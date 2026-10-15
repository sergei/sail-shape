"""Generic geometrically nonlinear 3D frame + sliding-cable statics solver.

Units are whatever the caller uses consistently (the rig model uses SI: m, N).

Beams: Euler-Bernoulli with cubic transverse shape functions and a moderate-rotation
(von Karman) axial strain measured in a fixed local frame,
    eps = du/L + (1/2L) q^T G q,
which is energy-consistent and captures beam-column (P-delta / P-small-delta)
amplification. Valid for small rotations (mast slopes of a few degrees), not for
large rigid-body rotations.

Cables: one element may pass over several nodes. The wire slides freely over the
intermediate nodes, so a single tension acts along the whole path (continuous
shroud over spreader tips). Cables are tension-only; a slack cable keeps a tiny
residual stiffness so the Newton system stays regular. Each cable is controlled
either by rest length L0 or by a target tension, in which case L0 becomes an extra
unknown solved alongside the displacements.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SLACK_STIFFNESS_RATIO = 1e-6
# Floor on the tension used in a cable's geometric (tangent-only) stiffness, as a fraction
# of EA. Keeps sliding-cable nodes regular when a wire starts at zero tension; it does not
# change the equilibrium, only the Newton path.
GEOMETRIC_TENSION_FLOOR = 1e-6


@dataclass
class Beam:
    n1: int
    n2: int
    EA: float
    EI_v: float  # bending stiffness for deflection along local y (e2)
    EI_w: float  # bending stiffness for deflection along local z (e3 = e1 x e2)
    GJ: float
    vref: tuple = (0.0, 0.0, 1.0)  # approximate direction of local y


@dataclass
class Cable:
    nodes: list
    EA: float
    L0: float | None = None
    tension: float | None = None  # target tension; when set, L0 is solved for
    L_start: float | None = None  # rest length at load factor 0 (defaults to initial length)
    name: str = ""


@dataclass
class Result:
    u: np.ndarray  # (n, 6) displacements and rotations
    x: np.ndarray  # (n, 3) deformed positions
    cable_tension: np.ndarray
    cable_L0: np.ndarray
    cable_length: np.ndarray
    beam_axial: np.ndarray  # + tension
    reactions: np.ndarray  # (n, 6) support force acting on the structure at fixed dofs
    converged: bool
    iterations: int
    min_eig: float  # smallest eigenvalue of the tangent stiffness at fixed rest lengths (>0: stable)


class Model:
    def __init__(self, X):
        self.X = np.asarray(X, float).reshape(-1, 3)
        n = len(self.X)
        self.beams: list[Beam] = []
        self.cables: list[Cable] = []
        self.fixed = np.zeros((n, 6), bool)
        self.loads = np.zeros((n, 6))
        self.springs: list[tuple[int, np.ndarray]] = []  # (node, 6x6) springs to ground

    @property
    def n(self):
        return len(self.X)

    def fix(self, node, dofs=range(6)):
        self.fixed[node, list(dofs)] = True

    def add_spring(self, node, K6):
        self.springs.append((node, np.asarray(K6, float)))

    def add_load(self, node, f):
        f = np.asarray(f, float)
        self.loads[node, : len(f)] += f


# ---------------------------------------------------------------- elements

def _beam_frame(X1, X2, vref):
    d = X2 - X1
    L = np.linalg.norm(d)
    e1 = d / L
    v = np.asarray(vref, float)
    e2 = v - (v @ e1) * e1
    if np.linalg.norm(e2) < 1e-8:
        raise ValueError("beam vref is parallel to the beam axis")
    e2 /= np.linalg.norm(e2)
    e3 = np.cross(e1, e2)
    return L, np.vstack([e1, e2, e3])


def _beam_matrices(b: Beam, L):
    Kb = np.zeros((12, 12))
    G = np.zeros((12, 12))
    iv, iw = [1, 5, 7, 11], [2, 4, 8, 10]
    kv = np.array([[12, 6 * L, -12, 6 * L], [6 * L, 4 * L * L, -6 * L, 2 * L * L],
                   [-12, -6 * L, 12, -6 * L], [6 * L, 2 * L * L, -6 * L, 4 * L * L]]) / L**3
    kw = kv * np.array([[1, -1, 1, -1], [-1, 1, -1, 1], [1, -1, 1, -1], [-1, 1, -1, 1]])
    gv = np.array([[36, 3 * L, -36, 3 * L], [3 * L, 4 * L * L, -3 * L, -L * L],
                   [-36, -3 * L, 36, -3 * L], [3 * L, -L * L, -3 * L, 4 * L * L]]) / (30 * L)
    gw = gv * np.array([[1, -1, 1, -1], [-1, 1, -1, 1], [1, -1, 1, -1], [-1, 1, -1, 1]])
    Kb[np.ix_(iv, iv)] = b.EI_v * kv
    Kb[np.ix_(iw, iw)] = b.EI_w * kw
    Kb[np.ix_([3, 9], [3, 9])] = b.GJ / L * np.array([[1, -1], [-1, 1]])
    G[np.ix_(iv, iv)] = gv
    G[np.ix_(iw, iw)] = gw
    a = np.zeros(12)
    a[0], a[6] = -1.0, 1.0
    return Kb, G, a


class _BeamData:
    def __init__(self, b: Beam, X):
        self.b = b
        self.L, R = _beam_frame(X[b.n1], X[b.n2], b.vref)
        self.T = np.kron(np.eye(4), R)
        self.Kb, self.G, self.a = _beam_matrices(b, self.L)
        self.dofs = np.r_[6 * b.n1: 6 * b.n1 + 6, 6 * b.n2: 6 * b.n2 + 6]

    def evaluate(self, ue):
        q = self.T @ ue
        Gq = self.G @ q
        N = self.b.EA * ((self.a @ q) / self.L + 0.5 * (q @ Gq) / self.L)
        bv = self.a + Gq
        f = self.Kb @ q + N * bv
        K = self.Kb + (self.b.EA / self.L) * np.outer(bv, bv) + N * self.G
        return self.T.T @ f, self.T.T @ K @ self.T, N


def _cable_state(c: Cable, pts, L0):
    d = np.diff(pts, axis=0)
    seg = np.linalg.norm(d, axis=1)
    e = d / seg[:, None]
    L = seg.sum()
    m = len(pts)
    g = np.zeros((m, 3))
    g[:-1] -= e
    g[1:] += e
    k = c.EA / L0 * (1.0 if L >= L0 else SLACK_STIFFNESS_RATIO)
    T = k * (L - L0)
    H = np.zeros((3 * m, 3 * m))
    for i in range(m - 1):
        P = (np.eye(3) - np.outer(e[i], e[i])) / seg[i]
        a, b = slice(3 * i, 3 * i + 3), slice(3 * i + 3, 3 * i + 6)
        H[a, a] += P
        H[b, b] += P
        H[a, b] -= P
        H[b, a] -= P
    gf = g.ravel()
    Tg = max(T, GEOMETRIC_TENSION_FLOOR * c.EA)
    return dict(L=L, T=T, dTdL=k, dTdL0=-k * L / L0, g=gf, K=k * np.outer(gf, gf) + Tg * H)


def _cable_dofs(c: Cable):
    return np.concatenate([np.arange(6 * n, 6 * n + 3) for n in c.nodes])


# ---------------------------------------------------------------- solver

class _System:
    def __init__(self, model: Model):
        self.m = model
        n = model.n
        self.beams = [_BeamData(b, model.X) for b in model.beams]
        has_rot = np.zeros(n, bool)
        for b in model.beams:
            has_rot[[b.n1, b.n2]] = True
        active = np.ones((n, 6), bool)
        active[~has_rot, 3:] = False
        self.free = np.flatnonzero((active & ~model.fixed).ravel())
        self.cdofs = [_cable_dofs(c) for c in model.cables]
        self.L_init = np.array([np.linalg.norm(np.diff(model.X[c.nodes], axis=0), axis=1).sum()
                                for c in model.cables])
        self.ctrl = [i for i, c in enumerate(model.cables) if c.tension is not None]
        for i, c in enumerate(model.cables):
            if c.tension is None and c.L0 is None:
                raise ValueError(f"cable {c.name or i} needs L0 or tension")

    def L0_length_controlled(self, lam):
        out = np.empty(len(self.m.cables))
        for i, c in enumerate(self.m.cables):
            if c.tension is None:
                start = self.L_init[i] if c.L_start is None else c.L_start
                out[i] = start + lam * (c.L0 - start)
        return out

    def assemble(self, u, L0):
        n6 = 6 * self.m.n
        f = np.zeros(n6)
        K = np.zeros((n6, n6))
        N = np.zeros(len(self.beams))
        for k, bd in enumerate(self.beams):
            fe, Ke, N[k] = bd.evaluate(u[bd.dofs])
            f[bd.dofs] += fe
            K[np.ix_(bd.dofs, bd.dofs)] += Ke
        for node, Ks in self.m.springs:
            d = np.arange(6 * node, 6 * node + 6)
            f[d] += Ks @ u[d]
            K[np.ix_(d, d)] += Ks
        X = self.m.X
        cs = []
        for c, d, L0c in zip(self.m.cables, self.cdofs, L0):
            pts = X[c.nodes] + u[d].reshape(-1, 3)
            s = _cable_state(c, pts, L0c)
            f[d] += s["T"] * s["g"]
            K[np.ix_(d, d)] += s["K"]
            cs.append(s)
        return f, K, N, cs


def _newton(sys: _System, lam, u, L0, max_iter, tol):
    m = sys.m
    free, ctrl = sys.free, sys.ctrl
    fext = lam * m.loads.ravel()
    targets = np.array([lam * m.cables[i].tension for i in ctrl])
    L0 = L0.copy()
    fixedL0 = sys.L0_length_controlled(lam)
    for i, c in enumerate(m.cables):
        if c.tension is None:
            L0[i] = fixedL0[i]
    scale = max(1.0, np.abs(fext).max(initial=0), np.abs(targets).max(initial=0),
                max((c.EA * 1e-6 for c in m.cables), default=0))

    def residual(u, L0):
        f, K, N, cs = sys.assemble(u, L0)
        r = np.concatenate([f[free] - fext[free], [cs[i]["T"] for i in ctrl] - targets])
        return r, f, K, N, cs

    r, f, K, N, cs = residual(u, L0)
    nf = len(free)
    nonmonotone_left = 8
    for it in range(1, max_iter + 1):
        J = np.zeros((nf + len(ctrl),) * 2)
        J[:nf, :nf] = K[np.ix_(free, free)]
        for j, i in enumerate(ctrl):
            gi = cs[i]["g"]
            gfull = np.zeros(6 * m.n)
            gfull[sys.cdofs[i]] = gi
            J[:nf, nf + j] = cs[i]["dTdL0"] * gfull[free]
            J[nf + j, :nf] = cs[i]["dTdL"] * gfull[free]
            J[nf + j, nf + j] = cs[i]["dTdL0"]
        try:
            dz = -np.linalg.solve(J, r)
        except np.linalg.LinAlgError:
            return None
        phi0 = r @ r
        step = 1.0
        for _ in range(12):
            u_try = u.copy()
            u_try[free] += step * dz[:nf]
            L0_try = L0.copy()
            L0_try[ctrl] += step * dz[nf:]
            if np.all(L0_try > 0):
                r_try, *rest = residual(u_try, L0_try)
                phi = r_try @ r_try
                # Stiff wire stretch and soft sliding/bending modes make the plain residual
                # norm a poor merit function, so a limited number of full Newton steps are
                # accepted even if the residual grows (non-monotone strategy).
                if phi < (1 - 1e-4 * step) * phi0 or phi0 < (tol * max(scale, np.abs(f).max())) ** 2:
                    break
                if step == 1.0 and nonmonotone_left > 0 and phi < 1e6 * phi0:
                    nonmonotone_left -= 1
                    break
            step *= 0.5
        else:
            return None
        u, L0, r = u_try, L0_try, r_try
        f, K, N, cs = rest
        # tolerance relative to the largest internal force, so round-off in heavily
        # pre-stressed rigs (tens of kN through the mast) does not block convergence
        if (np.abs(r).max() < tol * max(scale, np.abs(f).max())
                and np.abs(step * dz).max() < 1e-9 * (1 + np.abs(u).max())):
            return u, L0, f, K, N, cs, it
    return None


def solve(model: Model, max_iter=60, tol=1e-10, steps=None) -> Result:
    """Solve static equilibrium; ramps loads/targets/rest-length changes if a direct solve fails."""
    sys = _System(model)
    n6 = 6 * model.n
    schedules = [steps] if steps else [[1.0], list(np.linspace(0.25, 1, 4)), list(np.linspace(0.05, 1, 20))]
    for sched in schedules:
        u = np.zeros(n6)
        L0 = sys.L0_length_controlled(0.0)
        out, total_it = None, 0
        for lam in sched:
            for i in sys.ctrl:
                c = model.cables[i]
                if out is None:
                    L0[i] = sys.L_init[i] / (1 + lam * c.tension / c.EA)
            out = _newton(sys, lam, u, L0, max_iter, tol)
            if out is None:
                break
            u, L0, f, K, N, cs, it = out
            total_it += it
        if out is not None:
            break
    else:
        raise RuntimeError("rig solver did not converge")

    free = sys.free
    Kff = K[np.ix_(free, free)]
    min_eig = float(np.linalg.eigvalsh(0.5 * (Kff + Kff.T))[0])
    reactions = np.zeros(n6)
    fixed = model.fixed.ravel()
    reactions[fixed] = (f - model.loads.ravel())[fixed]
    U = u.reshape(-1, 6)
    return Result(u=U, x=model.X + U[:, :3],
                  cable_tension=np.array([s["T"] for s in cs]), cable_L0=L0,
                  cable_length=np.array([s["L"] for s in cs]), beam_axial=N,
                  reactions=reactions.reshape(-1, 6), converged=True,
                  iterations=total_it, min_eig=min_eig)
