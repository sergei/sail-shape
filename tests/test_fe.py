"""Validation of the generic solver against closed-form results."""
import numpy as np
import pytest
from scipy.optimize import brentq

from rigsolver.fe import Beam, Cable, Model, solve


def line_model(L, n_el, **beam):
    X = np.column_stack([np.linspace(0, L, n_el + 1), np.zeros(n_el + 1), np.zeros(n_el + 1)])
    m = Model(X)
    for i in range(n_el):
        m.beams.append(Beam(i, i + 1, **beam))
    return m


BEAM = dict(EA=1e8, EI_v=2e4, EI_w=5e4, GJ=1e4, vref=(0, 0, 1))


@pytest.mark.parametrize("direction,EI", [(2, BEAM["EI_v"]), (1, BEAM["EI_w"])])
def test_cantilever_tip_load(direction, EI):
    L, P = 3.0, 1.0  # small load -> linear regime
    m = line_model(L, 8, **BEAM)
    m.fix(0)
    f = np.zeros(3)
    f[direction] = P
    m.add_load(8, f)
    r = solve(m)
    assert r.u[8, direction] == pytest.approx(P * L**3 / (3 * EI), rel=1e-4)


def test_beam_column_amplification():
    """Pinned-pinned column, midspan load Q with axial compression P (Timoshenko & Gere)."""
    L, EI, Q = 4.0, 2e4, 10.0
    Pcr = np.pi**2 * EI / L**2
    P = 0.6 * Pcr
    m = line_model(L, 20, EA=1e9, EI_v=EI, EI_w=EI, GJ=1e4, vref=(0, 0, 1))
    m.fix(0, [0, 1, 2, 3])
    m.fix(20, [1, 2])
    m.add_load(20, [-P, 0, 0])
    m.add_load(10, [0, 0, Q])
    r = solve(m)
    u = L / 2 * np.sqrt(P / EI)
    expected = Q * L**3 / (48 * EI) * 3 * (np.tan(u) - u) / u**3
    assert r.u[10, 2] == pytest.approx(expected, rel=5e-3)
    assert r.min_eig > 0


def test_sliding_cable_v_hang():
    """Weight W hung on a continuous cable over a free node between two anchors."""
    a, h0, EA, W = 2.0, 0.5, 1e5, 500.0
    m = Model([[-a, 0, 0], [0, 0, -h0], [a, 0, 0]])
    m.fix(0), m.fix(2)
    L0 = 2 * np.hypot(a, h0)
    m.cables.append(Cable([0, 1, 2], EA=EA, L0=L0))
    m.add_load(1, [0, 0, -W])
    r = solve(m)

    def eq(h):
        s = np.hypot(a, h)
        return 2 * EA * (2 * s / L0 - 1) * h / s - W

    h = brentq(eq, h0, 2 * h0 + 1)
    assert -r.x[1, 2] == pytest.approx(h, rel=1e-6)
    assert r.x[1, 0] == pytest.approx(0, abs=1e-9)
    assert r.cable_tension[0] == pytest.approx(EA * (2 * np.hypot(a, h) / L0 - 1), rel=1e-6)


def test_sliding_cable_asymmetric_equal_tension():
    """Off-centre load: node slides until both sides carry the same tension and angles."""
    m = Model([[-2, 0, 0], [0.3, 0, -0.6], [1, 0, 0.5]])
    m.fix(0), m.fix(2)
    m.cables.append(Cable([0, 1, 2], EA=1e6, tension=2000.0))
    m.add_load(1, [0, 0, -1500.0])
    r = solve(m)
    T = r.cable_tension[0]
    e1 = (r.x[0] - r.x[1]) / np.linalg.norm(r.x[0] - r.x[1])
    e2 = (r.x[2] - r.x[1]) / np.linalg.norm(r.x[2] - r.x[1])
    np.testing.assert_allclose(T * (e1 + e2) + [0, 0, -1500.0], 0, atol=1e-6)
    assert T == pytest.approx(2000.0, rel=1e-9)
    assert r.cable_L0[0] == pytest.approx(r.cable_length[0] / (1 + T / 1e6), rel=1e-9)


def test_slack_cable_carries_nothing():
    m = Model([[0, 0, 1], [0, 0, 0], [0, 0, -1.5]])
    m.fix(0), m.fix(2)
    m.cables.append(Cable([0, 1], EA=1e5, L0=1.0))
    m.cables.append(Cable([1, 2], EA=1e5, L0=1.8))  # longer than the gap -> slack
    m.add_load(1, [0, 0, -100.0])
    r = solve(m)
    # the slack wire keeps a 1e-6 residual stiffness, i.e. a negligible push
    assert r.cable_tension[0] == pytest.approx(100.0, rel=1e-3)
    assert abs(r.cable_tension[1]) < 0.05


def test_guyed_mast_equilibrium():
    """Pinned mast with four stays: symmetric tune stays straight, reactions balance."""
    H, d = 10.0, 3.0
    X = [[0, 0, z] for z in np.linspace(0, H, 11)] + [[d, 0, 0], [-d, 0, 0], [0, d, 0], [0, -d, 0]]
    m = Model(X)
    for i in range(10):
        m.beams.append(Beam(i, i + 1, EA=5e8, EI_v=3e5, EI_w=3e5, GJ=1e5, vref=(1, 0, 0)))
    m.fix(0, [0, 1, 2, 5])
    for k in range(11, 15):
        m.fix(k)
    fwd = Cable([10, 11], EA=4e6, L0=np.hypot(H, d), name="fwd")
    m.cables += [fwd, Cable([10, 12], EA=4e6, tension=3000.0), Cable([10, 13], EA=4e6, tension=2000.0),
                 Cable([10, 14], EA=4e6, tension=2000.0)]
    r = solve(m)
    # mast leans aft until the elastic forestay balances; tensions equal to first order
    assert r.cable_tension[0] == pytest.approx(3000.0, rel=5e-3)
    np.testing.assert_allclose(r.u[:11, 1], 0, atol=1e-9)  # symmetric athwartships
    np.testing.assert_allclose(r.reactions[:, :3].sum(axis=0), 0, atol=1e-6)
    cos = H / np.hypot(H, d)
    assert -r.beam_axial[0] == pytest.approx(sum(r.cable_tension) * cos, rel=2e-3)
