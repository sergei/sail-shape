"""Sun Fast 3300 rig: builds a solver model from rig_model.json (geometry, ft) and
rig_properties.json (stiffness, SI) and solves it for a tuning state.

Coordinates follow rig_model.json (x forward, y starboard, z up, origin at mast base on
deck datum) converted to metres. Mast heights `s` are measured along the raked mast axis.

Wire controls (RigState.controls) are keyed by wire id (`cap_port`, ...) or by group
(`cap`, `intermediate`, `lower`, `backstay`, `forestay`; a group applies the same value to
each of its wires, so backstay tension is per leg). Each control is one of
    {"tension": N}        target tension (Loos gauge reading)
    {"rest_length": m}    absolute unstretched length
    {"shorten": m}        shortening relative to the unloaded reference geometry
    {"turns": n}          rigging-screw turns relative to the reference geometry
Uncontrolled wires sit at their reference length (stress-free in the unloaded geometry).

External loads are applied after tuning: wires given by tension are tuned first without
the loads, their rest lengths are then locked and the loads applied (dock tune -> sailing).
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import fe

FT = 0.3048
GRAVITY = 9.80665
DOCS = Path(__file__).resolve().parent.parent / "docs"
GEOMETRY_PATH = DOCS / "rig_model.json"
PROPERTIES_PATH = DOCS / "rig_properties.json"

GROUPS = {
    "cap": ["cap_port", "cap_stbd"],
    "intermediate": ["intermediate_port", "intermediate_stbd"],
    "lower": ["lower_port", "lower_stbd"],
    "backstay": ["backstay_port", "backstay_stbd"],
    "forestay": ["forestay"],
}
MAST_NODES = ["mast_base", "boom_gooseneck", "mast_lower_shroud", "mast_lower_spr", "mast_upper_spr",
              "mast_hounds", "mast_head"]

# Illustrative dock tune for demos and tests; NOT a recommended SF3300 tune.
EXAMPLE_TUNE = {"cap": {"tension": 7500.0}, "intermediate": {"tension": 4000.0},
                "lower": {"tension": 3000.0}, "backstay": {"tension": 2000.0}}


@dataclass
class RigState:
    controls: dict = field(default_factory=dict)
    loads: list = field(default_factory=list)  # {"at": node id or height_m along mast, "force_N": [fx, fy, fz]}
    gravity: bool = True

    @classmethod
    def from_dict(cls, d):
        return cls(controls=d.get("controls", {}), loads=d.get("loads", []), gravity=d.get("gravity", True))

    def to_dict(self):
        return {"controls": self.controls, "loads": self.loads, "gravity": self.gravity}


class Rig:
    def __init__(self, geometry: dict, properties: dict, params: dict | None = None, mesh_max_m=0.5):
        self.geometry = geometry
        self.properties = properties
        self.params = {k: v["value"] for k, v in properties["calibration"]["params"].items()}
        unknown = set(params or {}) - set(self.params)
        if unknown:
            raise KeyError(f"unknown calibration params: {sorted(unknown)}")
        self.params.update(params or {})
        self.mesh_max_m = mesh_max_m
        if self._prop("spreader_tip_contact") != "sliding":
            raise NotImplementedError("only sliding spreader-tip contact is implemented")

        self.nodes = {n["id"]: np.array(n["xyz"]) * FT for n in geometry["nodes"]}
        self.rake = np.radians(geometry["parameters"]["mast_rake_aft_deg"]["value"])
        self.axis = np.array([-np.sin(self.rake), 0.0, np.cos(self.rake)])
        self.fwd = np.array([np.cos(self.rake), 0.0, np.sin(self.rake)])  # forward, normal to the mast axis
        self.stbd = np.array([0.0, 1.0, 0.0])
        self.mast_s = {k: float(self.nodes[k] @ self.axis) for k in MAST_NODES}
        self.mast_length = self.mast_s["mast_head"]
        self.wires = {m["id"]: m for m in geometry["members"] if m["type"] == "wire"}
        self.spreaders = {m["id"]: m for m in geometry["members"] if m["type"] == "strut"}
        # reference (stress-free) wire lengths: the unloaded geometry incl. any pre-bend
        free_xyz = {n: self._stress_free_xyz(n) for n in self.nodes}
        self.ref_length = {w: sum(np.linalg.norm(free_xyz[b] - free_xyz[a])
                                  for a, b in zip(m["path"], m["path"][1:]))
                           for w, m in self.wires.items()}

    @classmethod
    def load(cls, geometry_path=GEOMETRY_PATH, properties_path=PROPERTIES_PATH, **kw):
        return cls(json.loads(Path(geometry_path).read_text()),
                   json.loads(Path(properties_path).read_text()), **kw)

    def with_params(self, **params) -> "Rig":
        merged = {**self.params, **params}
        return Rig(self.geometry, self.properties, merged, self.mesh_max_m)

    def _bow(self, s):
        """Stress-free forward pre-bend of the tube at mast height s."""
        return self.params["prebend_m"] * np.sin(np.pi * np.asarray(s) / self.mast_length)

    def _stress_free_xyz(self, nid):
        xyz = self.nodes[nid]
        if nid in self.mast_s:
            return xyz + self._bow(self.mast_s[nid]) * self.fwd
        if nid.startswith("spr_"):  # spreader tips follow their root's pre-bend offset
            root = "mast_upper_spr" if "upper" in nid else "mast_lower_spr"
            return xyz + self._bow(self.mast_s[root]) * self.fwd
        return xyz

    def _prop(self, key):
        return self.properties["parameters"][key]["value"]

    # ------------------------------------------------------------ properties
    def mast_section(self, s):
        """Effective (calibrated) mast stiffness at height s along the mast."""
        d = self.properties["derived"]
        lo, up = d["mast_section_lower"], d["mast_section_upper"]
        z0, z1 = self._prop("mast_taper_start_m"), self._prop("mast_taper_end_m")
        w = float(np.clip((s - z0) / (z1 - z0), 0, 1))
        p = self.params
        ku = p["k_EI_upper"] ** w
        mix = lambda k: (1 - w) * lo[k] + w * up[k]
        return dict(EA=mix("EA_N"), EI_fa=mix("EI_fa_Nm2") * p["k_EI_fa"] * ku,
                    EI_lat=mix("EI_lat_Nm2") * p["k_EI_lat"] * ku, GJ=mix("GJ_Nm2"))

    def wire_props(self, wire_id):
        return self.properties["wires"][self.wires[wire_id]["role"]]

    def wire_EA(self, wire_id):
        return self.wire_props(wire_id)["EA_N"] * self.params["k_EA_wire"]

    def _resolve_controls(self, controls):
        out = {}
        for key, ctl in controls.items():
            ids = GROUPS.get(key, [key])
            for w in ids:
                if w not in self.wires:
                    raise KeyError(f"unknown wire or group {key!r}")
                out[w] = ctl
        return out

    def _rest_length(self, wire_id, ctl):
        ref = self.ref_length[wire_id]
        if "rest_length" in ctl:
            return ctl["rest_length"]
        if "shorten" in ctl:
            return ref - ctl["shorten"]
        if "turns" in ctl:
            return ref - ctl["turns"] * self.wire_props(wire_id)["turn_shorten_m"]
        raise ValueError(f"control for {wire_id} needs tension, rest_length, shorten or turns: {ctl}")

    # ------------------------------------------------------------ model
    def _mast_stations(self, extra):
        key = sorted(set(self.mast_s.values()) | set(extra))
        s = [key[0]]
        for a, b in zip(key, key[1:]):
            n = max(1, int(np.ceil((b - a) / self.mesh_max_m - 1e-9)))
            s += list(np.linspace(a, b, n + 1)[1:])
        return np.array(s)

    def _load_heights(self, loads):
        return [ld["at"] for ld in loads if not isinstance(ld["at"], str)]

    def build_model(self, state: RigState, rest_lengths: dict | None = None, with_loads=True):
        """Returns (fe.Model, index) where index maps names to node / cable indices."""
        s = self._mast_stations(self._load_heights(state.loads))
        X = [si * self.axis + pb * self.fwd for si, pb in zip(s, self._bow(s))]
        idx = {"mast": list(range(len(s)))}
        for name, sv in self.mast_s.items():
            idx[name] = int(np.argmin(np.abs(s - sv)))
        for nid in self.nodes:
            if nid in idx or nid == "boom_clew":
                continue
            idx[nid] = len(X)
            X.append(self._stress_free_xyz(nid))
        m = fe.Model(np.array(X))

        for i in range(len(s) - 1):
            sec = self.mast_section(0.5 * (s[i] + s[i + 1]))
            # local y = starboard -> EI_v is lateral; local z is fore-aft
            m.beams.append(fe.Beam(i, i + 1, sec["EA"], sec["EI_lat"], sec["EI_fa"], sec["GJ"], vref=(0, 1, 0)))
        sp = self.properties["derived"]["spreader_section"]
        idx["spreader_beams"] = {}
        for sid, mem in self.spreaders.items():
            idx["spreader_beams"][sid] = len(m.beams)
            m.beams.append(fe.Beam(idx[mem["nodes"][0]], idx[mem["nodes"][1]], sp["EA_N"], sp["EI_vertical_Nm2"],
                                   sp["EI_horizontal_Nm2"], sp["GJ_Nm2"], vref=(0, 0, 1)))

        base = idx["mast_base"]
        m.fix(base, [0, 1, 2])
        GJ0 = self.mast_section(0)["GJ"]
        Krot = (1e3 * GJ0 * np.outer(self.axis, self.axis)  # torsion held at the step
                + self.params["base_k_rot_fa"] * np.outer(self.stbd, self.stbd)
                + self.params["base_k_rot_lat"] * np.outer(self.fwd, self.fwd))
        K6 = np.zeros((6, 6))
        K6[3:, 3:] = Krot
        m.add_spring(base, K6)
        for nid in self.nodes:
            if self.geometry_kind(nid) == "deck":
                m.fix(idx[nid])

        ctl = self._resolve_controls(state.controls)
        idx["cables"] = {}
        for w, mem in self.wires.items():
            c = ctl.get(w, {"shorten": 0.0})
            cable = fe.Cable([idx[n] for n in mem["path"]], EA=self.wire_EA(w), name=w,
                             L_start=self.ref_length[w])
            if rest_lengths and w in rest_lengths:
                cable.L0 = rest_lengths[w]
            elif "tension" in c:
                cable.tension = float(c["tension"])
            else:
                cable.L0 = self._rest_length(w, c)
            idx["cables"][w] = len(m.cables)
            m.cables.append(cable)

        if state.gravity:
            for i, wt in enumerate(self._mast_weights(s)):
                m.add_load(i, [0, 0, -wt])
        if with_loads:
            for ld in state.loads:
                at = ld["at"]
                node = idx[at] if isinstance(at, str) else int(np.argmin(np.abs(s - at)))
                m.add_load(node, ld["force_N"])
        idx["s"] = s
        return m, idx

    def geometry_kind(self, nid):
        return next(n["kind"] for n in self.geometry["nodes"] if n["id"] == nid)

    def _mast_weights(self, s):
        """Nodal weights from a linear mass distribution matching cert weight and CG height."""
        W = self._prop("mast_weight_kg") * GRAVITY
        L = self.mast_length
        cg = self.geometry["parameters"]["mast_cg_height"]["value"] * FT
        # w(s) = c0 + c1 s with  int w = W,  int s w = W cg
        A = np.array([[L, L**2 / 2], [L**2 / 2, L**3 / 3]])
        c0, c1 = np.linalg.solve(A, [W, W * cg])
        out = np.zeros(len(s))
        for i in range(len(s) - 1):
            le, sm = s[i + 1] - s[i], 0.5 * (s[i] + s[i + 1])
            out[i] += 0.5 * (c0 + c1 * sm) * le
            out[i + 1] += 0.5 * (c0 + c1 * sm) * le
        return out

    # ------------------------------------------------------------ solve
    def solve(self, state: RigState | dict | None = None) -> "RigResult":
        state = RigState.from_dict(state) if isinstance(state, dict) else (state or RigState())
        rest = None
        if state.loads and any("tension" in c for c in self._resolve_controls(state.controls).values()):
            model, idx = self.build_model(state, with_loads=False)
            tuned = fe.solve(model)
            rest = {w: tuned.cable_L0[i] for w, i in idx["cables"].items()}
        model, idx = self.build_model(state, rest_lengths=rest)
        return RigResult(self, state, model, idx, fe.solve(model))


class RigResult:
    def __init__(self, rig: Rig, state: RigState, model: fe.Model, idx: dict, res: fe.Result):
        self.rig, self.state, self.model, self.idx, self.fe = rig, state, model, idx, res
        mast = idx["mast"]
        self.s = idx["s"]
        self.mast_xyz = res.x[mast]
        straight = self.s[:, None] * rig.axis
        self.offset_fwd = (self.mast_xyz - straight) @ rig.fwd
        self.offset_stbd = (self.mast_xyz - straight) @ rig.stbd
        self.tensions = {w: float(res.cable_tension[i]) for w, i in idx["cables"].items()}
        self.rest_lengths = {w: float(res.cable_L0[i]) for w, i in idx["cables"].items()}
        self.shorten = {w: rig.ref_length[w] - L for w, L in self.rest_lengths.items()}
        nm = len(mast) - 1
        self.mast_axial = res.beam_axial[:nm]  # + tension, per element
        self.mast_axial_s = 0.5 * (self.s[1:] + self.s[:-1])
        self.spreader_axial = {k: float(res.beam_axial[i]) for k, i in idx["spreader_beams"].items()}
        self.stable = res.min_eig > 0

    def locked_controls(self, free=()):
        """Controls that hold every wire at its current rest length (as after a dock tune),
        except wires/groups in `free`, which can then be adjusted, e.g. free=("backstay",)."""
        free_ids = {w for k in free for w in GROUPS.get(k, [k])}
        return {w: {"rest_length": L} for w, L in self.rest_lengths.items() if w not in free_ids}

    @property
    def mast_compression_base(self):
        return float(-self.mast_axial[0])

    def position(self, s):
        """Deformed mast-axis position(s) at height(s) s (linear between nodes)."""
        s = np.atleast_1d(s)
        return np.column_stack([np.interp(s, self.s, self.mast_xyz[:, k]) for k in range(3)])

    def chord_offset(self, s, chord=None, plane="fore_aft"):
        """Offset of the mast from the straight line (e.g. a taut halyard) between two mast
        heights, in the deformed shape. fore_aft: + forward of the chord; lateral: + starboard."""
        s0, s1 = chord or (0.0, self.rig.mast_length)
        p0, p1 = self.position(s0)[0], self.position(s1)[0]
        c = (p1 - p0) / np.linalg.norm(p1 - p0)
        d = self.position(s) - p0
        perp = d - np.outer(d @ c, c)
        return perp @ (self.rig.fwd if plane == "fore_aft" else self.rig.stbd)

    def summary(self):
        r = self.rig
        lines = [f"converged in {self.fe.iterations} iterations, {'stable' if self.stable else 'UNSTABLE'}",
                 f"{'wire':20s} {'tension kN':>10s} {'% break':>8s} {'shorten mm':>11s}"]
        for w, T in self.tensions.items():
            brk = r.wire_props(w)["break_load_N"]["value"]
            lines.append(f"{w:20s} {T / 1e3:10.2f} {100 * T / brk:8.1f} {1e3 * self.shorten[w]:11.1f}")
        head = self.idx["mast_head"]
        bend = self.chord_offset(self.s)
        k = int(np.argmax(np.abs(bend)))
        lines += [f"mast compression at base: {self.mast_compression_base / 1e3:.2f} kN",
                  f"masthead moved: {1e3 * self.offset_fwd[head]:+.1f} mm fwd, {1e3 * self.offset_stbd[head]:+.1f} mm stbd",
                  f"max fore-aft bend vs base-head chord: {1e3 * bend[k]:+.1f} mm at s = {self.s[k]:.2f} m",
                  f"max lateral offset: {1e3 * np.abs(self.offset_stbd).max():.1f} mm",
                  f"spreaders axial kN: " + ", ".join(f"{k}={v / 1e3:+.2f}" for k, v in self.spreader_axial.items()),
                  f"calibration: {r.properties['calibration']['status']}"]
        return "\n".join(lines)
