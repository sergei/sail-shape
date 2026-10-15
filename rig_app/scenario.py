"""GUI inputs -> solver runs -> plain data (DataFrames / dicts) that is cheap to cache and plot.

A scenario is the sequence a crew actually follows:
  1. dock tune: shrouds set by tension (Loos gauge), forestay by length, a dock backstay;
  2. sailing: every wire locked at its dock-tune length except the backstay, which is
     re-tensioned, plus an optional sideways load at the hounds.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from rigsolver import Rig, RigResult, RigState

SLACK_N = 50.0
OVERLOAD_PCT = 30.0


@dataclass(frozen=True)
class Tune:
    cap_kN: float = 7.5
    intermediate_kN: float = 4.0
    lower_kN: float = 3.0
    backstay_kN: float = 2.0  # per leg
    forestay_shorten_mm: float = 0.0


@dataclass(frozen=True)
class Sailing:
    backstay_kN: float = 2.0  # per leg
    side_load_N: float = 0.0  # at the hounds, + = to starboard


@dataclass(frozen=True)
class Assumptions:
    k_EI_fa: float = 1.0
    k_EI_lat: float = 1.0
    base_k_rot: float = 0.0  # N*m/rad, both planes
    prebend_mm: float = 0.0


def make_rig(base: Rig, a: Assumptions) -> Rig:
    return base.with_params(k_EI_fa=a.k_EI_fa, k_EI_lat=a.k_EI_lat, base_k_rot_fa=a.base_k_rot,
                            base_k_rot_lat=a.base_k_rot, prebend_m=a.prebend_mm / 1000)


def tune_state(t: Tune) -> RigState:
    return RigState({"cap": {"tension": t.cap_kN * 1e3}, "intermediate": {"tension": t.intermediate_kN * 1e3},
                     "lower": {"tension": t.lower_kN * 1e3}, "backstay": {"tension": t.backstay_kN * 1e3},
                     "forestay": {"shorten": t.forestay_shorten_mm / 1e3}})


def sailing_state(dock: RigResult, s: Sailing) -> RigState:
    loads = [{"at": "mast_hounds", "force_N": [0.0, s.side_load_N, 0.0]}] if s.side_load_N else []
    return RigState({**dock.locked_controls(free=("backstay",)), "backstay": {"tension": s.backstay_kN * 1e3}},
                    loads=loads)


def snapshot(r: RigResult) -> dict:
    """Plain-data view of one solved state."""
    rig, m, res = r.rig, r.model, r.fe
    s_goose, s_head = rig.mast_s["boom_gooseneck"], rig.mast_s["mast_head"]
    U = res.u[:, :3]
    mast = pd.DataFrame({
        "s": r.s, "x": m.X[r.idx["mast"], 0], "y": m.X[r.idx["mast"], 1], "z": m.X[r.idx["mast"], 2],
        "ux": U[r.idx["mast"], 0], "uy": U[r.idx["mast"], 1], "uz": U[r.idx["mast"], 2],
        "offset_fwd_mm": 1e3 * r.offset_fwd, "offset_stbd_mm": 1e3 * r.offset_stbd,
        "bend_fa_mm": 1e3 * r.chord_offset(r.s, (s_goose, s_head), "fore_aft"),
        "bend_lat_mm": 1e3 * r.chord_offset(r.s, (s_goose, s_head), "lateral"),
    })
    rows = []
    for w, ci in r.idx["cables"].items():
        for k, n in enumerate(m.cables[ci].nodes):
            rows.append(("wire", w, rig.wires[w]["role"], k, *m.X[n], *U[n]))
    for sid, bi in r.idx["spreader_beams"].items():
        b = m.beams[bi]
        for k, n in enumerate((b.n1, b.n2)):
            rows.append(("spreader", sid, "spreader", k, *m.X[n], *U[n]))
    lines = pd.DataFrame(rows, columns=["kind", "member", "role", "order", "x", "y", "z", "ux", "uy", "uz"])

    trows = []
    for w, T in r.tensions.items():
        brk = rig.wire_props(w)["break_load_N"]["value"]
        trows.append({"wire": w, "tension_kN": T / 1e3, "pct_break": 100 * T / brk,
                      "shorten_mm": 1e3 * r.shorten[w]})
    tensions = pd.DataFrame(trows)

    head = r.idx["mast_head"]
    above = mast["s"] >= s_goose
    k = mast.loc[above, "bend_fa_mm"].abs().idxmax()
    metrics = {
        "forestay_kN": r.tensions["forestay"] / 1e3,
        "compression_kN": r.mast_compression_base / 1e3,
        "max_bend_mm": float(mast.at[k, "bend_fa_mm"]),
        "max_bend_height_m": float(mast.at[k, "s"]),
        "head_aft_mm": float(-1e3 * r.offset_fwd[head]),
        "head_side_mm": float(1e3 * r.offset_stbd[head]),
        "max_side_mm": float(np.abs(mast.loc[above, "bend_lat_mm"]).max()),
        "stable": bool(r.stable),
        "slack": [w for w, T in r.tensions.items() if T < SLACK_N],
        "overloaded": tensions.loc[tensions["pct_break"] > OVERLOAD_PCT, "wire"].tolist(),
    }
    return {"mast": mast, "lines": lines, "tensions": tensions, "metrics": metrics}


def run(base: Rig, a: Assumptions, t: Tune, s: Sailing) -> dict:
    """Solve dock tune and sailing state. Raises RuntimeError if no equilibrium is found."""
    rig = make_rig(base, a)
    dock = rig.solve(tune_state(t))
    same = s.side_load_N == 0 and s.backstay_kN == t.backstay_kN
    current = dock if same else rig.solve(sailing_state(dock, s))
    return {"dock": snapshot(dock), "current": snapshot(current), "same": same}


def backstay_sweep(base: Rig, a: Assumptions, t: Tune, values_kN) -> pd.DataFrame:
    """Forestay tension and bend vs backstay, other wires locked at the dock tune."""
    rig = make_rig(base, a)
    dock = rig.solve(tune_state(t))
    rows = []
    for b in values_kN:
        try:
            snap = snapshot(rig.solve(sailing_state(dock, Sailing(backstay_kN=float(b)))))
        except RuntimeError:
            continue
        mt = snap["metrics"]
        rows.append({"backstay_kN": float(b), "forestay_kN": mt["forestay_kN"], "max_bend_mm": mt["max_bend_mm"],
                     "cap_kN": float(snap["tensions"].set_index("wire").at["cap_port", "tension_kN"])})
    return pd.DataFrame(rows)
