"""Calibration hook: fit Rig calibration params to dockside measurements.

Not used yet (the model is uncalibrated); the format and fitting are in place so that
measurements can be dropped in later. A measurement file (see measurements/TEMPLATE.json):

{
  "id": "dock-2027-04-12-a",
  "date": "2027-04-12",
  "notes": "...",
  "state": {"controls": {"cap": {"tension": 7500}, "backstay": {"tension": 3000}, ...},
            "loads": [], "gravity": true},
  "observations": [
    {"type": "chord_offset", "plane": "fore_aft", "chord_m": [0.0, 13.5], "height_m": 6.5,
     "value_m": 0.045, "sigma_m": 0.003},
    {"type": "wire_tension", "wire": "forestay", "value_N": 9000, "sigma_N": 500}
  ]
}

chord_offset: mast offset from a straight line between two mast heights (taut halyard from
masthead to gooseneck, say); + forward (fore_aft) or + starboard (lateral). Heights are
along the mast from the deck datum.
"""
from __future__ import annotations

import datetime
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from .rig import Rig, RigState

# how each param is varied during a fit: factors in log space, absolutes linearly
PARAM_TRANSFORMS = {
    "factor": (np.log, np.exp),
    "absolute": (lambda v: v, lambda x: x),
}
ABSOLUTE_SCALES = {"base_k_rot_fa": 1e5, "base_k_rot_lat": 1e5, "prebend_m": 0.01}


@dataclass
class Measurement:
    id: str
    state: RigState
    observations: list
    date: str | None = None
    notes: str = ""

    @classmethod
    def from_dict(cls, d):
        return cls(id=d["id"], state=RigState.from_dict(d["state"]), observations=d["observations"],
                   date=d.get("date"), notes=d.get("notes", ""))

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))

    def to_dict(self):
        return {"id": self.id, "date": self.date, "notes": self.notes, "state": self.state.to_dict(),
                "observations": self.observations}


def predict(rig: Rig, m: Measurement) -> np.ndarray:
    """Model values for each observation of a measurement, in the observation's units."""
    res = rig.solve(m.state)
    out = []
    for ob in m.observations:
        if ob["type"] == "chord_offset":
            out.append(res.chord_offset(ob["height_m"], ob.get("chord_m"), ob.get("plane", "fore_aft"))[0])
        elif ob["type"] == "wire_tension":
            out.append(res.tensions[ob["wire"]])
        else:
            raise ValueError(f"unknown observation type {ob['type']!r}")
    return np.array(out)


def _observed(m: Measurement):
    vals, sig = [], []
    for ob in m.observations:
        unit = "m" if ob["type"] == "chord_offset" else "N"
        vals.append(ob[f"value_{unit}"])
        sig.append(ob[f"sigma_{unit}"])
    return np.array(vals), np.array(sig)


def residuals(rig: Rig, measurements) -> np.ndarray:
    """Sigma-weighted residuals (model - observed) over all measurements."""
    parts = []
    for m in measurements:
        obs, sig = _observed(m)
        parts.append((predict(rig, m) - obs) / sig)
    return np.concatenate(parts)


@dataclass
class FitResult:
    params: dict
    std: dict  # 1-sigma, in param units (linearised)
    rig: Rig
    cost: float
    residuals: np.ndarray
    success: bool
    message: str = ""
    measurement_ids: list = field(default_factory=list)


def fit(rig: Rig, measurements, params, bounds=None) -> FitResult:
    """Least-squares fit of the named calibration params, starting from rig.params."""
    kinds = {k: v["kind"] for k, v in rig.properties["calibration"]["params"].items()}
    scale = lambda k: 1.0 if kinds[k] == "factor" else ABSOLUTE_SCALES.get(k, 1.0)
    fwd = lambda k, v: PARAM_TRANSFORMS[kinds[k]][0](v) / (1 if kinds[k] == "factor" else scale(k))
    inv = lambda k, x: PARAM_TRANSFORMS[kinds[k]][1](x * (1 if kinds[k] == "factor" else scale(k)))
    x0 = np.array([fwd(k, rig.params[k]) for k in params])
    lo = np.array([0.0 if k.startswith("base_k_rot") else -np.inf for k in params])
    hi = np.full(len(params), np.inf)
    for i, k in enumerate(params):
        if bounds and k in bounds:
            lo[i], hi[i] = fwd(k, bounds[k][0]), fwd(k, bounds[k][1])
    x0 = np.clip(x0, lo + 1e-12, hi - 1e-12)

    def fun(x):
        return residuals(rig.with_params(**{k: inv(k, xi) for k, xi in zip(params, x)}), measurements)

    sol = least_squares(fun, x0, bounds=(lo, hi), diff_step=1e-5, x_scale=1.0)
    values = {k: float(inv(k, xi)) for k, xi in zip(params, sol.x)}
    try:
        cov = np.linalg.inv(sol.jac.T @ sol.jac)
        sx = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        sx = np.full(len(params), np.inf)
    # propagate to param units: d(param)/dx
    std = {k: float(abs(inv(k, xi + 1e-6) - inv(k, xi)) / 1e-6 * s) for k, xi, s in zip(params, sol.x, sx)}
    return FitResult(values, std, rig.with_params(**values), float(sol.cost), sol.fun, bool(sol.success),
                     sol.message, [m.id for m in measurements])


def write_calibration(fit_result: FitResult, properties_path, date=None):
    """Store fitted params in rig_properties.json (calibration block) with provenance."""
    path = Path(properties_path)
    props = json.loads(path.read_text())
    cal = props["calibration"]
    for k, v in fit_result.params.items():
        cal["params"][k]["value"] = v
        cal["params"][k]["std"] = fit_result.std[k]
        cal["params"][k]["source"] = "calibrated"
    cal["status"] = "calibrated"
    cal["fitted_on"] = date or datetime.date.today().isoformat()
    cal["measurement_sets"] = fit_result.measurement_ids
    path.write_text(json.dumps(props, indent=2))
