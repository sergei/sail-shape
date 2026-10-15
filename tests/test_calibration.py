"""Synthetic round trip: measurements generated with known params are fitted back."""
import json

import numpy as np
import pytest

from rigsolver.calibration import Measurement, fit, predict, write_calibration
from rigsolver.rig import EXAMPLE_TUNE, Rig, RigState

TRUE = {"k_EI_fa": 1.6, "k_EI_lat": 0.8}
HEIGHTS = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]


def synthetic(rig_true):
    """Two dock states: tune with light and heavier backstay, plus a sideways halyard pull."""
    tune = rig_true.solve(RigState(EXAMPLE_TUNE))
    states = {
        "light": RigState(EXAMPLE_TUNE),
        "backstay": RigState({**tune.locked_controls(free=("backstay",)), "backstay": {"tension": 3500.0}}),
        "side_pull": RigState(EXAMPLE_TUNE, loads=[{"at": "mast_hounds", "force_N": [0, 400.0, 0]}]),
    }
    out = []
    for name, st in states.items():
        plane = "lateral" if name == "side_pull" else "fore_aft"
        obs = [{"type": "chord_offset", "plane": plane, "chord_m": [0.0, 13.5], "height_m": h,
                "value_m": 0.0, "sigma_m": 0.002} for h in HEIGHTS]
        m = Measurement(name, st, obs)
        for ob, v in zip(obs, predict(rig_true, m)):
            ob["value_m"] = float(v)
        out.append(m)
    return out


@pytest.fixture(scope="module")
def setup():
    rig = Rig.load()
    return rig, synthetic(rig.with_params(**TRUE))


def test_fit_recovers_known_params(setup):
    rig, meas = setup
    res = fit(rig, meas, ["k_EI_fa", "k_EI_lat"])
    assert res.success
    for k, v in TRUE.items():
        assert res.params[k] == pytest.approx(v, rel=1e-3)
    assert np.abs(res.residuals).max() < 1e-3


def test_measurement_json_and_write_back(setup, tmp_path):
    rig, meas = setup
    m2 = Measurement.from_dict(json.loads(json.dumps(meas[0].to_dict())))
    np.testing.assert_allclose(predict(rig, m2), predict(rig, meas[0]))
    res = fit(rig, meas[:1], ["k_EI_fa"])
    props = tmp_path / "props.json"
    props.write_text(json.dumps(rig.properties))
    write_calibration(res, props, date="2027-01-01")
    cal = json.loads(props.read_text())["calibration"]
    assert cal["status"] == "calibrated" and cal["params"]["k_EI_fa"]["source"] == "calibrated"
    assert Rig.load(properties_path=props).params["k_EI_fa"] == pytest.approx(res.params["k_EI_fa"])
