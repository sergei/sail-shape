"""Behavioural checks of the Sun Fast 3300 rig model (estimated, uncalibrated stiffness)."""
import numpy as np
import pytest

from rigsolver.rig import EXAMPLE_TUNE, Rig, RigState


@pytest.fixture(scope="module")
def rig():
    return Rig.load()


@pytest.fixture(scope="module")
def tuned(rig):
    return rig.solve(RigState(EXAMPLE_TUNE))


def test_tune_hits_targets_and_is_symmetric(tuned):
    assert tuned.stable
    for w in ("cap_port", "cap_stbd"):
        assert tuned.tensions[w] == pytest.approx(7500.0, rel=1e-9)
    assert tuned.tensions["forestay"] > 0
    np.testing.assert_allclose(tuned.offset_stbd, 0, atol=1e-9)
    assert tuned.tensions["cap_port"] == pytest.approx(tuned.tensions["cap_stbd"], rel=1e-9)


def test_global_equilibrium(tuned):
    """Support reactions balance gravity; mast-step load equals base compression."""
    res, m = tuned.fe, tuned.model
    np.testing.assert_allclose(res.reactions[:, :3].sum(axis=0) + m.loads[:, :3].sum(axis=0), 0, atol=1e-6)
    base = tuned.idx["mast_base"]
    assert res.reactions[base, :3] @ tuned.rig.axis == pytest.approx(
        tuned.mast_compression_base - m.loads[base, 2] * tuned.rig.axis[2], rel=1e-6)


def test_continuous_shroud_single_tension(tuned):
    """Each cap/intermediate is one wire over the tips: stretch of the whole path matches its tension."""
    res = tuned.fe
    for w in ("cap_port", "intermediate_stbd"):
        i = tuned.idx["cables"][w]
        L, L0 = res.cable_length[i], res.cable_L0[i]
        assert tuned.tensions[w] == pytest.approx(tuned.rig.wire_EA(w) * (L - L0) / L0, rel=1e-9)
    assert all(v < 0 for v in tuned.spreader_axial.values())  # spreaders in compression


def test_tension_to_turns_round_trip(rig, tuned):
    turns = {w: {"turns": s / rig.wire_props(w)["turn_shorten_m"]} for w, s in tuned.shorten.items()}
    r = rig.solve(RigState(turns))
    for w, T in tuned.tensions.items():
        assert r.tensions[w] == pytest.approx(T, rel=1e-6)


def test_backstay_with_locked_shrouds(rig, tuned):
    """More backstay: forestay tightens, masthead goes aft, mast bows further forward."""
    out = []
    for bs in (2000.0, 4000.0):
        r = rig.solve(RigState({**tuned.locked_controls(free=("backstay",)), "backstay": {"tension": bs}}))
        out.append(r)
    lo, hi = out
    assert hi.tensions["forestay"] > lo.tensions["forestay"]
    head = lo.idx["mast_head"]
    assert hi.offset_fwd[head] < lo.offset_fwd[head]
    assert hi.chord_offset(hi.s).max() > lo.chord_offset(lo.s).max()


def test_lateral_load_after_tune(rig):
    """A sideways pull at the hounds is applied after the dock tune with lengths locked."""
    r = rig.solve(RigState(EXAMPLE_TUNE, loads=[{"at": "mast_hounds", "force_N": [0, 500.0, 0]}]))
    assert r.offset_stbd[r.idx["mast_hounds"]] > 0
    assert r.tensions["cap_port"] > 7500.0 > r.tensions["cap_stbd"]


def test_calibration_params_change_bend(rig, tuned):
    stiff = rig.with_params(k_EI_fa=2.0).solve(RigState(EXAMPLE_TUNE))
    assert stiff.chord_offset(stiff.s).max() < tuned.chord_offset(tuned.s).max()
    bent = rig.with_params(prebend_m=0.03).solve(RigState())
    mid = bent.chord_offset(bent.rig.mast_length / 2)[0]
    assert mid == pytest.approx(0.03, rel=0.1)
