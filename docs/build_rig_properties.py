#!/usr/bin/env python3
"""Generates rig_properties.json: structural properties for the Sun Fast 3300 rig model.

Companion to build_rig_model.py (geometry). Everything here is SI.
Every input carries provenance: cert | spec | estimate | assumed | calibrated.
Nothing in this file is calibrated yet: stiffnesses are first-principles estimates
and the `calibration.params` block holds the knobs that dockside measurements will
later fit (see rigsolver/calibration.py). Re-run with build(overrides) to change inputs."""
import copy, json, math, sys

FT, LB = 0.3048, 0.45359237

def P(v, u, src, unc=None, note=None):
    d = {"value": v, "unit": u, "source": src}
    if unc is not None: d["uncertainty"] = unc
    if note: d["note"] = note
    return d

PARAMS = {
 # --- mast laminate / section (cert dims, estimated laminate) ---
 "mast_weight_kg":          P(round(184.1 * LB, 2), "kg", "cert", note="ORR mast weight"),
 "mast_length_m":           P(round(44.318 * FT, 4), "m", "cert", note="ISPA, tube modelled from deck datum to masthead"),
 "mast_lower_fa_m":         P(round(0.505 * FT, 4), "m", "cert", note="MDL1 fore-aft depth"),
 "mast_lower_ath_m":        P(round(0.302 * FT, 4), "m", "cert", note="MDT1 athwart width"),
 "mast_upper_fa_m":         P(round(0.436 * FT, 4), "m", "cert", note="MDL2"),
 "mast_upper_ath_m":        P(round(0.302 * FT, 4), "m", "cert", note="MDT2"),
 "mast_section_shape":      P("ellipse", "-", "assumed", note="thin-walled ellipse through MDL x MDT"),
 "mast_taper_start_m":      P(round(27.4 * FT, 3), "m", "assumed", 1.5, "lower section constant up to here (upper spreader)"),
 "mast_taper_end_m":        P(round(41.037 * FT, 3), "m", "assumed", 1.0, "upper section from here up (hounds)"),
 "mast_E_axial_Pa":         P(9.0e10, "Pa", "estimate", 0.3, "effective longitudinal modulus of a production carbon mast laminate"),
 "mast_G_Pa":               P(1.5e10, "Pa", "estimate", 0.5, "only torsion; barely affects bend"),
 "mast_density_kgm3":       P(1550.0, "kg/m3", "estimate", 0.05),
 "mast_tube_mass_fraction": P(0.55, "-", "assumed", 0.15, "share of cert mast weight in the bare tube (rest: fittings, spreaders, halyards)"),
 # --- spreaders (aluminium airfoil assumed) ---
 "spreader_E_Pa":           P(7.0e10, "Pa", "assumed", 0.1, "aluminium"),
 "spreader_chord_m":        P(0.070, "m", "assumed", 0.015),
 "spreader_thickness_m":    P(0.025, "m", "assumed", 0.008),
 "spreader_wall_m":         P(0.0025, "m", "assumed", 0.001),
 "spreader_root":           P("fixed", "-", "assumed", note="bracket clamps the spreader to the mast in both planes"),
 "spreader_tip_contact":    P("sliding", "-", "user", note="shrouds are continuous over the tips; frictionless sliding modelled ('clamped' not implemented)"),
 # --- mast step ---
 "base_condition":          P("pinned", "-", "assumed", note="UNKNOWN keel- vs deck-stepped; use calibration params base_k_rot_* for partial fixity"),
}

# Dyform-type 1x19 wire assumed for all standing rigging; sizes are guesses.
WIRE_E_NOMINAL = P(1.2e11, "Pa", "estimate", 0.15, "compacted 1x19 strand, modulus on nominal circular area")
WIRES = {  # role: (diameter mm, approx break load kN)
 "cap_shroud":          (7.0, 50.8),
 "intermediate_shroud": (6.0, 37.4),
 "lower_shroud":        (7.0, 50.8),
 "forestay":            (7.0, 50.8),
 "split_backstay":      (6.0, 37.4),
}
TURN_SHORTEN_M = P(0.00254, "m/turn", "assumed", note="rigging screw, opposite-hand 1/2-20 UNF threads: 2 x 1.27 mm per turn")

CALIBRATION = {
 "status": "uncalibrated",
 "fitted_on": None,
 "measurement_sets": [],
 "params": {
  "k_EI_fa":       {"value": 1.0, "kind": "factor",   "note": "x fore-aft mast EI (whole mast)"},
  "k_EI_lat":      {"value": 1.0, "kind": "factor",   "note": "x athwartships mast EI (whole mast)"},
  "k_EI_upper":    {"value": 1.0, "kind": "factor",   "note": "x upper-section EI, both planes (taper stiffness ratio)"},
  "k_EA_wire":     {"value": 1.0, "kind": "factor",   "note": "x EA of all wires"},
  "base_k_rot_fa": {"value": 0.0, "kind": "absolute", "unit": "N*m/rad", "note": "mast-step rotational spring, fore-aft bending (0 = pinned)"},
  "base_k_rot_lat":{"value": 0.0, "kind": "absolute", "unit": "N*m/rad", "note": "mast-step rotational spring, lateral bending (0 = pinned)"},
  "prebend_m":     {"value": 0.0, "kind": "absolute", "unit": "m", "note": "stress-free forward bow of the tube, half-sine over its length"},
 },
}

def ellipse_perimeter(a, b):
    return math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))

def thin_ellipse(depth_bend, depth_other, t):
    """Second moment of a thin elliptical tube for bending in the `depth_bend` direction."""
    a, b = depth_bend / 2, depth_other / 2
    return math.pi / 4 * t * a * a * (a + 3 * b)

def section(fa, ath, t, E, G):
    a, b = fa / 2, ath / 2
    per = ellipse_perimeter(a, b)
    Am = math.pi * a * b
    return {"area_m2": per * t, "EA_N": E * per * t,
            "EI_fa_Nm2": E * thin_ellipse(fa, ath, t), "EI_lat_Nm2": E * thin_ellipse(ath, fa, t),
            "GJ_Nm2": G * 4 * Am * Am * t / per}

def build(overrides=None):
    pr = copy.deepcopy(PARAMS)
    for k, val in (overrides or {}).items():
        pr[k]["value"] = val
    v = lambda k: pr[k]["value"]
    # wall thickness from tube mass, integrating the perimeter over the (linear) taper
    L, z0, z1 = v("mast_length_m"), v("mast_taper_start_m"), v("mast_taper_end_m")
    per = lambda fa: ellipse_perimeter(fa / 2, v("mast_lower_ath_m") / 2)
    p_lo, p_up = per(v("mast_lower_fa_m")), per(v("mast_upper_fa_m"))
    perimeter_integral = p_lo * z0 + 0.5 * (p_lo + p_up) * (z1 - z0) + p_up * (L - z1)
    t = v("mast_weight_kg") * v("mast_tube_mass_fraction") / (v("mast_density_kgm3") * perimeter_integral)
    E, G = v("mast_E_axial_Pa"), v("mast_G_Pa")
    lower = section(v("mast_lower_fa_m"), v("mast_lower_ath_m"), t, E, G)
    upper = section(v("mast_upper_fa_m"), v("mast_upper_ath_m"), t, E, G)
    rel_unc = math.hypot(pr["mast_E_axial_Pa"]["uncertainty"],
                         pr["mast_tube_mass_fraction"]["uncertainty"] / v("mast_tube_mass_fraction"))
    # spreaders: thin elliptical airfoil
    c, h, w, Es = v("spreader_chord_m"), v("spreader_thickness_m"), v("spreader_wall_m"), v("spreader_E_Pa")
    spr = {"EA_N": Es * ellipse_perimeter(c / 2, h / 2) * w,
           "EI_vertical_Nm2": Es * thin_ellipse(h, c, w), "EI_horizontal_Nm2": Es * thin_ellipse(c, h, w),
           "GJ_Nm2": Es / 2.6 * 4 * (math.pi * c * h / 4) ** 2 * w / ellipse_perimeter(c / 2, h / 2)}
    wires = {}
    for role, (d_mm, brk) in WIRES.items():
        area = math.pi * (d_mm / 1000) ** 2 / 4
        wires[role] = {"diameter_mm": P(d_mm, "mm", "assumed", 1.0),
                       "EA_N": round(WIRE_E_NOMINAL["value"] * area, 1),
                       "break_load_N": P(brk * 1000, "N", "estimate", 0.1, "Dyform 316 catalogue order of magnitude"),
                       "turn_shorten_m": TURN_SHORTEN_M["value"]}
    rnd = lambda d: {k: float(f"{x:.5g}") for k, x in d.items()}
    return {
      "schema_version": "1.0",
      "meta": {"vessel": "Sun Dragon", "class": "Jeanneau Sun Fast 3300", "units": "SI (m, N, kg, Pa)",
               "geometry_file": "rig_model.json",
               "status": "first-principles estimates, NOT calibrated; treat absolute bend as +/- {:.0%}".format(rel_unc)},
      "parameters": pr,
      "wire_modulus": WIRE_E_NOMINAL,
      "derived": {
        "mast_wall_thickness_m": float(f"{t:.4g}"),
        "mast_section_lower": rnd(lower), "mast_section_upper": rnd(upper),
        "mast_EI_relative_uncertainty": round(rel_unc, 2),
        "spreader_section": rnd(spr)},
      "wires": wires,
      "calibration": CALIBRATION,
      "notes": [
        "Mast EI from cert section dims, thin-walled ellipse, wall thickness back-calculated from cert mast weight x tube_mass_fraction.",
        "Taper: lower section below mast_taper_start_m, upper above mast_taper_end_m, linear in between.",
        "Wire EA = E_nominal x nominal area; diameters are guesses until the rigging spec is known.",
        "Calibration params are applied on top of these values by rigsolver; dockside measurements fit them later."],
    }

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "rig_properties.json"
    m = build()
    json.dump(m, open(out, "w"), indent=2)
    print("wrote", out); print(json.dumps(m["derived"], indent=1))
