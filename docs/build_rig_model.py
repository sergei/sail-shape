#!/usr/bin/env python3
"""Generates rig_model.json for Sun Fast 3300 'Sun Dragon' (USA 33070) from parameters.
Edit PARAMS (or call build(overrides)) to re-run with corrected measurements.
Every parameter carries provenance: cert | user | photo_est | assumed."""
import json, math, sys, copy

def P(v, u, src, unc=None, note=None):
    d = {"value": v, "unit": u, "source": src}
    if unc is not None: d["uncertainty"] = unc
    if note: d["note"] = note
    return d

PARAMS = {
 # --- ORR certificate US43110 (2026) ---
 "I_forestay_height":   P(41.037, "ft", "cert", note="measured along mast from sheer datum"),
 "ISPA_masthead_height":P(44.318, "ft", "cert", note="spin halyard height; used as masthead; I + TL"),
 "J_fore_triangle_base":P(13.714, "ft", "cert", note="mast to forestay base at deck/sprit; ~0.25 ft mast half-depth not modelled"),
 "P_main_luff":         P(39.331, "ft", "cert"),
 "E_boom_length":       P(14.534, "ft", "cert"),
 "BAS_boom_height":     P(4.843,  "ft", "cert", note="boom above datum; consistent with P+BAS=44.17 ~ masthead"),
 "TL_top_length":       P(3.281,  "ft", "cert", note="ISPA - I"),
 "mast_weight":         P(184.1,  "lb", "cert"),
 "mast_cg_height":      P(15.659, "ft", "cert", note="assumed measured along mast from base"),
 "mast_section_lower":  P({"fore_aft_ft":0.505,"athwart_ft":0.302}, "ft", "cert", note="MDL1/MDT1 (my reading of abbreviations)"),
 "mast_section_upper":  P({"fore_aft_ft":0.436,"athwart_ft":0.302}, "ft", "cert", note="MDL2/MDT2 (my reading of abbreviations)"),
 # --- user supplied ---
 "mast_rake_aft_deg":   P(5.0, "deg", "user", note="mast axis leans aft from vertical, in x-z plane"),
 "chainplates_shared":  P(True, "bool", "user", note="cap, intermediate and lower shrouds land on the same chainplate per side"),
 # --- photo estimates (3 photos) ---
 "spreader_lower_height":     P(13.7, "ft", "photo_est", 1.0, "along mast"),
 "spreader_upper_height":     P(27.4, "ft", "photo_est", 1.0, "along mast"),
 "spreader_lower_athwart":    P(3.9,  "ft", "photo_est", 0.5, "tip distance from centerline"),
 "spreader_upper_athwart":    P(2.5,  "ft", "photo_est", 0.5, "tip distance from centerline"),
 "spreader_lower_sweep_aft":  P(2.5,  "ft", "photo_est", 0.8, "tip aft of mast axis at spreader height"),
 "spreader_upper_sweep_aft":  P(2.2,  "ft", "photo_est", 0.8, "tip aft of mast axis at spreader height"),
 "lower_shroud_mast_height":  P(13.4,  "ft", "photo_est", 0.8, "mast attachment just below lower spreader; along mast"),
 "chainplate_aft_of_mast_base":P(2.5, "ft", "photo_est", 0.8),
 "chainplate_half_span":      P(4.7,  "ft", "photo_est", 0.5, "athwartships from centerline; beam max 11.453 ft"),
 # --- assumptions ---
 "transom_attach_aft_of_mast_base": P(20.5, "ft", "assumed", 1.0, "photo-scaled; LOA 32.776 ft"),
 "transom_attach_half_span":  P(4.2, "ft", "assumed", 0.5, "split backstay to transom corners"),
 "forestay_base_height":      P(0.0, "ft", "assumed", note="bowsprit/deck datum"),
 "boom_attitude":             P("horizontal", "-", "assumed"),
}

SAILS = {
 "source": "cert",
 "main": {"desc":"ST Main","area_ft2":402.2,"rated_area_ft2":402.2,"MHB":4.724,"MUW":5.577,"MTW":7.513,"MHW":10.302,"MQW":12.500,"weight_lb":49.4},
 "jib":  {"desc":"Jib Medium","area_ft2":311.5,"rated_area_ft2":311.0,"HLP":13.714,"HLU":42.717,"HHB":0.262,"HUW":2.165,"HTW":4.003,"HHW":7.480,"HQW":10.728,"HLE":37.697},
 "asym_spinnaker": {"desc":"A1.5","area_ft2":1140.7,"SLU":50.033,"SLE":45.112,"SHW":28.379,"SFL":30.348,"tack":"centerline"},
 "ISPA_ft":44.318,"TPS_ft":17.552,"fstay_adjust":"via backstay","inner_forestay":"none","whisker_pole":False}

HULL = {
 "source": "cert",
 "LOA_ft":32.776,"beam_max_ft":11.453,"draft_sailing_ft":6.499,
 "displacement_empty_lb":8102,"displacement_sailing_lb":9218,
 "rated_crew_wgt_lb":351,"LPS_deg":117.9,"stability_index_deg":112.5,
 "righting_moment_ftlb_as_printed":{"RM2":970.8,"RM20":885.5,"RM40":637.8,"RM60":442.8,"RM90":191.4},
 "keel":"fixed","mast_material":"carbon","hull_construction":"cored"}

def build(overrides=None):
    pr = copy.deepcopy(PARAMS)
    for k, val in (overrides or {}).items():
        pr[k]["value"] = val
    v = lambda k: pr[k]["value"]
    r = math.radians(v("mast_rake_aft_deg"))
    mast = lambda h: [-math.sin(r)*h, 0.0, math.cos(r)*h]
    nodes = {}
    def add(i, xyz, kind, note=""):
        nodes[i] = {"id": i, "xyz": [round(c, 4) for c in xyz], "kind": kind}
        if note: nodes[i]["note"] = note
    add("mast_base", mast(0), "mast_axis", "origin, deck/sheer datum")
    add("mast_head", mast(v("ISPA_masthead_height")), "mast_axis", "backstay + spin halyard")
    add("mast_hounds", mast(v("I_forestay_height")), "mast_axis", "forestay + cap shroud attach")
    add("mast_upper_spr", mast(v("spreader_upper_height")), "mast_axis", "upper spreader root + intermediate attach")
    add("mast_lower_spr", mast(v("spreader_lower_height")), "mast_axis", "lower spreader root")
    add("mast_lower_shroud", mast(v("lower_shroud_mast_height")), "mast_axis", "lower shroud attach")
    gn = mast(v("BAS_boom_height")/math.cos(r))
    add("boom_gooseneck", gn, "boom")
    add("boom_clew", [gn[0]-v("E_boom_length"), 0.0, gn[2]], "boom", "assumes horizontal boom")
    add("forestay_base", [v("J_fore_triangle_base"), 0.0, v("forestay_base_height")], "deck")
    for s, sg in (("port", -1), ("stbd", 1)):
        up, lo = nodes["mast_upper_spr"]["xyz"], nodes["mast_lower_spr"]["xyz"]
        add(f"spr_upper_tip_{s}", [up[0]-v("spreader_upper_sweep_aft"), sg*v("spreader_upper_athwart"), up[2]], "spreader_tip")
        add(f"spr_lower_tip_{s}", [lo[0]-v("spreader_lower_sweep_aft"), sg*v("spreader_lower_athwart"), lo[2]], "spreader_tip")
        add(f"chainplate_{s}", [-v("chainplate_aft_of_mast_base"), sg*v("chainplate_half_span"), 0.0], "deck", "cap+intermediate+lower")
        add(f"transom_{s}", [-v("transom_attach_aft_of_mast_base"), sg*v("transom_attach_half_span"), 0.0], "deck", "backstay base")
    xyz = lambda i: nodes[i]["xyz"]
    dist = lambda a, b: math.dist(xyz(a), xyz(b))
    def ang(a, b, c):
        u = [p-q for p, q in zip(xyz(a), xyz(b))]; w = [p-q for p, q in zip(xyz(c), xyz(b))]
        d = sum(p*q for p, q in zip(u, w))/(math.hypot(*u)*math.hypot(*w))
        return round(180-math.degrees(math.acos(max(-1, min(1, d)))), 2)  # deviation from straight
    members = []
    def wire(i, role, path, contacts=()):
        segs = [round(dist(a, b), 3) for a, b in zip(path, path[1:])]
        m = {"id": i, "type": "wire", "role": role, "path": path, "segment_lengths_ft": segs,
             "total_length_ft": round(sum(segs), 3), "tension_only": True}
        if contacts:
            m["tip_contact_deflection_deg"] = {c: ang(path[path.index(c)-1], c, path[path.index(c)+1]) for c in contacts}
        members.append(m)
    members.append({"id": "mast", "type": "beam", "role": "mast", "nodes": ["mast_base", "mast_head"],
        "length_ft": round(dist("mast_base", "mast_head"), 3), "material": "carbon", "mass_lb": v("mast_weight"),
        "cg_height_ft": v("mast_cg_height"), "section_lower": v("mast_section_lower"), "section_upper": v("mast_section_upper"),
        "boundary_condition_at_base": "UNKNOWN (keel-stepped pinned vs deck-stepped); see unknowns"})
    for s in ("port", "stbd"):
        members.append({"id": f"spreader_upper_{s}", "type": "strut", "nodes": ["mast_upper_spr", f"spr_upper_tip_{s}"], "length_ft": round(dist("mast_upper_spr", f"spr_upper_tip_{s}"), 3), "compression_only": True})
        members.append({"id": f"spreader_lower_{s}", "type": "strut", "nodes": ["mast_lower_spr", f"spr_lower_tip_{s}"], "length_ft": round(dist("mast_lower_spr", f"spr_lower_tip_{s}"), 3), "compression_only": True})
        ut, lt, cp = f"spr_upper_tip_{s}", f"spr_lower_tip_{s}", f"chainplate_{s}"
        wire(f"cap_{s}", "cap_shroud", ["mast_hounds", ut, lt, cp], (ut, lt))
        wire(f"intermediate_{s}", "intermediate_shroud", ["mast_upper_spr", lt, cp], (lt,))
        wire(f"lower_{s}", "lower_shroud", ["mast_lower_shroud", cp])
        wire(f"backstay_{s}", "split_backstay", ["mast_head", f"transom_{s}"])
    wire("forestay", "forestay", ["mast_hounds", "forestay_base"])
    d = {m["id"]: m.get("total_length_ft", m.get("length_ft")) for m in members}
    fs = members[-1]
    derived = {
      "forestay_angle_from_vertical_deg": round(math.degrees(math.atan2(xyz("forestay_base")[0]-xyz("mast_hounds")[0], xyz("mast_hounds")[2]-xyz("forestay_base")[2])), 2),
      "backstay_angle_from_vertical_deg_sideview": round(math.degrees(math.atan2(xyz("mast_head")[0]-xyz("transom_stbd")[0], xyz("mast_head")[2])), 2),
      "masthead_aft_of_base_ft": round(-xyz("mast_head")[0], 3),
      "member_lengths_ft": d}
    return {
      "schema_version": "1.0",
      "meta": {"vessel": "Sun Dragon", "sail_number": "USA 33070", "class": "Jeanneau Sun Fast 3300", "orr_cert": "US43110 (issued 2026-05-18)",
               "rig": "fractional sloop, carbon mast, double swept spreaders, split backstay, no inner forestay, no runners",
               "status": "geometry is a schematic reconstruction; only items with source=cert are authoritative",
               "companion_visual": "https://claude.ai/artifact/GErRFApGJCMfQLmLx41rjk"},
      "units": {"length": "ft", "mass": "lb", "angle": "deg", "ft_to_m": 0.3048, "lb_to_kg": 0.45359237},
      "coordinate_system": {"origin": "mast axis at deck/sheer datum, on centerline", "x": "forward +", "y": "starboard +", "z": "up +",
                            "baseline_state": "upright, zero heel, no sail load, wires unloaded geometry",
                            "mast_axis": "straight line from origin raked aft by mast_rake_aft_deg in the x-z plane (no pre-bend modelled)"},
      "parameters": pr, "nodes": list(nodes.values()), "members": members, "derived": derived,
      "sails": SAILS, "hull_and_stability": HULL,
      "assumptions": [
        "Cap shroud is one continuous wire over both spreader tips (discontinuous rod rigs share the same node geometry; split the path at the tips).",
        "Intermediate shroud runs mast@upper spreader -> lower tip -> chainplate; lowers run mast -> same chainplate (user-confirmed shared chainplates).",
        "Spreaders are horizontal in side view, swept aft; tip positions from photos.",
        "Forestay attaches at the hounds node; masthead attach for backstay; boom horizontal at BAS height.",
        "I and ISPA are measured along the raked mast; vertical heights differ by <0.4% (cos 5 deg)."],
      "unknowns_needed_for_structural_models": [
        "wire/rod type, diameter, modulus (EA) and breaking loads for cap, intermediate, lower, forestay, backstay",
        "initial pretension / tuning (base tension, backstay adjuster range)",
        "mast wall thickness, laminate, EI(z), taper; spreader section/stiffness; tang and chainplate stiffness",
        "mast step (keel vs deck) and partner/gate constraints; mast pre-bend",
        "true spreader heights/lengths/angles, chainplate positions (replace photo estimates)",
        "wind/sail load cases (polars are in the ORR certificate PDF, not included here)"],
      "agent_notes": [
        "Treat source=cert and source=user as fixed; sweep photo_est/assumed parameters within their uncertainty (+/-) for sensitivity runs.",
        "Re-generate geometry with build(overrides={...}) instead of editing nodes by hand.",
        "Wires are tension-only; spreaders compression-only; contact at spreader tips (friction/pinned) is not specified.",
        "Convert to SI with units.ft_to_m before use if your solver expects metres."]}

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "rig_model.json"
    m = build()
    json.dump(m, open(out, "w"), indent=2)
    print("wrote", out); print(json.dumps(m["derived"], indent=1))
