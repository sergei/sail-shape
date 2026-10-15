import pandas as pd
import streamlit as st

from rig_app import charts
from rig_app.scenario import Assumptions, Sailing, Tune, backstay_sweep, run
from rigsolver import Rig

st.set_page_config(page_title="SF3300 mast shape", page_icon=":material/sailing:", layout="wide")


@st.cache_resource
def load_rig() -> Rig:
    return Rig.load()


@st.cache_data(max_entries=256)
def solve(a: Assumptions, t: Tune, s: Sailing):
    try:
        return run(load_rig(), a, t, s), None
    except RuntimeError as e:
        return None, str(e)


@st.cache_data(max_entries=32)
def sweep(a: Assumptions, t: Tune) -> pd.DataFrame:
    return backstay_sweep(load_rig(), a, t, [x / 2 for x in range(0, 17)])


WIRE_LABELS = {
    "cap_port": "Cap, port", "cap_stbd": "Cap, starboard",
    "intermediate_port": "Intermediate, port", "intermediate_stbd": "Intermediate, starboard",
    "lower_port": "Lower, port", "lower_stbd": "Lower, starboard",
    "backstay_port": "Backstay, port leg", "backstay_stbd": "Backstay, starboard leg", "forestay": "Forestay",
}

# ---------------------------------------------------------------- inputs
with st.sidebar:
    st.header("Dock tune")
    st.caption("Shroud tensions as read on a tension gauge. Wires are locked at these lengths when sailing.")
    cap = st.slider("Caps (kN)", 0.0, 20.0, 7.5, 0.25, key="cap", bind="query-params")
    inter = st.slider("Intermediates (kN)", 0.0, 12.0, 4.0, 0.25, key="inter", bind="query-params")
    lower = st.slider("Lowers (kN)", 0.0, 10.0, 3.0, 0.25, key="lower", bind="query-params")
    bs_dock = st.slider("Backstay at the dock, per leg (kN)", 0.0, 8.0, 2.0, 0.25, key="bs_dock",
                        bind="query-params")
    fs = st.slider("Forestay shortened (mm)", -60, 60, 0, 5, key="fs", bind="query-params",
                   help="Change of forestay length from the reference geometry; + is shorter (less rake).")

    st.header("Sailing")
    bs = st.slider("Backstay, per leg (kN)", 0.0, 10.0, 2.0, 0.25, key="bs", bind="query-params")
    side = st.slider("Side load at hounds (N)", -2000, 2000, 0, 50, key="side", bind="query-params",
                     help="Simple stand-in for sail load: a sideways pull at the hounds, + to starboard.")

    st.header("Display")
    exag = st.select_slider("Deflection exaggeration", [1, 2, 5, 10, 20, 50], value=5, key="exag",
                            format_func=lambda v: f"×{v}")

    with st.expander("Model assumptions", icon=":material/tune:"):
        st.caption("Stand-ins until the model is calibrated against dockside measurements.")
        k_fa = st.slider("Mast stiffness, fore-aft (× estimate)", 0.5, 4.0, 1.0, 0.1, key="k_fa")
        k_lat = st.slider("Mast stiffness, sideways (× estimate)", 0.5, 4.0, 1.0, 0.1, key="k_lat")
        step = st.select_slider("Mast step fixity (N·m/rad)", [0, 1e5, 3e5, 1e6, 3e6], value=0, key="step",
                                format_func=lambda v: "pinned" if v == 0 else f"{v:.0e}",
                                help="Rotational stiffness at the step. Pinned ≈ deck-stepped; "
                                     "larger ≈ keel-stepped with a tight deck partner.")
        prebend = st.slider("Built-in pre-bend (mm)", 0, 100, 0, 5, key="prebend")

a = Assumptions(k_EI_fa=k_fa, k_EI_lat=k_lat, base_k_rot=float(step), prebend_mm=float(prebend))
t = Tune(cap_kN=cap, intermediate_kN=inter, lower_kN=lower, backstay_kN=bs_dock, forestay_shorten_mm=float(fs))
s = Sailing(backstay_kN=bs, side_load_N=float(side))

# ---------------------------------------------------------------- header
st.title("Sun Fast 3300 mast shape")
st.caption(":orange-badge[:material/warning: Uncalibrated] Mast and wire stiffness are estimates "
           "(about ±40%). Use the app to compare settings, not for absolute millimetres.")

result, error = solve(a, t, s)
if error:
    st.error("The solver found no equilibrium for these settings. With the estimated stiffness the mast "
             "probably buckles fore-aft: lower the tensions, or raise the stiffness or step fixity "
             "under Model assumptions.", title="No equilibrium found", icon=":material/error:")
    st.stop()

cur, dock, same = result["current"], result["dock"], result["same"]
m, md = cur["metrics"], dock["metrics"]
dark = st.context.theme.type == "dark"
c = charts.palette(dark)
s_goose = load_rig().mast_s["boom_gooseneck"]


def delta(key):
    return None if same else round(m[key] - md[key], 1)


with st.container(horizontal=True):
    st.metric("Forestay tension", f"{m['forestay_kN']:.1f} kN", delta("forestay_kN"), border=True,
              delta_color="off", delta_description=None if same else "vs dock tune")
    st.metric("Mast bend", f"{m['max_bend_mm']:+.0f} mm", delta("max_bend_mm"), border=True,
              delta_color="off", delta_description=None if same else "vs dock tune",
              help=f"Largest fore-aft offset from a straight halyard between gooseneck and masthead, "
                   f"at {m['max_bend_height_m']:.1f} m. + means the middle of the mast bows forward.")
    st.metric("Masthead aft", f"{m['head_aft_mm']:.0f} mm", delta("head_aft_mm"), border=True,
              delta_color="off", delta_description=None if same else "vs dock tune",
              help="Movement of the masthead from the unloaded geometry.")
    st.metric("Mast compression", f"{m['compression_kN']:.1f} kN", delta("compression_kN"), border=True,
              delta_color="off", delta_description=None if same else "vs dock tune")

if not m["stable"]:
    st.error("The solved shape is not stable.", title="Unstable", icon=":material/error:")
if m["slack"]:
    st.warning(", ".join(WIRE_LABELS[w] for w in m["slack"]), title="Slack wires", icon=":material/warning:")
if m["overloaded"]:
    st.warning(", ".join(WIRE_LABELS[w] for w in m["overloaded"]) + " above 30% of the estimated breaking load.",
               title="High wire load", icon=":material/warning:")

# ---------------------------------------------------------------- views
side_col, front_col, bend_col = st.columns([3, 1.3, 2.6])
with side_col:
    with st.container(border=True):
        st.markdown(f"**Side view** · deflection ×{exag}")
        st.altair_chart(charts.rig_view(cur, dock, same, "side", exag, c), width="content",
                        alt="Side view of the rig with the mast's fore-aft deflection exaggerated")
with front_col:
    with st.container(border=True):
        st.markdown(f"**Front view** · ×{exag}")
        st.altair_chart(charts.rig_view(cur, dock, same, "front", exag, c), width="content",
                        alt="Front view of the rig with the mast's sideways deflection exaggerated")
with bend_col:
    with st.container(border=True):
        st.markdown("**Fore-aft bend**")
        st.altair_chart(charts.bend_chart(cur, dock, same, "fore_aft", c, s_goose),
                        alt="Fore-aft mast bend against height, relative to a halyard")
    with st.container(border=True):
        st.markdown("**Sideways bend**")
        st.altair_chart(charts.bend_chart(cur, dock, same, "lateral", c, s_goose),
                        alt="Sideways mast bend against height, relative to a halyard")

# ---------------------------------------------------------------- details
view = st.segmented_control("Details", ["Wire tensions", "Backstay sweep"], default="Wire tensions",
                            key="details", label_visibility="collapsed")

if view == "Wire tensions":
    turn_m = load_rig().properties["wires"]["cap_shroud"]["turn_shorten_m"]
    tbl = cur["tensions"].assign(
        name=lambda d: d["wire"].map(WIRE_LABELS),
        dock_kN=dock["tensions"]["tension_kN"],
        turns=lambda d: d["shorten_mm"] / (1e3 * turn_m),
    )
    st.dataframe(
        tbl, hide_index=True,
        column_order=["name", "dock_kN", "tension_kN", "pct_break", "shorten_mm", "turns"],
        column_config={
            "name": st.column_config.TextColumn("Wire"),
            "dock_kN": st.column_config.NumberColumn("Dock tune (kN)", format="%.2f"),
            "tension_kN": st.column_config.NumberColumn("Current (kN)", format="%.2f"),
            "pct_break": st.column_config.ProgressColumn("Of breaking load", min_value=0, max_value=50,
                                                         format="%.0f%%", help="Estimated breaking load"),
            "shorten_mm": st.column_config.NumberColumn("Shortened (mm)", format="%+.1f",
                                                        help="Rest length vs. the unloaded reference geometry"),
            "turns": st.column_config.NumberColumn("Rigging-screw turns", format="%+.1f",
                                                   help="Shortening expressed as turns (2.54 mm per turn assumed)"),
        },
        alt="Wire tensions, share of breaking load and rest-length changes",
    )
else:
    st.caption("Every wire locked at the dock tune; only the backstay changes. Points where no "
               "equilibrium exists are left out.")
    with st.spinner("Sweeping backstay…"):
        df = sweep(a, t)
    left, right = st.columns(2)
    with left, st.container(border=True):
        st.markdown("**Forestay tension**")
        st.altair_chart(charts.sweep_chart(df, "forestay_kN", "Forestay tension (kN)", bs, c),
                        alt="Forestay tension against backstay tension")
    with right, st.container(border=True):
        st.markdown("**Mast bend**")
        st.altair_chart(charts.sweep_chart(df, "max_bend_mm", "Max fore-aft bend (mm)", bs, c),
                        alt="Maximum fore-aft mast bend against backstay tension")
