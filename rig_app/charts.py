"""Altair chart builders for the rig app. All inputs are plain snapshots from scenario.py."""
from __future__ import annotations

import altair as alt
import pandas as pd

LIGHT = {"current": "#2a78d6", "dock": "#8a8984", "wire": "#b4b3ad", "spreader": "#6f6e69", "deck": "#6f6e69"}
DARK = {"current": "#3987e5", "dock": "#8a8984", "wire": "#55544f", "spreader": "#a3a29b", "deck": "#a3a29b"}

HALF_BEAM_M = 11.453 * 0.3048 / 2
STATES = ["Dock tune", "Current"]


def palette(dark: bool) -> dict:
    return DARK if dark else LIGHT


def _project(df: pd.DataFrame, plane: str, exag: float) -> pd.DataFrame:
    out = df.copy()
    out["h"] = (df["x"] + exag * df["ux"]) if plane == "side" else (df["y"] + exag * df["uy"])
    out["v"] = df["z"] + exag * df["uz"]
    return out


def _mast_frames(cur, dock, same):
    frames = [cur["mast"].assign(state="Current")]
    if not same:
        frames.insert(0, dock["mast"].assign(state="Dock tune"))
    return pd.concat(frames, ignore_index=True)


def _state_encodings(c, same, legend=True):
    states = ["Current"] if same else STATES
    colors = [c["current"]] if same else [c["dock"], c["current"]]
    color = alt.Color("state:N", scale=alt.Scale(domain=states, range=colors),
                      legend=alt.Legend(title=None, orient="top") if legend and not same else None)
    dash = alt.StrokeDash("state:N", scale=alt.Scale(domain=states, range=[[1, 0]] if same else [[5, 4], [1, 0]]),
                          legend=None)
    return color, dash


def rig_view(cur, dock, same, plane, exag, c, height=520):
    """Side ('side') or front ('front') view of the rig with deflections exaggerated."""
    lines = _project(cur["lines"], plane, exag)
    mast = _project(_mast_frames(cur, dock, same), plane, exag)
    deck_h = [lines["h"].min(), lines["h"].max()] if plane == "side" else [-HALF_BEAM_M, HALF_BEAM_M]
    deck = pd.DataFrame({"h": deck_h, "v": [0.0, 0.0]})

    hmin = min(lines["h"].min(), mast["h"].min(), deck_h[0]) - 0.4
    hmax = max(lines["h"].max(), mast["h"].max(), deck_h[1]) + 0.4
    vmin, vmax = -0.4, max(lines["v"].max(), mast["v"].max()) + 0.4
    width = max(140, int(height * (hmax - hmin) / (vmax - vmin)))
    x = alt.X("h:Q", scale=alt.Scale(domain=[hmin, hmax], nice=False),
              title="Fore-aft (m), forward →" if plane == "side" else "Starboard → (m)")
    y = alt.Y("v:Q", scale=alt.Scale(domain=[vmin, vmax], nice=False), title="Height (m)")

    base = alt.Chart().encode(x=x, y=y)
    wires = base.mark_line(strokeWidth=1, color=c["wire"]).encode(detail="member:N", order="order:Q").properties(
        data=lines[lines["kind"] == "wire"])
    spreaders = base.mark_line(strokeWidth=2.5, color=c["spreader"]).encode(detail="member:N").properties(
        data=lines[lines["kind"] == "spreader"])
    deck_l = base.mark_line(strokeWidth=2, color=c["deck"]).properties(data=deck)
    color, dash = _state_encodings(c, same, legend=plane == "side")
    mast_l = base.mark_line(strokeWidth=3, strokeCap="round").encode(
        color=color, strokeDash=dash, detail="state:N", order="s:Q").properties(data=mast)

    hover = alt.selection_point(on="pointerover", nearest=True, fields=["s"], empty=False, clear="pointerout")
    current = mast[mast["state"] == "Current"]
    tips = [alt.Tooltip("s:Q", title="Height on mast (m)", format=".2f"),
            alt.Tooltip("offset_fwd_mm:Q", title="Moved forward (mm)", format="+.0f"),
            alt.Tooltip("offset_stbd_mm:Q", title="Moved to starboard (mm)", format="+.0f")]
    points = base.mark_circle(size=90, color=c["current"]).encode(
        opacity=alt.condition(hover, alt.value(1), alt.value(0)), tooltip=tips
    ).add_params(hover).properties(data=current)

    return alt.layer(deck_l, wires, spreaders, mast_l, points).properties(width=width, height=height)


def bend_chart(cur, dock, same, plane, c, s_min=0.0, height=250):
    """Mast offset from a straight halyard (gooseneck to masthead) vs height."""
    col = "bend_fa_mm" if plane == "fore_aft" else "bend_lat_mm"
    data = _mast_frames(cur, dock, same)
    data = data[data["s"] >= s_min]  # the halyard only spans gooseneck to masthead
    title = "Bend vs halyard (mm), forward +" if plane == "fore_aft" else "Sideways bend vs halyard (mm), starboard +"
    x = alt.X(f"{col}:Q", title=title)
    y = alt.Y("s:Q", title="Height on mast (m)")
    color, dash = _state_encodings(c, same)
    zero = alt.Chart(pd.DataFrame({"z": [0.0]})).mark_rule(color=c["dock"], strokeWidth=1).encode(x="z:Q")
    line = alt.Chart(data).mark_line(strokeWidth=2).encode(x=x, y=y, color=color, strokeDash=dash,
                                                           detail="state:N", order="s:Q")
    hover = alt.selection_point(on="pointerover", nearest=True, fields=["s"], empty=False, clear="pointerout")
    pts = alt.Chart(data[data["state"] == "Current"]).mark_circle(size=80, color=c["current"]).encode(
        x=x, y=y, opacity=alt.condition(hover, alt.value(1), alt.value(0)),
        tooltip=[alt.Tooltip("s:Q", title="Height (m)", format=".2f"),
                 alt.Tooltip(f"{col}:Q", title="Offset (mm)", format="+.0f")]).add_params(hover)
    return alt.layer(zero, line, pts).properties(height=height)


def sweep_chart(df: pd.DataFrame, ycol: str, ytitle: str, current_kN: float, c, height=260):
    """One quantity vs backstay tension, with the current setting marked."""
    x = alt.X("backstay_kN:Q", title="Backstay tension per leg (kN)")
    y = alt.Y(f"{ycol}:Q", title=ytitle)
    hover = alt.selection_point(on="pointerover", nearest=True, fields=["backstay_kN"], empty=False,
                                clear="pointerout")
    line = alt.Chart(df).mark_line(strokeWidth=2, color=c["current"]).encode(x=x, y=y)
    pts = alt.Chart(df).mark_circle(size=80, color=c["current"]).encode(
        x=x, y=y, opacity=alt.condition(hover, alt.value(1), alt.value(0.35)),
        tooltip=[alt.Tooltip("backstay_kN:Q", title="Backstay (kN)", format=".1f"),
                 alt.Tooltip(f"{ycol}:Q", title=ytitle, format=".1f")]).add_params(hover)
    now = alt.Chart(pd.DataFrame({"b": [current_kN]})).mark_rule(color=c["dock"], strokeDash=[4, 3]).encode(x="b:Q")
    return alt.layer(now, line, pts).properties(height=height)
