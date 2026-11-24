import altair as alt
import pandas as pd
import pydeck as pdk
import streamlit as st

import app_core as core
import app_data

art = app_data.artifacts()
th = art.thresholds
horizon = art.results["horizon_hours"]
f = app_data.forecast(st.session_state.site, st.session_state.time)
row, pred = f["row"], f["pred"]
probs = [float(row["p_normal"]), float(row["p_warning"]), float(row["p_critical"])]
BADGE = ["green", "orange", "red"]
ICON = [":material/check_circle:", ":material/warning:", ":material/dangerous:"]
HAZARD_TEXT = {"wave": "rising waves", "wind": "strong wind"}


st.session_state.setdefault("map_version", 0)
map_key = f"site_map_{st.session_state.map_version}"


def pick_site_from_map():
    """Switch the sidebar site to the marker the user clicked.

    Streamlit remembers a click by row number and fades the other markers, which goes stale once
    the site changes. Its selection is read-only, so the map gets a new key to clear it.
    """
    objs = st.session_state[map_key].selection.objects.get("sites", [])
    if objs:
        st.session_state.site = objs[0]["site"]
    st.session_state.map_version += 1


# st.title("Offshore operations risk advisory", anchor=False)
st.caption(f"{core.site_label(row['site'])} · {core.time_label(row['time'])} · worst conditions from the "
           f"next hour to {horizon} hours ahead")

top_left, map_right = st.columns([3, 2], gap="large")

with top_left:
    with st.container(horizontal=True):
        with st.container(border=True):
            st.markdown("**Copilot Forecast**")
            st.badge(core.CLASS_NAMES[pred], icon=ICON[pred], color=BADGE[pred])
            st.caption(f"{probs[pred]:.0%} probability · {art.results['winner']}")
        with st.container(border=True):
            p = int(row["y_persist"])
            st.markdown("**Persistence baseline**")
            st.badge(core.CLASS_NAMES[p], icon=ICON[p], color=BADGE[p])
            st.caption(f"Assumes the next {horizon} hours look like now")
        with st.container(border=True):
            y = int(row["y"])
            st.markdown("**What actually happened**")
            st.badge(core.CLASS_NAMES[y], icon=ICON[y], color=BADGE[y])
            st.caption("Model correct" if pred == y else "Model wrong")

    limits = (f"{th['wind_crit']} m/s wind or {th['wave_crit']} m waves" if pred == 2
              else f"{th['wind_warn']} m/s wind or {th['wave_warn']} m waves")
    if pred == 2:
        st.error(f"Conditions are expected to reach the Critical limit ({limits}) within {horizon} hours, "
                 f"driven mainly by {HAZARD_TEXT[f['hazard']]}. Review weather-sensitive work such as "
                 "personnel transfers and crane lifts.",
                 title="Critical conditions expected", icon=ICON[2])
    elif pred == 1:
        st.warning(f"Conditions are expected to reach the Warning limit ({limits}) within {horizon} hours, "
                   f"driven mainly by {HAZARD_TEXT[f['hazard']]}. Plan weather-sensitive work with margin.",
                   title="Marginal conditions expected", icon=ICON[1])
    else:
        st.success(f"Wind and waves are expected to stay below the Warning limits for the next {horizon} hours.",
                   title="No hazard expected", icon=ICON[0])
    if pred > 0:
        st.page_link("app_pages/guidance.py", label="See safety guidance for this forecast",
                     icon=":material/menu_book:")
    st.caption(":material/info: Decision support for coursework only. Not for operational use.")
    

with map_right:
    # st.subheader("Regional picture")
    hour_rows = app_data.sample()
    hour_rows = hour_rows[hour_rows["time"] == row["time"]]
    sites = core.sites_frame(row["site"], hour_rows, dark=st.context.theme.type == "dark")
    deck = pdk.Deck(
        map_style=None,
        # Fits all sites, Greater Gabbard (51.9 N) to Brent (61 N), in a 440 px map. Closer zooms cut one off.
        initial_view_state=pdk.ViewState(latitude=56.7, longitude=-0.4, zoom=4.0),
        # pydeck treats bare strings as expressions, so unit names need inner quotes.
        layers=[
            pdk.Layer("ScatterplotLayer", id="sites", data=sites, get_position=["lon", "lat"],
                        get_fill_color="color", get_radius="radius", radius_units="'pixels'",
                        stroked=True, get_line_color="outline", get_line_width="outline_px",
                        line_width_units="'pixels'", pickable=True),
        ],
        tooltip={"text": "{label}\n{status}\nClick to select"},
    )
    st.pydeck_chart(deck, height=440, key=map_key, on_select=pick_site_from_map,
                    selection_mode="single-object",
                    alt=f"Map of the eight sites coloured by forecast class, {core.site_label(row['site'])} selected")
    with st.container(horizontal=True, vertical_alignment="center"):
        for i, name in enumerate(core.CLASS_NAMES):
            st.badge(f"{name} ({int((hour_rows['pred'] == i).sum())})", icon=ICON[i], color=BADGE[i])
    # st.caption("Each site's forecast for the same hour. Black ring: selected site. Click a marker to switch.")
    

st.divider()
class_col, chart_col = st.columns([4, 2], gap="large")

with class_col:
    st.subheader("Class probabilities")
    prob_df = pd.DataFrame({"Class": core.CLASS_NAMES, "Probability": probs})
    st.altair_chart(
        alt.Chart(prob_df).mark_bar().encode(
            x=alt.X("Probability:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
            y=alt.Y("Class:N", sort=core.CLASS_NAMES, title=None),
            color=alt.Color("Class:N", scale=alt.Scale(domain=core.CLASS_NAMES, range=core.CLASS_COLORS),
                            legend=None),
            tooltip=["Class", alt.Tooltip("Probability:Q", format=".1%")],
        ).properties(height=140),
        alt=f"Forecast probabilities, highest for {core.CLASS_NAMES[pred]} at {probs[pred]:.0%}",
    )

    st.subheader("Why this forecast")
    d = f["drivers"]
    drive_df = pd.DataFrame({
        "Feature": [core.feature_label(n) for n in d.index],
        "Effect": d.values,
        "Value": [float(row[n]) for n in d.index],
        "Typical": [float(art.medians[n]) for n in d.index],
    })
    drive_df["Direction"] = drive_df["Effect"].map(lambda v: "Raises" if v > 0 else "Lowers")
    st.altair_chart(
        alt.Chart(drive_df).mark_bar().encode(
            x=alt.X("Effect:Q", title=f"Change in P({core.CLASS_NAMES[pred]}) versus a typical value",
                    axis=alt.Axis(format="+.0%")),
            y=alt.Y("Feature:N", sort=None, title=None, axis=alt.Axis(labelLimit=320)),
            color=alt.Color("Direction:N", scale=alt.Scale(domain=["Raises", "Lowers"],
                                                            range=["#d64545", "#4a7bd0"]), legend=None),
            tooltip=["Feature", alt.Tooltip("Effect:Q", format="+.1%"),
                        alt.Tooltip("Value:Q", format=".2f", title="This hour"),
                        alt.Tooltip("Typical:Q", format=".2f", title="Training median")],
        ).properties(height=34 * len(drive_df) + 20),
        alt=f"Top factors behind the {core.CLASS_NAMES[pred]} forecast, led by {drive_df['Feature'].iloc[0]}",
    )
    st.caption("Each feature is swapped in turn for its training median. A bar shows how much the "
                "probability of the forecast class falls when this hour's value is replaced, "
                "so red bars are the conditions pushing the forecast towards this class.")

with chart_col:    
    st.subheader("Conditions now")
    cond = core.conditions(row, th)
    for key, unit in (("wind", "m/s"), ("wave", "m")):
        c = cond[key]
        st.metric(c["label"], f"{c['value']:.1f} {unit}", border=True,
                  help=f"Warning from {c['warn']} {unit}, Critical from {c['crit']} {unit}.")
        st.caption(f"Now at **{core.CLASS_NAMES[c['level']]}** level · Warning ≥ {c['warn']} {unit} · "
                   f"Critical ≥ {c['crit']} {unit}")
    with st.container(horizontal=True):
        st.metric("Gusts at 10 m", f"{row['wind_gusts_10m']:.1f} m/s")
        st.metric("Wave period", f"{row['wave_period']:.1f} s")
        st.metric("Pressure, 3 h change", f"{row['pressure_delta_3h']:+.1f} hPa")

    