"""North Sea Risk Copilot: Streamlit entry point.

Run with:
    streamlit run app.py
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import app_core as core  # noqa: E402
import app_data  # noqa: E402

st.set_page_config(page_title="North Sea Risk Advisory Copilot", page_icon=":material/waves:", layout="wide")

HOUR_FILTERS = ["All hours", "Critical outcome", "Model wrong"]

page = st.navigation(
    [
        st.Page("app_pages/forecast.py", title="Risk forecast", icon=":material/thunderstorm:", default=True,
                help="Forecast of the worst conditions in the next 6 hours for one site and hour, "
                     "with the reasons behind it and what actually happened."),
        st.Page("app_pages/guidance.py", title="Safety guidance", icon=":material/menu_book:",
                help="HSE and MAIB guidance that matches the forecast hazard, with cited passages "
                     "and links to the source documents."),
        st.Page("app_pages/performance.py", title="Model performance", icon=":material/monitoring:",
                help="Test-year results of every model against persistence, by site and month, "
                     "with bootstrap confidence intervals."),
        st.Page("app_pages/about.py", title="About", icon=":material/info:",
                help="Data sources, thresholds, method and limitations."),
    ],
    position="top",
)

df = app_data.sample()

with st.sidebar:
    st.header("North Sea Risk Advisory Copilot")
    sites = sorted(df["site"].unique())
    if "site" not in st.session_state:
        st.session_state.site = sites[0]
    st.selectbox("Location", sites, key="site", format_func=core.site_label, persist_state="session")
    st.segmented_control("Show hours", HOUR_FILTERS, key="hour_filter", default=HOUR_FILTERS[0],
                         required=True, persist_state="session")

    rows = df[df["site"] == st.session_state.site]
    if st.session_state.hour_filter == "Critical outcome":
        rows = rows[rows["y"] == 2]
    elif st.session_state.hour_filter == "Model wrong":
        rows = rows[rows["pred"] != rows["y"]]
    if rows.empty:
        st.warning("No hours match this filter at this site. Showing all hours.")
        rows = df[df["site"] == st.session_state.site]

    options = rows["time"].dt.strftime("%Y-%m-%d %H:%M").tolist()
    if st.session_state.get("time") not in options:
        st.session_state.time = options[-1]
    st.selectbox("Hour (2026 test year)", options, key="time",
                 format_func=core.time_label, persist_state="session",
                 help="Defaults to the latest hour in the sample for this site.")
    st.caption(f"{len(options)} of {int((df['site'] == st.session_state.site).sum())} sampled hours. "
               "The model never saw 2026 during training or selection.")

page.run()
