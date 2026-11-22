"""North Sea Offshore Operations Risk & Safety Copilot (CS5079 Task 1).

Replays held-out 2026 hours: a hazard forecast for the next six hours, the main drivers, and cited
HSE and MAIB passages. It is a coursework prototype and not an operational safety system.
"""

import inspect
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import config as cfg
import copilot
import explain
import retrieval

st.set_page_config(page_title="North Sea Risk Copilot", page_icon="🌊", layout="wide")

PLOT_KW = {"width": "stretch"} if "width" in inspect.signature(st.plotly_chart).parameters else {"use_container_width": True}
FRAME_KW = {"width": "stretch"} if "width" in inspect.signature(st.dataframe).parameters else {"use_container_width": True}
COLOURS = {"Normal": "#2a9d8f", "Warning": "#e9a23b", "Critical": "#d1495b"}
CONDITIONS = [("wind_speed_100m", "Wind 100 m", "m/s"), ("wind_gusts_10m", "Gusts 10 m", "m/s"), ("wave_height", "Wave height", "m"),
              ("wave_period", "Wave period", "s"), ("surface_pressure", "Pressure", "hPa"), ("temperature_2m", "Air temp", "°C")]
DISCLAIMER = ("Prototype for coursework. It forecasts when wind and wave limits are likely to be crossed in the next six hours, "
              "using reanalysis data and assumed limits. It does not predict accidents and must not be used for operational decisions.")


def plot(fig):
    """Draw a Plotly figure at full width."""
    st.plotly_chart(fig, **PLOT_KW)


def table(df, **kw):
    """Draw a dataframe at full width."""
    st.dataframe(df, hide_index=True, **FRAME_KW, **kw)


def read_json(path):
    """Parsed JSON file, or None if it does not exist."""
    path = Path(path)
    return json.load(open(path)) if path.exists() else None


def read_csv(path, **kw):
    """CSV file as a DataFrame, or None if it does not exist."""
    path = Path(path)
    return pd.read_csv(path, **kw) if path.exists() else None


@st.cache_resource(show_spinner="Loading model")
def load_model():
    """Model, scaler, feature list, training medians and the results file."""
    d = cfg.MODELS_DIR
    return (joblib.load(d / "best_model.joblib"), joblib.load(d / "scaler.joblib"), joblib.load(d / "features.joblib"),
            joblib.load(d / "feature_medians.joblib"), read_json(d / "results.json"))


@st.cache_data(show_spinner=False)
def load_sample():
    """The held-out test sample the app replays."""
    return pd.read_csv(cfg.TEST_SAMPLE_PATH, parse_dates=["time"])


@st.cache_resource(show_spinner="Loading safety corpus")
def load_search():
    """Core-only retrievers, the methods they support and the method chosen from the benchmark."""
    chunks = retrieval.load_chunks()
    embeddings, encode = None, None
    if cfg.EMBEDDINGS_PATH.exists():
        try:
            encode = retrieval.load_encoder()
            embeddings = np.load(cfg.EMBEDDINGS_PATH)
        except Exception:
            encode, embeddings = None, None
    R, available = copilot.core_retriever(chunks, embeddings, encode)
    method = copilot.best_method()
    return R, (method if method in available else available[-1]), available


def risk_badge(label, prob):
    """Coloured banner for the predicted class."""
    st.markdown(
        f"<div style='background:{COLOURS[label]};color:white;padding:14px 18px;border-radius:10px;font-size:1.4rem;"
        f"font-weight:600'>{label} risk in the next 6 hours <span style='font-weight:400;font-size:1rem'>"
        f"&nbsp;(model confidence {prob:.0%})</span></div>", unsafe_allow_html=True)


def drivers_chart(drivers):
    """Horizontal bars of the features that moved the predicted probability most."""
    d = drivers.iloc[::-1]
    fig = go.Figure(go.Bar(x=d.values, y=d.index, orientation="h", marker_color=["#d1495b" if v > 0 else "#2a9d8f" for v in d.values]))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Change in probability of the predicted class vs a typical value")
    return fig


def live_tab():
    """Replay tab: forecast, drivers and cited passages."""
    model, scaler, features, medians, _ = load_model()
    sample = load_sample()
    st.caption("Replay of held-out 2026 hours the model never saw in training or tuning.")
    c1, c2 = st.columns(2)
    site = c1.selectbox("Site", sorted(sample["site"].unique()))
    names = ["All"] + copilot.CLASS_NAMES
    truth = c2.selectbox("Show only moments where the observed worst class was", names)
    pool = sample[sample["site"] == site]
    if truth != "All":
        pool = pool[pool["y"] == names.index(truth) - 1]
    if pool.empty:
        st.info("No moments match this choice at this site. Try another class or site.")
        return
    stamps = pool["time"].sort_values().dt.strftime("%Y-%m-%d %H:%M").tolist()
    when = st.selectbox("Moment", stamps)
    row = pool[pool["time"] == pd.Timestamp(when)].iloc[0]

    proba = model.predict_proba(scaler.transform(pd.DataFrame([row[features].astype(float)], columns=features)))[0]
    pred = int(np.argmax(proba))
    label = copilot.CLASS_NAMES[pred]
    risk_badge(label, float(proba[pred]))
    left, right = st.columns([1, 1])
    with left:
        st.markdown("**Class probabilities**")
        fig = px.bar(x=copilot.CLASS_NAMES, y=proba, color=copilot.CLASS_NAMES, color_discrete_map=COLOURS)
        fig.update_layout(height=260, showlegend=False, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Probability", xaxis_title="")
        fig.update_yaxes(range=[0, 1])
        plot(fig)
        st.markdown(f"Observed worst class over the next 6 hours: **{copilot.CLASS_NAMES[int(row['y'])]}**. "
                    f"A no-model baseline that only reads the current conditions says **{copilot.CLASS_NAMES[int(row['y_persist'])]}**.")
    with right:
        st.markdown("**Current conditions**")
        shown = [(k, n, u) for k, n, u in CONDITIONS if k in row.index]
        cols = st.columns(3)
        for i, (k, n, u) in enumerate(shown):
            cols[i % 3].metric(n, f"{row[k]:.1f} {u}")
        st.caption(f"Limits used: Warning from {cfg.THRESHOLD_SETS[cfg.ACTIVE_THRESHOLDS]['wind_warn']} m/s wind at 100 m or "
                   f"{cfg.THRESHOLD_SETS[cfg.ACTIVE_THRESHOLDS]['wave_warn']} m waves, Critical from "
                   f"{cfg.THRESHOLD_SETS[cfg.ACTIVE_THRESHOLDS]['wind_crit']} m/s or {cfg.THRESHOLD_SETS[cfg.ACTIVE_THRESHOLDS]['wave_crit']} m.")

    drivers = explain.top_drivers(model, scaler, features, medians, row, pred)
    st.markdown(f"**What pushed the forecast towards {label}**")
    plot(drivers_chart(drivers))

    st.divider()
    st.subheader("Relevant safety guidance")
    note = st.text_area("Technician note (optional)", placeholder="For example: crew transfer planned at 14:00 to the turbine")
    R, method, available = load_search()
    query = copilot.build_query(pred, list(drivers.index), note)
    st.caption(f"Search text: {query}")
    st.caption(f"Retrieval method: {method}" + ("" if len(available) > 1 else " (keyword only, embedding model not available here)"))
    passages = R.top_chunks(method, query, k=3, core_only=False)
    for i, c in enumerate(passages, 1):
        with st.expander(f"{i}. {c.get('title', c['doc_id'])}", expanded=i == 1):
            text = c["text"]
            st.write(text if len(text) < 900 else text[:900].rsplit(" ", 1)[0] + " ...")
            st.markdown(f"Source: [{c.get('source', 'link')}]({c['url']})")
    st.warning(DISCLAIMER)


def eda_tab():
    """Data tab: summaries built by build_eda_summary.py."""
    facts = read_json(cfg.EDA_DIR / "eda_facts.json")
    if not facts:
        st.info("Run `python src/build_eda_summary.py` and commit data/processed/eda to show this tab.")
        return
    a, b, c, d = st.columns(4)
    a.metric("Hourly rows", f"{facts['rows']:,}")
    b.metric("Sites", len(facts["sites"]))
    c.metric("Period", f"{facts['first_hour'][:4]} to {facts['last_hour'][:4]}")
    d.metric("Safety documents", f"{facts.get('documents', 0):,}")
    st.caption("Weather from Open-Meteo (ERA5 reanalysis, ERA5-Ocean waves). Safety text from MAIB and HSE public pages.")

    monthly = read_csv(cfg.EDA_DIR / "eda_monthly.csv")
    if monthly is not None:
        st.subheader("Seasonal pattern by site")
        metric = st.selectbox("Measure", ["wind_mean", "wave_mean", "wave_p95", "gust_p95"], index=2)
        plot(px.line(monthly, x="month", y=metric, color="site", markers=True))
    share = read_csv(cfg.EDA_DIR / "eda_class_share.csv")
    if share is not None:
        st.subheader("Class balance")
        site = st.selectbox("Site", ["All sites"] + sorted(share["site"].unique()))
        s = share if site == "All sites" else share[share["site"] == site]
        s = s.groupby(["year", "class"], as_index=False)["hours"].sum()
        s["share"] = s["hours"] / s.groupby("year")["hours"].transform("sum")
        fig = px.bar(s, x="year", y="share", color="class", color_discrete_map=COLOURS, category_orders={"class": copilot.CLASS_NAMES})
        plot(fig)
    corr = read_csv(cfg.EDA_DIR / "eda_corr.csv", index_col=0)
    if corr is not None:
        st.subheader("Wave height correlation between sites")
        plot(px.imshow(corr, zmin=0.5, zmax=1, color_continuous_scale="Blues", text_auto=".2f"))
        st.caption("High correlation means the sites are not independent, so results across sites should not be read as separate trials.")
    corpus = read_csv(cfg.EDA_DIR / "eda_corpus.csv")
    if corpus is not None:
        st.subheader("Safety corpus")
        table(corpus)
        st.caption(f"{facts.get('core_documents', 0)} documents ({facts.get('core_chunks', 0):,} chunks) form the offshore-focused core that the copilot searches.")


def percent(x):
    """Format a fraction as a three decimal number."""
    return f"{x:.3f}"


def benchmark_tab():
    """Model and retrieval evidence, with the written justification and limitations."""
    _, _, _, _, res = load_model()
    st.subheader("Forecast model comparison")
    rows = [dict(model=n, validation_macro_F1=r["val_macro_f1"], test_macro_F1=r["test_macro_f1"], test_critical_recall=r["test_crit_recall"],
                 latency_ms=r.get("latency_ms"), size_mb=r.get("size_mb")) for n, r in res["models"].items()]
    table(pd.DataFrame(rows).round(3))
    st.caption(f"Winner: {res['winner']} ({res.get('feature_set', 'base')} features). Chosen on the validation year only, then scored once on 2026. "
               f"Rule: {res['selection_rule']}")

    cm = np.array(res["models"][res["winner"]]["cm_test"])
    fig = px.imshow(cm, x=copilot.CLASS_NAMES, y=copilot.CLASS_NAMES, text_auto=True, color_continuous_scale="Blues",
                    labels=dict(x="Predicted", y="Observed"))
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10))
    left, right = st.columns(2)
    with left:
        st.markdown("**Confusion matrix, 2026 test year**")
        plot(fig)
    with right:
        st.markdown("**Critical recall by month, model vs baseline**")
        m = pd.DataFrame(res["errors"]["by_month"]).T.rename_axis("month").reset_index()
        m = m[m["critical_hours"] > 0].melt("month", ["model_crit_recall", "persist_crit_recall"], "series", "recall")
        plot(px.bar(m, x="month", y="recall", color="series", barmode="group"))
    st.markdown("**Results by site (2026)**")
    site = pd.DataFrame(res["errors"]["by_site"]).T.rename_axis("site").reset_index()
    table(site.round(3))

    boot = read_json(cfg.MODELS_DIR / "bootstrap_test.json")
    if boot:
        st.markdown("**Uncertainty (day-block bootstrap, 95% intervals, test year)**")
        b = pd.DataFrame([dict(model=n, macro_F1=f"{v['macro_f1']:.3f} [{v['macro_f1_ci'][0]:.3f}, {v['macro_f1_ci'][1]:.3f}]",
                               critical_recall=f"{v['crit_recall']:.3f} [{v['crit_recall_ci'][0]:.3f}, {v['crit_recall_ci'][1]:.3f}]")
                          for n, v in boot["models"].items()])
        table(b)

    st.subheader("Retrieval comparison")
    ret = read_json(cfg.RETRIEVAL_RESULTS_PATH)
    if ret:
        r = pd.DataFrame(ret["methods"]).T.rename_axis("method").reset_index()
        table(r.round(3))
        st.caption(f"{ret['queries_scored']} queries scored against pooled relevance marks. Small, self-made benchmark: read differences with care.")
        if "by_complexity" in ret:
            rows = [dict(complexity=g, method=m, **v) for g, d in ret["by_complexity"].items() for m, v in d.items()]
            table(pd.DataFrame(rows).round(3))
    else:
        st.info("Retrieval results appear here after the benchmark is run.")

    st.subheader("Why this design")
    st.write("The forecast uses boosted trees because they scored highest on the validation year while meeting the size and speed limits, "
             "and they beat a persistence baseline that assumes the next six hours look like now. Safety passages come from a hybrid of keyword "
             "and embedding search so that exact terms and paraphrases both match.")
    st.subheader("Limits")
    st.write("Limits are assumed operating thresholds from public sources, not any operator's own. Inputs are reanalysis values, not live forecasts or "
             "observations. The model predicts threshold crossings, not accidents. Warning is the weakest class, and most missed Critical hours are "
             "predicted as Warning. The eight sites are correlated. The corpus is mostly incident reports, and no document states transfer limits. "
             "The retrieval benchmark is small and self-made.")


def main():
    """Page layout."""
    st.title("North Sea Offshore Operations Risk & Safety Copilot")
    st.caption("Six-hour hazard forecast for eight North Sea sites, with cited UK HSE and MAIB guidance.")
    needed = [cfg.MODELS_DIR / "best_model.joblib", cfg.MODELS_DIR / "results.json", cfg.TEST_SAMPLE_PATH, cfg.CHUNKS_PATH]
    missing = [str(p.relative_to(cfg.ROOT)) for p in needed if not p.exists()]
    if missing:
        st.error("Missing files: " + ", ".join(missing))
        st.stop()
    t1, t2, t3 = st.tabs(["Live Copilot", "Data & EDA", "Model Benchmarks"])
    with t1:
        live_tab()
    with t2:
        eda_tab()
    with t3:
        benchmark_tab()


main()
