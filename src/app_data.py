"""Cached loaders shared by the app pages."""

import pandas as pd
import streamlit as st

import app_core as core
import config as cfg
import retrieval


@st.cache_resource(show_spinner="Loading the forecast model...")
def artifacts():
    """The deployed model and its saved files."""
    return core.load_artifacts(cfg.MODELS_DIR)


@st.cache_data(show_spinner="Scoring the test sample...")
def sample():
    """The 2026 test sample with model predictions added."""
    return core.add_predictions(artifacts(), core.load_sample())


@st.cache_resource(show_spinner="Loading the search index and MiniLM encoder (first run takes a minute)...")
def retrievers():
    """BM25, dense and hybrid search over the safety corpus."""
    return retrieval.load_retrievers()


@st.cache_data
def json_file(path):
    """A results file, or None if it does not exist yet."""
    return core.load_json(path)


def selected_row():
    """The sample row for the site and hour chosen in the sidebar."""
    df = sample()
    t = pd.Timestamp(st.session_state.time)
    return df[(df["site"] == st.session_state.site) & (df["time"] == t)].iloc[0]


@st.cache_data(max_entries=500)
def forecast(site, time):
    """Prediction, explanation and suggested search text for one site and hour."""
    df = sample()
    row = df[(df["site"] == site) & (df["time"] == pd.Timestamp(time))].iloc[0]
    art = artifacts()
    pred = int(row["pred"])
    drive = core.drivers(art, row, pred)
    hazard = core.dominant_hazard(core.drivers(art, row, 2), row, art.thresholds)
    return dict(row=row, pred=pred, drivers=drive, hazard=hazard, query=core.hazard_query(pred, hazard),
                both=core.both_hazards(row, art.thresholds))
