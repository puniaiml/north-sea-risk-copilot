"""Offline smoke test of the model comparison and the explanation helper."""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import config as cfg
import explain
import train_models as tm
from test_labels import synthetic


def test_run_and_explain(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(cfg, "TEST_SAMPLE_PATH", tmp_path / "test_sample.csv")
    monkeypatch.setattr(cfg, "TEST_SAMPLE_PER_SITE", 50)
    res = tm.run(synthetic(), "cited", quick=True)
    assert res["winner"] in {"LogReg", "RandomForest", "XGBoost"}
    assert {"Persistence", "LogReg", "RandomForest", "XGBoost"} <= set(res["models"])
    assert (tmp_path / "models" / "best_model.joblib").exists()
    saved = json.loads((tmp_path / "models" / "results.json").read_text())
    assert saved["models"]["LogReg"]["cm_test"] and len(saved["features"]) == 28
    sample = pd.read_csv(tmp_path / "test_sample.csv")
    assert {"time", "site", "y", "y_persist"} <= set(sample.columns) and len(sample) <= 100

    model = joblib.load(tmp_path / "models" / "best_model.joblib")
    scaler = joblib.load(tmp_path / "models" / "scaler.joblib")
    feats = joblib.load(tmp_path / "models" / "features.joblib")
    med = joblib.load(tmp_path / "models" / "feature_medians.joblib")
    row = sample.iloc[0]
    out = explain.top_drivers(model, scaler, feats, med, row, cls=2, k=5)
    assert len(out) == 5 and out.abs().is_monotonic_decreasing


def test_pick_winner_rule():
    res = {
        "A": dict(val_crit_recall=0.90, val_macro_f1=0.70, size_mb=10, latency_ms=5),
        "B": dict(val_crit_recall=0.88, val_macro_f1=0.75, size_mb=10, latency_ms=5),
        "C": dict(val_crit_recall=0.99, val_macro_f1=0.99, size_mb=80, latency_ms=5),
        "D": dict(val_crit_recall=0.70, val_macro_f1=0.95, size_mb=10, latency_ms=5),
    }
    assert tm.pick_winner(res) == "B"


def test_extended_run_writes_to_custom_dir(tmp_path):
    out = tmp_path / "ext"
    res = tm.run(synthetic(), "cited", quick=True, out_dir=out, feature_set="extended")
    assert res["feature_set"] == "extended" and len(res["features"]) > 28
    assert (out / "best_model.joblib").exists() and (out / "test_sample.csv").exists()


def test_critical_weight_is_recorded_and_changes_model(tmp_path):
    from test_labels import synthetic
    df = synthetic()
    a = tm.run(df, "cited", quick=True, out_dir=tmp_path / "a", feature_set="extended")
    b = tm.run(df, "cited", quick=True, out_dir=tmp_path / "b", feature_set="extended", critical_weight=1.5)
    assert a["critical_weight"] == 1.0 and b["critical_weight"] == 1.5
    assert a["models"]["XGBoost"]["test_macro_f1"] != b["models"]["XGBoost"]["test_macro_f1"] or \
        a["models"]["XGBoost"]["cm_test"] != b["models"]["XGBoost"]["cm_test"]
