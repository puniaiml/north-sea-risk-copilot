"""Train and compare the hazard classifiers, pick the deployment model and export artefacts.

Models: Logistic Regression, Random Forest, XGBoost, plus the persistence baseline that
labels the next 6 hours from the current conditions.

Process
    1. Build labels and features, then split by time (train, validation 2025, test 2026).
    2. Fit the scaler on training data only. All models share the same features and balanced
       sample weights computed from the training labels.
    3. Tune a small hyperparameter grid for each model on the validation year only.
    4. Pick the winner with the selection rule below, which uses validation scores only.
    5. Only then score every model on the 2026 test year, once.
    6. Analyse the winner's errors by site and by month and export the artefacts.

Selection rule: among models that fit the size and latency limits, keep those whose
validation Critical recall is within cfg.RECALL_TOLERANCE of the best, then take the highest
validation macro-F1.

Usage:
    python src/train_models.py
    python src/train_models.py --thresholds guide_default
    python src/train_models.py --quick
    python src/train_models.py --features extended --out-dir models/extended
    python src/train_models.py --thresholds guide_default --out-dir models/sensitivity_guide_default
    python src/train_models.py --features extended --critical-weight 1.5
"""

import argparse
import json
import sys
import time
from itertools import product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
import explain
import labels

CLASSES = [0, 1, 2]
RULE_TEXT = (
    "Models within the size and latency limits are kept. Those with validation Critical recall "
    "within the tolerance of the best are shortlisted. The highest validation macro-F1 wins."
)


def macro_f1(y, p):
    """Macro-averaged F1 over the three classes."""
    return float(f1_score(y, p, average="macro", labels=CLASSES, zero_division=0))


def crit_recall(y, p):
    """Recall of the Critical class."""
    return float(recall_score(y, p, labels=[2], average=None, zero_division=0)[0])


def grids(quick):
    """Model factories and hyperparameter grids. The quick option shrinks them for smoke tests."""
    rf_trees = 50 if quick else 200
    spec = {
        "LogReg": (
            lambda p: LogisticRegression(max_iter=1000, random_state=cfg.SEED, **p),
            [dict(C=c) for c in (0.1, 1.0, 10.0)],
        ),
        "RandomForest": (
            lambda p: RandomForestClassifier(n_estimators=rf_trees, n_jobs=-1, random_state=cfg.SEED, **p),
            [dict(max_depth=d, min_samples_leaf=l) for d, l in product((8, 12), (5, 20))],
        ),
        "XGBoost": (
            lambda p: XGBClassifier(objective="multi:softprob", n_estimators=100 if quick else 300,
                                    subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                                    n_jobs=-1, random_state=cfg.SEED, **p),
            [dict(max_depth=d, learning_rate=lr) for d, lr in product((4, 6), (0.05, 0.1))],
        ),
    }
    if quick:
        spec = {k: (f, g[:1]) for k, (f, g) in spec.items()}
    return spec


def tune(factory, grid, X, y, w, Xv, yv):
    """Fit every grid entry on the training data, score on validation, return (best model, trials)."""
    trials, best, best_key = [], None, None
    for params in grid:
        t0 = time.perf_counter()
        m = factory(params)
        m.fit(X, y, sample_weight=w)
        pv = m.predict(Xv)
        trial = dict(params=params, val_macro_f1=macro_f1(yv, pv), val_crit_recall=crit_recall(yv, pv),
                     fit_seconds=round(time.perf_counter() - t0, 1))
        trials.append(trial)
        key = (trial["val_macro_f1"], trial["val_crit_recall"])
        if best_key is None or key > best_key:
            best, best_key = m, key
            best_params = params
    return best, best_params, trials


def single_row_latency_ms(model, row, repeats=200):
    """Mean time of one predict_proba call on one row, with a single thread."""
    if hasattr(model, "n_jobs"):
        model.set_params(n_jobs=1)
    model.predict_proba(row)
    t0 = time.perf_counter()
    for _ in range(repeats):
        model.predict_proba(row)
    return (time.perf_counter() - t0) / repeats * 1000


def pick_winner(res):
    """Apply the selection rule using validation scores only. Returns the winning model name."""
    cand = {k: r for k, r in res.items()
            if k != "Persistence" and r["size_mb"] <= cfg.MAX_SIZE_MB and r["latency_ms"] <= cfg.MAX_LATENCY_MS}
    if not cand:
        raise SystemExit("No model fits the size and latency limits.")
    best_rec = max(r["val_crit_recall"] for r in cand.values())
    ok = {k: r for k, r in cand.items() if r["val_crit_recall"] >= best_rec - cfg.RECALL_TOLERANCE}
    return max(ok, key=lambda k: ok[k]["val_macro_f1"])


def error_analysis(te, pred):
    """Winner versus persistence by site, and Critical recall by month."""
    te = te.assign(pred=pred)
    by_site = {}
    for s, g in te.groupby("site"):
        by_site[s] = dict(
            hours=len(g), critical_hours=int((g["y"] == 2).sum()),
            model_macro_f1=macro_f1(g["y"], g["pred"]), persist_macro_f1=macro_f1(g["y"], g["y_persist"]),
            model_crit_recall=crit_recall(g["y"], g["pred"]), persist_crit_recall=crit_recall(g["y"], g["y_persist"]))
    by_month = {}
    for m, g in te.groupby(te["time"].dt.month):
        by_month[int(m)] = dict(critical_hours=int((g["y"] == 2).sum()), model_crit_recall=crit_recall(g["y"], g["pred"]),
                                persist_crit_recall=crit_recall(g["y"], g["y_persist"]))
    missed = te[(te["y"] == 2) & (te["pred"] != 2)]
    return dict(by_site=by_site, by_month=by_month, missed_critical=int(len(missed)),
                missed_predicted_as={NAMES[k]: int(v) for k, v in missed["pred"].value_counts().items()})


NAMES = {0: "Normal", 1: "Warning", 2: "Critical"}


def run(df, set_name=None, quick=False, out_dir=None, feature_set="base", critical_weight=1.0):
    """Run the full comparison on a collected weather DataFrame and write all artefacts. Returns results.

    With a custom out_dir the artefacts and the test sample go there and the shared copies are left untouched.
    critical_weight multiplies the balanced sample weight of the Critical class for every model.
    """
    default_dir = cfg.MODELS_DIR
    out_dir = Path(out_dir) if out_dir else default_dir
    set_name = set_name or cfg.ACTIVE_THRESHOLDS
    th = cfg.THRESHOLD_SETS[set_name]
    np.random.seed(cfg.SEED)
    d = labels.build(df, th, feature_set)
    feats = labels.feature_columns(d)
    tr, va, te = labels.time_split(d)
    if quick:
        tr = tr.iloc[::4]
    print(f"Thresholds: {set_name} {th}; feature set: {feature_set}; Critical weight x{critical_weight}")
    print(f"Rows: train {len(tr)}, validation {len(va)}, test {len(te)}; features {len(feats)}")

    sc = StandardScaler().fit(tr[feats])
    X, Xv, Xt = (sc.transform(f[feats]) for f in (tr, va, te))
    y, yv, yt = tr["y"].to_numpy(), va["y"].to_numpy(), te["y"].to_numpy()
    w = compute_sample_weight("balanced", y)
    w[y == 2] *= critical_weight

    fitted, params, trials = {}, {}, {}
    for name, (factory, grid) in grids(quick).items():
        fitted[name], params[name], trials[name] = tune(factory, grid, X, y, w, Xv, yv)
        print(f"{name}: best {params[name]}, validation macro-F1 {max(t['val_macro_f1'] for t in trials[name]):.3f}")

    cand_dir = out_dir / "candidates"
    cand_dir.mkdir(parents=True, exist_ok=True)
    res = {}
    row = Xv[:1]
    for name, m in fitted.items():
        pv = m.predict(Xv)
        res[name] = dict(params=params[name], val_macro_f1=macro_f1(yv, pv), val_crit_recall=crit_recall(yv, pv),
                         latency_ms=single_row_latency_ms(m, row))
        path = cand_dir / f"{name}.joblib"
        joblib.dump(m, path, compress=3)
        res[name]["size_mb"] = path.stat().st_size / 1e6
    winner = pick_winner(res)
    print(f"Winner by the selection rule (validation only): {winner}")

    for name, m in fitted.items():
        pt = m.predict(Xt)
        res[name].update(test_macro_f1=macro_f1(yt, pt), test_crit_recall=crit_recall(yt, pt),
                         cm_test=confusion_matrix(yt, pt, labels=CLASSES).tolist())
    yp_val, yp_test = va["y_persist"].to_numpy(), te["y_persist"].to_numpy()
    res["Persistence"] = dict(val_macro_f1=macro_f1(yv, yp_val), val_crit_recall=crit_recall(yv, yp_val),
                              test_macro_f1=macro_f1(yt, yp_test), test_crit_recall=crit_recall(yt, yp_test),
                              cm_test=confusion_matrix(yt, yp_test, labels=CLASSES).tolist())

    errors = error_analysis(te, fitted[winner].predict(Xt))

    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted[winner], out_dir / "best_model.joblib", compress=3)
    joblib.dump(sc, out_dir / "scaler.joblib")
    joblib.dump(feats, out_dir / "features.joblib")
    joblib.dump(explain.feature_medians(tr, feats), out_dir / "feature_medians.joblib")

    share = lambda f: {NAMES[k]: round(float(v), 4) for k, v in f["y"].value_counts(normalize=True).sort_index().items()}
    results = dict(
        threshold_set=set_name, feature_set=feature_set, critical_weight=critical_weight, thresholds=th, horizon_hours=cfg.HORIZON, seed=cfg.SEED, quick=quick,
        versions=dict(sklearn=sklearn.__version__, xgboost=xgboost.__version__, pandas=pd.__version__),
        split=dict(train=len(tr), validation=len(va), test=len(te), train_class_share=share(tr),
                   validation_class_share=share(va), test_class_share=share(te)),
        features=feats, selection_rule=RULE_TEXT, limits=dict(max_size_mb=cfg.MAX_SIZE_MB, max_latency_ms=cfg.MAX_LATENCY_MS,
                                                           recall_tolerance=cfg.RECALL_TOLERANCE),
        trials=trials, models=res, winner=winner, errors=errors)
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    sample_path = cfg.TEST_SAMPLE_PATH if out_dir == default_dir else out_dir / "test_sample.csv"
    parts = [g.sample(min(len(g), cfg.TEST_SAMPLE_PER_SITE), random_state=cfg.SEED) for _, g in te.groupby("site")]
    sample = pd.concat(parts).sort_values(["site", "time"])
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    out = sample[["time", "site"] + feats + ["y", "y_persist"]].copy()
    out[feats] = out[feats].round(4)
    out.to_csv(sample_path, index=False)
    return results


def print_table(results):
    """Print the comparison table."""
    rows = []
    for name, r in results["models"].items():
        rows.append(dict(model=name, val_f1=r["val_macro_f1"], val_crit_rec=r["val_crit_recall"],
                         test_f1=r["test_macro_f1"], test_crit_rec=r["test_crit_recall"],
                         latency_ms=r.get("latency_ms"), size_mb=r.get("size_mb")))
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print("\nWinner:", results["winner"])
    print("Missed Critical hours:", results["errors"]["missed_critical"], results["errors"]["missed_predicted_as"])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--thresholds", default=cfg.ACTIVE_THRESHOLDS, choices=list(cfg.THRESHOLD_SETS))
    p.add_argument("--quick", action="store_true", help="small grids and fewer rows, for a smoke test")
    p.add_argument("--out-dir", help="write artefacts here instead of models/, for sensitivity or feature-set runs")
    p.add_argument("--features", default="base", choices=["base", "extended"], help="feature set")
    p.add_argument("--critical-weight", type=float, default=1.0, help="extra multiplier on the Critical class sample weight")
    a = p.parse_args()
    if not cfg.CSV_PATH.exists():
        raise SystemExit(f"{cfg.CSV_PATH} not found. Run collect_weather.py first.")
    frame = pd.read_csv(cfg.CSV_PATH, parse_dates=["time"])
    print_table(run(frame, a.thresholds, a.quick, a.out_dir, a.features, a.critical_weight))
