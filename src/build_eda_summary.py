"""Build the small summary tables that the app's data tab reads.

The full weather CSV is too large for the repo, so this script condenses it into a few small
files under data/processed/eda: monthly climate by site, class counts by site and year, the
wave height correlation between sites, the corpus composition, and a facts file.

Usage:
    python src/build_eda_summary.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
import labels

NAMES = {0: "Normal", 1: "Warning", 2: "Critical"}


def build(df, chunks_path=None, out_dir=None):
    """Write the summary files and return the facts dictionary."""
    out_dir = Path(out_dir or cfg.EDA_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    th = cfg.THRESHOLD_SETS[cfg.ACTIVE_THRESHOLDS]
    frames = {s: labels.regular(g.assign(site=s)) for s, g in df.groupby("site")}

    monthly, classes = [], []
    for s, g in frames.items():
        m = g.groupby(g.index.month).agg(
            wind_mean=("wind_speed_100m", "mean"), wave_mean=("wave_height", "mean"),
            wave_p95=("wave_height", lambda x: x.quantile(0.95)), gust_p95=("wind_gusts_10m", lambda x: x.quantile(0.95)))
        monthly.append(m.rename_axis("month").reset_index().assign(site=s))
        y, _ = labels.targets(g, th)
        f = pd.DataFrame({"site": s, "year": g.index.year, "y": y})
        classes.append(f[f["y"] >= 0].groupby(["site", "year", "y"]).size().rename("hours").reset_index())
    pd.concat(monthly).to_csv(out_dir / "eda_monthly.csv", index=False)
    cls = pd.concat(classes)
    cls["class"] = cls["y"].map(NAMES)
    cls.to_csv(out_dir / "eda_class_share.csv", index=False)
    pd.DataFrame({s: f["wave_height"] for s, f in frames.items()}).corr().round(3).to_csv(out_dir / "eda_corr.csv")

    facts = dict(rows=int(len(df)), sites=sorted(frames), first_hour=str(df["time"].min()), last_hour=str(df["time"].max()),
                 thresholds=th)
    cpath = Path(chunks_path or cfg.CHUNKS_PATH)
    if cpath.exists():
        chunks = pd.read_json(cpath, lines=True)
        corpus = (chunks.groupby(["source", "doc_type", "core"])
                  .agg(documents=("doc_id", "nunique"), chunks=("chunk_id", "size")).reset_index())
        corpus.to_csv(out_dir / "eda_corpus.csv", index=False)
        docs = chunks.drop_duplicates("doc_id")
        facts.update(chunks=int(len(chunks)), documents=int(docs["doc_id"].nunique()),
                     core_documents=int(docs["core"].sum()), core_chunks=int(chunks["core"].sum()))
    with open(out_dir / "eda_facts.json", "w") as f:
        json.dump(facts, f, indent=2)
    return facts


if __name__ == "__main__":
    if not cfg.CSV_PATH.exists():
        raise SystemExit(f"{cfg.CSV_PATH} not found.")
    print(build(pd.read_csv(cfg.CSV_PATH, parse_dates=["time"])))
    print("Wrote summaries to", cfg.EDA_DIR)
