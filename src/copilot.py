"""Pure helpers behind the Streamlit app: query building, method choice and the core-only retriever."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
import retrieval

CLASS_NAMES = ["Normal", "Warning", "Critical"]
RISK_PHRASE = {0: "routine offshore operations weather monitoring", 1: "offshore operations in rising wind and waves precautions",
               2: "offshore operations in severe weather suspend transfers heavy weather"}
FEATURE_PHRASES = [
    ("gust", "wind gusts"), ("wind", "high wind"), ("steepness", "steep waves"), ("wave", "wave height swell sea state"),
    ("pressure", "falling pressure weather deterioration"), ("marine_stress", "wind and waves combined"),
]


def driver_phrases(driver_names, limit=3):
    """Plain phrases for the strongest driving features, without repeats."""
    out = []
    for name in driver_names:
        for key, phrase in FEATURE_PHRASES:
            if key in name and phrase not in out:
                out.append(phrase)
                break
        if len(out) == limit:
            break
    return out


def build_query(risk_class, driver_names, note=""):
    """Search text from the predicted class, the main drivers and the technician's note."""
    parts = [RISK_PHRASE[int(risk_class)], *driver_phrases(driver_names), (note or "").strip()]
    return " ".join(p for p in parts if p)


def best_method(results_path=None, default="hybrid"):
    """Retrieval method with the highest MRR in the benchmark results, or the default when there are none."""
    path = Path(results_path or cfg.RETRIEVAL_RESULTS_PATH)
    if not path.exists():
        return default
    try:
        methods = json.load(open(path))["methods"]
        return max(methods, key=lambda m: methods[m]["mrr"])
    except (KeyError, ValueError):
        return default


def core_retriever(chunks, embeddings, encode):
    """Retrievers over the core offshore chunks only. Without embeddings or an encoder it is BM25 only.

    Returns (retrievers, available_methods).
    """
    mask = np.array([bool(c.get("core", False)) for c in chunks])
    subset = [c for c, keep in zip(chunks, mask) if keep] or list(chunks)
    if not mask.any():
        mask = np.ones(len(chunks), dtype=bool)
    if embeddings is not None and encode is not None and len(embeddings) == len(chunks):
        return retrieval.Retrievers(subset, np.asarray(embeddings)[mask], encode), list(retrieval.METHODS)
    dummy = np.zeros((len(subset), 1), dtype=np.float32)
    return retrieval.Retrievers(subset, dummy, lambda texts: np.zeros((len(texts), 1), dtype=np.float32)), ["bm25"]
