"""Offline tests of the app helpers, the EDA summaries and a headless run of the Streamlit app."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
import build_eda_summary as eda
import config as cfg
import copilot
import retrieval as rt
import train_models as tm
from test_labels import synthetic
from test_retrieval import TEXTS, fake_encode


def write_chunks(path):
    chunks = [dict(chunk_id=f"{d}-0", doc_id=d, title=f"Title {d}", url=f"https://x/{d}", source="MAIB", doc_type="report",
                   core=d != "d4", text=t * 8) for d, t in TEXTS.items()]
    with open(path, "w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")
    return chunks


def test_query_and_method(tmp_path):
    q = copilot.build_query(2, ["wave_height_max_6h", "wind_gusts_10m", "wave_delta_1h"], "crew transfer at 14:00")
    assert "wave height" in q and "wind gusts" in q and q.endswith("crew transfer at 14:00")
    assert copilot.best_method(tmp_path / "none.json") == "hybrid"
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"methods": {"bm25": {"mrr": 0.5}, "dense": {"mrr": 0.4}, "hybrid": {"mrr": 0.45}}}))
    assert copilot.best_method(p) == "bm25"


def test_core_retriever_modes():
    import tempfile
    chunks = write_chunks(Path(tempfile.mkdtemp()) / "c.jsonl")
    emb = fake_encode([c["text"] for c in chunks])
    R, avail = copilot.core_retriever(chunks, emb, fake_encode)
    assert avail == rt.METHODS and len(R.chunks) == 3
    R2, avail2 = copilot.core_retriever(chunks, None, None)
    assert avail2 == ["bm25"]
    assert R2.top_chunks("bm25", "helicopter helideck gusts", k=1)[0]["doc_id"] == "d3"


def test_eda_summary(tmp_path):
    cpath = tmp_path / "chunks.jsonl"
    write_chunks(cpath)
    facts = eda.build(synthetic(), cpath, tmp_path / "eda")
    assert facts["rows"] > 0 and facts["core_documents"] == 3 and facts["chunks"] == 4
    for name in ("eda_monthly.csv", "eda_class_share.csv", "eda_corr.csv", "eda_corpus.csv", "eda_facts.json"):
        assert (tmp_path / "eda" / name).exists()
    share = pd.read_csv(tmp_path / "eda" / "eda_class_share.csv")
    assert set(share["class"]) <= {"Normal", "Warning", "Critical"}


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    df = synthetic()
    mdir = tmp_path / "models"
    sample = tmp_path / "test_sample.csv"
    monkeypatch.setattr(cfg, "MODELS_DIR", mdir)
    monkeypatch.setattr(cfg, "TEST_SAMPLE_PATH", sample)
    tm.run(df, "cited", quick=True)
    chunks_path = tmp_path / "chunks.jsonl"
    chunks = write_chunks(chunks_path)
    np.save(tmp_path / "emb.npy", fake_encode([c["text"] for c in chunks]))
    monkeypatch.setattr(cfg, "CHUNKS_PATH", chunks_path)
    monkeypatch.setattr(cfg, "EMBEDDINGS_PATH", tmp_path / "emb.npy")
    monkeypatch.setattr(cfg, "EDA_DIR", tmp_path / "eda")
    monkeypatch.setattr(cfg, "RETRIEVAL_RESULTS_PATH", tmp_path / "ret.json")
    monkeypatch.setattr(rt, "load_encoder", lambda: fake_encode)
    eda.build(df, chunks_path, tmp_path / "eda")
    (tmp_path / "ret.json").write_text(json.dumps({
        "queries_scored": 5, "methods": {m: dict(hit_at_3=0.9, recall_at_3=0.8, mrr=0.7 + i / 100, mean_query_ms=5.0, queries=5)
                                         for i, m in enumerate(rt.METHODS)},
        "by_complexity": {"simple": {m: dict(hit_at_3=0.9, recall_at_3=0.8, mrr=0.7, mean_query_ms=5.0, queries=3) for m in rt.METHODS}}}))
    return tmp_path


def test_app_runs_headless(workspace):
    from streamlit.testing.v1 import AppTest
    import streamlit as st
    st.cache_resource.clear()
    st.cache_data.clear()
    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.tabs) == 3
    sites = at.selectbox[0].options
    at.selectbox[0].select(sites[-1]).run()
    assert not at.exception
    at.text_area[0].input("crew transfer to the turbine").run()
    assert not at.exception
    assert any("Search text" in c.value for c in at.caption)
    at.selectbox[1].select("Critical").run()
    assert not at.exception


def test_app_with_keyword_method_skips_embedding_model(workspace, monkeypatch):
    from streamlit.testing.v1 import AppTest
    import streamlit as st
    (workspace / "ret.json").write_text(json.dumps({
        "queries_scored": 5, "methods": {"bm25": dict(hit_at_3=1, recall_at_3=1, mrr=0.9, mean_query_ms=5.0, queries=5),
                                         "dense": dict(hit_at_3=1, recall_at_3=1, mrr=0.5, mean_query_ms=5.0, queries=5)}}))

    def boom():
        raise AssertionError("encoder must not load")

    monkeypatch.setattr(rt, "load_encoder", boom)
    st.cache_resource.clear()
    st.cache_data.clear()
    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Retrieval method: bm25" in c.value for c in at.caption)
