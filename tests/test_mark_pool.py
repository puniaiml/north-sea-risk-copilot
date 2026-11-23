"""Offline tests of pool marking and merging."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import config as cfg
import mark_pool as mp


@pytest.fixture
def pool(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "BENCH_DIR", tmp_path)
    monkeypatch.setattr(cfg, "POOL_PATH", tmp_path / "pool.csv")
    rows = [dict(qid=q, query=f"query {q}", doc_id=f"{q}d{i}", title=f"T{q}{i}", url=f"https://x/{q}{i}",
                 found_by="bm25+dense", snippet="text", relevant="") for q in ("Q1", "Q2") for i in range(3)]
    pd.DataFrame(rows).to_csv(cfg.POOL_PATH, index=False)
    return tmp_path


def scripted(answers):
    it = iter(answers)
    return lambda prompt: next(it)


def test_start_hides_method_and_keeps_query_order(pool):
    df = mp.start("a")
    assert "found_by" not in df.columns
    assert list(df["qid"].unique()) == ["Q1", "Q2"]
    assert df["relevant"].isna().all()


def test_mark_resume_and_open(pool):
    opened = []
    assert not mp.mark("a", ask=scripted(["1", "o", "0", "q"]), show=lambda *_: None, opener=opened.append)
    assert opened and mp.progress(pd.read_csv(mp.rater_path("a")))[0] == 2
    assert mp.mark("a", ask=scripted(["1", "1", "0", "0"]), show=lambda *_: None)
    assert mp.progress(pd.read_csv(mp.rater_path("a"))) == (6, 6)


def test_skip_leaves_row_unmarked(pool):
    mp.mark("a", ask=scripted(["s", "q"]), show=lambda *_: None)
    assert mp.progress(pd.read_csv(mp.rater_path("a")))[0] == 0


def test_kappa():
    assert mp.kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    assert abs(mp.kappa([1, 1, 0, 0], [1, 0, 1, 0])) < 1e-9


def test_merge_leaves_disagreements_blank(pool):
    for name, vals in (("a", [1, 1, 0, 0, 1, 0]), ("b", [1, 0, 0, 0, 1, 1])):
        base = pd.read_csv(cfg.POOL_PATH)[["qid", "query", "doc_id", "title", "url", "snippet"]]
        base["relevant"] = vals
        base.to_csv(mp.rater_path(name), index=False)
    res = mp.merge(["a", "b"])
    assert res["rows"] == 6 and res["unresolved"] == 2 and res["relevant"] == 2
    merged = pd.read_csv(cfg.POOL_PATH)
    assert merged["relevant"].isna().sum() == 2 and "found_by" in merged.columns
    assert (pool / "disagreements.csv").exists()
    assert 0 <= res["pairs"]["a vs b"]["agreement"] <= 1


def test_merge_rejects_unmarked(pool):
    mp.start("a")
    mp.start("b")
    with pytest.raises(SystemExit):
        mp.merge(["a", "b"])


def test_merge_single_rater(pool):
    base = pd.read_csv(cfg.POOL_PATH)[["qid", "query", "doc_id", "title", "url", "snippet"]]
    base["relevant"] = [1, 0, 0, 1, 1, 0]
    base.to_csv(mp.rater_path("solo"), index=False)
    res = mp.merge(["solo"])
    assert res["unresolved"] == 0 and res["relevant"] == 3 and res["pairs"] == {}
    assert pd.read_csv(cfg.POOL_PATH)["relevant"].tolist() == [1, 0, 0, 1, 1, 0]
