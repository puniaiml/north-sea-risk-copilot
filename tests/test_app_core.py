"""Tests of the app logic. They use the committed model and test sample but no network or MiniLM."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import app_core as core
import config as cfg

TH = cfg.THRESHOLD_SETS["cited"]


@pytest.fixture(scope="module")
def art():
    return core.load_artifacts(cfg.MODELS_DIR)


@pytest.fixture(scope="module")
def sample():
    return core.load_sample()


def test_level_boundaries():
    assert core.level(1.49, 1.5, 2.0) == 0
    assert core.level(1.5, 1.5, 2.0) == 1
    assert core.level(2.0, 1.5, 2.0) == 2


def test_conditions_use_threshold_wind_column():
    row = pd.Series({"wind_speed_100m": 16.0, "wind_gusts_10m": 5.0, "wave_height": 1.6})
    c = core.conditions(row, TH)
    assert c["wind"]["value"] == 16.0 and c["wind"]["level"] == 2
    assert c["wave"]["level"] == 1


@pytest.mark.parametrize("name,label", [
    ("wave_height", "Wave height"),
    ("wind_speed_100m_max_24h", "Wind speed at 100 m, highest in last 24 h"),
    ("wave_height_lag_3h", "Wave height, 3 h ago"),
    ("pressure_delta_12h", "Air pressure, 12 h change"),
    ("wave_delta_1h", "Wave height, 1 h change"),
    ("wind_gusts_10m_std_6h", "Wind gusts at 10 m, 6 h variability"),
    ("others_wave_max", "Other sites: highest wave height"),
])
def test_feature_label(name, label):
    assert core.feature_label(name) == label


def test_every_model_feature_has_a_readable_label(art):
    for f in art.features:
        assert "_" not in core.feature_label(f), f


def test_hazard_group():
    assert core.hazard_group("others_wave_mean") == "wave"
    assert core.hazard_group("wind_gusts_10m_max_6h") == "wind"
    assert core.hazard_group("marine_stress_index") is None
    assert core.hazard_group("hour_sin") is None


def test_dominant_hazard_from_drivers():
    row = pd.Series({"wind_speed_100m": 5.0, "wave_height": 0.5})
    drive = pd.Series({"wave_height_max_6h": 0.3, "wind_speed_100m": 0.1, "hour_sin": 0.5})
    assert core.dominant_hazard(drive, row, TH) == "wave"
    drive = pd.Series({"wave_height_max_6h": -0.3, "wind_gusts_10m": 0.2})
    assert core.dominant_hazard(drive, row, TH) == "wind"


def test_dominant_hazard_falls_back_to_conditions():
    drive = pd.Series({"wave_height": -0.1, "wind_speed_100m": -0.1})
    row = pd.Series({"wind_speed_100m": 15.0, "wave_height": 0.5})
    assert core.dominant_hazard(drive, row, TH) == "wind"
    row = pd.Series({"wind_speed_100m": 3.0, "wave_height": 1.9})
    assert core.dominant_hazard(drive, row, TH) == "wave"


def test_hazard_query():
    assert core.hazard_query(0, "wave") is None
    for cls in (1, 2):
        for hz in ("wave", "wind"):
            assert core.hazard_query(cls, hz)
    assert "swell" in core.hazard_query(2, "wave")


SUGGESTED = {"wave": ["a1", "a2", "a3", "a4", "a5"], "wind": ["b1", "b2", "b3"], "general": ["c1"]}


def test_suggestions_file_is_valid_and_differs_from_the_benchmark():
    table = core.load_suggestions()
    assert set(table) == {"wave", "wind", "general"}
    assert all(len(group) >= 4 for group in table.values())
    every = [q.lower() for group in table.values() for q in group]
    assert len(set(every)) == len(every)
    assert not set(every) & {q.lower() for q in pd.read_csv(cfg.QUERIES_PATH)["query"]}


def test_load_suggestions_tolerates_bad_files(tmp_path):
    p = tmp_path / "s.csv"
    assert core.load_suggestions(p) == {}
    for lines in ([""], ["topic,text", "wave,x"], ["hazard,query"]):
        p.write_text("\n".join(lines))
        assert core.load_suggestions(p) == {}
    p.write_bytes(b"hazard,query\nwave,caf\xe9\n")  # not UTF-8
    assert core.load_suggestions(p) == {}


def test_load_suggestions_cleans_rows(tmp_path):
    p = tmp_path / "s.csv"
    p.write_text("\n".join(["hazard,query", " Wave ,one ", "wave,", "wave,one", 'wind,"two, three"']), encoding="utf-8-sig")
    assert core.load_suggestions(p) == {"wave": ["one"], "wind": ["two, three"]}


def test_suggestions_follow_the_forecast():
    s = core.suggestions
    assert s(SUGGESTED, 0, "wave", both=True) == ["c1"]
    assert s(SUGGESTED, 2, "wave") == ["a1", "a2", "a3", "a4"]
    assert s(SUGGESTED, 1, "wind") == ["b1", "b2", "b3"]
    assert s(SUGGESTED, 2, "wind", both=True) == ["b1", "b2", "a1", "a2"]
    assert s({}, 2, "wave", both=True) == []


def test_both_hazards_needs_wind_and_waves_at_warning():
    row = pd.Series({"wind_speed_100m": TH["wind_warn"], "wave_height": TH["wave_warn"]})
    assert core.both_hazards(row, TH)
    assert not core.both_hazards(row.replace(TH["wave_warn"], 0.5), TH)


def test_sample_has_every_feature(art, sample):
    assert not [f for f in art.features if f not in sample.columns]
    assert {"time", "site", "y", "y_persist"} <= set(sample.columns)


def test_predictions(art, sample):
    out = core.add_predictions(art, sample.head(50))
    p = out[["p_normal", "p_warning", "p_critical"]].to_numpy()
    assert np.allclose(p.sum(axis=1), 1, atol=1e-5)
    assert (out["pred"] == p.argmax(axis=1)).all()


def test_predict_rejects_missing_features(art, sample):
    with pytest.raises(ValueError):
        core.predict_proba(art, sample.drop(columns=[art.features[0]]))


def test_drivers_shape(art, sample):
    d = core.drivers(art, sample.iloc[0], 2, k=5)
    assert len(d) == 5 and set(d.index) <= set(art.features)
    assert (d.abs().diff().dropna() <= 1e-9).all()


def test_sites_frame_highlights_one_site():
    df = core.sites_frame("EOWDC")
    assert df["selected"].sum() == 1
    assert df.iloc[-1]["site"] == "EOWDC"  # drawn last so it sits on top
    assert (df["radius"].iloc[:-1] < df["radius"].iloc[-1]).all()
    assert len(df) == len(cfg.SITES)
    assert (df["status"] == "no forecast").all()


def test_sites_frame_colours_by_forecast():
    hour = pd.DataFrame([dict(site="EOWDC", pred=2, p_normal=0.0, p_warning=0.1, p_critical=0.9),
                         dict(site="Teesside", pred=0, p_normal=0.8, p_warning=0.2, p_critical=0.0)])
    df = core.sites_frame("EOWDC", hour).set_index("site")
    assert df.loc["EOWDC", "color"] == core.CLASS_RGB[2]
    assert df.loc["EOWDC", "status"] == "Critical (90%)"
    assert df.loc["Teesside", "color"] == core.CLASS_RGB[0]
    assert df.loc["MorayEast", "color"] == core.NO_DATA_RGB


def test_sites_frame_ring_contrasts_with_basemap():
    assert core.sites_frame("EOWDC").iloc[-1]["outline"] == [20, 20, 20]
    assert core.sites_frame("EOWDC", dark=True).iloc[-1]["outline"] == [255, 255, 255]


def test_every_site_has_a_row_at_every_sampled_hour(sample):
    assert (sample.groupby("time")["site"].nunique() == len(cfg.SITES)).all()


def test_outcome_masks():
    df = pd.DataFrame(dict(y=[2, 2, 1, 0, 1], pred=[2, 1, 2, 2, 1]))
    got = {k: list(df.index[m]) for k, m in core.outcome_masks(df).items()}
    assert got == {"All": [0, 1, 2, 3, 4], "Critical": [0, 1], "Missed Critical": [1], "False alarm": [2, 3]}
    assert set(core.EMPTY_OUTCOME) == set(got)


TERMS = {"transfer", "swell", "vessel"}


def test_trim_drops_partial_sentences():
    text, a, b = core.trim_to_sentences("he contacted the coastguard. The crew boarded. It was dispatched his ve")
    assert text == "The crew boarded." and a and b


def test_trim_keeps_whole_text_and_single_fragment():
    assert core.trim_to_sentences("The crew boarded. All were safe.") == ("The crew boarded. All were safe.",
                                                                          False, False)
    assert core.trim_to_sentences("only a fragment")[0] == "only a fragment"


def test_highlight_bolds_terms_and_plurals_and_escapes():
    out = core.highlight("Vessels in heavy swell cost $5m * 2", TERMS)
    assert "**Vessels**" in out and "**swell**" in out
    assert "\\$5m" in out and "\\*" in out
    assert core.highlight("1. First point", TERMS).startswith("1\\.")


def test_best_sentences_prefers_matches_in_order():
    text = "Weather was fine. The transfer began in swell. Lunch was served. The vessel left."
    assert core.best_sentences(text, TERMS, n=2) == ["The transfer began in swell.", "The vessel left."]
    assert core.best_sentences("No match here. None here.", TERMS, n=1) == ["No match here."]


def test_format_passage_builds_bullets_and_paragraphs():
    text = "The plan was to: ■ check the swell; ■ stop the transfer. The vessel waited. It left."
    md = core.format_passage(text, TERMS, cut_start=True)
    assert md.startswith("…The plan was to:")
    assert "- check the **swell**\n- stop the **transfer**." in md
    assert md.endswith("The **vessel** waited. It left.")


def test_format_passage_ellipsis_beside_list_stays_separate():
    md = core.format_passage("■ one item; ■ two items", TERMS, cut_start=True, cut_end=True)
    assert md.startswith("…\n\n- one item") and md.endswith("\n\n…")


def test_doc_labels_and_titles():
    assert core.doc_type_label("maib_report") == "Investigation report"
    assert core.doc_type_label("new_kind") == "New kind"
    assert core.tidy_title("kp4 report") == "Kp4 report"


def test_preview_has_no_double_ellipsis():
    text = "The transfer " + "went on " * 60 + "late. The vessel left in swell."
    out = core.preview(text, TERMS, max_sentence_chars=60)
    assert "… …" not in out and out.count("…") == 1
    assert out.endswith("The **vessel** left in **swell**.")


def test_shorten():
    assert core.shorten("short", 10) == "short"
    assert core.shorten("one two three four", 10) == "one two …"


@pytest.mark.parametrize("site,label", [("DoggerBankA", "Dogger Bank A"), ("EOWDC", "EOWDC"),
                                        ("GreaterGabbard", "Greater Gabbard")])
def test_site_label(site, label):
    assert core.site_label(site) == label
