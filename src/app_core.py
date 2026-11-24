"""Logic behind the Streamlit app, kept free of Streamlit so it can be tested.

The app replays the 2026 test year: each row of the test sample is one site and hour
with its features, the true label y and the persistence label y_persist. This module
loads the saved model, predicts and explains a row, checks conditions against the
thresholds, and turns a forecast into a search query for the safety guidance.
"""

import json
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
import explain

CLASS_NAMES = ["Normal", "Warning", "Critical"]
CLASS_COLORS = ["#2e9d5b", "#e8a21a", "#d64545"]

BASE_LABELS = {
    "wind_speed_10m": "Wind speed at 10 m",
    "wind_speed_100m": "Wind speed at 100 m",
    "wind_gusts_10m": "Wind gusts at 10 m",
    "wave_height": "Wave height",
    "wave": "Wave height",
    "wave_period": "Wave period",
    "surface_pressure": "Air pressure",
    "pressure": "Air pressure",
    "temperature_2m": "Air temperature",
}
SPECIAL_LABELS = {
    "gust_factor": "Gust factor (gusts / mean wind)",
    "marine_stress_index": "Wave height x wind speed",
    "hour_sin": "Hour of day (sin)",
    "hour_cos": "Hour of day (cos)",
    "month_sin": "Month (sin)",
    "month_cos": "Month (cos)",
    "wave_steepness": "Wave steepness",
    "wind_dir_sin": "Wind direction (sin)",
    "wind_dir_cos": "Wind direction (cos)",
    "wave_dir_sin": "Wave direction (sin)",
    "wave_dir_cos": "Wave direction (cos)",
    "wind_wave_align": "Wind and wave alignment",
    "site_lat": "Site latitude",
    "site_lon": "Site longitude",
    "others_wind_mean": "Other sites: mean wind",
    "others_wind_max": "Other sites: highest wind",
    "others_wave_mean": "Other sites: mean wave height",
    "others_wave_max": "Other sites: highest wave height",
    "others_wind_mean_delta_3h": "Other sites: mean wind, 3 h change",
    "others_wave_mean_delta_3h": "Other sites: mean wave height, 3 h change",
}
SUFFIX_RE = re.compile(r"^(.*)_(mean|std|max|lag|delta)_(\d+)h$")
SUFFIX_TEXT = {"mean": "{w} h mean", "std": "{w} h variability", "max": "highest in last {w} h",
               "lag": "{w} h ago", "delta": "{w} h change"}

# Search text for each (class, hazard) pair. Wording follows the group's benchmark queries.
HAZARD_QUERIES = {
    (2, "wave"): "personnel transfer from vessel to offshore structure in heavy swell and rough sea",
    (2, "wind"): "suspending crane lifts and deck work on offshore installations in strong wind and gusts",
    (1, "wave"): "weather limits for crew transfer and small workboat operations in rough sea",
    (1, "wind"): "wind limits for offshore crane lifting operations",
}


@dataclass
class Artifacts:
    """Everything train_models.py saves for one model directory."""

    model: object
    scaler: object
    features: list
    medians: pd.Series
    results: dict

    @property
    def thresholds(self):
        return self.results["thresholds"]


def load_artifacts(model_dir=None):
    """Load the model, scaler, feature list, medians and results from a model directory."""
    d = Path(model_dir or cfg.MODELS_DIR)
    with warnings.catch_warnings():
        # XGBoost warns when a pickle comes from another 3.x release. The model still predicts the same.
        warnings.filterwarnings("ignore", category=UserWarning)
        model = joblib.load(d / "best_model.joblib")
    with open(d / "results.json", encoding="utf-8") as f:
        results = json.load(f)
    return Artifacts(model, joblib.load(d / "scaler.joblib"), list(joblib.load(d / "features.joblib")),
                     joblib.load(d / "feature_medians.joblib"), results)


def load_sample(path=None):
    """Read the test sample, sorted by site and time."""
    df = pd.read_csv(path or cfg.TEST_SAMPLE_PATH, parse_dates=["time"])
    return df.sort_values(["site", "time"]).reset_index(drop=True)


def predict_proba(art, rows):
    """Class probabilities (Normal, Warning, Critical) for a frame of feature rows."""
    missing = [f for f in art.features if f not in rows.columns]
    if missing:
        raise ValueError(f"Rows are missing model features: {missing[:5]}")
    return art.model.predict_proba(art.scaler.transform(rows[art.features]))


def add_predictions(art, df):
    """Copy of df with p_normal, p_warning, p_critical and pred columns."""
    p = predict_proba(art, df)
    out = df.copy()
    for i, name in enumerate(CLASS_NAMES):
        out[f"p_{name.lower()}"] = p[:, i]
    out["pred"] = p.argmax(axis=1)
    return out


def drivers(art, row, cls, k=8):
    """Top k features by how much swapping them for the training median changes P(cls)."""
    return explain.top_drivers(art.model, art.scaler, art.features, art.medians, row, cls, k).astype(float)


def feature_label(name):
    """Readable label for a feature column name."""
    if name in SPECIAL_LABELS:
        return SPECIAL_LABELS[name]
    if name in BASE_LABELS:
        return BASE_LABELS[name]
    m = SUFFIX_RE.match(name)
    if m and m.group(1) in BASE_LABELS:
        return f"{BASE_LABELS[m.group(1)]}, {SUFFIX_TEXT[m.group(2)].format(w=m.group(3))}"
    return name.replace("_", " ")


def hazard_group(name):
    """'wave', 'wind' or None for a feature name."""
    if "wave" in name and name != "marine_stress_index":
        return "wave"
    if "wind" in name or "gust" in name:
        return "wind"
    return None


def level(value, warn, crit):
    """0 Normal, 1 Warning, 2 Critical for one value against its limits."""
    return 2 if value >= crit else 1 if value >= warn else 0


def conditions(row, th):
    """Current wind and wave values with their limits and level."""
    wind, wave = float(row[th["wind_col"]]), float(row["wave_height"])
    return {
        "wind": dict(value=wind, warn=th["wind_warn"], crit=th["wind_crit"],
                     level=level(wind, th["wind_warn"], th["wind_crit"]), label=feature_label(th["wind_col"])),
        "wave": dict(value=wave, warn=th["wave_warn"], crit=th["wave_crit"],
                     level=level(wave, th["wave_warn"], th["wave_crit"]), label="Wave height"),
    }


def dominant_hazard(drive, row, th):
    """'wave' or 'wind': the hazard that pushes the forecast up most.

    Sums the positive driver values of wave and wind features. If neither is positive,
    picks the one closer to its Critical limit right now.
    """
    pos = drive[drive > 0]
    score = {"wave": 0.0, "wind": 0.0}
    for name, v in pos.items():
        g = hazard_group(name)
        if g:
            score[g] += float(v)
    if max(score.values()) > 0:
        return max(score, key=score.get)
    c = conditions(row, th)
    return "wave" if c["wave"]["value"] / c["wave"]["crit"] >= c["wind"]["value"] / c["wind"]["crit"] else "wind"


def hazard_query(cls, hazard):
    """Search text for a forecast class and hazard, or None for Normal."""
    return HAZARD_QUERIES.get((int(cls), hazard))


def time_label(ts):
    """Short timestamp label used in the hour picker."""
    return pd.Timestamp(ts).strftime("%a %d %b %Y, %H:%M")


def load_json(path):
    """Read a JSON file, or None if it does not exist."""
    p = Path(path)
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


DOC_TYPE_LABELS = {
    "maib_report": "Investigation report",
    "hse_notice": "Offshore notice",
    "hse_bulletin": "Safety bulletin",
    "hse_offshore_page": "Offshore guidance page",
}
BULLET = "■"
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])[\"'’”)]?\s+(?=[A-Z0-9\"'‘“(])")
SENTENCE_END_RE = re.compile(r"[.!?][\"'’”)]?$")
MD_CHARS_RE = re.compile(r"([\\`*_\[\]<>#|$~])")
WORD_RE = re.compile(r"[A-Za-z0-9]+")


def doc_type_label(doc_type):
    """Plain-words label for a corpus doc_type."""
    return DOC_TYPE_LABELS.get(doc_type, doc_type.replace("_", " ").capitalize())


def tidy_title(title):
    """Title with a capital first letter ('kp4 report' becomes 'Kp4 report')."""
    title = title.strip()
    return title[:1].upper() + title[1:]


def md_escape(text):
    """Escape characters that Markdown would treat as formatting, so text shows as written."""
    return MD_CHARS_RE.sub(r"\\\1", text)


def split_sentences(text):
    """Split text into sentences on end punctuation followed by a capital, digit or quote."""
    return [s.strip() for s in SENTENCE_SPLIT_RE.split(text.strip()) if s.strip()]


def trim_to_sentences(text):
    """Drop the partial sentence at each end of a chunk.

    A chunk starting with a lower-case letter began mid-sentence, and one not ending in end
    punctuation stops mid-sentence. Returns (text, cut_start, cut_end). If trimming would
    leave nothing, the text is returned unchanged.
    """
    sents = split_sentences(text)
    cut_start = cut_end = False
    if len(sents) > 1 and sents[0][:1].islower():
        sents, cut_start = sents[1:], True
    if len(sents) > 1 and not SENTENCE_END_RE.search(sents[-1]):
        sents, cut_end = sents[:-1], True
    return " ".join(sents), cut_start, cut_end


def _is_match(word, terms):
    w = word.lower()
    return w in terms or (w.endswith("s") and w[:-1] in terms)


def highlight(text, terms):
    """Escape text for Markdown and bold every word whose lower-case form (or singular) is in terms."""
    out, last = [], 0
    for m in WORD_RE.finditer(text):
        out.append(md_escape(text[last:m.start()]))
        out.append(f"**{m.group()}**" if _is_match(m.group(), terms) else m.group())
        last = m.end()
    out.append(md_escape(text[last:]))
    # A leading "1." or "-" would otherwise start a Markdown list.
    return re.sub(r"^(\d+)\.", r"\1\\.", re.sub(r"^([-+])", r"\\\1", "".join(out)))


def best_sentences(text, terms, n=3):
    """The n sentences with the most distinct query terms, in their original order.

    Falls back to the first n sentences when no sentence contains a term.
    """
    sents = [" ".join(s.split()) for s in split_sentences(text.replace(BULLET, " "))]
    score = [len({w.lower() for w in WORD_RE.findall(s) if _is_match(w, terms)}) for s in sents]
    if not any(score):
        return sents[:n]
    keep = sorted(sorted(range(len(sents)), key=lambda i: -score[i])[:n])
    return [sents[i] for i in keep if score[i] > 0]


def shorten(text, max_chars):
    """Cut text at a word boundary so it fits max_chars, adding an ellipsis when cut."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0].rstrip(",;:") + " …"


def preview(text, terms, n=3, max_sentence_chars=300):
    """Markdown preview: the best-matching sentences, highlighted, joined by ellipses."""
    parts = [shorten(s, max_sentence_chars) for s in best_sentences(text, terms, n)]
    # A shortened sentence already ends in an ellipsis, which doubles as the separator.
    out = ""
    for i, p in enumerate(parts):
        out += highlight(p.removesuffix(" …"), terms) + (" …" if p.endswith(" …") else "")
        if i < len(parts) - 1:
            out += " " if out.endswith("…") else " … "
    return out


def format_passage(text, terms, per_paragraph=3, cut_start=False, cut_end=False):
    """Markdown for a full passage: bullet lists from '■' items, other text in short paragraphs.

    cut_start and cut_end add an ellipsis where trim_to_sentences removed a partial sentence.
    """
    parts = [p.strip() for p in text.split(BULLET)]
    blocks = []
    intro, items = parts[0], [p for p in parts[1:] if p]
    if intro:
        sents = split_sentences(intro)
        for i in range(0, len(sents), per_paragraph):
            blocks.append(highlight(" ".join(sents[i:i + per_paragraph]), terms))
    if items:
        # Text after the last bullet that runs on past its first sentence is ordinary prose again.
        tail_sents = split_sentences(items[-1])
        tail = []
        if len(tail_sents) > 1:
            items[-1], tail = tail_sents[0], tail_sents[1:]
        blocks.append("\n".join(f"- {highlight(it.rstrip(';'), terms)}" for it in items))
        for i in range(0, len(tail), per_paragraph):
            blocks.append(highlight(" ".join(tail[i:i + per_paragraph]), terms))
    # Next to a bullet list the ellipsis gets its own line, so the list still renders.
    if blocks and cut_start:
        blocks = ["…"] + blocks if blocks[0].startswith("- ") else ["…" + blocks[0]] + blocks[1:]
    if blocks and cut_end:
        blocks = blocks + ["…"] if blocks[-1].startswith("- ") else blocks[:-1] + [blocks[-1] + "…"]
    return "\n\n".join(blocks)


def site_label(site):
    """Readable site name: 'DoggerBankA' becomes 'Dogger Bank A'."""
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", site)


CLASS_RGB = [[46, 157, 91], [232, 162, 26], [214, 69, 69]]
NO_DATA_RGB = [150, 150, 150]


EMPTY_OUTCOME = {"All": "No hours", "Critical": "No Critical hours", "Missed Critical": "No missed Critical hours",
                 "False alarm": "No false alarms"}


def outcome_masks(df):
    """Row masks for the hour filter, by comparing the forecast (pred) with the true outcome (y)."""
    critical, called = df["y"] == 2, df["pred"] == 2
    return {"All": critical | True, "Critical": critical, "Missed Critical": critical & ~called,
            "False alarm": called & ~critical}


def sites_frame(selected=None, hour_rows=None, dark=False):
    """Site table for the map, selected site last so it is drawn on top.

    hour_rows holds the sample rows for one hour (site, pred, p_* columns). Each marker is
    coloured by that site's forecast class, grey when the site has no row for the hour.
    The selected site's ring contrasts with the light or dark basemap.
    """
    by_site = {} if hour_rows is None else {r["site"]: r for _, r in hour_rows.iterrows()}
    rows = []
    for s, (lat, lon) in cfg.SITES.items():
        r = by_site.get(s)
        cls = None if r is None else int(r["pred"])
        status = "no forecast" if cls is None else f"{CLASS_NAMES[cls]} ({r[f'p_{CLASS_NAMES[cls].lower()}']:.0%})"
        rows.append(dict(site=s, label=site_label(s), lat=lat, lon=lon, selected=s == selected, pred=cls,
                         status=status, color=NO_DATA_RGB if cls is None else CLASS_RGB[cls]))
    df = pd.DataFrame(rows).sort_values("selected", kind="stable").reset_index(drop=True)
    df["radius"] = np.where(df["selected"], 9, 5)
    black, white = [20, 20, 20], [255, 255, 255]
    ring, rim = (white, black) if dark else (black, white)
    df["outline"] = [ring if s else rim for s in df["selected"]]
    df["outline_px"] = np.where(df["selected"], 2, 1)
    return df
