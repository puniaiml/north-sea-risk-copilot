"""Shared settings for data collection.

Site coordinates are approximate and must be verified with check_metocean.py
before the full run. Sites should sit at least 50 km apart because the
Open-Meteo marine data is a coarse grid and nearby points can share values.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
JSON_DIR = RAW_DIR / "json"
CSV_PATH = RAW_DIR / "north_sea_metocean.csv"
LOG_PATH = RAW_DIR / "collection_log.csv"

SITES = {
    "MorayEast": (58.18, -2.72),
    "EOWDC": (57.22, -1.97),
    "FortiesArea": (57.70, 1.00),
    "Teesside": (54.70, -1.00),
    "DoggerBankA": (54.80, 2.00),
    "HornseaOne": (53.90, 1.80),
    "BrentArea": (61.00, 1.70),
    "GreaterGabbard": (51.90, 2.00),
}

FIRST_YEAR = 2020
LAST_DATE = "2026-11-10"

ATM_URL = "https://archive-api.open-meteo.com/v1/archive"
MAR_URL = "https://marine-api.open-meteo.com/v1/marine"
ATM_VARS = (
    "wind_speed_10m,wind_speed_100m,wind_gusts_10m,wind_direction_100m,"
    "temperature_2m,surface_pressure"
)
MAR_VARS = "wave_height,wave_period,wave_direction"
MAR_MODEL = "era5_ocean"

USER_AGENT = "CS5079-student-project (University of Aberdeen coursework)"
REQUEST_DELAY_S = 1.5
MAX_RETRIES = 5
TIMEOUT_S = 90

CORPUS_DIR = RAW_DIR / "corpus"
DOCS_PATH = CORPUS_DIR / "documents.jsonl"
SCRAPE_LOG_PATH = CORPUS_DIR / "scrape_log.csv"

GOVUK_BASE = "https://www.gov.uk"
GOVUK_SEARCH_URL = "https://www.gov.uk/api/search.json"
MAIB_QUERIES = [
    "wind farm", "offshore", "crew transfer", "personnel transfer", "gangway",
    "heavy weather", "swell", "crane", "lifting operation", "supply vessel",
    "standby vessel", "platform", "weather", "mooring", "helicopter",
]
MAIB_MAX_PER_QUERY = 300

HSE_HOST = "www.hse.gov.uk"
HSE_SEEDS = [
    "https://www.hse.gov.uk/offshore/notices/index.htm",
    "https://www.hse.gov.uk/safetybulletins/bulletins-archive.htm",
    "https://www.hse.gov.uk/offshore/infosheets.htm",
    "https://www.hse.gov.uk/offshore/publications.htm",
    "https://www.hse.gov.uk/offshore/topics.htm",
]
HSE_PATH_PREFIXES = ("/safetybulletins/", "/offshore/")
HSE_MAX_DOCS = 400

RELEVANT_PATTERN = (
    r"offshore|wind ?farm|wind turbine|crew transfer|personnel transfer|gangway|"
    r"walk[- ]to[- ]work|heavy weather|swell|sea state|wave|crane|lifting|"
    r"platform|installation|standby vessel|supply vessel|mooring|helicopter|"
    r"dropped object|weather"
)
SCRAPE_DELAY_S = 2.0
MIN_CHARS = 400
PDF_MAX_PAGES = 60

PROCESSED_DIR = ROOT / "data" / "processed"
CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"
CHUNK_WORDS = 200
CHUNK_OVERLAP = 40
MIN_CHUNK_WORDS = 50
REPEAT_LINE_LIMIT = 4
CORE_PATTERN = (
    r"offshore|wind ?farm|wind turbine|turbine (tower|foundation)|crew transfer|personnel transfer|"
    r"gangway|walk[- ]to[- ]work|platform supply|standby (safety )?vessel|jack-?up|fpso|"
    r"drill ?ship|drilling (rig|unit)|oil (rig|platform)|oil (and|&) gas|helideck|pipelay|windcat"
)
CORE_HEAD_WORDS = 300

HORIZON = 6
VAL_START = "2025-01-01"
TEST_START = "2026-01-01"
THRESHOLD_SETS = {
    "cited": dict(wind_col="wind_speed_100m", wind_warn=12.9, wind_crit=15.4, wave_warn=1.5, wave_crit=2.0),
    "guide_default": dict(wind_col="wind_gusts_10m", wind_warn=15.0, wind_crit=25.0, wave_warn=1.5, wave_crit=3.0),
}
ACTIVE_THRESHOLDS = "cited"
NON_FEATURE_COLUMNS = ["y", "y_persist", "wind_direction_100m", "wave_direction", "grid_lat", "grid_lon"]

MODELS_DIR = ROOT / "models"
TEST_SAMPLE_PATH = PROCESSED_DIR / "test_sample.csv"
TEST_SAMPLE_PER_SITE = 400
SEED = 42
MAX_SIZE_MB = 50
MAX_LATENCY_MS = 50
RECALL_TOLERANCE = 0.03

BENCH_DIR = ROOT / "data" / "benchmark"
QUERIES_PATH = BENCH_DIR / "queries.csv"
POOL_PATH = BENCH_DIR / "pool.csv"
RETRIEVAL_RESULTS_PATH = BENCH_DIR / "retrieval_results.json"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDINGS_PATH = PROCESSED_DIR / "embeddings.npy"
POOL_TOP = 5
RRF_K = 60
RRF_DEPTH = 100

EDA_DIR = PROCESSED_DIR / "eda"
