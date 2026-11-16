"""Central configuration: paths, constants and label definitions.

Every other module imports paths/constants from here so that the whole
project can be relocated by changing PROJECT_ROOT only.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_CSV = RAW_DIR / "kazakhlawcorpus_kk_only.csv"
RAW_CSV_URL = (
    "https://huggingface.co/datasets/Arailym-tleubayeva/KazakhLawCorpus-clean/"
    "resolve/main/data/kazakhlawcorpus_kk_only.csv"
)
FULL_PARQUET = PROCESSED_DIR / "corpus.parquet"          # all rows, optimized dtypes
DATASET_PARQUET = PROCESSED_DIR / "dataset.parquet"      # labelled + split (model input)
DB_PATH = DATA_DIR / "kazlaw.db"                         # SQLite (corpus + experiments)

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

for _d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, RESULTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Reproducibility / hardware
# --------------------------------------------------------------------------
SEED = 42
CHUNKSIZE = 20_000            # rows per chunk when streaming the 337 MB CSV
# Only GPU 0 may be used on the shared machine.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# --------------------------------------------------------------------------
# Source-system placeholders that mean "missing"
# --------------------------------------------------------------------------
PLACEHOLDERS = {"", "ЖОҚ", "НЕТ", "N/A", "-", "---", "undefined", "null", "None", "nan"}

# --------------------------------------------------------------------------
# Task A: act form  (from the multi-valued, unordered `act_form` field)
# --------------------------------------------------------------------------
# Base (legal-instrument) forms in hierarchy order. When an act carries
# several base forms (e.g. joint acts "Қаулы, Шешім"), the highest one wins.
# Variants are mapped onto their base form.
FORM_ALIASES = {
    "Конституциялық Заң": "Заң",
    "Заң күші бар Жарлық": "Жарлық",
    "Нормативтік қаулы": "Қаулы",
}
FORM_PRIORITY = ["Заң", "Жарлық", "Қаулы", "Шешім", "Бұйрық", "Өкім"]
FORM_OTHER = "Басқа"          # "Other": agreements, conventions, protocols...

# --------------------------------------------------------------------------
# Task B: regulation area (from `database_section`, single-valued, curated)
# --------------------------------------------------------------------------
AREA_MIN_TRAIN_COUNT = 300    # rarer areas collapse into AREA_OTHER (decided on TRAIN only)
AREA_OTHER = "Басқа"

# --------------------------------------------------------------------------
# Temporal split (by date_adopted) - prevents look-ahead leakage
# --------------------------------------------------------------------------
TRAIN_END_YEAR = 2021         # train: year <= 2021
VAL_YEAR = 2022               # val:   year == 2022
# test: year >= 2023

TASKS = {
    "form": "form_label",     # task name -> label column in DATASET_PARQUET
    "area": "area_label",
}
