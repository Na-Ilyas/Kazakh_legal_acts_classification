"""Shared helpers for the model modules (tiny transformer, e5, benchmark).

Contents
--------
* dataset loading for one task (temporal split, NaN labels dropped),
* label encoding and metric computation (macro-F1 is the selection metric),
* loading the strict-test ids written by the leakage check,
* logging val / test / test_strict rows + test predictions to SQLite,
* small memory helpers (process RSS, peak RSS).

Environment overrides (used only for development / smoke tests):
    KAZLAW_DATASET   path to an alternative dataset parquet (e.g. a sample)
    KAZLAW_DB        path to an alternative SQLite file for experiment logs
"""
from __future__ import annotations

import json
import os
import resource
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src import config, experiments_db
from src import metrics as shared_metrics

DATA_COLUMNS = ["law_id", "title", "year", "split", "form_label", "area_label"]
SPLITS = ("train", "val", "test")
STRICT_IDS_CSV = config.RESULTS_DIR / "test_strict_ids.csv"
TOKEN_STATS_JSON = config.RESULTS_DIR / "token_lengths.json"
BPE_DIR = config.MODELS_DIR / "bpe_tokenizer"
E5_MODEL_NAME = "intfloat/multilingual-e5-small"
E5_PREFIX = "query: "          # e5 convention for classification / symmetric tasks
RUNS_DIR = config.PROJECT_ROOT / "runs"   # TensorBoard logs
QUICK_SUFFIX = "_quick"        # quick (smoke-test) runs never overwrite full-run results

# Dev override of the experiments DB: experiments_db.connect() binds its
# default db_path at import time, so we patch the default once here.
if os.environ.get("KAZLAW_DB"):
    experiments_db.connect.__defaults__ = (Path(os.environ["KAZLAW_DB"]),)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def dataset_path() -> Path:
    """Path of the model-input parquet (config.DATASET_PARQUET unless overridden)."""
    return Path(os.environ.get("KAZLAW_DATASET", str(config.DATASET_PARQUET)))


def load_dataset(columns: list[str] | None = None) -> pd.DataFrame:
    """Read only the needed columns of the labelled dataset."""
    path = dataset_path()
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}. Run the data preparation step first.")
    return pd.read_parquet(path, columns=columns or DATA_COLUMNS)


def load_task(task: str, max_train: int | None = None, max_eval: int | None = None,
              seed: int = config.SEED) -> dict[str, pd.DataFrame]:
    """Return {'train','val','test'} frames with columns law_id, title, year, label.

    Rows whose label is missing are dropped (area: no database_section; form: the
    acts without act_form, NaN since contract v2), so no NaN ever becomes a class.
    `max_train` / `max_eval` subsample (seeded) for quick smoke tests.
    """
    if task not in config.TASKS:
        raise ValueError(f"Unknown task {task!r}; expected one of {list(config.TASKS)}")
    label_col = config.TASKS[task]
    df = load_dataset(["law_id", "title", "year", "split", label_col])
    df = df[df[label_col].notna()]
    out: dict[str, pd.DataFrame] = {}
    for split in SPLITS:
        part = df[df["split"] == split]
        part = pd.DataFrame({
            "law_id": part["law_id"].astype(str).to_numpy(),
            "title": part["title"].astype(str).to_numpy(),
            "year": part["year"].to_numpy(),
            "label": part[label_col].astype(str).to_numpy(),
        })
        limit = max_train if split == "train" else max_eval
        if limit is not None and len(part) > limit:
            part = part.sample(n=limit, random_state=seed)
        out[split] = part.reset_index(drop=True)
    return out


def label_maps(train_labels: pd.Series | list[str]) -> tuple[list[str], dict[str, int]]:
    """Sorted class list and label->id mapping, decided on TRAIN labels only."""
    classes = sorted(set(map(str, train_labels)))
    return classes, {c: i for i, c in enumerate(classes)}


def encode_labels(labels, label2id: dict[str, int]) -> np.ndarray:
    """Map string labels to ids (unknown labels -> -1, counted as errors)."""
    return np.array([label2id.get(str(l), -1) for l in labels], dtype=np.int64)


def class_weights(y: np.ndarray, n_classes: int, kind: str = "sqrt_inv") -> np.ndarray:
    """Per-class loss weights. 'sqrt_inv' = 1/sqrt(freq), normalised to mean 1 per sample."""
    counts = np.bincount(y[y >= 0], minlength=n_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    if kind == "sqrt_inv":
        w = 1.0 / np.sqrt(counts)
    elif kind == "inv":
        w = 1.0 / counts
    else:
        raise ValueError(kind)
    w = w / (w * counts).sum() * counts.sum()   # E_sample[w] = 1
    return w.astype(np.float32)


# --------------------------------------------------------------------------
# Metrics / strict test / logging
# --------------------------------------------------------------------------
def compute_metrics(y_true, y_pred) -> dict:
    """f1_macro, accuracy, f1_weighted, n_eval via the shared src.metrics convention
    (macro-F1 averaged over the classes PRESENT in y_true)."""
    return shared_metrics.compute_metrics(y_true, y_pred)


def load_strict_ids() -> set[str] | None:
    """law_ids of the strict test subset, or None when the leakage check has not run yet."""
    try:
        ids = pd.read_csv(STRICT_IDS_CSV, dtype={"law_id": str})["law_id"]
        return set(ids.astype(str))
    except (FileNotFoundError, KeyError, pd.errors.EmptyDataError):
        return None


def with_db_retry(fn, *args, retries: int = 20, wait_s: float = 3.0, **kwargs):
    """Call a SQLite-writing function, retrying while another process holds the write lock."""
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc) and "busy" not in str(exc) or attempt == retries - 1:
                raise
            print(f"[db] busy ({exc}); retry {attempt + 1}/{retries}")
            time.sleep(wait_s)
    return None


def log_all_splits(model_name: str, family: str, task: str, preds: dict[str, pd.DataFrame],
                   common: dict, hyperparams: dict, notes: str = "") -> dict[str, dict]:
    """Log val / test / test_strict metrics and test predictions to SQLite.

    `preds[split]` is a frame with law_id, y_true, y_pred, confidence.
    `common` holds the split-independent metrics (n_params, size_mb, latency...).
    """
    results: dict[str, dict] = {}
    strict = load_strict_ids()
    frames = dict(preds)
    if strict is not None and "test" in preds:
        frames["test_strict"] = preds["test"][preds["test"]["law_id"].isin(strict)]
    for split, frame in frames.items():
        if len(frame) == 0:
            continue
        metrics = {**common, **compute_metrics(frame["y_true"], frame["y_pred"])}
        exp_id = with_db_retry(experiments_db.log_experiment, model_name, family, task, split, metrics,
                               hyperparams=hyperparams, notes=notes)
        if split == "test":
            with_db_retry(experiments_db.log_predictions, exp_id, frame["law_id"], frame["y_true"],
                          frame["y_pred"], frame["confidence"])
        results[split] = {"exp_id": exp_id, **metrics}
    return results


def prediction_frame(law_ids, y_true, probs: np.ndarray, classes: list[str]) -> pd.DataFrame:
    """Turn a probability matrix into the (law_id, y_true, y_pred, confidence) frame."""
    pred_idx = probs.argmax(axis=1)
    return pd.DataFrame({
        "law_id": list(map(str, law_ids)),
        "y_true": list(map(str, y_true)),
        "y_pred": [classes[i] for i in pred_idx],
        "confidence": probs.max(axis=1).astype(float),
    })


# --------------------------------------------------------------------------
# Misc helpers
# --------------------------------------------------------------------------
def rss_mb() -> float:
    """Current resident memory of this process in MB."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2**20
    except ImportError:          # pragma: no cover - psutil is in requirements
        return peak_rss_mb()


def peak_rss_mb() -> float:
    """Peak resident memory of this process in MB (Linux ru_maxrss is in KB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def dir_size_mb(path: Path) -> float:
    """Total size of the files under `path` (or of a single file) in MB."""
    path = Path(path)
    if path.is_file():
        return path.stat().st_size / 2**20
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 2**20


def prob_cache_dir(model_dir_name: str) -> Path:
    """artifacts/cache/<model dir>: cached class probabilities (kept out of the model dir so
    that size_mb measures only what is needed for inference)."""
    return config.ARTIFACTS_DIR / "cache" / model_dir_name


def save_probs(model_dir_name: str, split: str, probs: np.ndarray, law_ids, classes: list[str]) -> None:
    """Cache a probability matrix (float16) + row ids + class order for post-hoc methods."""
    d = prob_cache_dir(model_dir_name)
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / f"probs_{split}.npy", np.asarray(probs, dtype=np.float16))
    pd.Series(list(map(str, law_ids))).to_csv(d / f"law_ids_{split}.csv", index=False, header=["law_id"])
    save_json({"classes": list(classes)}, d / "classes.json")


def load_probs(model_dir_name: str, split: str) -> tuple[np.ndarray, list[str], list[str]] | None:
    """(probs float32, law_ids, classes) from the cache, or None if missing."""
    d = prob_cache_dir(model_dir_name)
    try:
        probs = np.load(d / f"probs_{split}.npy").astype(np.float32)
        ids = pd.read_csv(d / f"law_ids_{split}.csv", dtype=str)["law_id"].tolist()
        return probs, ids, load_json(d / "classes.json")["classes"]
    except (FileNotFoundError, KeyError):
        return None


def display_path(path: Path | str) -> str:
    """Path relative to the project root when inside it (portable logs), else absolute."""
    try:
        return str(Path(path).resolve().relative_to(config.PROJECT_ROOT))
    except ValueError:
        return str(path)


def save_json(obj, path: Path) -> None:
    """Write pretty UTF-8 JSON (Kazakh text stays readable)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, default=str)


def load_json(path: Path) -> dict:
    """Read a JSON file written by save_json."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def quick_name(name: str, quick: bool) -> str:
    """Append the quick-run suffix so smoke tests never clobber real results."""
    return f"{name}{QUICK_SUFFIX}" if quick else name


def max_len_for(kind: str, default: int) -> int:
    """max_len chosen from the p99 token length (results/token_lengths.json)."""
    try:
        return int(load_json(TOKEN_STATS_JSON)["chosen_max_len"][kind])
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        return default
