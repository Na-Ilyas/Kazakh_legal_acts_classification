"""SQLite storage for experiment results and per-row predictions.

Shared by every model module (baseline, tiny transformer, e5) so that the
model comparison is a single SQL query over one table.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Iterable

import pandas as pd

from src import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    exp_id              INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name          TEXT NOT NULL,      -- e.g. 'tfidf_char_lr', 'tiny_transformer_4L', 'e5_small_ft'
    model_family        TEXT NOT NULL,      -- 'baseline' | 'tfidf' | 'tiny_transformer' | 'pretrained'
    task                TEXT NOT NULL,      -- 'form' | 'area'
    eval_split          TEXT NOT NULL,      -- 'val' | 'test' | 'test_strict'
    f1_macro            REAL,
    accuracy            REAL,
    f1_weighted         REAL,
    n_params            INTEGER,
    size_mb             REAL,
    latency_ms_per_1000 REAL,               -- inference time for 1000 titles
    latency_device      TEXT,               -- 'cpu' | 'cuda'
    peak_mem_mb         REAL,
    train_time_s        REAL,
    n_train             INTEGER,
    n_eval              INTEGER,
    hyperparams         TEXT,               -- JSON
    notes               TEXT,
    created_at          TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS predictions (
    exp_id      INTEGER NOT NULL REFERENCES experiments(exp_id) ON DELETE CASCADE,
    law_id      TEXT NOT NULL,
    y_true      TEXT,
    y_pred      TEXT NOT NULL,
    confidence  REAL,
    PRIMARY KEY (exp_id, law_id)
);
CREATE INDEX IF NOT EXISTS idx_exp_task ON experiments(task, eval_split);
"""


def connect(db_path=config.DB_PATH) -> sqlite3.Connection:
    """Open the project database and make sure the experiment tables exist.

    The caller owns the connection (close it); inside this module use session().
    timeout=60: several processes (data + model agents) may write concurrently.
    """
    con = sqlite3.connect(db_path, timeout=60)
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript(SCHEMA)
    return con


@contextmanager
def session(db_path=None):
    """Connection that commits on success, rolls back on error and is ALWAYS closed.

    (`with sqlite3.connect(...)` only commits/rolls back - it never closes.)
    db_path=None defers to connect()'s default, so the KAZLAW_DB dev override applied by
    src.model_utils (which patches connect's default) also reaches session().
    """
    con = connect() if db_path is None else connect(db_path)
    try:
        with con:
            yield con
    finally:
        con.close()


def log_experiment(model_name: str, model_family: str, task: str, eval_split: str,
                   metrics: dict, hyperparams: dict | None = None, notes: str = "",
                   replace: bool = True) -> int:
    """Insert one result row and return its exp_id.

    `metrics` may contain any of: f1_macro, accuracy, f1_weighted, n_params,
    size_mb, latency_ms_per_1000, latency_device, peak_mem_mb, train_time_s,
    n_train, n_eval. With replace=True a previous run of the same
    (model_name, task, eval_split) is deleted first (keeps the table tidy on reruns).
    """
    cols = ["f1_macro", "accuracy", "f1_weighted", "n_params", "size_mb",
            "latency_ms_per_1000", "latency_device", "peak_mem_mb", "train_time_s",
            "n_train", "n_eval"]
    with session() as con:
        if replace:
            con.execute("DELETE FROM experiments WHERE model_name=? AND task=? AND eval_split=?",
                        (model_name, task, eval_split))
        cur = con.execute(
            f"INSERT INTO experiments (model_name, model_family, task, eval_split, {', '.join(cols)}, "
            f"hyperparams, notes, created_at) VALUES (?, ?, ?, ?, {', '.join('?' * len(cols))}, ?, ?, ?)",
            (model_name, model_family, task, eval_split, *[metrics.get(c) for c in cols],
             json.dumps(hyperparams or {}, ensure_ascii=False), notes,
             datetime.now().isoformat(timespec="seconds")),
        )
        return int(cur.lastrowid)


def log_predictions(exp_id: int, law_ids: Iterable[str], y_true: Iterable, y_pred: Iterable,
                    confidence: Iterable[float] | None = None) -> None:
    """Bulk-insert per-row predictions for one experiment."""
    law_ids, y_true, y_pred = list(law_ids), list(y_true), list(y_pred)
    conf = list(confidence) if confidence is not None else [None] * len(law_ids)
    rows = [(exp_id, str(l), None if t is None else str(t), str(p),
             None if c is None else float(c))
            for l, t, p, c in zip(law_ids, y_true, y_pred, conf)]
    with session() as con:
        con.execute("DELETE FROM predictions WHERE exp_id=?", (exp_id,))
        con.executemany("INSERT INTO predictions VALUES (?, ?, ?, ?, ?)", rows)


def load_experiments(task: str | None = None) -> pd.DataFrame:
    """All experiment rows (optionally for one task) as a DataFrame."""
    query = "SELECT * FROM experiments"
    params: tuple = ()
    if task:
        query += " WHERE task = ?"
        params = (task,)
    with session() as con:
        return pd.read_sql_query(query + " ORDER BY task, eval_split, f1_macro DESC", con, params=params)
