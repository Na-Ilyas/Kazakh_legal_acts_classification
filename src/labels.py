"""Label construction and leakage-free temporal split.

This module is the contract between the data pipeline and the models:
`build_dataset(df)` turns the cleaned corpus into the labelled, split table
saved as config.DATASET_PARQUET with (at least) the columns

    law_id, title, title_norm, date_adopted, year,
    act_form_raw, form_label, database_section, area_label, split

where split is one of "train" / "val" / "test".
"""
from __future__ import annotations

import re
import unicodedata

import pandas as pd

from src import config

_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[\"'«»“”„.,;:!?()\[\]{}№/\\-]+")
_DIGIT_RE = re.compile(r"\d+")


def normalize_title(title: str, mask_digits: bool = False) -> str:
    """Canonical form of a title used for duplicate / leakage detection."""
    if not isinstance(title, str):
        return ""
    text = unicodedata.normalize("NFC", title).lower()
    text = _PUNCT_RE.sub(" ", text)
    if mask_digits:
        text = _DIGIT_RE.sub("0", text)
    return _WS_RE.sub(" ", text).strip()


def split_act_form(raw: str) -> list[str]:
    """'Шешім, Өзгерту' -> ['Шешім', 'Өзгерту'] (tokens are unordered tags)."""
    if not isinstance(raw, str):
        return []
    return [t.strip() for t in raw.split(",") if t.strip() and t.strip() not in config.PLACEHOLDERS]


def primary_form(raw: str) -> str:
    """Highest-ranked base legal form of an act, or FORM_OTHER."""
    tokens = {config.FORM_ALIASES.get(t, t) for t in split_act_form(raw)}
    for form in config.FORM_PRIORITY:
        if form in tokens:
            return form
    return config.FORM_OTHER


def clean_section(raw) -> str | None:
    """database_section with placeholders / numeric junk codes mapped to None."""
    if not isinstance(raw, str) or pd.isna(raw):
        return None
    value = raw.strip()
    if value in config.PLACEHOLDERS or value.isdigit():
        return None
    return value


def assign_split(year: pd.Series) -> pd.Series:
    """Temporal split: train <= TRAIN_END_YEAR, val == VAL_YEAR, test after."""
    split = pd.Series("test", index=year.index, dtype="object")
    split[year <= config.TRAIN_END_YEAR] = "train"
    split[year == config.VAL_YEAR] = "val"
    return split


def build_dataset(df: pd.DataFrame, drop_cross_split_duplicates: bool = True) -> tuple[pd.DataFrame, dict]:
    """Create labels + split. Returns (dataset, stats) where stats documents every filter."""
    stats: dict = {"rows_in": int(len(df))}
    out = pd.DataFrame({
        "law_id": df["law_id"].astype(str),
        "title": df["title"].astype(str),
        "act_form_raw": df["act_form"],
        "database_section": df["database_section"],
    })
    out["date_adopted"] = pd.to_datetime(df["date_adopted"], errors="coerce")
    missing_date = out["date_adopted"].isna()
    stats["dropped_missing_date"] = int(missing_date.sum())
    out = out[~missing_date].copy()

    out["year"] = out["date_adopted"].dt.year.astype("int16")
    out["title_norm"] = out["title"].map(normalize_title)
    out["form_label"] = out["act_form_raw"].map(primary_form)
    out["database_section"] = out["database_section"].map(clean_section)
    out["split"] = assign_split(out["year"])

    # Area label: rare classes decided on TRAIN counts only (no peeking at test).
    train_counts = out.loc[out["split"] == "train", "database_section"].value_counts()
    keep = set(train_counts[train_counts >= config.AREA_MIN_TRAIN_COUNT].index)
    # pd.isna (not `is None`): under pandas 3 missing values arrive as NaN.
    out["area_label"] = out["database_section"].map(
        lambda s: None if pd.isna(s) else (s if s in keep else config.AREA_OTHER)
    )
    stats["area_classes"] = sorted(keep) + [config.AREA_OTHER]

    # Leakage guard: an identical (normalized) title must not appear both in
    # train and in val/test -> remove such rows from val/test.
    if drop_cross_split_duplicates:
        train_titles = set(out.loc[out["split"] == "train", "title_norm"])
        leak = (out["split"] != "train") & out["title_norm"].isin(train_titles)
        stats["dropped_eval_titles_seen_in_train"] = int(leak.sum())
        out = out[~leak]
        val_titles = set(out.loc[out["split"] == "val", "title_norm"])
        leak_vt = (out["split"] == "test") & out["title_norm"].isin(val_titles)
        stats["dropped_test_titles_seen_in_val"] = int(leak_vt.sum())
        out = out[~leak_vt]

    for col in ("form_label", "area_label", "split", "database_section"):
        out[col] = out[col].astype("category")
    stats["rows_out"] = int(len(out))
    stats["split_sizes"] = out["split"].value_counts().to_dict()
    return out.reset_index(drop=True), stats
