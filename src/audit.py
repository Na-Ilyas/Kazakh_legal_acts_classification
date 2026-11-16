"""Full data-quality audit of the raw corpus.

Covers: missing values (true empties vs. source placeholders, per field),
date format validity / ranges / logical anomalies, multi-valued field
statistics, constant columns, duplicates (law_id, title, title+description,
title+date), text-quality checks (mixed Latin/Cyrillic homoglyphs, titles
without Kazakh-specific letters, extreme lengths), label coverage, and a list
of cleaning decisions with their justification.

Outputs: results/audit.json, results/audit_report.md
"""
from __future__ import annotations

import collections
import json
import re
import time
from datetime import date

import pandas as pd

from src import config, labels
from src.data_io import iter_raw_chunks, load_corpus, load_dataset
from src.database import split_areas, split_form_tags, split_top_level

AUDIT_JSON = config.RESULTS_DIR / "audit.json"
AUDIT_MD = config.RESULTS_DIR / "audit_report.md"
DATE_COLS = ["date_adopted", "date_modified", "date_published", "ministry_registration_date"]
TODAY = date.today()
_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_MIXED_RE = re.compile(r"\b(?=\w*[a-zA-Z])(?=\w*[а-яА-ЯәғқңөұүһіӘҒҚҢӨҰҮҺІ])\w+\b")
_KAZ_RE = re.compile(r"[әғқңөұүһіӘҒҚҢӨҰҮҺІ]")


# --------------------------------------------------------------------------
# Streaming pass over the raw CSV
# --------------------------------------------------------------------------
def scan_raw() -> dict:
    """One chunked pass: empties, placeholders, date validity, constant columns."""
    n_rows = 0
    empty = collections.Counter()
    placeholder = collections.defaultdict(collections.Counter)
    whitespace_padded = collections.Counter()
    distinct: dict[str, set] = collections.defaultdict(set)
    date_stats = {c: {"non_empty": 0, "bad_format": 0, "unparseable": 0, "min": None, "max": None,
                      "examples_bad": []} for c in DATE_COLS}
    placeholder_set = config.PLACEHOLDERS - {""}
    for chunk in iter_raw_chunks():
        n_rows += len(chunk)
        for col in chunk.columns:
            s = chunk[col]
            stripped = s.str.strip()
            empty[col] += int((stripped == "").sum())
            ph = stripped[stripped.isin(placeholder_set)]
            for val, cnt in ph.value_counts().items():
                placeholder[col][val] += int(cnt)
            whitespace_padded[col] += int((s != stripped).sum())
            if len(distinct[col]) <= 2:          # only needed to detect constant columns
                distinct[col].update(s.unique()[:3].tolist())
        for col in DATE_COLS:
            s = chunk[col].str.strip()
            s = s[(s != "") & ~s.isin(placeholder_set)]
            st = date_stats[col]
            st["non_empty"] += int(len(s))
            bad = s[~s.str.match(_ISO_RE)]
            st["bad_format"] += int(len(bad))
            st["examples_bad"] = (st["examples_bad"] + bad.head(3).tolist())[:5]
            parsed = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
            st["unparseable"] += int(parsed.isna().sum())
            if parsed.notna().any():
                lo, hi = parsed.min(), parsed.max()
                st["min"] = lo if st["min"] is None else min(st["min"], lo)
                st["max"] = hi if st["max"] is None else max(st["max"], hi)
    missing = {}
    for col in empty:
        n_ph = int(sum(placeholder[col].values()))
        missing[col] = {
            "empty": int(empty[col]),
            "placeholder": n_ph,
            "placeholder_values": dict(placeholder[col].most_common(5)),
            "missing_total": int(empty[col]) + n_ph,
            "missing_pct": round(100 * (int(empty[col]) + n_ph) / n_rows, 2),
            "whitespace_padded": int(whitespace_padded[col]),
        }
    for st in date_stats.values():
        st["min"] = None if st["min"] is None else str(st["min"].date())
        st["max"] = None if st["max"] is None else str(st["max"].date())
    constant = [c for c, v in distinct.items() if len(v) <= 1]
    return {"n_rows": n_rows, "missing": missing, "dates": date_stats, "constant_columns": constant,
            "constant_values": {c: sorted(distinct[c]) for c in constant}}


# --------------------------------------------------------------------------
# Checks on the optimised corpus
# --------------------------------------------------------------------------
def date_anomalies(df: pd.DataFrame) -> dict:
    """Logical date checks (future dates, ordering, implausible years)."""
    today = pd.Timestamp(TODAY)
    adopted = df["date_adopted"]
    out = {
        "adopted_in_future": int((adopted > today).sum()),
        "adopted_before_1900": int((adopted.dt.year < 1900).sum()),
        "adopted_before_1991_independence": int((adopted.dt.year < 1991).sum()),
        "modified_before_adopted": int((df["date_modified"] < adopted).sum()),
        "published_before_adopted": int((df["date_published"] < adopted).sum()),
        "ministry_registration_before_adopted": int((df["ministry_registration_date"] < adopted).sum()),
        "modified_in_future": int((df["date_modified"] > today).sum()),
        "year_in_title_differs_from_adopted": None,
    }
    # Titles often contain "<YYYY> жылғы" (the date of the amended act) - informative only
    yr = df["title"].str.extract(r"(\d{4}) жылғы", expand=False)
    out["titles_with_year_mention_pct"] = round(100 * float(yr.notna().mean()), 2)
    out["acts_per_year_min_max"] = [int(adopted.dt.year.min()), int(adopted.dt.year.max())]
    del out["year_in_title_differs_from_adopted"]
    return out


def multivalue_stats(df: pd.DataFrame) -> dict:
    """Statistics of the multi-valued fields act_form / legal_relation_area / adopting_body."""
    forms = df["act_form"].astype(object)
    form_tags = forms.map(split_form_tags)
    raw_combos = forms.dropna().map(lambda s: " ".join(s.split())).nunique()
    set_combos = form_tags[form_tags.map(len) > 0].map(lambda t: tuple(sorted(t))).nunique()
    tag_counter = collections.Counter(t for tags in form_tags for t in tags)

    areas_raw = df["legal_relation_area"].astype(object)
    uniq_area_strings = areas_raw.dropna().unique()
    smart = {s: split_areas(s) for s in uniq_area_strings}
    naive_names = {f.strip() for s in uniq_area_strings for f in s.split(",") if f.strip()}
    smart_names = {n for v in smart.values() for n in v}
    whole_values = {" ".join(s.split()) for s in uniq_area_strings}
    n_areas = areas_raw.map(lambda s: len(smart.get(s, [])) if isinstance(s, str) else 0)
    numeric_codes = sum(1 for s in uniq_area_strings for f in s.split(",") if f.strip().isdigit())

    bodies = df["adopting_body"].astype(object).dropna()
    joint = bodies.map(lambda s: len([p for p in split_top_level(s) if p.strip()]) > 1)
    return {
        "act_form": {
            "distinct_raw_strings": int(raw_combos),
            "distinct_tag_sets": int(set_combos),
            "order_only_duplicates": int(raw_combos - set_combos),
            "distinct_tags": len(tag_counter),
            "tags_per_act_mean": round(float(form_tags.map(len).mean()), 2),
            "tags_per_act_max": int(form_tags.map(len).max()),
            "acts_with_multiple_tags_pct": round(100 * float((form_tags.map(len) > 1).mean()), 2),
            "top_tags": dict(tag_counter.most_common(15)),
        },
        "legal_relation_area": {
            "distinct_raw_strings": int(len(uniq_area_strings)),
            "distinct_names_naive_comma_split": len(naive_names),
            "distinct_names_vocab_aware_split": len(smart_names),
            "vocab_aware_names_seen_as_whole_value": len(smart_names & whole_values),
            "numeric_code_fragments_dropped": int(numeric_codes),
            "areas_per_act_mean": round(float(n_areas.mean()), 2),
            "areas_per_act_max": int(n_areas.max()),
            "acts_with_multiple_areas_pct": round(100 * float((n_areas > 1).mean()), 2),
        },
        "adopting_body": {
            "distinct": int(bodies.nunique()),
            "joint_acts_multiple_bodies_pct": round(100 * float(joint.mean()), 2),
        },
    }


def duplicate_stats(df: pd.DataFrame) -> dict:
    """Exact duplicates on several keys (count of redundant rows = duplicated())."""
    title_norm = df["title"].map(labels.normalize_title)
    keys = {
        "law_id": df["law_id"],
        "title_exact": df["title"],
        "title_normalized": title_norm,
        "title_plus_description": df["title"].astype(str) + "||" + df["description"].fillna("").astype(str),
        "title_plus_date_adopted": df["title"].astype(str) + "||" + df["date_adopted"].astype(str),
    }
    out = {}
    for name, key in keys.items():
        dup = key.duplicated(keep=False)
        out[name] = {"rows_in_duplicate_groups": int(dup.sum()),
                     "redundant_rows": int(key.duplicated().sum()),
                     "distinct_duplicated_values": int(key[dup].nunique())}
    # Conflicting labels among identical normalised titles (label noise indicator)
    forms = df["act_form"].astype(object).map(labels.primary_form)
    tmp = pd.DataFrame({"t": title_norm, "f": forms})
    grp = tmp[tmp["t"].duplicated(keep=False)].groupby("t")["f"].nunique()
    out["duplicate_titles_with_conflicting_form_label"] = int((grp > 1).sum())
    out["top_duplicated_titles"] = title_norm.value_counts().head(5).to_dict()
    return out


def text_quality(df: pd.DataFrame) -> dict:
    """Title-level text checks."""
    t = df["title"].astype(str)
    lens = t.str.len()
    words = t.str.split().str.len()
    return {
        "title_chars": {"min": int(lens.min()), "median": float(lens.median()),
                        "p95": float(lens.quantile(0.95)), "max": int(lens.max())},
        "title_words": {"min": int(words.min()), "median": float(words.median()),
                        "p95": float(words.quantile(0.95)), "max": int(words.max())},
        "titles_over_1000_chars": int((lens > 1000).sum()),
        "titles_under_20_chars": int((lens < 20).sum()),
        "titles_with_mixed_latin_cyrillic_word_pct": round(100 * float(t.str.contains(_MIXED_RE).mean()), 2),
        "titles_without_kazakh_specific_letters_pct": round(100 * (1 - float(t.str.contains(_KAZ_RE).mean())), 2),
        "titles_with_leading_trailing_space": int((t != t.str.strip()).sum()),
        "titles_with_double_space": int(t.str.contains("  ").sum()),
    }


def label_coverage() -> dict:
    """How many rows get a usable label for each task (from the built dataset)."""
    ds = load_dataset(["split", "form_label", "area_label", "act_form_raw", "database_section"])
    try:
        stats = json.loads((config.RESULTS_DIR / "split_stats.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        stats = {}
    return {
        "rows_in_dataset": int(len(ds)),
        "form_label_from_missing_act_form": int(ds["act_form_raw"].isna().sum()),
        "form_label_counts": ds["form_label"].value_counts().astype(int).to_dict(),
        "area_label_missing": int(ds["area_label"].isna().sum()),
        "area_label_n_classes": int(ds["area_label"].nunique()),
        "area_collapsed_to_other": int((ds["area_label"].astype(str) == config.AREA_OTHER).sum()),
        "removed_eval_rows_seen_in_train": stats.get("dropped_eval_titles_seen_in_train"),
        "removed_test_rows_seen_in_val": stats.get("dropped_test_titles_seen_in_val"),
        "split_sizes": stats.get("split_sizes"),
    }


CLEANING_DECISIONS = [
    ("Placeholders ('ЖОҚ', 'undefined', '', '-', ...) -> NA",
     "They are source-system fill values, not information; counting them as values would inflate cardinality and create fake classes."),
    ("`database_section` numeric codes ('3700', '3800') -> NA",
     "Not section names (19 rows); cannot be mapped to a curated area, so treated as missing for task `area`."),
    ("`is_archive` dropped from modelling", "Constant column ('f' for every row) - zero information."),
    ("`act_form` parsed as an unordered tag set (strip whitespace, split on ',')",
     "Raw strings differ only by tag order/padding; set semantics collapses order-only duplicates."),
    ("`legal_relation_area` split with a parenthesis- and case-aware splitter",
     "Naive ', ' splitting breaks names that contain commas; numeric classifier codes are dropped."),
    ("Primary form = highest base form in Заң > Жарлық > Қаулы > Шешім > Бұйрық > Өкім, else Басқа",
     "Joint acts carry several base forms; a hierarchy gives one deterministic label (aliases such as 'Нормативтік қаулы' -> Қаулы)."),
    ("Rare areas (< 300 train rows) collapsed to Басқа, decided on TRAIN only",
     "Classes with a handful of examples make macro-F1 unstable; using train counts only avoids peeking at test."),
    ("Temporal split train <= 2021, val 2022, test >= 2023", "Mimics deployment on future acts and prevents look-ahead leakage."),
    ("Val/test rows whose normalised title occurs in train removed; test_strict excludes near-duplicates (cos >= 0.9)",
     "Templated titles repeat every year; without this the test set measures memorisation."),
    ("Titles kept verbatim for models (no truncation at data level, NFC-normalised in text_prep)",
     "Length outliers are genuine (long amendment titles); models truncate at their own max length."),
    ("Dates parsed with strict ISO format; anomalies reported, not 'fixed'",
     "No ground truth to repair them; affected rows are few and dates only drive the split by year."),
]


# --------------------------------------------------------------------------
# Report rendering
# --------------------------------------------------------------------------
def _md_table(rows: list[dict], cols: list[str]) -> str:
    """Render a list of dicts as a markdown table."""
    head = "| " + " | ".join(cols) + " |\n|" + "|".join("---" for _ in cols) + "|\n"
    body = "".join("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |\n" for r in rows)
    return head + body


def render_markdown(audit: dict) -> str:
    """Human-readable audit report with markdown tables."""
    parts = [f"# Data-quality audit - KazakhLawCorpus ({audit['n_rows']:,} rows)\n",
             f"Generated by `python -m src.audit` on {audit['generated']}.\n"]
    parts.append("## 1. Missing values per field (empty vs. placeholder)\n")
    rows = [{"field": c, **{k: v for k, v in m.items() if k != "placeholder_values"},
             "placeholder values": ", ".join(f"{k}:{v}" for k, v in m["placeholder_values"].items())}
            for c, m in sorted(audit["missing"].items(), key=lambda x: -x[1]["missing_total"])]
    parts.append(_md_table(rows, ["field", "empty", "placeholder", "missing_total", "missing_pct",
                                  "whitespace_padded", "placeholder values"]))
    parts.append("\n## 2. Dates: format validity and ranges\n")
    rows = [{"field": c, **{k: v for k, v in d.items() if k != "examples_bad"}} for c, d in audit["dates"].items()]
    parts.append(_md_table(rows, ["field", "non_empty", "bad_format", "unparseable", "min", "max"]))
    parts.append("\n**Logical anomalies**\n\n" + _md_table(
        [{"check": k, "rows": v} for k, v in audit["date_anomalies"].items()], ["check", "rows"]))
    parts.append("\n## 3. Multi-valued fields\n")
    for field, st in audit["multivalue"].items():
        rows = [{"metric": k, "value": v} for k, v in st.items() if k != "top_tags"]
        parts.append(f"**{field}**\n\n" + _md_table(rows, ["metric", "value"]) + "\n")
    parts.append("\n## 4. Constant columns\n\n" + ", ".join(
        f"`{c}` = {audit['constant_values'][c]}" for c in audit["constant_columns"]) + "\n")
    parts.append("\n## 5. Duplicates\n")
    rows = [{"key": k, **v} for k, v in audit["duplicates"].items() if isinstance(v, dict) and "redundant_rows" in v]
    parts.append(_md_table(rows, ["key", "rows_in_duplicate_groups", "redundant_rows", "distinct_duplicated_values"]))
    parts.append(f"\nDuplicate normalised titles whose copies have **different** form labels: "
                 f"{audit['duplicates']['duplicate_titles_with_conflicting_form_label']}\n")
    parts.append("\n## 6. Title text quality\n\n" + _md_table(
        [{"check": k, "value": v} for k, v in audit["text_quality"].items()], ["check", "value"]))
    parts.append("\n## 7. Label coverage\n\n" + _md_table(
        [{"item": k, "value": v} for k, v in audit["label_coverage"].items()], ["item", "value"]))
    parts.append("\n## 8. Cleaning decisions and justification\n")
    parts.extend(f"{i}. **{d}** - {why}\n" for i, (d, why) in enumerate(CLEANING_DECISIONS, 1))
    return "\n".join(parts)


def run() -> dict:
    """Run the full audit and write results/audit.json + results/audit_report.md."""
    t0 = time.perf_counter()
    audit: dict = {"generated": str(TODAY)}
    audit.update(scan_raw())
    df = load_corpus(["law_id", "title", "description", "act_form", "legal_relation_area", "adopting_body",
                      "date_adopted", "date_modified", "date_published", "ministry_registration_date"])
    audit["date_anomalies"] = date_anomalies(df)
    audit["multivalue"] = multivalue_stats(df)
    audit["duplicates"] = duplicate_stats(df)
    audit["text_quality"] = text_quality(df)
    del df
    audit["label_coverage"] = label_coverage()
    audit["cleaning_decisions"] = [{"decision": d, "justification": j} for d, j in CLEANING_DECISIONS]
    audit["elapsed_s"] = round(time.perf_counter() - t0, 1)
    try:
        AUDIT_JSON.write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        AUDIT_MD.write_text(render_markdown(audit), encoding="utf-8")
    except OSError as exc:
        print(f"[audit] could not write outputs: {exc}")
    summary = {"rows": audit["n_rows"], "constant_columns": audit["constant_columns"],
               "title_dup_redundant_rows": audit["duplicates"]["title_exact"]["redundant_rows"],
               "elapsed_s": audit["elapsed_s"]}
    print(f"[audit] {summary}")
    return summary


if __name__ == "__main__":
    run()
