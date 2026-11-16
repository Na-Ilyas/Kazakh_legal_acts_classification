"""Collect the numbers shown in the presentation from SQLite + results/ into deck_data.json.

Run from the project root:  .venv/bin/python reports/deck/deck_data.py
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_name("deck_data.json")

# (display label, model_name) per task, in display order. Missing models are skipped.
MAIN_MODELS = {
    "form": [
        ("Majority class", "majority_dummy"),
        ("char 1-NN", "char_1nn"),
        ("TF-IDF word+char + LR", "tfidf_word_char_lr"),
        ("Tiny transformer 2L (scratch)", "tiny_transformer_2L_do30"),
        ("Tiny 4L + MLM + KD", "tiny_transformer_4L_mlm_kd_a90_T2"),
        ("mT5-small seq2seq (pruned)", "mt5_small_kkprune_text"),
        ("e5-small fine-tuned", "e5_small_ft"),
        ("e5-small pruned (25.5M)", "e5_small_kkprune_ft"),
        ("KazBERT fine-tuned", "kazbert_ft"),
        ("kaz-roberta fine-tuned", "kaz_roberta_ft"),
        ("Ensemble + prior adj.", "ensemble_tfidf_tiny_e5_prioradj"),
    ],
    "area": [
        ("Majority class", "majority_dummy"),
        ("char 1-NN", "char_1nn"),
        ("TF-IDF word+char + LR", "tfidf_word_char_lr"),
        ("Tiny transformer 2L (scratch)", "tiny_transformer_2L_do30"),
        ("Tiny 4L + MLM + KD", "tiny_transformer_4L_mlm_kd_a50_T2"),
        ("e5-small fine-tuned", "e5_small_ft"),
        ("e5-small pruned (25.5M)", "e5_small_kkprune_ft"),
        ("KazBERT fine-tuned", "kazbert_ft"),
        ("kaz-roberta LLRD+recency", "kaz_roberta_llrd_rec"),
        ("kaz-roberta refit train+val", "kaz_roberta_llrd_rec_trainval"),
        ("Ensemble (+kaz-roberta)", "ensemble_tfidf_tiny_e5_roberta"),
    ],
}


def fetch(con: sqlite3.Connection, model: str, task: str) -> dict | None:
    """val/test/test_strict F1 + cost columns of one model, or None if not logged."""
    rows = con.execute(
        "SELECT eval_split, f1_macro, n_params, size_mb, latency_ms_per_1000 FROM experiments "
        "WHERE model_name=? AND task=?", (model, task)).fetchall()
    if not rows:
        return None
    out = {"model": model}
    for split, f1, n_params, size_mb, lat in rows:
        out[split] = None if f1 is None else round(f1, 4)
        out.update(n_params=n_params, size_mb=size_mb, latency=lat)
    return out if "test" in out else None


EFFICIENCY_MODELS = [  # (label, form model, area model, highlight)
    ("TF-IDF word+char + LR", "tfidf_word_char_lr", "tfidf_word_char_lr", False),
    ("Tiny transformer 4L + MLM + KD (scratch)", "tiny_transformer_4L_mlm_kd_a90_T2", "tiny_transformer_4L_mlm_kd_a50_T2", True),
    ("e5-small fine-tuned", "e5_small_ft", "e5_small_ft", False),
    ("e5-small, Kazakh vocab pruned", "e5_small_kkprune_ft", "e5_small_kkprune_ft", False),
    ("kaz-roberta fine-tuned", "kaz_roberta_ft", "kaz_roberta_llrd_rec", False),
    ("KazBERT fine-tuned", "kazbert_ft", "kazbert_ft", False),
]


def efficiency_rows(con: sqlite3.Connection, tasks: dict) -> list[dict]:
    """Cost table: CPU latency from the unified benchmark (results/benchmark.csv) when present."""
    import csv
    bench: dict = {}
    try:
        with open(ROOT / "results" / "benchmark.csv", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                bench[(r["model_name"], r["task"])] = r
    except FileNotFoundError:
        pass
    rows = []
    for label, form_model, area_model, highlight in EFFICIENCY_MODELS:
        form, area = fetch(con, form_model, "form"), fetch(con, area_model, "area")
        if not form and not area:
            continue
        ref = form or area
        b = bench.get((form_model, "form")) or bench.get((area_model, "area"))
        rows.append({"label": label, "n_params": ref["n_params"],
                     "size_mb": float(b["size_mb"]) if b else ref["size_mb"],
                     "latency": float(b["latency_cpu_ms_per_1000"]) if b else ref["latency"],
                     "latency_source": "benchmark" if b else "training-time",
                     "form": form and form.get("test_strict"), "area": area and area.get("test_strict"),
                     "highlight": highlight})
    return rows


def main() -> None:
    con = sqlite3.connect(f"file:{ROOT / 'data' / 'kazlaw.db'}?mode=ro", uri=True)
    data: dict = {"tasks": {}}
    for task, models in MAIN_MODELS.items():
        rows = []
        for label, name in models:
            row = fetch(con, name, task)
            if row:
                row["label"] = label
                rows.append(row)
        data["tasks"][task] = rows
    for name in ("io_benchmark", "leakage_report", "split_stats", "eda_summary"):
        try:
            data[name] = json.loads((ROOT / "results" / f"{name}.json").read_text(encoding="utf-8"))
        except FileNotFoundError:
            data[name] = {}
    data["efficiency"] = efficiency_rows(con, data["tasks"])
    text_file = Path(__file__).with_name("deck_text.json")       # hand-written narrative slides
    if text_file.exists():
        data.update(json.loads(text_file.read_text(encoding="utf-8")))
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    for task, rows in data["tasks"].items():
        print(task)
        for r in rows:
            print(f"  {r['label']:32s} val={r.get('val')} test={r.get('test')} strict={r.get('test_strict')} "
                  f"params={r['n_params']} lat={r['latency']}")


if __name__ == "__main__":
    main()
