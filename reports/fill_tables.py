"""Fill the generated result tables in reports/report.md (between BEGIN/END markers).

Numbers come from SQLite (experiments), results/benchmark.csv (CPU latency, unified
protocol) and results/bootstrap_ci.csv (paired-bootstrap CIs) - nothing is typed by hand.
Usage:  .venv/bin/python reports/fill_tables.py
"""
from __future__ import annotations

import csv
import re
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "report.md"

# (label, model_name) per task; rows whose model is missing are skipped.
ROWS = {
    "form": [
        ("Majority class", "majority_dummy"),
        ("char 1-NN (template matching)", "char_1nn"),
        ("TF-IDF word+char + LR (baseline)", "tfidf_word_char_lr"),
        ("Tiny transformer 4L, from scratch", "tiny_transformer_4L"),
        ("Tiny transformer 2L d128, from scratch", "tiny_transformer_2L_d128"),
        ("Tiny 4L + MLM", "tiny_transformer_4L_mlm"),
        ("Tiny 4L + MLM + KD (e5 teacher)", "tiny_transformer_4L_mlm_kd_a90_T2"),
        ("Tiny 4L + MLM + KD (kaz-roberta teacher)", "tiny_transformer_4L_mlm_kdrob_a50_T2"),
        ("mT5-small seq2seq, vocab-pruned", "mt5_small_kkprune_text"),
        ("e5-small frozen + LR", "e5_frozen_lr"),
        ("e5-small fine-tuned", "e5_small_ft"),
        ("e5-small fine-tuned, vocab-pruned", "e5_small_kkprune_ft"),
        ("e5-small vocab-pruned + KD (kaz-roberta teacher)", "e5_small_kkprune_kdrob"),
        ("kaz-roberta fine-tuned", "kaz_roberta_ft"),
        ("kaz-roberta, vocab-pruned", "kaz_roberta_kkprune_ft"),
        ("KazBERT fine-tuned", "kazbert_ft"),
    ],
    "area": [
        ("Majority class", "majority_dummy"),
        ("char 1-NN (template matching)", "char_1nn"),
        ("TF-IDF word+char + LR (baseline)", "tfidf_word_char_lr"),
        ("Tiny transformer 4L, from scratch", "tiny_transformer_4L"),
        ("Tiny transformer 2L, from scratch (dropout 0.3)", "tiny_transformer_2L_do30"),
        ("Tiny 4L + MLM", "tiny_transformer_4L_mlm"),
        ("Tiny 4L + MLM + KD (e5 teacher)", "tiny_transformer_4L_mlm_kd_a50_T2"),
        ("Tiny 4L + MLM + KD (kaz-roberta teacher)", "tiny_transformer_4L_mlm_kdrob_a50_T2"),
        ("e5-small frozen + LR", "e5_frozen_lr"),
        ("e5-small fine-tuned", "e5_small_ft"),
        ("e5-small fine-tuned, vocab-pruned", "e5_small_kkprune_ft"),
        ("e5-small vocab-pruned + KD (kaz-roberta teacher)", "e5_small_kkprune_kdrob"),
        ("kaz-roberta fine-tuned", "kaz_roberta_ft"),
        ("kaz-roberta + LLRD + recency (val-selected)", "kaz_roberta_llrd_rec"),
        ("kaz-roberta, vocab-pruned", "kaz_roberta_kkprune_ft"),
        ("KazBERT fine-tuned", "kazbert_ft"),
    ],
}
REFERENCE = {  # not eligible as "the" model: refits, >120M, teacher
    "form": [("Ensemble TF-IDF+tiny+e5+kaz-roberta (208M, >120M)", "ensemble_tfidf_tiny_e5_roberta"),
             ("LLM teacher Qwen3-1.7B + LoRA (teacher only; latency on GPU)", "llm_teacher_qwen3_1p7b")],
    "area": [("kaz-roberta, refit on train+val", "kaz_roberta_llrd_rec_trainval"),
             ("Ensemble TF-IDF+tiny+e5+kaz-roberta (214M, >120M)", "ensemble_tfidf_tiny_e5_roberta"),
             ("LLM teacher Qwen3-1.7B + LoRA (teacher only; latency on GPU)", "llm_teacher_qwen3_1p7b")],
}


def load_csv(path: Path, key) -> dict:
    """Read a CSV into {key(row): row}; empty dict if the file does not exist."""
    try:
        with open(path, encoding="utf-8") as fh:
            return {key(r): r for r in csv.DictReader(fh)}
    except FileNotFoundError:
        return {}


def params(n) -> str:
    """Human-readable parameter count."""
    if not n:
        return "0"
    return f"{n / 1e6:.1f}M" if n >= 1e6 else f"{n / 1e3:.0f}k"


def table(con, task: str, rows: list, bench: dict, boot: dict) -> str:
    """Markdown table for one task."""
    out = ["| Model | Params | Size MB | CPU ms / 1000 | F1 val | F1 test [95 % CI] | Δ vs TF-IDF | F1 test_strict |",
           "|---|---|---|---|---|---|---|---|"]
    for label, name in rows:
        rec = {s: (f, n, mb, lat) for s, f, n, mb, lat in con.execute(
            "SELECT eval_split, f1_macro, n_params, size_mb, latency_ms_per_1000 FROM experiments "
            "WHERE model_name=? AND task=?", (name, task))}
        if "test" not in rec:
            continue
        _, n, mb, lat = rec["test"]
        b = bench.get((name, task))
        if b:
            mb, lat = float(b["size_mb"]), float(b["latency_cpu_ms_per_1000"])
        bt = boot.get((task, name))
        ci = f" [{float(bt['ci_low']):.3f}, {float(bt['ci_high']):.3f}]" if bt else ""
        delta = (f"{float(bt['delta_vs_tfidf']):+.3f} ({bt['verdict']})" if bt and bt["verdict"] != "baseline"
                 else ("–" if not bt else "baseline"))
        val = rec.get("val", (None,))[0]
        val_s = "in-sample" if name.endswith("_trainval") else ("–" if val is None else f"{val:.3f}")
        strict = rec.get("test_strict", (None,))[0]
        out.append(f"| {label} | {params(n)} | {mb:.0f} | {lat:,.0f} | {val_s} | {rec['test'][0]:.3f}{ci} | "
                   f"{delta} | {'–' if strict is None else f'{strict:.3f}'} |")
    return "\n".join(out)


def main() -> None:
    con = sqlite3.connect(f"file:{ROOT / 'data' / 'kazlaw.db'}?mode=ro", uri=True)
    bench = load_csv(ROOT / "results" / "benchmark.csv", lambda r: (r["model_name"], r["task"]))
    boot = load_csv(ROOT / "results" / "bootstrap_ci.csv", lambda r: (r["task"], r["model_name"]))
    text = REPORT.read_text(encoding="utf-8")
    for task in ("form", "area"):
        for kind, rows in (("main", ROWS[task]), ("ref", REFERENCE[task])):
            marker = f"{kind}_{task}"
            new = table(con, task, rows, bench, boot)
            text, n = re.subn(rf"(<!-- BEGIN:{marker} -->).*?(<!-- END:{marker} -->)",
                              lambda m: m.group(1) + "\n" + new + "\n" + m.group(2), text, flags=re.S)
            print(f"{marker}: {'filled' if n else 'marker not found'}")
    REPORT.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
