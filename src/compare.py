"""Final model comparison built from the SQLite experiments table.

Steps
1. (Re)compute the test_strict rows from the stored test predictions for any
   experiment where that row is missing or stale (strict ids written later).
2. Pivot experiments into one row per (task, model): val / test / test_strict
   macro-F1, accuracy, #params, size MB, CPU + CUDA latency per 1000 titles,
   memory, train time -> results/model_comparison.csv and .md.
3. Figures: macro-F1 vs CPU latency bubble chart (bubble = #params), grouped
   F1 bars per task, test vs test_strict F1.
4. results/best_models.json: best model per (task, family) chosen on VAL F1.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src import config, experiments_db
from src.model_utils import (QUICK_SUFFIX, compute_metrics, load_strict_ids, save_json, with_db_retry)

COMPARISON_CSV = config.RESULTS_DIR / "model_comparison.csv"
COMPARISON_MD = config.RESULTS_DIR / "model_comparison.md"
BEST_JSON = config.RESULTS_DIR / "best_models.json"
FAMILY_COLORS = {"baseline": "#b0b0b0", "tfidf": "#4d4d4d", "tiny_transformer": "#2a6fdb",
                 "pretrained": "#d62728", "pretrained_frozen": "#e07b39"}

RANK_SQL = """
WITH test_rows AS (
    SELECT task, model_name, model_family, f1_macro, latency_ms_per_1000, n_params
    FROM experiments WHERE eval_split = 'test'
)
SELECT task, model_name, model_family, ROUND(f1_macro, 4) AS f1_macro,
       RANK() OVER (PARTITION BY task ORDER BY f1_macro DESC) AS rank_f1,
       RANK() OVER (PARTITION BY task ORDER BY latency_ms_per_1000 ASC) AS rank_speed,
       ROUND(f1_macro - MAX(f1_macro) OVER (PARTITION BY task), 4) AS gap_to_best
FROM test_rows ORDER BY task, rank_f1
"""


# --------------------------------------------------------------------------
# test_strict recomputation
# --------------------------------------------------------------------------
def recompute_test_strict(force: bool = False) -> list[dict]:
    """Log test_strict rows computed from the predictions table where missing/stale."""
    strict = load_strict_ids()
    if strict is None:
        print("[compare] results/test_strict_ids.csv missing - test_strict not computed")
        return []
    done = []
    with experiments_db.connect() as con:
        tests = pd.read_sql_query("SELECT * FROM experiments WHERE eval_split='test'", con)
        existing = pd.read_sql_query("SELECT model_name, task, n_eval FROM experiments "
                                     "WHERE eval_split='test_strict'", con)
    have = {(r.model_name, r.task): r.n_eval for r in existing.itertuples()}
    for row in tests.itertuples():
        with experiments_db.connect() as con:
            preds = pd.read_sql_query("SELECT law_id, y_true, y_pred FROM predictions WHERE exp_id=?",
                                      con, params=(row.exp_id,))
        if preds.empty:
            continue
        sub = preds[preds["law_id"].astype(str).isin(strict)]
        if sub.empty:
            continue
        key = (row.model_name, row.task)
        if not force and key in have and have[key] == len(sub):
            continue                                  # already present and consistent
        metrics = compute_metrics(sub["y_true"], sub["y_pred"])
        for col in ("n_params", "size_mb", "latency_ms_per_1000", "latency_device", "peak_mem_mb",
                    "train_time_s", "n_train"):
            metrics[col] = getattr(row, col)
        hp = json.loads(row.hyperparams or "{}")
        with_db_retry(experiments_db.log_experiment, row.model_name, row.model_family, row.task,
                      "test_strict", metrics, hyperparams=hp,
                      notes=(row.notes or "") + " | test_strict recomputed from predictions table")
        done.append({"model_name": row.model_name, "task": row.task, "f1_macro": round(metrics["f1_macro"], 4),
                     "n_eval": metrics["n_eval"]})
    if done:
        print(f"[compare] recomputed test_strict for {len(done)} experiments")
    return done


# --------------------------------------------------------------------------
# Comparison table
# --------------------------------------------------------------------------
def load_rows(quick: bool = False) -> pd.DataFrame:
    """Full-run rows; with quick=True the smoke-test (_quick) rows plus the baselines."""
    df = experiments_db.load_experiments()
    is_quick = df["model_name"].str.endswith(QUICK_SUFFIX)
    if quick:
        return df[is_quick | df["model_family"].isin(["baseline", "tfidf"])]
    return df[~is_quick]


def _hp(value: str, key: str):
    """Read one key from a hyperparams JSON string."""
    try:
        return json.loads(value or "{}").get(key)
    except (json.JSONDecodeError, AttributeError):
        return None


def build_table(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per (task, model) with metrics of every split + cost columns."""
    f1 = rows.pivot_table(index=["task", "model_name"], columns="eval_split", values="f1_macro", aggfunc="first")
    f1 = f1.rename(columns=lambda s: f"f1_{s}")
    acc = rows[rows["eval_split"] == "test"].set_index(["task", "model_name"])["accuracy"].rename("acc_test")
    # cost columns are split-independent: take them from the test row (fallback: any row)
    order = {"test": 0, "val": 1, "test_strict": 2}
    meta = (rows.assign(_o=rows["eval_split"].map(order).fillna(9)).sort_values("_o")
            .groupby(["task", "model_name"]).first())
    meta = meta[["model_family", "n_params", "size_mb", "latency_ms_per_1000", "peak_mem_mb",
                 "train_time_s", "n_train", "hyperparams"]]
    table = meta.join(f1).join(acc).reset_index()
    table["latency_cuda_ms_per_1000"] = table["hyperparams"].map(lambda h: _hp(h, "latency_cuda_ms_per_1000"))
    table["infer_ram_mb"] = table["hyperparams"].map(lambda h: _hp(h, "infer_ram_mb"))
    table = table.drop(columns="hyperparams")
    cols = ["task", "model_name", "model_family", "f1_val", "f1_test", "f1_test_strict", "acc_test",
            "n_params", "size_mb", "latency_ms_per_1000", "latency_cuda_ms_per_1000", "peak_mem_mb",
            "infer_ram_mb", "train_time_s", "n_train"]
    for c in cols:
        if c not in table:
            table[c] = np.nan
    table = table[cols].sort_values(["task", "f1_test"], ascending=[True, False])
    return table.reset_index(drop=True)


def to_markdown(table: pd.DataFrame) -> str:
    """GitHub markdown, one section per task (no extra dependency)."""
    show = {"model_name": "model", "model_family": "family", "f1_val": "F1 val", "f1_test": "F1 test",
            "f1_test_strict": "F1 test_strict", "acc_test": "acc test", "n_params": "params",
            "size_mb": "size MB", "latency_ms_per_1000": "CPU ms/1000", "latency_cuda_ms_per_1000": "GPU ms/1000",
            "peak_mem_mb": "train peak MB", "infer_ram_mb": "infer RAM MB", "train_time_s": "train s"}
    lines = ["# Model comparison", "",
             "Macro-F1 on val (2022), test (>=2023) and test_strict (test without near-duplicates of train "
             "titles). Latency = end-to-end inference for 1000 test titles (batch 64, 8 CPU threads, median "
             "of 3). `train peak MB` = peak CUDA memory while training for neural models (as logged by the "
             "baseline module for TF-IDF models); `infer RAM MB` = RAM growth of a fresh process while "
             "loading the model and predicting on CPU.", ""]
    for task, part in table.groupby("task", sort=False):
        lines += [f"## Task `{task}`", "", "| " + " | ".join(show.values()) + " |",
                  "|" + "---|" * len(show)]
        for _, r in part.iterrows():
            cells = []
            for col in show:
                v = r[col]
                if isinstance(v, float) and np.isnan(v) or v is None:
                    cells.append("–")
                elif col.startswith(("f1", "acc")):
                    cells.append(f"{v:.4f}")
                elif col == "n_params":
                    cells.append(f"{v / 1e6:.2f}M" if v >= 1e5 else f"{int(v):,}")
                elif isinstance(v, (int, float, np.floating, np.integer)):
                    cells.append(f"{v:,.1f}")
                else:
                    cells.append(str(v))
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def seed_section() -> list[str]:
    """Markdown summary of results/tiny_transformer_seed_study.csv (if the study was run)."""
    path = config.RESULTS_DIR / "tiny_transformer_seed_study.csv"
    try:
        df = pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return []
    cols = [c for c in ("f1_val", "f1_test", "f1_test_strict") if c in df]
    agg = df.groupby(["task", "arch", "loss"])[cols].agg(["mean", "std"]).round(4)
    lines = ["## Seed variance of the tiny transformer", "",
             f"Same data and hyper-parameters, seeds {sorted(df['seed'].unique().tolist())} "
             "(weight init + batch order). Differences between tiny-transformer variants smaller than "
             "about 2 std are within seed noise.", "",
             "| task | arch | loss | " + " | ".join(f"{c} mean ± std" for c in cols) + " |",
             "|" + "---|" * (3 + len(cols))]
    for (task, arch, loss), r in agg.iterrows():
        cells = [f"{r[(c, 'mean')]:.4f} ± {r[(c, 'std')]:.4f}" for c in cols]
        lines.append(f"| {task} | {arch} | {loss} | " + " | ".join(cells) + " |")
    return lines + [""]


def best_models(table: pd.DataFrame) -> dict:
    """Best model per task and family, selected on VALIDATION macro-F1 (never on test)."""
    out: dict = {}
    for (task, fam), part in table.dropna(subset=["f1_val"]).groupby(["task", "model_family"]):
        best = part.sort_values("f1_val", ascending=False).iloc[0]
        out.setdefault(task, {})[fam] = {"model_name": best["model_name"], "f1_val": round(best["f1_val"], 4),
                                         "f1_test": None if pd.isna(best["f1_test"]) else round(best["f1_test"], 4)}
    for task, fams in out.items():
        overall = max(fams.values(), key=lambda d: d["f1_val"])
        out[task]["_overall"] = overall
    return out


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def short_name(name: str) -> str:
    """Compact label for crowded plots ('tiny_transformer_4L_d128' -> 'tiny 4L d128')."""
    return (name.replace("tiny_transformer_", "tiny ").replace("e5_small_ft", "e5 fine-tuned")
            .replace("e5_frozen_lr", "e5 frozen+LR").replace("_", " "))


def plot_bubble(table: pd.DataFrame, path) -> None:
    """Test macro-F1 vs CPU latency (log x); bubble area ~ #params.

    The majority-class dummy is left out (F1 ~0.02-0.05 would squash the y axis).
    """
    plt = _plt()
    tasks = list(table["task"].unique())
    fig, axes = plt.subplots(1, len(tasks), figsize=(8 * len(tasks), 6.5), squeeze=False)
    offsets = [(7, 7), (7, -13), (-10, 12), (-10, -18)]
    for ax, task in zip(axes[0], tasks):
        part = table[(table["task"] == task) & table["latency_ms_per_1000"].notna()
                     & ~table["model_name"].str.contains("dummy")].sort_values("latency_ms_per_1000")
        for i, (_, r) in enumerate(part.iterrows()):
            params = r["n_params"] if pd.notna(r["n_params"]) and r["n_params"] > 0 else 1e5
            size = 40 + 1500 * np.sqrt(params / 1.2e8)
            ax.scatter(r["latency_ms_per_1000"], r["f1_test"], s=size, alpha=0.55,
                       color=FAMILY_COLORS.get(r["model_family"], "#555555"), edgecolors="black", linewidths=0.6)
            ax.annotate(f"{short_name(r['model_name'])} ({params / 1e6:.1f}M)",
                        (r["latency_ms_per_1000"], r["f1_test"]), fontsize=7,
                        xytext=offsets[i % len(offsets)], textcoords="offset points",
                        ha="left" if offsets[i % len(offsets)][0] > 0 else "right")
        ax.set_xscale("log")
        ax.set_xlabel("CPU latency, ms per 1000 titles (log scale, 8 threads, batch 64)")
        ax.set_ylabel("macro-F1 on test (>=2023)")
        ax.set_title(f"Task '{task}': quality vs speed (bubble area ~ #params)")
        ax.grid(alpha=0.3, which="both")
        if len(part):
            lo, hi = part["f1_test"].min(), part["f1_test"].max()
            ax.set_ylim(lo - 0.03, hi + 0.03)
            ax.set_xlim(part["latency_ms_per_1000"].min() / 2.5, part["latency_ms_per_1000"].max() * 3)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=f, markersize=9)
               for f, c in FAMILY_COLORS.items() if f in set(table["model_family"])]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_grouped_bars(table: pd.DataFrame, path) -> None:
    """Grouped bars: val / test / test_strict macro-F1 per model, one panel per task."""
    plt = _plt()
    tasks = list(table["task"].unique())
    fig, axes = plt.subplots(len(tasks), 1, figsize=(12, 4.5 * len(tasks)), squeeze=False)
    splits = [("f1_val", "val 2022", "#9ecae1"), ("f1_test", "test >=2023", "#3182bd"),
              ("f1_test_strict", "test_strict", "#08519c")]
    for ax, task in zip(axes[:, 0], tasks):
        part = table[table["task"] == task].sort_values("f1_test", ascending=False)
        x = np.arange(len(part))
        for i, (col, label, color) in enumerate(splits):
            ax.bar(x + (i - 1) * 0.27, part[col].fillna(0), width=0.27, label=label, color=color)
        ax.set_xticks(x)
        ax.set_xticklabels([short_name(n) for n in part["model_name"]], rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("macro-F1")
        ax.set_ylim(0, 1)
        ax.set_title(f"Task '{task}': macro-F1 per model and evaluation split")
        ax.legend(fontsize=8, loc="lower left")
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_test_vs_strict(table: pd.DataFrame, path) -> None:
    """Scatter of test vs test_strict macro-F1 with the identity line."""
    plt = _plt()
    part = table.dropna(subset=["f1_test", "f1_test_strict"])
    fig, ax = plt.subplots(figsize=(7, 6))
    markers = {"form": "o", "area": "s"}
    for _, r in part.iterrows():
        ax.scatter(r["f1_test"], r["f1_test_strict"], marker=markers.get(r["task"], "o"), s=60,
                   color=FAMILY_COLORS.get(r["model_family"], "#555555"), edgecolors="black", linewidths=0.5)
        ax.annotate(f"{r['task']}: {short_name(r['model_name'])}", (r["f1_test"], r["f1_test_strict"]), fontsize=6,
                    xytext=(4, -8), textcoords="offset points")
    if len(part):
        lo = float(min(part["f1_test"].min(), part["f1_test_strict"].min())) - 0.03
        hi = float(max(part["f1_test"].max(), part["f1_test_strict"].max())) + 0.03
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="test_strict = test")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
    ax.set_xlabel("macro-F1 on test")
    ax.set_ylabel("macro-F1 on test_strict (no near-duplicates of train)")
    ax.set_title("Robustness to near-duplicate removal (circle = form, square = area)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(quick: bool = False, force_strict: bool = False) -> dict:
    """Recompute test_strict, build the comparison table + figures + best-model map."""
    sfx = "_quick" if quick else ""          # smoke tests never overwrite the real outputs
    recomputed = recompute_test_strict(force=force_strict)
    rows = load_rows(quick=quick)
    if rows.empty:
        print("[compare] no experiments logged yet")
        return {"n_models": 0}
    table = build_table(rows)
    try:
        table.to_csv(COMPARISON_CSV.with_name(f"model_comparison{sfx}.csv"), index=False)
        md = to_markdown(table) + "\n" + "\n".join(seed_section() if not quick else [])
        COMPARISON_MD.with_name(f"model_comparison{sfx}.md").write_text(md, encoding="utf-8")
        with experiments_db.connect() as con:
            ranking = pd.read_sql_query(RANK_SQL, con)
        keep = ranking["model_name"].isin(set(table["model_name"]))
        ranking[keep].to_csv(config.RESULTS_DIR / f"model_ranking_sql{sfx}.csv", index=False)
    except OSError as exc:
        print(f"[compare] could not write results: {exc}")
    best = best_models(table)
    save_json(best, BEST_JSON.with_name(f"best_models{sfx}.json"))
    plot_bubble(table, config.FIGURES_DIR / f"f1_vs_latency{sfx}.png")
    plot_grouped_bars(table, config.FIGURES_DIR / f"f1_grouped_bars{sfx}.png")
    plot_test_vs_strict(table, config.FIGURES_DIR / f"f1_test_vs_strict{sfx}.png")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(table[["task", "model_name", "f1_val", "f1_test", "f1_test_strict", "n_params", "size_mb",
                     "latency_ms_per_1000"]].to_string(index=False))
    return {"n_models": int(len(table)), "recomputed_test_strict": len(recomputed),
            "best": {t: v["_overall"]["model_name"] for t, v in best.items()}}


if __name__ == "__main__":
    run()
