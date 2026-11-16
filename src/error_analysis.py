"""Error analysis of the best model of every family, for both tasks.

Uses SQL over the `predictions` table (joined with `acts` / `bodies` built by
src.database; a temporary `acts` table is created from the parquet if that
table is missing) plus pandas/sklearn:

* normalised confusion matrices (best model per family and task),
* per-class precision / recall / F1 table,
* top confused class pairs with 3 example titles each (highest confidence),
* error rate by year, by title-length bucket and by adopting-body type,
* agreement / disagreement between model families (e.g. TF-IDF right while the
  tiny transformer is wrong) + examples,
* results/error_analysis.md with the concrete observations.
"""
from __future__ import annotations

import json
import sqlite3

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from src import config, experiments_db
from src.model_utils import QUICK_SUFFIX, load_dataset, load_json

REPORT_MD = config.RESULTS_DIR / "error_analysis.md"
NN_SIM_PARQUET = config.PROCESSED_DIR / "nn_similarity.parquet"   # max char-ngram cosine to train (src.leakage)
PER_CLASS_CSV = config.RESULTS_DIR / "per_class_metrics.csv"
FAMILY_ORDER = ["baseline", "tfidf", "pretrained_frozen", "tiny_transformer", "pretrained", "posthoc"]
FAMILY_LABEL = {"baseline": "baseline", "tfidf": "TF-IDF+LR", "pretrained_frozen": "e5 frozen+LR",
                "tiny_transformer": "tiny transformer", "pretrained": "e5-small fine-tuned",
                "posthoc": "ensemble / prior-adjusted"}

SQL_CONFUSED_PAIRS = """
SELECT y_true, y_pred, COUNT(*) AS n
FROM predictions
WHERE exp_id = ? AND y_true <> y_pred
GROUP BY y_true, y_pred
ORDER BY n DESC
LIMIT ?
"""
SQL_PAIR_EXAMPLES = """
SELECT a.title, ROUND(p.confidence, 3) AS confidence
FROM predictions p JOIN acts a ON a.law_id = p.law_id
WHERE p.exp_id = ? AND p.y_true = ? AND p.y_pred = ?
ORDER BY p.confidence DESC
LIMIT 3
"""
SQL_ERROR_BY_YEAR = """
SELECT a.year AS year, COUNT(*) AS n, SUM(p.y_pred <> p.y_true) AS errors,
       ROUND(AVG(p.y_pred <> p.y_true), 4) AS error_rate
FROM predictions p JOIN acts a ON a.law_id = p.law_id
WHERE p.exp_id = ?
GROUP BY a.year
ORDER BY a.year
"""
SQL_ERROR_BY_LENGTH = """
WITH bucketed AS (
    SELECT p.y_true, p.y_pred,
           CASE WHEN a.title_len_words <= 10 THEN '01-10'
                WHEN a.title_len_words <= 20 THEN '11-20'
                WHEN a.title_len_words <= 30 THEN '21-30'
                WHEN a.title_len_words <= 50 THEN '31-50'
                ELSE '51+' END AS length_bucket
    FROM predictions p JOIN acts a ON a.law_id = p.law_id
    WHERE p.exp_id = ?
)
SELECT length_bucket, COUNT(*) AS n, SUM(y_pred <> y_true) AS errors,
       ROUND(AVG(y_pred <> y_true), 4) AS error_rate
FROM bucketed GROUP BY length_bucket ORDER BY length_bucket
"""
SQL_ERROR_BY_BODY = """
SELECT COALESCE(b.body_type, 'unknown') AS body_type, COUNT(*) AS n,
       SUM(p.y_pred <> p.y_true) AS errors, ROUND(AVG(p.y_pred <> p.y_true), 4) AS error_rate
FROM predictions p
JOIN acts a ON a.law_id = p.law_id
LEFT JOIN bodies b ON b.body_id = a.body_id
WHERE p.exp_id = ?
GROUP BY COALESCE(b.body_type, 'unknown')
HAVING COUNT(*) >= 30
ORDER BY error_rate DESC
"""
SQL_ERROR_BY_SIMILARITY = """
WITH s AS (
    SELECT p.y_true, p.y_pred,
           CASE WHEN n.max_cos < 0.60 THEN '1: <0.60'
                WHEN n.max_cos < 0.75 THEN '2: 0.60-0.75'
                WHEN n.max_cos < 0.85 THEN '3: 0.75-0.85'
                WHEN n.max_cos < 0.90 THEN '4: 0.85-0.90'
                WHEN n.max_cos < 0.95 THEN '5: 0.90-0.95'
                ELSE '6: >=0.95' END AS sim_bucket
    FROM predictions p JOIN nn_sim n ON n.law_id = p.law_id
    WHERE p.exp_id = ?
)
SELECT sim_bucket, COUNT(*) AS n, SUM(y_pred <> y_true) AS errors,
       ROUND(AVG(y_pred <> y_true), 4) AS error_rate
FROM s GROUP BY sim_bucket ORDER BY sim_bucket
"""
# Does a model reproduce the label of the most similar TRAIN title? ({col} = form_label / area_label)
SQL_NEIGHBOUR_LABEL = """
SELECT COUNT(*) AS n,
       SUM(nb.{col} = p.y_true)                              AS neighbour_label_is_true,
       SUM(p.y_pred = nb.{col})                              AS prediction_equals_neighbour,
       SUM(p.y_pred <> p.y_true)                             AS errors,
       SUM(p.y_pred <> p.y_true AND p.y_pred = nb.{col})     AS errors_copying_neighbour
FROM predictions p
JOIN nn_sim n ON n.law_id = p.law_id
JOIN acts nb ON nb.law_id = n.nn_law_id
WHERE p.exp_id = ?
"""
SQL_AGREEMENT = """
SELECT COUNT(*) AS n,
       SUM(a.y_pred = a.y_true AND b.y_pred = b.y_true)   AS both_right,
       SUM(a.y_pred = a.y_true AND b.y_pred <> b.y_true)  AS only_a_right,
       SUM(a.y_pred <> a.y_true AND b.y_pred = b.y_true)  AS only_b_right,
       SUM(a.y_pred <> a.y_true AND b.y_pred <> b.y_true) AS both_wrong,
       SUM(a.y_pred = b.y_pred)                           AS same_prediction
FROM predictions a JOIN predictions b ON a.law_id = b.law_id
WHERE a.exp_id = ? AND b.exp_id = ?
"""
SQL_ONLY_A_RIGHT = """
SELECT t.title, a.y_true, b.y_pred AS wrong_prediction, ROUND(b.confidence, 3) AS wrong_confidence
FROM predictions a
JOIN predictions b ON a.law_id = b.law_id
JOIN acts t ON t.law_id = a.law_id
WHERE a.exp_id = ? AND b.exp_id = ? AND a.y_pred = a.y_true AND b.y_pred <> b.y_true
ORDER BY b.confidence DESC
LIMIT ?
"""


# --------------------------------------------------------------------------
# Data access
# --------------------------------------------------------------------------
def open_db() -> sqlite3.Connection:
    """Connection with an `acts` table available (TEMP copy from parquet if needed)."""
    con = experiments_db.connect()
    has_acts = con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='acts'").fetchone()
    if not has_acts:
        df = load_dataset(["law_id", "title", "year", "form_label", "area_label"])
        df["title_len_words"] = df["title"].str.split().str.len()
        con.execute("CREATE TEMP TABLE acts (law_id TEXT PRIMARY KEY, title TEXT, year INTEGER, "
                    "title_len_words INTEGER, form_label TEXT, area_label TEXT, body_id INTEGER)")
        cols = ["law_id", "title", "year", "title_len_words", "form_label", "area_label"]
        con.executemany("INSERT INTO acts VALUES (?, ?, ?, ?, ?, ?, NULL)",
                        df[cols].astype(object).where(df[cols].notna(), None).itertuples(index=False))
        print("[error_analysis] `acts` table missing - using a temporary copy from the parquet")
    try:   # nearest-train-neighbour similarity from the leakage check -> TEMP table for SQL joins
        nn = pd.read_parquet(NN_SIM_PARQUET, columns=["law_id", "max_cos", "nn_law_id"])
        con.execute("CREATE TEMP TABLE nn_sim (law_id TEXT PRIMARY KEY, max_cos REAL, nn_law_id TEXT)")
        con.executemany("INSERT INTO nn_sim VALUES (?, ?, ?)",
                        zip(nn["law_id"].astype(str), nn["max_cos"].astype(float), nn["nn_law_id"].astype(str)))
    except (FileNotFoundError, OSError, KeyError) as exc:
        print(f"[error_analysis] nn_similarity unavailable ({exc}); similarity analysis skipped")
    # Commit so that a later failing query (pandas calls rollback()) cannot wipe the TEMP tables.
    con.commit()
    return con


def has_table(con: sqlite3.Connection, name: str) -> bool:
    """True if a (temp or main) table exists."""
    q = "SELECT 1 FROM sqlite_master WHERE name=? UNION ALL SELECT 1 FROM sqlite_temp_master WHERE name=?"
    return con.execute(q, (name, name)).fetchone() is not None


def select_models(con: sqlite3.Connection, quick: bool = False) -> pd.DataFrame:
    """Best model (by VAL macro-F1) per task and family, with the exp_id of its test row."""
    exps = pd.read_sql_query("SELECT exp_id, model_name, model_family, task, eval_split, f1_macro "
                             "FROM experiments", con)
    exps = exps[~exps["model_name"].str.endswith("_trainval")]   # refits trained on val: not selectable
    is_quick = exps["model_name"].str.endswith(QUICK_SUFFIX)
    exps = exps[is_quick | exps["model_family"].isin(["baseline", "tfidf"])] if quick else exps[~is_quick]
    val = exps[exps["eval_split"] == "val"]
    test = exps[exps["eval_split"] == "test"][["model_name", "task", "exp_id", "f1_macro"]]
    test = test.rename(columns={"f1_macro": "f1_test", "exp_id": "test_exp_id"})
    best = (val.sort_values("f1_macro", ascending=False).groupby(["task", "model_family"]).head(1)
            .rename(columns={"f1_macro": "f1_val"}).merge(test, on=["model_name", "task"]))
    n_preds = pd.read_sql_query("SELECT exp_id AS test_exp_id, COUNT(*) AS n_pred FROM predictions "
                                "GROUP BY exp_id", con)
    best = best.merge(n_preds, on="test_exp_id")
    best["family_rank"] = best["model_family"].map({f: i for i, f in enumerate(FAMILY_ORDER)}).fillna(99)
    return best.sort_values(["task", "family_rank"]).reset_index(drop=True)


def load_preds(con: sqlite3.Connection, exp_id: int) -> pd.DataFrame:
    """Test predictions of one experiment."""
    return pd.read_sql_query("SELECT law_id, y_true, y_pred, confidence FROM predictions WHERE exp_id=?",
                             con, params=(int(exp_id),))


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _short(label: str, n: int = 28) -> str:
    return label if len(label) <= n else label[:n - 1] + "…"


def plot_confusion(preds: pd.DataFrame, title: str, path, classes: list[str]) -> np.ndarray:
    """Row-normalised confusion matrix heatmap (rows = true class)."""
    plt = _plt()
    cm = confusion_matrix(preds["y_true"], preds["y_pred"], labels=classes).astype(float)
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    k = len(classes)
    size = 6 if k <= 10 else 13
    fig, ax = plt.subplots(figsize=(size + 1, size))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    fs = 9 if k <= 10 else 6
    ax.set_xticklabels([_short(c) for c in classes], rotation=60, ha="right", fontsize=fs)
    ax.set_yticklabels([f"{_short(c)} (n={int(cm[i].sum())})" for i, c in enumerate(classes)], fontsize=fs)
    for i in range(k):
        for j in range(k):
            if cm_norm[i, j] >= (0.01 if k <= 10 else 0.05):
                ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center", fontsize=fs - 1,
                        color="white" if cm_norm[i, j] > 0.5 else "black")
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.04, label="share of true class")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return cm


def plot_per_class_f1(per_class: pd.DataFrame, task: str, path) -> None:
    """Horizontal grouped bars: per-class F1 of each family's best model."""
    plt = _plt()
    part = per_class[per_class["task"] == task]
    pivot = part.pivot_table(index="class", columns="family_label", values="f1", aggfunc="first")
    support = part.groupby("class")["support"].first()
    pivot = pivot.loc[support.sort_values().index]
    fig, ax = plt.subplots(figsize=(10, max(4, 0.32 * len(pivot) * max(1, len(pivot.columns)) / 2)))
    h = 0.8 / max(1, len(pivot.columns))
    y = np.arange(len(pivot))
    for i, col in enumerate(pivot.columns):
        ax.barh(y + i * h, pivot[col].fillna(0), height=h, label=col)
    ax.set_yticks(y + h * (len(pivot.columns) - 1) / 2)
    ax.set_yticklabels([f"{_short(c, 34)} (n={int(support[c])})" for c in pivot.index], fontsize=7)
    ax.set_xlabel("F1 on test")
    ax.set_xlim(0, 1)
    ax.set_title(f"Task '{task}': per-class F1 of the best model per family (classes sorted by test support)")
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_error_breakdown(year_df: pd.DataFrame, len_df: pd.DataFrame, path) -> None:
    """Error rate by year and by title-length bucket (rows = task, lines = family)."""
    plt = _plt()
    tasks = list(year_df["task"].unique())
    fig, axes = plt.subplots(len(tasks), 2, figsize=(12, 4 * len(tasks)), squeeze=False)
    for row, task in enumerate(tasks):
        for col, (df, key, xlabel) in enumerate([(year_df, "year", "year adopted (test split)"),
                                                 (len_df, "length_bucket", "title length, words")]):
            ax = axes[row, col]
            part = df[df["task"] == task]
            for fam, g in part.groupby("family_label", sort=False):
                ax.plot(g[key].astype(str), g["error_rate"], "o-", label=fam)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("error rate (1 - accuracy)")
            ax.set_title(f"Task '{task}': error rate by {key.replace('_', ' ')}")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_similarity(sim_df: pd.DataFrame, path) -> None:
    """Error rate per bucket of max cosine similarity to the training set (one panel per task)."""
    plt = _plt()
    tasks = list(sim_df["task"].unique())
    fig, axes = plt.subplots(1, len(tasks), figsize=(7 * len(tasks), 4.5), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        part = sim_df[sim_df["task"] == task]
        for label, g in part.groupby("family_label", sort=False):
            ax.plot(g["sim_bucket"].str[3:], g["error_rate"], "o-", label=label)
        counts = part.groupby("sim_bucket")["n"].first()
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels([f"{b[3:]}\n(n={n})" for b, n in counts.items()], fontsize=7)
        ax.set_xlabel("max char-ngram cosine to any train title")
        ax.set_ylabel("error rate (1 - accuracy)")
        ax.set_title(f"Task '{task}': errors vs similarity to the training set (test)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------
def per_class_table(preds: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    """Precision / recall / F1 / support per class."""
    p, r, f, s = precision_recall_fscore_support(preds["y_true"], preds["y_pred"], labels=classes,
                                                 zero_division=0)
    return pd.DataFrame({"class": classes, "precision": p.round(4), "recall": r.round(4),
                         "f1": f.round(4), "support": s})


def confused_pairs(con, exp_id: int, top: int = 5) -> list[dict]:
    """Most frequent (true -> predicted) errors with 3 high-confidence example titles each."""
    pairs = pd.read_sql_query(SQL_CONFUSED_PAIRS, con, params=(int(exp_id), top))
    out = []
    for r in pairs.itertuples():
        ex = pd.read_sql_query(SQL_PAIR_EXAMPLES, con, params=(int(exp_id), r.y_true, r.y_pred))
        out.append({"true": r.y_true, "pred": r.y_pred, "n": int(r.n),
                    "examples": [(t, c) for t, c in zip(ex["title"], ex["confidence"])]})
    return out


def agreement(con, a: pd.Series, b: pd.Series, n_examples: int = 4) -> dict:
    """Contingency of correctness between two models + examples where only A is right."""
    row = pd.read_sql_query(SQL_AGREEMENT, con, params=(int(a.test_exp_id), int(b.test_exp_id))).iloc[0]
    ex_a = pd.read_sql_query(SQL_ONLY_A_RIGHT, con, params=(int(a.test_exp_id), int(b.test_exp_id), n_examples))
    ex_b = pd.read_sql_query(SQL_ONLY_A_RIGHT, con, params=(int(b.test_exp_id), int(a.test_exp_id), n_examples))
    return {"a": a.model_name, "b": b.model_name, **{k: int(v) for k, v in row.items()},
            "examples_only_a_right": ex_a.to_dict("records"), "examples_only_b_right": ex_b.to_dict("records")}


def confidence_profile(preds: pd.DataFrame) -> dict:
    """Accuracy of high- vs low-confidence predictions (is the confidence useful?)."""
    correct = preds["y_true"] == preds["y_pred"]
    conf = preds["confidence"].astype(float)
    hi = conf >= 0.9
    return {"share_conf_ge_0.9": round(float(hi.mean()), 4),
            "acc_conf_ge_0.9": round(float(correct[hi].mean()), 4) if hi.any() else None,
            "acc_conf_lt_0.9": round(float(correct[~hi].mean()), 4) if (~hi).any() else None,
            "share_of_errors_with_conf_ge_0.9": round(float((hi & ~correct).sum() / max(1, (~correct).sum())), 4)}


def _md_table(df: pd.DataFrame) -> list[str]:
    """Minimal markdown table."""
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    for row in df.itertuples(index=False):           # itertuples keeps int columns as ints
        cells = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(c.replace("|", "/") for c in cells) + " |")
    return lines


def analyse_task(con, task: str, models: pd.DataFrame, md: list[str], sfx: str = "") -> dict:
    """All analyses for one task; appends markdown to `md` and returns tables for plotting."""
    part = models[models["task"] == task]
    if part.empty:
        return {}
    overall = part.sort_values("f1_val", ascending=False).iloc[0]
    md += [f"## Task `{task}`", "",
           "Models analysed (best of each family, chosen on validation macro-F1):", ""]
    md += _md_table(part[["model_family", "model_name", "f1_val", "f1_test"]].round(4))
    md += ["", f"Overall best on validation: **{overall.model_name}** "
               f"(val {overall.f1_val:.4f}, test {overall.f1_test:.4f}).", ""]

    preds_by_model = {r.model_name: load_preds(con, r.test_exp_id) for r in part.itertuples()}
    classes = sorted(set().union(*[set(p["y_true"]) for p in preds_by_model.values()]))
    per_class_rows, year_rows, len_rows, body_rows, sim_rows, neighbour_rows = [], [], [], [], [], []
    for r in part.itertuples():
        preds = preds_by_model[r.model_name]
        fam = f"{FAMILY_LABEL.get(r.model_family, r.model_family)}: {r.model_name}"   # unique series label
        plot_confusion(preds, f"Task '{task}' - {r.model_name} ({fam}): normalised confusion matrix (test)",
                       config.FIGURES_DIR / f"confusion_{task}_{r.model_name}{sfx}.png", classes)
        pc = per_class_table(preds, classes)
        pc.insert(0, "model_name", r.model_name)
        pc.insert(0, "family_label", fam)
        pc.insert(0, "task", task)
        per_class_rows.append(pc)
        for sql, rows in ((SQL_ERROR_BY_YEAR, year_rows), (SQL_ERROR_BY_LENGTH, len_rows)):
            df = pd.read_sql_query(sql, con, params=(int(r.test_exp_id),))
            df.insert(0, "model_name", r.model_name)
            df.insert(0, "family_label", fam)
            df.insert(0, "task", task)
            rows.append(df)
        if has_table(con, "nn_sim"):
            df = pd.read_sql_query(SQL_ERROR_BY_SIMILARITY, con, params=(int(r.test_exp_id),))
            df.insert(0, "model_name", r.model_name)
            df.insert(0, "family_label", fam)
            df.insert(0, "task", task)
            sim_rows.append(df)
            nb = pd.read_sql_query(SQL_NEIGHBOUR_LABEL.format(col=config.TASKS[task]), con,
                                   params=(int(r.test_exp_id),)).iloc[0]
            neighbour_rows.append({"task": task, "model_name": r.model_name, **{k: int(v) for k, v in nb.items()}})
        if has_table(con, "bodies"):          # built by src.database (absent in quick runs)
            df = pd.read_sql_query(SQL_ERROR_BY_BODY, con, params=(int(r.test_exp_id),))
            df.insert(0, "model_name", r.model_name)
            df.insert(0, "family_label", fam)
            df.insert(0, "task", task)
            body_rows.append(df)
    per_class = pd.concat(per_class_rows)
    year_df, len_df = pd.concat(year_rows), pd.concat(len_rows)
    body_df = pd.concat(body_rows) if body_rows else pd.DataFrame()
    sim_df = pd.concat(sim_rows) if sim_rows else pd.DataFrame()

    # Per-class observations for the overall best model.
    best_pc = per_class[per_class["model_name"] == overall.model_name].sort_values("f1")
    md += ["### Per-class quality (best model)", "",
           f"Hardest classes for {overall.model_name}: " + "; ".join(
               f"**{c}** F1 {f:.2f} (P {p:.2f}, R {rc:.2f}, n={s})" for c, f, p, rc, s in
               best_pc[["class", "f1", "precision", "recall", "support"]].head(3).itertuples(index=False)) + ".",
           f"Easiest: " + "; ".join(f"**{c}** F1 {f:.2f} (n={s})" for c, f, s in
                                    best_pc[["class", "f1", "support"]].tail(3).iloc[::-1].itertuples(index=False))
           + ".", f"Full table: `results/{PER_CLASS_CSV.name}`; figure `reports/figures/per_class_f1_{task}.png`.", ""]

    # Confused pairs.
    pairs = confused_pairs(con, overall.test_exp_id)
    md += ["### Most confused class pairs (best model, test)", ""]
    for p in pairs:
        md.append(f"- **{p['true']} → {p['pred']}**: {p['n']} errors. Examples (highest confidence):")
        for title, conf in p["examples"]:
            md.append(f"    - \"{title[:220]}\" (conf {conf})")
    md.append("")

    # Year / length / body.
    yb = year_df[year_df["model_name"] == overall.model_name]
    lb = len_df[len_df["model_name"] == overall.model_name]
    md += ["### Error rate by year and title length (best model)", ""]
    md += _md_table(yb[["year", "n", "errors", "error_rate"]])
    md += [""] + _md_table(lb[["length_bucket", "n", "errors", "error_rate"]]) + [""]
    worst_len = lb.sort_values("error_rate", ascending=False).iloc[0]
    best_len = lb.sort_values("error_rate").iloc[0]
    md += [f"Error rate ranges from {best_len.error_rate:.3f} ({best_len.length_bucket} words) to "
           f"{worst_len.error_rate:.3f} ({worst_len.length_bucket} words); by year from "
           f"{yb.error_rate.min():.3f} ({int(yb.loc[yb.error_rate.idxmin(), 'year'])}) to "
           f"{yb.error_rate.max():.3f} ({int(yb.loc[yb.error_rate.idxmax(), 'year'])}).", ""]
    if not body_df.empty:
        bb = body_df[body_df["model_name"] == overall.model_name].head(6)
        md += ["Error rate by adopting-body type (SQL JOIN predictions-acts-bodies, groups with n>=30):", ""]
        md += _md_table(bb[["body_type", "n", "errors", "error_rate"]]) + [""]

    # Similarity to the nearest training title (from the leakage check).
    if not sim_df.empty:
        piv = sim_df.pivot_table(index="sim_bucket", columns="model_name", values="error_rate", aggfunc="first")
        counts = sim_df.groupby("sim_bucket")["n"].first()
        piv.insert(0, "n", counts)
        md += ["### Error rate vs similarity to the nearest train title (SQL JOIN predictions-nn_sim)", "",
               "`max_cos` = highest char-ngram TF-IDF cosine between a test title and any train title "
               "(data/processed/nn_similarity.parquet; >= 0.9 means a near-duplicate, excluded from test_strict).", ""]
        md += _md_table(piv.reset_index()) + [""]
        nbd = pd.DataFrame(neighbour_rows)
        best_nb = nbd[nbd["model_name"] == overall.model_name].iloc[0]
        md += [f"The label of the nearest train title equals the true label for "
               f"{best_nb.neighbour_label_is_true / best_nb.n:.1%} of test titles (a 1-NN 'copy' rule). "
               f"{overall.model_name} predicts the neighbour's label for "
               f"{best_nb.prediction_equals_neighbour / best_nb.n:.1%} of titles, and "
               f"{best_nb.errors_copying_neighbour / max(1, best_nb.errors):.1%} of its errors are exactly the "
               f"neighbour's label (errors inherited from a similar training title).", ""]
        md += _md_table(nbd.assign(
            neighbour_acc=(nbd.neighbour_label_is_true / nbd.n).round(4),
            copy_rate=(nbd.prediction_equals_neighbour / nbd.n).round(4),
            errors_copying_share=(nbd.errors_copying_neighbour / nbd.errors.clip(lower=1)).round(4))
            [["model_name", "n", "errors", "neighbour_acc", "copy_rate", "errors_copying_share"]]) + [""]

    # Confidence.
    conf = confidence_profile(preds_by_model[overall.model_name])
    md += [f"Confidence: {conf['share_conf_ge_0.9']:.1%} of test predictions have confidence >= 0.9 with "
           f"accuracy {conf['acc_conf_ge_0.9']}, vs {conf['acc_conf_lt_0.9']} below 0.9; "
           f"{conf['share_of_errors_with_conf_ge_0.9']:.1%} of all errors are confident (>= 0.9).", ""]

    # Agreement between families.
    md += ["### Agreement between model families (test, SQL self-join on predictions)", ""]
    agree_rows, agreements = [], []
    fams = list(part.itertuples(index=False))
    for i in range(len(fams)):
        for j in range(i + 1, len(fams)):
            a, b = part.iloc[i], part.iloc[j]
            ag = agreement(con, a, b)
            agreements.append(ag)
            agree_rows.append({"A": a.model_name, "B": b.model_name, "both right": ag["both_right"],
                               "only A right": ag["only_a_right"], "only B right": ag["only_b_right"],
                               "both wrong": ag["both_wrong"],
                               "same prediction": f"{ag['same_prediction'] / ag['n']:.3f}"})
    if agree_rows:
        md += _md_table(pd.DataFrame(agree_rows)) + [""]
    all_preds = pd.concat({m: p.set_index("law_id")["y_pred"] for m, p in preds_by_model.items()}, axis=1)
    truth = next(iter(preds_by_model.values())).set_index("law_id")["y_true"].reindex(all_preds.index)
    right = all_preds.eq(truth, axis=0)
    md += [f"Oracle (at least one of the {len(preds_by_model)} models right): {right.any(axis=1).mean():.4f} "
           f"accuracy; all models wrong on {(~right.any(axis=1)).sum()} of {len(right)} test titles.", ""]
    # Focus comparison: TF-IDF (or other baseline) vs our tiny transformer, both directions.
    tfidf = part[part["model_family"] == "tfidf"]
    if tfidf.empty:
        tfidf = part[part["model_family"] == "baseline"]
    tiny = part[part["model_family"] == "tiny_transformer"]
    if len(tfidf) and len(tiny):
        a, b = tfidf.iloc[0], tiny.iloc[0]
        ag = agreement(con, a, b, n_examples=4)
        md += [f"Examples where **{a.model_name}** is right and **{b.model_name}** is wrong:", ""]
        md += [f"- \"{e['title'][:200]}\" — true {e['y_true']}, {b.model_name} predicted "
               f"{e['wrong_prediction']} (conf {e['wrong_confidence']})" for e in ag["examples_only_a_right"]]
        md += ["", f"Examples where **{b.model_name}** is right and **{a.model_name}** is wrong:", ""]
        md += [f"- \"{e['title'][:200]}\" — true {e['y_true']}, {a.model_name} predicted "
               f"{e['wrong_prediction']} (conf {e['wrong_confidence']})" for e in ag["examples_only_b_right"]]
        md.append("")
    plot_per_class_f1(per_class, task, config.FIGURES_DIR / f"per_class_f1_{task}{sfx}.png")
    return {"per_class": per_class, "year": year_df, "length": len_df, "body": body_df, "similarity": sim_df,
            "pairs": pairs, "agreements": agreements, "best": overall.model_name}


def run(tasks: tuple[str, ...] = ("form", "area"), quick: bool = False) -> dict:
    """Run every analysis, save tables/figures and results/error_analysis.md."""
    sfx = "_quick" if quick else ""          # smoke tests never overwrite the real outputs
    con = open_db()
    try:
        models = select_models(con, quick=quick)
        if models.empty:
            print("[error_analysis] no experiments with test predictions found")
            return {"n_models": 0}
        md = ["# Error analysis", "",
              "All numbers come from the test predictions stored in SQLite (`predictions` table, test split "
              ">= 2023). Figures: `reports/figures/confusion_*.png`, `per_class_f1_*.png`, "
              "`error_by_year_length.png`, `error_by_similarity.png`.", ""]
        outs = {task: analyse_task(con, task, models, md, sfx) for task in tasks}
        outs = {k: v for k, v in outs.items() if v}
        per_class = pd.concat([o["per_class"] for o in outs.values()])
        per_class.to_csv(PER_CLASS_CSV.with_name(f"per_class_metrics{sfx}.csv"), index=False)
        year_df = pd.concat([o["year"] for o in outs.values()])
        len_df = pd.concat([o["length"] for o in outs.values()])
        year_df.to_csv(config.RESULTS_DIR / f"error_by_year{sfx}.csv", index=False)
        len_df.to_csv(config.RESULTS_DIR / f"error_by_length{sfx}.csv", index=False)
        bodies = [o["body"] for o in outs.values() if not o["body"].empty]
        if bodies:
            pd.concat(bodies).to_csv(config.RESULTS_DIR / f"error_by_body_type{sfx}.csv", index=False)
        plot_error_breakdown(year_df, len_df, config.FIGURES_DIR / f"error_by_year_length{sfx}.png")
        sims = [o["similarity"] for o in outs.values() if not o["similarity"].empty]
        if sims:
            sim_all = pd.concat(sims)
            sim_all.to_csv(config.RESULTS_DIR / f"error_by_similarity{sfx}.csv", index=False)
            plot_similarity(sim_all, config.FIGURES_DIR / f"error_by_similarity{sfx}.png")
        report = REPORT_MD.with_name(f"error_analysis{sfx}.md")
        report.write_text("\n".join(md) + "\n", encoding="utf-8")
        print(f"[error_analysis] wrote {report}")
        return {"n_models": int(len(models)), "best": {t: o["best"] for t, o in outs.items()},
                "report": str(report)}
    finally:
        con.close()


if __name__ == "__main__":
    run()
