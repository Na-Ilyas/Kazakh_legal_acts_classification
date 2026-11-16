"""Leakage analysis: temporal split, exact and near-duplicate titles.

Steps (``run()``):
  * exact overlap of ``title_norm`` train vs val/test after labels.build_dataset
    (must be 0) + how many rows build_dataset removed;
  * digit-masked overlap (titles identical up to numbers / dates);
  * within-train duplicate rate;
  * near duplicates: char_wb 3-5-gram TF-IDF fitted on the DIGIT-MASKED train
    titles (``normalize_title(t, mask_digits=True)``), max cosine of every
    val/test masked title to any train masked title. Masking matters: titles of
    yearly templates ("... 2022 жылғы ... N 15 шешіміне ...") differ only by
    numbers and must count as near duplicates. Computed in small batches with a
    thread pool (scipy sparse matmul releases the GIL); only row max / argmax
    are kept, the full similarity matrix is never materialised;
  * test_strict = test rows with max masked cosine < 0.9 AND no masked-exact
    match in train -> results/test_strict_ids.csv;
  * nearest LABELLED train neighbour per task = 1-NN classifier ('char_1nn');
  * results/leakage_report.json, data/processed/nn_similarity.parquet, figure.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from src import config, experiments_db, labels
from src.data_io import _peak_rss_bytes, _reset_peak_rss, load_dataset
from src.metrics import compute_metrics

NEAR_DUP_THRESHOLD = 0.9
STRICT_IDS_CSV = config.RESULTS_DIR / "test_strict_ids.csv"
REPORT_JSON = config.RESULTS_DIR / "leakage_report.json"
NN_PARQUET = config.PROCESSED_DIR / "nn_similarity.parquet"
HIST_PNG = config.FIGURES_DIR / "leakage_max_cosine_hist.png"
BATCH_ROWS = 100      # eval rows per sparse product (100 x 156k dense ~ 60 MB)
N_THREADS = 6         # <= 8 per the shared-machine rules


def mask_title(title: str) -> str:
    """Key used for near-duplicate detection: normalised title with digits -> '0'."""
    return labels.normalize_title(title, mask_digits=True)


# --------------------------------------------------------------------------
# Exact / masked duplicates
# --------------------------------------------------------------------------
def exact_overlap(ds: pd.DataFrame, key: str = "title_norm") -> dict:
    """Number of val/test rows whose `key` also occurs in train (and val->test)."""
    train = set(ds.loc[ds["split"] == "train", key])
    val = set(ds.loc[ds["split"] == "val", key])
    out = {}
    for split in ("val", "test"):
        sub = ds.loc[ds["split"] == split, key]
        out[f"train_{split}_rows"] = int(sub.isin(train).sum())
        out[f"train_{split}_pct"] = round(100 * float(sub.isin(train).mean()), 3)
    test = ds.loc[ds["split"] == "test", key]
    out["val_test_rows"] = int(test.isin(val).sum())
    return out


def within_split_duplicates(ds: pd.DataFrame, split: str = "train") -> dict:
    """Duplicate rate of title_norm inside one split."""
    titles = ds.loc[ds["split"] == split, "title_norm"]
    return {
        "rows": int(len(titles)),
        "rows_in_duplicate_groups": int(titles.duplicated(keep=False).sum()),
        "redundant_copies": int(titles.duplicated().sum()),
        "redundant_copies_pct": round(100 * float(titles.duplicated().mean()), 2),
        "distinct_titles": int(titles.nunique()),
    }


# --------------------------------------------------------------------------
# Near duplicates (batched sparse cosine)
# --------------------------------------------------------------------------
def fit_char_vectorizer(train_titles: list[str]) -> tuple[TfidfVectorizer, sp.csr_matrix]:
    """char_wb 3-5-gram TF-IDF fitted on (masked) train titles only (L2-normalised rows)."""
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                          sublinear_tf=True, dtype=np.float32)
    X = vec.fit_transform(train_titles)
    return vec, X


def max_cosine_to_train(X_eval: sp.csr_matrix, X_train_T: sp.csr_matrix,
                        excluded: dict[str, np.ndarray] | None = None,
                        batch_rows: int = BATCH_ROWS, n_threads: int = N_THREADS) -> dict[str, tuple]:
    """Max cosine / argmax of every eval row over the train rows.

    Returns {"all": (best, arg), <name>: (best, arg), ...}: "all" searches every
    train row; each entry of `excluded` (name -> train column indices to skip,
    e.g. unlabelled rows of one task) gives the best match among the others.
    Rows are L2-normalised so the dot product is the cosine. Each batch result
    (batch_rows x n_train) is densified only for that batch, then reduced.
    """
    n = X_eval.shape[0]
    excluded = excluded or {}
    out = {k: (np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.int64)) for k in ["all", *excluded]}

    def _job(start: int) -> None:
        stop = min(start + batch_rows, n)
        rows = np.arange(stop - start)
        prod = (X_eval[start:stop] @ X_train_T).toarray()
        a = prod.argmax(axis=1)
        out["all"][0][start:stop], out["all"][1][start:stop] = prod[rows, a], a
        for name, cols in excluded.items():
            if len(cols) == 0:
                out[name][0][start:stop], out[name][1][start:stop] = prod[rows, a], a
                continue
            saved = prod[:, cols].copy()
            prod[:, cols] = -1.0
            a2 = prod.argmax(axis=1)
            out[name][0][start:stop], out[name][1][start:stop] = prod[rows, a2], a2
            prod[:, cols] = saved

    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        list(pool.map(_job, range(0, n, batch_rows)))
    return out


def similarity_distribution(sim: np.ndarray) -> dict:
    """Quantiles and share above thresholds of a max-cosine vector."""
    qs = [0.05, 0.25, 0.5, 0.75, 0.95]
    return {
        "n": int(len(sim)),
        "mean": round(float(sim.mean()), 4),
        "quantiles": {str(q): round(float(np.quantile(sim, q)), 4) for q in qs},
        "pct_ge_0.95": round(100 * float((sim >= 0.95).mean()), 2),
        "pct_ge_0.9": round(100 * float((sim >= 0.9).mean()), 2),
        "pct_ge_0.8": round(100 * float((sim >= 0.8).mean()), 2),
        "pct_ge_0.7": round(100 * float((sim >= 0.7).mean()), 2),
        "pct_lt_0.5": round(100 * float((sim < 0.5).mean()), 2),
    }


def one_nn_evaluation(ds: pd.DataFrame, eval_df: pd.DataFrame, search: dict, strict_ids: set[str]) -> dict:
    """1-NN classifier per task: label of the nearest LABELLED train title.

    Uses the shared metric convention (src.metrics: macro-F1 over classes present
    in y_true). confidence = cosine to that neighbour.
    """
    train = ds[ds["split"] == "train"].reset_index(drop=True)
    out: dict = {}
    for task, col in config.TASKS.items():
        best, arg = search[task]
        train_labels = train[col].astype(object).to_numpy()
        frame = eval_df[["law_id", "split", "max_cos"]].assign(
            y_pred=[str(v) for v in train_labels[arg]], nn_cos=best.round(4))
        frame = frame.merge(ds[["law_id", col]], on="law_id").dropna(subset=[col])
        frame["y_true"] = frame[col].astype(str)
        res: dict = {"n_train_labelled": int(train[col].notna().sum())}
        for split in ("val", "test", "test_strict"):
            if split == "test_strict":
                sub = frame[(frame["split"] == "test") & frame["law_id"].isin(strict_ids)]
            else:
                sub = frame[frame["split"] == split]
            m = compute_metrics(sub["y_true"], sub["y_pred"])
            res[split] = {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}
            res[split]["_frame"] = sub
        test = frame[frame["split"] == "test"]
        bins = pd.cut(test["max_cos"], [0, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0001], right=False)
        res["test_agreement_by_similarity"] = {
            str(b): {"n": int(len(g)), "agreement": round(float((g["y_true"] == g["y_pred"]).mean()), 4)}
            for b, g in test.groupby(bins, observed=True)
        }
        out[task] = res
    return out


def log_one_nn(results: dict, fit_time: float, model_bytes: int, nnz: int,
               latency_ms: float, peak_mb: float) -> None:
    """Store the 1-NN baseline in the experiments DB (val, test, test_strict)."""
    for task, res in results.items():
        for split in ("val", "test", "test_strict"):
            frame = res[split].pop("_frame")
            metrics = dict(res[split], n_params=int(nnz), size_mb=round(model_bytes / 2**20, 2),
                           latency_ms_per_1000=round(latency_ms, 1), latency_device="cpu",
                           peak_mem_mb=round(peak_mb, 1), train_time_s=round(fit_time, 2),
                           n_train=res["n_train_labelled"])
            try:
                exp_id = experiments_db.log_experiment(
                    "char_1nn", "baseline", task, split, metrics,
                    hyperparams={"analyzer": "char_wb", "ngram_range": [3, 5], "min_df": 2,
                                 "sublinear_tf": True, "k": 1, "metric": "cosine",
                                 "input": "normalize_title(mask_digits=True)"},
                    notes=("1-nearest-neighbour on char_wb 3-5 TF-IDF of digit-masked normalised titles; "
                           "the neighbour is searched among train rows labelled for this task (n_train). "
                           "n_params = non-zeros stored in the train TF-IDF matrix (non-parametric). "
                           "confidence = cosine to the neighbour. Metrics: src.metrics (macro over classes "
                           "present in y_true). peak_mem = peak RSS of the whole leakage process "
                           "(vectoriser fit + search for all eval titles, 6 threads); latency measured with "
                           "the same 6-thread batched search."))
                if split == "test":
                    experiments_db.log_predictions(exp_id, frame["law_id"], frame["y_true"],
                                                   frame["y_pred"], frame["nn_cos"])
            except Exception as exc:  # DB locked / schema problems must not kill the analysis
                print(f"[leakage] could not log char_1nn {task}/{split}: {exc}")


def plot_histogram(nn: pd.DataFrame, path=HIST_PNG) -> None:
    """Histogram of max (masked) cosine to train for val and test titles."""
    fig, ax = plt.subplots(figsize=(9, 4.8))
    bins = np.linspace(0.2, 1.0001, 41)
    for split, color in (("val", "#eb6834"), ("test", "#1baf7a")):   # same split colours as src.eda
        vals = nn.loc[nn["split"] == split, "max_cos"].to_numpy()
        weights = np.full(len(vals), 100.0 / max(len(vals), 1))     # % of the split -> comparable shapes
        ax.hist(vals, bins=bins, weights=weights, histtype="stepfilled", alpha=0.18, color=color)
        ax.hist(vals, bins=bins, weights=weights, histtype="step", lw=2, color=color,
                label=f"{split} (n={len(vals):,}; ≥0.9: {100 * (vals >= 0.9).mean():.1f}%)")
    ax.axvline(NEAR_DUP_THRESHOLD, color="#0b0b0b", ls="--", lw=1, label="near-duplicate threshold 0.9")
    ax.set_title("Max char 3-5-gram TF-IDF cosine of each eval title to any train title\n"
                 "(titles normalised, digits masked)")
    ax.set_xlabel("max cosine similarity to the train set")
    ax.set_ylabel("% of the split's titles")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left")
    fig.tight_layout()
    try:
        fig.savefig(path, dpi=150)
    except OSError as exc:
        print(f"[leakage] could not save figure: {exc}")
    plt.close(fig)


def measure_latency(vec: TfidfVectorizer, X_train_T: sp.csr_matrix, titles: list[str],
                    repeats: int = 3) -> float:
    """Median ms to classify 1000 raw titles with 1-NN (mask + vectorise + search), CPU."""
    sample = titles[:1000]
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        X = vec.transform([mask_title(t) for t in sample])
        max_cosine_to_train(X, X_train_T)
        times.append(time.perf_counter() - t0)
    return 1000 * float(np.median(times))


def _write_outputs(nn: pd.DataFrame, strict: pd.DataFrame) -> None:
    """Save nn_similarity.parquet and test_strict_ids.csv (fail loudly: models depend on them)."""
    try:
        nn.to_parquet(NN_PARQUET, index=False)
        tmp = STRICT_IDS_CSV.with_suffix(".csv.part")
        strict[["law_id"]].to_csv(tmp, index=False)
        tmp.replace(STRICT_IDS_CSV)
    except OSError as exc:
        raise RuntimeError(f"[leakage] could not write leakage outputs: {exc}") from exc


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def run(log_to_db: bool = True) -> dict:
    """Run every leakage check and save the report, strict ids, figure."""
    t_start = time.perf_counter()
    _reset_peak_rss()
    ds = load_dataset(["law_id", "title", "title_norm", "split", "year", "form_label", "area_label"])
    ds["split"] = ds["split"].astype(str)
    try:
        split_stats = json.loads((config.RESULTS_DIR / "split_stats.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        split_stats = {}

    report: dict = {"temporal_split": {
        sp_: {"years": [int(g["year"].min()), int(g["year"].max())], "rows": int(len(g))}
        for sp_, g in ds.groupby("split")
    }}
    report["removed_by_build_dataset"] = {
        "eval_titles_seen_in_train": split_stats.get("dropped_eval_titles_seen_in_train"),
        "test_titles_seen_in_val": split_stats.get("dropped_test_titles_seen_in_val"),
    }
    report["exact_overlap_after_build"] = exact_overlap(ds, "title_norm")
    ds["title_masked"] = ds["title"].map(mask_title)
    report["digit_masked_overlap"] = exact_overlap(ds, "title_masked")
    report["within_train_duplicates"] = within_split_duplicates(ds, "train")
    assert report["exact_overlap_after_build"]["train_test_rows"] == 0, "exact leakage train->test!"
    assert report["exact_overlap_after_build"]["train_val_rows"] == 0, "exact leakage train->val!"
    print(f"[leakage] exact overlap OK; digit-masked overlap: {report['digit_masked_overlap']}")

    # ---- near duplicates on digit-masked titles
    is_train = (ds["split"] == "train").to_numpy()
    train = ds[is_train].reset_index(drop=True)
    t0 = time.perf_counter()
    vec, X_train = fit_char_vectorizer(train["title_masked"].tolist())
    fit_time = time.perf_counter() - t0
    X_train_T = X_train.T.tocsr()
    del X_train
    print(f"[leakage] vectorizer fitted in {fit_time:.1f}s, train matrix {X_train_T.shape[::-1]}, nnz={X_train_T.nnz:,}")

    eval_df = ds.loc[~is_train, ["law_id", "split", "title_masked"]].reset_index(drop=True)
    unlabelled = {task: np.flatnonzero(train[col].isna().to_numpy()) for task, col in config.TASKS.items()}
    t0 = time.perf_counter()
    X_eval = vec.transform(eval_df["title_masked"].tolist())
    search = max_cosine_to_train(X_eval, X_train_T, excluded=unlabelled)
    search_time = time.perf_counter() - t0
    print(f"[leakage] nearest-neighbour search for {len(eval_df):,} titles: {search_time:.1f}s")

    best, arg = search["all"]
    train_masked = set(train["title_masked"])
    eval_df["max_cos"] = best.round(4)
    nn = pd.DataFrame({"law_id": eval_df["law_id"], "split": eval_df["split"], "max_cos": eval_df["max_cos"],
                       "nn_law_id": train["law_id"].to_numpy()[arg],
                       "masked_exact_in_train": eval_df["title_masked"].isin(train_masked).to_numpy()})

    report["near_duplicates"] = {
        "method": ("char_wb 3-5-gram TF-IDF (sublinear tf, min_df=2) fitted on train "
                   "normalize_title(mask_digits=True); cosine"),
        "vocab_size": int(len(vec.vocabulary_)),
        "fit_time_s": round(fit_time, 1),
        "search_time_s": round(search_time, 1),
        "val": similarity_distribution(nn.loc[nn["split"] == "val", "max_cos"].to_numpy()),
        "test": similarity_distribution(nn.loc[nn["split"] == "test", "max_cos"].to_numpy()),
    }
    test_nn = nn[nn["split"] == "test"]
    below = test_nn["max_cos"] < NEAR_DUP_THRESHOLD
    strict = test_nn[below & ~test_nn["masked_exact_in_train"]]
    _write_outputs(nn, strict)
    report["test_strict"] = {
        "definition": "test rows with max masked cosine < 0.9 and no digit-masked exact match in train",
        "threshold": NEAR_DUP_THRESHOLD, "n_test": int(len(test_nn)),
        "n_below_threshold": int(below.sum()),
        "n_masked_exact_below_threshold": int((below & test_nn["masked_exact_in_train"]).sum()),
        "n_masked_exact_total": int(test_nn["masked_exact_in_train"].sum()),
        "n_strict": int(len(strict)),
        "pct_kept": round(100 * len(strict) / max(len(test_nn), 1), 2)}
    print(f"[leakage] test_strict: {len(strict):,}/{len(test_nn):,} test rows ({report['test_strict']})")

    # ---- 1-NN baseline (label of the nearest labelled train title)
    one_nn = one_nn_evaluation(ds, eval_df, search, set(strict["law_id"]))
    latency_ms = measure_latency(vec, X_train_T, ds.loc[ds["split"] == "test", "title"].tolist())
    model_bytes = X_train_T.data.nbytes + X_train_T.indices.nbytes + X_train_T.indptr.nbytes
    peak_mb = _peak_rss_bytes() / 2**20
    if log_to_db:
        log_one_nn(one_nn, fit_time, model_bytes, X_train_T.nnz, latency_ms, peak_mb)
    else:
        for res in one_nn.values():
            for split in ("val", "test", "test_strict"):
                res[split].pop("_frame", None)
    report["char_1nn"] = one_nn
    report["char_1nn_cost"] = {"latency_ms_per_1000": round(latency_ms, 1),
                               "index_size_mb": round(model_bytes / 2**20, 1),
                               "peak_rss_mb": round(peak_mb, 1)}
    plot_histogram(nn)
    report["elapsed_s"] = round(time.perf_counter() - t_start, 1)
    try:
        REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as exc:
        print(f"[leakage] could not write report: {exc}")
    summary = {
        "test_pct_ge_0.9": report["near_duplicates"]["test"]["pct_ge_0.9"],
        "test_pct_ge_0.8": report["near_duplicates"]["test"]["pct_ge_0.8"],
        "n_test_strict": report["test_strict"]["n_strict"],
        "char_1nn_f1_test": {t: one_nn[t]["test"]["f1_macro"] for t in one_nn},
        "char_1nn_f1_test_strict": {t: one_nn[t]["test_strict"]["f1_macro"] for t in one_nn},
        "elapsed_s": report["elapsed_s"],
    }
    print(f"[leakage] {summary}")
    return summary


if __name__ == "__main__":
    run()
