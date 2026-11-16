"""Classical baselines for both tasks (form, area): Dummy + TF-IDF + LogisticRegression.

Models
  (a) majority_dummy      DummyClassifier(most_frequent)
  (b) tfidf_word_lr       word 1-2-grams (text_prep.tokenize: NFC, lower, homoglyph folding, digits->num)
  (c) tfidf_char_lr       char_wb 2-5-grams on text_prep.normalize(title)
  (d) tfidf_stem_lr       1-2-grams of Kazakh stems without stop words (text_prep.analyzer_stems)
  (e) tfidf_word_char_lr  hstack(word, char) features (FeatureUnion)

Classifier: L2-regularised multinomial-style logistic regression (log-loss,
one-vs-rest) optimised with SGD (``SGDClassifier(loss="log_loss")``). The
lbfgs ``LogisticRegression`` was tried first: fine for word features, but for
char 2-5-grams (~80M non-zeros, 33 classes) each fit needed >1.5 GB of lbfgs
history and tens of minutes, beyond the shared-machine budget. The same
optimiser is used for every representation so the comparison is fair.

Protocol: fit on train, choose the regularisation (alpha = 1 / (C * n_train),
C in {1, 4, 16, 64}) and class_weight on val macro-F1 (parallel threads share the
sparse matrices), refit the chosen configuration alone so its train time /
peak memory are measured cleanly, report val / test / test_strict. Every result is logged to SQLite
(experiments_db) and test predictions are stored with confidence = max proba.
"""
from __future__ import annotations

import json
import os
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import FeatureUnion, Pipeline
from threadpoolctl import threadpool_limits

from src import config, experiments_db, text_prep
from src.data_io import load_dataset
from src.metrics import compute_metrics, eval_labels, macro_f1

RESULTS_SUBDIR = config.RESULTS_DIR / "baseline"
SUMMARY_JSON = config.RESULTS_DIR / "baseline_summary.json"
STRICT_IDS_CSV = config.RESULTS_DIR / "test_strict_ids.csv"
C_GRID = [1.0, 4.0, 16.0, 64.0]
N_THREADS = 2            # grid configs in parallel ...
SGD_JOBS = 4             # ... x one-vs-rest classes in parallel = 8 cores max
# tol / n_iter_no_change = sklearn defaults (stop when the epoch loss stops improving by
# 1e-3 for 5 epochs); max_iter is a generous cap, convergence is recorded per model.
LR_KW = dict(loss="log_loss", penalty="l2", max_iter=100, tol=1e-3, n_iter_no_change=5,
             random_state=config.SEED)


# --------------------------------------------------------------------------
# Vectorisers (module-level callables => the pipelines are picklable)
# --------------------------------------------------------------------------
def make_vectorizer(kind: str) -> TfidfVectorizer:
    """TF-IDF vectoriser for one representation."""
    common = dict(sublinear_tf=True, dtype=np.float32, lowercase=False)
    if kind == "word":
        return TfidfVectorizer(tokenizer=text_prep.tokenize, token_pattern=None, ngram_range=(1, 2),
                               min_df=2, max_features=400_000, **common)
    if kind == "char":
        return TfidfVectorizer(analyzer="char_wb", preprocessor=text_prep.normalize, ngram_range=(2, 5),
                               min_df=3, max_features=300_000, **common)
    if kind == "stem":
        return TfidfVectorizer(tokenizer=text_prep.analyzer_stems, token_pattern=None, ngram_range=(1, 2),
                               min_df=2, max_features=400_000, **common)
    raise ValueError(f"unknown vectorizer kind {kind!r}")


MODELS = {  # model_name -> (vectorizer kind(s), description)
    "tfidf_word_lr": (["word"], "TF-IDF word 1-2-grams + LR"),
    "tfidf_char_lr": (["char"], "TF-IDF char_wb 2-5-grams + LR"),
    "tfidf_stem_lr": (["stem"], "TF-IDF Kazakh stems 1-2-grams (no stop words) + LR"),
    "tfidf_word_char_lr": (["word", "char"], "TF-IDF word 1-2 + char_wb 2-5 (FeatureUnion) + LR"),
}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def load_strict_ids() -> set[str]:
    """law_ids of test_strict (written by src.leakage). Raises if unavailable:
    silently evaluating on the full test set would mislabel results as 'strict'."""
    try:
        ids = set(pd.read_csv(STRICT_IDS_CSV, dtype=str)["law_id"])
    except (OSError, KeyError) as exc:
        raise FileNotFoundError(f"{STRICT_IDS_CSV} unavailable ({exc}) - run `python -m src.leakage` first") from exc
    if not ids:
        raise ValueError(f"{STRICT_IDS_CSV} is empty")
    return ids


def model_dir(name: str, task: str) -> Path:
    """artifacts/models/<name>_<task>/"""
    return config.MODELS_DIR / f"{name}_{task}"


def save_model(model, name: str, task: str, meta: dict) -> float:
    """joblib-dump the fitted pipeline + meta.json; returns size on disk in MB."""
    d = model_dir(name, task)
    d.mkdir(parents=True, exist_ok=True)
    path = d / "model.joblib"
    try:
        joblib.dump(model, path)
        (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as exc:
        print(f"[baseline] could not save {path}: {exc}")
        return float("nan")
    return path.stat().st_size / 2**20


def load_baseline(name: str, task: str):
    """Load a saved baseline pipeline (vectoriser + classifier)."""
    path = model_dir(name, task) / "model.joblib"
    try:
        return joblib.load(path)
    except (OSError, FileNotFoundError) as exc:
        raise FileNotFoundError(f"{path} not found - run `python -m src.baseline` first") from exc


def predict_baseline(model, texts: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict raw titles -> (labels, class-probability matrix, class names)."""
    probas = model.predict_proba(list(texts))
    classes = model.classes_ if hasattr(model, "classes_") else model[-1].classes_
    return classes[probas.argmax(axis=1)], probas, np.asarray(classes)


def best_baseline(task: str) -> str | None:
    """Name of the best baseline on val for a task (from results/baseline_summary.json)."""
    try:
        return json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))["best_by_val"][task]["model_name"]
    except (OSError, KeyError, json.JSONDecodeError):
        return None


def measure_latency(model, titles: list[str], repeats: int = 3) -> float:
    """Median ms to classify 1000 raw titles on CPU (vectoriser transform included)."""
    sample = list(titles[:1000])
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        model.predict_proba(sample)
        times.append(time.perf_counter() - t0)
    return 1000 * float(np.median(times))


def _fit_lr(X, y, C: float, cw) -> tuple[SGDClassifier, float]:
    """Fit one logistic regression (SGD, alpha = 1/(C*n)); return (model, wall time)."""
    t0 = time.perf_counter()
    clf = SGDClassifier(alpha=1.0 / (C * X.shape[0]), class_weight=cw, n_jobs=SGD_JOBS, **LR_KW).fit(X, y)
    return clf, time.perf_counter() - t0


def tune_on_val(X_tr, y_tr, X_val, y_val) -> tuple[dict, list[dict]]:
    """Grid over C (class_weight='balanced'), then class_weight=None at the best C. Parallel threads."""
    def _eval(cfg):
        clf, dt = _fit_lr(X_tr, y_tr, cfg["C"], cfg["class_weight"])
        f1 = macro_f1(y_val, clf.predict(X_val))
        return {**cfg, "val_f1_macro": round(float(f1), 4), "fit_s_parallel": round(dt, 1),
                "epochs": int(clf.n_iter_)}

    with ThreadPoolExecutor(max_workers=N_THREADS) as pool:
        grid = list(pool.map(_eval, [{"C": c, "class_weight": "balanced"} for c in C_GRID]))
        best = max(grid, key=lambda r: r["val_f1_macro"])
        grid.append(_eval({"C": best["C"], "class_weight": None}))
    best = max(grid, key=lambda r: r["val_f1_macro"])
    return {"C": best["C"], "class_weight": best["class_weight"]}, grid


def evaluate_and_log(name: str, family: str, task: str, model, splits: dict, cost: dict,
                     hyperparams: dict, notes: str) -> dict:
    """Score val/test/test_strict, log to SQLite, store test predictions and per-class report."""
    out = {}
    for split, (titles, y, ids) in splits.items():
        probas = model.predict_proba(titles)
        classes = model.classes_ if hasattr(model, "classes_") else model[-1].classes_
        y_pred = np.asarray(classes)[probas.argmax(axis=1)]
        m = compute_metrics(y, y_pred)
        out[split] = {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}
        try:
            exp_id = experiments_db.log_experiment(name, family, task, split, {**m, **cost},
                                                   hyperparams=hyperparams, notes=notes)
            if split == "test":
                experiments_db.log_predictions(exp_id, ids, y, y_pred, probas.max(axis=1))
        except Exception as exc:  # keep training even if the DB is busy
            print(f"[baseline] could not log {name}/{task}/{split}: {exc}")
        if split in ("test", "test_strict"):
            rep = classification_report(y, y_pred, labels=eval_labels(y), output_dict=True, zero_division=0)
            try:
                RESULTS_SUBDIR.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(rep).T.round(4).to_csv(RESULTS_SUBDIR / f"{name}_{task}_{split}_report.csv")
            except OSError as exc:
                print(f"[baseline] could not save report: {exc}")
    return out


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------
def _task_data(ds: pd.DataFrame, task: str, strict_ids: set[str]) -> dict:
    """Per-split (titles, labels, law_ids, row positions) for one task (NaN labels dropped)."""
    col = config.TASKS[task]
    data = {}
    for split in ("train", "val", "test"):
        sub = ds[(ds["split"] == split) & ds[col].notna()]
        data[split] = sub
    test = data["test"]
    data["test_strict"] = test[test["law_id"].isin(strict_ids)]
    return data


def run_dummy(ds: pd.DataFrame, strict_ids: set[str], results: dict) -> None:
    """(a) majority-class baseline for both tasks."""
    for task, col in config.TASKS.items():
        data = _task_data(ds, task, strict_ids)
        tr = data["train"]
        tracemalloc.start()
        t0 = time.perf_counter()
        model = DummyClassifier(strategy="most_frequent").fit(tr["title"].to_numpy().reshape(-1, 1), tr[col].astype(str))
        model = Pipeline([("identity", _Reshape()), ("clf", model)])
        train_time = time.perf_counter() - t0
        peak_mb = tracemalloc.get_traced_memory()[1] / 2**20
        tracemalloc.stop()
        size = save_model(model, "majority_dummy", task, {"task": task, "classes": list(model[-1].classes_)})
        cost = {"n_params": 0, "size_mb": round(size, 4), "train_time_s": round(train_time, 3),
                "latency_ms_per_1000": round(measure_latency(model, data["test"]["title"].tolist()), 2),
                "latency_device": "cpu", "peak_mem_mb": round(peak_mb, 2), "n_train": int(len(tr))}
        splits = {s: (data[s]["title"].tolist(), data[s][col].astype(str).to_numpy(), data[s]["law_id"].tolist())
                  for s in ("val", "test", "test_strict")}
        results.setdefault(task, {})["majority_dummy"] = evaluate_and_log(
            "majority_dummy", "baseline", task, model, splits, cost, {"strategy": "most_frequent"},
            "Predicts the most frequent train class.")


class _Reshape:
    """Pipeline step turning a list of titles into a 2-D array (for DummyClassifier)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.asarray(X, dtype=object).reshape(-1, 1)


def run_tfidf_model(name: str, feats: dict, vecs: dict, ds: pd.DataFrame, strict_ids: set[str],
                    results: dict, tuning: dict) -> None:
    """Tune / refit / evaluate / save one TF-IDF + LR model for both tasks."""
    kinds, desc = MODELS[name]
    vec_time = sum(feats[k]["fit_time_s"] for k in kinds)
    for task, col in config.TASKS.items():
        data = _task_data(ds, task, strict_ids)

        def _X(split):
            # `pos` = row index inside the split's feature matrix (labels may drop rows)
            rows = data[split]["pos"].to_numpy()
            mats = [feats[k][split][rows] for k in kinds]
            return sp.hstack(mats, format="csr") if len(mats) > 1 else mats[0]

        y = {s: data[s][col].astype(str).to_numpy() for s in ("train", "val", "test", "test_strict")}
        X_tr, X_val = _X("train"), _X("val")
        t0 = time.perf_counter()
        best_cfg, grid = tune_on_val(X_tr, y["train"], X_val, y["val"])
        tuning[f"{name}/{task}"] = {"grid": grid, "best": best_cfg, "tuning_wall_s": round(time.perf_counter() - t0, 1)}
        print(f"[baseline] {name}/{task}: grid {[(g['C'], g['class_weight'], g['val_f1_macro']) for g in grid]}")

        # Clean, sequential refit of the chosen configuration (measured).
        # peak_mem = train feature matrix bytes + tracemalloc peak of the fit (tracemalloc,
        # not RSS: pages freed by the grid fits are reused, so an RSS delta reads ~0).
        del X_val
        feat_bytes = X_tr.data.nbytes + X_tr.indices.nbytes + X_tr.indptr.nbytes
        tracemalloc.start()
        clf, fit_time = _fit_lr(X_tr, y["train"], best_cfg["C"], best_cfg["class_weight"])
        peak_mb = (tracemalloc.get_traced_memory()[1] + feat_bytes) / 2**20
        tracemalloc.stop()
        del X_tr
        vec = vecs[kinds[0]] if len(kinds) == 1 else FeatureUnion([(k, vecs[k]) for k in kinds])
        model = Pipeline([("tfidf", vec), ("clf", clf)])
        n_params = int(clf.coef_.size + clf.intercept_.size)
        nnz = int(np.count_nonzero(clf.coef_))
        size = save_model(model, name, task, {"task": task, "description": desc, "classes": list(clf.classes_),
                                              "best_config": best_cfg})
        cost = {"n_params": n_params, "size_mb": round(size, 2),
                "train_time_s": round(vec_time + fit_time, 1),
                "latency_ms_per_1000": round(measure_latency(model, data["test"]["title"].tolist()), 1),
                "latency_device": "cpu", "peak_mem_mb": round(peak_mb, 1), "n_train": int(len(data["train"]))}
        hp = {"C": best_cfg["C"], "alpha": 1.0 / (best_cfg["C"] * len(data["train"])),
              "class_weight": best_cfg["class_weight"], "features": kinds, "optimizer": "SGD",
              "n_features": int(clf.coef_.shape[1]), "epochs_run": int(clf.n_iter_),
              "converged": bool(clf.n_iter_ < LR_KW["max_iter"]), "c_on_grid_edge": best_cfg["C"] == max(C_GRID),
              **LR_KW}
        notes = (f"{desc}; logistic regression (log-loss, L2, OvR) fitted with SGD. "
                 f"n_params = coef_.size + intercept_.size ({nnz:,} non-zero coef). "
                 f"train_time = vectoriser fit ({vec_time:.1f}s, shared by both tasks) + LR fit ({fit_time:.1f}s). "
                 "peak_mem = train feature matrix bytes + tracemalloc peak of the LR fit "
                 "(vectoriser fitting excluded). Metrics: src.metrics (macro over classes present in y_true). "
                 "Latency: 1000 raw titles, vectoriser + predict_proba, single process on CPU.")
        splits = {}
        for s in ("val", "test", "test_strict"):
            splits[s] = (data[s]["title"].tolist(), y[s], data[s]["law_id"].tolist())
        # evaluation uses the full pipeline on raw titles (exactly what predict.py does)
        results.setdefault(task, {})[name] = evaluate_and_log(name, "tfidf", task, model, splits, cost, hp, notes)
        results[task][name]["cost"] = cost
        print(f"[baseline] {name}/{task}: {results[task][name]['test']} strict={results[task][name]['test_strict']['f1_macro']}")


def featurize(kind: str, ds: pd.DataFrame) -> tuple[dict, TfidfVectorizer]:
    """Fit a vectoriser on train titles, transform train / val / test once."""
    vec = make_vectorizer(kind)
    t0 = time.perf_counter()
    out = {"train": vec.fit_transform(ds.loc[ds["split"] == "train", "title"].tolist())}
    out["fit_time_s"] = time.perf_counter() - t0
    for split in ("val", "test"):
        out[split] = vec.transform(ds.loc[ds["split"] == split, "title"].tolist())
    if hasattr(vec, "stop_words_"):
        vec.stop_words_ = None          # huge set of pruned terms, not needed for inference
    print(f"[baseline] features {kind}: train {out['train'].shape}, nnz={out['train'].nnz:,}, "
          f"fit {out['fit_time_s']:.1f}s")
    return out, vec


def run(models: list[str] | None = None) -> dict:
    """Train / evaluate all baselines for both tasks; returns summary dict."""
    t_start = time.perf_counter()
    models = models or ["majority_dummy", "tfidf_word_lr", "tfidf_char_lr", "tfidf_word_char_lr", "tfidf_stem_lr"]
    ds = load_dataset(["law_id", "title", "split", "form_label", "area_label"])
    ds["split"] = ds["split"].astype(str)
    for col in config.TASKS.values():
        ds[col] = ds[col].astype(object)
    # position of each row inside its split's feature matrix
    ds["pos"] = ds.groupby("split").cumcount()
    strict_ids = load_strict_ids()
    text_prep.get_lexicon()            # stemmer lexicon (built by src.text_prep)
    results: dict = {}
    tuning: dict = {}
    timings: dict = {}

    with threadpool_limits(limits=1):   # BLAS single-threaded; parallelism comes from the grid threads
        if "majority_dummy" in models:
            run_dummy(ds, strict_ids, results)
        feats: dict = {}
        vecs: dict = {}
        order = [("word", ["tfidf_word_lr"]), ("char", ["tfidf_char_lr", "tfidf_word_char_lr"]),
                 ("stem", ["tfidf_stem_lr"])]
        for kind, names in order:
            names = [n for n in names if n in models]
            if not names:
                continue
            t0 = time.perf_counter()
            feats[kind], vecs[kind] = featurize(kind, ds)
            for name in names:
                if any(k not in feats for k in MODELS[name][0]):
                    continue
                t1 = time.perf_counter()
                run_tfidf_model(name, feats, vecs, ds, strict_ids, results, tuning)
                timings[name] = round(time.perf_counter() - t1, 1)
            if kind == "char":                     # biggest matrices: free as soon as possible
                feats.pop("char")
            timings[f"features_{kind}"] = round(time.perf_counter() - t0, 1)

    best = {}
    for task, res in results.items():
        name = max((n for n in res if n != "majority_dummy"), key=lambda n: res[n]["val"]["f1_macro"], default=None)
        if name:
            best[task] = {"model_name": name, "val_f1_macro": res[name]["val"]["f1_macro"],
                          "test_f1_macro": res[name]["test"]["f1_macro"],
                          "test_strict_f1_macro": res[name]["test_strict"]["f1_macro"]}
    summary = {"results": results, "best_by_val": best, "tuning": tuning, "timings_s": timings,
               "elapsed_s": round(time.perf_counter() - t_start, 1)}
    try:
        SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    except OSError as exc:
        print(f"[baseline] could not write summary: {exc}")
    text_prep.attach_downstream_f1()
    print(f"[baseline] best by val: {best}; elapsed {summary['elapsed_s']} s")
    return {"best_by_val": best, "elapsed_s": summary["elapsed_s"]}


if __name__ == "__main__":
    # Re-import under the package name so pickled classes (e.g. _Reshape) are stored
    # as `src.baseline._Reshape`, not `__main__._Reshape` (loadable from predict.py).
    from src import baseline as _module
    _module.run()
