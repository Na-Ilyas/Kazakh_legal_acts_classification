"""Preprocessing ablation: does homoglyph folding / digit masking change F1?

Word 1-2-gram TF-IDF + logistic regression (exactly the tfidf_word_lr recipe of
src.baseline, with its val-selected C / class_weight per task kept FIXED so
only the preprocessing changes) under four tokenisations:
  fold_num     NFC + lower + Latin->Cyrillic homoglyph folding + digits -> 'num'  (= text_prep.tokenize, default)
  nofold_num   no homoglyph folding (Latin 'i' inside Kazakh words kept)
  fold_digits  folding, digit runs kept as tokens (years / act numbers visible)
  raw          no folding, digits kept (plain lower-case regex tokens)
The transformer side of the ablation (homoglyph folding before e5 tokenisation)
is run by `python -m src.vocab_prune --tasks form --fold` (model e5_small_kkprune_fold_ft).
Logged to SQLite as 'ablation_tfidf_word_<variant>' (family 'tfidf'); summary
results/prep_ablation.json.
"""
from __future__ import annotations

import json
import re
import time
import unicodedata

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src import config, text_prep
from src.baseline import _fit_lr, _task_data, evaluate_and_log, load_strict_ids
from src.data_io import load_dataset
from src.metrics import macro_f1
from sklearn.pipeline import Pipeline

SUMMARY_JSON = config.RESULTS_DIR / "prep_ablation.json"
_TOKEN_RE = re.compile(rf"[{text_prep.KAZ_LETTERS}a-z]+|\d+")


def tok_fold_num(text: str) -> list[str]:
    """Default project tokenisation (text_prep.tokenize)."""
    return text_prep.tokenize(text)


def tok_nofold_num(text: str) -> list[str]:
    """No homoglyph folding; digits -> num."""
    low = unicodedata.normalize("NFC", text).lower() if isinstance(text, str) else ""
    return ["num" if t.isdigit() else t for t in _TOKEN_RE.findall(low)]


def tok_fold_digits(text: str) -> list[str]:
    """Homoglyph folding; digits kept."""
    return _TOKEN_RE.findall(text_prep.normalize(text))


def tok_raw(text: str) -> list[str]:
    """No folding, digits kept."""
    low = unicodedata.normalize("NFC", text).lower() if isinstance(text, str) else ""
    return _TOKEN_RE.findall(low)


VARIANTS = {"fold_num": tok_fold_num, "nofold_num": tok_nofold_num, "fold_digits": tok_fold_digits, "raw": tok_raw}


def selected_config(task: str) -> dict:
    """C / class_weight chosen on val for tfidf_word_lr by src.baseline (kept fixed here)."""
    try:
        summary = json.loads((config.RESULTS_DIR / "baseline_summary.json").read_text(encoding="utf-8"))
        return summary["tuning"][f"tfidf_word_lr/{task}"]["best"]
    except (OSError, KeyError, json.JSONDecodeError):
        return {"C": 16.0, "class_weight": None}


def run() -> dict:
    """Fit every variant for both tasks, log val/test/test_strict, save the summary."""
    t0 = time.perf_counter()
    ds = load_dataset(["law_id", "title", "split", "form_label", "area_label"])
    ds["split"] = ds["split"].astype(str)
    for col in config.TASKS.values():
        ds[col] = ds[col].astype(object)
    ds["pos"] = ds.groupby("split").cumcount()
    strict_ids = load_strict_ids()
    out: dict = {}
    for name, tok in VARIANTS.items():
        vec = TfidfVectorizer(tokenizer=tok, token_pattern=None, lowercase=False, ngram_range=(1, 2), min_df=2,
                              max_features=400_000, sublinear_tf=True, dtype=np.float32)
        X_tr = vec.fit_transform(ds.loc[ds["split"] == "train", "title"].tolist())
        X_val = vec.transform(ds.loc[ds["split"] == "val", "title"].tolist())
        if hasattr(vec, "stop_words_"):
            vec.stop_words_ = None
        out[name] = {"vocab": len(vec.vocabulary_)}
        for task, col in config.TASKS.items():
            data = _task_data(ds, task, strict_ids)
            cfg = selected_config(task)
            y_tr = data["train"][col].astype(str).to_numpy()
            clf, fit_s = _fit_lr(X_tr[data["train"]["pos"].to_numpy()], y_tr, cfg["C"], cfg["class_weight"])
            val_f1 = macro_f1(data["val"][col].astype(str), clf.predict(X_val[data["val"]["pos"].to_numpy()]))
            model = Pipeline([("tfidf", vec), ("clf", clf)])
            splits = {s: (data[s]["title"].tolist(), data[s][col].astype(str).to_numpy(), data[s]["law_id"].tolist())
                      for s in ("val", "test", "test_strict")}
            res = evaluate_and_log(f"ablation_tfidf_word_{name}", "tfidf", task, model, splits,
                                   {"n_params": int(clf.coef_.size + clf.intercept_.size), "train_time_s": round(fit_s, 1),
                                    "n_train": len(y_tr), "latency_device": "cpu"},
                                   {"variant": name, **cfg, "optimizer": "SGD"},
                                   f"preprocessing ablation of tfidf_word_lr: tokenisation '{name}' "
                                   "(C / class_weight fixed to the tfidf_word_lr val selection)")
            out[name][task] = {s: res[s]["f1_macro"] for s in res}
            print(f"[prep_ablation] {name}/{task}: {out[name][task]} (val check {val_f1:.4f})")
    out["elapsed_s"] = round(time.perf_counter() - t0, 1)
    SUMMARY_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    print(run())
