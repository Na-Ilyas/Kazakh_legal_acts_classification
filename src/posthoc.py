"""Post-hoc improvements on top of trained models (no retraining).

1. Prior-shift (label-shift) adjustment. The class prior moves strongly over time
   (form: Шешім 36% of train vs 64% of test), so we re-weight the predicted
   probabilities:  p_adj(c|x) ∝ p(c|x) * (p_target(c) / p_train(c)) ** lam.
   Validation (2022) is split deterministically (hash of law_id) into two halves:
   half A estimates p_target from its LABEL distribution, half B selects lam on
   macro-F1. The val rows of ALL posthoc models are reported on half B only, so they are
   comparable with each other (not with the full-val rows of single models).
2. Ensemble: weighted average of the probabilities of TF-IDF word+char LR, the
   best tiny transformer and the fine-tuned e5 (weights on a 0.1 simplex grid,
   chosen on full validation), optionally followed by the prior adjustment.

Models are written as small configs under artifacts/models/posthoc_<name>_<task>/
(members + weights + prior ratio), so src.predict and src.benchmark load them like
any other model; latency/size/params are those of all members together.
"""
from __future__ import annotations

import hashlib
import itertools
import json

import numpy as np
import pandas as pd

from src import config, experiments_db
from src.metrics import macro_f1
from src.model_utils import (compute_metrics, display_path, label_maps, load_probs, load_task,
                             log_all_splits, save_json, save_probs)

LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)
PRIOR_PSEUDO_COUNT = 100        # shrink the half-A label distribution toward the train prior
RATIO_CLIP = (0.5, 2.0)         # never down-/up-weight a class by more than 2x (rare classes!)
MIN_GAIN = 0.002                # keep an adjusted variant only if it gains this much on val half B
PARAM_BUDGET = 120_000_000      # course limit for the compact pretrained model
TEMPERATURES = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0)   # calibration of each ensemble member (val NLL)
WEIGHT_STEP = 0.1
FAMILY = "posthoc"


# --------------------------------------------------------------------------
# Probabilities of the member models, aligned to load_task() rows and classes
# --------------------------------------------------------------------------
def member_specs(task: str) -> dict[str, dict]:
    """TF-IDF best (data side), best tiny transformer and e5 fine-tuned, chosen on VAL F1."""
    with experiments_db.connect() as con:
        val = pd.read_sql_query("SELECT model_name, model_family, f1_macro FROM experiments "
                                "WHERE task=? AND eval_split='val'", con, params=(task,))
    val = val[~val["model_name"].str.endswith(("_trainval", "_quick"))]    # refits saw val
    out = {}
    for fam, key in (("tfidf", "tfidf"), ("tiny_transformer", "tiny"), ("pretrained", "e5")):
        part = val[val["model_family"] == fam]
        if fam == "pretrained":                       # fine-tuned e5 variants only (best on val)
            part = part[part["model_name"].str.startswith("e5_small_ft")]
        if len(part):
            name = part.sort_values("f1_macro", ascending=False).iloc[0]["model_name"]
            out[key] = {"name": name, "family": fam}
    # Kazakh RoBERTa variants of the research module (loadable through their meta.json "loader")
    rob = val[val["model_name"].str.startswith("kaz_roberta")].sort_values("f1_macro", ascending=False)
    for name in rob["model_name"]:
        d = config.MODELS_DIR / f"{name}_{task}"
        try:
            if "loader" in json.loads((d / "meta.json").read_text(encoding="utf-8")):
                out["roberta"] = {"name": name, "family": "pretrained", "kind": "external", "path": str(d)}
                break
        except (OSError, ValueError):
            continue
    for key, spec in out.items():
        name = spec["name"]
        if key == "roberta":
            continue
        if key == "tfidf":
            spec.update(kind="baseline", path=name)
        elif key == "tiny":
            spec.update(kind="tiny", path=str(config.MODELS_DIR / name.replace("tiny_transformer_",
                                                                               f"tiny_transformer_{task}_")))
        else:
            suffix = name[len("e5_small_ft"):]
            spec.update(kind="e5_ft", path=str(config.MODELS_DIR / f"e5_small_ft_{task}{suffix}"))
    return out


def aligned_probs(spec: dict, task: str, split: str, frame: pd.DataFrame, classes: list[str]) -> np.ndarray:
    """[n_rows, n_classes] probabilities of one member for `frame` rows in `classes` order."""
    from pathlib import Path
    if spec["kind"] == "baseline":
        cache_name = f"{spec['name']}_{task}"
        cached = load_probs(cache_name, split)
        if cached is None:                         # TF-IDF is cheap: predict on CPU and cache
            from src.inference import BaselinePredictor
            pred = BaselinePredictor(spec["name"], task)
            probs = pred.predict_proba(frame["title"].tolist(), batch_size=4096)
            save_probs(cache_name, split, probs, frame["law_id"], pred.classes)
            cached = load_probs(cache_name, split)
    else:
        cached = load_probs(Path(spec["path"]).name, split)
    if cached is None:
        raise FileNotFoundError(f"no cached probabilities for {spec['name']} ({split})")
    probs, ids, m_classes = cached
    order = pd.Series(range(len(ids)), index=ids).reindex(frame["law_id"].astype(str))
    if order.isna().any():
        raise ValueError(f"{spec['name']}: cached rows do not match the {split} split")
    out = np.zeros((len(frame), len(classes)), dtype=np.float32)
    for j, c in enumerate(classes):
        if c in m_classes:
            out[:, j] = probs[order.to_numpy().astype(int), m_classes.index(c)]
    return out / out.sum(axis=1, keepdims=True).clip(1e-12)


# --------------------------------------------------------------------------
# Selection helpers
# --------------------------------------------------------------------------
def val_halves(law_ids: pd.Series) -> np.ndarray:
    """Deterministic 50/50 split of val rows by md5(law_id): True = half A (prior estimate)."""
    return np.array([int(hashlib.md5(str(i).encode()).hexdigest(), 16) % 2 == 0 for i in law_ids])


def class_freq(labels, classes: list[str]) -> np.ndarray:
    """Label distribution with add-one smoothing (no zero priors)."""
    counts = pd.Series(list(map(str, labels))).value_counts()
    freq = np.array([counts.get(c, 0) + 1.0 for c in classes])
    return freq / freq.sum()


def prior_ratio(y_a, y_train, classes: list[str]) -> np.ndarray:
    """p_target / p_train with p_target = half-A label counts shrunk toward the train prior
    (pseudo-count PRIOR_PSEUDO_COUNT), clipped to RATIO_CLIP. A class with no half-A rows keeps
    ratio ~1 instead of being suppressed (bug M2 of the review: ratio 0.029 zeroed a class)."""
    p_train = class_freq(y_train, classes)
    counts = pd.Series(list(map(str, y_a))).value_counts()
    n_a = np.array([counts.get(c, 0) for c in classes], dtype=float)
    p_target = (n_a + PRIOR_PSEUDO_COUNT * p_train) / (n_a.sum() + PRIOR_PSEUDO_COUNT)
    return np.clip(p_target / p_train, *RATIO_CLIP)


def adjust(probs: np.ndarray, ratio: np.ndarray, lam: float) -> np.ndarray:
    """p * ratio**lam, renormalised (lam = 0 -> unchanged)."""
    q = probs * ratio[None, :] ** lam
    return q / q.sum(axis=1, keepdims=True)


def pick_lambda(probs_b: np.ndarray, y_b, ratio: np.ndarray, classes: list[str]) -> tuple[float, list[dict]]:
    """lam maximising macro-F1 on val half B."""
    rows = []
    for lam in LAMBDAS:
        pred = np.array(classes)[adjust(probs_b, ratio, lam).argmax(1)]
        rows.append({"lam": lam, "val_b_f1": round(macro_f1(y_b, pred), 4)})
    best = max(rows, key=lambda r: (r["val_b_f1"], -r["lam"]))     # ties -> smaller adjustment
    return best["lam"], rows


def temper(probs: np.ndarray, temp: float) -> np.ndarray:
    """Temperature scaling on probabilities: softmax(log p / T)."""
    logp = np.log(np.clip(probs, 1e-9, 1.0)) / temp
    logp -= logp.max(axis=1, keepdims=True)
    q = np.exp(logp)
    return q / q.sum(axis=1, keepdims=True)


def fit_temperature(probs: np.ndarray, y: np.ndarray, classes: list[str]) -> float:
    """Temperature minimising the negative log-likelihood on validation (calibration only)."""
    idx = np.array([classes.index(c) for c in y])
    nll = {t: -np.log(np.clip(temper(probs, t)[np.arange(len(idx)), idx], 1e-9, 1.0)).mean()
           for t in TEMPERATURES}
    return min(nll, key=nll.get)


def weight_grid(k: int, step: float = WEIGHT_STEP):
    """All weight vectors on a simplex grid with every member weight > 0."""
    n = int(round(1 / step))
    for combo in itertools.product(range(1, n), repeat=k):
        if sum(combo) == n:
            yield np.array(combo, dtype=np.float64) / n


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
def member_costs(task: str, names: list[str]) -> dict:
    """Sum of members' params / size / CPU latency / train time (from their test rows)."""
    with experiments_db.connect() as con:
        df = pd.read_sql_query("SELECT model_name, n_params, size_mb, latency_ms_per_1000, train_time_s, "
                               "hyperparams FROM experiments WHERE task=? AND eval_split='test'", con,
                               params=(task,))
    df = df[df["model_name"].isin(names)]
    cuda = 0.0
    for h in df["hyperparams"]:
        try:
            cuda += float(json.loads(h or "{}").get("latency_cuda_ms_per_1000") or 0)
        except (ValueError, TypeError):
            pass
    return {"n_params": int(df["n_params"].fillna(0).sum()), "size_mb": float(df["size_mb"].fillna(0).sum()),
            "latency_ms_per_1000": float(df["latency_ms_per_1000"].fillna(0).sum()), "latency_device": "cpu",
            "train_time_s": float(df["train_time_s"].fillna(0).sum()), "latency_cuda_sum": cuda}


def save_and_log(task: str, model_name: str, members: list[dict], weights: np.ndarray, ratio: np.ndarray,
                 lam: float, classes: list[str], frames: dict, probs: dict, val_mask: np.ndarray | None,
                 hp: dict, notes: str, n_train: int) -> dict:
    """Write the posthoc config, cache probabilities and log val/test/test_strict rows."""
    out_dir = config.MODELS_DIR / f"posthoc_{model_name}_{task}"
    save_json({"task": task, "model_name": model_name, "kind": "posthoc", "classes": classes,
               "members": [{k: m[k] for k in ("kind", "name", "path")} for m in members],
               "weights": [float(w) for w in weights], "prior_ratio": [float(r) for r in ratio],
               "lam": float(lam), **{k: v for k, v in hp.items() if k != "search"}}, out_dir / "config.json")
    preds = {}
    for split in ("val", "test"):
        frame, p = frames[split], probs[split]
        save_probs(out_dir.name, split, p, frame["law_id"], classes)
        key = split
        if split == "val":
            # Only val half B is (partly) held out, and it is in-sample for lam / weights:
            # logged as eval_split='val_half_b', never compared with full-val scores of single models.
            frame, p, key = frame[val_mask], p[val_mask], "val_half_b"
        pred_idx = p.argmax(1)
        preds[key] = pd.DataFrame({"law_id": frame["law_id"].astype(str).to_numpy(),
                                     "y_true": frame["label"].astype(str).to_numpy(),
                                     "y_pred": np.array(classes)[pred_idx], "confidence": p.max(1)})
    costs = member_costs(task, [m["name"] for m in members])
    hp = {**hp, "members": [m["name"] for m in members], "weights": [round(float(w), 2) for w in weights],
          "lam": lam, "latency_cuda_ms_per_1000": costs.pop("latency_cuda_sum"),
          "model_dir": display_path(out_dir)}
    common = {**costs, "peak_mem_mb": None, "n_train": n_train}
    if costs["n_params"] > PARAM_BUDGET:
        notes = f"OUTSIDE the 120M parameter budget ({costs['n_params'] / 1e6:.0f}M) - reference only; " + notes
        hp["outside_param_budget"] = True
    logged = log_all_splits(model_name, FAMILY, task, preds, common, hp, notes=notes)
    return {s: round(v["f1_macro"], 4) for s, v in logged.items()}


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def purge_previous(task: str) -> None:
    """Remove earlier posthoc outputs of this task (rows, configs, caches): the family is fully
    regenerated on every run, so a changed member set can never leave stale variants behind."""
    import shutil
    from src.model_utils import prob_cache_dir, with_db_retry

    def delete_rows():
        with experiments_db.connect() as con:
            con.execute("DELETE FROM experiments WHERE model_family=? AND task=?", (FAMILY, task))
    with_db_retry(delete_rows)
    for d in config.MODELS_DIR.glob(f"posthoc_*_{task}"):
        shutil.rmtree(d, ignore_errors=True)
        shutil.rmtree(prob_cache_dir(d.name), ignore_errors=True)


def run_task(task: str) -> list[dict]:
    """Prior adjustment for each member + ensemble (+ adjustment) for one task."""
    purge_previous(task)
    data = load_task(task)
    classes, _ = label_maps(data["train"]["label"])
    frames = {s: data[s] for s in ("val", "test")}
    specs = member_specs(task)
    probs = {}
    for k, spec in list(specs.items()):
        try:
            probs[k] = {s: aligned_probs(spec, task, s, frames[s], classes) for s in ("val", "test")}
        except (FileNotFoundError, ValueError) as exc:
            if k != "roberta":
                raise
            print(f"[posthoc] {task}: kaz-roberta member skipped ({exc}); run --cache-external first")
            specs.pop(k)
    y_val = frames["val"]["label"].astype(str).to_numpy()
    half_a = val_halves(frames["val"]["law_id"])
    ratio = prior_ratio(y_val[half_a], data["train"]["label"], classes)
    rows = []

    def f1_b(p: np.ndarray) -> float:
        """macro-F1 on val half B of a probability matrix over full val."""
        return round(macro_f1(y_val[~half_a], np.array(classes)[p[~half_a].argmax(1)]), 4)

    def try_adjust(name: str, members: list[dict], weights: np.ndarray, p: dict, base_hp: dict, what: str):
        lam, search = pick_lambda(p["val"][~half_a], y_val[~half_a], ratio, classes)
        base_b = f1_b(p["val"])
        best_b = max(r["val_b_f1"] for r in search)
        print(f"[posthoc] {task} {name}: lam={lam} (val-B F1 {base_b:.4f} -> {best_b:.4f})", flush=True)
        if lam == 0.0 or best_b - base_b < MIN_GAIN:
            rows.append({"task": task, "model_name": f"{name}_prioradj", "selected_lam": lam,
                         "note": f"val-B gain {best_b - base_b:+.4f} < {MIN_GAIN} -> not logged"})
            return
        adj = {s: adjust(p[s], ratio, lam) for s in ("val", "test")}
        f1s = save_and_log(task, f"{name}_prioradj", members, weights, ratio, lam, classes, frames, adj,
                           ~half_a, {**base_hp, "prior_search": search, "prior_estimated_on": "val half A",
                                     "lam_selected_on": "val half B", "base_val_half_b_f1": base_b,
                                     "prior_pseudo_count": PRIOR_PSEUDO_COUNT, "ratio_clip": list(RATIO_CLIP)},
                           f"{what} + prior-shift adjustment p*(p_val/p_train)^lam (p_val from val half A, "
                           f"shrunk to the train prior, ratio clipped to {list(RATIO_CLIP)}); lam={lam} chosen "
                           f"on val half B (val_half_b row is in-sample for lam; base model there: {base_b})",
                           len(data["train"]))
        rows.append({"task": task, "model_name": f"{name}_prioradj", "selected_lam": lam, **f1s})

    # 1) prior adjustment of each single model
    for key, spec in specs.items():
        try_adjust(spec["name"], [spec], np.array([1.0]), probs[key], {"base_model": spec["name"]},
                   f"{spec['name']}")
    # 2) ensemble of the three families: members calibrated by temperature (val NLL), then the
    #    mixing weights are gridded on full val macro-F1
    member_sets = [[k for k in ("tfidf", "tiny", "e5") if k in specs]]
    if "roberta" in specs:
        member_sets.append([k for k in ("tfidf", "tiny", "e5", "roberta") if k in specs])
    for keys in member_sets:
        if len(keys) < 2:
            continue
        temps = {k: fit_temperature(probs[k]["val"], y_val, classes) for k in keys}
        cal = {k: {s: temper(probs[k][s], temps[k]) for s in ("val", "test")} for k in keys}
        print(f"[posthoc] {task} member temperatures (val NLL): {temps}", flush=True)
        best = None
        for w in weight_grid(len(keys)):
            p = sum(wi * cal[k]["val"] for wi, k in zip(w, keys))
            f1 = macro_f1(y_val, np.array(classes)[p.argmax(1)])
            if best is None or f1 > best[0] + 1e-12:
                best = (f1, w)
        f1_val, w = best
        members = [specs[k] for k in keys]
        ens = {s: sum(wi * cal[k][s] for wi, k in zip(w, keys)) for s in ("val", "test")}
        name = "ensemble_" + "_".join(keys)
        hp = {"weights_selected_on": "full val", "weight_step": WEIGHT_STEP,
              "members_val_half_b_f1": {specs[k]["name"]: f1_b(probs[k]["val"]) for k in keys},
              "member_temperatures": {specs[k]["name"]: temps[k] for k in keys},
              "temperatures_selected_on": "val NLL"}
        print(f"[posthoc] {task} {name}: weights {dict(zip(keys, [round(float(x), 2) for x in w]))} "
              f"val F1 {f1_val:.4f}", flush=True)
        f1s = save_and_log(task, name, members, w, ratio, 0.0, classes, frames, ens, ~half_a, hp,
                           f"weighted average of member probabilities ({', '.join(m['name'] for m in members)}), "
                           f"weights {[round(float(x), 2) for x in w]} chosen on full val (val_half_b row is in-sample for the weights; not comparable with single-model val)",
                           len(data["train"]))
        rows.append({"task": task, "model_name": name, "weights": [round(float(x), 2) for x in w], **f1s})
        try_adjust(name, members, w, ens, hp, "ensemble")
    return rows


def cache_external_probs(tasks: tuple[str, ...] = ("form", "area")) -> dict:
    """One GPU inference pass: cache val/test probabilities of the kaz-roberta member
    (run inside the shared GPU lock: flock .gpu.lock python -m src.posthoc --cache-external)."""
    import torch
    from src.inference import ModelSpec, load_predictor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    done = {}
    for task in tasks:
        spec = member_specs(task).get("roberta")
        if spec is None:
            continue
        if all(load_probs(Path(spec["path"]).name, s) is not None for s in ("val", "test")):
            done[task] = f"{spec['name']} (cache already present)"
            continue
        data = load_task(task)
        pred = load_predictor(ModelSpec("external", spec["name"], task, spec["path"]), device)
        for split in ("val", "test"):
            frame = data[split]
            probs = pred.predict_proba(frame["title"].tolist(), batch_size=256)
            save_probs(Path(spec["path"]).name, split, probs, frame["law_id"], pred.classes)
        done[task] = spec["name"]
        del pred
        torch.cuda.empty_cache()
    return done


def run(tasks: tuple[str, ...] = ("form", "area"), quick: bool = False) -> dict:
    """Prior-shift adjustment and ensembles for every task; summary CSV in results/."""
    if quick:
        print("[posthoc] skipped in quick mode (needs the full-run members)")
        return {"skipped": True}
    rows = []
    for task in tasks:
        rows += run_task(task)
    pd.DataFrame(rows).to_csv(config.RESULTS_DIR / "posthoc_summary.csv", index=False)
    return {"n_variants": len(rows)}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prior-shift adjustment and ensembles")
    parser.add_argument("--tasks", default="form,area")
    parser.add_argument("--cache-external", action="store_true",
                        help="GPU pass caching kaz-roberta probabilities (run under the GPU lock)")
    args = parser.parse_args()
    if args.cache_external:
        print(cache_external_probs(tuple(args.tasks.split(","))))
    else:
        print(run(tuple(args.tasks.split(","))))
