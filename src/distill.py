"""Improving the tiny transformer: knowledge distillation and recency weighting.

1. Knowledge distillation (KD): the fine-tuned e5-small (118M params, teacher)
   is distilled into the 4-layer d=256 tiny transformer (5.2M params, student).
   Loss = alpha * T^2 * KL(softmax(teacher/T) || softmax(student/T)) + (1-alpha) * CE(gold).
   Teacher signal = e5 log-probabilities on TRAIN titles only (cached by
   src.finetune_e5 in artifacts/cache/e5_small_ft_<task>/probs_train.npy).
   alpha, T and the initialisation (random vs MLM-pretrained encoder) are chosen
   on validation macro-F1.
2. Recency weighting (temporal drift): train rows are weighted by
   exp((year - 2021) / tau), normalised to mean 1, so recent acts count more;
   tau is chosen on validation macro-F1.

Every variant is a real run logged to SQLite (val / test / test_strict + test
predictions) with an explicit model_name, so the selection is auditable.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src import config  # noqa: F401  (pins GPU 0 before torch)
import torch

from src import bpe_tokenizer
from src.mlm_pretrain import MLM_DIR
from src.model_utils import (BPE_DIR, label_maps, load_probs, load_task, max_len_for, quick_name, save_json)
from src.tiny_transformer import (MAIN_ARCH, arch_name, encode_splits, finalize, train_variant)
from src.trainer import TrainConfig, get_device

KD_GRID = [(0.5, 2.0), (0.9, 2.0), (0.9, 4.0)]                 # (alpha, temperature)
TAU_GRID = (3.0, 8.0)                                           # recency weighting, in years
SUMMARY_CSV = config.RESULTS_DIR / "tiny_transformer_improvements.csv"


def chosen_loss(task: str) -> str:
    """Loss selected on val for the from-scratch main model (results/tiny_transformer_loss_search.csv)."""
    try:
        search = pd.read_csv(config.RESULTS_DIR / "tiny_transformer_loss_search.csv")
        part = search[(search["task"] == task) & (search["arch"] == arch_name(MAIN_ARCH))]
        return str(part.sort_values("val_f1_macro", ascending=False).iloc[0]["loss"])
    except (FileNotFoundError, IndexError, KeyError):
        return "ce"


def teacher_log_probs(task: str, train_ids: list[str], classes: list[str], device,
                      quick: bool = False) -> torch.Tensor:
    """Teacher log-probabilities for the TRAIN rows, aligned to the student's rows/classes."""
    cached = load_probs(quick_name(f"e5_small_ft_{task}", quick), "train")
    if cached is None:
        raise FileNotFoundError(f"teacher probabilities missing - run src.finetune_e5 for task {task}")
    probs, ids, t_classes = cached
    order = pd.Series(range(len(ids)), index=ids).reindex(train_ids)
    if order.isna().any():
        raise ValueError("teacher rows do not cover the training set")
    col = [t_classes.index(c) for c in classes]
    probs = probs[order.to_numpy().astype(int)][:, col]
    logp = np.log(np.clip(probs, 1e-6, 1.0))
    return torch.from_numpy(logp.astype(np.float16)).to(device)


def recency_weights(years: np.ndarray, tau: float, device) -> torch.Tensor:
    """exp((year - last train year) / tau), normalised to mean 1."""
    w = np.exp((years.astype(np.float64) - config.TRAIN_END_YEAR) / tau)
    w = w / w.mean()
    return torch.from_numpy(w.astype(np.float32)).to(device)


def _run_variant(task: str, name_suffix: str, data: dict, tensors: dict, classes: list[str],
                 cfg: TrainConfig, max_len: int, vocab: int, quick: bool, extra_hp: dict, notes: str,
                 init_from=None, teacher=None, weights=None, arch: dict | None = None,
                 label: str | None = None, tok_dir=BPE_DIR) -> dict:
    """Train one variant, benchmark + log it. Returns the summary row.

    Model name = tiny_transformer_<label><name_suffix>, label defaults to the main arch ('4L').
    """
    import shutil

    arch = arch or MAIN_ARCH
    label = label or arch_name(MAIN_ARCH)
    dir_name = quick_name(f"tiny_transformer_{task}_{label}{name_suffix}", quick)
    out_dir = config.MODELS_DIR / dir_name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    res = train_variant(task, data, tensors, classes, arch, cfg, out_dir, max_len, vocab,
                        quick_name(f"tiny_{task}{name_suffix}", quick), tok_dir=tok_dir, init_from=init_from,
                        teacher_logp=teacher, sample_weight=weights)
    model_name = quick_name(f"tiny_transformer_{label}{name_suffix}", quick)
    try:                                   # persist the variant settings (refits need e.g. recency_tau)
        from src.model_utils import load_json
        cfg_json = load_json(out_dir / "config.json")
        save_json({**cfg_json, "extra": extra_hp}, out_dir / "config.json")
    except (FileNotFoundError, ValueError):
        pass
    f1s = finalize(task, model_name, out_dir, res, len(data["train"]), quick, extra_hp)
    _set_notes(model_name, task, notes)
    row = {"task": task, "model_name": model_name, "val_f1": round(res["best_val_f1"], 4), **f1s,
           "best_epoch": res["best_epoch"], "train_time_s": res["train_time_s"], **extra_hp}
    print(f"[improve] {row}", flush=True)
    return row


def _set_notes(model_name: str, task: str, notes: str) -> None:
    """Replace the generic notes written by finalize() with a precise description."""
    from src import experiments_db
    from src.model_utils import with_db_retry

    def upd():
        with experiments_db.connect() as con:
            con.execute("UPDATE experiments SET notes=? WHERE model_name=? AND task=?", (notes, model_name, task))
    with_db_retry(upd)


def run_task(task: str, quick: bool = False, do_kd: bool = True, do_recency: bool = True) -> list[dict]:
    """KD grid (+ MLM init for the best KD setting) and recency-weighting grid for one task."""
    device = get_device()
    tok = bpe_tokenizer.load_tokenizer(BPE_DIR)
    max_len = max_len_for("bpe", 96)
    # quick: same seeded subsample as the e5 quick run, so the teacher covers every train row
    data = load_task(task, max_train=8_000 if quick else None, max_eval=2_000 if quick else None)
    classes, label2id = label_maps(data["train"]["label"])
    tensors = encode_splits(data, tok, max_len, label2id, device)
    loss = chosen_loss(task)
    base = {"epochs": 2 if quick else 12, "patience": 3, "loss": loss}
    rows: list[dict] = []
    vocab = tok.get_vocab_size()

    if do_kd:
        teacher = teacher_log_probs(task, data["train"]["law_id"].tolist(), classes, device, quick)
        kd_rows = []
        for alpha, temp in (KD_GRID[:1] if quick else KD_GRID):
            cfg = TrainConfig(**base, kd_alpha=alpha, kd_temperature=temp)
            sfx = f"_kd_a{int(alpha * 100)}_T{int(temp)}"
            kd_rows.append(_run_variant(
                task, sfx, data, tensors, classes, cfg, max_len, vocab, quick,
                {"kd_alpha": alpha, "kd_temperature": temp, "teacher": "e5_small_ft", "init": "random",
                 "loss_selected_on_val": loss},
                f"KD student: e5_small_ft teacher (train-title logits), alpha={alpha}, T={temp}, random init",
                teacher=teacher))
        best = max(kd_rows, key=lambda r: r["val_f1"])
        rows += kd_rows
        enc = MLM_DIR / "encoder.pt"
        if enc.exists() and not quick:      # best (alpha, T) again, from the MLM-pretrained encoder
            alpha, temp = best["kd_alpha"], best["kd_temperature"]
            cfg = TrainConfig(**{**base, "lr": 3e-4}, kd_alpha=alpha, kd_temperature=temp)
            rows.append(_run_variant(
                task, f"_mlm_kd_a{int(alpha * 100)}_T{int(temp)}", data, tensors, classes, cfg, max_len,
                vocab, quick, {"kd_alpha": alpha, "kd_temperature": temp, "teacher": "e5_small_ft",
                               "init": "MLM-pretrained encoder", "lr": 3e-4, "loss_selected_on_val": loss},
                f"KD student from MLM-pretrained encoder: e5_small_ft teacher, alpha={alpha}, T={temp}",
                init_from=enc, teacher=teacher))
        del teacher

    if do_recency:
        years = data["train"]["year"].to_numpy()
        for tau in (TAU_GRID[:1] if quick else TAU_GRID):
            cfg = TrainConfig(**base)
            rows.append(_run_variant(
                task, f"_recency_tau{int(tau)}", data, tensors, classes, cfg, max_len, vocab, quick,
                {"recency_tau": tau, "init": "random", "loss_selected_on_val": loss},
                f"recency-weighted training: weight=exp((year-2021)/{tau}), mean 1",
                weights=recency_weights(years, tau, device)))
    del tensors
    torch.cuda.empty_cache()
    return rows


def _hp_from_db(task: str, model_name: str, key: str):
    """One hyperparameter of a logged model (val row), or None."""
    import json
    from src import experiments_db
    with experiments_db.connect() as con:
        row = con.execute("SELECT hyperparams FROM experiments WHERE task=? AND model_name=? AND "
                          "eval_split='val'", (task, model_name)).fetchone()
    try:
        return json.loads(row[0] or "{}").get(key) if row else None
    except ValueError:
        return None


def best_tiny_on_val(task: str) -> str | None:
    """Best tiny-transformer model on VAL macro-F1 (refits excluded)."""
    from src import experiments_db
    with experiments_db.connect() as con:
        df = pd.read_sql_query("SELECT model_name, f1_macro FROM experiments WHERE task=? AND "
                               "eval_split='val' AND model_family='tiny_transformer'", con, params=(task,))
    df = df[~df["model_name"].str.endswith(("_trainval", "_quick"))]
    return None if df.empty else df.sort_values("f1_macro", ascending=False).iloc[0]["model_name"]


def refit_trainval(task: str, model_name: str | None = None) -> dict:
    """Retrain the val-selected tiny model on train+val with the SAME config and epoch count.

    The val row of the refit is not a held-out score (val is in its training data).
    """
    import shutil
    from src.model_utils import load_json

    model_name = model_name or best_tiny_on_val(task)
    src_dir = config.MODELS_DIR / model_name.replace("tiny_transformer_", f"tiny_transformer_{task}_")
    cfg_json = load_json(src_dir / "config.json")
    arch = {k: cfg_json["model"][k] for k in ("n_layers", "d_model", "n_heads", "dropout")}
    train_cfg = {k: v for k, v in cfg_json["train"].items() if k in TrainConfig.__dataclass_fields__}
    epochs = int(cfg_json["best_epoch"])
    cfg = TrainConfig(**{**train_cfg, "stop_after": epochs})
    device = get_device()
    tok_dir = bpe_tokenizer.tokenizer_dir(int(cfg_json["model"]["vocab_size"]))   # 8k or the 4k ablation
    tok = bpe_tokenizer.load_tokenizer(tok_dir)
    max_len = int(cfg_json["max_len"])
    data = load_task(task)
    data["train"] = pd.concat([data["train"], data["val"]], ignore_index=True)
    classes, label2id = label_maps(data["train"]["label"])
    tensors = encode_splits(data, tok, max_len, label2id, device)
    teacher = None
    if cfg.kd_alpha > 0:                     # teacher log-probs for train AND val rows
        tr = load_probs(f"e5_small_ft_{task}", "train")
        va = load_probs(f"e5_small_ft_{task}", "val")
        probs = np.vstack([tr[0], va[0]])
        ids = tr[1] + va[1]
        order = pd.Series(range(len(ids)), index=ids).reindex(data["train"]["law_id"].astype(str))
        col = [tr[2].index(c) for c in classes]
        logp = np.log(np.clip(probs[order.to_numpy().astype(int)][:, col], 1e-6, 1.0))
        teacher = torch.from_numpy(logp.astype(np.float16)).to(device)
    init = MLM_DIR / "encoder.pt" if "_mlm" in model_name else None
    tau = (cfg_json.get("extra") or {}).get("recency_tau") or _hp_from_db(task, model_name, "recency_tau")
    weights = recency_weights(data["train"]["year"].to_numpy(), float(tau), device) if tau else None
    label = model_name[len("tiny_transformer_"):]
    return _run_variant(task, "_trainval", data, tensors, classes, cfg, max_len, tok.get_vocab_size(), False,
                        {"refit_of": model_name, "epochs_fixed": epochs, "trainval": True,
                         "kd_alpha": cfg.kd_alpha, "kd_temperature": cfg.kd_temperature, "recency_tau": tau},
                        f"REFIT of {model_name} on train+val for {epochs} epochs (same config"
                        f"{f', recency weights tau={tau}' if tau else ''}; val row is not held-out)",
                        init_from=init, teacher=teacher, weights=weights, arch=arch, label=label,
                        tok_dir=tok_dir)


def run_refits(tasks: tuple[str, ...] = ("form", "area"), quick: bool = False) -> dict:
    """Train+val refit of the best tiny transformer (on val) for every task."""
    if quick:
        return {"skipped": "quick"}
    return {t: refit_trainval(t) for t in tasks}


def run(tasks: tuple[str, ...] = ("form", "area"), quick: bool = False, kd: bool = True,
        recency: bool = True) -> dict:
    """Run the improvement grids for every task and save a summary CSV (selection on val)."""
    t0 = time.time()
    rows = []
    for task in tasks:
        rows += run_task(task, quick=quick, do_kd=kd, do_recency=recency)
    df = pd.DataFrame(rows)
    if len(df) and not quick:
        df.to_csv(SUMMARY_CSV, index=False)
    best = {t: g.sort_values("val_f1", ascending=False).iloc[0]["model_name"]
            for t, g in df.groupby("task")} if len(df) else {}
    save_json({"best_on_val": best, "time_s": round(time.time() - t0, 1)},
              config.RESULTS_DIR / ("tiny_transformer_improvements_quick.json" if quick else
                                    "tiny_transformer_improvements.json"))
    return {"n_runs": len(rows), "best_on_val": best}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="KD (e5 -> tiny) and recency weighting")
    parser.add_argument("--tasks", default="form,area")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-kd", action="store_true")
    parser.add_argument("--no-recency", action="store_true")
    parser.add_argument("--refit", action="store_true", help="train+val refit of the best tiny model on val")
    args = parser.parse_args()
    if args.refit:
        print(run_refits(tuple(args.tasks.split(","))))
        raise SystemExit(0)
    print(run(tuple(args.tasks.split(",")), quick=args.quick, kd=not args.no_kd, recency=not args.no_recency))
