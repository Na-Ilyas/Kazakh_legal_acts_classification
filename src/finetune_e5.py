"""Pretrained compact model: intfloat/multilingual-e5-small (118M params).

Two experiments per task:
* e5_frozen_lr : FROZEN e5 mean-pooled, L2-normalised sentence embeddings +
  scikit-learn LogisticRegression (representation baseline; together with
  TF-IDF it answers "sparse n-grams vs dense pretrained embeddings").
* e5_small_ft  : full fine-tuning, masked mean pooling + linear head, AdamW,
  warmup + cosine, bf16, best epoch on validation macro-F1.

Inputs follow the e5 convention "query: <title>"; max_len comes from the p99
token length (results/token_lengths.json).
"""
from __future__ import annotations

import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src import config  # noqa: F401  (pins GPU 0 before torch)
import torch
import torch.nn as nn

from src.metrics import macro_f1
from src.model_utils import (E5_MODEL_NAME, display_path, save_probs, E5_PREFIX, encode_labels, label_maps, load_task,
                             log_all_splits, max_len_for, peak_rss_mb, prediction_frame, quick_name,
                             save_json)
from src.trainer import (TokenData, TrainConfig, fit, get_device, plot_learning_curve, predict_proba,
                         set_seed)

FT_NAME, FROZEN_NAME = "e5_small_ft", "e5_frozen_lr"
LR_C = 1.0          # LogisticRegression C on standardised embeddings


class E5Classifier(nn.Module):
    """e5 encoder -> masked mean pooling -> dropout -> linear head."""

    def __init__(self, encoder: nn.Module, n_classes: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(encoder.config.hidden_size, n_classes)

    def pool(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Masked mean of the last hidden states."""
        hidden = self.encoder(input_ids=ids, attention_mask=mask.long()).last_hidden_state
        m = mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * m).sum(1) / m.sum(1).clamp_min(1.0)

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.head(self.dropout(self.pool(ids, mask)))


def load_e5():
    """(tokenizer, encoder) of the pretrained checkpoint (HF cache)."""
    from transformers import AutoModel, AutoTokenizer
    return AutoTokenizer.from_pretrained(E5_MODEL_NAME), AutoModel.from_pretrained(E5_MODEL_NAME)


def encode_e5(hf_tok, titles: list[str], max_len: int, chunk: int = 20_000) -> tuple[np.ndarray, np.ndarray]:
    """'query: ' + title -> int32 [N, max_len] padded ids + lengths (compact, chunked)."""
    ids = np.full((len(titles), max_len), hf_tok.pad_token_id, dtype=np.int32)
    lengths = np.zeros(len(titles), dtype=np.int16)
    for start in range(0, len(titles), chunk):
        batch = [E5_PREFIX + t for t in titles[start:start + chunk]]
        enc = hf_tok(batch, truncation=True, max_length=max_len)["input_ids"]
        for i, row in enumerate(enc, start=start):
            ids[i, :len(row)] = row
            lengths[i] = len(row)
    return ids, lengths


def prepare(task: str, quick: bool, merge_val: bool = False) -> tuple[dict, list[str], dict, int]:
    """Load the task data and tokenise every split once (tensors on the GPU).

    merge_val=True: train on train+val (final refit after all choices were made on val).
    """
    data = load_task(task, max_train=8_000 if quick else None, max_eval=2_000 if quick else None)
    if merge_val:
        data["train"] = pd.concat([data["train"], data["val"]], ignore_index=True)
    classes, label2id = label_maps(data["train"]["label"])
    max_len = max_len_for("e5", 112)
    hf_tok, _ = load_e5_tokenizer_only()
    device = get_device()
    tensors = {}
    for split, frame in data.items():
        ids, lengths = encode_e5(hf_tok, frame["title"].tolist(), max_len)
        tensors[split] = TokenData.build(ids, lengths, encode_labels(frame["label"], label2id), device)
    return {"data": data, "tensors": tensors, "tokenizer": hf_tok}, classes, label2id, max_len


def load_e5_tokenizer_only():
    """Tokenizer only (cheap) - the encoder is loaded where needed."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(E5_MODEL_NAME), None


def _benchmark_and_log(kind: str, model_name: str, family: str, task: str, out_dir: Path,
                       preds: dict, n_params: int, train_time: float, peak_mem: float | None,
                       n_train: int, hp: dict, quick: bool, notes: str) -> dict:
    """Shared tail: subprocess benchmark + SQLite logging of val/test/test_strict."""
    from src import benchmark

    bench = benchmark.measure_subprocess(kind, str(out_dir), quick=quick)
    common = {"n_params": bench.get("n_params", n_params), "size_mb": bench.get("size_mb"),
              "latency_ms_per_1000": bench.get("latency_cpu_ms_per_1000"), "latency_device": "cpu",
              "peak_mem_mb": peak_mem, "train_time_s": round(train_time, 2), "n_train": n_train}
    hp = {**hp, "latency_cuda_ms_per_1000": bench.get("latency_cuda_ms_per_1000"),
          "infer_peak_gpu_mb": bench.get("infer_peak_gpu_mb"), "infer_ram_mb": bench.get("infer_ram_mb"),
          "model_dir": display_path(out_dir)}
    logged = log_all_splits(model_name, family, task, preds, common, hp, notes=notes)
    return {split: round(v["f1_macro"], 4) for split, v in logged.items()}


# --------------------------------------------------------------------------
# Frozen embeddings + LogisticRegression
# --------------------------------------------------------------------------
@torch.no_grad()
def embed(encoder: nn.Module, data: TokenData, batch_size: int = 512) -> np.ndarray:
    """L2-normalised masked-mean e5 embeddings (float32 [N, 384])."""
    clf = E5Classifier(encoder, n_classes=1)
    clf.head = nn.Identity()          # forward() now returns dropout(pooled) = pooled in eval mode
    emb = predict_like(clf, data, batch_size)
    return emb / np.linalg.norm(emb, axis=1, keepdims=True).clip(1e-12)


@torch.no_grad()
def predict_like(model: nn.Module, data: TokenData, batch_size: int) -> np.ndarray:
    """Raw model outputs for all rows (length-sorted batches, bf16 on CUDA)."""
    from src.trainer import make_batch

    device = next(model.parameters()).device
    model.eval()
    order = np.argsort(data.lengths, kind="stable")
    chunks = []
    for start in range(0, len(order), batch_size):
        ids, mask = make_batch(data, order[start:start + batch_size], device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            chunks.append(model(ids, mask).float().cpu().numpy())
    out = np.concatenate(chunks)
    result = np.empty_like(out)
    result[order] = out
    return result


def frozen_lr_task(task: str, quick: bool = False) -> dict:
    """Frozen e5 embeddings + LogisticRegression (class_weight chosen on val macro-F1)."""
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits

    prep, classes, label2id, max_len = prepare(task, quick)
    data, tensors = prep["data"], prep["tensors"]
    device = get_device()
    _, encoder = load_e5()
    encoder = encoder.to(device).eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    emb = {split: embed(encoder, tensors[split]) for split in ("train", "val", "test")}
    embed_time = time.time() - t0
    peak_gpu = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None
    del encoder
    torch.cuda.empty_cache()
    y_train = data["train"]["label"].to_numpy()

    search, best = [], None
    t1 = time.time()
    with threadpool_limits(8):                       # shared machine: <= 8 BLAS threads
        for class_weight in (None, "balanced"):
            ts = time.time()
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(C=LR_C, max_iter=1000, class_weight=class_weight))
            clf.fit(emb["train"], y_train)
            val_f1 = macro_f1(data["val"]["label"], clf.predict(emb["val"]))
            search.append({"class_weight": str(class_weight), "C": LR_C, "val_f1_macro": round(float(val_f1), 4),
                           "fit_time_s": round(time.time() - ts, 1), "n_iter": int(np.max(clf[-1].n_iter_))})
            print(f"[e5_frozen_lr] {task} class_weight={class_weight}: val F1 {val_f1:.4f} "
                  f"({time.time() - ts:.0f}s, {int(np.max(clf[-1].n_iter_))} iters)")
            if best is None or val_f1 > best[1]:
                best = (clf, val_f1, class_weight)
    lr_time = time.time() - t1
    clf, _, cw = best
    out_dir = config.MODELS_DIR / quick_name(f"{FROZEN_NAME}_{task}", quick)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, out_dir / "lr.joblib")
    save_json({"task": task, "model_name": quick_name(FROZEN_NAME, quick), "base_model": E5_MODEL_NAME,
               "max_len": max_len, "pooling": "mean, L2-normalised", "class_weight": str(cw), "C": LR_C,
               "search": search}, out_dir / "config.json")
    preds = {split: prediction_frame(data[split]["law_id"], data[split]["label"],
                                     clf.predict_proba(emb[split]), [str(c) for c in clf.classes_])
             for split in ("val", "test")}
    hp = {"base_model": E5_MODEL_NAME, "max_len": max_len, "prefix": E5_PREFIX, "C": LR_C, "scaler": "StandardScaler",
          "class_weight": str(cw), "max_iter": 1000, "search": search,
          "embed_time_s": round(embed_time, 1), "lr_fit_time_s": round(lr_time, 1),
          "train_peak_gpu_mb": None if peak_gpu is None else round(peak_gpu, 1),
          "train_peak_rss_mb": round(peak_rss_mb(), 1),
          "peak_mem_mb_meaning": "CUDA peak while embedding (frozen encoder)"}
    del emb, clf, best                       # free RAM before the benchmark subprocess starts
    gc.collect()
    f1s = _benchmark_and_log("e5_frozen_lr", quick_name(FROZEN_NAME, quick), "pretrained_frozen", task,
                             out_dir, preds, 0, embed_time + lr_time, peak_gpu, len(data["train"]), hp,
                             quick, "frozen multilingual-e5-small embeddings + LogisticRegression")
    print(f"[e5_frozen_lr] {task}: {f1s}")
    return {"class_weight": str(cw), **f1s, "train_time_s": round(embed_time + lr_time, 1)}


# --------------------------------------------------------------------------
# Fine-tuning
# --------------------------------------------------------------------------
def finetune_task(task: str, quick: bool = False, lr: float = 5e-5, epochs: int = 3,
                  batch_size: int = 64, patience: int | None = None, suffix: str = "",
                  trainval: bool = False, stop_after: int = 0) -> dict:
    """Fine-tune e5-small end-to-end; best epoch (val macro-F1) saved in HF format + head.

    suffix: model-name suffix for variants (e.g. '_5ep').
    trainval: refit on train+val for exactly `stop_after` epochs (choices made on val
              beforehand); its val row is NOT a held-out score.
    """
    prep, classes, label2id, max_len = prepare(task, quick, merge_val=trainval)
    data, tensors, hf_tok = prep["data"], prep["tensors"], prep["tokenizer"]
    device = get_device()
    _, encoder = load_e5()
    set_seed(config.SEED)                       # reproducible head init
    model = E5Classifier(encoder, len(classes)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[e5_ft] task={task}: {n_params:,} parameters, {len(classes)} classes, "
          f"train={len(data['train']):,}, max_len={max_len}, device={device}")
    cfg = TrainConfig(epochs=1 if quick else epochs, batch_size=batch_size, lr=lr, weight_decay=0.01,
                      warmup_ratio=0.06, loss="ce", patience=patience or epochs, eval_batch_size=512,
                      bucket=True, stop_after=stop_after)
    name = quick_name(FT_NAME + suffix, quick)
    out_dir = config.MODELS_DIR / quick_name(f"{FT_NAME}_{task}{suffix}", quick)
    run_name = out_dir.name
    res = fit(model, tensors["train"], tensors["val"], len(classes), cfg, out_dir / "model.pt", run_name,
              log_csv=config.RESULTS_DIR / f"train_log_{run_name}.csv")
    probs = {split: predict_proba(model, tensors[split], 512) for split in ("train", "val", "test")}
    preds = {split: prediction_frame(data[split]["law_id"], data[split]["label"], probs[split], classes)
             for split in ("val", "test")}
    for split in ("train", "val", "test"):    # train probs = teacher signal for src.distill
        save_probs(out_dir.name, split, probs[split], data[split]["law_id"], classes)
    del probs
    # Persist in a portable format: HF encoder (safetensors) + tokenizer + linear head.
    model.encoder.save_pretrained(out_dir / "encoder")
    hf_tok.save_pretrained(out_dir / "encoder")
    torch.save(model.head.state_dict(), out_dir / "head.pt")
    (out_dir / "model.pt").unlink(missing_ok=True)
    save_json({"task": task, "model_name": name, "base_model": E5_MODEL_NAME, "trainval": trainval,
               "max_len": max_len, "prefix": E5_PREFIX, "pooling": "masked mean", "train": res["train_config"],
               "best_epoch": res["best_epoch"], "best_val_f1": res["best_val_f1"]}, out_dir / "config.json")
    save_json(label2id, out_dir / "label2id.json")
    plot_learning_curve(res["history"], f"multilingual-e5-small fine-tuning - task '{task}'",
                        config.FIGURES_DIR / f"learning_curve_{run_name}.png")
    del model, encoder
    torch.cuda.empty_cache()
    hp = {"base_model": E5_MODEL_NAME, "max_len": max_len, "prefix": E5_PREFIX, **res["train_config"],
          "best_epoch": res["best_epoch"], "epochs_run": res["epochs_run"],
          "train_peak_gpu_mb": res["train_peak_gpu_mb"], "peak_mem_mb_meaning": "training peak CUDA memory",
          "trainval": trainval}
    notes = "fine-tuned multilingual-e5-small, mean pooling + linear head"
    if trainval:
        notes += f"; REFIT on train+val for {stop_after} epochs (val row is not held-out)"
    elif suffix:
        notes += f"; variant {suffix}: up to {epochs} epochs, early stopping patience {patience or epochs}"
    f1s = _benchmark_and_log("e5_ft", name, "pretrained", task, out_dir, preds,
                             n_params, res["train_time_s"], res["train_peak_gpu_mb"], len(data["train"]),
                             hp, quick, notes)
    print(f"[e5_ft] {task}: {f1s}")
    return {**f1s, "best_epoch": res["best_epoch"], "train_time_s": res["train_time_s"],
            "train_peak_gpu_mb": res["train_peak_gpu_mb"]}


def run(tasks: tuple[str, ...] = ("form", "area"), quick: bool = False, frozen: bool = True,
        finetune: bool = True) -> dict:
    """Frozen-embedding baseline and fine-tuning for every task."""
    results: dict = {}
    for task in tasks:
        results[task] = {}
        if frozen:
            results[task][FROZEN_NAME] = frozen_lr_task(task, quick)
        if finetune:
            results[task][FT_NAME] = finetune_task(task, quick)
    return results


def run_5ep(tasks: tuple[str, ...] = ("form", "area"), quick: bool = False) -> dict:
    """e5 upgrade: up to 5 epochs with early stopping (patience 2) on val macro-F1."""
    if quick:
        return {"skipped": "quick"}
    return {t: finetune_task(t, epochs=5, patience=2, suffix="_5ep") for t in tasks}


def refit_trainval(tasks: tuple[str, ...] = ("form", "area"), quick: bool = False) -> dict:
    """Refit the e5 variant that is best on VAL on train+val with the same epoch count."""
    from src import experiments_db
    from src.model_utils import load_json

    if quick:
        return {"skipped": "quick"}
    out = {}
    for t in tasks:
        with experiments_db.connect() as con:            # e5 variant with the best VAL macro-F1
            rows = con.execute("SELECT model_name, f1_macro FROM experiments WHERE task=? AND "
                               "eval_split='val' AND model_name IN (?, ?)",
                               (t, FT_NAME, FT_NAME + "_5ep")).fetchall()
        chosen = max(rows, key=lambda r: r[1])[0] if rows else FT_NAME
        best = load_json(config.MODELS_DIR / f"{FT_NAME}_{t}{chosen[len(FT_NAME):]}" / "config.json")
        print(f"[e5 refit] {t}: refitting {chosen} (best on val), {best.get('best_epoch')} epochs")
        out[t] = finetune_task(t, epochs=int(best["train"]["epochs"]), suffix="_trainval", trainval=True,
                               stop_after=int(best.get("best_epoch") or 3))
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="e5-small: frozen+LR and fine-tuning")
    parser.add_argument("--tasks", default="form,area")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-frozen", action="store_true")
    parser.add_argument("--no-finetune", action="store_true")
    parser.add_argument("--variant", choices=["5ep", "trainval"], default=None,
                        help="5ep: up to 5 epochs with patience 2; trainval: refit on train+val")
    args = parser.parse_args()
    if args.variant == "5ep":
        print(run_5ep(tuple(args.tasks.split(","))))
    elif args.variant == "trainval":
        print(refit_trainval(tuple(args.tasks.split(","))))
    else:
        print(run(tuple(args.tasks.split(",")), quick=args.quick, frozen=not args.no_frozen,
                  finetune=not args.no_finetune))
