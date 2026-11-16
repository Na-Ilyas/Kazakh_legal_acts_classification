"""Generic GPU training loop for token-id classifiers (tiny transformer, e5).

Any model with signature `model(input_ids: LongTensor[B,T], mask: BoolTensor[B,T]) -> logits`
can be trained here. Features: AdamW (no decay on norms/biases), linear warmup +
cosine decay, bf16 autocast on CUDA, gradient clipping, CE loss with optional
sqrt-inverse-frequency class weights or label smoothing, early stopping on
validation macro-F1 with the best checkpoint saved to disk, per-epoch CSV log
and TensorBoard scalars.

The whole (int16/int32) token matrix lives on the GPU; each batch is sliced to
the longest sequence it contains, so padding costs almost nothing.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src import config  # noqa: F401  (sets CUDA_VISIBLE_DEVICES before torch)
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

from src.model_utils import RUNS_DIR, class_weights


@dataclass
class TrainConfig:
    """Optimisation hyper-parameters (all logged to SQLite as JSON)."""
    epochs: int = 12
    batch_size: int = 256
    lr: float = 7e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    grad_clip: float = 1.0
    loss: str = "ce"                 # 'ce' | 'weighted' (sqrt-inv freq) | 'smooth' (label smoothing)
    label_smoothing: float = 0.1
    patience: int = 3                # early stopping on val macro-F1
    eval_batch_size: int = 1024
    seed: int = config.SEED
    bf16: bool = True
    bucket: bool = False             # length-bucketed batches (less padding, faster)


@dataclass
class TokenData:
    """Padded token ids (+ lengths, + label ids) of one split, kept on `device`."""
    ids: torch.Tensor        # [N, T] int16/int32
    lengths: np.ndarray      # [N] (CPU, used to trim batches)
    y: np.ndarray | None     # [N] int64 label ids

    @classmethod
    def build(cls, ids: np.ndarray, lengths: np.ndarray, y: np.ndarray | None, device) -> "TokenData":
        return cls(torch.from_numpy(ids).to(device), np.asarray(lengths).astype(np.int64), y)

    def __len__(self) -> int:
        return len(self.lengths)


def get_device() -> torch.device:
    """GPU 0 if available (CUDA_VISIBLE_DEVICES is pinned to 0 by src.config)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    """Seed python/numpy/torch for reproducible runs."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_batch(data: TokenData, idx: np.ndarray, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Slice rows `idx`, trim to the longest one, return (LongTensor ids, BoolTensor mask)."""
    max_len = int(data.lengths[idx].max())
    idx_t = torch.from_numpy(idx).to(device, non_blocking=True)
    ids = data.ids.index_select(0, idx_t)[:, :max_len].long()
    lens = torch.from_numpy(data.lengths[idx]).to(device, non_blocking=True)
    mask = torch.arange(max_len, device=device)[None, :] < lens[:, None]
    return ids, mask


def epoch_batches(lengths: np.ndarray, batch_size: int, rng: np.random.Generator,
                  bucket: bool) -> list[np.ndarray]:
    """Shuffled batches of row indices.

    bucket=True: shuffle, cut into chunks of 50 batches, sort each chunk by
    length, split into batches and shuffle the batch order - batches contain
    titles of similar length, so far less padding is computed.
    """
    perm = rng.permutation(len(lengths))
    if not bucket:
        return [perm[i:i + batch_size] for i in range(0, len(perm), batch_size)]
    batches = []
    chunk = batch_size * 50
    for start in range(0, len(perm), chunk):
        part = perm[start:start + chunk]
        part = part[np.argsort(lengths[part], kind="stable")]
        batches.extend(part[i:i + batch_size] for i in range(0, len(part), batch_size))
    order = rng.permutation(len(batches))
    return [batches[i] for i in order]


def param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """AdamW groups: no weight decay for biases, norms and embeddings' 1-D params."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 or "norm" in name.lower() else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def warmup_cosine(total_steps: int, warmup_steps: int):
    """LR multiplier: linear warmup to 1, then cosine decay to 0."""
    def fn(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return fn


@torch.no_grad()
def predict_proba(model: nn.Module, data: TokenData, batch_size: int = 1024, bf16: bool = True) -> np.ndarray:
    """Softmax probabilities for every row; rows are length-sorted to minimise padding."""
    device = next(model.parameters()).device
    model.eval()
    order = np.argsort(data.lengths, kind="stable")
    out = np.zeros((len(data), 0), dtype=np.float32)
    chunks = []
    use_amp = bf16 and device.type == "cuda"
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        ids, mask = make_batch(data, idx, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            logits = model(ids, mask)
        chunks.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
    probs = np.concatenate(chunks) if chunks else out
    result = np.empty_like(probs)
    result[order] = probs
    return result


def build_criterion(cfg: TrainConfig, y_train: np.ndarray, n_classes: int, device) -> nn.Module:
    """Cross-entropy, optionally class-weighted or label-smoothed."""
    weight = None
    if cfg.loss == "weighted":
        weight = torch.tensor(class_weights(y_train, n_classes, "sqrt_inv"), device=device)
    smoothing = cfg.label_smoothing if cfg.loss == "smooth" else 0.0
    if cfg.loss not in ("ce", "weighted", "smooth"):
        raise ValueError(f"unknown loss {cfg.loss!r}")
    return nn.CrossEntropyLoss(weight=weight, label_smoothing=smoothing)


def fit(model: nn.Module, train: TokenData, val: TokenData, n_classes: int, cfg: TrainConfig,
        ckpt_path: Path, run_name: str, log_csv: Path | None = None) -> dict:
    """Train with early stopping on val macro-F1; best weights are saved to `ckpt_path`.

    Returns a summary with the history DataFrame, best epoch/F1, train time and
    peak CUDA memory. The model is left loaded with the BEST weights.
    """
    set_seed(cfg.seed)
    device = next(model.parameters()).device
    criterion = build_criterion(cfg, train.y, n_classes, device)
    use_amp = cfg.bf16 and device.type == "cuda"
    opt = torch.optim.AdamW(param_groups(model, cfg.weight_decay), lr=cfg.lr, betas=(0.9, 0.98),
                            eps=1e-6, fused=device.type == "cuda")
    steps_per_epoch = math.ceil(len(train) / cfg.batch_size)
    total = steps_per_epoch * cfg.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, warmup_cosine(total, int(cfg.warmup_ratio * total)))
    y_train_t = torch.from_numpy(train.y).to(device)
    writer = _tensorboard(run_name)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    history, best_f1, best_epoch, bad, step = [], -1.0, 0, 0, 0
    rng = np.random.default_rng(cfg.seed)
    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        loss_sum = torch.zeros((), device=device)
        te = time.time()
        for idx in epoch_batches(train.lengths, cfg.batch_size, rng, cfg.bucket):
            ids, mask = make_batch(train, idx, device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                logits = model(ids, mask)
            loss = criterion(logits.float(), y_train_t[torch.from_numpy(idx).to(device)])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            sched.step()
            loss_sum += loss.detach()
            step += 1
            if writer is not None and step % 50 == 0:
                writer.add_scalar("train/loss_step", loss.item(), step)
        train_loss = float(loss_sum.item() / steps_per_epoch)

        probs = predict_proba(model, val, cfg.eval_batch_size, cfg.bf16)
        val_pred = probs.argmax(1)
        val_loss = float(F.nll_loss(torch.log(torch.from_numpy(probs).clamp_min(1e-9)),
                                    torch.from_numpy(np.where(val.y < 0, 0, val.y))).item())
        val_f1 = float(f1_score(val.y, val_pred, average="macro", zero_division=0))
        val_acc = float((val_pred == val.y).mean())
        row = {"epoch": epoch, "train_loss": round(train_loss, 5), "val_loss": round(val_loss, 5),
               "val_f1_macro": round(val_f1, 5), "val_accuracy": round(val_acc, 5),
               "lr_end": opt.param_groups[0]["lr"], "epoch_time_s": round(time.time() - te, 2)}
        history.append(row)
        if writer is not None:
            for k in ("train_loss", "val_loss", "val_f1_macro", "val_accuracy"):
                writer.add_scalar(f"epoch/{k}", row[k], epoch)
        improved = val_f1 > best_f1
        print(f"[{run_name}] epoch {epoch:2d} loss {train_loss:.4f} val_loss {val_loss:.4f} "
              f"val_f1 {val_f1:.4f} acc {val_acc:.4f} ({row['epoch_time_s']}s){' *' if improved else ''}",
              flush=True)
        if improved:
            best_f1, best_epoch, bad = val_f1, epoch, 0
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)
        else:
            bad += 1
            if bad >= cfg.patience:
                print(f"[{run_name}] early stopping (no val-F1 gain for {cfg.patience} epochs)")
                break
    train_time = time.time() - t0
    peak_gpu = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None
    if writer is not None:
        writer.close()
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    hist = pd.DataFrame(history)
    if log_csv is not None:
        log_csv.parent.mkdir(parents=True, exist_ok=True)
        hist.to_csv(log_csv, index=False)
    return {"history": hist, "best_epoch": best_epoch, "best_val_f1": best_f1,
            "epochs_run": len(history), "train_time_s": round(train_time, 2),
            "train_peak_gpu_mb": None if peak_gpu is None else round(peak_gpu, 1),
            "train_config": asdict(cfg)}


def _tensorboard(run_name: str):
    """SummaryWriter under runs/<run_name>, or None if tensorboard is unavailable."""
    try:
        from torch.utils.tensorboard import SummaryWriter
        return SummaryWriter(log_dir=str(RUNS_DIR / run_name))
    except Exception as exc:  # tensorboard is an optional extra
        print(f"[trainer] TensorBoard disabled: {exc}")
        return None


def plot_learning_curve(hist: pd.DataFrame, title: str, path: Path) -> None:
    """Two-panel figure: train/val loss and val macro-F1 per epoch."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(hist["epoch"], hist["train_loss"], "o-", label="train loss")
    a1.plot(hist["epoch"], hist["val_loss"], "s-", label="val loss (CE)")
    a1.set_xlabel("epoch"); a1.set_ylabel("loss"); a1.set_title("Loss"); a1.legend()
    a2.plot(hist["epoch"], hist["val_f1_macro"], "o-", color="#3a9e5f", label="val macro-F1")
    a2.plot(hist["epoch"], hist["val_accuracy"], "s--", color="#999999", label="val accuracy")
    best = hist.loc[hist["val_f1_macro"].idxmax()]
    a2.axvline(best["epoch"], color="#e07b39", ls=":", label=f"best epoch {int(best['epoch'])}")
    a2.set_xlabel("epoch"); a2.set_ylabel("score"); a2.set_title("Validation (2022)"); a2.legend()
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
