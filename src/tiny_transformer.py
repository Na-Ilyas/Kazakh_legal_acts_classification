"""Tiny transformer encoder trained FROM SCRATCH on our own BPE tokens.

Architecture (all modules written here, no nn.TransformerEncoder):
    token embedding + learned positional embedding -> dropout
    N x [ pre-LN -> own multi-head self-attention -> residual
          pre-LN -> GELU feed-forward            -> residual ]
    final LayerNorm -> masked mean pooling (or [CLS]) -> dropout -> linear head

run() trains both tasks: first the loss variant (plain CE / sqrt-inverse class
weights / label smoothing) is chosen on validation macro-F1 with the main 4-layer
d=256 model, then an ablation (2 vs 4 layers, d_model 256 vs 128) is trained with
the chosen loss. Every final model is benchmarked and logged (val/test/test_strict).
"""
from __future__ import annotations

import math
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src import config  # noqa: F401  (pins GPU 0 before torch is imported)
import torch
import torch.nn as nn
import torch.nn.functional as F

from src import bpe_tokenizer
from src.model_utils import (BPE_DIR, label_maps, encode_labels, load_task, log_all_splits,
                             max_len_for, prediction_frame, quick_name, save_json)
from src.trainer import (TokenData, TrainConfig, fit, get_device, plot_learning_curve, predict_proba,
                         set_seed)

FAMILY = "tiny_transformer"
MAX_PARAMS = 10_000_000
MAIN_ARCH = {"n_layers": 4, "d_model": 256, "n_heads": 4}
# Ablation grid (trained with the loss chosen for the main architecture).
ABLATION = [{"n_layers": 2, "d_model": 256, "n_heads": 4},
            {"n_layers": 4, "d_model": 128, "n_heads": 4},
            {"n_layers": 2, "d_model": 128, "n_heads": 4}]
LOSSES = ("ce", "weighted", "smooth")


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention written out explicitly.

    One fused linear layer produces Q, K and V; they are split into `n_heads`
    heads of size d_model/n_heads, attention softmax(QK^T/sqrt(d_head))V is
    computed per head (padding keys masked out) and the heads are concatenated
    and projected back to d_model.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads, self.d_head = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = dropout

    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                return_weights: bool = False) -> tuple[torch.Tensor, torch.Tensor | None]:
        """x: [B,T,D]; mask: [B,T] bool (True = real token). Returns (out, weights|None)."""
        b, t, d = x.shape
        # [B,T,3D] -> [3,B,H,T,Dh]
        q, k, v = self.qkv(x).view(b, t, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        key_mask = mask[:, None, None, :]            # broadcast over heads and query positions
        if return_weights:                           # explicit path (for inspection / teaching)
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
            scores = scores.masked_fill(~key_mask, float("-inf"))
            weights = scores.softmax(dim=-1)
            ctx = weights @ v
        else:                                        # same maths, fused flash/memory-efficient kernel
            weights = None
            ctx = F.scaled_dot_product_attention(
                q, k, v, attn_mask=key_mask, dropout_p=self.attn_dropout if self.training else 0.0)
        ctx = ctx.transpose(1, 2).reshape(b, t, d)    # concatenate heads
        return self.out_proj(ctx), weights


class EncoderBlock(nn.Module):
    """Pre-LayerNorm transformer block: x + Attn(LN(x)); x + FFN(LN(x))."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(d_ff, d_model))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(self.norm1(x), mask)
        x = x + self.drop(attn_out)
        return x + self.drop(self.ffn(self.norm2(x)))


class TinyTransformerClassifier(nn.Module):
    """Encoder-only transformer + pooling + linear classification head."""

    def __init__(self, vocab_size: int, n_classes: int, max_len: int = 128, d_model: int = 256,
                 n_heads: int = 4, n_layers: int = 4, d_ff: int | None = None, dropout: float = 0.1,
                 pooling: str = "mean", pad_id: int = 0):
        super().__init__()
        if pooling not in ("mean", "cls"):
            raise ValueError("pooling must be 'mean' or 'cls'")
        self.pooling, self.max_len = pooling, max_len
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.emb_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(EncoderBlock(d_model, n_heads, d_ff or 4 * d_model, dropout)
                                    for _ in range(n_layers))
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, n_classes))
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """GPT/BERT-style init: N(0, 0.02) weights, zero biases."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
            if isinstance(module, nn.Embedding) and module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()

    def hidden_states(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Contextual token vectors [B, T, d_model] after the final LayerNorm."""
        t = ids.size(1)
        pos = torch.arange(t, device=ids.device)
        x = self.emb_drop(self.tok_emb(ids) + self.pos_emb(pos)[None, :, :])
        for block in self.blocks:
            x = block(x, mask)
        return self.norm(x)

    def encode(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Pooled sentence vector [B, d_model]."""
        x = self.hidden_states(ids, mask)
        if self.pooling == "cls":
            return x[:, 0]
        m = mask.unsqueeze(-1).to(x.dtype)
        return (x * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(ids, mask))


def count_params(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --------------------------------------------------------------------------
# Training helpers
# --------------------------------------------------------------------------
def arch_name(arch: dict) -> str:
    """'4L' for the main d=256 model, '4L_d128' for other widths, '_do30' for dropout 0.3."""
    suffix = "" if arch["d_model"] == MAIN_ARCH["d_model"] else f"_d{arch['d_model']}"
    if arch.get("dropout", 0.1) != 0.1:
        suffix += f"_do{int(round(arch['dropout'] * 100))}"
    return f"{arch['n_layers']}L{suffix}"


def encode_splits(data: dict[str, pd.DataFrame], tok, max_len: int, label2id: dict, device) -> dict[str, TokenData]:
    """Tokenise every split once into compact int16 tensors on the GPU."""
    out = {}
    for split, frame in data.items():
        ids, lengths = bpe_tokenizer.encode_titles(tok, frame["title"].tolist(), max_len)
        out[split] = TokenData.build(ids, lengths, encode_labels(frame["label"], label2id), device)
    return out


def train_variant(task: str, data: dict, tensors: dict, classes: list[str], arch: dict,
                  cfg: TrainConfig, out_dir: Path, max_len: int, vocab_size: int, run_name: str,
                  tok_dir: Path = BPE_DIR, init_from: Path | None = None) -> dict:
    """Train one architecture/loss variant; best checkpoint + config are saved in out_dir.

    init_from: encoder weights from MLM pretraining (src.mlm_pretrain); the
    classification head is always freshly initialised.
    """
    device = get_device()
    set_seed(cfg.seed)                     # weight init is controlled by the run's seed
    model = TinyTransformerClassifier(vocab_size=vocab_size, n_classes=len(classes), max_len=max_len,
                                      **arch).to(device)
    if init_from is not None:
        state = torch.load(init_from, map_location=device, weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        assert all(k.startswith("head.") for k in missing) and not unexpected, (missing, unexpected)
        print(f"[tiny] {run_name}: encoder initialised from {Path(init_from).name}")
    n_params = count_params(model)
    print(f"[tiny] {run_name}: {n_params:,} parameters on {device}")
    assert n_params <= MAX_PARAMS, f"{n_params} params exceeds the 10M budget"
    res = fit(model, tensors["train"], tensors["val"], len(classes), cfg, out_dir / "model.pt",
              run_name, log_csv=out_dir / "train_log.csv")
    model_cfg = {"vocab_size": vocab_size, "n_classes": len(classes), "max_len": max_len, "dropout": 0.1,
                 "pooling": "mean", "pad_id": bpe_tokenizer.PAD_ID, **arch}
    save_json({"model": model_cfg, "train": res["train_config"], "task": task, "max_len": max_len,
               "n_params": n_params, "best_epoch": res["best_epoch"], "best_val_f1": res["best_val_f1"]},
              out_dir / "config.json")
    save_json({c: i for i, c in enumerate(classes)}, out_dir / "label2id.json")
    shutil.copy(tok_dir / "tokenizer.json", out_dir / "tokenizer.json")
    preds = {split: prediction_frame(data[split]["law_id"], data[split]["label"],
                                     predict_proba(model, tensors[split]), classes)
             for split in ("val", "test")}
    del model
    torch.cuda.empty_cache()
    return {**res, "n_params": n_params, "preds": preds, "model_cfg": model_cfg}


def finalize(task: str, model_name: str, out_dir: Path, res: dict, n_train: int,
             quick: bool, extra_hp: dict) -> dict:
    """Benchmark (subprocess), write curves, log val/test/test_strict to SQLite."""
    from src import benchmark

    log_name = out_dir.name
    hist = res["history"]
    hist.to_csv(config.RESULTS_DIR / f"train_log_{log_name}.csv", index=False)
    plot_learning_curve(hist, f"Tiny transformer {arch_label(res['model_cfg'])} - task '{task}'",
                        config.FIGURES_DIR / f"learning_curve_{log_name}.png")
    bench = benchmark.measure_subprocess("tiny", str(out_dir), quick=quick)
    common = {"n_params": res["n_params"], "size_mb": bench.get("size_mb"),
              "latency_ms_per_1000": bench.get("latency_cpu_ms_per_1000"), "latency_device": "cpu",
              "peak_mem_mb": res["train_peak_gpu_mb"], "train_time_s": res["train_time_s"],
              "n_train": n_train}
    hp = {**res["model_cfg"], **res["train_config"], **extra_hp, "best_epoch": res["best_epoch"],
          "epochs_run": res["epochs_run"], "peak_mem_mb_meaning": "training peak CUDA memory",
          "train_peak_gpu_mb": res["train_peak_gpu_mb"],
          "latency_cuda_ms_per_1000": bench.get("latency_cuda_ms_per_1000"),
          "infer_peak_gpu_mb": bench.get("infer_peak_gpu_mb"),
          "infer_ram_mb": bench.get("infer_ram_mb"), "model_dir": str(out_dir.relative_to(config.PROJECT_ROOT))}
    logged = log_all_splits(model_name, FAMILY, task, res["preds"], common, hp,
                            notes="own BPE + transformer encoder from scratch")
    return {split: round(v["f1_macro"], 4) for split, v in logged.items()}


def arch_label(model_cfg: dict) -> str:
    """Human-readable architecture summary for plot titles."""
    return f"{model_cfg['n_layers']}L d{model_cfg['d_model']} h{model_cfg['n_heads']}"


def train_task(task: str, quick: bool = False, losses: tuple[str, ...] = LOSSES,
               ablation: list[dict] | None = None, main_arch: dict | None = None,
               vocab_size: int = bpe_tokenizer.MAIN_VOCAB, init_from: Path | None = None,
               cfg_overrides: dict | None = None) -> dict:
    """Loss selection on the main architecture, then the ablation, for one task.

    vocab_size != 8000 uses the alternative BPE tokenizer (tokenizer ablation);
    those models get the suffix '_bpe4k' etc.
    """
    main_arch = main_arch or MAIN_ARCH
    device = get_device()
    tok_dir = bpe_tokenizer.tokenizer_dir(vocab_size)
    vsfx = "" if vocab_size == bpe_tokenizer.MAIN_VOCAB else f"_bpe{vocab_size // 1000}k"
    if init_from is not None:
        vsfx += "_mlm"                   # encoder pretrained with masked-language modelling
    tok = bpe_tokenizer.load_tokenizer(tok_dir)
    max_len = max_len_for("bpe" if not vsfx else f"bpe_{vocab_size // 1000}k", 96)
    data = load_task(task, max_train=20_000 if quick else None, max_eval=4_000 if quick else None)
    classes, label2id = label_maps(data["train"]["label"])
    t0 = time.time()
    tensors = encode_splits(data, tok, max_len, label2id, device)
    print(f"[tiny] task={task}: {len(classes)} classes, train={len(data['train']):,} "
          f"val={len(data['val']):,} test={len(data['test']):,}, max_len={max_len}, "
          f"tokenised in {time.time() - t0:.1f}s")
    base_cfg = TrainConfig(**{"epochs": 2 if quick else 12, "patience": 3, **(cfg_overrides or {})})
    search_root = config.MODELS_DIR / "_search_tiny"
    summary: dict = {"task": task, "search": [], "final": {}}

    # 1) choose the loss on validation macro-F1 with the main architecture
    best = None
    for loss in losses:
        cfg = TrainConfig(**{**base_cfg.__dict__, "loss": loss})
        out_dir = search_root / f"{task}_{arch_name(main_arch)}{vsfx}_{loss}"
        res = train_variant(task, data, tensors, classes, main_arch, cfg, out_dir, max_len,
                            tok.get_vocab_size(),
                            quick_name(f"tiny_{task}_{arch_name(main_arch)}{vsfx}_{loss}", quick), tok_dir,
                            init_from)
        summary["search"].append({"task": task, "arch": arch_name(main_arch) + vsfx, "loss": loss,
                                  "best_epoch": res["best_epoch"], "val_f1_macro": round(res["best_val_f1"], 4),
                                  "train_time_s": res["train_time_s"], "n_params": res["n_params"]})
        if best is None or res["best_val_f1"] > best[1]["best_val_f1"]:
            best = (loss, res, out_dir)
    best_loss, best_res, best_dir = best
    print(f"[tiny] task={task}: chosen loss '{best_loss}' (val F1 {best_res['best_val_f1']:.4f})")

    runs = [(main_arch, best_res, best_dir)]
    # 2) ablation with the chosen loss
    for arch in (ablation if ablation is not None else ([] if quick else ABLATION)):
        cfg = TrainConfig(**{**base_cfg.__dict__, "loss": best_loss})
        out_dir = search_root / f"{task}_{arch_name(arch)}{vsfx}_{best_loss}"
        res = train_variant(task, data, tensors, classes, arch, cfg, out_dir, max_len,
                            tok.get_vocab_size(), quick_name(f"tiny_{task}_{arch_name(arch)}{vsfx}", quick),
                            tok_dir, init_from)
        runs.append((arch, res, out_dir))

    for arch, res, src_dir in runs:
        final_dir = config.MODELS_DIR / quick_name(f"tiny_transformer_{task}_{arch_name(arch)}{vsfx}", quick)
        if final_dir.exists():
            shutil.rmtree(final_dir)
        shutil.move(str(src_dir), str(final_dir))
        model_name = quick_name(f"tiny_transformer_{arch_name(arch)}{vsfx}", quick)
        f1s = finalize(task, model_name, final_dir, res, len(data["train"]), quick,
                       {"loss_selected_on_val": best_loss, "loss_candidates": list(losses),
                        "bpe_vocab_size": vocab_size,
                        "init": "MLM-pretrained encoder" if init_from is not None else "random"})
        summary["final"][model_name] = {"val_f1": round(res["best_val_f1"], 4), **f1s,
                                        "n_params": res["n_params"], "train_time_s": res["train_time_s"],
                                        "best_epoch": res["best_epoch"]}
        print(f"[tiny] {task} {model_name}: {summary['final'][model_name]}")
    shutil.rmtree(search_root, ignore_errors=True)   # leftover losing loss variants
    del tensors
    torch.cuda.empty_cache()
    return summary


DROPOUT_ARCHS = [{**MAIN_ARCH, "dropout": 0.3}, {"n_layers": 2, "d_model": 256, "n_heads": 4, "dropout": 0.3}]


def run_dropout_variants(tasks: tuple[str, ...] = ("form", "area")) -> dict:
    """Regularisation ablation: dropout 0.3 instead of 0.1, with the loss already chosen on val."""
    import pandas as pd
    search = pd.read_csv(config.RESULTS_DIR / "tiny_transformer_loss_search.csv")
    out = {}
    for task in tasks:
        part = search[(search["task"] == task) & (search["arch"] == arch_name(MAIN_ARCH))]
        loss = part.sort_values("val_f1_macro", ascending=False).iloc[0]["loss"]
        out[task] = train_task(task, losses=(loss,), ablation=DROPOUT_ARCHS[1:],
                               main_arch=DROPOUT_ARCHS[0])["final"]
    return out


def run_seed_study(tasks: tuple[str, ...] = ("form", "area"), seeds: tuple[int, ...] = (1, 2, 3),
                   archs: list[dict] | None = None) -> pd.DataFrame:
    """Seed variance of the tiny transformer (same data/config, different init + batch order).

    Not logged as separate experiments; writes results/tiny_transformer_seed_study.csv
    so that differences between configurations can be compared with seed noise.
    """
    from src.model_utils import compute_metrics, load_strict_ids

    search = pd.read_csv(config.RESULTS_DIR / "tiny_transformer_loss_search.csv")
    strict = load_strict_ids()
    device = get_device()
    tok = bpe_tokenizer.load_tokenizer(BPE_DIR)
    max_len = max_len_for("bpe", 96)
    rows = []
    for task in tasks:
        part = search[(search["task"] == task) & (search["arch"] == arch_name(MAIN_ARCH))]
        loss = part.sort_values("val_f1_macro", ascending=False).iloc[0]["loss"]
        data = load_task(task)
        classes, label2id = label_maps(data["train"]["label"])
        tensors = encode_splits(data, tok, max_len, label2id, device)
        for arch in archs or [MAIN_ARCH, {"n_layers": 2, "d_model": 256, "n_heads": 4}]:
            for seed in seeds:
                cfg = TrainConfig(epochs=12, patience=3, loss=loss, seed=seed)
                out_dir = config.MODELS_DIR / "_seed_study" / f"{task}_{arch_name(arch)}_s{seed}"
                res = train_variant(task, data, tensors, classes, arch, cfg, out_dir, max_len,
                                    tok.get_vocab_size(), f"seed_{task}_{arch_name(arch)}_s{seed}")
                row = {"task": task, "arch": arch_name(arch), "loss": loss, "seed": seed,
                       "best_epoch": res["best_epoch"], "train_time_s": res["train_time_s"]}
                for split in ("val", "test"):
                    row[f"f1_{split}"] = round(compute_metrics(res["preds"][split]["y_true"],
                                                               res["preds"][split]["y_pred"])["f1_macro"], 4)
                if strict is not None:
                    t = res["preds"]["test"]
                    t = t[t["law_id"].isin(strict)]
                    row["f1_test_strict"] = round(compute_metrics(t["y_true"], t["y_pred"])["f1_macro"], 4)
                rows.append(row)
                print(f"[seed-study] {row}", flush=True)
        del tensors
        torch.cuda.empty_cache()
    shutil.rmtree(config.MODELS_DIR / "_seed_study", ignore_errors=True)
    df = pd.DataFrame(rows)
    df.to_csv(config.RESULTS_DIR / "tiny_transformer_seed_study.csv", index=False)
    summary = df.groupby(["task", "arch"])[[c for c in df.columns if c.startswith("f1_")]].agg(["mean", "std"])
    print(summary.round(4).to_string())
    return df


def run_vocab_variant(tasks: tuple[str, ...] = ("form",), vocab_size: int = 4000,
                      arch: dict | None = None) -> dict:
    """Tokenizer ablation: same model + chosen loss, BPE vocabulary 4k instead of 8k."""
    search = pd.read_csv(config.RESULTS_DIR / "tiny_transformer_loss_search.csv")
    out = {}
    for task in tasks:
        part = search[(search["task"] == task) & (search["arch"] == arch_name(MAIN_ARCH))]
        loss = part.sort_values("val_f1_macro", ascending=False).iloc[0]["loss"]
        out[task] = train_task(task, losses=(loss,), ablation=[], main_arch=arch or MAIN_ARCH,
                               vocab_size=vocab_size)["final"]
    return out


def run(tasks: tuple[str, ...] = ("form", "area"), quick: bool = False,
        losses: tuple[str, ...] | None = None, ablation: list[dict] | None = None,
        extra_ablations: bool = True) -> dict:
    """Train the tiny transformer for every task (quick=True: small smoke test).

    Full runs also train the extra ablations: dropout 0.3 (both tasks) and the
    4k-vocabulary BPE tokenizer (form task).
    """
    if not (BPE_DIR / "tokenizer.json").exists():
        bpe_tokenizer.run(quick=quick)
    losses = losses or (("ce",) if quick else LOSSES)
    results, search_rows = {}, []
    for task in tasks:
        res = train_task(task, quick=quick, losses=losses, ablation=ablation)
        results[task] = res["final"]
        search_rows.extend(res["search"])
    if search_rows:
        path = config.RESULTS_DIR / f"tiny_transformer_loss_search{'_quick' if quick else ''}.csv"
        pd.DataFrame(search_rows).to_csv(path, index=False)
    if extra_ablations and not quick:
        results["dropout_ablation"] = run_dropout_variants(tasks)
        if "form" in tasks:
            results["vocab_ablation"] = run_vocab_variant(("form",), vocab_size=4000)
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the tiny transformer")
    parser.add_argument("--tasks", default="form,area")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--losses", default=None, help="comma list of ce,weighted,smooth")
    parser.add_argument("--no-ablation", action="store_true")
    parser.add_argument("--dropout-variant", action="store_true",
                        help="train only the high-dropout (0.3) variants of the best 2L/4L models")
    parser.add_argument("--seed-study", action="store_true",
                        help="3 seeds of the 4L and 2L models (variance estimate, not logged to SQLite)")
    parser.add_argument("--vocab", type=int, default=None,
                        help="tokenizer ablation: train the main model with this BPE vocab (e.g. 4000)")
    args = parser.parse_args()
    if args.dropout_variant:
        print(run_dropout_variants(tuple(args.tasks.split(","))))
    elif args.vocab:
        print(run_vocab_variant(tuple(args.tasks.split(",")), vocab_size=args.vocab))
    elif args.seed_study:
        run_seed_study(tuple(args.tasks.split(",")))
    else:
        print(run(tuple(args.tasks.split(",")), quick=args.quick,
                  losses=tuple(args.losses.split(",")) if args.losses else None,
                  ablation=[] if args.no_ablation else None))
