"""Optional extension: masked-language-model (MLM) pretraining of the tiny transformer.

Self-supervised pretraining on unlabeled in-domain text, then fine-tuning on
the classification tasks (model name suffix `_mlm`).

* Text: TRAIN-period acts only (year <= 2021): their titles AND their
  `description` field (e.g. "... министрінің 2004 жылғы 15 шілдедегі № 417
  бұйрығы"). No val/test text is ever seen. The classifier input stays the title.
* Objective: BERT-style masking of 15% of the non-special tokens
  (80% -> [MASK], 10% -> random token, 10% unchanged); a small transform +
  decoder tied to the token embedding predicts the original tokens.
* The same 4-layer d=256 encoder as the main tiny transformer; afterwards
  src.tiny_transformer.train_task fine-tunes it (fresh linear head) with the
  loss chosen on validation for the from-scratch model.
"""
from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd

from src import config  # noqa: F401  (pins GPU 0 before torch)
import torch
import torch.nn as nn
import torch.nn.functional as F

from src import bpe_tokenizer
from src.model_utils import BPE_DIR, load_dataset, max_len_for, save_json
from src.tiny_transformer import MAIN_ARCH, TinyTransformerClassifier, arch_name, count_params, train_task
from src.trainer import TokenData, epoch_batches, get_device, make_batch, param_groups, set_seed, warmup_cosine

MLM_DIR = config.MODELS_DIR / f"tiny_mlm_pretrained_{arch_name(MAIN_ARCH)}"
MASK_PROB = 0.15


def mlm_dir(quick: bool = False):
    """Output directory (quick smoke tests never overwrite the real encoder)."""
    return MLM_DIR.with_name(MLM_DIR.name + "_quick") if quick else MLM_DIR


class MLMHead(nn.Module):
    """Dense + GELU + LayerNorm, then a decoder tied to the input embedding matrix."""

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.transform = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.LayerNorm(d_model))
        self.bias = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, hidden: torch.Tensor, embedding: nn.Embedding) -> torch.Tensor:
        return F.linear(self.transform(hidden), embedding.weight, self.bias)


def mask_tokens(ids: torch.Tensor, mask: torch.Tensor, vocab_size: int,
                gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """BERT masking. Returns (corrupted ids, labels) with label -100 where not predicted."""
    special = ids < len(bpe_tokenizer.SPECIAL_TOKENS)           # [PAD] [UNK] [CLS] [SEP] [MASK]
    prob = torch.full(ids.shape, MASK_PROB, device=ids.device)
    prob.masked_fill_(special | ~mask, 0.0)
    selected = torch.bernoulli(prob, generator=gen).bool()
    labels = torch.where(selected, ids, torch.full_like(ids, -100))
    r = torch.rand(ids.shape, device=ids.device, generator=gen)
    corrupted = ids.clone()
    corrupted[selected & (r < 0.8)] = bpe_tokenizer.MASK_ID
    random_pos = selected & (r >= 0.8) & (r < 0.9)
    random_ids = torch.randint(len(bpe_tokenizer.SPECIAL_TOKENS), vocab_size, ids.shape,
                               device=ids.device, generator=gen)
    corrupted[random_pos] = random_ids[random_pos]
    return corrupted, labels


def pretraining_texts(quick: bool = False) -> tuple[list[str], list[str]]:
    """(train texts = train titles + train descriptions, val titles for monitoring only)."""
    ds = load_dataset(["law_id", "title", "split"])
    train_ids = set(ds.loc[ds["split"] == "train", "law_id"].astype(str))
    corpus = pd.read_parquet(config.FULL_PARQUET, columns=["law_id", "description"])
    desc = corpus[corpus["law_id"].astype(str).isin(train_ids)]["description"].dropna().astype(str)
    desc = desc[desc.str.len() > 10]
    texts = ds.loc[ds["split"] == "train", "title"].astype(str).tolist() + desc.tolist()
    val_titles = ds.loc[ds["split"] == "val", "title"].astype(str).tolist()
    if quick:
        rng = np.random.default_rng(config.SEED)
        texts = [texts[i] for i in rng.choice(len(texts), 20_000, replace=False)]
        val_titles = val_titles[:2_000]
    return texts, val_titles


@torch.no_grad()
def mlm_eval(model, head, data: TokenData, vocab: int, batch_size: int = 1024) -> tuple[float, float]:
    """Masked-token loss and accuracy on held-out titles (fixed masking seed)."""
    model.eval()
    head.eval()
    device = next(model.parameters()).device
    gen = torch.Generator(device=device).manual_seed(123)
    total_loss, total_correct, total = 0.0, 0, 0
    for start in range(0, len(data), batch_size):
        idx = np.arange(start, min(start + batch_size, len(data)))
        ids, mask = make_batch(data, idx, device)
        inp, labels = mask_tokens(ids, mask, vocab, gen)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            hidden = model.hidden_states(inp, mask)
            sel = labels != -100
            logits = head(hidden[sel], model.tok_emb).float()
        total_loss += F.cross_entropy(logits, labels[sel], reduction="sum").item()
        total_correct += (logits.argmax(-1) == labels[sel]).sum().item()
        total += int(sel.sum())
    return total_loss / max(total, 1), total_correct / max(total, 1)


def pretrain(epochs: int = 10, batch_size: int = 256, lr: float = 5e-4, quick: bool = False) -> dict:
    """Run MLM pretraining and save the encoder weights to MLM_DIR/encoder.pt."""
    set_seed(config.SEED)
    device = get_device()
    tok = bpe_tokenizer.load_tokenizer(BPE_DIR)
    vocab = tok.get_vocab_size()
    max_len = max_len_for("bpe", 96)
    texts, val_titles = pretraining_texts(quick)
    t0 = time.time()
    ids, lengths = bpe_tokenizer.encode_titles(tok, texts, max_len)
    train = TokenData.build(ids, lengths, None, device)
    vids, vlens = bpe_tokenizer.encode_titles(tok, val_titles, max_len)
    val = TokenData.build(vids, vlens, None, device)
    print(f"[mlm] {len(texts):,} train-period texts ({len(val_titles):,} val titles for monitoring), "
          f"tokenised in {time.time() - t0:.1f}s")

    model = TinyTransformerClassifier(vocab_size=vocab, n_classes=1, max_len=max_len, **MAIN_ARCH).to(device)
    head = MLMHead(MAIN_ARCH["d_model"], vocab).to(device)
    params = param_groups(model, 0.01) + param_groups(head, 0.01)
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.98), eps=1e-6, fused=device.type == "cuda")
    epochs = 1 if quick else epochs
    steps = math.ceil(len(train) / batch_size) * epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, warmup_cosine(steps, int(0.06 * steps)))
    gen = torch.Generator(device=device).manual_seed(config.SEED)
    rng = np.random.default_rng(config.SEED)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    history, t_start = [], time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        head.train()
        loss_sum, n_batches, te = torch.zeros((), device=device), 0, time.time()
        for idx in epoch_batches(train.lengths, batch_size, rng, bucket=True):
            x, mask = make_batch(train, idx, device)
            inp, labels = mask_tokens(x, mask, vocab, gen)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                hidden = model.hidden_states(inp, mask)
                sel = labels != -100
                logits = head(hidden[sel], model.tok_emb)          # only masked positions -> cheap
            loss = F.cross_entropy(logits.float(), labels[sel])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), 1.0)
            opt.step()
            sched.step()
            loss_sum += loss.detach()
            n_batches += 1
        val_loss, val_acc = mlm_eval(model, head, val, vocab)
        row = {"epoch": epoch, "train_mlm_loss": round(float(loss_sum.item() / n_batches), 4),
               "val_mlm_loss": round(val_loss, 4), "val_masked_token_acc": round(val_acc, 4),
               "epoch_time_s": round(time.time() - te, 1)}
        history.append(row)
        print(f"[mlm] epoch {epoch}: {row}", flush=True)
    train_time = time.time() - t_start
    out_dir = mlm_dir(quick)
    out_dir.mkdir(parents=True, exist_ok=True)
    state = {k: v for k, v in model.state_dict().items() if not k.startswith("head.")}
    torch.save(state, out_dir / "encoder.pt")
    hist = pd.DataFrame(history)
    sfx = "_quick" if quick else ""
    hist.to_csv(config.RESULTS_DIR / f"train_log_tiny_mlm_pretrain{sfx}.csv", index=False)
    plot_mlm_curve(hist, config.FIGURES_DIR / f"learning_curve_tiny_mlm_pretrain{sfx}.png")
    info = {"arch": MAIN_ARCH, "vocab_size": vocab, "max_len": max_len, "n_texts": len(texts),
            "epochs": epochs, "batch_size": batch_size, "lr": lr, "mask_prob": MASK_PROB,
            "encoder_params": count_params(model) - sum(p.numel() for p in model.head.parameters()),
            "train_time_s": round(train_time, 1),
            "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1) if device.type == "cuda" else None,
            "final": history[-1], "data": "train-period (<=2021) titles + descriptions only"}
    save_json(info, out_dir / "config.json")
    del model, head, train, val
    torch.cuda.empty_cache()
    return info


def plot_mlm_curve(hist: pd.DataFrame, path) -> None:
    """MLM loss and masked-token accuracy per epoch."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    a1.plot(hist["epoch"], hist["train_mlm_loss"], "o-", label="train (masked tokens)")
    a1.plot(hist["epoch"], hist["val_mlm_loss"], "s-", label="val titles 2022")
    a1.set_xlabel("epoch"); a1.set_ylabel("cross-entropy"); a1.set_title("MLM loss"); a1.legend()
    a2.plot(hist["epoch"], hist["val_masked_token_acc"], "o-", color="#3a9e5f")
    a2.set_xlabel("epoch"); a2.set_ylabel("accuracy"); a2.set_title("Masked-token accuracy (val titles)")
    fig.suptitle("Tiny transformer MLM pretraining on train-period titles + descriptions")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(tasks: tuple[str, ...] = ("form", "area"), quick: bool = False, epochs: int = 10,
        force: bool = False, finetune_lr: float = 3e-4) -> dict:
    """Pretrain (unless already done) and fine-tune for every task."""
    enc = mlm_dir(quick) / "encoder.pt"
    info = pretrain(epochs=epochs, quick=quick) if (force or quick or not enc.exists()) else \
        {"reused": str(enc)}
    search = pd.read_csv(config.RESULTS_DIR / "tiny_transformer_loss_search.csv")
    out = {"pretrain": info}
    for task in tasks:
        part = search[(search["task"] == task) & (search["arch"] == arch_name(MAIN_ARCH))]
        loss = part.sort_values("val_f1_macro", ascending=False).iloc[0]["loss"] if len(part) else "ce"
        out[task] = train_task(task, quick=quick, losses=(loss,), ablation=[], init_from=enc,
                               cfg_overrides={"lr": finetune_lr})["final"]
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MLM pretraining + fine-tuning of the tiny transformer")
    parser.add_argument("--tasks", default="form,area")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(run(tuple(args.tasks.split(",")), quick=args.quick, epochs=args.epochs, force=args.force))
