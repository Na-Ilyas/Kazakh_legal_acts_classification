"""Seq2seq label generation with google/mt5-small (title -> label text).

* Kazakh vocabulary pruning (as in src.vocab_prune): mT5-small has ~300M
  parameters, mostly the 250k-token shared embedding and the (untied) LM head.
  We keep the SentencePiece ids used by the TRAIN inputs + every candidate
  label + specials, and copy the corresponding rows of both matrices
  -> well under the 120M budget. Unseen input ids map to <unk>.
* Training: teacher-forced cross-entropy on the label sequence.
* Inference = constrained scoring over the fixed label set: every candidate
  label is scored by its sequence log-likelihood given the title, so the output
  is always a valid class. Sum vs. length-normalised (mean) log-likelihood is
  chosen on VAL only. Confidence = softmax over candidate scores.
* Target variants: 'text' (the Kazakh label string) and 'code' (short ASCII
  codes "l0", "l1", ...).
SQLite rows: model_name 'mt5_small_kkprune_<targets>', family 'seq2seq'.
"""
from __future__ import annotations

import json
import math
import time

import numpy as np

from src import config  # noqa: F401  (pins GPU 0 before torch)
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.metrics import macro_f1
from src.model_utils import dir_size_mb, label_maps, load_task, log_all_splits, prediction_frame, save_json
from src.trainer import get_device, set_seed, warmup_cosine

MODEL_ID = "google/mt5-small"
SUMMARY_JSON = config.RESULTS_DIR / "seq2seq_mt5.json"
SRC_MAX_LEN = 96


def load_tokenizer():
    """mT5 fast tokenizer (tokenizer.json shipped in the repo's onnx/ folder - no sentencepiece needed)."""
    from huggingface_hub import hf_hub_download
    from transformers import PreTrainedTokenizerFast
    path = hf_hub_download(MODEL_ID, "onnx/tokenizer.json")
    return PreTrainedTokenizerFast(tokenizer_file=path, pad_token="<pad>", eos_token="</s>", unk_token="<unk>")


def tokenize(tok, texts: list[str], max_len: int, chunk: int = 20_000) -> tuple[np.ndarray, np.ndarray]:
    """Padded int32 ids [N, max_len] + lengths (pad id 0; eos appended by the tokenizer post-processor)."""
    ids = np.zeros((len(texts), max_len), dtype=np.int32)
    lengths = np.zeros(len(texts), dtype=np.int16)
    for start in range(0, len(texts), chunk):
        enc = tok(texts[start:start + chunk], truncation=True, max_length=max_len)["input_ids"]
        for i, row in enumerate(enc, start=start):
            ids[i, :len(row)] = row
            lengths[i] = len(row)
    return ids, lengths


def source_text(task: str, title: str) -> str:
    """Encoder input."""
    return f"{task}: {title}"


def target_strings(classes: list[str], targets: str) -> list[str]:
    """Label sequences to generate: Kazakh label text or short ASCII codes."""
    return list(classes) if targets == "text" else [f"l{i}" for i in range(len(classes))]


# --------------------------------------------------------------------------
# Pruning
# --------------------------------------------------------------------------
def prune_mt5(model, kept: np.ndarray):
    """Keep only `kept` rows of the shared embedding and of the (untied) LM head."""
    kept_t = torch.as_tensor(kept)
    old_in = model.get_input_embeddings()
    new_in = nn.Embedding(len(kept), old_in.embedding_dim)
    old_out = model.get_output_embeddings()
    new_out = nn.Linear(old_out.in_features, len(kept), bias=False)
    with torch.no_grad():
        new_in.weight.copy_(old_in.weight[kept_t])
        new_out.weight.copy_(old_out.weight[kept_t])
    model.set_input_embeddings(new_in)          # also re-points encoder/decoder embed_tokens
    model.set_output_embeddings(new_out)
    model.config.vocab_size = len(kept)
    return model


def count_params(model: nn.Module) -> int:
    """Number of parameters."""
    return int(sum(p.numel() for p in model.parameters()))


# --------------------------------------------------------------------------
# Scoring over the label set
# --------------------------------------------------------------------------
@torch.no_grad()
def score_candidates(model, src_ids: torch.Tensor, src_len: np.ndarray, cand: torch.Tensor,
                     cand_len: torch.Tensor, batch_size: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """(sum, mean) log-likelihood of every candidate label for every row: two [N, K] arrays."""
    device = src_ids.device
    K, L = cand.shape
    # logits are [B*K, L, V]: bound B*K*L (~10k decoder tokens per batch) to keep memory flat for many labels
    batch_size = max(4, min(batch_size, 10_000 // max(K * L, 1)))
    tok_mask = (torch.arange(L, device=device)[None, :] < cand_len[:, None]).float()      # [K, L]
    dec_in = torch.cat([torch.zeros(K, 1, dtype=cand.dtype, device=device), cand[:, :-1]], dim=1)  # start = pad(0)
    sums, means = [], []
    model.eval()
    for start in range(0, len(src_len), batch_size):
        stop = min(start + batch_size, len(src_len))
        t = int(src_len[start:stop].max())
        ids = src_ids[start:stop, :t].long()
        mask = ids != 0
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            enc = model.encoder(input_ids=ids, attention_mask=mask).last_hidden_state          # [B, T, d]
            B = enc.shape[0]
            enc_rep = enc.repeat_interleave(K, dim=0)
            mask_rep = mask.repeat_interleave(K, dim=0)
            logits = model(encoder_outputs=(enc_rep,), attention_mask=mask_rep,
                           decoder_input_ids=dec_in.repeat(B, 1).long()).logits                     # [B*K, L, V]
        logp = torch.log_softmax(logits.float(), dim=-1)
        tok_lp = logp.gather(-1, cand.repeat(B, 1).long().unsqueeze(-1)).squeeze(-1).view(B, K, L)
        s = (tok_lp * tok_mask[None]).sum(-1)
        sums.append(s.cpu().numpy())
        means.append((s / cand_len[None].float()).cpu().numpy())
    return np.concatenate(sums), np.concatenate(means)


def softmax_np(x: np.ndarray) -> np.ndarray:
    """Row-wise softmax."""
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------
# Train / evaluate one task
# --------------------------------------------------------------------------
def run_task(task: str, targets: str = "text", epochs: int = 2, lr: float = 5e-4, batch_size: int = 64,
             max_train: int | None = None, quick: bool = False) -> dict:
    """Prune mT5-small, fine-tune title -> label, evaluate with label-set scoring, log."""
    from transformers import MT5ForConditionalGeneration

    t_start = time.perf_counter()
    data = load_task(task, max_train=8_000 if quick else max_train, max_eval=2_000 if quick else None)
    classes, label2id = label_maps(data["train"]["label"])
    cand_str = target_strings(classes, targets)
    tok = load_tokenizer()

    src = {s: tokenize(tok, [source_text(task, t) for t in f["title"]], SRC_MAX_LEN) for s, f in data.items()}
    cand_ids, cand_len = tokenize(tok, cand_str, 32)
    cand_ids, cand_len = cand_ids[:, :int(cand_len.max())], cand_len
    kept = np.union1d(np.union1d(np.unique(src["train"][0]), np.unique(cand_ids)),
                      np.array([0, 1, 2], dtype=np.int32)).astype(np.int64)          # pad, eos, unk
    old_vocab = 250112
    lookup = np.full(old_vocab + 1000, 2, dtype=np.int32)                              # default -> unk (new id 2)
    lookup[kept] = np.arange(len(kept), dtype=np.int32)
    assert lookup[0] == 0 and lookup[1] == 1 and lookup[2] == 2
    oov = {}
    for s, (ids, lens) in src.items():
        m = np.arange(ids.shape[1])[None, :] < lens[:, None]
        oov[s] = round(100 * float((lookup[ids[m]] == 2).mean() - (ids[m] == 2).mean()), 4)

    model = MT5ForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    n_full = count_params(model)
    model = prune_mt5(model, kept)
    n_params = count_params(model)
    device = get_device()
    model.to(device)
    print(f"[mt5] {task}/{targets}: vocab {old_vocab:,} -> {len(kept):,}; params {n_full:,} -> {n_params:,}; "
          f"OOV val/test {oov.get('val')}%/{oov.get('test')}%; train={len(data['train']):,}")

    to_dev = lambda a: torch.from_numpy(a.astype(np.int64)).to(device)
    src_t = {s: to_dev(lookup[ids]) for s, (ids, _) in src.items()}
    src_len = {s: lens.astype(np.int64) for s, (_, lens) in src.items()}
    cand_t = to_dev(lookup[cand_ids])
    cand_len_t = torch.from_numpy(cand_len.astype(np.int64)).to(device)
    y = {s: np.array([label2id[l] for l in data[s]["label"]]) for s in data}

    set_seed(config.SEED)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    n_train = len(src_len["train"])
    steps_per_epoch = math.ceil(n_train / batch_size)
    total = steps_per_epoch * (1 if quick else epochs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, warmup_cosine(total, int(0.06 * total)))
    rng = np.random.default_rng(config.SEED)
    best = {"f1": -1.0, "epoch": 0, "state": None}
    history = []
    t_train = time.perf_counter()
    for epoch in range(1, (1 if quick else epochs) + 1):
        model.train()
        order = rng.permutation(n_train)
        losses = []
        for start in range(0, n_train, batch_size):
            idx = order[start:start + batch_size]
            t = int(src_len["train"][idx].max())
            ids = src_t["train"][idx, :t]
            lab = cand_t[y["train"][idx]].clone()
            lab[torch.arange(lab.shape[1], device=device)[None, :] >= cand_len_t[y["train"][idx]][:, None]] = -100
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss = model(input_ids=ids, attention_mask=ids != 0, labels=lab).loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(float(loss))
        s_sum, s_mean = score_candidates(model, src_t["val"], src_len["val"], cand_t, cand_len_t)
        f1_sum = macro_f1(y["val"], s_sum.argmax(1))
        f1_mean = macro_f1(y["val"], s_mean.argmax(1))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_f1_sum": f1_sum,
                        "val_f1_mean": f1_mean})
        print(f"[mt5] {task}/{targets} epoch {epoch}: loss {np.mean(losses):.4f} val F1 sum {f1_sum:.4f} mean {f1_mean:.4f}")
        if max(f1_sum, f1_mean) > best["f1"]:
            best = {"f1": max(f1_sum, f1_mean), "epoch": epoch, "score": "sum" if f1_sum >= f1_mean else "mean",
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
    train_time = time.perf_counter() - t_train
    peak_gpu = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None
    model.load_state_dict(best["state"])
    use = 0 if best["score"] == "sum" else 1

    preds = {}
    for s in ("val", "test"):
        scores = score_candidates(model, src_t[s], src_len[s], cand_t, cand_len_t)[use]
        preds[s] = prediction_frame(data[s]["law_id"], data[s]["label"], softmax_np(scores), classes)

    name = f"mt5_small_kkprune_{targets}" + ("_quick" if quick else "")
    out_dir = config.MODELS_DIR / f"{name}_{task}"
    model.save_pretrained(out_dir)
    np.save(out_dir / "id_lookup.npy", lookup)
    save_json({"classes": classes, "targets": cand_str, "score": best["score"], "src_max_len": SRC_MAX_LEN,
               "source_template": f"{task}: <title>", "tokenizer": f"{MODEL_ID} (onnx/tokenizer.json)"},
              out_dir / "labels.json")
    size_mb = dir_size_mb(out_dir)
    latency = cpu_latency_ms(model, tok, lookup, task, cand_t.cpu(), cand_len_t.cpu(), use)
    common = {"n_params": n_params, "size_mb": round(size_mb, 2), "latency_ms_per_1000": round(latency, 1),
              "latency_device": "cpu", "peak_mem_mb": None if peak_gpu is None else round(peak_gpu, 1),
              "train_time_s": round(train_time, 1), "n_train": n_train}
    hp = {"base_model": MODEL_ID, "targets": targets, "n_params_full": n_full, "vocab_kept": int(len(kept)),
          "oov_pct": oov, "epochs": epochs, "best_epoch": best["epoch"], "lr": lr, "batch_size": batch_size,
          "score": best["score"], "src_max_len": SRC_MAX_LEN, "history": history,
          "peak_mem_mb_meaning": "training peak CUDA memory",
          "latency_protocol": "1000 seeded test titles, CPU 8 threads, batch 64, encoder + K-candidate scoring"}
    notes = (f"mT5-small seq2seq (title -> {'Kazakh label text' if targets == 'text' else 'ASCII label code'}), "
             "vocabulary pruned to train ids + labels; prediction = argmax over the fixed label set of the "
             f"sequence log-likelihood ({best['score']}, chosen on val).")
    logged = log_all_splits(name, "seq2seq", task, preds, common, hp, notes=notes)
    f1 = {s: round(v["f1_macro"], 4) for s, v in logged.items()}
    print(f"[mt5] {task}/{targets}: {f1} params {n_full:,} -> {n_params:,}, {size_mb:.0f} MB, {latency:.0f} ms/1000")
    del model, src_t
    torch.cuda.empty_cache()
    return {"f1_macro": f1, "n_params": n_params, "n_params_full": n_full, "vocab_kept": int(len(kept)),
            "oov_pct": oov, "size_mb": round(size_mb, 1), "latency_cpu_ms_per_1000": round(latency, 1),
            "train_time_s": round(train_time, 1), "best_epoch": best["epoch"], "score": best["score"],
            "history": history, "elapsed_s": round(time.perf_counter() - t_start, 1)}


@torch.no_grad()
def cpu_latency_ms(model, tok, lookup: np.ndarray, task: str, cand: torch.Tensor, cand_len: torch.Tensor,
                   use: int, repeats: int = 3) -> float:
    """Median ms for 1000 raw titles on CPU (8 threads, batch 64): tokenise + remap + score all labels."""
    from src import benchmark

    texts = [source_text(task, t) for t in benchmark.benchmark_texts(1000)]
    prev = torch.get_num_threads()
    torch.set_num_threads(8)
    cpu = model.to("cpu").float().eval()
    lut = torch.from_numpy(lookup.astype(np.int64))

    def _predict():
        for start in range(0, len(texts), 64):
            enc = tok(texts[start:start + 64], truncation=True, max_length=SRC_MAX_LEN, padding=True,
                      return_tensors="pt")
            ids = lut[enc["input_ids"]]
            score_candidates(cpu, ids, (ids != 0).sum(1).numpy(), cand, cand_len, batch_size=64)

    _predict()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _predict()
        times.append((time.perf_counter() - t0) * 1000)
    torch.set_num_threads(prev)
    return float(np.median(times))


def run(tasks: tuple[str, ...] = ("form",), targets: tuple[str, ...] = ("text",), epochs: int = 2,
        quick: bool = False) -> dict:
    """Run the requested (task, target variant) combinations; save results/seq2seq_mt5.json."""
    out = {}
    for task in tasks:
        for tg in targets:
            out[f"{task}/{tg}"] = run_task(task, tg, epochs=epochs, quick=quick)
    if not quick:
        try:
            old = json.loads(SUMMARY_JSON.read_text(encoding="utf-8")) if SUMMARY_JSON.exists() else {}
        except (OSError, json.JSONDecodeError):
            old = {}
        old.update(out)
        SUMMARY_JSON.write_text(json.dumps(old, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="mT5-small seq2seq label generation (vocab-pruned)")
    parser.add_argument("--tasks", default="form")
    parser.add_argument("--targets", default="text")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    print(run(tuple(args.tasks.split(",")), tuple(args.targets.split(",")), args.epochs, args.quick))
