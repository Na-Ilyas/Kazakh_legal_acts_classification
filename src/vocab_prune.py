"""Kazakh vocabulary pruning of multilingual-e5-small, then fine-tuning.

multilingual-e5-small has 118M parameters, ~96M of them in the 250k x 384
token-embedding matrix, while Kazakh legal titles use only a small fraction of
the SentencePiece vocabulary. Pruning:
  1. tokenise the TRAIN titles ("query: " + title, as for e5) -> set of used ids;
  2. keep those ids + all special tokens, sorted; old id -> new id lookup table
     (ids never seen in train map to <unk>);
  3. replace the embedding matrix by the kept rows (exact copy - nothing else
     in the network changes), then fine-tune exactly like src.finetune_e5
     (same classifier, optimiser, schedule, epochs, best epoch on val).

Inference: HF tokenizer -> old ids -> lookup -> pruned model. The OOV rate
(tokens of val/test titles that were not seen in train) is reported.
Outputs: artifacts/models/e5_small_kkprune_ft_<task>/, results/vocab_prune.json,
SQLite rows (model_name 'e5_small_kkprune_ft', family 'pretrained').
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from src import config  # noqa: F401  (pins GPU 0 before torch)
import torch
import torch.nn as nn

from src.finetune_e5 import E5Classifier, encode_e5, load_e5, predict_like
from src.model_utils import (E5_MODEL_NAME, E5_PREFIX, dir_size_mb, encode_labels, label_maps, load_task,
                             log_all_splits, max_len_for, prediction_frame, save_json)
from src.trainer import TokenData, TrainConfig, fit, get_device, plot_learning_curve, predict_proba, set_seed

NAME = "e5_small_kkprune_ft"
SUMMARY_JSON = config.RESULTS_DIR / "vocab_prune.json"


# --------------------------------------------------------------------------
# Vocabulary pruning
# --------------------------------------------------------------------------
def kept_token_ids(train_ids: np.ndarray, special_ids: list[int]) -> np.ndarray:
    """Sorted old ids to keep: every id used by the train titles + special tokens."""
    return np.union1d(np.unique(train_ids), np.asarray(special_ids, dtype=train_ids.dtype)).astype(np.int64)


def build_lookup(kept: np.ndarray, old_vocab_size: int, unk_old: int) -> np.ndarray:
    """old id -> new id (int32); ids not kept map to the new id of <unk>."""
    lookup = np.empty(old_vocab_size, dtype=np.int32)
    new_of = {int(o): i for i, o in enumerate(kept)}
    lookup[:] = new_of[unk_old]
    lookup[kept] = np.arange(len(kept), dtype=np.int32)
    return lookup


def prune_encoder(encoder: nn.Module, kept: np.ndarray, pad_new: int) -> nn.Module:
    """Replace the input embedding matrix by its kept rows (exact copy)."""
    old = encoder.get_input_embeddings()
    new = nn.Embedding(len(kept), old.embedding_dim, padding_idx=pad_new)
    with torch.no_grad():
        new.weight.copy_(old.weight[torch.as_tensor(kept)])
    encoder.set_input_embeddings(new)
    encoder.config.vocab_size = len(kept)
    encoder.config.pad_token_id = pad_new
    return encoder


def oov_rate(ids_old: np.ndarray, lengths: np.ndarray, lookup: np.ndarray, unk_old: int, unk_new: int) -> float:
    """% of real (non-pad) tokens that become <unk> only because of pruning."""
    mask = np.arange(ids_old.shape[1])[None, :] < lengths[:, None]
    toks = ids_old[mask]
    return round(100 * float(((lookup[toks] == unk_new) & (toks != unk_old)).mean()), 4)


def fold_title(title: str) -> str:
    """Preprocessing ablation: NFC + Latin->Cyrillic homoglyph folding inside Kazakh words (case kept)."""
    import unicodedata
    from src import labels
    return labels._fold_homoglyphs(unicodedata.normalize("NFC", title)) if isinstance(title, str) else ""


def count_params(model: nn.Module) -> int:
    """Number of parameters."""
    return int(sum(p.numel() for p in model.parameters()))


# --------------------------------------------------------------------------
# Latency (same protocol as src.benchmark: 1000 seeded test titles, CPU 8 threads, batch 64)
# --------------------------------------------------------------------------
@torch.no_grad()
def cpu_latency_ms(model: nn.Module, hf_tok, lookup: np.ndarray, max_len: int, repeats: int = 3,
                   fold: bool = False) -> float:
    """Median ms for 1000 raw titles: tokenise + remap + forward on CPU."""
    from src import benchmark

    texts = benchmark.benchmark_texts(1000)
    if fold:
        texts = [fold_title(t) for t in texts]
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(8)
    cpu_model = model.to("cpu").eval()
    lut = torch.from_numpy(lookup.astype(np.int64))

    def _predict():
        for start in range(0, len(texts), 64):
            enc = hf_tok([E5_PREFIX + t for t in texts[start:start + 64]], truncation=True,
                         max_length=max_len, padding=True, return_tensors="pt")
            cpu_model(lut[enc["input_ids"]], enc["attention_mask"].bool())

    _predict()                                   # warm-up
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _predict()
        times.append((time.perf_counter() - t0) * 1000)
    torch.set_num_threads(prev_threads)
    return float(np.median(times))


# --------------------------------------------------------------------------
# Train one task
# --------------------------------------------------------------------------
def run_task(task: str, lr: float = 5e-5, epochs: int = 3, batch_size: int = 64, quick: bool = False,
             fold: bool = False) -> dict:
    """Prune e5-small to the Kazakh train vocabulary, fine-tune, evaluate, log."""
    data = load_task(task, max_train=8_000 if quick else None, max_eval=2_000 if quick else None)
    base_name = NAME.replace("_ft", "_fold_ft") if fold else NAME
    if fold:
        for frame in data.values():
            frame["title"] = frame["title"].map(fold_title)
    classes, label2id = label_maps(data["train"]["label"])
    max_len = max_len_for("e5", 112)
    hf_tok, encoder = load_e5()
    n_params_full = count_params(encoder)
    old_vocab = encoder.get_input_embeddings().num_embeddings
    unk_old, pad_old = hf_tok.unk_token_id, hf_tok.pad_token_id

    ids_old = {s: encode_e5(hf_tok, f["title"].tolist(), max_len) for s, f in data.items()}
    kept = kept_token_ids(ids_old["train"][0], hf_tok.all_special_ids)
    lookup = build_lookup(kept, old_vocab, unk_old)
    unk_new, pad_new = int(lookup[unk_old]), int(lookup[pad_old])
    oov = {s: oov_rate(ids, lens, lookup, unk_old, unk_new) for s, (ids, lens) in ids_old.items()}

    encoder = prune_encoder(encoder, kept, pad_new)
    device = get_device()
    set_seed(config.SEED)
    model = E5Classifier(encoder, len(classes)).to(device)
    n_params = count_params(model)
    print(f"[vocab_prune] {task}: vocab {old_vocab:,} -> {len(kept):,}; params {n_params_full:,} (encoder) -> "
          f"{n_params:,} (pruned encoder + head); OOV val/test {oov.get('val')}%/{oov.get('test')}%")
    tensors = {s: TokenData.build(lookup[ids], lens, encode_labels(data[s]["label"], label2id), device)
               for s, (ids, lens) in ids_old.items()}
    del ids_old

    cfg = TrainConfig(epochs=1 if quick else epochs, batch_size=batch_size, lr=lr, weight_decay=0.01,
                      warmup_ratio=0.06, loss="ce", patience=epochs, eval_batch_size=512, bucket=True)
    run_name = f"{base_name}_{task}" + ("_quick" if quick else "")
    out_dir = config.MODELS_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    res = fit(model, tensors["train"], tensors["val"], len(classes), cfg, out_dir / "model.pt", run_name,
              log_csv=config.RESULTS_DIR / f"train_log_{run_name}.csv")
    preds = {s: prediction_frame(data[s]["law_id"], data[s]["label"], predict_proba(model, tensors[s], 512), classes)
             for s in ("val", "test")}
    # Portable artefacts: pruned HF encoder + head + id lookup (old -> new) + labels
    model.encoder.save_pretrained(out_dir / "encoder")
    torch.save(model.head.state_dict(), out_dir / "head.pt")
    np.save(out_dir / "id_lookup.npy", lookup)
    (out_dir / "model.pt").unlink(missing_ok=True)
    save_json(label2id, out_dir / "label2id.json")
    save_json({"task": task, "base_model": E5_MODEL_NAME, "tokenizer": E5_MODEL_NAME, "max_len": max_len,
               "prefix": E5_PREFIX, "kept_vocab": int(len(kept)), "unk_new": unk_new, "pad_new": pad_new},
              out_dir / "config.json")
    plot_learning_curve(res["history"], f"e5-small, Kazakh-pruned vocabulary - task '{task}'",
                        config.FIGURES_DIR / f"learning_curve_{run_name}.png")
    size_mb = dir_size_mb(out_dir)
    latency = cpu_latency_ms(model, hf_tok, lookup, max_len, fold=fold)
    del tensors
    torch.cuda.empty_cache()

    common = {"n_params": n_params, "size_mb": round(size_mb, 2), "latency_ms_per_1000": round(latency, 1),
              "latency_device": "cpu", "peak_mem_mb": res["train_peak_gpu_mb"],
              "train_time_s": res["train_time_s"], "n_train": len(data["train"])}
    hp = {"base_model": E5_MODEL_NAME, "vocab_full": old_vocab, "vocab_kept": int(len(kept)),
          "n_params_full_model": n_params_full + model.head.weight.numel() + model.head.bias.numel(),
          "oov_pct": oov, "max_len": max_len, "best_epoch": res["best_epoch"], "epochs_run": res["epochs_run"],
          **res["train_config"], "peak_mem_mb_meaning": "training peak CUDA memory",
          "latency_protocol": "1000 seeded test titles, CPU 8 threads, batch 64, tokenise+remap+forward, median of 3"}
    notes = ("multilingual-e5-small with its token-embedding matrix pruned to the SentencePiece ids seen in "
             "TRAIN titles (+specials); unseen ids -> <unk>. Fine-tuned exactly like e5_small_ft "
             "(mean pooling + linear head, AdamW, warmup+cosine, best val epoch)."
             + (" Preprocessing ablation: titles NFC + homoglyph-folded before tokenisation." if fold else ""))
    name = base_name + ("_quick" if quick else "")
    logged = log_all_splits(name, "pretrained", task, preds, common, hp, notes=notes)
    f1 = {s: round(v["f1_macro"], 4) for s, v in logged.items()}
    print(f"[vocab_prune] {task}: {f1} params={n_params:,} size={size_mb:.1f} MB latency={latency:.0f} ms/1000")
    return {"f1_macro": f1, "n_params": n_params, "n_params_full": hp["n_params_full_model"],
            "vocab_kept": int(len(kept)), "vocab_full": old_vocab, "oov_pct": oov, "size_mb": round(size_mb, 1),
            "latency_cpu_ms_per_1000": round(latency, 1), "train_time_s": res["train_time_s"],
            "best_epoch": res["best_epoch"]}


def run(tasks: tuple[str, ...] = ("form", "area"), quick: bool = False, fold: bool = False) -> dict:
    """Prune + fine-tune for every task; save results/vocab_prune.json."""
    out = {}
    for task in tasks:
        out[task + ("/fold" if fold else "")] = run_task(task, quick=quick, fold=fold)
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

    parser = argparse.ArgumentParser(description="Kazakh vocabulary pruning of e5-small + fine-tuning")
    parser.add_argument("--tasks", default="form,area")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--fold", action="store_true", help="homoglyph-fold titles before tokenisation (ablation)")
    args = parser.parse_args()
    print(run(tuple(args.tasks.split(",")), quick=args.quick, fold=args.fold))


def latency_compare(task: str = "form", update_db: bool = True) -> dict:
    """Fair CPU latency: full e5_small_ft vs the pruned model, same process, same protocol, back to back.

    (The latency logged during training can be inflated by other jobs on the shared CPU.)
    """
    from transformers import AutoModel, AutoTokenizer
    from src import experiments_db

    res = {}
    for name in ("e5_small_ft", NAME):
        d = config.MODELS_DIR / f"{name}_{task}"
        enc = AutoModel.from_pretrained(d / "encoder")
        head = torch.load(d / "head.pt", map_location="cpu", weights_only=True)
        model = E5Classifier(enc, head["weight"].shape[0])
        model.head.load_state_dict(head)
        tok = AutoTokenizer.from_pretrained(E5_MODEL_NAME)
        lookup = np.load(d / "id_lookup.npy") if (d / "id_lookup.npy").exists() else np.arange(len(tok), dtype=np.int32)
        res[name] = {"latency_cpu_ms_per_1000": round(cpu_latency_ms(model, tok, lookup, max_len_for("e5", 112)), 1),
                     "n_params": count_params(model)}
        del model, enc
    if update_db:
        with experiments_db.session() as con:
            con.execute("UPDATE experiments SET latency_ms_per_1000=? WHERE model_name=? AND task=?",
                        (res[NAME]["latency_cpu_ms_per_1000"], NAME, task))
    print(f"[vocab_prune] latency compare ({task}): {res}")
    return res
