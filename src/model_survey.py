"""Survey of compact (<=120M params) Kazakh-capable encoders + one fine-tune.

1. For each candidate: parameter count (HF metadata / config), tokenizer
   fertility on 3,000 seeded TRAIN titles (tokens per whitespace word), share
   of <unk> tokens and share of titles longer than 128 tokens.
2. Fine-tune the most promising candidate (monolingual Kazakh RoBERTa
   kz-transformers/kaz-roberta-conversational, 83.5M params) on task `form`
   with the same recipe as e5_small_ft (masked mean pooling + linear head,
   AdamW 5e-5, warmup + cosine, 3 epochs, best val epoch).
Outputs: results/model_survey.md, results/model_survey.json, SQLite rows
(model_name 'kaz_roberta_ft', family 'pretrained').
"""
from __future__ import annotations

import json
import time

import numpy as np

from src import config  # noqa: F401  (pins GPU 0 before torch)
import torch

from src.finetune_e5 import E5Classifier
from src.model_utils import dir_size_mb, encode_labels, label_maps, load_task, log_all_splits, prediction_frame, save_json
from src.trainer import TokenData, TrainConfig, fit, get_device, predict_proba, set_seed

SURVEY_MD = config.RESULTS_DIR / "model_survey.md"
SURVEY_JSON = config.RESULTS_DIR / "model_survey.json"
CANDIDATES = {   # model id -> (params, note); params from HF safetensors metadata (checked 2026-11-16)
    "intfloat/multilingual-e5-small": (117_654_272, "multilingual MiniLM-L12, XLM-R SentencePiece 250k (project baseline)"),
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": (117_654_272, "same architecture/tokenizer as e5-small"),
    "kz-transformers/kaz-roberta-conversational": (83_504_930, "monolingual Kazakh RoBERTa"),
    "Eraly-ml/KazBERT": (110_650_880, "Kazakh BERT"),
    "google/mt5-small": (300_176_768, "seq2seq; >120M unless vocabulary-pruned (src.seq2seq_mt5)"),
    "FacebookAI/xlm-roberta-base": (278_885_778, "too large (>120M) without pruning"),
}
FT_MODEL = "kz-transformers/kaz-roberta-conversational"
PRUNED_ROWS = [("e5_small_kkprune_ft", "25.5M", "e5-small with Kazakh-pruned vocabulary (src.vocab_prune)"),
               ("kaz_roberta_kkprune_ft", "see SQLite", "kaz-roberta with Kazakh-train-pruned vocabulary"),
               ("mt5_small_kkprune_text", "58.7M", "mT5-small seq2seq, pruned vocab, label-set scoring (src.seq2seq_mt5)")]
FT_NAME = "kaz_roberta_ft"
PRUNED_FT_NAME = "kaz_roberta_kkprune_ft"
MAX_LEN = 128


def load_tok(model_id: str):
    """Tokenizer of a candidate (mT5 via its fast tokenizer.json)."""
    if model_id == "google/mt5-small":
        from src.seq2seq_mt5 import load_tokenizer
        return load_tokenizer()
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_id)


def tokenizer_stats(model_id: str, titles: list[str]) -> dict:
    """Fertility (tokens / word), <unk> share and >128-token share on sample titles."""
    tok = load_tok(model_id)
    enc = tok(titles, add_special_tokens=False)["input_ids"]
    n_tok = np.array([len(e) for e in enc])
    n_words = np.array([max(len(t.split()), 1) for t in titles])
    unk = tok.unk_token_id
    n_unk = sum(sum(1 for i in e if i == unk) for e in enc) if unk is not None else 0
    return {"vocab_size": int(len(tok)), "fertility_tokens_per_word": round(float(n_tok.sum() / n_words.sum()), 3),
            "median_tokens_per_title": float(np.median(n_tok)), "unk_pct": round(100 * n_unk / max(n_tok.sum(), 1), 4),
            "pct_titles_over_128_tokens": round(100 * float((n_tok > MAX_LEN - 2).mean()), 2)}


def finetune(task: str = "form", model_id: str = FT_MODEL, lr: float = 5e-5, epochs: int = 3,
             batch_size: int = 64, prune: bool = False) -> dict:
    """Fine-tune one encoder (mean pooling + linear head) and log val/test/test_strict.

    prune=True first prunes the token-embedding matrix to the ids used by the TRAIN
    titles (+ specials), exactly like src.vocab_prune does for e5-small.
    """
    from transformers import AutoModel, AutoTokenizer
    from src import vocab_prune

    name = PRUNED_FT_NAME if prune else FT_NAME
    data = load_task(task)
    classes, label2id = label_maps(data["train"]["label"])
    tok = AutoTokenizer.from_pretrained(model_id)
    device = get_device()
    raw = {}
    for s, f in data.items():
        enc = tok(f["title"].tolist(), truncation=True, max_length=MAX_LEN)["input_ids"]
        ids = np.full((len(enc), MAX_LEN), tok.pad_token_id, dtype=np.int32)
        lens = np.array([len(e) for e in enc], dtype=np.int64)
        for i, e in enumerate(enc):
            ids[i, :len(e)] = e
        raw[s] = (ids, lens)
    encoder = AutoModel.from_pretrained(model_id)
    lookup, extra_hp = None, {}
    if prune:
        n_full = int(sum(p.numel() for p in encoder.parameters()))
        kept = vocab_prune.kept_token_ids(raw["train"][0], tok.all_special_ids)
        lookup = vocab_prune.build_lookup(kept, len(tok), tok.unk_token_id)
        pad_new = int(lookup[tok.pad_token_id])
        # RoBERTa derives position ids from the pad id: it must not move
        assert pad_new == tok.pad_token_id, (pad_new, tok.pad_token_id)
        oov = {s: vocab_prune.oov_rate(ids, lens, lookup, tok.unk_token_id, int(lookup[tok.unk_token_id]))
               for s, (ids, lens) in raw.items()}
        encoder = vocab_prune.prune_encoder(encoder, kept, pad_new)
        raw = {s: (lookup[ids], lens) for s, (ids, lens) in raw.items()}
        extra_hp = {"vocab_full": len(tok), "vocab_kept": int(len(kept)), "oov_pct": oov, "n_params_full_encoder": n_full}
        print(f"[survey] pruned {model_id}: vocab {len(tok):,} -> {len(kept):,}, OOV {oov}")
    tensors = {s: TokenData.build(ids, lens, encode_labels(data[s]["label"], label2id), device)
               for s, (ids, lens) in raw.items()}
    set_seed(config.SEED)
    model = E5Classifier(encoder, len(classes)).to(device)
    n_params = int(sum(p.numel() for p in model.parameters()))
    cfg = TrainConfig(epochs=epochs, batch_size=batch_size, lr=lr, weight_decay=0.01, warmup_ratio=0.06,
                      loss="ce", patience=epochs, eval_batch_size=512, bucket=True)
    out_dir = config.MODELS_DIR / f"{name}_{task}"
    out_dir.mkdir(parents=True, exist_ok=True)
    res = fit(model, tensors["train"], tensors["val"], len(classes), cfg, out_dir / "model.pt", f"{name}_{task}",
              log_csv=config.RESULTS_DIR / f"train_log_{name}_{task}.csv")
    preds = {s: prediction_frame(data[s]["law_id"], data[s]["label"], predict_proba(model, tensors[s], 512), classes)
             for s in ("val", "test")}
    model.encoder.save_pretrained(out_dir / "encoder")
    tok.save_pretrained(out_dir / "encoder")
    torch.save(model.head.state_dict(), out_dir / "head.pt")
    (out_dir / "model.pt").unlink(missing_ok=True)
    save_json(label2id, out_dir / "label2id.json")
    if lookup is not None:
        np.save(out_dir / "id_lookup.npy", lookup)
    size_mb = dir_size_mb(out_dir)
    latency = _cpu_latency(model, tok, lookup=lookup)
    common = {"n_params": n_params, "size_mb": round(size_mb, 2), "latency_ms_per_1000": round(latency, 1),
              "latency_device": "cpu", "peak_mem_mb": res["train_peak_gpu_mb"], "train_time_s": res["train_time_s"],
              "n_train": len(data["train"])}
    hp = {"base_model": model_id, "max_len": MAX_LEN, "best_epoch": res["best_epoch"], **res["train_config"], **extra_hp,
          "peak_mem_mb_meaning": "training peak CUDA memory",
          "latency_protocol": "1000 seeded test titles, CPU 8 threads, batch 64, median of 3"}
    logged = log_all_splits(name, "pretrained", task, preds, common, hp,
                            notes=f"fine-tuned {model_id} (monolingual Kazakh RoBERTa), mean pooling + linear head, "
                                  "same recipe as e5_small_ft"
                                  + ("; token-embedding matrix pruned to train ids (unseen -> <unk>)" if prune else ""))
    del model, tensors
    torch.cuda.empty_cache()
    return {"task": task, "f1_macro": {s: round(v["f1_macro"], 4) for s, v in logged.items()}, "n_params": n_params,
            "size_mb": round(size_mb, 1), "latency_cpu_ms_per_1000": round(latency, 1),
            "train_time_s": res["train_time_s"], "best_epoch": res["best_epoch"]}


@torch.no_grad()
def _cpu_latency(model, tok, repeats: int = 3, lookup: np.ndarray | None = None) -> float:
    """Median ms per 1000 seeded test titles on CPU (8 threads, batch 64)."""
    from src import benchmark
    texts = benchmark.benchmark_texts(1000)
    prev = torch.get_num_threads()
    torch.set_num_threads(8)
    cpu = model.to("cpu").eval()
    lut = None if lookup is None else torch.from_numpy(lookup.astype(np.int64))

    def _go():
        for start in range(0, len(texts), 64):
            enc = tok(texts[start:start + 64], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
            ids = enc["input_ids"] if lut is None else lut[enc["input_ids"]]
            cpu(ids, enc["attention_mask"].bool())

    _go()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _go()
        times.append((time.perf_counter() - t0) * 1000)
    torch.set_num_threads(prev)
    return float(np.median(times))


def write_markdown(rows: list[dict], ft: dict | None) -> None:
    """results/model_survey.md"""
    lines = ["# Compact Kazakh-capable encoders - survey", "",
             "Tokenizer statistics on 3,000 seeded TRAIN titles (fertility = tokens per whitespace word; lower is "
             "better). Params from HF metadata. F1 = macro-F1 on task `form` where the model was trained in this "
             "project (see SQLite `experiments`).", "",
             "| model | params | <=120M | vocab | fertility | median tok/title | unk % | >128 tok % | note | form F1 val / test / strict | area F1 val / test / strict |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        f1 = r.get("f1", "")
        lines.append(f"| {r['model']} | {r['params'] / 1e6:.1f}M | {'yes' if r['params'] <= 120e6 else 'no'} | "
                     f"{r.get('vocab_size', '')} | {r.get('fertility_tokens_per_word', '')} | "
                     f"{r.get('median_tokens_per_title', '')} | {r.get('unk_pct', '')} | "
                     f"{r.get('pct_titles_over_128_tokens', '')} | {r['note']} | {f1} | {r.get('f1_area', '')} |")
    for name, params, note in PRUNED_ROWS:
        lines.append(f"| {name} (this project) | {params} | yes | train vocab | = base | = base | 0 | = base | {note} | "
                     f"{existing_f1(name, 'form')} | {existing_f1(name, 'area')} |")
    if ft:
        lines += ["", f"Fine-tuned `{FT_MODEL}`: {ft['n_params']:,} params, {ft['size_mb']} MB, "
                      f"{ft['latency_cpu_ms_per_1000']} ms / 1000 titles on a busy shared CPU, best epoch "
                      f"{ft['best_epoch']}, train {ft['train_time_s']} s (task {ft.get('task', 'form')})."]
    SURVEY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def existing_f1(model_name: str, task: str = "form") -> str:
    """'val / test / strict' macro-F1 of an already logged model (or '')."""
    from src import experiments_db
    try:
        d = experiments_db.load_experiments(task)
    except Exception:
        return ""
    d = d[d["model_name"] == model_name].set_index("eval_split")["f1_macro"]
    return " / ".join(f"{d[s]:.4f}" for s in ("val", "test", "test_strict") if s in d) if len(d) else ""


def run(finetune_model: bool = True, tasks: tuple[str, ...] = ("form", "area")) -> dict:
    """Tokenizer survey for every candidate (+ optional fine-tune of FT_MODEL on `tasks`)."""
    data = load_task("form")
    titles = data["train"]["title"].sample(3000, random_state=config.SEED).tolist()
    rows = []
    for mid, (params, note) in CANDIDATES.items():
        try:
            st = tokenizer_stats(mid, titles)
        except Exception as exc:  # gated / unavailable tokenizer must not stop the survey
            st = {"error": repr(exc)[:120]}
        rows.append({"model": mid, "params": params, "note": note, **st})
        print(f"[survey] {mid}: {st}")
    ft = None
    for task in (tasks if finetune_model else ()):
        ft = finetune(task)
    logged_as = {"intfloat/multilingual-e5-small": "e5_small_ft", FT_MODEL: FT_NAME}
    for r in rows:
        r["f1"] = existing_f1(logged_as.get(r["model"], "-"), "form")
        r["f1_area"] = existing_f1(logged_as.get(r["model"], "-"), "area")
    write_markdown(rows, ft)
    SURVEY_JSON.write_text(json.dumps({"candidates": rows, "finetune": ft}, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    return {"rows": len(rows), "finetune": ft}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Kazakh encoder survey")
    parser.add_argument("--no-finetune", action="store_true")
    parser.add_argument("--tasks", default="form,area")
    parser.add_argument("--prune", action="store_true", help="fine-tune the vocabulary-pruned FT_MODEL instead")
    a = parser.parse_args()
    if a.prune:
        for t in a.tasks.split(","):
            print(finetune(t, prune=True))
        print(run(finetune_model=False))          # refresh the survey table from SQLite
    else:
        print(run(finetune_model=not a.no_finetune, tasks=tuple(a.tasks.split(","))))
