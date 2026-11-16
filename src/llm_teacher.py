"""LLM as a TEACHER: Qwen3-1.7B-Base + LoRA classifier -> soft labels -> KD into the tiny transformer.

Question answered here: can a (Kazakh-capable) LLM be distilled into our small models?
The deployed model stays small (tiny transformer, 5.2M params); the LLM only produces
soft labels on TRAIN titles. The LLM itself is far above the 120M budget and is logged
(family 'llm_teacher') only so that the teacher quality is auditable.

Why Qwen3-1.7B-Base and not Sherkala-8B: `inception42/Llama-3.1-Sherkala-8B-Chat` (and
`issai/LLama-3.1-KazLLM-1.0-8B`) are gated on the HF Hub (HTTP 401 without an accepted
licence + token, and there is no token on this machine). Qwen3 is open, lists Kazakh
among its 119 pre-training languages, and the 1.7B size fits the time box (a LoRA epoch
on tens of thousands of titles in minutes on one GPU, shared with other jobs).

Stages (each one is a separate short GPU job so that other agents can interleave):
    train    LoRA (r=16, attention projections) + linear head on a stratified TRAIN subsample,
             last-token pooling; checkpoints scored on VAL macro-F1, best adapter kept.
    infer    teacher probabilities for val/test (+ SQLite logging of the teacher) or train
             (cached in artifacts/cache/<tag>_<task>/probs_<split>.npy, the format of
             src.model_utils.save_probs that src.distill reads).
    student  KD into the 4L d=256 tiny transformer initialised from the MLM-pretrained
             encoder (alpha in {0.5, 0.9}, T=2; selection on val), logged as
             tiny_transformer_4L_mlm_kdllm_a{..}_T{..}.

Result of the reported run (results/llm_teacher.json, SQLite family 'llm_teacher'): val macro-F1
form 0.754 (40k titles, 1 epoch, 5 min), area 0.549 (81k titles, 1 epoch, 9 min) - below
kaz-roberta (0.883 / 0.616) and even below e5-small (0.866 / 0.584), the existing KD teacher.
Since a teacher must be at least as good as kaz-roberta on val, the KD stages were NOT run.

Usage (always under the shared GPU lock):
    flock .gpu.lock .venv/bin/python -m src.llm_teacher --stage train --task form
    flock .gpu.lock .venv/bin/python -m src.llm_teacher --stage infer --task form --splits val,test
    flock .gpu.lock .venv/bin/python -m src.llm_teacher --stage infer --task form --splits train
    flock .gpu.lock .venv/bin/python -m src.llm_teacher --stage student --task form
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src import config  # noqa: F401  (pins GPU 0 before torch)
import torch

from src.model_utils import (display_path, dir_size_mb, label_maps, load_json, load_probs, load_task,
                             log_all_splits, peak_rss_mb, prediction_frame, save_json, save_probs,
                             compute_metrics)

BASE_MODEL = "Qwen/Qwen3-1.7B-Base"
TAG = "llm_teacher_qwen3_1p7b"           # model dir / prob-cache / DB name stem
FAMILY = "llm_teacher"
MAX_LEN = 128                            # Qwen tokens; median title = 95, p95 = 207 -> head+tail truncation
N_TRAIN_SUB = 40_000                     # stratified TRAIN subsample for the LoRA teacher (time box)
# form: 40k = one epoch in ~5 min; area (33 classes, harder): 80k = one epoch inside the 540 s budget
N_SUB_BY_TASK = {"form": 40_000, "area": 80_000}
MIN_PER_CLASS = 400                      # floor per class in the subsample (rare classes stay learnable)
LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05
LR, BATCH, EVAL_BATCH = 2e-4, 32, 128
N_EVALS = 3                              # val checkpoints per epoch (best one is kept)
TIME_BUDGET_S = 540                      # training-loop budget per locked job (keeps the job < 12 min)
KD_ALPHAS, KD_T = (0.5, 0.9), 2.0
SUMMARY_JSON = config.RESULTS_DIR / "llm_teacher.json"


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def model_dir(task: str) -> Path:
    """artifacts/models/<tag>_<task>: LoRA adapter + head + meta."""
    return config.MODELS_DIR / f"{TAG}_{task}"


def cache_name(task: str) -> str:
    """Name of the probability cache dir (artifacts/cache/<name>)."""
    return f"{TAG}_{task}"


def stratified_subsample(df: pd.DataFrame, n: int, floor: int, seed: int = config.SEED) -> pd.DataFrame:
    """Proportional per-class sample of ~n rows with at least min(count, floor) rows per class."""
    frac = n / len(df)
    parts = []
    for _, grp in df.groupby("label"):
        k = min(len(grp), max(int(round(len(grp) * frac)), floor))
        parts.append(grp.sample(n=k, random_state=seed))
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def load_tokenizer():
    """Qwen tokenizer; pad with an unused special token so that EOS can close every title."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    pad_id = tok.convert_tokens_to_ids("<|fim_pad|>")
    return tok, int(pad_id), int(tok.eos_token_id)


def tokenize(tok, titles: list[str], eos_id: int, max_len: int = MAX_LEN,
             chunk: int = 20_000) -> tuple[np.ndarray, np.ndarray]:
    """[N, max_len] int32 ids (+ lengths). Long titles keep head AND tail (issuer at the start,
    the amended act's form at the end); every title ends with EOS (the pooled position)."""
    body = max_len - 1
    head = body // 2
    ids = np.zeros((len(titles), max_len), dtype=np.int32)
    lengths = np.zeros(len(titles), dtype=np.int32)
    for start in range(0, len(titles), chunk):
        enc = tok(titles[start:start + chunk], add_special_tokens=False)["input_ids"]
        for j, seq in enumerate(enc):
            if len(seq) > body:
                seq = seq[:head] + seq[-(body - head):]
            seq = seq + [eos_id]
            ids[start + j, :len(seq)] = seq
            lengths[start + j] = len(seq)
    return ids, lengths


def batch_tensors(ids: np.ndarray, lengths: np.ndarray, idx: np.ndarray, pad_id: int,
                  device) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-padded (input_ids, attention_mask) for the rows idx, trimmed to their max length."""
    width = int(lengths[idx].max())
    x = ids[idx, :width].copy()
    mask = np.arange(width)[None, :] < lengths[idx][:, None]
    x[~mask] = pad_id
    return (torch.from_numpy(x.astype(np.int64)).to(device),
            torch.from_numpy(mask.astype(np.int64)).to(device))


def bucketed_batches(lengths: np.ndarray, bs: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Shuffled batches of similar length (less padding -> faster)."""
    perm = rng.permutation(len(lengths))
    batches = []
    for start in range(0, len(perm), bs * 50):
        chunk = perm[start:start + bs * 50]
        chunk = chunk[np.argsort(lengths[chunk], kind="stable")]
        batches += [chunk[i:i + bs] for i in range(0, len(chunk), bs)]
    rng.shuffle(batches)
    return batches


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
def load_model(n_classes: int, pad_id: int, adapter_dir: Path | None = None):
    """bf16 base loaded straight onto GPU 0 (never materialised in CPU RAM) + LoRA.

    adapter_dir=None -> fresh trainable LoRA; else the saved adapter + head for inference.
    """
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForSequenceClassification

    base = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=n_classes, dtype=torch.bfloat16, device_map={"": 0}, low_cpu_mem_usage=True)
    base.config.pad_token_id = pad_id          # last non-pad token (= EOS) is pooled
    if adapter_dir is not None:
        model = PeftModel.from_pretrained(base, str(adapter_dir))
        return model.eval()
    lcfg = LoraConfig(task_type="SEQ_CLS", r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    model = get_peft_model(base, lcfg)
    for p in model.parameters():               # trainable params in fp32 (bf16 Adam updates are lossy)
        if p.requires_grad:
            p.data = p.data.float()
    return model


def n_params(model) -> tuple[int, int]:
    """(total, trainable) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, train


@torch.no_grad()
def predict_proba(model, ids: np.ndarray, lengths: np.ndarray, pad_id: int, bs: int = EVAL_BATCH) -> np.ndarray:
    """Softmax probabilities (float32 [N, C]); rows processed in length order."""
    model.eval()
    device = torch.device("cuda")
    order = np.argsort(lengths, kind="stable")
    out = None
    for start in range(0, len(order), bs):
        idx = order[start:start + bs]
        x, m = batch_tensors(ids, lengths, idx, pad_id, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=x, attention_mask=m).logits.float()
        p = torch.softmax(logits, dim=-1).cpu().numpy()
        if out is None:
            out = np.zeros((len(order), p.shape[1]), dtype=np.float32)
        out[idx] = p
    return out


# --------------------------------------------------------------------------
# Stage 1: LoRA teacher training (selection on VAL)
# --------------------------------------------------------------------------
def train_teacher(task: str, n_sub: int = N_TRAIN_SUB, budget_s: float = TIME_BUDGET_S) -> dict:
    """Train the LoRA teacher on a stratified TRAIN subsample; keep the best checkpoint on val."""
    t_start = time.time()
    data = load_task(task)
    classes, label2id = label_maps(data["train"]["label"])
    sub = stratified_subsample(data["train"], n_sub, MIN_PER_CLASS)
    tok, pad_id, eos_id = load_tokenizer()
    tr_ids, tr_len = tokenize(tok, sub["title"].tolist(), eos_id)
    va_ids, va_len = tokenize(tok, data["val"]["title"].tolist(), eos_id)
    y = torch.tensor([label2id[l] for l in sub["label"]], dtype=torch.long)
    print(f"[llm] {task}: subsample {len(sub):,} of {len(data['train']):,} train rows, {len(classes)} classes, "
          f"mean len {tr_len.mean():.1f}, truncated {(tr_len == MAX_LEN).mean():.1%}", flush=True)

    torch.manual_seed(config.SEED)
    torch.cuda.reset_peak_memory_stats()
    model = load_model(len(classes), pad_id)
    total, trainable = n_params(model)
    print(f"[llm] params total {total:,}, trainable {trainable:,}; load {time.time() - t_start:.0f}s", flush=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.0)
    batches = bucketed_batches(tr_len, BATCH, np.random.default_rng(config.SEED))
    n_steps = len(batches)
    warm = max(1, int(0.05 * n_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (n_steps - s) / max(1, n_steps - warm)))
    eval_at = {int(n_steps * k / N_EVALS) - 1 for k in range(1, N_EVALS + 1)}
    out_dir = model_dir(task)
    out_dir.mkdir(parents=True, exist_ok=True)
    history, best = [], {"val_f1": -1.0}
    device = torch.device("cuda")
    t_train = time.time()
    model.train()
    for step, idx in enumerate(batches):
        x, m = batch_tensors(tr_ids, tr_len, idx, pad_id, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=x, attention_mask=m).logits.float()
        loss = torch.nn.functional.cross_entropy(logits, y[idx].to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        if step % 100 == 0:
            el = time.time() - t_train
            print(f"[llm] step {step}/{n_steps} loss {loss.item():.4f} {el:.0f}s "
                  f"(eta epoch {el / (step + 1) * n_steps:.0f}s)", flush=True)
        over_budget = time.time() - t_train > budget_s
        if step in eval_at or over_budget:
            probs = predict_proba(model, va_ids, va_len, pad_id)
            f1 = compute_metrics(data["val"]["label"], [classes[i] for i in probs.argmax(1)])["f1_macro"]
            row = {"step": step + 1, "samples_seen": (step + 1) * BATCH, "val_f1": round(f1, 4),
                   "elapsed_s": round(time.time() - t_train, 1)}
            history.append(row)
            print(f"[llm] eval {row}", flush=True)
            if f1 > best["val_f1"]:
                best = row
                model.save_pretrained(str(out_dir))
            model.train()
        if over_budget:
            print(f"[llm] time budget {budget_s}s reached at step {step + 1}/{n_steps}", flush=True)
            break
    meta = {"task": task, "base_model": BASE_MODEL, "classes": classes, "max_len": MAX_LEN, "pad_id": pad_id,
            "eos_id": eos_id, "n_sub": len(sub), "n_train_full": len(data["train"]), "min_per_class": MIN_PER_CLASS,
            "lora": {"r": LORA_R, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT,
                     "targets": ["q_proj", "k_proj", "v_proj", "o_proj"]},
            "lr": LR, "batch": BATCH, "steps_planned": n_steps, "history": history, "best": best,
            "n_params_total": total, "n_params_trainable": trainable,
            "train_time_s": round(time.time() - t_train, 1),
            "train_peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1),
            "peak_rss_mb": round(peak_rss_mb(), 1), "pooling": "last token (EOS), head+tail truncation"}
    save_json(meta, out_dir / "teacher_meta.json")
    print(f"[llm] best on val: {best}; peak GPU {meta['train_peak_gpu_mb']} MB", flush=True)
    return {"task": task, "best": best, "train_time_s": meta["train_time_s"]}


# --------------------------------------------------------------------------
# Stage 2: teacher inference -> prob cache (+ SQLite logging of the teacher)
# --------------------------------------------------------------------------
def infer_teacher(task: str, splits: tuple[str, ...]) -> dict:
    """Cache teacher probabilities for `splits`; log the teacher once val+test are available."""
    out_dir = model_dir(task)
    meta = load_json(out_dir / "teacher_meta.json")
    classes = meta["classes"]
    data = load_task(task)
    tok, pad_id, eos_id = load_tokenizer()
    torch.cuda.reset_peak_memory_stats()
    model = load_model(len(classes), pad_id, adapter_dir=out_dir)
    timings = meta.get("infer_s", {})
    for split in splits:
        t0 = time.time()
        ids, lens = tokenize(tok, data[split]["title"].tolist(), eos_id)
        probs = predict_proba(model, ids, lens, pad_id)
        save_probs(cache_name(task), split, probs, data[split]["law_id"], classes)
        timings[split] = round(time.time() - t0, 1)
        if split != "train":
            f1 = compute_metrics(data[split]["label"], [classes[i] for i in probs.argmax(1)])["f1_macro"]
            print(f"[llm] {task} {split}: macro-F1 {f1:.4f} ({timings[split]}s)", flush=True)
        else:
            print(f"[llm] {task} train probs cached: {len(probs):,} rows ({timings[split]}s)", flush=True)
    meta["infer_s"] = timings
    if "test" in splits:
        meta["latency_cuda_ms_per_1000"] = _latency(model, tok, data["test"]["title"].tolist()[:1000],
                                                    pad_id, eos_id)
    meta["infer_peak_gpu_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    save_json(meta, out_dir / "teacher_meta.json")
    del model
    torch.cuda.empty_cache()
    if load_probs(cache_name(task), "val") is not None and load_probs(cache_name(task), "test") is not None:
        return log_teacher(task)
    return {"task": task, "cached": list(splits)}


def _latency(model, tok, titles: list[str], pad_id: int, eos_id: int) -> float:
    """GPU ms per 1000 titles (tokenisation included, batch 128, after one warm-up pass)."""
    for rep in range(2):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        ids, lens = tokenize(tok, titles, eos_id)
        predict_proba(model, ids, lens, pad_id)
        torch.cuda.synchronize()
        el = (time.perf_counter() - t0) * 1000 * 1000 / len(titles)
    return round(el, 1)


def log_teacher(task: str) -> dict:
    """Log the teacher's val / test / test_strict (+ test predictions) from the prob cache."""
    meta = load_json(model_dir(task) / "teacher_meta.json")
    data = load_task(task)
    preds = {}
    for split in ("val", "test"):
        probs, ids, classes = load_probs(cache_name(task), split)
        frame = data[split].set_index("law_id").loc[ids]
        preds[split] = prediction_frame(ids, frame["label"], probs, classes)
    base_mb = dir_size_mb(_base_snapshot()) if _base_snapshot() else None
    common = {"n_params": meta["n_params_total"],
              "size_mb": round((base_mb or 0) + dir_size_mb(model_dir(task)), 1),
              "latency_ms_per_1000": meta.get("latency_cuda_ms_per_1000"), "latency_device": "cuda",
              "peak_mem_mb": meta["train_peak_gpu_mb"], "train_time_s": meta["train_time_s"],
              "n_train": meta["n_sub"]}
    hp = {k: meta[k] for k in ("base_model", "max_len", "n_sub", "min_per_class", "lora", "lr", "batch",
                               "best", "n_params_trainable", "pooling", "infer_s", "infer_peak_gpu_mb")
          if k in meta}
    hp["model_dir"] = display_path(model_dir(task))
    hp["peak_mem_mb_meaning"] = "training peak CUDA memory"
    logged = log_all_splits(TAG, FAMILY, task, preds, common, hp,
                            notes="TEACHER ONLY (exceeds the 120M budget): Qwen3-1.7B-Base + LoRA r16 "
                                  "seq-cls head on a stratified train subsample; soft labels for KD")
    f1s = {s: round(v["f1_macro"], 4) for s, v in logged.items()}
    _update_summary(task, {"teacher": f1s, "teacher_meta": {k: meta.get(k) for k in (
        "n_sub", "train_time_s", "train_peak_gpu_mb", "infer_s", "infer_peak_gpu_mb",
        "latency_cuda_ms_per_1000", "n_params_total", "n_params_trainable", "best")}})
    print(f"[llm] teacher logged {task}: {f1s}", flush=True)
    return {"task": task, "teacher": f1s}


REFERENCE = {"form": "kaz_roberta_ft_form", "area": "kaz_roberta_llrd_rec_area"}   # kaz-roberta prob caches


def per_class_val(task: str) -> pd.DataFrame:
    """Per-class VAL F1 of the LLM teacher vs the kaz-roberta reference (CPU, from prob caches)."""
    from sklearn.metrics import f1_score

    data = load_task(task)
    truth = data["val"].set_index("law_id")["label"]
    rows = {}
    for name in (cache_name(task), REFERENCE[task]):
        cached = load_probs(name, "val")
        if cached is None:
            continue
        probs, ids, classes = cached
        y_true = truth.loc[ids].to_numpy()
        y_pred = np.array(classes)[probs.argmax(1)]
        labels = sorted(set(y_true))
        rows[name] = pd.Series(f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0),
                               index=labels)
        rows.setdefault("n_val", pd.Series(y_true).value_counts().reindex(labels))
    df = pd.DataFrame(rows)
    df.to_csv(config.RESULTS_DIR / f"llm_teacher_per_class_val_{task}.csv")
    return df


def _base_snapshot() -> Path | None:
    """Local HF snapshot dir of the base model (for the size_mb column)."""
    try:
        from huggingface_hub import try_to_load_from_cache
        cfg = try_to_load_from_cache(BASE_MODEL, "config.json")
        return Path(cfg).parent if isinstance(cfg, str) else None
    except Exception:            # noqa: BLE001 - size is informative only
        return None


def _update_summary(task: str, payload: dict) -> None:
    """Merge one task's results into results/llm_teacher.json."""
    try:
        summary = load_json(SUMMARY_JSON)
    except (FileNotFoundError, ValueError):
        summary = {}
    summary.setdefault(task, {}).update(payload)
    save_json(summary, SUMMARY_JSON)


# --------------------------------------------------------------------------
# Stage 3: KD into the tiny transformer (MLM-pretrained init)
# --------------------------------------------------------------------------
def teacher_log_probs(task: str, train_ids: list[str], classes: list[str], device) -> torch.Tensor:
    """LLM-teacher log-probs for the TRAIN rows, aligned to the student's rows/classes."""
    cached = load_probs(cache_name(task), "train")
    if cached is None:
        raise FileNotFoundError(f"LLM teacher train probabilities missing - run --stage infer --splits train")
    probs, ids, t_classes = cached
    order = pd.Series(range(len(ids)), index=ids).reindex(train_ids)
    if order.isna().any():
        raise ValueError("teacher rows do not cover the training set")
    probs = probs[order.to_numpy().astype(int)][:, [t_classes.index(c) for c in classes]]
    return torch.from_numpy(np.log(np.clip(probs, 1e-6, 1.0)).astype(np.float16)).to(device)


def distill_student(task: str, alphas: tuple[float, ...] = KD_ALPHAS, temp: float = KD_T) -> dict:
    """KD grid into the MLM-initialised 4L tiny transformer; selection on val."""
    from src import bpe_tokenizer
    from src.distill import _run_variant, chosen_loss
    from src.mlm_pretrain import MLM_DIR
    from src.model_utils import BPE_DIR, max_len_for
    from src.tiny_transformer import encode_splits
    from src.trainer import TrainConfig, get_device

    device = get_device()
    tok = bpe_tokenizer.load_tokenizer(BPE_DIR)
    max_len = max_len_for("bpe", 96)
    data = load_task(task)
    classes, label2id = label_maps(data["train"]["label"])
    tensors = encode_splits(data, tok, max_len, label2id, device)
    teacher = teacher_log_probs(task, data["train"]["law_id"].tolist(), classes, device)
    loss = chosen_loss(task)
    enc = MLM_DIR / "encoder.pt"
    rows = []
    for alpha in alphas:
        cfg = TrainConfig(epochs=12, patience=3, loss=loss, lr=3e-4, kd_alpha=alpha, kd_temperature=temp)
        sfx = f"_mlm_kdllm_a{int(alpha * 100)}_T{int(temp)}"
        rows.append(_run_variant(
            task, sfx, data, tensors, classes, cfg, max_len, tok.get_vocab_size(), False,
            {"kd_alpha": alpha, "kd_temperature": temp, "teacher": TAG, "init": "MLM-pretrained encoder",
             "lr": 3e-4, "loss_selected_on_val": loss},
            f"KD student from MLM-pretrained encoder: LLM teacher {BASE_MODEL}+LoRA ({TAG}), "
            f"alpha={alpha}, T={temp}", init_from=enc, teacher=teacher))
    del teacher, tensors
    torch.cuda.empty_cache()
    best = max(rows, key=lambda r: r["val_f1"])
    _update_summary(task, {"students": rows, "best_student_on_val": best["model_name"]})
    return {"task": task, "rows": rows, "best_on_val": best["model_name"]}


def run(tasks: tuple[str, ...] = ("form", "area"), stage: str = "teacher", quick: bool = False) -> dict:
    """Pipeline entry.

    stage='teacher' (default, what the reported run did): LoRA training + val/test inference and
    logging. The teacher lost to kaz-roberta (and to e5-small) on VAL for both tasks, so the
    KD stages were not run; stage='all' adds train-title inference + the KD students.
    Needs ~4 GB of Qwen weights and ~13 GPU-minutes per task for 'teacher', so --quick skips it.
    """
    if quick:
        return {"skipped": "quick (LLM teacher is GPU-heavy)"}
    out = {}
    for task in tasks:
        if stage in ("teacher", "all"):
            out[f"{task}_train"] = train_teacher(task, n_sub=N_SUB_BY_TASK.get(task, N_TRAIN_SUB))
            out[f"{task}_infer"] = infer_teacher(task, ("val", "test"))
        if stage == "all":
            out[f"{task}_infer_train"] = infer_teacher(task, ("train",))
            out[f"{task}_student"] = distill_student(task)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM teacher (Qwen3-1.7B + LoRA) and KD into the tiny model")
    parser.add_argument("--stage", choices=["train", "infer", "student", "log", "perclass", "all"], default="all")
    parser.add_argument("--task", default="form")
    parser.add_argument("--splits", default="val,test", help="infer stage: comma list of splits")
    parser.add_argument("--n-sub", type=int, default=N_TRAIN_SUB)
    parser.add_argument("--budget", type=float, default=TIME_BUDGET_S, help="train-loop seconds")
    parser.add_argument("--alphas", default="0.5,0.9")
    args = parser.parse_args()
    if args.stage == "train":
        print(train_teacher(args.task, n_sub=args.n_sub, budget_s=args.budget))
    elif args.stage == "infer":
        print(infer_teacher(args.task, tuple(args.splits.split(","))))
    elif args.stage == "log":
        print(log_teacher(args.task))
    elif args.stage == "perclass":
        print(per_class_val(args.task).round(3).to_string())
    elif args.stage == "student":
        print(distill_student(args.task, tuple(float(a) for a in args.alphas.split(","))))
    else:
        print(run((args.task,), "all"))      # full chain incl. KD students (not run for the report)
