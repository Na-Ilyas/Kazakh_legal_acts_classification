"""Kazakh teachers for the compact students: KazBERT fine-tune + kaz-roberta distillation.

1. KazBERT (`Eraly-ml/KazBERT`, BERT-base, 110.7M params) fine-tuned with exactly the
   kaz-roberta recipe of src.roberta_tune (masked mean pooling + linear head):
     * form: survey recipe of kaz_roberta_ft (lr 5e-5, 3 epochs, CE, best val epoch);
     * area: the val-selected kaz-roberta recipe (LLRD 0.8, lr 1e-4, <= 5 epochs, early
       stopping on val, recency sample weights).
   Logged as `kazbert_ft` (family 'pretrained'), loadable with RobertaPredictor.
2. Knowledge distillation with the val-selectable kaz-roberta models as teachers
   (kaz_roberta_ft_form, kaz_roberta_llrd_rec_area). Teacher probabilities are computed
   on TRAIN titles only (cached in artifacts/cache/<teacher dir>/probs_train.npy); test is
   used for final scoring only. Loss = alpha * T^2 * KL(teacher_T || student_T) + (1-alpha) * CE.
   Students:
     a) tiny transformer 4L d256 from the MLM-pretrained encoder (best init with the e5
        teacher), alpha in {0.5, 0.9}, T = 2 -> tiny_transformer_4L_mlm_kdrob_a{50,90}_T2;
     b) the vocabulary-pruned e5-small (25.5M) -> e5_small_kkprune_kdrob, with the alpha
        selected on val for the tiny student of the same task.
Every run logs val / test / test_strict (+ test predictions) to SQLite through
src.model_utils.log_all_splits (shared src.metrics).

GPU commands must go through the shared lock, one short job each:
    flock .gpu.lock .venv/bin/python -m src.distill_roberta teacher
    flock .gpu.lock .venv/bin/python -m src.distill_roberta kazbert --task form
    flock .gpu.lock .venv/bin/python -m src.distill_roberta tiny --task area
    flock .gpu.lock .venv/bin/python -m src.distill_roberta e5 --task area
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from src import config  # noqa: F401  (pins GPU 0 before torch)
import torch

from src.model_utils import (E5_MODEL_NAME, E5_PREFIX, dir_size_mb, encode_labels, label_maps, load_json,
                             load_probs, load_task, log_all_splits, max_len_for, prediction_frame, save_json,
                             save_probs)
from src.trainer import TokenData, TrainConfig, fit, get_device, plot_learning_curve, predict_proba, set_seed

KAZBERT = "Eraly-ml/KazBERT"
KAZBERT_NAME = "kazbert_ft"
TEACHERS = {"form": "kaz_roberta_ft_form", "area": "kaz_roberta_llrd_rec_area"}   # val-selectable teachers
KAZBERT_TEACHERS = {"form": "kazbert_ft_form", "area": "kazbert_ft_area"}           # optional second teacher
ALPHAS, TEMP = (0.5, 0.9), 2.0
E5_KD_NAME = "e5_small_kkprune_kdrob"
SUMMARY_JSON = config.RESULTS_DIR / "distill_roberta.json"
SECTION_MARKER = "## Kazakh teachers: KazBERT fine-tune and kaz-roberta distillation (src.distill_roberta)"


# --------------------------------------------------------------------------
# Teacher probabilities (TRAIN titles only)
# --------------------------------------------------------------------------
def teacher_probs(tasks: tuple[str, ...] = ("form", "area"), batch_size: int = 256,
                  teachers: dict[str, str] | None = None) -> dict:
    """Cache teacher probabilities on TRAIN (and val/test if missing) for every task.

    teachers: task -> model dir (default: the kaz-roberta TEACHERS; KAZBERT_TEACHERS for the second teacher).
    """
    from src.roberta_tune import RobertaPredictor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    teachers = teachers or TEACHERS
    done = {}
    for task in tasks:
        tdir = teachers[task]
        data = load_task(task)
        todo = [s for s in ("train", "val", "test") if load_probs(tdir, s) is None]
        if not todo:
            done[task] = "cached"
            continue
        pred = RobertaPredictor(config.MODELS_DIR / tdir, device=device)
        for split in todo:
            titles = data[split]["title"].tolist()
            order = np.argsort([len(t) for t in titles])          # length-sorted batches: less padding
            t0 = time.time()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                probs_sorted = pred.predict_proba([titles[i] for i in order], batch_size=batch_size)
            probs = np.empty_like(probs_sorted)
            probs[order] = probs_sorted
            save_probs(tdir, split, probs, data[split]["law_id"], pred.classes)
            acc = float((np.array(pred.classes)[probs.argmax(1)] == data[split]["label"].to_numpy()).mean())
            print(f"[distill_roberta] teacher {tdir} {split}: {len(titles):,} titles in {time.time() - t0:.0f} s, "
                  f"accuracy {acc:.4f}", flush=True)
        done[task] = todo
        del pred
        torch.cuda.empty_cache()
    return done


def _aligned_train_probs(tdir: str, train_ids: list[str], classes: list[str]) -> np.ndarray:
    """Cached TRAIN probabilities of one teacher, aligned to the student's rows and class order."""
    cached = load_probs(tdir, "train")
    if cached is None:
        raise FileNotFoundError(f"teacher probabilities of {tdir} missing - run `python -m src.distill_roberta "
                                "teacher` first")
    probs, ids, t_classes = cached
    order = pd.Series(range(len(ids)), index=ids).reindex(train_ids)
    if order.isna().any():
        raise ValueError("teacher rows do not cover the training set")
    col = [t_classes.index(c) for c in classes]
    return probs[order.to_numpy().astype(int)][:, col]


def teacher_log_probs(task: str, train_ids: list[str], classes: list[str], device,
                      mode: str = "rob") -> torch.Tensor:
    """Teacher log-probabilities for the TRAIN rows: 'rob' = kaz-roberta, 'avg' = mean of kaz-roberta and KazBERT."""
    probs = _aligned_train_probs(TEACHERS[task], train_ids, classes)
    if mode == "avg":
        probs = 0.5 * (probs + _aligned_train_probs(KAZBERT_TEACHERS[task], train_ids, classes))
    logp = np.log(np.clip(probs, 1e-6, 1.0))
    return torch.from_numpy(logp.astype(np.float16)).to(device)


# --------------------------------------------------------------------------
# 1. KazBERT fine-tune (kaz-roberta recipe)
# --------------------------------------------------------------------------
def kazbert(task: str) -> dict:
    """Fine-tune KazBERT with the kaz-roberta recipe of src.roberta_tune and log it as kazbert_ft."""
    from src import roberta_tune as rt
    rt.BASE_MODEL = KAZBERT          # roberta_tune reads BASE_MODEL at call time (tokenizer, encoder, meta)
    ctx = rt.Context(task)
    base = TrainConfig(epochs=rt.EPOCHS, batch_size=64, lr=rt.LR, weight_decay=0.01, warmup_ratio=0.06, loss="ce",
                       patience=rt.PATIENCE, eval_batch_size=512, bucket=True)
    if task == "form":               # recipe of the val-selectable kaz_roberta_ft
        cfg, decay, rec = replace(base, epochs=3, lr=5e-5, patience=3), None, False
        note = "kaz_roberta_ft recipe: lr 5e-5, 3-epoch schedule, CE, no LLRD, best val epoch"
    else:                            # recipe of the val-selected kaz_roberta_llrd_rec
        cfg, decay, rec = base, rt.DECAY, True
        note = (f"kaz_roberta_llrd_rec recipe: LLRD {rt.DECAY}, lr {rt.LR}, <= {rt.EPOCHS} epochs, early stop on "
                f"val F1, recency weights (half-life {rt.REC_HALF_LIFE} y, floor {rt.REC_FLOOR}); the same decay "
                "over 12 layers gives the lower layers smaller LRs than in the 6-layer kaz-roberta")
    res = rt.train_variant(ctx, KAZBERT_NAME, cfg, decay, rec, note=note)
    _save_summary(f"kazbert_{task}", res)
    return res


# --------------------------------------------------------------------------
# 2a. Tiny transformer student (MLM init) with the kaz-roberta teacher
# --------------------------------------------------------------------------
def kd_tiny(task: str, alphas: tuple[float, ...] = ALPHAS, temp: float = TEMP, mode: str = "rob") -> list[dict]:
    """tiny_transformer_4L_mlm_kd{mode}_a{..}_T{..}: same student/config as the e5-teacher run, new teacher.

    mode 'rob' = kaz-roberta teacher; 'avg' = average of the kaz-roberta and KazBERT probabilities.
    """
    from src import bpe_tokenizer
    from src.distill import _run_variant, chosen_loss
    from src.mlm_pretrain import MLM_DIR
    from src.model_utils import BPE_DIR
    from src.tiny_transformer import encode_splits

    device = get_device()
    tok = bpe_tokenizer.load_tokenizer(BPE_DIR)
    max_len = max_len_for("bpe", 96)
    data = load_task(task)
    classes, label2id = label_maps(data["train"]["label"])
    tensors = encode_splits(data, tok, max_len, label2id, device)
    teacher = teacher_log_probs(task, data["train"]["law_id"].tolist(), classes, device, mode)
    rob = TEACHERS[task].rsplit("_", 1)[0]
    t_name = rob if mode == "rob" else f"mean({rob}, {KAZBERT_NAME})"
    loss = chosen_loss(task)
    rows = []
    for alpha in alphas:
        cfg = TrainConfig(epochs=12, patience=3, loss=loss, lr=3e-4, kd_alpha=alpha, kd_temperature=temp)
        sfx = f"_mlm_kd{mode}_a{int(alpha * 100)}_T{int(temp)}"
        rows.append(_run_variant(
            task, sfx, data, tensors, classes, cfg, max_len, tok.get_vocab_size(), False,
            {"kd_alpha": alpha, "kd_temperature": temp, "teacher": t_name,
             "init": "MLM-pretrained encoder", "lr": 3e-4, "loss_selected_on_val": loss},
            f"KD student from MLM-pretrained encoder: {t_name} teacher (probabilities on TRAIN titles), "
            f"alpha={alpha}, T={temp}",
            init_from=MLM_DIR / "encoder.pt", teacher=teacher))
    _save_summary(f"tiny_{mode}_{task}", rows)
    del tensors, teacher
    torch.cuda.empty_cache()
    return rows


def best_tiny_alpha(task: str) -> float:
    """alpha of the tiny kaz-roberta student with the best VAL macro-F1 (default 0.5)."""
    from src import experiments_db
    try:
        d = experiments_db.load_experiments(task)
    except Exception:
        return ALPHAS[0]
    d = d[d["model_name"].str.startswith("tiny_transformer_4L_mlm_kdrob_") & (d["eval_split"] == "val")]
    if d.empty:
        return ALPHAS[0]
    best = d.sort_values("f1_macro", ascending=False).iloc[0]["model_name"]
    return int(best.split("_a")[-1].split("_")[0]) / 100


# --------------------------------------------------------------------------
# 2b. Vocabulary-pruned e5-small student with the kaz-roberta teacher
# --------------------------------------------------------------------------
def kd_e5_pruned(task: str, alpha: float | None = None, temp: float = TEMP, epochs: int = 3) -> dict:
    """Same pruning / recipe as src.vocab_prune.run_task (e5_small_kkprune_ft) + kaz-roberta KD loss."""
    from src.finetune_e5 import E5Classifier, encode_e5, load_e5
    from src.vocab_prune import build_lookup, count_params, cpu_latency_ms, kept_token_ids, oov_rate, prune_encoder

    alpha = best_tiny_alpha(task) if alpha is None else alpha
    data = load_task(task)
    classes, label2id = label_maps(data["train"]["label"])
    max_len = max_len_for("e5", 112)
    hf_tok, encoder = load_e5()
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
    tensors = {s: TokenData.build(lookup[ids], lens, encode_labels(data[s]["label"], label2id), device)
               for s, (ids, lens) in ids_old.items()}
    del ids_old
    teacher = teacher_log_probs(task, data["train"]["law_id"].tolist(), classes, device)
    cfg = TrainConfig(epochs=epochs, batch_size=64, lr=5e-5, weight_decay=0.01, warmup_ratio=0.06, loss="ce",
                      patience=epochs, eval_batch_size=512, bucket=True, kd_alpha=alpha, kd_temperature=temp)
    run_name = f"{E5_KD_NAME}_{task}"
    out_dir = config.MODELS_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    res = fit(model, tensors["train"], tensors["val"], len(classes), cfg, out_dir / "model.pt", run_name,
              log_csv=config.RESULTS_DIR / f"train_log_{run_name}.csv", teacher_logp=teacher)
    probs = {s: predict_proba(model, tensors[s], 512) for s in ("val", "test")}
    preds = {s: prediction_frame(data[s]["law_id"], data[s]["label"], probs[s], classes) for s in ("val", "test")}
    # same artefacts as e5_small_kkprune_ft + tokenizer and meta.json, so RobertaPredictor / predict can load it
    model.encoder.save_pretrained(out_dir / "encoder")
    hf_tok.save_pretrained(out_dir / "encoder")
    torch.save(model.head.state_dict(), out_dir / "head.pt")
    np.save(out_dir / "id_lookup.npy", lookup)
    (out_dir / "model.pt").unlink(missing_ok=True)
    save_json(label2id, out_dir / "label2id.json")
    save_json({"task": task, "base_model": E5_MODEL_NAME, "tokenizer": E5_MODEL_NAME, "max_len": max_len,
               "prefix": E5_PREFIX, "kept_vocab": int(len(kept)), "unk_new": unk_new, "pad_new": pad_new},
              out_dir / "config.json")
    save_json({"model_type": "hf_encoder_meanpool_linear", "loader": "src.roberta_tune.RobertaPredictor",
               "model_name": E5_KD_NAME, "task": task, "base_model": E5_MODEL_NAME, "classes": classes,
               "max_len": max_len, "prefix": E5_PREFIX, "pooling": "masked mean", "id_lookup": "id_lookup.npy",
               "best_epoch": res["best_epoch"], "best_val_f1": res["best_val_f1"], "kd_alpha": alpha,
               "kd_temperature": temp, "teacher": TEACHERS[task]}, out_dir / "meta.json")
    for s in ("val", "test"):
        save_probs(run_name, s, probs[s], data[s]["law_id"], classes)
    plot_learning_curve(res["history"], f"e5-small pruned, kaz-roberta KD - task '{task}'",
                        config.FIGURES_DIR / f"learning_curve_{run_name}.png")
    size_mb = dir_size_mb(out_dir)
    latency = cpu_latency_ms(model, hf_tok, lookup, max_len)
    del tensors, teacher
    torch.cuda.empty_cache()
    common = {"n_params": n_params, "size_mb": round(size_mb, 2), "latency_ms_per_1000": round(latency, 1),
              "latency_device": "cpu", "peak_mem_mb": res["train_peak_gpu_mb"], "train_time_s": res["train_time_s"],
              "n_train": len(data["train"])}
    hp = {"base_model": E5_MODEL_NAME, "vocab_full": old_vocab, "vocab_kept": int(len(kept)), "oov_pct": oov,
          "max_len": max_len, "best_epoch": res["best_epoch"], "epochs_run": res["epochs_run"], **res["train_config"],
          "teacher": TEACHERS[task], "alpha_selected_from": "val of tiny_transformer_4L_mlm_kdrob_* (same task)",
          "peak_mem_mb_meaning": "training peak CUDA memory",
          "latency_protocol": "1000 seeded test titles, CPU 8 threads, batch 64, tokenise+remap+forward, median of 3"}
    notes = (f"e5_small_kkprune_ft recipe (pruned e5-small, lr 5e-5, {epochs} epochs, best val epoch) + KD from "
             f"{TEACHERS[task].rsplit('_', 1)[0]} (kaz-roberta probabilities on TRAIN titles), alpha={alpha}, T={temp}; "
             "alpha taken from the val-best tiny kaz-roberta student")
    logged = log_all_splits(E5_KD_NAME, "pretrained", task, preds, common, hp, notes=notes)
    f1 = {s: round(v["f1_macro"], 4) for s, v in logged.items()}
    out = {"name": E5_KD_NAME, "task": task, "alpha": alpha, "f1_macro": f1, "n_params": n_params,
           "size_mb": round(size_mb, 1), "latency_cpu_ms_per_1000": round(latency, 1),
           "best_epoch": res["best_epoch"], "train_time_s": res["train_time_s"]}
    print(f"[distill_roberta] {out}", flush=True)
    _save_summary(f"e5kd_{task}", out)
    return out


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def _save_summary(key: str, value) -> None:
    """Merge one result into results/distill_roberta.json."""
    try:
        allres = load_json(SUMMARY_JSON) if SUMMARY_JSON.exists() else {}
    except (OSError, json.JSONDecodeError):
        allres = {}
    allres[key] = value
    save_json(allres, SUMMARY_JSON)


def write_survey_section() -> str:
    """Fill the KazBERT row of results/model_survey.md and insert a section with this module's runs."""
    from src import experiments_db
    from src.model_survey import SURVEY_MD, existing_f1
    try:
        d = experiments_db.load_experiments()
    except Exception as exc:
        print(f"[distill_roberta] survey skipped: {exc}")
        return ""
    names = ("kazbert_ft", "e5_small_kkprune_kdrob", "tiny_transformer_4L_mlm_kdrob_", "tiny_transformer_4L_mlm_kdavg_", "kaz_roberta_ft",
             "kaz_roberta_llrd_rec", "e5_small_kkprune_ft", "tiny_transformer_4L_mlm_kd_")
    d = d[d["model_name"].apply(lambda n: any(n == k or (k.endswith("_") and n.startswith(k)) for k in names))]
    piv = d.pivot_table(index=["task", "model_name"], columns="eval_split", values="f1_macro", aggfunc="first")
    meta = d.groupby(["task", "model_name"]).agg(n_params=("n_params", "first"), size_mb=("size_mb", "first"),
                                                 lat=("latency_ms_per_1000", "first"))
    piv = piv.join(meta).reset_index().sort_values(["task", "val"], ascending=[True, False])
    lines = [SECTION_MARKER, "",
             "KazBERT fine-tuned with the kaz-roberta recipe (form: lr 5e-5, 3 epochs; area: LLRD 0.8 + recency "
             "weights). Distillation students use teacher probabilities on TRAIN titles from the val-selectable "
             "kaz-roberta models (`kaz_roberta_ft` form, `kaz_roberta_llrd_rec` area); `kd_` rows (e5 teacher) are "
             "shown for reference. Macro-F1 from SQLite; selection on val only. CPU ms per 1000 titles "
             "(8 threads, busy shared machine).", "",
             "| task | model | val | test | test_strict | params | MB | CPU ms/1000 |", "|---|---|---|---|---|---|---|---|"]
    for r in piv.to_dict("records"):
        lines.append(f"| {r['task']} | {r['model_name']} | {r.get('val', float('nan')):.4f} | "
                     f"{r.get('test', float('nan')):.4f} | {r.get('test_strict', float('nan')):.4f} | "
                     f"{r['n_params'] / 1e6:.1f}M | {r['size_mb']:.0f} | {r['lat']:.0f} |")
    section = "\n".join(lines) + "\n"
    try:
        text = SURVEY_MD.read_text(encoding="utf-8") if SURVEY_MD.exists() else ""
        row_start = "| Eraly-ml/KazBERT |"
        out_lines = []
        for line in text.split("\n"):
            if line.startswith(row_start):    # fill the two F1 columns of the survey table
                cells = line.split(" | ")
                cells[-2] = existing_f1(KAZBERT_NAME, "form")
                cells[-1] = existing_f1(KAZBERT_NAME, "area") + " |"
                line = " | ".join(cells)
            out_lines.append(line)
        text = "\n".join(out_lines)
        if SECTION_MARKER in text:            # replace our old section (it ends at the next '## ' heading)
            head, rest = text.split(SECTION_MARKER, 1)
            nxt = rest.find("\n## ")
            text = head.rstrip() + "\n\n" + section + ("\n" + rest[nxt + 1:] if nxt >= 0 else "")
        else:                                 # insert before roberta_tune's section (it rewrites the file tail)
            marker = "## kaz-roberta improvements (src.roberta_tune)"
            if marker in text:
                head, tail = text.split(marker, 1)
                text = head.rstrip() + "\n\n" + section + "\n" + marker + tail
            else:
                text = text.rstrip() + "\n\n" + section
        SURVEY_MD.write_text(text, encoding="utf-8")
    except OSError as exc:
        print(f"[distill_roberta] could not write {SURVEY_MD}: {exc}")
    return section


def run(tasks: tuple[str, ...] = ("form", "area"), kazbert_ft: bool = True, tiny: bool = True,
        e5: bool = True, quick: bool = False) -> dict:
    """Pipeline entry: teacher probabilities, KazBERT, tiny + pruned-e5 students, survey section.

    Needs the full-run kaz-roberta teachers, so a quick (smoke) pipeline run skips it.
    """
    if quick:
        return {"skipped": "quick (needs the full-run kaz-roberta teachers)"}
    out: dict = {"teacher": teacher_probs(tasks)}
    for task in tasks:
        if kazbert_ft:
            out[f"kazbert_{task}"] = kazbert(task)["f1_macro"]
        if tiny:
            out[f"tiny_{task}"] = {r["model_name"]: r["val_f1"] for r in kd_tiny(task)}
        if e5:
            out[f"e5_{task}"] = kd_e5_pruned(task)["f1_macro"]
    write_survey_section()
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KazBERT fine-tune + kaz-roberta distillation")
    parser.add_argument("cmd", choices=["teacher", "kazbert", "tiny", "e5", "survey", "all"])
    parser.add_argument("--task", default="form,area")
    parser.add_argument("--alphas", default=",".join(str(a) for a in ALPHAS))
    parser.add_argument("--alpha", type=float, default=None, help="e5 student alpha (default: val-best tiny)")
    parser.add_argument("--mode", default="rob", choices=["rob", "avg"], help="tiny: teacher (avg = + KazBERT)")
    parser.add_argument("--kazbert-teacher", action="store_true", help="teacher: cache KazBERT probabilities")
    a = parser.parse_args()
    tasks = tuple(a.task.split(","))
    if a.cmd == "teacher":
        print(teacher_probs(tasks, teachers=KAZBERT_TEACHERS if a.kazbert_teacher else None))
    elif a.cmd == "kazbert":
        for t in tasks:
            print(kazbert(t))
    elif a.cmd == "tiny":
        for t in tasks:
            print(kd_tiny(t, tuple(float(x) for x in a.alphas.split(",")), mode=a.mode))
    elif a.cmd == "e5":
        for t in tasks:
            print(kd_e5_pruned(t, a.alpha))
    elif a.cmd == "survey":
        print(write_survey_section())
    else:
        print(json.dumps(run(tasks), ensure_ascii=False, indent=1, default=str))
