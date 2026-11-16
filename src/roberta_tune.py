"""Val-selected improvements of the fine-tuned Kazakh RoBERTa (best pretrained model so far).

Base: kz-transformers/kaz-roberta-conversational (6 layers, 83.5M params), masked mean
pooling + linear head (same architecture as src.model_survey.finetune / e5_small_ft).
Cheap ideas, each trained as its own variant and SELECTED ON VAL macro-F1 only:

* ep5    - survey recipe (lr 5e-5, no LLRD) but up to 5 epochs with early stopping (patience 2)
* llrd   - layer-wise LR decay: head lr = LR, layer i lr = LR * DECAY^(L - i),
           embeddings LR * DECAY^(L + 1); 5 epochs with early stopping (patience 2)
* cw     - + class-weighted CE (1/sqrt(freq)) for the long-tail area classes
* ls     - + label-smoothed CE (0.1)
* rec    - + recency sample weights (half-life 6 years before 2021, floor 0.1; the
           test years >= 2023 have a strongly shifted label distribution)
* combo  - llrd + every component that beat plain llrd on val
* *_trainval - refit of the val-selected config on train+val with the selected epoch
           count (its val row is in-sample and NOT a selection score)

Every variant logs val / test / test_strict to SQLite (src.metrics via model_utils) and
is saved for inference under artifacts/models/<name>_<task>/ with meta.json
(see RobertaPredictor at the bottom for loading).
"""
from __future__ import annotations

import argparse
import json
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from src import config  # noqa: F401  (pins GPU 0 before torch)
import torch
import torch.nn as nn

from src import trainer
from src.finetune_e5 import E5Classifier
from src.model_utils import (compute_metrics, dir_size_mb, encode_labels, label_maps, load_json, load_task,
                             log_all_splits, prediction_frame, prob_cache_dir, save_json, save_probs)
from src.trainer import TokenData, TrainConfig, fit, get_device, predict_proba, set_seed

BASE_MODEL = "kz-transformers/kaz-roberta-conversational"
MAX_LEN = 128
PREFIX = ""                       # kaz-roberta needs no instruction prefix (unlike e5 'query: ')
LR, DECAY = 1e-4, 0.8
EPOCHS, PATIENCE = 5, 2
REC_HALF_LIFE, REC_FLOOR = 6.0, 0.1
SUMMARY_JSON = config.RESULTS_DIR / "roberta_tune.json"
FT_REFERENCE = "kaz_roberta_ft"   # survey recipe (src.model_survey), the reference for every delta"


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def encode(tok, titles: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Titles -> int32 [N, MAX_LEN] padded ids + int64 lengths."""
    enc = tok([PREFIX + t for t in titles], truncation=True, max_length=MAX_LEN)["input_ids"]
    ids = np.full((len(enc), MAX_LEN), tok.pad_token_id, dtype=np.int32)
    lens = np.array([len(e) for e in enc], dtype=np.int64)
    for i, e in enumerate(enc):
        ids[i, :len(e)] = e
    return ids, lens


def recency_weights(years: np.ndarray) -> np.ndarray:
    """0.5 ** ((TRAIN_END_YEAR - year) / half-life), floored, normalised to mean 1."""
    age = np.clip(config.TRAIN_END_YEAR - years.astype(np.float64), 0, None)
    w = np.maximum(0.5 ** (age / REC_HALF_LIFE), REC_FLOOR)
    return (w / w.mean()).astype(np.float32)


def llrd_groups(model: nn.Module, weight_decay: float, base_lr: float, decay: float) -> list[dict]:
    """AdamW groups with layer-wise LR decay (no weight decay on biases / norms)."""
    n_layers = model.encoder.config.num_hidden_layers
    groups: dict[tuple, dict] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("encoder.embeddings"):
            depth = n_layers + 1
        elif ".layer." in name:
            depth = n_layers - int(name.split(".layer.")[1].split(".")[0])
        else:                                    # head (+ unused pooler)
            depth = 0
        no_decay = p.ndim < 2 or "norm" in name.lower()
        g = groups.setdefault((depth, no_decay), {"params": [], "lr": base_lr * decay ** depth,
                                                  "weight_decay": 0.0 if no_decay else weight_decay})
        g["params"].append(p)
    return list(groups.values())


@contextmanager
def patched_param_groups(decay: float | None, base_lr: float):
    """Temporarily make trainer.fit build LLRD groups (fit looks the function up at call time)."""
    original = trainer.param_groups
    if decay is not None:
        trainer.param_groups = lambda model, wd: llrd_groups(model, wd, base_lr, decay)
    try:
        yield
    finally:
        trainer.param_groups = original


# --------------------------------------------------------------------------
# One variant
# --------------------------------------------------------------------------
class Context:
    """Tokenised task data shared by all variants of one process (tokenise once)."""

    def __init__(self, task: str, prune: bool = False, quick: bool = False):
        from transformers import AutoTokenizer
        from src import vocab_prune
        self.task, self.prune, self.quick = task, prune, quick
        self.data = load_task(task, max_train=2000, max_eval=500) if quick else load_task(task)
        self.classes, self.label2id = label_maps(self.data["train"]["label"])
        self.tok = AutoTokenizer.from_pretrained(BASE_MODEL)
        self.raw = {s: encode(self.tok, f["title"].tolist()) for s, f in self.data.items()}
        self.lookup, self.kept = None, None
        if prune:
            self.kept = vocab_prune.kept_token_ids(self.raw["train"][0], self.tok.all_special_ids)
            self.lookup = vocab_prune.build_lookup(self.kept, len(self.tok), self.tok.unk_token_id)
            assert int(self.lookup[self.tok.pad_token_id]) == self.tok.pad_token_id  # RoBERTa position ids
            self.raw = {s: (self.lookup[ids], lens) for s, (ids, lens) in self.raw.items()}
        self.latency: float | None = None


def build_model(ctx: Context, device) -> E5Classifier:
    """Fresh pretrained encoder (+ optional vocab pruning) and a new linear head."""
    from transformers import AutoModel
    from src import vocab_prune
    encoder = AutoModel.from_pretrained(BASE_MODEL)
    if ctx.prune:
        encoder = vocab_prune.prune_encoder(encoder, ctx.kept, ctx.tok.pad_token_id)
    set_seed(config.SEED)
    return E5Classifier(encoder, len(ctx.classes)).to(device)


def train_variant(ctx: Context, name: str, cfg: TrainConfig, decay: float | None, recency: bool,
                  trainval: bool = False, note: str = "") -> dict:
    """Train, evaluate, log (val/test/test_strict) and save one variant."""
    from src.model_survey import _cpu_latency
    device = get_device()
    y = {s: encode_labels(ctx.data[s]["label"], ctx.label2id) for s in ctx.data}
    train_ids, train_lens, y_train = ctx.raw["train"][0], ctx.raw["train"][1], y["train"]
    years = ctx.data["train"]["year"].to_numpy()
    if trainval:                              # refit: val joins the training data
        train_ids = np.concatenate([train_ids, ctx.raw["val"][0]])
        train_lens = np.concatenate([train_lens, ctx.raw["val"][1]])
        y_train = np.concatenate([y_train, y["val"]])
        years = np.concatenate([years, ctx.data["val"]["year"].to_numpy()])
    tensors = {"train": TokenData.build(train_ids, train_lens, y_train, device)}
    tensors.update({s: TokenData.build(*ctx.raw[s], y[s], device) for s in ("val", "test")})
    sw = torch.from_numpy(recency_weights(years)).to(device) if recency else None
    model = build_model(ctx, device)
    n_params = int(sum(p.numel() for p in model.parameters()))
    out_dir = config.MODELS_DIR / f"{name}_{ctx.task}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with patched_param_groups(decay, cfg.lr):
        res = fit(model, tensors["train"], tensors["val"], len(ctx.classes), cfg, out_dir / "model.pt",
                  f"{name}_{ctx.task}", log_csv=config.RESULTS_DIR / f"train_log_{name}_{ctx.task}.csv",
                  sample_weight=sw)
    probs = {s: predict_proba(model, tensors[s], 512) for s in ("val", "test")}
    preds = {s: prediction_frame(ctx.data[s]["law_id"], ctx.data[s]["label"], probs[s], ctx.classes)
             for s in ("val", "test")}
    save_model(model, ctx, out_dir, name, res, cfg, decay, recency, trainval)
    for s in ("val", "test"):
        save_probs(out_dir.name, s, probs[s], ctx.data[s]["law_id"], ctx.classes)
    if ctx.latency is None:                   # same architecture for every variant -> measure once
        ctx.latency = _cpu_latency(model, ctx.tok, lookup=ctx.lookup)
    size_mb = dir_size_mb(out_dir)
    common = {"n_params": n_params, "size_mb": round(size_mb, 2), "latency_ms_per_1000": round(ctx.latency, 1),
              "latency_device": "cpu", "peak_mem_mb": res["train_peak_gpu_mb"], "train_time_s": res["train_time_s"],
              "n_train": len(y_train)}
    hp = {"base_model": BASE_MODEL, "max_len": MAX_LEN, "llrd_decay": decay, "recency_weighting": recency,
          "recency_half_life_years": REC_HALF_LIFE if recency else None, "trainval_refit": trainval,
          "vocab_pruned": ctx.prune, "best_epoch": res["best_epoch"], **res["train_config"],
          "peak_mem_mb_meaning": "training peak CUDA memory",
          "latency_protocol": "1000 seeded test titles, CPU 8 threads, batch 64, median of 3 (measured once "
                              "per architecture in this process)"}
    full_note = (f"fine-tuned {BASE_MODEL}{' (vocab pruned to train ids)' if ctx.prune else ''}, mean pooling + "
                 f"linear head; src.roberta_tune variant '{name}': {note}")
    if trainval:
        full_note += ("; REFIT on train+val with the epoch count selected on val - the val row is in-sample "
                      "and is NOT a valid selection score")
    if ctx.quick:       # smoke test: never written to SQLite, artifacts removed at the end of run()
        logged = {s: compute_metrics(f["y_true"], f["y_pred"]) for s, f in preds.items()}
    else:
        logged = log_all_splits(name, "pretrained", ctx.task, preds, common, hp, notes=full_note)
    f1 = {s: round(v["f1_macro"], 4) for s, v in logged.items()}
    print(f"[roberta_tune] {name} {ctx.task}: {f1} best_epoch {res['best_epoch']} ({res['train_time_s']} s)",
          flush=True)
    del model, tensors
    torch.cuda.empty_cache()
    return {"name": name, "task": ctx.task, "f1_macro": f1, "best_epoch": res["best_epoch"],
            "epochs_run": res["epochs_run"], "n_params": n_params, "size_mb": round(size_mb, 1),
            "latency_cpu_ms_per_1000": round(ctx.latency, 1), "train_time_s": res["train_time_s"],
            "path": str(out_dir.relative_to(config.PROJECT_ROOT))}


def save_model(model: E5Classifier, ctx: Context, out_dir: Path, name: str, res: dict, cfg: TrainConfig,
               decay: float | None, recency: bool, trainval: bool) -> None:
    """encoder/ (HF format, + tokenizer), head.pt, label2id.json, id_lookup.npy, meta.json + config.json."""
    model.encoder.save_pretrained(out_dir / "encoder")
    ctx.tok.save_pretrained(out_dir / "encoder")
    torch.save(model.head.state_dict(), out_dir / "head.pt")
    (out_dir / "model.pt").unlink(missing_ok=True)
    save_json(ctx.label2id, out_dir / "label2id.json")
    if ctx.lookup is not None:
        np.save(out_dir / "id_lookup.npy", ctx.lookup)
    meta = {"model_type": "hf_encoder_meanpool_linear", "loader": "src.roberta_tune.RobertaPredictor",
            "model_name": name, "task": ctx.task, "base_model": BASE_MODEL, "classes": ctx.classes,
            "max_len": MAX_LEN, "prefix": PREFIX, "pooling": "masked mean",
            "id_lookup": "id_lookup.npy" if ctx.lookup is not None else None,
            "best_epoch": res["best_epoch"], "best_val_f1": res["best_val_f1"], "trainval_refit": trainval,
            "llrd_decay": decay, "recency_weighting": recency, "loss": cfg.loss, "lr": cfg.lr}
    save_json(meta, out_dir / "meta.json")
    save_json({k: meta[k] for k in ("task", "model_name", "base_model", "max_len", "prefix", "pooling",
                                     "best_epoch", "best_val_f1")}, out_dir / "config.json")


# --------------------------------------------------------------------------
# Sweep (selection on VAL only)
# --------------------------------------------------------------------------
def run(task: str = "area", prune: bool = False, refit: bool = True, only: str = "", quick: bool = False,
        refit_epochs: int = 0) -> dict:
    """llrd -> {cw, ls, rec} -> combo of the components that helped on val -> train+val refit.

    refit_epochs > 0 with a single `only` variant: skip training on train and ONLY run the train+val
    refit of that (already val-selected) config, stopping after `refit_epochs` epochs.

    The refit keeps the selected run's LR schedule (same `epochs`) and stops after its best epoch.
    quick=True: 2,000 train / 500 eval rows, 1 epoch, names end in '_quick' (smoke test only).
    """
    t0 = time.time()
    ctx = Context(task, prune, quick)
    stem = ("kaz_roberta_kkprune" if prune else "kaz_roberta") + ("_quick" if quick else "")
    base = TrainConfig(epochs=1 if quick else EPOCHS, batch_size=64, lr=LR, weight_decay=0.01, warmup_ratio=0.06, loss="ce",
                       patience=PATIENCE, eval_batch_size=512, bucket=True)
    variants = {   # name -> (cfg, llrd decay, recency, note)
        "ep5": (replace(base, lr=5e-5), None, False, f"no LLRD, lr 5e-5 (survey recipe), <= {EPOCHS} epochs, "
                                                      "early stop on val F1"),
        "llrd": (base, DECAY, False, f"LLRD {DECAY}, lr {LR}, <= {EPOCHS} epochs, early stop on val F1"),
        "llrd_cw": (replace(base, loss="weighted"), DECAY, False, "LLRD + class-weighted CE (1/sqrt freq)"),
        "llrd_ls": (replace(base, loss="smooth", label_smoothing=0.1), DECAY, False, "LLRD + label smoothing 0.1"),
        "llrd_rec": (base, DECAY, True, f"LLRD + recency weights (half-life {REC_HALF_LIFE} y, floor {REC_FLOOR})"),
    }
    explicit = {   # explicit combinations, only run when requested with `only` (e.g. on the pruned encoder)
        "llrd_cw_rec": (replace(base, loss="weighted"), DECAY, True, "LLRD + class-weighted CE + recency weights"),
        "llrd_ls_rec": (replace(base, loss="smooth", label_smoothing=0.1), DECAY, True,
                        "LLRD + label smoothing 0.1 + recency weights"),
        "ep5_rec": (replace(base, lr=5e-5), None, True, "no LLRD, lr 5e-5, <= 5 epochs + recency weights"),
        "ft": (replace(base, epochs=3, lr=5e-5, patience=3), None, False,
               "survey recipe of src.model_survey (lr 5e-5, 3-epoch schedule, CE, no LLRD)"),
    }
    if only:
        variants = {k: v for k, v in {**variants, **explicit}.items() if k in only.split(",")}
    if refit_epochs > 0:
        if len(variants) != 1:
            raise ValueError("refit_epochs needs exactly one variant in `only`")
        key, (cfg, decay, rec, note) = next(iter(variants.items()))
        res = train_variant(ctx, f"{stem}_{key}_trainval", replace(cfg, stop_after=refit_epochs, patience=99),
                            decay, rec, trainval=True,
                            note=f"{note}; {refit_epochs} epochs (best val epoch of '{stem}_{key}')")
        summary = {"stem": f"{stem}_{key}_refit", "task": task, "prune": prune, "refit": res,
                   "wall_time_s": round(time.time() - t0, 1)}
        _save_summary(summary)
        return summary
    results = {}
    for key, (cfg, decay, rec, note) in variants.items():
        results[key] = {**train_variant(ctx, f"{stem}_{key}", cfg, decay, rec, note=note),
                        "cfg": (cfg, decay, rec, note)}
    val = {k: r["f1_macro"]["val"] for k, r in results.items()}
    # combo: plain llrd + every component that improved val over plain llrd
    if "llrd" in val:
        loss_keys = [k for k in ("llrd_cw", "llrd_ls") if k in val and val[k] > val["llrd"]]
        best_loss = max(loss_keys, key=val.get) if loss_keys else None
        use_rec = "llrd_rec" in val and val["llrd_rec"] > val["llrd"]
        if best_loss and use_rec:
            cfg = results[best_loss]["cfg"][0]
            results["combo"] = {**train_variant(ctx, f"{stem}_combo", cfg, DECAY, True,
                                                note=f"LLRD + {cfg.loss} loss + recency weights (components "
                                                     "that beat plain LLRD on val)"),
                                "cfg": (cfg, DECAY, True, "combo")}
            val["combo"] = results["combo"]["f1_macro"]["val"]
    best = max(val, key=val.get)
    print(f"[roberta_tune] val macro-F1 {val} -> selected '{best}'", flush=True)
    summary = {"stem": stem, "task": task, "prune": prune, "val_f1": val, "selected": f"{stem}_{best}",
               "variants": {k: {kk: vv for kk, vv in r.items() if kk != "cfg"} for k, r in results.items()}}
    if refit:
        cfg, decay, rec, note = results[best]["cfg"]
        n_ep = max(1, results[best]["best_epoch"])
        summary["refit"] = train_variant(ctx, f"{stem}_{best}_trainval",
                                         replace(cfg, stop_after=n_ep, patience=99), decay, rec,
                                         trainval=True, note=f"{note}; {n_ep} epochs (best val epoch of '{best}')")
    summary["wall_time_s"] = round(time.time() - t0, 1)
    if quick:
        _remove_quick_outputs(stem, task)
    else:
        _save_summary(summary)
    return summary


def _remove_quick_outputs(stem: str, task: str) -> None:
    """Delete every file a quick (smoke) run wrote: model dirs, probability caches, train logs, TB runs."""
    import shutil
    from src.model_utils import RUNS_DIR
    for d in list(config.MODELS_DIR.glob(f"{stem}_*_{task}")) + list(RUNS_DIR.glob(f"{stem}_*_{task}")):
        shutil.rmtree(d, ignore_errors=True)
        shutil.rmtree(prob_cache_dir(d.name), ignore_errors=True)
    for f in config.RESULTS_DIR.glob(f"train_log_{stem}_*_{task}.csv"):
        f.unlink(missing_ok=True)


def run_all(tasks: tuple[str, ...] = ("area", "form"), quick: bool = False) -> dict:
    """Pipeline entry: the val-selected sweep + refit for every task, then the survey table section."""
    out = {t: run(t, quick=quick) for t in tasks}
    write_survey_section()
    return {t: {"selected": r["selected"], "val_f1": r["val_f1"]} for t, r in out.items()}


def write_survey_section() -> str:
    """(Re)write the 'kaz-roberta improvements' section of results/model_survey.md from SQLite."""
    from src import experiments_db
    from src.model_survey import SURVEY_MD
    marker = "## kaz-roberta improvements (src.roberta_tune)"
    try:
        d = experiments_db.load_experiments()
    except Exception as exc:          # DB busy / missing: keep the old file
        print(f"[roberta_tune] survey section skipped: {exc}")
        return ""
    d = d[d["model_name"].str.startswith("kaz_roberta") & ~d["model_name"].str.contains("quick")]
    piv = d.pivot_table(index=["task", "model_name"], columns="eval_split", values="f1_macro", aggfunc="first")
    meta = d.groupby(["task", "model_name"]).agg(n_params=("n_params", "first"), size_mb=("size_mb", "first"),
                                                 lat=("latency_ms_per_1000", "first"))
    lines = [marker, "",
             "All rows from SQLite `experiments` (macro-F1, shared `src.metrics`). Selection is on **val** only; "
             "`*_trainval` rows are refits on train+val with the selected epoch count, so their val column is "
             "in-sample (marked *) and was never used for selection. Latency: CPU, 8 threads, batch 64, "
             "ms per 1000 titles on a busy shared machine.", "",
             "| task | model | val | test | test_strict | params | MB | CPU ms/1000 |", "|---|---|---|---|---|---|---|---|"]
    piv = piv.join(meta).reset_index()
    piv["refit"] = piv["model_name"].str.endswith("_trainval")
    piv = piv.sort_values(["task", "refit", "val"], ascending=[True, True, False])
    for r in piv.to_dict("records"):
        task, name, m = r["task"], r["model_name"], r
        star = "*" if name.endswith("_trainval") else ""
        lines.append(f"| {task} | {name} | {r.get('val', float('nan')):.4f}{star} | {r.get('test', float('nan')):.4f} | "
                     f"{r.get('test_strict', float('nan')):.4f} | {m['n_params'] / 1e6:.1f}M | {m['size_mb']:.0f} | "
                     f"{m['lat']:.0f} |")
    lines.append("")
    for task, g in piv.groupby("task"):          # val-selected model vs the survey recipe, per task
        sel = g[~g["refit"]].iloc[0]
        ref = g[g["model_name"] == FT_REFERENCE]
        if len(ref) and sel["model_name"] != FT_REFERENCE:
            ref = ref.iloc[0]
            lines.append(f"- `{task}`: val-selected **{sel['model_name']}**; vs `{FT_REFERENCE}` (lr 5e-5, 3 epochs): "
                         + ", ".join(f"{s_} {sel[s_] - ref[s_]:+.4f}" for s_ in ("val", "test", "test_strict")) + "."
                         + (" The val gain does NOT transfer to test (2022 val vs >=2023 test shift); reported as "
                            "selected, test was not used to change the choice." if sel["test"] < ref["test"] else ""))
        else:
            lines.append(f"- `{task}`: val-selected **{sel['model_name']}**.")
    section = "\n".join(lines) + "\n"
    try:
        text = SURVEY_MD.read_text(encoding="utf-8") if SURVEY_MD.exists() else ""
        text = text.split(marker)[0].rstrip() + "\n\n" + section
        SURVEY_MD.write_text(text, encoding="utf-8")
    except OSError as exc:
        print(f"[roberta_tune] could not write {SURVEY_MD}: {exc}")
    return section


def backfill_meta() -> list[str]:
    """Write meta.json (+ e5-style config.json) for kaz-roberta dirs saved by src.model_survey."""
    done = []
    for d in sorted(config.MODELS_DIR.glob("kaz_roberta*_ft_*")):
        if (d / "meta.json").exists() or not (d / "head.pt").exists():
            continue
        task = d.name.rsplit("_", 1)[1]
        l2id = load_json(d / "label2id.json")
        name = d.name[: -len(task) - 1]
        meta = {"model_type": "hf_encoder_meanpool_linear", "loader": "src.roberta_tune.RobertaPredictor",
                "model_name": name, "task": task, "base_model": BASE_MODEL,
                "classes": [c for c, _ in sorted(l2id.items(), key=lambda kv: kv[1])], "max_len": MAX_LEN,
                "prefix": PREFIX, "pooling": "masked mean",
                "id_lookup": "id_lookup.npy" if (d / "id_lookup.npy").exists() else None,
                "trainval_refit": False, "saved_by": "src.model_survey.finetune"}
        save_json(meta, d / "meta.json")
        if not (d / "config.json").exists():
            save_json({k: meta[k] for k in ("task", "model_name", "base_model", "max_len", "prefix", "pooling")},
                      d / "config.json")
        done.append(d.name)
    return done


def _save_summary(summary: dict) -> None:
    """Merge this run into results/roberta_tune.json (keyed by task + prune)."""
    try:
        allres = load_json(SUMMARY_JSON) if SUMMARY_JSON.exists() else {}
    except (OSError, json.JSONDecodeError):
        allres = {}
    allres[f"{summary['stem']}_{summary['task']}"] = summary
    save_json(allres, SUMMARY_JSON)


# --------------------------------------------------------------------------
# Inference loader (for src.predict / src.inference)
# --------------------------------------------------------------------------
class RobertaPredictor:
    """Load a model saved by this module (or src.model_survey) and predict class probabilities.

    >>> p = RobertaPredictor("artifacts/models/kaz_roberta_llrd_area")
    >>> p.predict(["Қазақстан Республикасының Білім туралы заңы"], top_k=3)
    """

    def __init__(self, model_dir: str | Path, device: str = "cpu"):
        from transformers import AutoModel, AutoTokenizer
        self.model_dir = Path(model_dir)
        if not self.model_dir.is_absolute():
            self.model_dir = config.PROJECT_ROOT / self.model_dir
        meta_path = self.model_dir / "meta.json"
        meta = load_json(meta_path) if meta_path.exists() else {}
        l2id = load_json(self.model_dir / "label2id.json")
        self.classes = meta.get("classes") or [c for c, _ in sorted(l2id.items(), key=lambda kv: kv[1])]
        self.max_len = int(meta.get("max_len", MAX_LEN))
        self.prefix = meta.get("prefix", PREFIX)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir / "encoder")
        encoder = AutoModel.from_pretrained(self.model_dir / "encoder")
        self.model = E5Classifier(encoder, len(self.classes)).eval()
        self.model.head.load_state_dict(torch.load(self.model_dir / "head.pt", map_location="cpu",
                                                   weights_only=True))
        lut = self.model_dir / "id_lookup.npy"
        self.lookup = torch.from_numpy(np.load(lut).astype(np.int64)) if lut.exists() else None
        self.n_params = int(sum(p.numel() for p in self.model.parameters()))
        self.size_mb = dir_size_mb(self.model_dir)
        self.to(device)

    def to(self, device: str) -> "RobertaPredictor":
        """Move the model to 'cpu' or 'cuda' (same interface as src.inference predictors)."""
        self.device = device
        self.model.to(device)
        return self

    @torch.inference_mode()
    def predict_proba(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Softmax probabilities [N, n_classes] in the order of self.classes."""
        out = []
        for start in range(0, len(texts), batch_size):
            enc = self.tokenizer([self.prefix + t for t in texts[start:start + batch_size]], truncation=True,
                                 max_length=self.max_len, padding=True, return_tensors="pt")
            ids = enc["input_ids"] if self.lookup is None else self.lookup[enc["input_ids"]]
            logits = self.model(ids.to(self.device), enc["attention_mask"].bool().to(self.device))
            out.append(torch.softmax(logits.float(), -1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, len(self.classes)), dtype=np.float32)

    def predict(self, texts: list[str], top_k: int = 3) -> list[dict]:
        """[{'label', 'confidence', 'top_k': [(label, p), ...]}] per text."""
        probs = self.predict_proba(list(texts))
        res = []
        for row in probs:
            order = np.argsort(-row)[:top_k]
            res.append({"label": self.classes[order[0]], "confidence": float(row[order[0]]),
                        "top_k": [(self.classes[i], float(row[i])) for i in order]})
        return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Val-selected kaz-roberta improvements")
    parser.add_argument("--task", default="area", choices=list(config.TASKS))
    parser.add_argument("--prune", action="store_true", help="use the train-vocabulary-pruned encoder")
    parser.add_argument("--no-refit", action="store_true")
    parser.add_argument("--only", default="", help="comma list of variants (ep5,llrd,llrd_cw,llrd_ls,llrd_rec)")
    parser.add_argument("--quick", action="store_true", help="tiny smoke test (names end in _quick)")
    parser.add_argument("--refit-epochs", type=int, default=0, help="only refit the single --only variant")
    a = parser.parse_args()
    print(json.dumps(run(a.task, a.prune, not a.no_refit, a.only, a.quick, a.refit_epochs), ensure_ascii=False,
                     indent=1,
                     default=str))
