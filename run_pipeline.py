"""Non-interactive reproduction of the whole project.

Examples (from the project root):
    .venv/bin/python run_pipeline.py --steps all            # full run
    .venv/bin/python run_pipeline.py --steps all --quick    # smoke test, a few minutes
    .venv/bin/python run_pipeline.py --steps tiny,e5,benchmark,compare,errors

--quick trains every model on a small subset for 1-2 epochs and writes ALL its
outputs (models, results, figures, SQLite) to a separate scratch folder
(default: <tmp>/kazlaw_quick, see --quick-dir), so a smoke test never touches
the full-run deliverables. In quick mode, the slow data steps whose outputs
already exist (parquet, leakage ids, audit, EDA, baselines) are skipped; pass
--force-data to rerun them anyway.
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

from src import config

# step name -> (module, kwargs for a full run, kwargs for a quick run)
STEPS: dict[str, tuple[str, dict, dict]] = {
    # --- data side -------------------------------------------------------------------------
    "data": ("src.data_io", {}, {"benchmark": False}),
    "audit": ("src.audit", {}, {}),
    "sql": ("src.database", {}, {"rebuild": False}),
    "eda": ("src.eda", {}, {}),
    "leakage": ("src.leakage", {}, {}),
    "text_prep": ("src.text_prep", {}, {}),
    "baseline": ("src.baseline", {}, {}),
    # --- own tokenizer + tiny transformer, e5 ----------------------------------------------
    "tokenizer": ("src.bpe_tokenizer", {}, {}),
    "tiny": ("src.tiny_transformer", {}, {}),     # loss search, 2L/4L x d128/d256, dropout, 4k vocab
    "seeds": ("src.tiny_transformer:run_seed_study", {}, {}),   # 3-seed variance of the tiny model
    "mlm": ("src.mlm_pretrain", {}, {}),          # MLM-pretrained tiny transformer
    "e5": ("src.finetune_e5", {}, {}),            # frozen e5 + LR, e5 fine-tuning (= KD teacher)
    "distill": ("src.distill", {}, {}),           # KD e5 -> tiny transformer, recency weighting
    "refit": ("src.distill:run_refits", {}, {}),  # train+val refit of the val-best tiny model
    # --- research modules: vocabulary pruning, mT5 seq2seq, model survey, Kazakh RoBERTa -----
    "prep_ablation": ("src.prep_ablation", {}, {}),
    "vocab_prune": ("src.vocab_prune", {}, {}),
    "seq2seq": ("src.seq2seq_mt5", {}, {}),
    "survey": ("src.model_survey", {}, {"finetune_model": False}),
    # val-selected kaz-roberta improvements: LLRD, <=5 epochs + early stopping, class-weighted / smoothed CE,
    # recency weights, train+val refit (GPU, ~20 min per task)
    "roberta_tune": ("src.roberta_tune:run_all", {}, {}),
    # KazBERT fine-tune (kaz-roberta recipe) + kaz-roberta as KD teacher for the tiny transformer and
    # the vocabulary-pruned e5-small (GPU, ~25 min; needs the kaz-roberta teachers; skipped in --quick)
    "distill_roberta": ("src.distill_roberta", {}, {}),
    # LLM teacher feasibility (Qwen3-1.7B-Base + LoRA; Sherkala-8B is gated): LoRA teacher + val/test
    # logging only - it lost to kaz-roberta on val, so KD was not run (GPU ~13 min per task, ~4 GB
    # download; skipped in --quick)
    "llm_teacher": ("src.llm_teacher", {"tasks": ("form", "area"), "stage": "teacher"}, {}),
    # --- post-hoc, measurement, reporting ----------------------------------------------------
    "posthoc": ("src.posthoc", {}, {}),           # prior-shift adjustment + ensembles (needs kaz-roberta cache)
    "benchmark": ("src.benchmark", {"select": "main"}, {}),   # same protocol, main-table models only
    "compare": ("src.compare", {}, {}),
    "errors": ("src.error_analysis", {}, {}),
    "predict": ("src.predict", {}, {}),
    # refresh results/sql/*.csv so the SQL query outputs include the final experiment tables
    "sql_final": ("src.database:run_queries", {}, {}),
}
# Outputs that let quick mode skip an expensive, already-completed data step.
DONE_MARKERS = {
    "data": config.PROCESSED_DIR / "READY",
    "leakage": config.RESULTS_DIR / "test_strict_ids.csv",
    "audit": config.RESULTS_DIR / "audit.json",
    "text_prep": config.RESULTS_DIR / "text_prep_comparison.json",
    "eda": config.RESULTS_DIR / "eda_summary.json",
    # src.baseline has no quick mode (it trains all TF-IDF models, ~10 min): in quick mode it is
    # skipped as soon as one trained baseline exists on disk.
    "baseline": config.MODELS_DIR / "tfidf_word_lr_form" / "model.joblib",
}


def redirect_outputs(quick_dir: Path) -> None:
    """Point every OUTPUT path of src.config to `quick_dir` (inputs stay in place).

    Must run before any step module is imported: modules derive their output paths
    from src.config at import time. Small inputs that the model steps read from the
    results/models folders (strict test ids, token stats, tokenizer) are copied.
    """
    real_results, real_models = config.RESULTS_DIR, config.MODELS_DIR
    config.ARTIFACTS_DIR = quick_dir
    for attr, sub in (("RESULTS_DIR", "results"), ("FIGURES_DIR", "figures"), ("MODELS_DIR", "models")):
        target = quick_dir / sub
        target.mkdir(parents=True, exist_ok=True)
        setattr(config, attr, target)
    config.DB_PATH = quick_dir / "kazlaw_quick.db"
    os.environ["KAZLAW_DB"] = str(config.DB_PATH)          # read by src.model_utils
    os.environ["KAZLAW_RUNS_DIR"] = str(quick_dir / "runs")  # TensorBoard logs of quick runs
    for name in ("test_strict_ids.csv", "token_lengths.json", "tiny_transformer_loss_search.csv"):
        if (real_results / name).exists():
            shutil.copy(real_results / name, config.RESULTS_DIR / name)
    for name in ("bpe_tokenizer", "bpe_tokenizer_4k"):
        if (real_models / name).exists():
            shutil.copytree(real_models / name, config.MODELS_DIR / name, dirs_exist_ok=True)
    print(f"[quick] all outputs go to {quick_dir}")


def parse_steps(spec: str) -> list[str]:
    """'all' or a comma-separated list -> validated, ordered list of step names."""
    if spec.strip().lower() == "all":
        return list(STEPS)
    steps = [s.strip() for s in spec.split(",") if s.strip()]
    unknown = [s for s in steps if s not in STEPS]
    if unknown:
        raise SystemExit(f"Unknown step(s) {unknown}. Valid: {', '.join(STEPS)} or 'all'")
    return [s for s in STEPS if s in steps]          # canonical order


def run_step(name: str, quick: bool, force_data: bool) -> dict:
    """Run one step; return a status record (never raises)."""
    module_name, full_kwargs, quick_kwargs = STEPS[name]
    marker = DONE_MARKERS.get(name)
    if quick and not force_data and marker is not None and marker.exists():
        print(f"[skip] {name}: output already present ({marker})")
        return {"step": name, "status": "skipped"}
    t0 = time.perf_counter()
    try:
        module_path, _, func_name = module_name.partition(":")
        func = getattr(importlib.import_module(module_path), func_name or "run")
        params = inspect.signature(func).parameters
        kwargs = dict(quick_kwargs if quick else full_kwargs)
        kwargs = {k: v for k, v in kwargs.items() if k in params}
        if quick and "quick" in params:
            kwargs["quick"] = True
        print(f"\n===== {name}: {module_name}({kwargs}) =====", flush=True)
        result = func(**kwargs)
        status = {"step": name, "status": "ok", "seconds": round(time.perf_counter() - t0, 1)}
        if isinstance(result, dict):
            status["summary"] = {k: v for k, v in list(result.items())[:8]}
        return status
    except Exception as exc:
        print(f"[error] step {name} failed: {type(exc).__name__}: {exc}")
        return {"step": name, "status": "error", "error": f"{type(exc).__name__}: {exc}",
                "seconds": round(time.perf_counter() - t0, 1)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", default="all", help=f"'all' or comma list of: {', '.join(STEPS)}")
    parser.add_argument("--quick", action="store_true", help="small, fast smoke-test run")
    parser.add_argument("--force-data", action="store_true", help="in quick mode, rerun finished data steps")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--quick-dir", default=str(Path(tempfile.gettempdir()) / "kazlaw_quick"),
                        help="where --quick writes its outputs (never inside the deliverables)")
    args = parser.parse_args(argv)

    steps = parse_steps(args.steps)
    if args.quick:
        redirect_outputs(Path(args.quick_dir))
    print(f"Pipeline steps: {steps} | quick={args.quick}")
    log = []
    t0 = time.perf_counter()
    for step in steps:
        record = run_step(step, args.quick, args.force_data)
        log.append(record)
        if record["status"] == "error" and args.stop_on_error:
            break
    total = round(time.perf_counter() - t0, 1)
    out = config.RESULTS_DIR / "pipeline_log.json"         # quick runs: inside --quick-dir
    try:
        with open(out, "w", encoding="utf-8") as fh:
            json.dump({"finished": datetime.now().isoformat(timespec="seconds"), "quick": args.quick,
                       "total_seconds": total, "steps": log}, fh, ensure_ascii=False, indent=2, default=str)
    except OSError as exc:
        print(f"could not write {out}: {exc}")
    print("\nSummary:")
    for rec in log:
        print(f"  {rec['step']:<10} {rec['status']:<8} {rec.get('seconds', '')}")
    print(f"Total {total}s; log written to {out}")
    return 0 if all(r["status"] != "error" for r in log) else 1


if __name__ == "__main__":
    sys.exit(main())
