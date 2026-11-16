"""Non-interactive reproduction of the whole project.

Examples (from the project root):
    .venv/bin/python run_pipeline.py --steps all            # full run
    .venv/bin/python run_pipeline.py --steps all --quick    # smoke test, a few minutes
    .venv/bin/python run_pipeline.py --steps tiny,e5,benchmark,compare,errors

--quick trains every model on a small subset for 1-2 epochs and tags the
results with the suffix "_quick" (models, SQLite rows, output files), so a
smoke test never overwrites the full-run results. In quick mode, the slow data
steps whose outputs already exist (parquet, leakage ids, SQLite corpus) are
skipped; pass --force-data to rerun them anyway.
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import sys
import time
from datetime import datetime

from src import config

# step name -> (module, kwargs for a full run, kwargs for a quick run)
STEPS: dict[str, tuple[str, dict, dict]] = {
    "data": ("src.data_io", {}, {"benchmark": False}),
    "audit": ("src.audit", {}, {}),
    "sql": ("src.database", {}, {"rebuild": False}),
    "eda": ("src.eda", {}, {}),
    "leakage": ("src.leakage", {}, {}),
    "text_prep": ("src.text_prep", {}, {}),
    "baseline": ("src.baseline", {}, {}),
    "tokenizer": ("src.bpe_tokenizer", {}, {}),
    "tiny": ("src.tiny_transformer", {}, {}),
    "e5": ("src.finetune_e5", {}, {}),
    "mlm": ("src.mlm_pretrain", {}, {}),          # optional extension: MLM-pretrained tiny transformer
    "benchmark": ("src.benchmark", {}, {}),
    "compare": ("src.compare", {}, {}),
    "errors": ("src.error_analysis", {}, {}),
    "predict": ("src.predict", {}, {}),
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
        print(f"[skip] {name}: output already present ({marker.relative_to(config.PROJECT_ROOT)})")
        return {"step": name, "status": "skipped"}
    t0 = time.perf_counter()
    try:
        module = importlib.import_module(module_name)
        params = inspect.signature(module.run).parameters
        kwargs = dict(quick_kwargs if quick else full_kwargs)
        kwargs = {k: v for k, v in kwargs.items() if k in params}
        if quick and "quick" in params:
            kwargs["quick"] = True
        print(f"\n===== {name}: {module_name}.run({kwargs}) =====", flush=True)
        result = module.run(**kwargs)
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
    args = parser.parse_args(argv)

    steps = parse_steps(args.steps)
    print(f"Pipeline steps: {steps} | quick={args.quick}")
    log = []
    t0 = time.perf_counter()
    for step in steps:
        record = run_step(step, args.quick, args.force_data)
        log.append(record)
        if record["status"] == "error" and args.stop_on_error:
            break
    total = round(time.perf_counter() - t0, 1)
    out = config.RESULTS_DIR / f"pipeline_log{'_quick' if args.quick else ''}.json"
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
