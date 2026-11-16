"""Classify a free-text title with every available model.

    predict(text, task) -> [{"model", "family", "label", "top3": [(label, prob)...], "latency_ms"}, ...]

Models are loaded lazily on first use and cached; a model that is missing or
fails to load is skipped with a warning (the demo never crashes because one
artifact is absent). By default one model per family is shown: the best one on
validation according to results/best_models.json (written by src.compare);
without that file every model found on disk is used.
"""
from __future__ import annotations

import time

from src import config
from src.inference import ModelSpec, discover_models, load_predictor
from src.model_utils import load_json

_CACHE: dict[str, list[tuple[ModelSpec, object]]] = {}
_FAILED: set[str] = set()
EXAMPLES = [
    "Алматы облысы Талғар аудандық мәслихатының 2023 жылғы бюджеті туралы",
    "Қазақстан Республикасы Денсаулық сақтау министрінің кейбір бұйрықтарына өзгерістер енгізу туралы",
    "Салық және бюджетке төленетін басқа да міндетті төлемдер туралы",
]


def _selected_names(task: str) -> set[str] | None:
    """Names of the best model per family (by val F1) or None if compare has not run."""
    try:
        best = load_json(config.RESULTS_DIR / "best_models.json").get(task, {})
        return {v["model_name"] for k, v in best.items() if not k.startswith("_")}
    except (FileNotFoundError, ValueError):
        return None


def get_predictors(task: str, device: str = "cpu", all_models: bool = False) -> list[tuple[ModelSpec, object]]:
    """Load (once) and return the predictors for a task."""
    key = f"{task}:{device}:{all_models}"
    if key in _CACHE:
        return _CACHE[key]
    specs = discover_models(task)
    names = None if all_models else _selected_names(task)
    if names:
        specs = [s for s in specs if s.name in names] or specs
    loaded = []
    for spec in specs:
        ident = f"{spec.kind}:{spec.name}:{task}"
        if ident in _FAILED:
            continue
        try:
            t0 = time.perf_counter()
            loaded.append((spec, load_predictor(spec, device)))
            print(f"  loaded {spec.name} [{task}] in {time.perf_counter() - t0:.1f}s")
        except Exception as exc:          # missing file, missing dependency, corrupt artifact...
            _FAILED.add(ident)
            print(f"  ! skipping {spec.name} [{task}]: {type(exc).__name__}: {exc}")
    order = {"baseline": 0, "tiny": 1, "e5_frozen_lr": 2, "e5_ft": 3}
    loaded.sort(key=lambda sp: (order.get(sp[0].kind, 9), sp[0].name))
    _CACHE[key] = loaded
    return loaded


def predict(text: str, task: str, device: str = "cpu", all_models: bool = False) -> list[dict]:
    """Prediction of every available model for one title."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("empty title")
    if task not in config.TASKS:
        raise ValueError(f"unknown task {task!r}")
    results = []
    for spec, pred in get_predictors(task, device, all_models):
        try:
            t0 = time.perf_counter()
            out = pred.predict([text.strip()], top_k=3)[0]
            results.append({"model": spec.name, "kind": spec.kind, "label": out["label"],
                            "top3": [(lab, round(p, 4)) for lab, p in out["top"]],
                            "latency_ms": round((time.perf_counter() - t0) * 1000, 1)})
        except Exception as exc:
            results.append({"model": spec.name, "kind": spec.kind, "label": f"ERROR: {exc}",
                            "top3": [], "latency_ms": None})
    return results


def format_results(task: str, results: list[dict]) -> str:
    """Readable console table for one task."""
    if not results:
        return f"  [{task}] no trained models available"
    lines = [f"  Task '{task}':"]
    for r in results:
        top = ", ".join(f"{lab} {p:.2f}" for lab, p in r["top3"])
        lat = "-" if r["latency_ms"] is None else f"{r['latency_ms']:.1f} ms"
        lines.append(f"    {r['model']:<28} -> {r['label']:<30} | top3: {top} | {lat}")
    return "\n".join(lines)


def run(texts: list[str] | None = None, tasks: tuple[str, ...] = ("form", "area")) -> dict:
    """Non-interactive demo on example titles (the interactive loop lives in main.py)."""
    out: dict = {}
    for text in texts or EXAMPLES:
        print(f"\nTitle: {text}")
        out[text] = {}
        for task in tasks:
            res = predict(text, task)
            out[text][task] = res
            print(format_results(task, res))
    return {"n_texts": len(out)}


if __name__ == "__main__":
    run()
