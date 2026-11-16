"""One measurement protocol for ALL models (baselines, tiny transformer, e5).

Protocol (identical for every model, end-to-end from raw title to probabilities,
tokenisation/vectorisation included):
* texts: 1000 test titles, fixed seeded sample (config.SEED);
* CPU: torch.set_num_threads(8) (and 8 BLAS/OpenMP threads), batch size 64,
  one warm-up pass, median of 3 timed runs -> latency_ms_per_1000 (primary);
* CUDA (neural models only): same, with torch.cuda.synchronize(), bf16 autocast;
* size on disk (MB) of everything needed to run the model; n_params;
* memory: CPU RAM growth of a fresh process while loading + predicting
  (infer_ram_mb, from peak RSS) and CUDA max_memory_allocated during inference.

Each model is measured in a fresh subprocess (`python -m src.benchmark --kind ...`)
so that memory numbers are not polluted by other models in the same process.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

N_TEXTS, BATCH, REPEATS, CPU_THREADS = 1000, 64, 3, 8
BENCH_CSV_NAME = "benchmark.csv"


def benchmark_texts(n: int = N_TEXTS) -> list[str]:
    """Fixed sample of test titles (same for every model)."""
    from src import config
    from src.model_utils import load_dataset

    df = load_dataset(["title", "split"])
    test = df[df["split"] == "test"]["title"].astype(str)
    return test.sample(n=min(n, len(test)), random_state=config.SEED).tolist()


def reset_peak_rss() -> bool:
    """Reset the kernel's RSS high-water mark of this process (Linux: write 5 to clear_refs)."""
    try:
        with open("/proc/self/clear_refs", "w") as fh:
            fh.write("5")
        return True
    except OSError:
        return False


def hwm_rss_mb() -> float:
    """Peak RSS since the last reset (VmHWM), falling back to the lifetime peak."""
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    from src.model_utils import peak_rss_mb
    return peak_rss_mb()


def time_predictions(pred, texts: list[str], repeats: int, cuda: bool) -> float:
    """Median wall time (ms) to predict all `texts` after one warm-up pass."""
    import torch

    def sync():
        if cuda:
            torch.cuda.synchronize()

    pred.predict_proba(texts[:BATCH * 2], batch_size=BATCH)     # warm-up (lazy init, kernels)
    sync()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        pred.predict_proba(texts, batch_size=BATCH)
        sync()
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


def measure(kind: str, path: str, name: str = "", task: str = "", n_texts: int = N_TEXTS,
            repeats: int = REPEATS, with_cuda: bool = True) -> dict:
    """Measure one model in THIS process (call via measure_subprocess for clean RAM numbers)."""
    from src import config  # noqa: F401  (GPU 0 only)
    import torch
    from src.inference import ModelSpec, load_predictor
    import gc
    from src.model_utils import rss_mb

    torch.set_num_threads(CPU_THREADS)
    texts = benchmark_texts(n_texts)
    scale = 1000.0 / len(texts)
    gc.collect()
    rss_before = rss_mb()
    # The parquet read above has its own RSS peak; reset the high-water mark so that
    # infer_ram_mb only covers loading the model + predicting.
    reset_peak_rss()
    t0 = time.perf_counter()
    pred = load_predictor(ModelSpec(kind, name, task, path), device="cpu")
    load_s = time.perf_counter() - t0
    cpu_ms = time_predictions(pred, texts, repeats, cuda=False)
    out = {"kind": kind, "path": path, "model_name": name, "task": task,
           "n_params": pred.n_params, "size_mb": None if pred.size_mb is None else round(pred.size_mb, 2),
           "load_time_s": round(load_s, 2), "n_texts": len(texts), "batch_size": BATCH,
           "cpu_threads": CPU_THREADS, "repeats": repeats,
           "latency_cpu_ms_per_1000": round(cpu_ms * scale, 1),
           "infer_ram_mb": round(hwm_rss_mb() - rss_before, 1)}
    if with_cuda and pred.supports_cuda and torch.cuda.is_available():
        pred.to("cuda")
        torch.cuda.reset_peak_memory_stats()
        cuda_ms = time_predictions(pred, texts, repeats, cuda=True)
        out["latency_cuda_ms_per_1000"] = round(cuda_ms * scale, 1)
        out["infer_peak_gpu_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    return out


def measure_subprocess(kind: str, path: str, name: str = "", task: str = "", quick: bool = False,
                       timeout: int = 900) -> dict:
    """Run `measure` in a fresh Python process and return its JSON result ({} on failure)."""
    from src import config

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as fh:
        out_path = fh.name
    cmd = [sys.executable, "-m", "src.benchmark", "--kind", kind, "--path", path,
           "--name", name, "--task", task, "--json-out", out_path]
    if quick:
        cmd += ["--n-texts", "200", "--repeats", "1"]
    env = {**os.environ, "OMP_NUM_THREADS": str(CPU_THREADS), "MKL_NUM_THREADS": str(CPU_THREADS),
           "OPENBLAS_NUM_THREADS": str(CPU_THREADS), "TOKENIZERS_PARALLELISM": "false"}
    try:
        proc = subprocess.run(cmd, cwd=config.PROJECT_ROOT, env=env, capture_output=True,
                              text=True, timeout=timeout)
        if proc.returncode != 0:
            print(f"[benchmark] {kind}:{name or path} failed:\n{proc.stderr[-2000:]}")
            return {}
        with open(out_path, encoding="utf-8") as fh:
            result = json.load(fh)
        print(f"[benchmark] {kind}:{name or Path(path).name} cpu={result.get('latency_cpu_ms_per_1000')} ms/1000 "
              f"cuda={result.get('latency_cuda_ms_per_1000')} size={result.get('size_mb')} MB "
              f"params={result.get('n_params')}")
        return result
    except (subprocess.TimeoutExpired, OSError, json.JSONDecodeError) as exc:
        print(f"[benchmark] {kind}:{name or path} error: {exc}")
        return {}
    finally:
        Path(out_path).unlink(missing_ok=True)


def update_experiment_rows(result: dict) -> None:
    """Write the benchmark numbers into every experiments row of that model/task."""
    from src import experiments_db

    if not result or not result.get("model_name"):
        return
    with experiments_db.connect() as con:
        rows = con.execute("SELECT exp_id, hyperparams FROM experiments WHERE model_name=? AND task=?",
                           (result["model_name"], result["task"])).fetchall()
        for exp_id, hp_json in rows:
            try:
                hp = json.loads(hp_json or "{}")
            except json.JSONDecodeError:
                hp = {}
            hp.update({k: result.get(k) for k in ("latency_cuda_ms_per_1000", "infer_peak_gpu_mb",
                                                  "infer_ram_mb") if result.get(k) is not None})
            hp["benchmark_protocol"] = f"{result.get('n_texts')} test titles, batch {BATCH}, " \
                                       f"{CPU_THREADS} CPU threads, median of {result.get('repeats')}"
            con.execute("UPDATE experiments SET n_params=COALESCE(?, n_params), size_mb=COALESCE(?, size_mb), "
                        "latency_ms_per_1000=?, latency_device='cpu', hyperparams=? WHERE exp_id=?",
                        (result.get("n_params"), result.get("size_mb"),
                         result.get("latency_cpu_ms_per_1000"), json.dumps(hp, ensure_ascii=False), exp_id))


def run(tasks: tuple[str, ...] = ("form", "area"), quick: bool = False, update_db: bool = True) -> dict:
    """Benchmark every model found on disk; save results/benchmark.csv and update SQLite."""
    import pandas as pd
    from src import config
    from src.inference import discover_models

    rows = []
    for task in tasks:
        for spec in discover_models(task, include_quick=quick):
            if quick != spec.name.endswith("_quick"):
                continue              # quick runs only touch _quick models (never the real rows)
            res = measure_subprocess(spec.kind, spec.path, spec.name, spec.task, quick=quick)
            if res:
                rows.append(res)
                if update_db:
                    update_experiment_rows(res)
    df = pd.DataFrame(rows)
    out = config.RESULTS_DIR / (BENCH_CSV_NAME if not quick else "benchmark_quick.csv")
    if len(df):
        df.to_csv(out, index=False)
    return {"n_models": len(rows), "csv": str(out)}


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Benchmark one model (or all with --all)")
    parser.add_argument("--kind")
    parser.add_argument("--path", default="")
    parser.add_argument("--name", default="")
    parser.add_argument("--task", default="")
    parser.add_argument("--n-texts", type=int, default=N_TEXTS)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--json-out")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    if args.all or not args.kind:
        print(run())
        return
    result = measure(args.kind, args.path, args.name, args.task, args.n_texts, args.repeats)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False)
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _cli()
