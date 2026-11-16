"""Data ingestion: download, chunked CSV reading, dtype optimisation, Parquet.

Pipeline (``run()``):
  1. ``download_if_missing()`` - fetch the raw CSV from Hugging Face if absent.
  2. Pass 1 over the CSV (chunked): exact cardinality of every column and the
     memory a *default* pandas read would need.
  3. Pass 2 over the CSV (chunked): placeholders -> NA, optimised dtypes
     (category / bool / datetime / Int / string), memory before vs after,
     incremental write of ``config.FULL_PARQUET`` with one ParquetWriter.
  4. CSV vs Parquet benchmark (size, read time, peak memory) in fresh
     sub-processes so that the peak-RSS numbers are not polluted.
  5. ``labels.build_dataset`` -> ``config.DATASET_PARQUET`` + split stats +
     ``data/processed/READY`` marker for the models agent.

Never loads the full CSV with object dtype in one go (shared 6 GB box).
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import resource
import statistics
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src import config, labels

# --------------------------------------------------------------------------
# Column roles (decided from a profiling pass, see results/memory_optimization.json)
# --------------------------------------------------------------------------
TEXT_COLS = ["law_id", "title", "description", "official_publication_info", "adilet_url", "ekb_url"]
BOOL_COLS = ["is_archive", "is_repealed"]
DATE_COLS = ["date_adopted", "date_modified", "date_published", "ministry_registration_date"]
INT_COLS = ["source_id", "state_registry_number"]
# Everything else is a candidate for `category` (low cardinality relative to rows).
CATEGORY_MAX_RATIO = 0.5          # category if n_unique <= 50 % of rows

READY_MARKER = config.PROCESSED_DIR / "READY"
MEMORY_JSON = config.RESULTS_DIR / "memory_optimization.json"
IO_BENCH_JSON = config.RESULTS_DIR / "io_benchmark.json"
SPLIT_STATS_JSON = config.RESULTS_DIR / "split_stats.json"
BENCH_COLS = ["law_id", "title", "date_adopted", "act_form", "database_section"]


def _mb(n_bytes: float) -> float:
    """Bytes -> megabytes rounded to 2 decimals."""
    return round(float(n_bytes) / 2**20, 2)


def _save_json(obj: dict, path: Path) -> None:
    """Write a JSON file (UTF-8, Kazakh text kept readable)."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    except OSError as exc:
        print(f"[data_io] could not write {path}: {exc}")


# --------------------------------------------------------------------------
# 1. Download
# --------------------------------------------------------------------------
def download_if_missing(url: str = config.RAW_CSV_URL, dest: Path = config.RAW_CSV) -> Path:
    """Download the raw CSV from Hugging Face unless it already exists."""
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[data_io] raw CSV present: {dest} ({_mb(dest.stat().st_size)} MB)")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    try:
        print(f"[data_io] downloading {url} ...")
        with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as fh:
            while True:
                block = resp.read(1 << 20)
                if not block:
                    break
                fh.write(block)
        tmp.replace(dest)
    except Exception as exc:  # network errors, HTTP errors, disk errors
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Download failed ({exc}). Put the CSV manually at {dest}") from exc
    return dest


# --------------------------------------------------------------------------
# 2. Chunked reading helpers
# --------------------------------------------------------------------------
def iter_raw_chunks(path: Path = config.RAW_CSV, chunksize: int = config.CHUNKSIZE,
                    usecols: list[str] | None = None):
    """Yield raw CSV chunks with every column as Python ``object`` strings.

    keep_default_na=False so that *we* decide what is missing (placeholders),
    i.e. the raw text is preserved exactly ('' stays '').
    """
    return pd.read_csv(path, chunksize=chunksize, dtype=object, keep_default_na=False,
                       na_values=[], usecols=usecols)


def mask_placeholders(s: pd.Series) -> pd.Series:
    """Replace source-system placeholders ('', 'ЖОҚ', 'undefined', ...) by NA."""
    stripped = s.str.strip()
    return s.mask(stripped.isin(config.PLACEHOLDERS))


def profile_columns(path: Path = config.RAW_CSV) -> dict:
    """Pass 1: exact n_unique per column (placeholders excluded) and the memory
    a default pandas read (pandas-3 dtype inference) needs, summed over chunks."""
    uniques: dict[str, set] = {}
    default_mem = 0
    n_rows = 0
    for chunk in pd.read_csv(path, chunksize=config.CHUNKSIZE):
        default_mem += int(chunk.memory_usage(deep=True).sum())
        n_rows += len(chunk)
    for chunk in iter_raw_chunks(path):
        for col in chunk.columns:
            if col in TEXT_COLS:          # high-cardinality free text: skip set tracking
                continue
            vals = mask_placeholders(chunk[col]).dropna().unique()
            uniques.setdefault(col, set()).update(vals.tolist())
    return {
        "n_rows": n_rows,
        "default_read_mem_bytes": default_mem,
        "n_unique": {c: len(v) for c, v in uniques.items()},
        "categories": {c: sorted(v) for c, v in uniques.items()},
    }


def build_dtype_plan(profile: dict) -> dict[str, object]:
    """Decide the target dtype for every column from the profiling pass."""
    plan: dict[str, object] = {}
    n_rows = profile["n_rows"]
    for col, n_unique in profile["n_unique"].items():
        if col in BOOL_COLS or col in DATE_COLS or col in INT_COLS:
            continue
        if n_unique <= CATEGORY_MAX_RATIO * n_rows:
            # Fixed category list => identical dtype (and Arrow schema) in every chunk.
            plan[col] = pd.CategoricalDtype(categories=profile["categories"][col])
        else:
            plan[col] = "string"
    for col in TEXT_COLS:
        plan[col] = "string"
    for col in BOOL_COLS:
        plan[col] = "bool"
    for col in DATE_COLS:
        plan[col] = "datetime64[s]"
    plan["source_id"] = "int32"
    plan["state_registry_number"] = "Int32"
    return plan


def optimize_chunk(chunk: pd.DataFrame, plan: dict[str, object]) -> pd.DataFrame:
    """Raw object chunk -> optimised dtypes (placeholders become NA first)."""
    out = {}
    for col in chunk.columns:
        target = plan.get(col, "string")
        raw = mask_placeholders(chunk[col])
        if col in BOOL_COLS:
            mapped = raw.map({"t": True, "f": False, "true": True, "false": False})
            out[col] = mapped.astype("boolean") if mapped.isna().any() else mapped.astype(bool)
        elif col in DATE_COLS:
            out[col] = pd.to_datetime(raw, format="%Y-%m-%d", errors="coerce").astype("datetime64[s]")
        elif col == "source_id":
            out[col] = pd.to_numeric(raw, errors="coerce").astype("int32")
        elif col == "state_registry_number":
            # stored as '174547.0' in the CSV -> nullable integer
            out[col] = pd.to_numeric(raw, errors="coerce").round().astype("Int32")
        elif isinstance(target, pd.CategoricalDtype):
            out[col] = raw.astype(target)
        else:
            out[col] = raw.astype("string")
    return pd.DataFrame(out, index=chunk.index)


# --------------------------------------------------------------------------
# 3. CSV -> Parquet (streaming) with memory measurements
# --------------------------------------------------------------------------
def csv_to_parquet(plan: dict[str, object], src: Path = config.RAW_CSV,
                   dest: Path = config.FULL_PARQUET, compression: str = "zstd") -> dict:
    """Stream the CSV into one Parquet file (one row group per chunk)."""
    mem_object = 0
    mem_optimized_chunks = 0
    n_rows = 0
    n_chunks = 0
    writer: pq.ParquetWriter | None = None
    schema: pa.Schema | None = None
    t0 = time.perf_counter()
    tmp = dest.with_suffix(".parquet.part")
    try:
        for chunk in iter_raw_chunks(src):
            mem_object += int(chunk.memory_usage(deep=True).sum())
            opt = optimize_chunk(chunk, plan)
            mem_optimized_chunks += int(opt.memory_usage(deep=True).sum())
            if schema is None:
                schema = pa.Schema.from_pandas(opt, preserve_index=False)
                writer = pq.ParquetWriter(tmp, schema, compression=compression)
            table = pa.Table.from_pandas(opt, schema=schema, preserve_index=False)
            writer.write_table(table)
            n_rows += len(opt)
            n_chunks += 1
    except (OSError, pa.ArrowException) as exc:
        if writer is not None:
            writer.close()
            writer = None
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"writing {dest} failed after {n_rows} rows: {exc}") from exc
    finally:
        if writer is not None:
            writer.close()
    try:
        tmp.replace(dest)                 # atomic: readers never see a half-written file
    except OSError as exc:
        raise RuntimeError(f"could not move {tmp} to {dest}: {exc}") from exc
    return {
        "rows": n_rows,
        "chunks": n_chunks,
        "chunksize": config.CHUNKSIZE,
        "object_dtype_mem_bytes_sum_chunks": mem_object,
        "optimized_mem_bytes_sum_chunks": mem_optimized_chunks,
        "write_time_s": round(time.perf_counter() - t0, 2),
        "parquet_compression": compression,
    }


def load_corpus(columns: list[str] | None = None) -> pd.DataFrame:
    """Read the optimised full corpus (optionally only some columns)."""
    try:
        return pd.read_parquet(config.FULL_PARQUET, columns=columns)
    except (OSError, FileNotFoundError) as exc:
        raise FileNotFoundError(f"{config.FULL_PARQUET} missing - run `python -m src.data_io` first") from exc


def load_dataset(columns: list[str] | None = None) -> pd.DataFrame:
    """Read the labelled + split model dataset."""
    try:
        return pd.read_parquet(config.DATASET_PARQUET, columns=columns)
    except (OSError, FileNotFoundError) as exc:
        raise FileNotFoundError(f"{config.DATASET_PARQUET} missing - run `python -m src.data_io` first") from exc


# --------------------------------------------------------------------------
# 4. CSV vs Parquet benchmark (each case in a fresh process for clean peak RSS)
# --------------------------------------------------------------------------
def _rss_bytes() -> int:
    """Current resident set size of this process."""
    try:
        import psutil
        return int(psutil.Process().memory_info().rss)
    except ImportError:
        with open("/proc/self/statm") as fh:
            return int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")


def _peak_rss_bytes() -> int:
    """Peak RSS (VmHWM) of the *current address space*.

    Unlike getrusage().ru_maxrss, VmHWM is not inherited across exec, so a
    spawned child reports its own peak, and it can be reset (see below)."""
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def _reset_peak_rss() -> None:
    """Reset VmHWM to the current RSS (Linux: write '5' to clear_refs)."""
    try:
        with open("/proc/self/clear_refs", "w") as fh:
            fh.write("5")
    except OSError:
        pass


def _bench_worker(case: str, columns: list[str] | None, repeats: int, queue) -> None:
    """Child process: time `repeats` reads, report median time and peak RSS delta."""
    import gc
    import warnings
    warnings.simplefilter("ignore")      # DtypeWarning on date_published in the plain CSV read
    try:
        times, peaks, bases = [], [], []
        n_rows = n_cols = 0
        frame_mem = None
        for _ in range(repeats):
            gc.collect()
            _reset_peak_rss()
            bases.append(_rss_bytes())
            t0 = time.perf_counter()
            if case == "csv":
                df = pd.read_csv(config.RAW_CSV, usecols=columns)
            elif case == "csv_chunked":
                n_rows = 0
                for ch in pd.read_csv(config.RAW_CSV, usecols=columns, chunksize=config.CHUNKSIZE):
                    n_rows += len(ch)
                    n_cols = ch.shape[1]
                df = None
            else:
                df = pd.read_parquet(config.FULL_PARQUET, columns=columns)
            times.append(time.perf_counter() - t0)
            peaks.append(_peak_rss_bytes())
            if df is not None:
                n_rows, n_cols = df.shape
                frame_mem = int(df.memory_usage(deep=True).sum())
            del df
        queue.put({"ok": True, "times": times, "peak_rss": max(peaks),
                   "peak_delta": max(p - b for p, b in zip(peaks, bases)), "base_rss": bases[0],
                   "rows": n_rows, "cols": n_cols, "frame_mem": frame_mem})
    except Exception as exc:  # report the failure instead of hanging the parent
        queue.put({"ok": False, "error": repr(exc)})


def benchmark_io(repeats: int = 3) -> dict:
    """Benchmark CSV vs Parquet: file size, read time (full / column subset), peak memory."""
    ctx = mp.get_context("spawn")
    cases = [
        ("csv", "full", None),
        ("csv", "5 cols", BENCH_COLS),
        ("csv_chunked", "full", None),
        ("parquet", "full", None),
        ("parquet", "5 cols", BENCH_COLS),
    ]
    results = []
    for fmt, label, cols in cases:
        queue = ctx.Queue()
        proc = ctx.Process(target=_bench_worker, args=(fmt, cols, repeats, queue))
        proc.start()
        try:
            res = queue.get(timeout=900)
        except Exception as exc:
            res = {"ok": False, "error": repr(exc)}
        proc.join()
        if not res.get("ok"):
            print(f"[data_io] benchmark {fmt}/{label} failed: {res.get('error')}")
            results.append({"format": fmt, "columns": label, "error": res.get("error")})
            continue
        row = {
            "format": fmt,
            "columns": label,
            "read_time_s_median": round(statistics.median(res["times"]), 3),
            "read_times_s": [round(t, 3) for t in res["times"]],
            "peak_rss_mb": _mb(res["peak_rss"]),
            "peak_rss_delta_mb": _mb(res["peak_delta"]),
            "frame_mem_mb": None if res["frame_mem"] is None else _mb(res["frame_mem"]),
            "rows": int(res["rows"]),
            "cols": int(res["cols"]),
        }
        print(f"[data_io] bench {fmt:12s} {label:7s} {row['read_time_s_median']:7.3f}s "
              f"peakΔ={row['peak_rss_delta_mb']} MB")
        results.append(row)
    return {
        "csv_size_mb": _mb(config.RAW_CSV.stat().st_size),
        "parquet_size_mb": _mb(config.FULL_PARQUET.stat().st_size),
        "size_ratio_csv_over_parquet": round(config.RAW_CSV.stat().st_size / config.FULL_PARQUET.stat().st_size, 2),
        "subset_columns": BENCH_COLS,
        "repeats": repeats,
        "notes": ("csv = pd.read_csv with pandas-3 default dtype inference (Arrow-backed str); "
                  "csv_chunked = streamed with chunksize, frame never fully materialised; "
                  "parquet = optimised dtypes (category/bool/datetime). Each case runs in a fresh "
                  "spawned process; peak_rss_delta = VmHWM (reset via /proc/self/clear_refs before "
                  "every read) minus RSS just before the read, max over repeats."),
        "cases": results,
    }


# --------------------------------------------------------------------------
# 5. Labelled dataset
# --------------------------------------------------------------------------
def split_class_distribution(ds: pd.DataFrame) -> dict:
    """Per task: counts of every class in every split (+ warning flags)."""
    out: dict = {}
    for task, col in config.TASKS.items():
        sub = ds.dropna(subset=[col])
        table = pd.crosstab(sub[col].astype(str), sub["split"].astype(str))
        out[task] = {cls: {sp: int(v) for sp, v in row.items()} for cls, row in table.iterrows()}
    return out


def class_distribution_warnings(dist: dict, min_eval: int = 30) -> list[str]:
    """Flag classes with too few val/test examples for a stable per-class F1."""
    warns = []
    for task, classes in dist.items():
        for cls, counts in classes.items():
            for split in ("val", "test"):
                n = counts.get(split, 0)
                if n < min_eval:
                    warns.append(f"{task}: class '{cls}' has only {n} rows in {split}")
    return warns


def build_and_save_dataset() -> dict:
    """Run labels.build_dataset on the corpus and save the model input table."""
    cols = ["law_id", "title", "act_form", "database_section", "date_adopted"]
    corpus = load_corpus(cols)
    # Plain object columns: labels.py maps python functions over these values and
    # NA must reach them (Categorical.map could skip NA).
    corpus["act_form"] = corpus["act_form"].astype(object)
    corpus["database_section"] = corpus["database_section"].astype(object)
    ds, stats = labels.build_dataset(corpus)
    del corpus
    # Contract checks: a missing section must give a missing area label, a missing
    # act_form a missing form label (both rows are simply excluded from that task).
    assert not (ds["database_section"].isna() & ds["area_label"].notna()).any(), "area label without section"
    stats["form_label_missing_by_split"] = ds.groupby("split", observed=True)["form_label"].apply(
        lambda s: int(s.isna().sum())).to_dict()
    tmp = config.DATASET_PARQUET.with_suffix(".parquet.part")
    try:
        ds.to_parquet(tmp, index=False)
        tmp.replace(config.DATASET_PARQUET)
    except (OSError, ValueError) as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"could not write {config.DATASET_PARQUET}: {exc}") from exc
    dist = split_class_distribution(ds)
    stats["class_distribution"] = dist
    stats["warnings"] = class_distribution_warnings(dist)
    stats["area_missing_by_split"] = ds.groupby("split", observed=True)["area_label"].apply(
        lambda s: int(s.isna().sum())).to_dict()
    stats["year_range_by_split"] = {
        sp: [int(g["year"].min()), int(g["year"].max())] for sp, g in ds.groupby("split", observed=True)
    }
    stats["columns"] = list(ds.columns)
    _save_json(stats, SPLIT_STATS_JSON)
    try:
        READY_MARKER.touch()
    except OSError as exc:
        print(f"[data_io] could not create READY marker: {exc}")
    return stats


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def run(benchmark: bool = True, repeats: int = 3) -> dict:
    """Full ingestion pipeline. Returns a small summary dict."""
    t0 = time.perf_counter()
    download_if_missing()
    READY_MARKER.unlink(missing_ok=True)

    profile = profile_columns()
    plan = build_dtype_plan(profile)
    conv = csv_to_parquet(plan)

    # "After" memory measured on the real, fully materialised optimised frame.
    full = load_corpus()
    mem_after = int(full.memory_usage(deep=True).sum())
    per_col_after = {c: _mb(v) for c, v in full.memory_usage(deep=True, index=False).items()}
    dtypes_after = {c: str(t) for c, t in full.dtypes.items()}
    del full

    memory = {
        "rows": conv["rows"],
        "object_dtype_mb": _mb(conv["object_dtype_mem_bytes_sum_chunks"]),
        "pandas_default_mb": _mb(profile["default_read_mem_bytes"]),
        "optimized_full_frame_mb": _mb(mem_after),
        "optimized_sum_of_chunks_mb": _mb(conv["optimized_mem_bytes_sum_chunks"]),
        "reduction_vs_object_pct": round(100 * (1 - mem_after / conv["object_dtype_mem_bytes_sum_chunks"]), 1),
        "reduction_vs_pandas_default_pct": round(100 * (1 - mem_after / profile["default_read_mem_bytes"]), 1),
        "how_measured": ("object/default: DataFrame.memory_usage(deep=True) summed over "
                         f"{conv['chunks']} chunks of {config.CHUNKSIZE} rows (never the full object frame); "
                         "optimized: memory_usage(deep=True) of the full frame read back from Parquet."),
        "n_unique": profile["n_unique"],
        "dtypes_after": dtypes_after,
        "per_column_mb_after": per_col_after,
        "csv_to_parquet_write_time_s": conv["write_time_s"],
    }
    _save_json(memory, MEMORY_JSON)
    print(f"[data_io] memory: object {memory['object_dtype_mb']} MB | pandas default "
          f"{memory['pandas_default_mb']} MB | optimised {memory['optimized_full_frame_mb']} MB")

    # Dataset first (models agent is waiting for it), benchmark afterwards.
    stats = build_and_save_dataset()
    print(f"[data_io] dataset: {stats['split_sizes']} -> {config.DATASET_PARQUET}")
    for w in stats["warnings"]:
        print(f"[data_io] WARNING {w}")

    bench = {}
    if benchmark:
        bench = benchmark_io(repeats=repeats)
        bench["memory"] = {k: memory[k] for k in ("object_dtype_mb", "pandas_default_mb",
                                                  "optimized_full_frame_mb", "reduction_vs_object_pct")}
        _save_json(bench, IO_BENCH_JSON)

    summary = {
        "rows": conv["rows"],
        "memory_mb": {k: memory[k] for k in ("object_dtype_mb", "pandas_default_mb", "optimized_full_frame_mb")},
        "csv_mb": _mb(config.RAW_CSV.stat().st_size),
        "parquet_mb": _mb(config.FULL_PARQUET.stat().st_size),
        "split_sizes": stats["split_sizes"],
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }
    print(f"[data_io] done in {summary['elapsed_s']} s")
    return summary


if __name__ == "__main__":
    run()
