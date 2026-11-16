# Tiny Transformer for Kazakh Legal Act Classification

Final project for **Programming for AI** (Astana IT University, MSc Applied AI), topic **#24**, cluster 5: *Lightweight Transformer Models for Kazakh Text*.
Team AAI-2603M: Ilyas Nagiyev, Beksultan Aliyev, Yernur Baimukhambetov.

We classify Kazakh normative legal acts **from their titles only**, on two tasks:

- **`form`**, the act form (7 classes): Заң, Жарлық, Қаулы, Шешім, Бұйрық, Өкім, Басқа.
- **`area`**, the regulation area (33 classes).

We compare three kinds of model:

- a **tiny transformer trained from scratch** (own BPE tokenizer, ≤10M parameters);
- **TF-IDF + logistic regression**;
- **fine-tuned compact pretrained encoders** (≤120M parameters: multilingual-e5-small, Kazakh RoBERTa, KazBERT).

The comparison reports macro-F1, parameter count, model size and latency, on a temporal, leakage-controlled split.

The dataset is [KazakhLawCorpus-clean](https://huggingface.co/datasets/Arailym-tleubayeva/KazakhLawCorpus-clean): 215,889 acts, a 337 MB CSV, CC BY 4.0.

## Key results

The tables show macro-F1 on the 2023–2026 test acts and on **test_strict**, which drops test titles that have a templated near-duplicate in the training years. The CPU column is the time to classify 1000 titles on CPU.

**Act form (7 classes)**

| Model | Params | CPU ms / 1000 | Test | Strict | vs TF-IDF (95 % CI) |
|---|---|---|---|---|---|
| TF-IDF word+char + LR (baseline) | 1.5M | ~190 | 0.820 | 0.825 | — |
| Tiny transformer, from scratch (2L, d128) | 1.4M | ~150 | 0.810 | 0.809 | tie |
| **Tiny transformer + MLM + distillation** | **5.2M** | **~540** | **0.827** | **0.829** | tie |
| e5-small fine-tuned | 117.7M | ~3,700 | 0.842 | 0.846 | better |
| e5-small, Kazakh vocab-pruned | 25.5M | — | 0.845 | 0.852 | better |
| kaz-roberta fine-tuned | 83.5M | ~9,700 | 0.853 | 0.868 | better |
| **KazBERT fine-tuned** | 110.6M | — | **0.860** | 0.866 | better |

**Regulation area (33 classes)**

| Model | Params | CPU ms / 1000 | Test | Strict | vs TF-IDF (95 % CI) |
|---|---|---|---|---|---|
| TF-IDF word+char + LR (baseline) | 7.1M | ~230 | 0.583 | 0.556 | — |
| Tiny transformer, from scratch (4L) | 5.2M | ~500 | 0.584 | 0.564 | tie |
| **Tiny transformer + MLM + distillation** | **5.2M** | **~520** | **0.598** | **0.576** | better |
| e5-small fine-tuned | 117.7M | ~3,500 | 0.601 | 0.579 | better |
| **kaz-roberta + LLRD + recency** | 83.5M | ~9,800 | **0.629** | **0.608** | better |

The full tables, with every variant, latency, memory and bootstrap CIs, are in `results/model_comparison.md` and `results/ablations.md`. The written report is `reports/report.md` (also `.docx`), and the slides are `reports/presentation.pptx`.

**Takeaways**

- The tiny transformer needs MLM pretraining and distillation to catch the TF-IDF baseline. With them, it matches the 23× larger e5-small on the area task at about 1/7 of its CPU latency.
- Kazakh-specific encoders are the most accurate models within the 120M-parameter budget.
- Kazakh vocabulary pruning shrinks e5-small by 78 % at equal F1.

## Setup

Tested on Ubuntu with Python 3.12 and one NVIDIA RTX 5090 (CUDA 12.8). Training on CPU works too, just slower.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128   # or the CPU wheel
pip install -r requirements.txt
```

The dataset CSV is downloaded automatically to `data/raw/` on the first run (menu item 1 or `run_pipeline.py --steps data`).

## How to run

**Interactive console menu** (requirement 7):

```bash
python main.py
```

| # | Action | # | Action |
|---|---|---|---|
| 1 | Prepare data (download, Parquet, labels, split) | 8 | Fine-tune e5-small (+ frozen e5 + LR) |
| 2 | Data-quality audit | 9 | Compare models (benchmark table + figures) |
| 3 | Build SQLite + run the SQL queries | 10 | Error analysis |
| 4 | EDA figures | 11 | **Classify your own title** with every trained model |
| 5 | Leakage check (near-duplicates, test_strict) | 12 | Distillation / recency weighting for the tiny model |
| 6 | Train TF-IDF + LR baselines | 13 | Prior-shift adjustment + ensembles |
| 7 | Train BPE tokenizer + tiny transformer | 0 | Exit |

**Whole pipeline, non-interactive:**

```bash
python run_pipeline.py --steps all            # full reproduction
python run_pipeline.py --steps all --quick    # smoke test in a few minutes (writes to a temp folder)
python run_pipeline.py --steps tiny,compare   # selected steps
```

Every module can also be run alone, e.g. `python -m src.leakage`. Each module exposes `run()`.

Measured wall-clock times on the machine above: data, audit, SQL and EDA take about 1 min. The leakage check takes about 3.5 min, the baselines about 7 min, each tiny-transformer run 0.5–1 min, and each e5 or kaz-roberta fine-tune 1.5–5 min.

## Project structure

```
main.py                 interactive console menu
run_pipeline.py         non-interactive reproduction (argparse)
src/
  config.py             paths, constants, label definitions, split years
  labels.py             label construction + temporal, leakage-free split
  metrics.py            shared macro-F1 / accuracy (same label set for every model)
  experiments_db.py     SQLite logging of experiments + predictions
  data_io.py            chunked CSV -> Parquet, dtype optimisation, memory + I/O benchmark
  audit.py              data-quality audit (missing, placeholders, dates, duplicates)
  database.py           normalised SQLite schema, ER diagram, SQL query runner
  eda.py                EDA figures and data-backed findings
  leakage.py            exact / digit-masked / near-duplicate analysis, test_strict
  text_prep.py          Kazakh normalisation, stop words, suffix stemmer
  baseline.py           majority, TF-IDF (word / char / stem / word+char) + LR
  bpe_tokenizer.py      own byte-level BPE (train titles only)
  tiny_transformer.py   transformer encoder written from scratch (own attention)
  mlm_pretrain.py       masked-LM pretraining of the tiny encoder
  distill.py            knowledge distillation (teacher -> tiny student), recency weighting
  finetune_e5.py        multilingual-e5-small fine-tuning + frozen-embedding baseline
  vocab_prune.py        Kazakh vocabulary pruning of e5-small (118M -> 25.5M)
  seq2seq_mt5.py        mT5-small label generation with constrained decoding
  model_survey.py       survey + fine-tuning of compact Kazakh encoders
  roberta_tune.py       kaz-roberta improvements (LLRD, recency, train+val refit)
  posthoc.py            prior-shift adjustment + ensembles
  benchmark.py          params / size / CPU+GPU latency / memory, one protocol for all
  compare.py            final comparison tables and figures from SQLite
  error_analysis.py     confusion matrices, per-class F1, error slices
  inference.py, predict.py   model loading + "classify your own title"
sql/                    schema.sql, indexes.sql, queries.sql (17 queries)
results/                JSON / CSV / Markdown results (comparison, audit, leakage, SQL outputs ...)
reports/                report.md / report.docx, presentation, figures/
docs/                   project brief, ER diagram (Mermaid)
```

Generated and not in git: `data/raw/` (CSV), `data/processed/` (Parquet), `data/kazlaw.db` (SQLite) and `artifacts/models/` (trained models). All of them are rebuilt by the pipeline.

## How the course requirements are met

| # | Requirement | Where |
|---|---|---|
| 1–4 | functions, lists/dicts, if/elif/else, loops | throughout `src/`; menu loop in `main.py` |
| 5 | file I/O (CSV, JSON, Parquet, SQLite) | `data_io.py`, `results/`, `experiments_db.py` |
| 6 | try/except | user input in `main.py`, downloads and I/O in `data_io.py`, model loading in `inference.py` |
| 7 | console menu | `main.py` |
| 8 | modularity | 30 modules in `src/`, stdlib `pathlib`, `json`, `datetime`, `os` |
| 9 | comparison of approaches + baseline (quality, time, memory) | `results/model_comparison.md`, `results/io_benchmark.json`, `results/text_prep_comparison.json` |
| 10 | SQLite, ER diagram, ≥ 8 SQL queries | `sql/`, `reports/figures/er_diagram.png`, `results/sql/` |
| 11 | large data: chunksize, dtypes, Parquet, memory measurement | `data_io.py`, `results/memory_optimization.json` |

## Reproducibility notes

- **Seeds:** the seed is fixed (`config.SEED = 42`). GPU kernels are not fully deterministic, so expect differences in the third decimal.
- **Leakage-safe fitting:** all tokenizers, vocabularies, vocab pruning and teacher signals are fitted on **train titles only**.
- **Model selection:** all choices are made on the 2022 validation year. Test scores are reported once.
- **`test_strict`:** 17,629 of the 38,471 test titles. It excludes titles with a digit-masked near-duplicate (char n-gram cosine ≥ 0.9) in the training set. See `results/leakage_report.json`.
