"""Our own byte-level BPE tokenizer for Kazakh legal titles.

* Trained with HuggingFace `tokenizers` on TRAIN titles only (val/test text is
  never seen, so the vocabulary cannot leak information from the future).
* NFC normalisation + byte-level pre-tokenisation (GPT-2 regex): every Unicode
  character, including the Kazakh-specific letters ә ғ қ ң ө ұ ү һ і, is
  representable, so there are no [UNK] tokens and decoding round-trips exactly.
* Special tokens: [PAD]=0 [UNK]=1 [CLS]=2 [SEP]=3 [MASK]=4; encodings are
  wrapped as "[CLS] title [SEP]".

run() trains an 8k (main) and a 4k vocabulary, compares token lengths and
fertility (tokens per word) with the e5 SentencePiece tokenizer, picks max_len
from the p99 length and writes results/token_lengths.json plus
reports/figures/token_length_hist.png.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.model_utils import (BPE_DIR, E5_MODEL_NAME, E5_PREFIX, TOKEN_STATS_JSON, display_path, load_dataset,
                             load_json, save_json)

SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
PAD_ID, UNK_ID, CLS_ID, SEP_ID, MASK_ID = range(5)
VOCAB_SIZES = (8000, 4000)
MAIN_VOCAB = 8000
FIG_PATH = config.FIGURES_DIR / "token_length_hist.png"


def tokenizer_dir(vocab_size: int) -> Path:
    """artifacts/models/bpe_tokenizer (main 8k) or bpe_tokenizer_<k>k for other sizes."""
    return BPE_DIR if vocab_size == MAIN_VOCAB else BPE_DIR.with_name(f"bpe_tokenizer_{vocab_size // 1000}k")


def build_tokenizer():
    """Untrained BPE pipeline: NFC -> byte-level split -> BPE -> [CLS] x [SEP]."""
    from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers

    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.normalizer = normalizers.NFC()
    # add_prefix_space: the first word gets the same "Ġword" form as words mid-title.
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tok.decoder = decoders.ByteLevel()
    return tok


def train_tokenizer(titles: list[str], vocab_size: int, min_frequency: int = 2):
    """Train BPE on the given (train-only) titles and attach the CLS/SEP template."""
    from tokenizers import pre_tokenizers, processors, trainers

    tok = build_tokenizer()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=min_frequency,
                                  special_tokens=SPECIAL_TOKENS, show_progress=False,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(titles, trainer=trainer, length=len(titles))
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", tok.token_to_id("[CLS]")), ("[SEP]", tok.token_to_id("[SEP]"))])
    assert tok.token_to_id("[PAD]") == PAD_ID and tok.token_to_id("[SEP]") == SEP_ID
    return tok


def save_tokenizer(tok, vocab_size: int) -> Path:
    """Save to artifacts/models/<dir>/tokenizer.json and return the path."""
    out_dir = tokenizer_dir(vocab_size)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "tokenizer.json"
    tok.save(str(path))
    return path


def load_tokenizer(path: Path | str | None = None):
    """Load a saved tokenizer (default: the main 8k one)."""
    from tokenizers import Tokenizer

    path = Path(path) if path else BPE_DIR / "tokenizer.json"
    if path.is_dir():
        path = path / "tokenizer.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run `python -m src.bpe_tokenizer` first")
    return Tokenizer.from_file(str(path))


def encode_titles(tok, titles: list[str], max_len: int, chunk: int = 20_000) -> tuple[np.ndarray, np.ndarray]:
    """Encode to a compact int16 [N, max_len] matrix (PAD=0) + int16 lengths.

    Over-long titles keep their first max_len-1 tokens and the closing [SEP].
    Done in chunks so the Rust Encoding objects never pile up in RAM.
    """
    n = len(titles)
    dtype = np.int16 if tok.get_vocab_size() < 32_767 else np.int32
    ids = np.zeros((n, max_len), dtype=dtype)
    lengths = np.zeros(n, dtype=np.int16)
    for start in range(0, n, chunk):
        for i, enc in enumerate(tok.encode_batch(titles[start:start + chunk]), start=start):
            row = enc.ids
            if len(row) > max_len:
                row = row[:max_len - 1] + [SEP_ID]
            ids[i, :len(row)] = row
            lengths[i] = len(row)
    return ids, lengths


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------
def length_stats(lengths: np.ndarray) -> dict:
    """mean / median / p95 / p99 / max of a length array."""
    lengths = np.asarray(lengths)
    return {"mean": round(float(lengths.mean()), 2), "median": float(np.median(lengths)),
            "p95": float(np.percentile(lengths, 95)), "p99": float(np.percentile(lengths, 99)),
            "max": int(lengths.max()), "n": int(len(lengths))}


def bpe_lengths(tok, titles: list[str], with_special: bool = True, chunk: int = 20_000) -> np.ndarray:
    """Token count per title for our BPE tokenizer."""
    out = []
    for start in range(0, len(titles), chunk):
        encs = tok.encode_batch(titles[start:start + chunk], add_special_tokens=with_special)
        out.extend(len(e.ids) for e in encs)
    return np.array(out, dtype=np.int32)


def e5_lengths(hf_tok, titles: list[str], with_prefix_and_special: bool, chunk: int = 20_000) -> np.ndarray:
    """Token count per title for the e5 SentencePiece tokenizer."""
    out = []
    for start in range(0, len(titles), chunk):
        batch = titles[start:start + chunk]
        if with_prefix_and_special:
            batch = [E5_PREFIX + t for t in batch]
        enc = hf_tok(batch, add_special_tokens=with_prefix_and_special)["input_ids"]
        out.extend(len(x) for x in enc)
    return np.array(out, dtype=np.int32)


def fertility(token_counts: np.ndarray, titles: list[str]) -> dict:
    """Tokens per whitespace word and characters per token (compression)."""
    words = sum(len(t.split()) for t in titles)
    chars = sum(len(t) for t in titles)
    toks = int(np.sum(token_counts))
    return {"tokens_per_word": round(toks / max(words, 1), 3),
            "chars_per_token": round(chars / max(toks, 1), 3),
            "n_words": int(words), "n_tokens": toks}


def roundtrip_errors(tok, titles: list[str]) -> int:
    """Number of titles whose decode(encode(x)) differs from NFC(x)."""
    import unicodedata

    bad = 0
    for start in range(0, len(titles), 20_000):
        batch = titles[start:start + 20_000]
        encs = tok.encode_batch(batch, add_special_tokens=False)
        decoded = tok.decode_batch([e.ids for e in encs])
        for orig, dec in zip(batch, decoded):
            if dec.strip() != unicodedata.normalize("NFC", orig).strip():
                bad += 1
    return bad


def round_up(x: float, multiple: int = 8) -> int:
    """Round up to a multiple of 8 (tensor-core friendly sequence length)."""
    return int(np.ceil(x / multiple) * multiple)


def plot_lengths(series: dict[str, np.ndarray], max_lens: dict[str, int], path: Path = FIG_PATH) -> None:
    """Overlaid histogram of token lengths per tokenizer with the chosen max_len."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    upper = int(max(np.percentile(v, 99.8) for v in series.values())) + 5
    bins = np.arange(0, upper + 2, 2)
    colors = ["#2a6fdb", "#e07b39", "#3a9e5f"]
    for (name, lens), color in zip(series.items(), colors):
        ax.hist(np.clip(lens, 0, upper), bins=bins, alpha=0.45, label=f"{name} (mean {lens.mean():.1f})",
                color=color)
        if name in max_lens:
            ax.axvline(max_lens[name], color=color, ls="--", lw=1.5,
                       label=f"{name} max_len={max_lens[name]} (p99)")
    ax.set_title("Title length in tokens: own BPE vs e5 SentencePiece (all splits)")
    ax.set_xlabel(f"tokens per title (incl. special tokens; values > {upper} shown in the last bin)")
    ax.set_ylabel("number of titles")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def example_tokens(tok, hf_tok, titles: list[str]) -> list[dict]:
    """Show how each tokenizer segments a few titles (readable pieces)."""
    rows = []
    for t in titles:
        ids = tok.encode(t, add_special_tokens=False).ids
        rows.append({"title": t,
                     "bpe_pieces": [tok.decode([i]) for i in ids],
                     "e5_pieces": hf_tok.tokenize(t)})
    return rows


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def run(vocab_sizes: tuple[int, ...] = VOCAB_SIZES, quick: bool = False, force: bool = False) -> dict:
    """Train tokenizers on train titles, compute length/fertility stats, pick max_len.

    quick=True reuses an existing tokenizer + stats if present (smoke test).
    """
    if quick and not force and (BPE_DIR / "tokenizer.json").exists() and TOKEN_STATS_JSON.exists():
        print("[bpe] quick mode: reusing existing tokenizer and token stats")
        return load_json(TOKEN_STATS_JSON)

    t0 = time.time()
    ds = load_dataset(["title", "split"])        # every title, whatever its labels
    train_titles = ds.loc[ds["split"] == "train", "title"].astype(str).tolist()
    heldout_titles = ds.loc[ds["split"] != "train", "title"].astype(str).tolist()
    all_titles = train_titles + heldout_titles
    print(f"[bpe] training on {len(train_titles):,} train titles; held-out {len(heldout_titles):,}")

    from transformers import AutoTokenizer
    hf_tok = AutoTokenizer.from_pretrained(E5_MODEL_NAME)

    report: dict = {"trained_on": "train titles only", "n_train_titles": len(train_titles),
                    "n_heldout_titles": len(heldout_titles), "tokenizers": {}}
    hist_series: dict[str, np.ndarray] = {}
    for vs in vocab_sizes:
        ts = time.time()
        tok = train_tokenizer(train_titles, vs)
        path = save_tokenizer(tok, vs)
        lens_all = bpe_lengths(tok, all_titles)
        lens_train = lens_all[:len(train_titles)]
        lens_held_raw = bpe_lengths(tok, heldout_titles, with_special=False)
        name = f"bpe_{vs // 1000}k"
        report["tokenizers"][name] = {
            "path": display_path(path), "vocab_size": tok.get_vocab_size(),
            "train_time_s": round(time.time() - ts, 2),
            "length_all_with_special": length_stats(lens_all),
            "length_train_with_special": length_stats(lens_train),
            "fertility_heldout": fertility(lens_held_raw, heldout_titles),
            "unk_tokens_heldout": int(sum(e.ids.count(UNK_ID) for e in tok.encode_batch(heldout_titles[:20000]))),
            "roundtrip_errors_heldout": roundtrip_errors(tok, heldout_titles),
        }
        hist_series[f"BPE-{vs // 1000}k"] = lens_all
        print(f"[bpe] {name}: {report['tokenizers'][name]['length_all_with_special']} "
              f"fertility={report['tokenizers'][name]['fertility_heldout']['tokens_per_word']}")

    e5_all = e5_lengths(hf_tok, all_titles, with_prefix_and_special=True)
    e5_held_raw = e5_lengths(hf_tok, heldout_titles, with_prefix_and_special=False)
    report["tokenizers"]["e5_sentencepiece"] = {
        "model": E5_MODEL_NAME, "vocab_size": hf_tok.vocab_size,
        "length_all_with_prefix_special": length_stats(e5_all),
        "length_train_with_prefix_special": length_stats(e5_all[:len(train_titles)]),
        "fertility_heldout": fertility(e5_held_raw, heldout_titles),
    }
    hist_series["e5 (with 'query: ')"] = e5_all

    # max_len = p99 of TRAIN lengths (rounded up to a multiple of 8); longer titles are truncated.
    main = f"bpe_{MAIN_VOCAB // 1000}k"
    chosen = {
        "bpe": round_up(report["tokenizers"][main]["length_train_with_special"]["p99"]),
        "e5": round_up(report["tokenizers"]["e5_sentencepiece"]["length_train_with_prefix_special"]["p99"]),
    }
    for vs in vocab_sizes:
        key = f"bpe_{vs // 1000}k"
        chosen[key] = round_up(report["tokenizers"][key]["length_train_with_special"]["p99"])
    report["chosen_max_len"] = chosen
    report["truncated_share"] = {
        "bpe": round(float((hist_series[f"BPE-{MAIN_VOCAB // 1000}k"] > chosen["bpe"]).mean()), 4),
        "e5": round(float((e5_all > chosen["e5"]).mean()), 4)}
    main_tok = load_tokenizer(BPE_DIR)
    report["examples"] = example_tokens(main_tok, hf_tok, heldout_titles[:3])
    report["total_time_s"] = round(time.time() - t0, 1)
    save_json(report, TOKEN_STATS_JSON)
    plot_lengths(hist_series, {f"BPE-{MAIN_VOCAB // 1000}k": chosen["bpe"],
                               "e5 (with 'query: ')": chosen["e5"]})
    print(f"[bpe] max_len chosen from p99: {chosen}; saved {TOKEN_STATS_JSON.name} and {FIG_PATH.name}")
    return {"chosen_max_len": chosen, "fertility": {k: v["fertility_heldout"]["tokens_per_word"]
                                                     for k, v in report["tokenizers"].items()}}


if __name__ == "__main__":
    run()
