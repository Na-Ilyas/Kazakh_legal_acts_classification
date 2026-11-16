"""Kazakh text preprocessing for the classical (TF-IDF) models.

* ``normalize``      - NFC, lower-case, Latin->Cyrillic homoglyph folding inside
                       Cyrillic words (the corpus writes Kazakh 'і' as Latin 'i'
                       in ~11 % of titles: 'жөнiндегi' vs 'жөніндегі').
* ``tokenize``       - regex word tokenizer keeping Kazakh letters
                       (әғқңөұүһі); digit runs become the token 'NUM'.
* ``KAZAKH_STOPWORDS`` - general function words + frequent legal boilerplate;
                       ``stopword_report`` justifies them with train document
                       frequency and how far their class distribution is from
                       the prior (KL divergence).
* ``stem``           - rule-based light inflectional stemmer: iteratively strips
                       case, possessive and plural endings (vowel-harmony
                       variants) with a minimum stem length and final-consonant
                       restoration (бұйрығы -> бұйрығ -> бұйрық).
* ``run()``          - compares word forms / word forms without stop words /
                       stems / char n-grams: vocabulary size, OOV of test
                       tokens w.r.t. the train vocabulary, and the downstream
                       macro-F1 of the matching baseline (if already logged).
"""
from __future__ import annotations

import json
import math
import re
import time
import unicodedata
from collections import Counter
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from src import config
from src.data_io import load_dataset

COMPARISON_JSON = config.RESULTS_DIR / "text_prep_comparison.json"

KAZ_LETTERS = "аәбвгғдеёжзийкқлмнңоөпрстуұүфхһцчшщъыіьэюя"
_TOKEN_RE = re.compile(rf"[{KAZ_LETTERS}a-z]+|\d+")
_CYR_RE = re.compile(rf"[{KAZ_LETTERS}]")
VOWELS = set("аәеёиоөуұүыіэюя")

# Latin look-alikes -> Cyrillic (applied only inside words that contain Cyrillic)
HOMOGLYPHS = str.maketrans({"i": "і", "a": "а", "e": "е", "o": "о", "p": "р", "c": "с", "x": "х",
                            "y": "у", "k": "к", "m": "м", "h": "һ", "t": "т", "b": "в"})

# --------------------------------------------------------------------------
# Stop words
# --------------------------------------------------------------------------
GENERAL_STOPWORDS = {
    # conjunctions / postpositions / particles / pronouns
    "және", "мен", "бен", "пен", "да", "де", "та", "те", "немесе", "не", "әрі", "бірақ", "үшін",
    "туралы", "бойынша", "арналған", "жөніндегі", "жөнінде", "қатысты", "байланысты", "сәйкес",
    "арасындағы", "арасында", "кейбір", "барлық", "осы", "сол", "бұл", "ол", "оның", "олардың", "өз",
    "өзінің", "тиісті", "дейін", "кейін", "бастап", "соң", "қоса", "ретінде", "ішінде", "ма", "ме",
}
LEGAL_BOILERPLATE = {
    # appear in a large share of titles of every form (see stopword_report)
    "қазақстан", "республикасы", "республикасының", "республикасындағы", "жылғы", "жылы",
    "n", "num", "нөмірі", "күні",
}
KAZAKH_STOPWORDS = frozenset(GENERAL_STOPWORDS | LEGAL_BOILERPLATE)

# --------------------------------------------------------------------------
# Stemmer
# --------------------------------------------------------------------------
MIN_STEM = 3
_CASE = [
    "дан", "ден", "тан", "тен", "нан", "нен",                       # ablative
    "ның", "нің", "дың", "дің", "тың", "тің",                       # genitive
    "ға", "ге", "қа", "ке", "на", "не",                             # dative
    "да", "де", "та", "те", "нда", "нде",                           # locative
    "ны", "ні", "ды", "ді", "ты", "ті",                             # accusative
    "мен", "бен", "пен",                                            # instrumental
    "дағы", "дегі", "тағы", "тегі", "ндағы", "ндегі",                # locative + relative -ғы
]
_POSS_CASE = [  # possessive 3sg + case, e.g. облыс-ы-ның, қала-сы-на
    "ының", "інің", "сының", "сінің", "ына", "іне", "сына", "сіне", "ында", "інде", "сында", "сінде",
    "ынан", "інен", "сынан", "сінен", "ын", "ін", "сын", "сін", "ындағы", "індегі", "сындағы", "сіндегі",
]
# only 3rd-person possessive: legal titles never use 1st/2nd person, and '-ім' would
# wrongly cut 'шешім' (decision) -> 'шеш'
_POSS = ["ы", "і", "сы", "сі"]
_PLURAL = ["лар", "лер", "дар", "дер", "тар", "тер"]
SUFFIXES = sorted(set(_CASE + _POSS_CASE + _POSS + _PLURAL), key=len, reverse=True)
_SUFFIX_BY_LEN: dict[int, set[str]] = {}
for _s in SUFFIXES:
    _SUFFIX_BY_LEN.setdefault(len(_s), set()).add(_s)
_S_POSSESSIVE = {x for x in _POSS + _POSS_CASE if x.startswith("с")}
_FINAL_RESTORE = {"ғ": "қ", "г": "к", "б": "п"}          # voicing alternation before vowel suffixes


def _allowed(stem: str, suffix: str) -> bool:
    """Phonotactic guard: 'с'-initial possessives follow vowels, 'ы/і'-initial follow consonants."""
    if len(stem) < MIN_STEM:
        return False
    if suffix in _S_POSSESSIVE:
        return stem[-1] in VOWELS
    if suffix[0] in "ыі":
        return stem[-1] not in VOWELS
    return True


STEM_EXCEPTIONS = {"сауда", "кеден", "шешім", "заң", "қала"}   # look inflected, are roots
LEXICON_PATH = config.ARTIFACTS_DIR / "text_prep" / "lexicon.txt"
LEXICON_MIN_COUNT = 3
_LEXICON: frozenset[str] | None = None


def build_lexicon(train_titles: list[str], min_count: int = LEXICON_MIN_COUNT) -> frozenset[str]:
    """Known word forms = train tokens seen >= min_count times (train only: no leakage). Saved to disk."""
    counts = Counter(t for title in train_titles for t in tokenize(title))
    lex = frozenset(w for w, c in counts.items() if c >= min_count)
    try:
        LEXICON_PATH.parent.mkdir(parents=True, exist_ok=True)
        LEXICON_PATH.write_text("\n".join(sorted(lex)), encoding="utf-8")
    except OSError as exc:
        print(f"[text_prep] could not save lexicon: {exc}")
    set_lexicon(lex)
    return lex


def set_lexicon(lex: frozenset[str] | None) -> None:
    """Install a lexicon (clears the stem cache)."""
    global _LEXICON
    _LEXICON = lex
    stem.cache_clear()


def get_lexicon() -> frozenset[str]:
    """Lazy-load the saved lexicon (empty set -> pure rule-based stemming)."""
    global _LEXICON
    if _LEXICON is None:
        try:
            _LEXICON = frozenset(LEXICON_PATH.read_text(encoding="utf-8").split("\n"))
        except OSError:
            _LEXICON = frozenset()
    return _LEXICON


def _candidates(word: str) -> list[str]:
    """All allowed one-suffix strips of `word` (longest suffix first), with consonant restoration."""
    out = []
    for length in sorted(_SUFFIX_BY_LEN, reverse=True):
        if len(word) - length < MIN_STEM:
            continue
        suf = word[-length:]
        if suf in _SUFFIX_BY_LEN[length] and _allowed(word[:-length], suf):
            cand = word[:-length]
            if suf[0] in VOWELS and cand[-1] in _FINAL_RESTORE:
                cand = cand[:-1] + _FINAL_RESTORE[cand[-1]]
            out.append(cand)
    return out


@lru_cache(maxsize=None)
def stem(word: str, max_rounds: int = 4) -> str:
    """Lexicon-aware iterative suffix stripping (cached per word).

    Each round: among the allowed strips, take the longest one whose result is a
    known train word form (resolves облы+сының vs облыс+ының). If none is known
    and the word itself is known, stop (protects roots ending like a suffix,
    e.g. 'қаулы'); if the word is unknown (OOV), take the longest strip.
    """
    if len(word) <= MIN_STEM or word in STEM_EXCEPTIONS or not _CYR_RE.search(word):
        return word
    lex = get_lexicon()
    for _ in range(max_rounds):
        cands = _candidates(word)
        if not cands:
            break
        known = [c for c in cands if c in lex]
        if known:
            word = known[0]
        elif word in lex:
            break
        else:
            word = cands[0]
        if word in STEM_EXCEPTIONS:
            break
    return word


# --------------------------------------------------------------------------
# Normalisation / tokenisation / analyzers (module-level => picklable)
# --------------------------------------------------------------------------
def _fold_token(tok: str) -> str:
    """Fold Latin homoglyphs only in tokens that also contain Cyrillic letters."""
    return tok.translate(HOMOGLYPHS) if _CYR_RE.search(tok) and re.search(r"[a-z]", tok) else tok


def normalize(text: str) -> str:
    """NFC + lower-case + homoglyph folding inside Cyrillic words."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text).lower()
    return re.sub(r"\w+", lambda m: _fold_token(m.group(0)), text)


def tokenize(text: str) -> list[str]:
    """Regex word tokens of normalised text; numbers -> 'num'."""
    return ["num" if t.isdigit() else t for t in _TOKEN_RE.findall(normalize(text))]


def remove_stopwords(tokens: list[str]) -> list[str]:
    """Drop Kazakh stop words / legal boilerplate."""
    return [t for t in tokens if t not in KAZAKH_STOPWORDS]


def analyzer_words(text: str) -> list[str]:
    """Word forms (all tokens) - used by the word TF-IDF baseline via ngram helper."""
    return tokenize(text)


def analyzer_words_nostop(text: str) -> list[str]:
    """Word forms without stop words."""
    return remove_stopwords(tokenize(text))


def analyzer_stems(text: str) -> list[str]:
    """Stems of non-stop-word tokens."""
    return [stem(t) for t in remove_stopwords(tokenize(text))]


def preprocess(text: str, mode: str = "stems") -> str:
    """Space-joined representation ('words' | 'nostop' | 'stems') for vectorisers that need strings."""
    if mode == "words":
        toks = tokenize(text)
    elif mode == "nostop":
        toks = analyzer_words_nostop(text)
    elif mode == "stems":
        toks = analyzer_stems(text)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return " ".join(toks)


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------
def stopword_report(train: pd.DataFrame, label_col: str = "form_label", top: int = 40) -> list[dict]:
    """Document frequency + class-distribution KL(word || prior) of each stop word.

    Low KL = the word is spread like the classes overall -> little information.
    """
    prior = train[label_col].astype(str).value_counts(normalize=True)
    doc_tokens = train["title"].map(lambda t: set(tokenize(t)))
    rows = []
    for word in sorted(KAZAKH_STOPWORDS):
        mask = doc_tokens.map(lambda s, w=word: w in s)
        df_pct = 100 * float(mask.mean())
        if df_pct == 0:
            continue
        dist = train.loc[mask, label_col].astype(str).value_counts(normalize=True)
        kl = sum(p * math.log(p / prior[c]) for c, p in dist.items() if p > 0)
        rows.append({"word": word, "train_doc_freq_pct": round(df_pct, 2), "kl_vs_prior": round(kl, 3)})
    return sorted(rows, key=lambda r: -r["train_doc_freq_pct"])[:top]


def vocab_and_oov(train_texts: list[str], test_texts: list[str], analyzer, min_df: int = 1) -> dict:
    """Vocabulary size on train and OOV rate of test tokens (token and type level)."""
    cv = CountVectorizer(analyzer=analyzer, min_df=min_df, lowercase=False)
    cv.fit(train_texts)
    vocab = cv.vocabulary_
    total, unseen = 0, 0
    types, unseen_types = set(), set()
    for doc in test_texts:
        toks = analyzer(doc)
        total += len(toks)
        for t in toks:
            types.add(t)
            if t not in vocab:
                unseen += 1
                unseen_types.add(t)
    return {"vocab_size": len(vocab), "test_tokens": total,
            "oov_token_rate_pct": round(100 * unseen / max(total, 1), 2),
            "oov_type_rate_pct": round(100 * len(unseen_types) / max(len(types), 1), 2)}


def _char_analyzer():
    """char_wb 3-5-gram analyzer on normalised text (same as the char baseline)."""
    base = CountVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=False).build_analyzer()
    return lambda text: base(normalize(text))


def downstream_f1() -> dict:
    """Macro-F1 of the baselines that correspond to each representation (if logged)."""
    mapping = {"words": "tfidf_word_lr", "stems": "tfidf_stem_lr", "char_3_5": "tfidf_char_lr"}
    try:
        from src import experiments_db
        exp = experiments_db.load_experiments()
    except Exception as exc:  # DB missing or locked
        return {"error": repr(exc)}
    out: dict = {}
    for rep, model in mapping.items():
        sub = exp[exp["model_name"] == model]
        out[rep] = {f"{r.task}/{r.eval_split}": round(float(r.f1_macro), 4) for r in sub.itertuples()}
    return out


def attach_downstream_f1() -> None:
    """Update text_prep_comparison.json with baseline F1 (called by baseline.run)."""
    try:
        data = json.loads(COMPARISON_JSON.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    data["downstream_macro_f1"] = downstream_f1()
    COMPARISON_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def run() -> dict:
    """Compare representations and write results/text_prep_comparison.json."""
    t0 = time.perf_counter()
    ds = load_dataset(["title", "split", "form_label"])
    train = ds[ds["split"] == "train"]
    test = ds[ds["split"] == "test"]
    train_titles, test_titles = train["title"].tolist(), test["title"].tolist()
    lexicon = build_lexicon(train_titles)

    homoglyph_titles = int(ds["title"].map(lambda t: normalize(t) != unicodedata.normalize("NFC", t).lower()).sum())
    reps = {
        "words": analyzer_words,
        "words_no_stopwords": analyzer_words_nostop,
        "stems": analyzer_stems,
        "char_3_5": _char_analyzer(),
    }
    comparison = {}
    for name, analyzer in reps.items():
        t1 = time.perf_counter()
        comparison[name] = vocab_and_oov(train_titles, test_titles, analyzer)
        comparison[name]["min_df_2_vocab_size"] = len(
            CountVectorizer(analyzer=analyzer, min_df=2, lowercase=False).fit(train_titles).vocabulary_)
        comparison[name]["time_s"] = round(time.perf_counter() - t1, 1)
        print(f"[text_prep] {name}: {comparison[name]}")

    examples = ["Қазақстан Республикасы Үкіметінің 2015 жылғы 12 мамырдағы № 330 қаулысына өзгерістер енгізу туралы",
                "Ақтөбе облысы әкімдігінің кейбір қаулыларының күші жойылды деп тану туралы",
                "Жамбыл ауданы мәслихатының 2022 жылғы 23 желтоқсандағы №39-2 шешіміне өзгерістер енгізу туралы",
                "Ветеринария саласындағы мемлекеттік көрсетілетін қызметтер регламенттерін бекіту туралы"]
    stem_examples = [{"title": e, "tokens": tokenize(e), "stems": analyzer_stems(e)} for e in examples]
    word_counts = Counter(t for title in train_titles[:50_000] for t in remove_stopwords(tokenize(title)))
    top_word_stems = [{"word": w, "stem": stem(w), "count_in_50k_train": c} for w, c in word_counts.most_common(25)]

    out = {
        "n_train": len(train_titles), "n_test": len(test_titles),
        "normalization": {"steps": ["NFC", "lower-case", "Latin->Cyrillic homoglyph folding inside Cyrillic words",
                                    "digits -> 'num'"],
                          "titles_changed_by_homoglyph_folding": homoglyph_titles,
                          "titles_changed_pct": round(100 * homoglyph_titles / len(ds), 2)},
        "stopwords": {"n": len(KAZAKH_STOPWORDS), "general": sorted(GENERAL_STOPWORDS),
                      "legal_boilerplate": sorted(LEGAL_BOILERPLATE),
                      "justification": ("Function words and boilerplate that occur in a large share of titles of "
                                        "every class; kl_vs_prior close to 0 means the word's class distribution "
                                        "equals the overall prior, i.e. it carries little class information. "
                                        "Words with high KL (e.g. 'республикасының', typical of national acts) are "
                                        "informative and their removal is tested downstream (stems vs words)."),
                      "report_form_label": stopword_report(train)},
        "stemmer": {"min_stem_len": MIN_STEM, "n_suffixes": len(SUFFIXES), "max_rounds": 4,
                    "lexicon_size": len(lexicon), "lexicon_min_count": LEXICON_MIN_COUNT,
                    "exceptions": sorted(STEM_EXCEPTIONS),
                    "examples": stem_examples, "top_words": top_word_stems},
        "representations": comparison,
        "downstream_macro_f1": downstream_f1(),
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }
    try:
        COMPARISON_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as exc:
        print(f"[text_prep] could not write {COMPARISON_JSON}: {exc}")
    summary = {k: {"vocab": v["vocab_size"], "oov_token_pct": v["oov_token_rate_pct"]} for k, v in comparison.items()}
    print(f"[text_prep] {summary}")
    return summary


if __name__ == "__main__":
    run()
