"""Exploratory data analysis figures + data-backed findings.

Figures (reports/figures/, dpi 150, titled, labelled axes):
  eda_class_distribution_by_split.png  label shift between train/val/test (both tasks)
  eda_acts_per_year_by_form.png        stacked bars from SQL query `acts_per_year_by_form`
  eda_title_length_by_form.png         title length (chars, words) per form class
  eda_missingness.png                  missing values per field: empty vs placeholder (audit)
  eda_duplicates_overview.png          exact / masked / near duplicates (audit + leakage)
  eda_top_bodies.png                   top-15 adopting bodies from SQL query `top_adopting_bodies`
  eda_area_form_heatmap.png            area x form from SQL query `area_form_matrix`
Findings: results/eda_summary.json (1-2 sentences per figure with the numbers).
"""
from __future__ import annotations

import json
import re
import textwrap
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config, database
from src.data_io import load_dataset

SUMMARY_JSON = config.RESULTS_DIR / "eda_summary.json"
FIG = config.FIGURES_DIR
# Reference categorical palette (fixed order, never cycled)
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
FORM_ORDER = ["Шешім", "Қаулы", "Бұйрық", "Өкім", "Жарлық", "Заң", "Басқа"]
FORM_COLORS = dict(zip(FORM_ORDER, PALETTE))
SPLIT_COLORS = {"train": PALETTE[0], "val": PALETTE[1], "test": PALETTE[2]}
INK, MUTED = "#0b0b0b", "#52514e"

plt.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": 150, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.titlesize": 12, "axes.titleweight": "bold", "font.size": 9.5,
})


def _save(fig, name: str) -> str:
    """Save a figure into reports/figures and close it."""
    path = FIG / name
    try:
        fig.tight_layout()
        fig.savefig(path, dpi=150, facecolor="white", bbox_inches="tight")
    except OSError as exc:
        print(f"[eda] could not save {path}: {exc}")
    plt.close(fig)
    return str(path.relative_to(config.PROJECT_ROOT))


def _load_json(name: str) -> dict:
    """Read a results/*.json produced by another module ({} if missing)."""
    try:
        return json.loads((config.RESULTS_DIR / name).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        print(f"[eda] {name} missing - run the module that creates it first")
        return {}


def _form_order(present) -> list[str]:
    """Known form classes in display order, then any other present class (never NaN)."""
    present = [str(c) for c in present if isinstance(c, str)]
    return [f for f in FORM_ORDER if f in present] + sorted(c for c in present if c not in FORM_ORDER)


def _short(name: str, width: int = 48) -> str:
    """Shorten long Kazakh names for tick labels (drop parenthesised history)."""
    base = re.sub(r"\s*\([^()]*\)", "", name)
    return textwrap.shorten(base, width=width, placeholder="…")


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def fig_class_distribution(ds: pd.DataFrame) -> dict:
    """Share of each class within each split, for both tasks (label shift)."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 8), gridspec_kw={"width_ratios": [1, 1.6]})
    findings = {}
    for ax, (task, col) in zip(axes, config.TASKS.items()):
        sub = ds.dropna(subset=[col])
        share = pd.crosstab(sub[col].astype(str), sub["split"].astype(str), normalize="columns") * 100
        share = share[["train", "val", "test"]].sort_values("train")
        y = np.arange(len(share))
        h = 0.26
        for i, split in enumerate(["train", "val", "test"]):
            ax.barh(y + (i - 1) * h, share[split], height=h * 0.92, color=SPLIT_COLORS[split],
                    label=f"{split} (n={int((sub['split'] == split).sum()):,})")
        ax.set_yticks(y)
        ax.set_yticklabels([_short(c, 40) for c in share.index], fontsize=8 if task == "area" else 9.5)
        ax.set_xlabel("share of the split (%)")
        ax.set_ylabel(f"{task} class")
        ax.set_title(f"Task `{task}`: class share per temporal split")
        ax.legend(loc="lower right")
        delta = share["test"] - share["train"]
        up, down = delta.idxmax(), delta.idxmin()
        findings[task] = {
            "up": up, "up_pct": [round(float(share.loc[up, "train"]), 1), round(float(share.loc[up, "test"]), 1)],
            "down": down, "down_pct": [round(float(share.loc[down, "train"]), 1), round(float(share.loc[down, "test"]), 1)],
            "test_top": share["test"].idxmax(), "test_top_pct": round(float(share["test"].max()), 1)}
    path = _save(fig, "eda_class_distribution_by_split.png")
    f, a = findings["form"], findings["area"]
    text = (f"Label shift between the temporal splits: in `form`, '{f['up']}' goes from {f['up_pct'][0]}% of train "
            f"to {f['up_pct'][1]}% of test while '{f['down']}' goes from {f['down_pct'][0]}% to {f['down_pct'][1]}%; "
            f"the largest test class covers {f['test_top_pct']}% of test, so accuracy mostly reflects one class and "
            f"macro-F1 is the headline metric. In `area`, '{_short(a['up'], 30)}' goes from {a['up_pct'][0]}% "
            f"(train) to {a['up_pct'][1]}% (test).")
    return {"figure": path, "finding": text, "numbers": findings}


def fig_acts_per_year(q: pd.DataFrame) -> dict:
    """Stacked bars of acts per year and form (SQL result)."""
    full = q.pivot_table(index="year", columns="form_label", values="n_acts", fill_value=0)
    forms = _form_order(full.columns)
    full = full[forms]
    piv = full.loc[full.index >= 1990]
    fig, ax = plt.subplots(figsize=(13, 5.5))
    bottom = np.zeros(len(piv))
    for form in piv.columns:
        ax.bar(piv.index, piv[form], bottom=bottom, color=FORM_COLORS.get(form, MUTED), width=0.85, label=form,
               edgecolor="white", linewidth=0.6)
        bottom += piv[form].to_numpy()
    top_y = bottom.max() * 1.03
    for x in (config.TRAIN_END_YEAR + 0.5, config.VAL_YEAR + 0.5):
        ax.axvline(x, color=INK, ls="--", lw=1)
    ax.text(config.TRAIN_END_YEAR + 0.3, top_y, f"train ≤{config.TRAIN_END_YEAR}", ha="right", fontsize=8.5, color=INK)
    ax.text(config.VAL_YEAR, top_y * 1.04, "val", ha="center", fontsize=8.5, color=INK)
    ax.text(config.VAL_YEAR + 0.7, top_y, f"test ≥{config.VAL_YEAR + 1}", ha="left", fontsize=8.5, color=INK)
    ax.set_ylim(0, top_y * 1.12)
    last_year = int(full.index.max())
    ax.set_title(f"Acts adopted per year by primary legal form (SQL: acts_per_year_by_form), 1990-{last_year}"
                 + (" (last year partial)" if full.loc[last_year].sum() < 0.5 * full.sum(axis=1).median() else ""))
    ax.set_xlabel("year adopted")
    ax.set_ylabel("number of acts")
    ax.legend(title="form", ncol=len(forms), loc="upper left", fontsize=8.5)
    path = _save(fig, "eda_acts_per_year_by_form.png")
    totals = full.sum(axis=1)
    peak = int(totals.idxmax())
    shares = full.div(totals, axis=0) * 100
    # Compare the form that dominates the test years with its share 10 years before the split
    test_years = shares.index[shares.index > config.VAL_YEAR]
    dom = full.loc[test_years].sum().idxmax()
    ref = config.TRAIN_END_YEAR - 10
    direction = "rises" if shares.loc[peak, dom] > shares.loc[ref, dom] else "falls"
    text = (f"Volume peaks in {peak} ({int(totals.max()):,} acts). The share of '{dom}', the most frequent form in "
            f"the test years, {direction} from {shares.loc[ref, dom]:.0f}% in {ref} to {shares.loc[peak, dom]:.0f}% "
            f"in {peak}, which drives the form label shift of the temporal split. "
            f"{int(totals.loc[totals.index < 2000].sum()):,} acts predate 2000 "
            f"(earliest year: {int(totals.index.min())}).")
    return {"figure": path, "finding": text, "sql_query": "acts_per_year_by_form"}

def fig_title_length(ds: pd.DataFrame) -> dict:
    """Title length in characters and words per form (box plots, outliers hidden)."""
    d = ds.dropna(subset=["form_label"])[["title", "form_label"]].copy()
    d["form_label"] = d["form_label"].astype(str)
    d["chars"] = d["title"].str.len()
    d["words"] = d["title"].str.split().str.len()
    forms = _form_order(d["form_label"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, col, unit in ((axes[0], "chars", "characters"), (axes[1], "words", "whitespace words")):
        data = [d.loc[d["form_label"] == f, col].to_numpy() for f in forms]
        bp = ax.boxplot(data, tick_labels=forms, showfliers=False, patch_artist=True, widths=0.6,
                        medianprops={"color": INK})
        for patch, f in zip(bp["boxes"], forms):
            patch.set_facecolor(FORM_COLORS.get(f, MUTED))
            patch.set_alpha(0.75)
        ax.set_title(f"Length in {unit}")
        ax.set_xlabel("form class")
        ax.set_ylabel(f"title length ({unit})")
    fig.suptitle("Title length by form class (boxes = IQR, whiskers = 1.5 IQR, outliers hidden)",
                 fontweight="bold", fontsize=12)
    path = _save(fig, "eda_title_length_by_form.png")
    med = d.groupby("form_label")[["chars", "words"]].median()
    short, long_ = med["chars"].idxmin(), med["chars"].idxmax()
    text = (f"Median title length is {d['chars'].median():.0f} chars / {d['words'].median():.0f} words "
            f"(p95 {d['chars'].quantile(0.95):.0f}, p99 {d['chars'].quantile(0.99):.0f}, max {d['chars'].max():,} chars). "
            f"By median, {short} titles are the shortest ({med.loc[short, 'chars']:.0f} chars) and {long_} the "
            f"longest ({med.loc[long_, 'chars']:.0f} chars); {int((d['chars'] > 1000).sum())} titles "
            f"({100 * (d['chars'] > 1000).mean():.2f}%) exceed 1000 chars.")
    return {"figure": path, "finding": text, "median_chars_by_form": med["chars"].round(1).to_dict()}

def fig_missingness(audit: dict) -> dict:
    """Missing values per field split into true empties and placeholders."""
    miss = pd.DataFrame(audit.get("missing", {})).T
    if miss.empty:
        return {"figure": None, "finding": "audit.json missing"}
    n = audit["n_rows"]
    miss = miss[miss["missing_total"] > 0].sort_values("missing_total")
    fig, ax = plt.subplots(figsize=(11, 7))
    y = np.arange(len(miss))
    emp = miss["empty"].astype(float) / n * 100
    ph = miss["placeholder"].astype(float) / n * 100
    ax.barh(y, emp, color=PALETTE[0], label="true empty", height=0.7)
    ax.barh(y, ph, left=emp, color=PALETTE[1], label="placeholder ('ЖОҚ', 'НЕТ', 'undefined', ...)", height=0.7,
            edgecolor="white", linewidth=0.6)
    for yi, tot in zip(y, emp + ph):
        ax.text(tot + 0.8, yi, f"{tot:.1f}%", va="center", fontsize=8, color=MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels(miss.index)
    ax.set_xlim(0, 100)
    ax.set_xlabel(f"rows missing (% of {n:,})")
    ax.set_ylabel("field")
    ax.set_title("Missing values per field: true empties vs. source placeholders")
    ax.legend(loc="lower right")
    path = _save(fig, "eda_missingness.png")
    allm = audit["missing"]
    hidden = miss.sort_values("placeholder").iloc[-1]
    looks_filled = 100 * (1 - float(hidden["empty"]) / n)
    text = (f"Model inputs are nearly complete (missing: title {allm['title']['missing_pct']}%, date_adopted "
            f"{allm['date_adopted']['missing_pct']}%, act_form {allm['act_form']['missing_pct']}%, database_section "
            f"{allm['database_section']['missing_pct']}%), but placeholders hide missingness: '{hidden.name}' looks "
            f"{looks_filled:.2f}% filled when only empty strings are counted, yet {int(hidden['placeholder']):,} rows "
            f"({100 * float(hidden['placeholder']) / n:.1f}%) are placeholders that a naive isna() would miss.")
    return {"figure": path, "finding": text}


def exact_overlap_before_removal() -> dict:
    """% of val / test rows whose normalised title occurs in train BEFORE build_dataset removed them."""
    from src import labels
    from src.data_io import load_corpus
    c = load_corpus(["title", "date_adopted"])
    c["split"] = labels.assign_split(c["date_adopted"].dt.year)
    c["t"] = c["title"].map(labels.normalize_title)
    train = set(c.loc[c["split"] == "train", "t"])
    return {sp_: round(100 * float(c.loc[c["split"] == sp_, "t"].isin(train).mean()), 1) for sp_ in ("val", "test")}


def fig_duplicates(audit: dict, leak: dict) -> dict:
    """Exact duplicate keys (audit) and train->eval overlap levels (leakage)."""
    dup = audit.get("duplicates", {})
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    keys = ["law_id", "title_exact", "title_normalized", "title_plus_date_adopted", "title_plus_description"]
    vals = [dup.get(k, {}).get("rows_in_duplicate_groups", 0) for k in keys]
    axes[0].bar(range(len(keys)), vals, color=PALETTE[0], width=0.6)
    for i, v in enumerate(vals):
        axes[0].text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=8.5, color=INK)
    axes[0].set_xticks(range(len(keys)))
    axes[0].set_xticklabels(["law_id", "title", "normalised\ntitle", "title +\ndate", "title +\ndescription"])
    axes[0].set_ylabel("rows in duplicate groups")
    axes[0].set_xlabel("duplicate key")
    axes[0].set_title("Whole corpus: rows sharing a key with another row")

    nd = leak.get("near_duplicates", {})
    ex = leak.get("removed_by_build_dataset", {})
    masked = leak.get("digit_masked_overlap", {})
    exact = exact_overlap_before_removal()
    cats = ["exact title\n(removed, % before)", "same title up\nto digits", "near-dup\ncos ≥ 0.9",
            "near-dup\ncos ≥ 0.8"]
    test_vals = [exact.get("test", 0), masked.get("train_test_pct", 0), nd.get("test", {}).get("pct_ge_0.9", 0),
                 nd.get("test", {}).get("pct_ge_0.8", 0)]
    val_vals = [exact.get("val", 0), masked.get("train_val_pct", 0), nd.get("val", {}).get("pct_ge_0.9", 0),
                nd.get("val", {}).get("pct_ge_0.8", 0)]
    x = np.arange(len(cats))
    axes[1].bar(x - 0.2, val_vals, width=0.38, color=SPLIT_COLORS["val"], label=f"val ({config.VAL_YEAR})")
    axes[1].bar(x + 0.2, test_vals, width=0.38, color=SPLIT_COLORS["test"], label=f"test ({config.VAL_YEAR + 1}+)")
    for xi, v, w in zip(x, test_vals, val_vals):
        axes[1].text(xi + 0.2, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=8.5)
        axes[1].text(xi - 0.2, w, f"{w:.1f}%", ha="center", va="bottom", fontsize=8.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(cats)
    axes[1].set_ylabel("% of eval titles")
    axes[1].set_xlabel("overlap type with a TRAIN title")
    axes[1].set_title("Eval titles overlapping a TRAIN title\n(bars 2-4: after exact removal; cosine on digit-masked titles)")
    axes[1].legend()
    path = _save(fig, "eda_duplicates_overview.png")
    td = dup.get("title_exact", {})
    lid = dup.get("law_id", {}).get("redundant_rows", 0)
    strict = leak.get("test_strict", {})
    text = (f"{td.get('rows_in_duplicate_groups', 0):,} rows share their exact title with another row "
            f"({td.get('distinct_duplicated_values', 0):,} distinct titles); law_id has {lid:,} duplicated rows. "
            f"After removing {ex.get('eval_titles_seen_in_train') or 0:,} eval rows whose normalised title occurs in "
            f"train, {masked.get('train_test_pct', 0):.1f}% of test titles still match a train title up to digits and "
            f"{nd.get('test', {}).get('pct_ge_0.9')}% have a near-duplicate (masked cosine ≥ 0.9); test_strict keeps "
            f"{strict.get('n_strict', 0):,} of {strict.get('n_test', 0):,} test rows ({strict.get('pct_kept', 0)}%).")
    return {"figure": path, "finding": text}


def fig_top_bodies(q: pd.DataFrame, by_type: pd.DataFrame) -> dict:
    """Top-15 adopting bodies (SQL result), coloured by body type."""
    q = q.sort_values("n_acts")
    types = list(dict.fromkeys(q["body_type"][::-1]))
    colors = {t: PALETTE[i % len(PALETTE)] for i, t in enumerate(types)}
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.barh(range(len(q)), q["n_acts"], color=[colors[t] for t in q["body_type"]], height=0.7)
    for i, (n, rep) in enumerate(zip(q["n_acts"], q["pct_repealed"])):
        ax.text(n, i, f" {n:,}  ({rep:.0f}% repealed)", va="center", fontsize=8, color=MUTED)
    ax.set_yticks(range(len(q)))
    ax.set_yticklabels([_short(b, 60) for b in q["body"]], fontsize=8.5)
    ax.set_xlabel("number of acts adopted")
    ax.set_ylabel("adopting body")
    ax.set_title("Top-15 adopting bodies (SQL: top_adopting_bodies)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[t]) for t in types]
    ax.legend(handles, types, title="body type", loc="lower right")
    ax.set_xlim(0, q["n_acts"].max() * 1.3)
    path = _save(fig, "eda_top_bodies.png")
    top = q.iloc[-1]
    total = int(by_type["n_acts"].sum())
    top_share = 100 * q["n_acts"].sum() / total
    local = by_type[by_type["body_type"].isin(["Мәслихат", "Әкімдік", "Әкім"])]
    text = (f"The largest single issuer is '{_short(top['body'], 50)}' ({int(top['n_acts']):,} acts, "
            f"{top['pct_repealed']:.0f}% repealed), yet the top-{len(q)} bodies together adopt only {top_share:.1f}% of "
            f"all {total:,} acts. Local bodies (types Мәслихат / Әкімдік / Әкім; {int(local['n_bodies'].sum()):,} "
            f"distinct bodies) adopt {local['pct_of_acts'].sum():.1f}% of the acts (SQL: acts_by_body_type).")
    return {"figure": path, "finding": text, "sql_query": ["top_adopting_bodies", "acts_by_body_type"]}


def fig_area_form_heatmap(q: pd.DataFrame) -> dict:
    """Area x form heatmap (row-normalised %) from SQL."""
    piv = q.pivot_table(index="area_label", columns="form_label", values="n_acts", fill_value=0)
    piv = piv[_form_order(piv.columns)]
    piv = piv.loc[piv.sum(axis=1).sort_values(ascending=False).index]
    pct = piv.div(piv.sum(axis=1), axis=0) * 100
    fig, ax = plt.subplots(figsize=(11, 12))
    im = ax.imshow(pct.to_numpy(), cmap="Blues", aspect="auto", vmin=0, vmax=100)
    for i in range(pct.shape[0]):
        for j in range(pct.shape[1]):
            v = pct.iat[i, j]
            if v >= 1:
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if v > 60 else INK)
    ax.set_xticks(range(pct.shape[1]))
    ax.set_xticklabels(pct.columns)
    ax.set_yticks(range(pct.shape[0]))
    ax.set_yticklabels([f"{_short(a, 42)} (n={int(piv.loc[a].sum()):,})" for a in pct.index], fontsize=7.5)
    ax.set_xlabel("form class")
    ax.set_ylabel("area class (sorted by size)")
    ax.grid(False)
    ax.set_title("Area x form: % of each area's acts per form (SQL: area_form_matrix)")
    fig.colorbar(im, ax=ax, fraction=0.03, label="% of the area's acts")
    path = _save(fig, "eda_area_form_heatmap.png")
    dominant = pct.idxmax(axis=1)
    mono = pct.max(axis=1)
    most, least = mono.idxmax(), mono.idxmin()
    text = (f"The two targets are correlated but not redundant: {int((mono >= 70).sum())} of {len(pct)} areas have one "
            f"form covering ≥70% of their acts (most concentrated: '{_short(most, 30)}', {mono.max():.1f}% "
            f"{dominant.loc[most]}), while the most mixed area '{_short(least, 30)}' has at most {mono.min():.1f}% "
            "in any single form, so knowing the form does not determine the area.")
    return {"figure": path, "finding": text, "sql_query": "area_form_matrix"}


def run() -> dict:
    """Create all EDA figures and results/eda_summary.json."""
    t0 = time.perf_counter()
    ds = load_dataset(["title", "split", "form_label", "area_label", "year"])
    audit = _load_json("audit.json")
    leak = _load_json("leakage_report.json")
    if not database.db_has_corpus():
        print("[eda] corpus DB missing - building it")
        database.build_database()
    sql = database.run_queries(["acts_per_year_by_form", "top_adopting_bodies", "area_form_matrix",
                                "acts_by_body_type"])
    summary: dict = {}
    steps = [
        ("class_distribution", lambda: fig_class_distribution(ds)),
        ("acts_per_year_by_form", lambda: fig_acts_per_year(sql["acts_per_year_by_form"])),
        ("title_length", lambda: fig_title_length(ds)),
        ("missingness", lambda: fig_missingness(audit)),
        ("duplicates", lambda: fig_duplicates(audit, leak)),
        ("top_bodies", lambda: fig_top_bodies(sql["top_adopting_bodies"], sql["acts_by_body_type"])),
        ("area_form_heatmap", lambda: fig_area_form_heatmap(sql["area_form_matrix"])),
    ]
    for name, fn in steps:
        try:
            summary[name] = fn()
            print(f"[eda] {name}: {summary[name]['finding']}")
        except Exception as exc:  # one broken figure must not stop the others
            print(f"[eda] {name} failed: {exc!r}")
            summary[name] = {"figure": None, "finding": f"failed: {exc!r}"}
    summary["elapsed_s"] = round(time.perf_counter() - t0, 1)
    try:
        SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as exc:
        print(f"[eda] could not write summary: {exc}")
    return {"figures": [v["figure"] for v in summary.values() if isinstance(v, dict)], "elapsed_s": summary["elapsed_s"]}


if __name__ == "__main__":
    run()
