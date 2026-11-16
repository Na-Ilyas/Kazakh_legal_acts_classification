"""Normalised SQLite corpus database, ER diagram and analytical SQL queries.

``run()``:
  1. executes sql/schema.sql (drops + recreates only the corpus tables; the
     experiments / predictions tables of experiments_db are left untouched);
  2. streams data/processed/corpus.parquet in row batches and loads acts,
     lookup tables and the two many-to-many tables with executemany
     (one short transaction per chunk so concurrent experiment logging
     by other processes is not blocked);
  3. creates indexes (sql/indexes.sql);
  4. draws the ER diagram (matplotlib) + writes a Mermaid version;
  5. runs every query of sql/queries.sql into pandas -> results/sql/<name>.csv.
"""
from __future__ import annotations

import re
import sqlite3
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pyarrow.parquet as pq
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src import config, experiments_db, labels

SQL_DIR = config.PROJECT_ROOT / "sql"
SCHEMA_SQL = SQL_DIR / "schema.sql"
INDEXES_SQL = SQL_DIR / "indexes.sql"
QUERIES_SQL = SQL_DIR / "queries.sql"
SQL_RESULTS_DIR = config.RESULTS_DIR / "sql"
ER_PNG = config.FIGURES_DIR / "er_diagram.png"
ER_MERMAID = config.PROJECT_ROOT / "docs" / "er_diagram.md"

_DIGITS_RE = re.compile(r"^\d+$")

# Rule-based body types (first matching keyword wins; checked on lower-cased name).
BODY_TYPE_RULES: list[tuple[str, str]] = [
    ("мәслихат", "Мәслихат"),
    ("әкімдіг", "Әкімдік"),
    ("әкімі", "Әкім"),
    ("еуразиялық", "Халықаралық / ЕАЭО"),
    ("халықаралық", "Халықаралық / ЕАЭО"),
    ("үкіметі", "Үкімет"),
    ("премьер", "Үкімет"),
    ("президент", "Президент"),
    ("парламент", "Парламент"),
    ("сот", "Сот / прокуратура"),
    ("прокуратура", "Сот / прокуратура"),
    ("ұлттық банк", "Ұлттық Банк"),
    ("министр", "Министрлік"),
    ("агенттіг", "Агенттік / комитет"),
    ("комитет", "Агенттік / комитет"),
    ("комиссия", "Агенттік / комитет"),
    ("бюро", "Агенттік / комитет"),
    ("палата", "Агенттік / комитет"),
]
_PAREN_RE = re.compile(r"\([^()]*\)")


# --------------------------------------------------------------------------
# Multi-valued field parsing
# --------------------------------------------------------------------------
def split_top_level(raw: str, sep: str = ",") -> list[str]:
    """Split on `sep` only outside parentheses ('A (x, y), B' -> ['A (x, y)', ' B'])."""
    parts, buf, depth = [], [], 0
    for ch in raw:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        if ch == sep and depth == 0:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    parts.append("".join(buf))
    return parts


def split_areas(raw) -> list[str]:
    """Vocabulary-aware split of `legal_relation_area`.

    Area names may contain commas ("Халықаралық шарттар, келісімдер және
    конвенциялар"). Rules: split outside parentheses; a fragment starting with
    a lower-case letter continues the previous name; purely numeric fragments
    (classifier codes like '005022003000') are dropped; placeholders dropped.
    """
    if not isinstance(raw, str):
        return []
    names: list[str] = []
    for part in split_top_level(raw):
        frag = " ".join(part.split())
        if not frag or frag in config.PLACEHOLDERS or _DIGITS_RE.match(frag):
            continue
        if names and frag[0].islower():
            names[-1] = f"{names[-1]}, {frag}"
        else:
            names.append(frag)
    return list(dict.fromkeys(names))          # de-duplicate, keep order


def split_form_tags(raw) -> list[str]:
    """`act_form` -> unique tags (unordered set semantics, whitespace-normalised)."""
    return list(dict.fromkeys(" ".join(t.split()) for t in labels.split_act_form(raw)))


def body_type(name: str) -> str:
    """Coarse type of an adopting body from keywords; joint acts -> 'Бірлескен'."""
    if len([p for p in split_top_level(name) if p.strip()]) > 1:
        return "Бірлескен (joint)"
    # classify on the current name only: parenthesised segments hold history
    # ("ҚР Президентінің ... Жарлығымен құрылды; бұрынғы атауы: ...") or the oblast
    # ("Есіл ауданының (Солтүстік Қазақстан облысы) мәслихаты") and must not decide the type
    low = name
    while _PAREN_RE.search(low):
        low = _PAREN_RE.sub(" ", low)
    low = low.lower().replace("i", "і")          # Latin 'i' homoglyph -> Kazakh 'і'

    for key, label in BODY_TYPE_RULES:
        if key in low:
            return label
    return "Басқа"


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def _connect(db_path: Path = config.DB_PATH) -> sqlite3.Connection:
    """Connection with FK checks, WAL (concurrent readers) and a generous busy timeout."""
    con = experiments_db.connect(db_path)        # also guarantees experiments/predictions exist
    con.execute("PRAGMA busy_timeout = 60000")
    con.execute("PRAGMA journal_mode = WAL")
    return con


def _read_sql_file(path: Path) -> str:
    """Read an .sql file (clear error if missing)."""
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise FileNotFoundError(f"cannot read {path}: {exc}") from exc


class _Lookup:
    """In-memory name -> id dictionary for one lookup table; new names are
    inserted in bulk per chunk (ids are assigned here, not by SQLite)."""

    def __init__(self, table: str, id_col: str, extra=None):
        self.table, self.id_col, self.extra = table, id_col, extra
        self.ids: dict[str, int] = {}
        self.pending: list[tuple] = []

    def get(self, name) -> int | None:
        if not isinstance(name, str) or not name:
            return None
        idx = self.ids.get(name)
        if idx is None:
            idx = len(self.ids) + 1
            self.ids[name] = idx
            extra = () if self.extra is None else (self.extra(name),)
            self.pending.append((idx, name, *extra))
        return idx

    def flush(self, con: sqlite3.Connection, columns: str) -> None:
        if self.pending:
            marks = ", ".join("?" * len(self.pending[0]))
            con.executemany(f"INSERT INTO {self.table} ({columns}) VALUES ({marks})", self.pending)
            self.pending = []


def _iso(ts) -> str | None:
    """Timestamp -> 'YYYY-MM-DD' (None for NaT)."""
    return None if pd.isna(ts) else ts.strftime("%Y-%m-%d")


def load_corpus(con: sqlite3.Connection, batch_size: int = config.CHUNKSIZE) -> dict:
    """Stream the corpus parquet into the normalised tables."""
    try:
        ds = pd.read_parquet(config.DATASET_PARQUET, columns=["law_id", "split", "area_label"])
    except OSError as exc:
        raise FileNotFoundError("dataset.parquet missing - run src.data_io first") from exc
    split_of = dict(zip(ds["law_id"], ds["split"].astype(object)))
    area_of = dict(zip(ds["law_id"], ds["area_label"].astype(object)))
    area_classes = {c for c in ds["area_label"].dropna().unique()}
    del ds

    bodies = _Lookup("bodies", "body_id", extra=body_type)
    regions = _Lookup("regions", "region_id")
    sections = _Lookup("sections", "section_id", extra=lambda n: int(n in area_classes))
    forces = _Lookup("legal_forces", "legal_force_id")

    def _base_form(tag: str):
        base = config.FORM_ALIASES.get(tag, tag)
        return base if base in config.FORM_PRIORITY else None

    tags = _Lookup("act_form_tags", "tag_id", extra=_base_form)
    areas = _Lookup("areas", "area_id")

    cols = ["law_id", "source_id", "title", "date_adopted", "date_modified", "date_published",
            "is_repealed", "act_form", "legal_relation_area", "legal_force", "adopting_body",
            "region", "database_section"]
    counts = {"acts": 0, "act_act_form": 0, "act_area": 0}
    pf = pq.ParquetFile(config.FULL_PARQUET)
    for batch in pf.iter_batches(batch_size=batch_size, columns=cols):
        df = batch.to_pandas()
        for c in ("act_form", "legal_relation_area", "legal_force", "adopting_body", "region", "database_section"):
            df[c] = df[c].astype(object)
        act_rows, form_rows, area_rows = [], [], []
        for r in df.itertuples(index=False):
            law_id = str(r.law_id)
            title = str(r.title)
            section = labels.clean_section(r.database_section)
            area_lab = area_of.get(law_id)
            act_rows.append((
                law_id, int(r.source_id), title, len(title), len(title.split()),
                _iso(r.date_adopted), _iso(r.date_modified), _iso(r.date_published),
                None if pd.isna(r.date_adopted) else int(r.date_adopted.year),
                int(bool(r.is_repealed)), labels.primary_form(r.act_form),
                None if area_lab is None or pd.isna(area_lab) else str(area_lab),
                split_of.get(law_id),
                bodies.get(r.adopting_body), regions.get(r.region), sections.get(section),
                forces.get(r.legal_force),
            ))
            form_rows.extend((law_id, tags.get(t)) for t in split_form_tags(r.act_form))
            area_rows.extend((law_id, areas.get(a)) for a in split_areas(r.legal_relation_area))
        with con:                                   # one short transaction per chunk
            bodies.flush(con, "body_id, name, body_type")
            regions.flush(con, "region_id, name")
            sections.flush(con, "section_id, name, is_area_class")
            forces.flush(con, "legal_force_id, name")
            tags.flush(con, "tag_id, name, base_form")
            areas.flush(con, "area_id, name")
            con.executemany(f"INSERT INTO acts VALUES ({', '.join('?' * 17)})", act_rows)
            con.executemany("INSERT INTO act_act_form VALUES (?, ?)", form_rows)
            con.executemany("INSERT INTO act_area VALUES (?, ?)", area_rows)
        counts["acts"] += len(act_rows)
        counts["act_act_form"] += len(form_rows)
        counts["act_area"] += len(area_rows)
    counts.update({"bodies": len(bodies.ids), "regions": len(regions.ids), "sections": len(sections.ids),
                   "legal_forces": len(forces.ids), "act_form_tags": len(tags.ids), "areas": len(areas.ids)})
    return counts


def build_database(db_path: Path = config.DB_PATH) -> dict:
    """(Re)create the corpus tables, load them in chunks and index them."""
    con = _connect(db_path)
    try:
        con.executescript(_read_sql_file(SCHEMA_SQL))
        t0 = time.perf_counter()
        counts = load_corpus(con)
        con.executescript(_read_sql_file(INDEXES_SQL))
        con.execute("ANALYZE")
        counts["load_time_s"] = round(time.perf_counter() - t0, 1)
        fk_errors = con.execute("PRAGMA foreign_key_check").fetchall()
        counts["foreign_key_violations"] = len(fk_errors)
    finally:
        con.close()
    return counts


# --------------------------------------------------------------------------
# ER diagram
# --------------------------------------------------------------------------
ER_TABLES: dict[str, dict] = {
    "bodies":        {"xy": (0.02, 0.70), "cols": ["PK body_id", "name (UQ)", "body_type"]},
    "regions":       {"xy": (0.02, 0.45), "cols": ["PK region_id", "name (UQ)"]},
    "acts":          {"xy": (0.34, 0.30), "cols": ["PK law_id", "source_id", "title", "title_len_chars / _words",
                                                   "date_adopted / _modified / _published", "year", "is_repealed",
                                                   "form_label, area_label, split",
                                                   "FK body_id", "FK region_id", "FK section_id",
                                                   "FK legal_force_id"]},
    "sections":      {"xy": (0.72, 0.70), "cols": ["PK section_id", "name (UQ)", "is_area_class"]},
    "legal_forces":  {"xy": (0.72, 0.47), "cols": ["PK legal_force_id", "name (UQ)"]},
    "act_act_form":  {"xy": (0.20, 0.19), "cols": ["PK,FK law_id", "PK,FK tag_id"]},
    "act_form_tags": {"xy": (0.02, 0.02), "cols": ["PK tag_id", "name (UQ)", "base_form"]},
    "act_area":      {"xy": (0.47, 0.08), "cols": ["PK,FK law_id", "PK,FK area_id"]},
    "areas":         {"xy": (0.72, 0.24), "cols": ["PK area_id", "name (UQ)"]},
    "experiments":   {"xy": (0.72, 0.83), "cols": ["PK exp_id", "model_name, family", "task, eval_split",
                                                   "f1_macro, n_params, ..."]},
    "predictions":   {"xy": (0.36, 0.82), "cols": ["PK,FK exp_id", "PK law_id (-> acts)", "y_true, y_pred",
                                                   "confidence"]},
}
# (child, parent, label) - arrow from the FK side ("many") to the referenced table ("one")
ER_EDGES = [
    ("acts", "bodies", "N:1"), ("acts", "regions", "N:1"), ("acts", "sections", "N:1"),
    ("acts", "legal_forces", "N:1"), ("act_act_form", "acts", "N:1"), ("act_act_form", "act_form_tags", "N:1"),
    ("act_area", "acts", "N:1"), ("act_area", "areas", "N:1"), ("predictions", "experiments", "N:1"),
    ("predictions", "acts", "N:1 (logical)"),
]


def draw_er_diagram(path: Path = ER_PNG) -> None:
    """Draw the ER diagram with matplotlib boxes and arrows (graphviz not required)."""
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.01, 1.01)          # headroom: rounded box edges must not touch the title
    ax.axis("off")
    line_h, width = 0.026, 0.25
    boxes = {}
    for name, spec in ER_TABLES.items():
        x, y = spec["xy"]
        h = line_h * (len(spec["cols"]) + 1.4)
        color = "#FCE9D9" if name in ("experiments", "predictions") else (
            "#DDEBF7" if name == "acts" else "#EAF4E4" if name in ("act_act_form", "act_area") else "#F2F2F2")
        ax.add_patch(FancyBboxPatch((x, y), width, h, boxstyle="round,pad=0.004", fc=color, ec="#333", lw=1.2))
        ax.text(x + width / 2, y + h - line_h * 0.75, name, ha="center", va="center", weight="bold", fontsize=11)
        ax.plot([x, x + width], [y + h - line_h * 1.3] * 2, color="#333", lw=0.8)
        for i, col in enumerate(spec["cols"]):
            ax.text(x + 0.008, y + h - line_h * (2.0 + i), col, va="center", fontsize=8.5,
                    family="monospace", weight="bold" if col.startswith("PK") else "normal")
        boxes[name] = (x, y, width, h)

    def _anchor(src, dst):
        sx, sy, sw, sh = boxes[src]
        dx, dy, dw, dh = boxes[dst]
        scx, scy, dcx, dcy = sx + sw / 2, sy + sh / 2, dx + dw / 2, dy + dh / 2
        if abs(scx - dcx) > abs(scy - dcy) * 1.2:      # mostly horizontal
            return ((sx + sw if dcx > scx else sx), scy), ((dx if dcx > scx else dx + dw), dcy)
        return (scx, (sy + sh if dcy > scy else sy)), (dcx, (dy if dcy > scy else dy + dh))

    for child, parent, lab in ER_EDGES:
        (x1, y1), (x2, y2) = _anchor(child, parent)
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                                     color="#555", lw=1.1, connectionstyle="arc3,rad=0.05",
                                     ls="--" if "logical" in lab else "-"))
        ax.text((x1 + x2) / 2, (y1 + y2) / 2, lab, fontsize=8, color="#A23", ha="center",
                bbox=dict(fc="white", ec="none", pad=0.5))
    ax.set_title("ER diagram - normalised KazakhLawCorpus + experiment tracking (SQLite data/kazlaw.db)\n"
                 "arrows point from the foreign-key (many) side to the referenced (one) table; "
                 "green = M2M link tables, orange = experiment tables", fontsize=12)
    fig.tight_layout()
    try:
        fig.savefig(path, dpi=150)
    except OSError as exc:
        print(f"[database] could not save ER diagram: {exc}")
    plt.close(fig)


MERMAID = """# ER diagram (Mermaid)

Rendered version: `reports/figures/er_diagram.png` (generated by `python -m src.database`).
Schema source: `sql/schema.sql`; experiment tables: `src/experiments_db.py`.

```mermaid
erDiagram
    bodies ||--o{ acts : "adopts"
    regions ||--o{ acts : "located in"
    sections ||--o{ acts : "classified in"
    legal_forces ||--o{ acts : "has force"
    acts ||--o{ act_act_form : "tagged"
    act_form_tags ||--o{ act_act_form : "tag"
    acts ||--o{ act_area : "regulates"
    areas ||--o{ act_area : "area"
    experiments ||--o{ predictions : "produces"
    acts ||--o{ predictions : "predicted (logical FK)"

    bodies { INTEGER body_id PK
             TEXT name UK
             TEXT body_type }
    regions { INTEGER region_id PK
              TEXT name UK }
    sections { INTEGER section_id PK
               TEXT name UK
               INTEGER is_area_class }
    legal_forces { INTEGER legal_force_id PK
                   TEXT name UK }
    act_form_tags { INTEGER tag_id PK
                    TEXT name UK
                    TEXT base_form }
    areas { INTEGER area_id PK
            TEXT name UK }
    acts { TEXT law_id PK
           INTEGER source_id
           TEXT title
           INTEGER title_len_chars
           INTEGER title_len_words
           TEXT date_adopted
           TEXT date_modified
           TEXT date_published
           INTEGER year
           INTEGER is_repealed
           TEXT form_label
           TEXT area_label
           TEXT split
           INTEGER body_id FK
           INTEGER region_id FK
           INTEGER section_id FK
           INTEGER legal_force_id FK }
    act_act_form { TEXT law_id PK, FK
                   INTEGER tag_id PK, FK }
    act_area { TEXT law_id PK, FK
               INTEGER area_id PK, FK }
    experiments { INTEGER exp_id PK
                  TEXT model_name
                  TEXT model_family
                  TEXT task
                  TEXT eval_split
                  REAL f1_macro
                  INTEGER n_params
                  REAL size_mb
                  REAL latency_ms_per_1000 }
    predictions { INTEGER exp_id PK, FK
                  TEXT law_id PK
                  TEXT y_true
                  TEXT y_pred
                  REAL confidence }
```

Normalisation notes:
* `act_form` and `legal_relation_area` are multi-valued -> many-to-many link tables
  (`act_act_form`, `act_area`), so the 863 raw `act_form` strings collapse to a small tag dictionary
  and the ~25k area combinations collapse to a few hundred area names.
* Repeated text attributes (body, region, section, legal force) are lookup tables referenced by integer FKs (3NF).
* `form_label`, `area_label`, `split` are derived ML labels kept on `acts` for analysis convenience.
"""


def write_mermaid(path: Path = ER_MERMAID) -> None:
    """Save the Mermaid ER diagram next to the docs."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(MERMAID, encoding="utf-8")
    except OSError as exc:
        print(f"[database] could not write {path}: {exc}")


# --------------------------------------------------------------------------
# Queries
# --------------------------------------------------------------------------
def parse_queries(path: Path = QUERIES_SQL) -> dict[str, dict]:
    """Parse sql/queries.sql into {name: {"sql": ..., "answers": ..., "uses": ...}}."""
    text = _read_sql_file(path)
    queries: dict[str, dict] = {}
    current = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("-- name:"):
            current = stripped.split(":", 1)[1].strip()
            queries[current] = {"sql": [], "answers": "", "uses": ""}
        elif current is None:
            continue
        elif stripped.startswith("-- answers:"):
            queries[current]["answers"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("-- uses:"):
            queries[current]["uses"] = stripped.split(":", 1)[1].strip()
        elif not stripped.startswith("--"):
            queries[current]["sql"].append(line)
    for q in queries.values():
        q["sql"] = "\n".join(q["sql"]).strip().rstrip(";")
    return queries


def run_queries(names: list[str] | None = None, save: bool = True) -> dict[str, pd.DataFrame]:
    """Execute (selected) queries into DataFrames and save results/sql/<name>.csv."""
    SQL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, pd.DataFrame] = {}
    con = _connect()
    try:
        for name, q in parse_queries().items():
            if names and name not in names:
                continue
            try:
                df = pd.read_sql_query(q["sql"], con)
            except (sqlite3.Error, pd.errors.DatabaseError) as exc:
                print(f"[database] query {name} failed: {exc}")
                df = pd.DataFrame()
            results[name] = df
            if save:
                try:
                    df.to_csv(SQL_RESULTS_DIR / f"{name}.csv", index=False)
                except OSError as exc:
                    print(f"[database] could not save {name}.csv: {exc}")
    finally:
        con.close()
    return results


def db_has_corpus() -> bool:
    """True if the acts table exists and is populated."""
    try:
        con = sqlite3.connect(config.DB_PATH)
        n = con.execute("SELECT COUNT(*) FROM acts").fetchone()[0]
        con.close()
        return n > 0
    except sqlite3.Error:
        return False


def run(rebuild: bool = True, queries_only: bool = False) -> dict:
    """Build the DB (unless present and rebuild=False), draw ER, run all queries.

    queries_only=True only re-runs sql/queries.sql against the current DB and
    refreshes results/sql/*.csv (use as the very last pipeline step, after all
    models have logged): `python -m src.database --queries-only`.
    """
    t0 = time.perf_counter()
    if queries_only:
        results = run_queries()
        rows = {k: len(v) for k, v in results.items()}
        print(f"[database] queries -> rows: {rows}")
        return {"query_rows": rows, "elapsed_s": round(time.perf_counter() - t0, 1)}
    counts = build_database() if (rebuild or not db_has_corpus()) else {"rebuilt": False}
    print(f"[database] loaded: {counts}")
    draw_er_diagram()
    write_mermaid()
    results = run_queries()
    rows = {k: len(v) for k, v in results.items()}
    print(f"[database] queries -> rows: {rows}")
    return {"tables": counts, "query_rows": rows, "elapsed_s": round(time.perf_counter() - t0, 1)}


if __name__ == "__main__":
    import sys
    run(queries_only="--queries-only" in sys.argv[1:])
