"""Convert reports/report.md into reports/report.docx (python-docx, no pandoc needed).

Supports the Markdown subset used in the report: #/##/### headings, paragraphs
with **bold** and `code`, "- " bullets, "1. " numbered items, pipe tables and
images on their own line: ![caption](relative/path.png).

Usage:  .venv/bin/python reports/build_report.py
"""
from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

REPORT_DIR = Path(__file__).resolve().parent
SRC = REPORT_DIR / "report.md"
OUT = REPORT_DIR / "report.docx"
INLINE_RE = re.compile(r"(\*\*[^*]+\*\*|`[^`]+`)")
IMAGE_RE = re.compile(r"^!\[(.*)\]\((.+)\)\s*$")


def add_inline(paragraph, text: str) -> None:
    """Add text to a paragraph, rendering **bold** and `code` spans."""
    for part in INLINE_RE.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            paragraph.add_run(part[2:-2]).bold = True
        elif part.startswith("`") and part.endswith("`"):
            run = paragraph.add_run(part[1:-1])
            run.font.name = "Consolas"
            run.font.size = Pt(9.5)
        else:
            paragraph.add_run(part)


def parse_row(line: str) -> list[str]:
    """Split a markdown pipe-table row into stripped cells."""
    return [c.strip() for c in line.strip().strip("|").split("|")]


def add_table(doc: Document, rows: list[list[str]]) -> None:
    """Render a markdown table (header + body; separator row already removed)."""
    table = doc.add_table(rows=len(rows), cols=len(rows[0]))
    table.style = "Light Grid Accent 1"
    for i, row in enumerate(rows):
        for j, cell_text in enumerate(row[: len(rows[0])]):
            cell = table.cell(i, j)
            cell.text = ""
            par = cell.paragraphs[0]
            add_inline(par, cell_text)
            for run in par.runs:
                run.font.size = Pt(8.5)
                if i == 0:
                    run.bold = True
    doc.add_paragraph()


def build() -> Path:
    """Parse report.md line by line and write report.docx."""
    doc = Document()
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(11)
    lines = SRC.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if line.startswith("|"):
            block = []
            while i < len(lines) and lines[i].startswith("|"):
                if not re.match(r"^\|\s*:?-{2,}", lines[i]):
                    block.append(parse_row(lines[i]))
                i += 1
            add_table(doc, block)
            continue
        if (m := IMAGE_RE.match(line)):
            path = (REPORT_DIR / m.group(2)).resolve()
            try:
                doc.add_picture(str(path), width=Inches(6.0))
                doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
                cap = doc.add_paragraph()
                cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
                run = cap.add_run(m.group(1))
                run.italic = True
                run.font.size = Pt(9)
            except FileNotFoundError:
                doc.add_paragraph(f"[missing figure: {m.group(2)}]")
        elif line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            text = line.lstrip("#").strip()
            if level == 1:
                doc.add_heading(text, level=0)
            else:
                doc.add_heading(text, level=min(level - 1, 3))
        elif line.lstrip().startswith("- "):
            nested = line.startswith(" ")
            add_inline(doc.add_paragraph(style="List Bullet 2" if nested else "List Bullet"), line.lstrip()[2:])
        elif line.startswith("  "):                     # continuation paragraph of a list item
            par = doc.add_paragraph()
            par.paragraph_format.left_indent = Inches(0.25)
            add_inline(par, line.strip())
        elif re.match(r"^\d+\. ", line):
            add_inline(doc.add_paragraph(style="List Number"), re.sub(r"^\d+\. ", "", line))
        else:
            add_inline(doc.add_paragraph(), line)
        i += 1
    doc.save(OUT)
    return OUT


if __name__ == "__main__":
    print(f"Wrote {build()}")
