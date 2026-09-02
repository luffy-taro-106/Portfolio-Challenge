"""Render REPORT.md to a print-ready PDF.

Markdown is converted to HTML with a print stylesheet, then Chrome's headless
renderer produces the PDF. Chrome is used because it lays out tables and page
breaks properly, which the lightweight Python PDF writers do not.
"""

from __future__ import annotations

import base64
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "REPORT.md"
OUTPUT = ROOT / "REPORT.pdf"

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

STYLE = """
@page { size: A4; margin: 15mm 14mm; }

body {
  font-family: "Charter", "Georgia", serif;
  font-size: 9.3pt;
  line-height: 1.32;
  color: #16191d;
  margin: 0;
}

h1 {
  font-size: 17pt;
  margin: 0 0 2px 0;
  letter-spacing: -0.2px;
}
h1 + h3 {
  font-size: 10pt;
  font-weight: 400;
  color: #5a6472;
  margin: 0 0 14px 0;
  padding-bottom: 9px;
  border-bottom: 2px solid #1a3f6b;
}
h2 {
  font-size: 11.4pt;
  color: #1a3f6b;
  margin: 11px 0 5px 0;
  padding-bottom: 3px;
  border-bottom: 1px solid #d3d9e0;
  page-break-after: avoid;
}
h3 {
  font-size: 9.8pt;
  margin: 10px 0 4px 0;
  color: #24303f;
  page-break-after: avoid;
}

p { margin: 0 0 6px 0; text-align: justify; }
ul, ol { margin: 0 0 6px 0; padding-left: 17px; }
li { margin-bottom: 2px; }

table {
  border-collapse: collapse;
  width: 100%;
  margin: 7px 0 9px 0;
  font-size: 8.5pt;
  page-break-inside: avoid;
}
th {
  background: #1a3f6b;
  color: #ffffff;
  text-align: left;
  padding: 5px 7px;
  font-weight: 600;
}
td { padding: 4px 7px; border-bottom: 1px solid #e2e6eb; }
tr:nth-child(even) td { background: #f6f8fa; }

/* Alignment is decided per column from the cell contents, not by position. */
td, th { text-align: left; }
td.num, th.num { text-align: right; }

code {
  font-family: "SF Mono", Menlo, monospace;
  font-size: 8.2pt;
  background: #eef1f5;
  padding: 1px 3px;
  border-radius: 2px;
}

img {
  width: 76%;
  display: block;
  margin: 7px auto;
  page-break-inside: avoid;
}

hr {
  border: none;
  border-top: 1px solid #dfe4ea;
  margin: 10px 0;
}

em { color: #3d4653; }

/* Keep a heading from being stranded at the foot of a page. */
h2, h3, table, img { break-inside: avoid; }
"""


CELL = re.compile(r"<(th|td)([^>]*)>(.*?)</\1>", re.S)
ROW = re.compile(r"<tr>(.*?)</tr>", re.S)
TABLE = re.compile(r"<table>.*?</table>", re.S)


def _looks_numeric(cell: str) -> bool:
    """True for values, false for prose. Used to align table columns."""
    text = re.sub(r"<[^>]+>", "", cell).strip()
    if not text or not re.search(r"\d", text):
        return False
    # A figure may carry a short unit or qualifier; a sentence carries many
    # more letters than that.
    return len(re.findall(r"[A-Za-z]", text)) <= 10


def align_columns(html: str) -> str:
    """Right-align any column whose body cells are mostly numeric."""

    def fix_table(match: re.Match) -> str:
        table = match.group(0)
        rows = ROW.findall(table)
        if not rows:
            return table

        body = [CELL.findall(r) for r in rows if "<td" in r]
        if not body:
            return table

        width = max(len(r) for r in body)
        numeric = []
        for column in range(width):
            values = [r[column][2] for r in body
                      if len(r) > column and re.sub(r"<[^>]+>", "", r[column][2]).strip()]
            numeric.append(
                bool(values) and sum(map(_looks_numeric, values)) >= len(values) * 0.6
            )

        def rewrite_row(row_match: re.Match) -> str:
            inner = row_match.group(1)
            index = [0]

            def rewrite_cell(cell_match: re.Match) -> str:
                tag, attrs, content = cell_match.groups()
                column = index[0]
                index[0] += 1
                if column < len(numeric) and numeric[column]:
                    attrs = f'{attrs} class="num"'
                return f"<{tag}{attrs}>{content}</{tag}>"

            return f"<tr>{CELL.sub(rewrite_cell, inner)}</tr>"

        return ROW.sub(rewrite_row, table)

    return TABLE.sub(fix_table, html)


def inline_images(html: str) -> str:
    """Embed figures as data URIs so the PDF is self-contained."""

    def replace(match: re.Match) -> str:
        path = ROOT / match.group(1)
        if not path.exists():
            print(f"  warning: missing figure {path}", file=sys.stderr)
            return match.group(0)
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f'src="data:image/png;base64,{encoded}"'

    return re.sub(r'src="([^"]+)"', replace, html)


def main() -> None:
    if not SOURCE.exists():
        raise SystemExit(f"{SOURCE} not found")
    if not Path(CHROME).exists():
        raise SystemExit(f"Chrome not found at {CHROME}")

    body = markdown.markdown(
        SOURCE.read_text(),
        extensions=["tables", "attr_list", "sane_lists"],
    )
    html = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<style>{STYLE}</style></head>"
        f"<body>{inline_images(align_columns(body))}</body></html>"
    )

    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "report.html"
        source.write_text(html)
        # Chrome refuses to write into a directory it does not own on some
        # setups, so render into the temp directory and move the result.
        rendered = Path(tmp) / "report.pdf"
        subprocess.run(
            [
                CHROME,
                "--headless",
                "--disable-gpu",
                "--no-pdf-header-footer",
                f"--print-to-pdf={rendered}",
                source.as_uri(),
            ],
            check=True,
            capture_output=True,
            timeout=120,
        )
        if not rendered.exists():
            raise SystemExit("Chrome did not produce a PDF")
        shutil.move(str(rendered), OUTPUT)

    print(f"Wrote {OUTPUT} ({OUTPUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
