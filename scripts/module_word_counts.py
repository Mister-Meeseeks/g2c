#!/usr/bin/env python3
"""Word counts for each module's lesson page, split into lecture vs. post-lecture.

Lecture = everything before the first homework-style heading
(`## What you'll build`, `## Scaffolding...`, or `## Setting up the experiments`,
whichever appears first). Post = that heading and everything after.

Word counting matches `wc -w` (whitespace-separated tokens, including code
fences, ASCII diagrams, and frontmatter).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULES_DIR = REPO_ROOT / "docs" / "modules"

SPLIT_PATTERN = re.compile(
    r"^## (What you'll build|Scaffolding|Setting up the experiments)",
)


def split_lecture_post(text: str) -> tuple[str, str, str | None]:
    """Return (lecture, post, matched_heading) given lesson markdown."""
    lines = text.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if SPLIT_PATTERN.match(line):
            return "".join(lines[:i]), "".join(lines[i:]), line.strip()
    return text, "", None


def wc_words(text: str) -> int:
    return len(text.split())


def render_table(rows: list[dict], fmt: str) -> str:
    if fmt == "csv":
        out = ["module,lecture,post,total,split_heading"]
        for r in rows:
            heading = r["heading"] or ""
            out.append(f"{r['module']},{r['lecture']},{r['post']},{r['total']},{heading}")
        return "\n".join(out)

    headers = ["Module", "Lecture", "Post", "Total", "Split heading"]
    table = [headers]
    for r in rows:
        table.append([
            r["module"],
            f"{r['lecture']:,}",
            f"{r['post']:,}" if r["heading"] else "—",
            f"{r['total']:,}",
            r["heading"] or "(no split heading)",
        ])
    widths = [max(len(row[c]) for row in table) for c in range(len(headers))]

    def fmt_row(row: list[str]) -> str:
        cells = []
        for c, cell in enumerate(row):
            if c == 0 or c == 4:
                cells.append(cell.ljust(widths[c]))
            else:
                cells.append(cell.rjust(widths[c]))
        return "  ".join(cells).rstrip()

    lines = [fmt_row(table[0]), "  ".join("-" * w for w in widths)]
    for row in table[1:]:
        lines.append(fmt_row(row))
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--format",
        choices=("table", "csv"),
        default="table",
        help="Output format (default: table).",
    )
    parser.add_argument(
        "--modules-dir",
        type=Path,
        default=MODULES_DIR,
        help="Directory containing module .md files.",
    )
    args = parser.parse_args()

    paths = sorted(args.modules_dir.glob("*.md"))
    if not paths:
        print(f"No module .md files in {args.modules_dir}", file=sys.stderr)
        return 1

    rows = []
    totals = {"lecture": 0, "post": 0, "total": 0}
    for path in paths:
        text = path.read_text()
        lecture, post, heading = split_lecture_post(text)
        lec_n, post_n = wc_words(lecture), wc_words(post)
        total_n = wc_words(text)
        rows.append({
            "module": path.stem,
            "lecture": lec_n,
            "post": post_n,
            "total": total_n,
            "heading": heading,
        })
        totals["lecture"] += lec_n
        totals["post"] += post_n
        totals["total"] += total_n

    print(render_table(rows, args.format))
    if args.format == "table":
        print()
        print(
            f"Totals: lecture={totals['lecture']:,}  "
            f"post={totals['post']:,}  total={totals['total']:,}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
