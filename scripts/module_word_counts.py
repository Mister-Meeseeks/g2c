#!/usr/bin/env python3
"""Word counts for each module's lesson page.

Default mode splits each module into lecture vs. post-lecture.

Lecture = everything before the first homework-style heading
(`## What you'll build`, `## Scaffolding...`, or `## Setting up the experiments`,
whichever appears first). Post = that heading and everything after.

Pass ``--breakdown`` for a deeper section-level count using the course's
canonical module headings.

Pass ``--diff`` or ``--pct`` to compare each row against its column average.

Word counting matches `wc -w` (whitespace-separated tokens, including code
fences, ASCII diagrams, and frontmatter).
"""
from __future__ import annotations

import argparse
import csv
import io
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULES_DIR = REPO_ROOT / "docs" / "modules"

SPLIT_PATTERN = re.compile(
    r"^## (What you'll build|Scaffolding|Setting up the experiments)",
)
HEADING_PATTERN = re.compile(r"^#{2,6}\s+")


@dataclass(frozen=True)
class SectionSpec:
    key: str
    label: str
    pattern: re.Pattern[str]


DEEP_SECTIONS = [
    SectionSpec("intro", "Intro", re.compile(r"$^")),
    SectionSpec(
        "before_start",
        "Before",
        re.compile(r"^## Before you start\s*$", re.IGNORECASE),
    ),
    SectionSpec(
        "where_fits",
        "Where fits",
        re.compile(r"^## Where this fits(?: in)?\s*$", re.IGNORECASE),
    ),
    SectionSpec(
        "big_idea",
        "Big idea",
        re.compile(r"^## The big idea\s*$", re.IGNORECASE),
    ),
    SectionSpec("body", "Body", re.compile(r"$^")),
    SectionSpec(
        "concepts",
        "Concepts",
        re.compile(r"^## Concepts to internalize\s*$", re.IGNORECASE),
    ),
    SectionSpec(
        "build",
        "Build",
        re.compile(r"^## What you'll build\s*$", re.IGNORECASE),
    ),
    SectionSpec(
        "exercises",
        "Exercises",
        re.compile(r"^## Exercises\s*$", re.IGNORECASE),
    ),
    SectionSpec(
        "pitfalls",
        "Pitfalls",
        re.compile(r"^## Pitfalls(?: to (?:expect|avoid)| avoid)?\s*$", re.IGNORECASE),
    ),
    SectionSpec(
        "m_series",
        "M-series",
        re.compile(r"^## M-series notes\s*$", re.IGNORECASE),
    ),
    SectionSpec(
        "reading",
        "Reading",
        re.compile(r"^## Readings?\s*$", re.IGNORECASE),
    ),
    SectionSpec(
        "deliverables",
        "Deliverables",
        re.compile(r"^## Deliverable(?: checklist|s)?\s*$", re.IGNORECASE),
    ),
]


def split_lecture_post(text: str) -> tuple[str, str, str | None]:
    """Return (lecture, post, matched_heading) given lesson markdown."""
    lines = text.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if SPLIT_PATTERN.match(line):
            return "".join(lines[:i]), "".join(lines[i:]), line.strip()
    return text, "", None


def wc_words(text: str) -> int:
    return len(text.split())


def column_averages(rows: list[dict], keys: list[str]) -> dict[str, float]:
    if not rows:
        return {key: 0.0 for key in keys}
    return {key: sum(float(row[key]) for row in rows) / len(rows) for key in keys}


def deep_breakdown(text: str) -> dict[str, int]:
    """Return word counts for canonical module section ranges.

    `intro` is everything before `## Before you start`, excluding that header.
    `big_idea` is only the `## The big idea` section through the next markdown
    heading of any level. `body` is everything after that first post-big-idea
    heading until `## Concepts to internalize`. Every other present section
    includes its own `##` header and runs until the next recognized canonical
    heading in document order. Missing sections count as zero.
    """
    lines = text.splitlines(keepends=True)
    starts = _section_starts(lines)
    ordered_starts = sorted(starts.items(), key=lambda item: item[1])
    first_known_start = ordered_starts[0][1] if ordered_starts else len(lines)
    before_start = starts.get("before_start")
    intro_end = before_start if before_start is not None else first_known_start

    counts = {spec.key: 0 for spec in DEEP_SECTIONS}
    counts["intro"] = wc_words("".join(lines[:intro_end]))

    for key, start in ordered_starts:
        if key == "intro":
            continue
        if key == "big_idea":
            counts["big_idea"], counts["body"] = _count_big_idea_and_body(
                lines,
                big_idea_start=start,
                concepts_start=starts.get("concepts"),
            )
            continue
        if key == "body":
            continue
        end = next(
            (other_start for _, other_start in ordered_starts if other_start > start),
            len(lines),
        )
        counts[key] = wc_words("".join(lines[start:end]))
    return counts


def _count_big_idea_and_body(
    lines: list[str],
    *,
    big_idea_start: int,
    concepts_start: int | None,
) -> tuple[int, int]:
    if concepts_start is None or concepts_start <= big_idea_start:
        body_end = len(lines)
        body_count = 0
    else:
        body_end = concepts_start
        body_count = None
    next_heading = next(
        (
            index
            for index in range(big_idea_start + 1, body_end)
            if HEADING_PATTERN.match(lines[index])
        ),
        body_end,
    )
    big_idea_count = wc_words("".join(lines[big_idea_start:next_heading]))
    if body_count is None:
        body_count = wc_words("".join(lines[next_heading:body_end]))
    return big_idea_count, body_count


def _section_starts(lines: list[str]) -> dict[str, int]:
    starts: dict[str, int] = {}
    for index, line in enumerate(lines):
        if not line.startswith("## "):
            continue
        for spec in DEEP_SECTIONS:
            if spec.key in {"intro", "body"} or spec.key in starts:
                continue
            if spec.pattern.match(line.strip()):
                starts[spec.key] = index
                break
    return starts


# Technical prose with math and code reads slower than general text; 180 wpm is a
# deliberately conservative planning figure. This covers *reading the lesson only* —
# it says nothing about how long the implementation or exercises take, which depends
# far more on the reader than on the page.
WORDS_PER_MINUTE = 180


def reading_minutes(words: int) -> int:
    return max(1, round(words / WORDS_PER_MINUTE))


def render_table(rows: list[dict], fmt: str, *, relative: str = "none") -> str:
    numeric_keys = ["lecture", "post", "total"]
    averages = column_averages(rows, numeric_keys)
    if fmt == "csv":
        buffer = io.StringIO()
        fieldnames = ["module", *numeric_keys, "split_heading"]
        writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "module": row["module"],
                **{
                    key: format_numeric_cell(
                        row[key],
                        averages[key],
                        relative=relative,
                        for_csv=True,
                    )
                    for key in numeric_keys
                },
                "split_heading": row["heading"] or "",
            })
        return buffer.getvalue().rstrip("\n")

    headers = ["Module", "Lecture", "Post", "Total", "Split heading"]
    table = [headers]
    for r in rows:
        table.append([
            r["module"],
            format_numeric_cell(r["lecture"], averages["lecture"], relative=relative),
            (
                format_numeric_cell(r["post"], averages["post"], relative=relative)
                if r["heading"] or relative != "none"
                else "—"
            ),
            format_numeric_cell(r["total"], averages["total"], relative=relative),
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


def render_breakdown_table(rows: list[dict], fmt: str, *, relative: str = "none") -> str:
    section_specs = DEEP_SECTIONS
    numeric_keys = [*(spec.key for spec in section_specs), "total"]
    averages = column_averages(rows, numeric_keys)
    fieldnames = ["module", *numeric_keys]
    if fmt == "csv":
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "module": row["module"],
                **{
                    key: format_numeric_cell(
                        row[key],
                        averages[key],
                        relative=relative,
                        for_csv=True,
                    )
                    for key in numeric_keys
                },
            })
        return buffer.getvalue().rstrip("\n")

    headers = ["Module", *(spec.label for spec in section_specs), "Total"]
    table = [headers]
    for row in rows:
        table.append([
            row["module"],
            *(
                format_numeric_cell(
                    row[spec.key],
                    averages[spec.key],
                    relative=relative,
                )
                for spec in section_specs
            ),
            format_numeric_cell(row["total"], averages["total"], relative=relative),
        ])
    widths = [max(len(line[column]) for line in table) for column in range(len(headers))]

    def fmt_row(row: list[str]) -> str:
        cells = []
        for column, cell in enumerate(row):
            if column == 0:
                cells.append(cell.ljust(widths[column]))
            else:
                cells.append(cell.rjust(widths[column]))
        return "  ".join(cells).rstrip()

    lines = [fmt_row(table[0]), "  ".join("-" * width for width in widths)]
    for row in table[1:]:
        lines.append(fmt_row(row))
    return "\n".join(lines)


def format_numeric_cell(
    value: int | float,
    average: float,
    *,
    relative: str,
    for_csv: bool = False,
) -> str:
    if relative == "none":
        return str(int(value)) if for_csv else f"{int(value):,}"

    delta = float(value) - average
    if relative == "diff":
        rounded = int(round(delta))
        return f"{rounded:+d}" if for_csv else f"{rounded:+,d}"

    if relative == "pct":
        if average == 0:
            return "" if for_csv else "—"
        pct = (delta / average) * 100
        if abs(pct) < 0.05:
            pct = 0.0
        return f"{pct:+.1f}%"

    raise ValueError(f"unknown relative mode: {relative}")


def format_average_summary(
    label: str,
    averages: dict[str, float],
    keys: list[str],
) -> str:
    parts = "  ".join(f"{key}={averages[key]:,.1f}" for key in keys)
    return f"{label}: {parts}"


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
    parser.add_argument(
        "--breakdown",
        "--deep",
        action="store_true",
        help="Break counts down by canonical module sections instead of lecture/post.",
    )
    parser.add_argument(
        "--reading-time",
        action="store_true",
        help=(
            "Report estimated lesson reading time per module instead of word counts. "
            "Covers reading only — not implementation or exercises."
        ),
    )
    relative_group = parser.add_mutually_exclusive_group()
    relative_group.add_argument(
        "--diff",
        action="store_true",
        help="Show signed difference from each numeric column's average.",
    )
    relative_group.add_argument(
        "--pct",
        action="store_true",
        help="Show signed percentage difference from each numeric column's average.",
    )
    args = parser.parse_args()
    relative = "diff" if args.diff else "pct" if args.pct else "none"

    paths = sorted(args.modules_dir.glob("*.md"))
    if not paths:
        print(f"No module .md files in {args.modules_dir}", file=sys.stderr)
        return 1

    if args.reading_time:
        rows = []
        for path in paths:
            lecture, post, _ = split_lecture_post(path.read_text())
            rows.append({
                "module": path.stem,
                "lecture": reading_minutes(wc_words(lecture)),
                "post": reading_minutes(wc_words(post)) if post.strip() else 0,
            })
        width = max(len(r["module"]) for r in rows)
        if args.format == "csv":
            buffer = io.StringIO()
            writer = csv.DictWriter(
                buffer,
                fieldnames=["module", "lecture_min", "homework_min", "total_min"],
                lineterminator="\n",
            )
            writer.writeheader()
            for r in rows:
                writer.writerow({
                    "module": r["module"],
                    "lecture_min": r["lecture"],
                    "homework_min": r["post"],
                    "total_min": r["lecture"] + r["post"],
                })
            print(buffer.getvalue().rstrip("\n"))
        else:
            print(f"{'Module'.ljust(width)}  Lecture  Homework  Total")
            print(f"{'-' * width}  -------  --------  -----")
            for r in rows:
                total = r["lecture"] + r["post"]
                post = f"{r['post']}m" if r["post"] else "—"
                print(
                    f"{r['module'].ljust(width)}  {r['lecture']:>6}m  {post:>8}  {total:>4}m"
                )
            print(
                f"\nReading only, at {WORDS_PER_MINUTE} wpm. Implementation and exercise "
                "time is not\nestimated here — it varies far more by reader than by module."
            )
        return 0

    if args.breakdown:
        rows = []
        totals = {spec.key: 0 for spec in DEEP_SECTIONS}
        totals["total"] = 0
        for path in paths:
            text = path.read_text()
            counts = deep_breakdown(text)
            row = {"module": path.stem, **counts, "total": wc_words(text)}
            rows.append(row)
            for key in totals:
                totals[key] += row[key]

        numeric_keys = [*(spec.key for spec in DEEP_SECTIONS), "total"]
        print(render_breakdown_table(rows, args.format, relative=relative))
        if args.format == "table":
            section_text = "  ".join(
                f"{spec.key}={totals[spec.key]:,}" for spec in DEEP_SECTIONS
            )
            print()
            if relative == "none":
                print(f"Totals: {section_text}  total={totals['total']:,}")
            else:
                averages = column_averages(rows, numeric_keys)
                print(format_average_summary("Averages", averages, numeric_keys))
        return 0

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

    print(render_table(rows, args.format, relative=relative))
    if args.format == "table":
        print()
        if relative == "none":
            print(
                f"Totals: lecture={totals['lecture']:,}  "
                f"post={totals['post']:,}  total={totals['total']:,}"
            )
        else:
            averages = column_averages(rows, ["lecture", "post", "total"])
            print(format_average_summary("Averages", averages, ["lecture", "post", "total"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
