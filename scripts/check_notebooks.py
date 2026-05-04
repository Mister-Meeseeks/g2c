#!/usr/bin/env python3
"""Validate notebook JSON and Python code-cell syntax.

This is a lightweight scaffold check, not a notebook executor. By default it
checks canonical clean notebooks only, because solution notebooks may contain
in-progress student work.
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
CLEAN_DIR = REPO_ROOT / "notebooks" / "clean"
SOLUTIONS_DIR = REPO_ROOT / "notebooks" / "solutions"


def default_notebooks() -> list[Path]:
    return sorted(CLEAN_DIR.glob("*.ipynb"))


def all_notebooks() -> list[Path]:
    return sorted([*CLEAN_DIR.glob("*.ipynb"), *SOLUTIONS_DIR.glob("*.ipynb")])


def source_text(cell: dict[str, Any]) -> str:
    source = cell.get("source", "")
    if isinstance(source, str):
        return source
    if isinstance(source, list):
        return "".join(str(part) for part in source)
    raise TypeError(f"cell source must be str or list[str], got {type(source).__name__}")


def python_source_for_ast(source: str, *, strict_python: bool) -> str | None:
    """Return source suitable for `ast.parse`, or None for non-Python cells.

    Clean notebooks sometimes use small IPython conveniences such as
    `%load_ext autoreload` before ordinary Python. Those are valid notebook
    cells but invalid Python source, so the default check strips line magics
    and shell escapes. Cell magics (`%%...`) are treated as non-Python cells.
    """
    if strict_python:
        return source

    lines = source.splitlines(keepends=True)
    if lines and lines[0].lstrip().startswith("%%"):
        return None

    python_lines: list[str] = []
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("%") or stripped.startswith("!"):
            continue
        python_lines.append(line)
    return "".join(python_lines)


def check_notebook(path: Path, *, strict_python: bool) -> list[str]:
    errors: list[str] = []

    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        return [f"{path}: invalid notebook JSON: {exc}"]

    cells = data.get("cells")
    if not isinstance(cells, list):
        return [f"{path}: notebook has no top-level cells list"]

    for index, cell in enumerate(cells):
        if not isinstance(cell, dict):
            errors.append(f"{path}: cell {index} is not an object")
            continue
        if cell.get("cell_type") != "code":
            continue

        try:
            source = python_source_for_ast(
                source_text(cell),
                strict_python=strict_python,
            )
            if source is None or not source.strip():
                continue
            ast.parse(source, filename=f"{path}:cell-{index}")
        except SyntaxError as exc:
            line = exc.lineno or "?"
            col = exc.offset or "?"
            errors.append(f"{path}: code cell {index} syntax error at {line}:{col}: {exc.msg}")
        except Exception as exc:
            errors.append(f"{path}: code cell {index} could not be read: {exc}")

    return errors


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate notebook JSON and Python code-cell syntax.",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="notebooks to check; defaults to notebooks/clean/*.ipynb",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="check notebooks/clean and notebooks/solutions",
    )
    parser.add_argument(
        "--strict-python",
        action="store_true",
        help="parse raw code-cell source without allowing IPython magics",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    if args.paths and args.all:
        print("error: pass explicit paths or --all, not both", file=sys.stderr)
        return 2

    paths = args.paths or (all_notebooks() if args.all else default_notebooks())
    if not paths:
        print("No notebooks found.")
        return 0

    failures: list[str] = []
    for path in paths:
        failures.extend(check_notebook(path, strict_python=args.strict_python))

    if failures:
        for failure in failures:
            print(failure, file=sys.stderr)
        return 1

    print(f"Checked {len(paths)} notebook(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
