#!/usr/bin/env python3
"""Strip worked solution bodies back to scaffold TODOs.

This repo has two long-running branches:

* ``master`` keeps the course pristine, with pedagogical functions left as
  ``# TODO`` + ``raise NotImplementedError``.
* ``solutions`` keeps those same files implemented so the author can run worked
  notebooks and model-training experiments.

When course edits happen on ``solutions`` and need to flow back to ``master``,
plain Git merges are not enough: they also bring filled-in student exercise
code. This tool makes that cleanup deterministic.

It strips functions by either signal:

1. A ``# SOLUTION`` marker anywhere inside the function body.
2. ``--reference REF``: if the same qualified function in ``REF`` was a
   scaffolded TODO, strip the target function too. This is useful immediately
   after merging ``solutions`` into ``master``: ``HEAD`` is the pre-merge
   scaffold reference.

Examples:

    python scripts/strip_solutions.py --reference HEAD g2c
    python scripts/strip_solutions.py --reference master g2c --check
    python scripts/strip_solutions.py g2c
"""
from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


SOLUTION_MARKERS = ("# SOLUTION", "# BEGIN SOLUTION", "# END SOLUTION")


@dataclass(frozen=True)
class FunctionInfo:
    """Line span and qualified name for one Python function."""

    qualname: str
    lineno: int
    end_lineno: int
    body_start: int
    indent: int
    segment: str
    is_scaffold: bool
    has_solution_marker: bool


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def is_docstring(stmt: ast.stmt) -> bool:
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and isinstance(stmt.value.value, str)
    )


def is_not_implemented_raise(stmt: ast.stmt) -> bool:
    if not isinstance(stmt, ast.Raise) or stmt.exc is None:
        return False
    exc = stmt.exc
    if isinstance(exc, ast.Name):
        return exc.id == "NotImplementedError"
    if isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name):
        return exc.func.id == "NotImplementedError"
    return False


def body_start_line(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    if node.body and is_docstring(node.body[0]):
        return node.body[0].end_lineno + 1
    if node.body:
        return node.body[0].lineno
    return node.lineno + 1


def function_is_scaffold(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    segment: str,
) -> bool:
    """Return True for direct pedagogical TODO stubs.

    The ``# TODO`` check intentionally excludes abstract methods such as
    ``Module.forward`` whose ``raise NotImplementedError`` is normal class
    design, not a student exercise.
    """
    body = node.body[1:] if node.body and is_docstring(node.body[0]) else node.body
    return "# TODO" in segment and any(is_not_implemented_raise(stmt) for stmt in body)


def has_solution_marker(segment: str) -> bool:
    return any(marker in segment for marker in SOLUTION_MARKERS)


def collect_functions(source: str, filename: str) -> dict[str, FunctionInfo]:
    tree = ast.parse(source, filename=filename)
    functions: dict[str, FunctionInfo] = {}

    def visit_body(body: list[ast.stmt], prefix: list[str]) -> None:
        for stmt in body:
            if isinstance(stmt, ast.ClassDef):
                visit_body(stmt.body, prefix + [stmt.name])
            elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualname = ".".join(prefix + [stmt.name])
                segment = ast.get_source_segment(source, stmt) or ""
                functions[qualname] = FunctionInfo(
                    qualname=qualname,
                    lineno=stmt.lineno,
                    end_lineno=stmt.end_lineno,
                    body_start=body_start_line(stmt),
                    indent=stmt.col_offset + 4,
                    segment=segment,
                    is_scaffold=function_is_scaffold(stmt, segment),
                    has_solution_marker=has_solution_marker(segment),
                )
                visit_body(stmt.body, prefix + [stmt.name])

    visit_body(tree.body, [])
    return functions


def git_show(ref: str, rel_path: Path, root: Path) -> str | None:
    result = subprocess.run(
        ["git", "show", f"{ref}:{rel_path.as_posix()}"],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout


def iter_python_files(paths: list[Path], root: Path) -> list[Path]:
    files: list[Path] = []
    for raw_path in paths:
        path = raw_path if raw_path.is_absolute() else root / raw_path
        if path.is_file() and path.suffix == ".py":
            files.append(path)
        elif path.is_dir():
            files.extend(
                sorted(
                    child
                    for child in path.rglob("*.py")
                    if "__pycache__" not in child.parts
                )
            )
        else:
            raise SystemExit(f"not a Python file or directory: {raw_path}")
    return sorted(dict.fromkeys(files))


def scaffold_body(indent: int) -> list[str]:
    prefix = " " * indent
    return [
        f"{prefix}# TODO\n",
        f"{prefix}raise NotImplementedError\n",
    ]


def strip_file(
    path: Path,
    *,
    root: Path,
    reference: str | None,
    check: bool,
) -> list[str]:
    rel_path = path.relative_to(root)
    source = path.read_text()
    lines = source.splitlines(keepends=True)
    target_functions = collect_functions(source, str(path))

    reference_scaffolds: set[str] = set()
    if reference is not None:
        reference_source = git_show(reference, rel_path, root)
        if reference_source is not None:
            reference_functions = collect_functions(
                reference_source,
                f"{reference}:{rel_path.as_posix()}",
            )
            reference_scaffolds = {
                qualname
                for qualname, info in reference_functions.items()
                if info.is_scaffold
            }

    replacements: list[tuple[int, int, list[str], str]] = []
    changed: list[str] = []
    for qualname, info in target_functions.items():
        should_strip = info.has_solution_marker or qualname in reference_scaffolds
        if not should_strip or info.is_scaffold:
            continue
        replacements.append(
            (
                info.body_start,
                info.end_lineno,
                scaffold_body(info.indent),
                qualname,
            )
        )
        changed.append(f"{rel_path}:{qualname}")

    if replacements and not check:
        for start, end, new_lines, _ in sorted(replacements, reverse=True):
            lines[start - 1 : end] = new_lines
        path.write_text("".join(lines))

    return changed


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strip worked solution bodies back to TODO scaffolds.",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[Path("g2c")],
        help="Python files or directories to scan. Defaults to g2c/.",
    )
    parser.add_argument(
        "--reference",
        help=(
            "Git ref whose scaffolded functions should be stripped in the "
            "working tree, e.g. HEAD before committing a solutions merge."
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report files that would change, but do not rewrite them.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    root = repo_root()
    files = iter_python_files(args.paths, root)

    changed: list[str] = []
    for path in files:
        changed.extend(
            strip_file(
                path,
                root=root,
                reference=args.reference,
                check=args.check,
            )
        )

    verb = "would strip" if args.check else "stripped"
    if changed:
        print(f"{verb} {len(changed)} function(s):")
        for item in changed:
            print(f"  {item}")
    else:
        print("no solution bodies found")

    return 1 if args.check and changed else 0


if __name__ == "__main__":
    raise SystemExit(main())
