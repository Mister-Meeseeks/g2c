#!/usr/bin/env python3
"""One-shot: build g2c/solutions/ mirror files from the solutions branch.

For each Python file under g2c/ that contains a scaffolded pedagogical
function (`# TODO` + `raise NotImplementedError`), look up the same file
on the `solutions` branch via `git show`, extract the matching method or
function bodies, and write a mirror file at `g2c/solutions/<rel>.py`.

Generated mirror layout (Recipe B):

    # g2c/solutions/autodiff/value.py
    \"""Solutions for g2c.autodiff.value pedagogical scaffolds.\"""
    from __future__ import annotations
    # ... imports copied from solutions:g2c/autodiff/value.py ...
    from g2c.autodiff.value import Value


    class _ValueImpl:
        \"""Method holder. Patched onto Value by g2c.solutions.apply().\"""

        def __add__(self, other):
            ...


    def some_module_level_function(x):
        ...

The patcher (g2c.solutions._patcher) walks these mirror modules and
binds the methods of `_<Name>Impl` onto the real `<Name>` class on the
matching `g2c/<rel>.py` module, plus any free functions onto the module.

This script is idempotent — re-running rewrites every mirror file from
scratch. It is intended to run once during migration and on subsequent
syncs from the solutions branch.
"""
from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
G2C = REPO_ROOT / "g2c"
MIRROR_ROOT = G2C / "solutions"
SOURCE_BRANCH = "solutions"


# ---------------------------------------------------------------------------
# Scaffold detection (copied from strip_solutions.py for consistency).
# ---------------------------------------------------------------------------


def _is_docstring(stmt: ast.stmt) -> bool:
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and isinstance(stmt.value.value, str)
    )


def _is_not_implemented_raise(stmt: ast.stmt) -> bool:
    if not isinstance(stmt, ast.Raise) or stmt.exc is None:
        return False
    exc = stmt.exc
    if isinstance(exc, ast.Name):
        return exc.id == "NotImplementedError"
    if isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name):
        return exc.func.id == "NotImplementedError"
    return False


def _function_is_scaffold(
    node: ast.FunctionDef | ast.AsyncFunctionDef, segment: str
) -> bool:
    body = node.body[1:] if node.body and _is_docstring(node.body[0]) else node.body
    return "# TODO" in segment and any(_is_not_implemented_raise(s) for s in body)


# ---------------------------------------------------------------------------
# Scaffold collection on main (current g2c/) and solutions (via git show).
# ---------------------------------------------------------------------------


def collect_scaffolds(source: str, filename: str) -> list[tuple[str | None, str]]:
    """Return [(class_name_or_None, function_name), ...] for scaffolds."""
    tree = ast.parse(source, filename=filename)
    out: list[tuple[str | None, str]] = []
    for stmt in tree.body:
        if isinstance(stmt, ast.ClassDef):
            for inner in stmt.body:
                if isinstance(inner, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    seg = ast.get_source_segment(source, inner) or ""
                    if _function_is_scaffold(inner, seg):
                        out.append((stmt.name, inner.name))
        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            seg = ast.get_source_segment(source, stmt) or ""
            if _function_is_scaffold(stmt, seg):
                out.append((None, stmt.name))
    return out


def git_show(ref: str, rel_path: Path) -> str | None:
    r = subprocess.run(
        ["git", "show", f"{ref}:{rel_path.as_posix()}"],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return r.stdout if r.returncode == 0 else None


# ---------------------------------------------------------------------------
# Extraction from the solutions-branch source.
# ---------------------------------------------------------------------------


def _function_block_lines(
    source: str, source_lines: list[str], node: ast.FunctionDef | ast.AsyncFunctionDef
) -> list[str]:
    """Return source lines for a function definition, including decorators."""
    start = node.lineno
    if node.decorator_list:
        start = min(d.lineno for d in node.decorator_list)
    end = node.end_lineno
    return source_lines[start - 1 : end]


def extract_function(
    source: str, class_name: str | None, func_name: str
) -> list[str] | None:
    tree = ast.parse(source)
    source_lines = source.splitlines(keepends=True)
    if class_name is None:
        for stmt in tree.body:
            if (
                isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef))
                and stmt.name == func_name
            ):
                return _function_block_lines(source, source_lines, stmt)
        return None
    for stmt in tree.body:
        if isinstance(stmt, ast.ClassDef) and stmt.name == class_name:
            for inner in stmt.body:
                if (
                    isinstance(inner, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and inner.name == func_name
                ):
                    return _function_block_lines(source, source_lines, inner)
            return None
    return None


def _rewrite_relative_import(stmt: ast.ImportFrom, target_package: str) -> str:
    """Render a relative `from .x import Y` as absolute against target_package.

    `target_package` is the dotted package the SOLUTIONS module lived in
    on the solutions branch (e.g. `g2c.agent` for `g2c/agent/agent.py`).
    """
    base_parts = target_package.split(".")
    # `level == 1` -> same package; `level == 2` -> parent; etc.
    if stmt.level > len(base_parts):
        raise ValueError(f"relative import beyond top: level={stmt.level}")
    parent_parts = base_parts[: len(base_parts) - (stmt.level - 1)]
    if stmt.module:
        full = ".".join(parent_parts + [stmt.module])
    else:
        full = ".".join(parent_parts)
    names = ", ".join(
        alias.name + (f" as {alias.asname}" if alias.asname else "")
        for alias in stmt.names
    )
    return f"from {full} import {names}\n"


def extract_imports_and_constants(
    source: str, target_package: str
) -> list[str]:
    """Return top-level lines worth copying into the mirror file.

    Includes:
      * imports (relative imports rewritten to absolute)
      * module-level Name/AnnAssign assignments (e.g. `Number = Union[int, float]`)

    Excludes:
      * `from __future__` imports (mirror emits its own)
      * class and function definitions (handled separately)
    """
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    out: list[str] = []
    for stmt in tree.body:
        if isinstance(stmt, ast.ImportFrom):
            if stmt.module == "__future__":
                continue
            if stmt.level > 0:
                out.append(_rewrite_relative_import(stmt, target_package))
            else:
                out.extend(lines[stmt.lineno - 1 : stmt.end_lineno])
        elif isinstance(stmt, ast.Import):
            out.extend(lines[stmt.lineno - 1 : stmt.end_lineno])
        elif isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            # Module-level constants / type aliases (e.g. Number = Union[int, float]).
            out.extend(lines[stmt.lineno - 1 : stmt.end_lineno])
    return out


# ---------------------------------------------------------------------------
# Mirror file generation.
# ---------------------------------------------------------------------------


GENERATED_BANNER = (
    "# AUTO-GENERATED by scripts/build_solutions_mirror.py — do not edit by hand.\n"
    "# Regenerate after updating the solutions branch:\n"
    "#     source .venv/bin/activate && python scripts/build_solutions_mirror.py\n"
)


def _dedent_method(block: list[str]) -> list[str]:
    """Remove one level (4 spaces) of indentation from a method block."""
    out = []
    for line in block:
        if line.startswith("    "):
            out.append(line[4:])
        elif line.strip() == "":
            out.append(line)
        else:
            out.append(line)
    return out


def _reindent_to_method(block: list[str]) -> list[str]:
    """Add 4-space indent so a top-level function reads as a class method."""
    out = []
    for line in block:
        if line.strip() == "":
            out.append(line)
        else:
            out.append("    " + line)
    return out


def build_mirror_file(
    rel_path: Path,
    scaffolds: list[tuple[str | None, str]],
    sol_source: str,
) -> str:
    """Render the mirror file source for a single g2c file."""
    target_module_dotted = "g2c." + ".".join(rel_path.with_suffix("").parts[1:])

    by_class: dict[str | None, list[str]] = defaultdict(list)
    for class_name, func_name in scaffolds:
        by_class[class_name].append(func_name)

    # Per-class extraction (each method already 4-space-indented in solutions).
    class_blocks: dict[str, list[str]] = {}
    for class_name, methods in by_class.items():
        if class_name is None:
            continue
        method_chunks: list[str] = []
        for method in methods:
            block = extract_function(sol_source, class_name, method)
            if block is None:
                method_chunks.append(
                    f"    # MISSING in solutions branch: {class_name}.{method}\n"
                    f"    def {method}(self, *args, **kwargs):\n"
                    f"        raise NotImplementedError\n\n"
                )
                continue
            method_chunks.extend(block)
            method_chunks.append("\n")
        class_blocks[class_name] = method_chunks

    # Free functions (extracted at module top level, no re-indent).
    free_blocks: list[list[str]] = []
    for func_name in by_class.get(None, []):
        block = extract_function(sol_source, None, func_name)
        if block is None:
            free_blocks.append(
                [
                    f"# MISSING in solutions branch: {func_name}\n",
                    f"def {func_name}(*args, **kwargs):\n",
                    f"    raise NotImplementedError\n",
                ]
            )
            continue
        free_blocks.append(block)

    # Imports + module-level constants from the solutions branch source.
    sol_package = ".".join(rel_path.with_suffix("").parts[:-1])  # e.g. "g2c.agent"
    sol_imports = extract_imports_and_constants(sol_source, sol_package)
    target_imports: list[str] = []
    if class_blocks:
        names = ", ".join(sorted(class_blocks.keys()))
        target_imports.append(f"from {target_module_dotted} import {names}\n")

    # Assemble.
    out: list[str] = []
    out.append(GENERATED_BANNER)
    out.append('"""Solutions for ' + target_module_dotted + " pedagogical scaffolds.\n")
    out.append("\n")
    out.append("Patched onto the scaffold targets by g2c.solutions.apply().\n")
    out.append('"""\n')
    out.append("from __future__ import annotations\n\n")
    out.extend(sol_imports)
    if target_imports:
        out.append("\n")
        out.extend(target_imports)

    for class_name, chunks in class_blocks.items():
        out.append("\n\n")
        out.append(
            f"class _{class_name}Impl:  # patched onto {class_name} by apply()\n"
        )
        out.extend(chunks)

    for block in free_blocks:
        out.append("\n\n")
        out.extend(block)

    return "".join(out)


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--check",
        action="store_true",
        help="Report files that would be written without writing them.",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)

    # Discover scaffold files in current g2c/.
    file_scaffolds: dict[Path, list[tuple[str | None, str]]] = {}
    for path in sorted(G2C.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        if "solutions" in path.parts:
            continue  # do not mirror the mirror
        rel = path.relative_to(REPO_ROOT)
        source = path.read_text()
        scaffolds = collect_scaffolds(source, str(rel))
        if scaffolds:
            file_scaffolds[rel] = scaffolds

    written = 0
    missing_sol_source = 0
    for rel, scaffolds in file_scaffolds.items():
        sol_source = git_show(SOURCE_BRANCH, rel)
        if sol_source is None:
            print(f"NO SOLUTIONS BRANCH FILE: {rel}", file=sys.stderr)
            missing_sol_source += 1
            continue

        mirror_rel = Path("g2c/solutions") / rel.relative_to("g2c")
        mirror_path = REPO_ROOT / mirror_rel

        content = build_mirror_file(rel, scaffolds, sol_source)
        if args.check:
            print(f"would write {mirror_rel} ({len(scaffolds)} scaffolds)")
            continue
        mirror_path.parent.mkdir(parents=True, exist_ok=True)
        mirror_path.write_text(content)
        # Make sure every parent has an __init__.py so the mirror is importable.
        parent = mirror_path.parent
        while parent != MIRROR_ROOT and parent != REPO_ROOT:
            init = parent / "__init__.py"
            if not init.exists():
                init.write_text(GENERATED_BANNER + '"""Mirror subpackage."""\n')
            parent = parent.parent
        written += 1

    print(
        f"\n{'would write' if args.check else 'wrote'} {written} mirror files; "
        f"{missing_sol_source} files had no solutions-branch source"
    )
    return 0 if missing_sol_source == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
