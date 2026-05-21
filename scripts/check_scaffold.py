#!/usr/bin/env python3
"""Check that scaffolded pedagogical functions stayed scaffolded.

This is the invariant check for clean backports from ``solutions`` to
``main``. It compares the working tree against a reference ref and fails if
any function that was scaffolded in the reference is no longer a direct
``# TODO`` + ``raise NotImplementedError`` stub.

Typical use after merging course edits from ``solutions`` into ``main``:

    python scripts/strip_solutions.py --reference HEAD g2c
    python scripts/check_scaffold.py --reference HEAD g2c
    python scripts/test_clean.py

``HEAD`` works in that flow because it is still the pre-merge pristine scaffold
commit. For other situations, pass the relevant clean ref explicitly.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from strip_solutions import (
    collect_functions,
    git_show,
    iter_python_files,
    repo_root,
)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify scaffold TODO bodies were not filled in.",
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
        default="HEAD",
        help="Git ref containing the expected scaffold state. Defaults to HEAD.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    root = repo_root()
    files = iter_python_files(args.paths, root)

    violations: list[str] = []
    checked = 0

    for path in files:
        rel_path = path.relative_to(root)
        reference_source = git_show(args.reference, rel_path, root)
        if reference_source is None:
            continue

        current_source = path.read_text()
        reference_functions = collect_functions(
            reference_source,
            f"{args.reference}:{rel_path.as_posix()}",
        )
        current_functions = collect_functions(current_source, str(path))

        for qualname, reference_info in reference_functions.items():
            if not reference_info.is_scaffold:
                continue
            checked += 1
            current_info = current_functions.get(qualname)
            if current_info is None:
                violations.append(f"{rel_path}:{qualname} missing")
            elif not current_info.is_scaffold:
                violations.append(f"{rel_path}:{qualname} is no longer scaffolded")

        for qualname, current_info in current_functions.items():
            if current_info.has_solution_marker:
                violations.append(f"{rel_path}:{qualname} still has # SOLUTION marker")

    if violations:
        print(
            f"scaffold invariant failed against {args.reference}: "
            f"{len(violations)} issue(s)"
        )
        for item in violations:
            print(f"  {item}")
        return 1

    print(
        f"scaffold invariant passed against {args.reference}: "
        f"{checked} scaffold function(s) checked"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
