#!/usr/bin/env python3
"""Run the tests expected to pass on the pristine scaffold branch.

The full test suite intentionally contains many failing tests on `master`:
those are the student implementation targets. This runner reads the
checked-in manifest at `tests/clean_tests.txt` and invokes pytest only on
the boilerplate / already-implemented tests that should pass cleanly.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_nodeids(path: Path) -> list[str]:
    nodeids: list[str] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            nodeids.append(line)
    if not nodeids:
        raise SystemExit(f"No clean tests listed in {path}")
    return nodeids


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    root = repo_root()
    manifest = root / "tests" / "clean_tests.txt"
    nodeids = load_nodeids(manifest)

    print(
        f"Running {len(nodeids)} clean scaffold tests from {manifest.relative_to(root)}",
        flush=True,
    )
    cmd = [sys.executable, "-m", "pytest", *nodeids, *argv]
    return subprocess.run(cmd, cwd=root).returncode


if __name__ == "__main__":
    raise SystemExit(main())
