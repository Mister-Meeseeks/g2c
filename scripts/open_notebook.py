#!/usr/bin/env python3
"""Open a module notebook from the working solutions directory.

Clean notebooks live in notebooks/clean/ and should stay pristine. This script
creates or resumes the matching notebook in notebooks/solutions/.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CLEAN_DIR = REPO_ROOT / "notebooks" / "clean"
SOLUTIONS_DIR = REPO_ROOT / "notebooks" / "solutions"
REVISIONS_DIR = SOLUTIONS_DIR / "revisions"
JUPYTER_DIR = REPO_ROOT / ".jupyter"
JUPYTER_DATA_DIR = JUPYTER_DIR / "data"
JUPYTER_RUNTIME_DIR = JUPYTER_DIR / "runtime"
JUPYTER_CONFIG_DIR = JUPYTER_DIR / "config"
IPYTHON_DIR = JUPYTER_DIR / "ipython"


# Beyond modules are addressed by slug, not number (see docs/beyond/).
BEYOND_SLUGS = {
    "moe",
    "linear-attention",
    "rl",
    "multimodal",
    "harness",
    "midtraining",
    "muon",
    "specdec",
}


def module_prefix(raw: str) -> str:
    """Normalize module ids like '1', '01', '03b' — or a Beyond slug."""
    lowered = raw.lower()
    if lowered in BEYOND_SLUGS:
        return lowered
    match = re.fullmatch(r"(\d+)([a-zA-Z]?)", raw)
    if not match:
        raise argparse.ArgumentTypeError(
            "module must look like 1, 01, or 03b — or a Beyond slug: "
            + ", ".join(sorted(BEYOND_SLUGS))
        )
    number, suffix = match.groups()
    return f"{int(number):02d}{suffix.lower()}"


def find_clean_notebook(prefix: str) -> Path:
    matches = sorted(CLEAN_DIR.glob(f"{prefix}-*.ipynb"))
    if not matches:
        pattern = CLEAN_DIR / f"{prefix}-*.ipynb"
        raise FileNotFoundError(f"No clean notebook found matching {pattern}")
    if len(matches) > 1:
        names = ", ".join(path.name for path in matches)
        raise RuntimeError(f"Multiple clean notebooks match module {prefix}: {names}")
    return matches[0]


def revision_path_for(solution_path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    candidate = REVISIONS_DIR / f"{solution_path.stem}.{timestamp}{solution_path.suffix}"

    suffix = 2
    while candidate.exists():
        filename = f"{solution_path.stem}.{timestamp}.{suffix}{solution_path.suffix}"
        candidate = REVISIONS_DIR / filename
        suffix += 1

    return candidate


def prepare_solution_notebook(clean_path: Path, fresh: bool) -> tuple[Path, Path | None, str]:
    SOLUTIONS_DIR.mkdir(parents=True, exist_ok=True)
    solution_path = SOLUTIONS_DIR / clean_path.name

    if solution_path.exists() and not fresh:
        return solution_path, None, "existing"

    revision_path = None
    if solution_path.exists() and fresh:
        REVISIONS_DIR.mkdir(parents=True, exist_ok=True)
        revision_path = revision_path_for(solution_path)
        shutil.copy2(solution_path, revision_path)

    shutil.copy2(clean_path, solution_path)
    return solution_path, revision_path, "fresh" if revision_path else "created"


def notebook_env() -> dict[str, str]:
    """Return an environment with project-local writable Jupyter state."""
    for path in (JUPYTER_DATA_DIR, JUPYTER_RUNTIME_DIR, JUPYTER_CONFIG_DIR, IPYTHON_DIR):
        path.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["JUPYTER_DATA_DIR"] = str(JUPYTER_DATA_DIR)
    env["JUPYTER_RUNTIME_DIR"] = str(JUPYTER_RUNTIME_DIR)
    env["JUPYTER_CONFIG_DIR"] = str(JUPYTER_CONFIG_DIR)
    env["IPYTHONDIR"] = str(IPYTHON_DIR)
    return env


def launch_jupyter(notebook_path: Path) -> int:
    try:
        import jupyter  # noqa: F401
    except ModuleNotFoundError:
        print(
            "Jupyter is not installed in this Python environment. Run ./setup.sh, "
            "then use .venv/bin/python scripts/open_notebook.py <module>.",
            file=sys.stderr,
        )
        return 1

    process = subprocess.Popen(
        [sys.executable, "-m", "jupyter", "notebook", str(notebook_path)],
        env=notebook_env(),
    )

    while True:
        try:
            return process.wait()
        except KeyboardInterrupt:
            # Jupyter handles Ctrl-C itself by asking whether to shut down.
            # Keep the wrapper alive while the child process handles that prompt.
            continue


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create or open a working notebook copy for a module.",
    )
    parser.add_argument("module", type=module_prefix, help="module id, e.g. 1, 01, or 03b")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="archive any existing solution notebook, then replace it from notebooks/clean/",
    )
    parser.add_argument(
        "--no-launch",
        action="store_true",
        help="prepare the solution notebook but do not launch Jupyter",
    )
    parser.add_argument(
        "--solutions",
        nargs="?",
        const="1",
        default=None,
        metavar="WHICH",
        help=(
            "launch the notebook with worked implementations from g2c/solutions/ "
            "applied at import time (sets G2C_APPLY_SOLUTIONS). Bare --solutions "
            "applies every module; --solutions=01-07 applies only those, leaving "
            "your own code in place everywhere else"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # `notebook.sh` exports G2C_APPLY_SOLUTIONS itself and strips the flag
    # before handing off, so this only fires on direct invocation. Pass the
    # selector through verbatim rather than forcing "1", or a narrowed
    # selection would silently widen to every module.
    if args.solutions is not None:
        os.environ["G2C_APPLY_SOLUTIONS"] = args.solutions

    try:
        clean_path = find_clean_notebook(args.module)
        solution_path, revision_path, state = prepare_solution_notebook(clean_path, args.fresh)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if state == "existing":
        print(f"Opening existing notebook: {solution_path.relative_to(REPO_ROOT)}")
    elif state == "created":
        print(f"Created notebook from clean scaffold: {solution_path.relative_to(REPO_ROOT)}")
    else:
        assert revision_path is not None
        print(f"Archived previous notebook: {revision_path.relative_to(REPO_ROOT)}")
        print(f"Reset notebook from clean scaffold: {solution_path.relative_to(REPO_ROOT)}")

    if args.no_launch:
        return 0

    return launch_jupyter(solution_path)


if __name__ == "__main__":
    raise SystemExit(main())
