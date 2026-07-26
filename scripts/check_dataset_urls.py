"""Check that every dataset download URL the course ships is still reachable.

External corpora move. When they do, the failure surfaces to a student running
``./setup.sh`` on a fresh clone, which is the worst possible place to find out.
This script checks the URLs ahead of them.

The URLs are *extracted from the shell scripts*, not restated here, so this
check cannot drift away from what the scripts actually download. The rule: any
shell variable whose name ends in ``_URL`` or ``_url`` is treated as a dataset
download and gets checked. Install hints (Homebrew, pyenv, the Ollama download
page) live in help text rather than such variables, so they are ignored.

Adding a new download? Assign it to a ``*_URL`` variable and it is covered.

Usage:
    python3 scripts/check_dataset_urls.py

Exits 0 when every URL is reachable, 1 otherwise. Uses only the standard
library, so it runs without the project venv.
"""

from __future__ import annotations

import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Shell scripts that download course data.
SCRIPTS = ("setup.sh", "datasets.sh", "checkpoints.sh")

# `FOO_URL="https://..."` or `local foo_url="https://..."`.
URL_ASSIGNMENT = re.compile(
    r'^\s*(?:local\s+)?([A-Za-z_][A-Za-z0-9_]*_(?:URL|url))\s*=\s*"([^"]+)"',
    re.MULTILINE,
)

TIMEOUT_SECONDS = 30

# Some CDNs reject HEAD but honor a ranged GET, so we ask for a single byte
# rather than pulling down a 1.9GB corpus to prove it exists.
USER_AGENT = "g2c-dataset-url-check"


def discover_urls() -> dict[str, list[str]]:
    """Map each distinct dataset URL to the ``script:variable`` sites using it.

    The same corpus can be referenced from more than one place (TinyStories is
    downloaded whole and also sampled), so URLs are deduplicated: each is
    fetched once, but every site is reported so a fix lands everywhere.
    """
    found: dict[str, list[str]] = {}
    for script in SCRIPTS:
        path = REPO_ROOT / script
        if not path.exists():
            print(f"warn: {script} not found; skipping")
            continue
        for name, url in URL_ASSIGNMENT.findall(path.read_text()):
            # Skip templated URLs (e.g. .../{repo}/resolve/main/{filename});
            # those are formatted at runtime and have no fixed target.
            if "{" in url:
                continue
            site = f"{script}:{name}"
            sites = found.setdefault(url, [])
            # The same variable name can appear in two functions; list it once.
            if site not in sites:
                sites.append(site)
    return found


def check(url: str) -> tuple[bool, str]:
    """Return (ok, detail) for a single URL."""
    request = urllib.request.Request(
        url,
        method="GET",
        headers={"User-Agent": USER_AGENT, "Range": "bytes=0-0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            return True, f"HTTP {response.status}"
    except urllib.error.HTTPError as exc:
        # 416 means the server ignored our Range but the resource is there.
        if exc.code == 416:
            return True, "HTTP 416 (range ignored; resource exists)"
        return False, f"HTTP {exc.code} {exc.reason}"
    except urllib.error.URLError as exc:
        return False, f"unreachable: {exc.reason}"
    except OSError as exc:
        return False, f"error: {exc}"


def main() -> int:
    urls = discover_urls()
    if not urls:
        print("FAIL: no dataset URLs discovered — has the assignment style changed?")
        return 1

    print(f"Checking {len(urls)} dataset URL(s)\n")
    failures = []
    for url, sites in urls.items():
        ok, detail = check(url)
        status = " ok " if ok else "FAIL"
        print(f"[{status}] {', '.join(sites)}\n         {url}\n         {detail}\n")
        if not ok:
            failures.append((url, sites, detail))

    if failures:
        print(f"{len(failures)} of {len(urls)} dataset URL(s) unreachable:\n")
        for url, sites, detail in failures:
            print(f"  {url}\n    {detail}\n    referenced by: {', '.join(sites)}")
        print(
            "\nFix the URL at each site listed above. A dead URL here means "
            "./setup.sh or ./datasets.sh fails on a fresh clone."
        )
        return 1

    print(f"All {len(urls)} dataset URL(s) reachable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
