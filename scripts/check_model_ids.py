"""Check that the pretrained-model ids the course pulls still resolve upstream.

Modules 13-20 stand on models this repo does not host: BaseLM weights come
from the Hugging Face hub and the ProdLM/RAG models from the Ollama registry.
An upstream rename strands every new student entering Part II with nothing in
the repo having changed — the model-id analogue of dataset link rot, and the
reason this runs on the same weekly schedule as check_dataset_urls.py.

Like that checker, ids are *extracted from their source of truth*, not
restated here, so this check cannot drift from what the course actually pulls:

  - the ``DEFAULT_BASELM_MODEL_ID`` constant in ``g2c/artifacts/baselm.py``
    (Hugging Face model id)
  - every literal ``*MODEL_ID="..."`` default assignment in ``prodlm.sh``
    (Ollama model tags; the ``${2:?...}`` argument re-assignments contain a
    ``$`` and are skipped)

Checks are metadata-only: the HF model API, and the Ollama registry manifest
endpoint that ``ollama pull`` itself resolves tags against. Nothing larger
than a JSON document is fetched.

Usage:
    python3 scripts/check_model_ids.py

Exits 0 when every id resolves, 1 otherwise. Uses only the standard library,
so it runs without the project venv.
"""

from __future__ import annotations

import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

BASELM_SOURCE = "g2c/artifacts/baselm.py"
BASELM_CONSTANT = re.compile(r'^DEFAULT_BASELM_MODEL_ID\s*=\s*"([^"]+)"', re.MULTILINE)

PRODLM_SOURCE = "prodlm.sh"
OLLAMA_ASSIGNMENT = re.compile(r'^([A-Za-z_]*MODEL_ID)="([^"$]+)"', re.MULTILINE)

TIMEOUT_SECONDS = 30
USER_AGENT = "g2c-model-id-check"


def discover_ids() -> list[tuple[str, str, str]]:
    """Return (kind, site, model_id) triples from the scripts' own defaults."""
    ids: list[tuple[str, str, str]] = []

    baselm = (REPO_ROOT / BASELM_SOURCE).read_text()
    for match in BASELM_CONSTANT.finditer(baselm):
        ids.append(("huggingface", f"{BASELM_SOURCE}:DEFAULT_BASELM_MODEL_ID", match.group(1)))

    prodlm = (REPO_ROOT / PRODLM_SOURCE).read_text()
    for match in OLLAMA_ASSIGNMENT.finditer(prodlm):
        ids.append(("ollama", f"{PRODLM_SOURCE}:{match.group(1)}", match.group(2)))

    return ids


def check_url(url: str) -> tuple[bool, str]:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            # The Ollama registry speaks the Docker registry v2 protocol.
            "Accept": "application/vnd.docker.distribution.manifest.v2+json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            return True, f"HTTP {response.status}"
    except urllib.error.HTTPError as error:
        return False, f"HTTP {error.code}"
    except urllib.error.URLError as error:
        return False, f"unreachable ({error.reason})"


def probe_url(kind: str, model_id: str) -> str:
    if kind == "huggingface":
        return f"https://huggingface.co/api/models/{model_id}"
    # Ollama: "name" or "name:tag", optionally "namespace/name:tag".
    # Un-namespaced models live under "library/".
    name, _, tag = model_id.partition(":")
    if "/" not in name:
        name = f"library/{name}"
    return f"https://registry.ollama.ai/v2/{name}/manifests/{tag or 'latest'}"


def main() -> int:
    ids = discover_ids()
    if not ids:
        print("error: no model ids extracted — did the constants move or get renamed?")
        return 1

    failures = 0
    for kind, site, model_id in ids:
        ok, detail = check_url(probe_url(kind, model_id))
        status = "ok  " if ok else "FAIL"
        print(f"{status} [{kind}] {model_id}  ({site}; {detail})")
        failures += 0 if ok else 1

    if failures:
        print(f"\n{failures} model id(s) no longer resolve. If a model moved or was")
        print("renamed upstream, update the default in the file named above — the")
        print("id is extracted from there, so fixing the source fixes this check.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
