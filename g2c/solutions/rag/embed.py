# Hand-maintained solutions mirror — canonical, lives in `main` (see AGENTS.md).
"""Solutions for g2c.rag.embed pedagogical scaffolds.

Patched onto the scaffold targets by g2c.solutions.apply().
"""
from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Any
import numpy as np
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_EMBED_MODEL = "nomic-embed-text"

from g2c.rag.embed import HashEmbedder


class _HashEmbedderImpl:  # patched onto HashEmbedder by apply()
    def embed(self, texts: list[str]) -> np.ndarray:
        """Hash-embed `texts` to a `(len(texts), self._dim)` float32 array.

        Args:
            texts: list of strings. May be empty. Each text may be
                empty (it embeds to a zero row).

        Returns:
            `(len(texts), self._dim)` float32, L2-normalized rows.

        Recipe:
            1. # Allocate the output.
               out = np.zeros((len(texts), self._dim), dtype=np.float32)

            2. # Tally n-gram bucket counts per text.
               n_min, n_max = self._ngram_range
               for i, text in enumerate(texts):
                   t = text.lower()
                   for n in range(n_min, n_max + 1):
                       if len(t) < n:
                           continue
                       for j in range(len(t) - n + 1):
                           ngram = t[j:j + n]
                           bucket = self._bucket(ngram)
                           out[i, bucket] += 1.0

            3. # L2-normalize each row.
               return _l2_normalize_rows(out)

        Implementation notes:

          * `texts` may legitimately be empty; the early return is a
            zero-row array. Don't raise on empty input — callers like
            `NumpyVectorStore.add` may have nothing to add.

          * Empty individual texts produce zero rows. They stay zero
            after `_l2_normalize_rows` (the helper handles zero norms).
            Cosine similarity against a zero row is 0 — which is the
            right behavior: "this thing has no signal."

          * The lowercase-then-window-then-hash pipeline is order-
            sensitive. Lowercasing AFTER hashing would make `"the"` and
            `"The"` collide differently — defeating the point of the
            normalization.

          * Character n-grams are robust to morphology and minor
            spelling differences ("running" and "runs" share `"run"`).
            Word n-grams are stronger when you have a tokenizer; we
            don't pull one in for this module.

          * BLAKE2b is overkill for a hash embedder (a one-line
            polynomial hash like FNV-1a would also work). The reason
            we use it is determinism: Python's built-in `hash()` is
            salted differently in each process, which breaks
            reproducibility — embeddings produced today wouldn't match
            embeddings produced tomorrow against the same text. Stable
            tests want stable hashes.

        Sanity values:

          * `embed([])` → shape `(0, dim)` float32 array.

          * `embed([""])` → shape `(1, dim)`, the single row is all
            zeros (no n-grams).

          * `embed(["abc", "abc"])` → both rows identical.

          * `embed(["abc"])` row is L2-normalized: `(out**2).sum() ==
            1.0` to within float32 precision.
        """
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        n_min, n_max = self._ngram_range

        for row, text in enumerate(texts):
            lowered = text.lower()
            for n in range(n_min, n_max + 1):
                if len(lowered) < n:
                    continue
                for start in range(len(lowered) - n + 1):
                    bucket = self._bucket(lowered[start : start + n])
                    out[row, bucket] += 1.0

        return _l2_normalize_rows(out).astype(np.float32, copy=False)

