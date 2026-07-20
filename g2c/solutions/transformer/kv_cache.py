# Hand-maintained solutions mirror — canonical, lives in `main` (see AGENTS.md).
"""Solutions for g2c.transformer.kv_cache pedagogical scaffolds.

Patched onto the scaffold targets by g2c.solutions.apply().
"""
from __future__ import annotations

import torch

from g2c.transformer.kv_cache import LayerKVCache


class _LayerKVCacheImpl:  # patched onto LayerKVCache by apply()
    def append(self, key: torch.Tensor, value: torch.Tensor) -> LayerKVCache:
        """Append one or more key/value positions and return ``self``.

        Args:
            key: ``(B, H, T_new, head_dim)``
            value: ``(B, H, T_new, head_dim)``

        Returns:
            ``self``, with ``keys`` and ``values`` each grown by ``T_new``
            positions along the sequence axis.

        This is the whole idea of a KV cache in one method: keys and values
        for past tokens are never recomputed, they are *kept* and extended.
        """
        self._validate_append(key, value)

        if self.keys is None:
            self.keys = key
            self.values = value
            return self

        self.keys = torch.cat([self.keys, key], dim=-2)
        self.values = torch.cat([self.values, value], dim=-2)
        return self
