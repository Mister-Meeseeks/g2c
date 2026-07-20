"""Small KV-cache containers for decoder-only inference.

The cache stores key/value projections that have already been computed during
autoregressive decoding. This is intentionally simple and pedagogical:

* one ``LayerKVCache`` per transformer block;
* tensors are appended along the sequence axis;
* no batching tricks, paging, eviction, or rolling windows.

The interesting math lives in ``MultiHeadAttention.forward_cached``. These
containers handle the shape bookkeeping so students can focus on the cache
idea itself.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class LayerKVCache:
    """Cached keys and values for one attention layer.

    Stored shapes are ``(B, H, T_cache, head_dim)``. ``keys`` and ``values`` are
    either both ``None`` for an empty cache or both tensors with matching shape.
    """

    keys: torch.Tensor | None = None
    values: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if (self.keys is None) != (self.values is None):
            raise ValueError("keys and values must both be None or both tensors")
        if self.keys is not None:
            _validate_kv_pair(self.keys, self.values)

    @property
    def length(self) -> int:
        """Number of cached sequence positions."""
        if self.keys is None:
            return 0
        return int(self.keys.shape[-2])

    def _validate_append(self, key: torch.Tensor, value: torch.Tensor) -> None:
        """Check that ``(key, value)`` may be appended to this cache.

        Provided for you — this is bookkeeping, not the concept. Raises
        ``ValueError`` if the pair is malformed on its own, or if it
        disagrees with already-cached rows on batch, head count, head_dim,
        device, or dtype.
        """
        _validate_kv_pair(key, value)
        if self.keys is None:
            return

        assert self.values is not None
        if key.shape[:2] != self.keys.shape[:2] or key.shape[-1] != self.keys.shape[-1]:
            raise ValueError(
                "new key/value tensors must match existing batch, head, and "
                f"head_dim; got {tuple(key.shape)} after {tuple(self.keys.shape)}"
            )
        if key.device != self.keys.device or value.device != self.values.device:
            raise ValueError("new key/value tensors must be on the cache device")
        if key.dtype != self.keys.dtype or value.dtype != self.values.dtype:
            raise ValueError("new key/value tensors must match the cache dtype")

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

        Recipe (validation is already called for you below — start after it):

            1. If the cache is still empty (``self.keys is None``), the new
               tensors simply become the cache. Store both and return.

            2. Otherwise concatenate onto what's already there:
                   self.keys = torch.cat([self.keys, key], dim=-2)

               and the same for ``self.values``.

               ``dim=-2`` is the ``T_cache`` slot of
               ``(B, H, T_cache, head_dim)``. This axis is the one thing
               to get right: ``dim=-1`` would grow head_dim and ``dim=0``
               would grow the batch. Both "work" here and then fail as a
               confusing shape error inside attention, far from the cause.

            3. Return ``self`` so callers can chain.
        """
        self._validate_append(key, value)

        # TODO
        raise NotImplementedError


@dataclass
class KVCache:
    """KV cache for a full transformer stack."""

    layers: list[LayerKVCache]

    @classmethod
    def empty(cls, num_layers: int) -> KVCache:
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        return cls([LayerKVCache() for _ in range(num_layers)])

    def __len__(self) -> int:
        return len(self.layers)

    @property
    def length(self) -> int:
        """Common cache length across all layers.

        A valid decoder cache has the same number of cached positions in every
        layer. If a partial update left layers out of sync, fail loudly.
        """
        if not self.layers:
            return 0
        lengths = {layer.length for layer in self.layers}
        if len(lengths) != 1:
            raise ValueError(f"KV cache layers have inconsistent lengths: {lengths}")
        return lengths.pop()


def _validate_kv_pair(key: torch.Tensor, value: torch.Tensor | None) -> None:
    if value is None:
        raise ValueError("value tensor is required when key tensor is present")
    if key.dim() != 4 or value.dim() != 4:
        raise ValueError(
            "key and value tensors must have shape (B, H, T, head_dim); "
            f"got {tuple(key.shape)} and {tuple(value.shape)}"
        )
    if key.shape != value.shape:
        raise ValueError(
            f"key and value shapes must match; got {tuple(key.shape)} and "
            f"{tuple(value.shape)}"
        )
