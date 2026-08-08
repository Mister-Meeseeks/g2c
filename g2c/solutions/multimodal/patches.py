# Hand-maintained solutions mirror — canonical, lives in `main` (see AGENTS.md).
"""Solutions for g2c.multimodal.patches pedagogical scaffolds.

Patched onto the scaffold targets by g2c.solutions.apply().
"""
from __future__ import annotations

import torch


def patchify(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    B, H, W = images.shape
    p = patch_size
    if H % p != 0 or W % p != 0:
        raise ValueError(
            f"image size ({H}, {W}) not divisible by patch_size {p}"
        )
    gh, gw = H // p, W // p
    x = images.reshape(B, gh, p, gw, p)
    x = x.permute(0, 1, 3, 2, 4)
    return x.reshape(B, gh * gw, p * p)


class _PatchEmbeddingImpl:  # patched onto PatchEmbedding by apply()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        from g2c.multimodal.patches import patchify

        patches = patchify(images, self.patch_size)
        return self.proj(patches)
