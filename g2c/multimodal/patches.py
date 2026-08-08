"""Patchify and PatchEmbedding — the tokenizer for images.

Module 04 solved "text is continuous, models need chunks" with BPE.
The field's answer for images (from ViT) is simpler: cut the pixel
grid into fixed-size squares, flatten each square, and apply one
learned linear projection. The projection plays the embedding table's
role — with the structural difference that there is no discrete
vocabulary. A patch is transformed, not looked up.

`patchify` is pure index arithmetic (no learned parameters);
`PatchEmbedding.forward` composes it with the projection. Both are
scaffolded — the index arithmetic IS the exercise.
"""
from __future__ import annotations

from collections.abc import Iterable

import torch

from g2c.nn import Linear, Module


def patchify(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Slice a batch of square images into flattened patches, row-major.

    Args:
        images: `(B, H, W)` float tensor. `H` and `W` must both be
            divisible by `patch_size`.
        patch_size: side length `p` of each square patch.

    Returns:
        `(B, num_patches, p*p)` where `num_patches = (H//p) * (W//p)`.
        Patch order is row-major over the patch grid: the patch at
        grid position (row `r`, column `c`) lands at index
        `r * (W//p) + c`. Within a patch, pixels are row-major too.

    Raises:
        ValueError: if `H` or `W` is not divisible by `patch_size`.

    Recipe:
        1. B, H, W = images.shape
           validate divisibility, gh, gw = H // p, W // p
        2. x = images.reshape(B, gh, p, gw, p)
           # axes: (batch, patch-row, pixel-row, patch-col, pixel-col)
        3. x = x.permute(0, 1, 3, 2, 4)
           # → (batch, patch-row, patch-col, pixel-row, pixel-col):
           # bring the two grid axes together, the two pixel axes
           # together. This permute is the entire trick.
        4. return x.reshape(B, gh * gw, p * p)

    The operation is lossless — `test_patchify_roundtrip` inverts it
    exactly. If your version passes the shape test but fails the
    roundtrip, the permute in step 3 is missing or wrong: plain
    reshape without it interleaves pixels across patches.
    """
    # TODO
    raise NotImplementedError


class PatchEmbedding(Module):
    """Project flattened patches into the residual stream.

    Args:
        patch_size: side length of each square patch (`p`).
        embedding_dim: output channel dim (`D`) — the same `D` as the
            language model the patches will be spliced into.

    Attributes:
        proj: `Linear(p*p, D)` — the "embedding table" for patches.
    """

    patch_size: int
    embedding_dim: int
    proj: Linear

    def __init__(self, patch_size: int, embedding_dim: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.proj = Linear(patch_size * patch_size, embedding_dim)

    def parameters(self) -> Iterable[torch.Tensor]:
        return list(self.proj.parameters())

    def num_patches(self, height: int, width: int) -> int:
        """How many patches one `(height, width)` image produces."""
        return (height // self.patch_size) * (width // self.patch_size)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Patchify and project: images in, patch vectors out.

        Args:
            images: `(B, H, W)` float tensor, already normalized (the
                batch builder handles pixel scaling — see
                `build_caption_batch`; don't normalize twice).

        Returns:
            `(B, num_patches, embedding_dim)` — one vector per patch,
            ready to be spliced into the token stream.

        Recipe:
            1. patches = patchify(images, self.patch_size)
            2. return self.proj(patches)

        Two lines. This is enough to expose the modality-to-residual-stream
        interface on MNIST. Production systems usually place a substantial
        vision encoder and often a resampler before this projection; the
        raw-pixel linear layer is the teaching simplification, not a claim
        about sufficient real-world visual capacity.
        """
        # TODO
        raise NotImplementedError
