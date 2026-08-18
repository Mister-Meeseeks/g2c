"""MultimodalLM — patch vectors spliced into an unmodified TransformerLM.

The design fact this file demonstrates: the transformer needs NO
architectural change to accept images. `MultimodalLM` wraps a Module 09
`TransformerLM` and a `PatchEmbedding`; its forward pass runs the same
embed → blocks → norm → unembed pipeline, with exactly one twist — the
embedding rows at `<image_patch>` placeholder positions are overwritten
with patch vectors before the blocks run. Retained `<img>` and `</img>`
tokens mark the image boundaries.

The placeholder convention keeps every shape static: the token sequence
contains `num_patches` copies of the placeholder id per image, so the
splice is a 1:1 masked overwrite — no length changes, no re-alignment
of positions or loss masks.

`build_caption_batch` is provided plumbing (it reuses Module 13's
shift-and-mask discipline); `MultimodalLM.forward` — the splice — is
the scaffold.
"""
from __future__ import annotations

from collections.abc import Iterable

import torch

from g2c.nn import Module
from g2c.transformer import TransformerLM

from .patches import PatchEmbedding
from .vocab import (
    IMAGE_END_ID,
    IMAGE_PATCH_ID,
    IMAGE_START_ID,
    PAD_ID,
    caption_ids,
)


class MultimodalLM(Module):
    """A `TransformerLM` that reads patch vectors at placeholder positions.

    Args:
        lm: a Module 09 `TransformerLM`, unmodified. Its vocab must
            include the placeholder id (`patch_token_id`).
        patch_size: side length of the square patches.
        patch_token_id: the vocab id whose embedding rows get replaced
            by patch vectors. Defaults to `<image_patch>`.

    Attributes:
        patch_embed: the `PatchEmbedding` producing splice-ready
            vectors at the LM's `embedding_dim`.
    """

    lm: TransformerLM
    patch_embed: PatchEmbedding
    patch_token_id: int

    def __init__(
        self,
        lm: TransformerLM,
        patch_size: int,
        *,
        patch_token_id: int = IMAGE_PATCH_ID,
    ) -> None:
        super().__init__()
        self.lm = lm
        self.patch_embed = PatchEmbedding(patch_size, lm.embedding_dim)
        self.patch_token_id = patch_token_id

    def parameters(self) -> Iterable[torch.Tensor]:
        return [*self.lm.parameters(), *self.patch_embed.parameters()]

    @property
    def max_seq_len(self) -> int:
        return self.lm.max_seq_len

    def forward(
        self, token_ids: torch.Tensor, images: torch.Tensor
    ) -> torch.Tensor:
        """Embed, splice, and run the wrapped transformer.

        Args:
            token_ids: `(B, T)` LongTensor. Must contain exactly
                `num_patches × n_images` placeholder ids per row, in
                the order the images' patches should occupy them.
            images: `(B, H, W)` for one image per sequence, or
                `(B, N, H, W)` for `N` images per sequence. Already
                normalized.

        Returns:
            `(B, T, vocab_size)` logits — same contract as
            `TransformerLM.forward`.

        Raises:
            ValueError: if the number of placeholder positions does
                not match the number of patch vectors.

        Recipe:
            1. if images.dim() == 3: images = images.unsqueeze(1)
               B, N, H, W = images.shape
            2. patches = self.patch_embed(images.reshape(B * N, H, W))
               patches = patches.reshape(B, -1, self.lm.embedding_dim)
               # (B, N * num_patches, D) — images in order, patches
               # row-major, matching the placeholder order.
            3. tok = self.lm.token_embed(token_ids)          # (B, T, D)
               mask = token_ids == self.patch_token_id       # (B, T)
               counts = mask.sum(dim=1)                      # (B,)
               expected = patches.shape[1]                   # per example
               if not torch.all(counts == expected):
                   raise ValueError(...)

               Validate per example, not only across the whole batch. A
               batch-wide total would let one row's missing placeholder be
               silently filled by another row's extra placeholder.
            4. tok = tok.clone()
               tok[mask] = patches.reshape(-1, self.lm.embedding_dim)
               # The splice. Boolean assignment fills mask positions in
               # row-major order — image order must match placeholder
               # order, which the sequence builder must guarantee.
            5. # The rest is TransformerLM.forward, verbatim — run it
               # here over the spliced embeddings:
               _, T = token_ids.shape
               x = tok + self.lm.pos_embed(T)
               for block in self.lm.blocks:
                   x = block(x)
               x = self.lm.ln_final(x)
               return x @ self.lm.token_embed.weight.T + self.lm.head_bias

        Step 4's clone-then-assign keeps autograd happy (in-place
        assignment into a fresh clone is differentiable) and keeps
        gradient flowing BOTH into `patch_embed.proj` (through the
        assigned rows) and into the token embedding table (through the
        untouched text rows).
        """
        # TODO
        raise NotImplementedError


def build_caption_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    patch_size: int,
    *,
    pad_to: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assemble a caption-training batch: `(x, images, y, loss_mask)`.

    Provided plumbing. Builds, per example, the id sequence

        [<img>] + [<image_patch>] * num_patches + [</img>]
        + caption_ids(label)

    then applies Module 13's shift-and-mask: `x = ids[:, :-1]`,
    `y = ids[:, 1:]`, with `loss_mask` zero on every position whose
    target is an image boundary, placeholder, or padding and one on
    caption targets.
    Feed the result to `masked_cross_entropy(model(x, images), y, mask)`.

    Args:
        images: `(B, H, W)` uint8 or float MNIST digits. uint8 input
            is scaled to [0, 1] here — the one and only normalization
            site, per the lesson's "normalize once" pitfall.
        labels: `(B,)` integer digit labels.
        patch_size: patch side length (determines placeholder count).
        pad_to: optional fixed length for the un-shifted sequences;
            defaults to the natural length (all captions are equal
            length, so padding only matters if you extend the format).

    Returns:
        x:         `(B, T-1)` LongTensor input ids (image structure kept).
        images:    `(B, H, W)` float tensor, normalized.
        y:         `(B, T-1)` LongTensor shifted targets.
        loss_mask: `(B, T-1)` LongTensor — 1 exactly where the target
                   is a caption token.
    """
    if images.dtype == torch.uint8:
        images = images.to(torch.float32) / 255.0
    else:
        images = images.to(torch.float32)
    B, H, W = images.shape
    num_patches = (H // patch_size) * (W // patch_size)

    rows: list[list[int]] = []
    masks: list[list[int]] = []
    for label in labels.tolist():
        caption = caption_ids(int(label))
        image_span = (
            [IMAGE_START_ID]
            + [IMAGE_PATCH_ID] * num_patches
            + [IMAGE_END_ID]
        )
        ids = image_span + caption
        mask = [0] * len(image_span) + [1] * len(caption)
        rows.append(ids)
        masks.append(mask)

    length = pad_to if pad_to is not None else max(len(r) for r in rows)
    for ids, mask in zip(rows, masks):
        if len(ids) > length:
            raise ValueError(
                f"sequence length {len(ids)} exceeds pad_to {length}"
            )
        while len(ids) < length:
            ids.append(PAD_ID)
            mask.append(0)

    ids_b = torch.tensor(rows, dtype=torch.long)
    mask_b = torch.tensor(masks, dtype=torch.long)
    x = ids_b[:, :-1]
    y = ids_b[:, 1:]
    loss_mask = mask_b[:, 1:]
    return x, images, y, loss_mask
