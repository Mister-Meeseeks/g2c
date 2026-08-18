"""Tests for g2c/multimodal — Beyond module: multimodal language models.

Suggested order to implement & turn green:

1. `patchify` — unblocks `test_patchify_*`. The roundtrip test is the
   anchor: patchify is pure indexing, and if the inverse reconstructs
   the image exactly, the fiddliest arithmetic is verified before any
   model exists.
2. `PatchEmbedding.forward` — unblocks `test_patch_embedding_*`.
3. `MultimodalLM.forward` — unblocks the splice tests.
   `test_splice_preserves_text_positions` pins the seam where real
   bugs live. (Requires Modules 05/09 forwards — Beyond modules assume
   the numbered prerequisites are done.)

The caption vocab and `build_caption_batch` are provided plumbing and
pass from the start.
"""
from __future__ import annotations

import pytest
import torch

from g2c.multimodal import (
    IMG_ID,
    MultimodalLM,
    PatchEmbedding,
    build_caption_batch,
    caption_ids,
    decode_ids,
    patchify,
)
from g2c.multimodal.vocab import VOCAB_SIZE
from g2c.transformer import TransformerLM

D = 16


# ---------------------------------------------------------------------------
# Provided plumbing — green from the start.
# ---------------------------------------------------------------------------


def test_caption_vocab_roundtrip():
    ids = caption_ids(7)
    assert len(ids) == 6
    assert decode_ids(ids) == "This is a 7 . <end>"
    with pytest.raises(ValueError):
        caption_ids(10)


def test_build_caption_batch_shapes_and_mask():
    images = torch.randint(0, 256, (3, 28, 28), dtype=torch.uint8)
    labels = torch.tensor([0, 7, 9])
    x, imgs, y, mask = build_caption_batch(images, labels, patch_size=7)
    T = 16 + 6  # patches + caption
    assert x.shape == (3, T - 1)
    assert y.shape == (3, T - 1)
    assert mask.shape == (3, T - 1)
    # loss fires exactly on the six caption targets per row
    assert (mask.sum(dim=-1) == 6).all()
    # every placeholder survives the shift into x
    assert ((x == IMG_ID).sum(dim=-1) == 16).all()
    # normalization happened exactly once
    assert imgs.dtype == torch.float32
    assert imgs.max() <= 1.0


# ---------------------------------------------------------------------------
# patchify
# ---------------------------------------------------------------------------


def test_patchify_shape():
    images = torch.randn(2, 28, 28)
    patches = patchify(images, 7)
    assert patches.shape == (2, 16, 49)


def test_patchify_rejects_indivisible():
    with pytest.raises(ValueError):
        patchify(torch.randn(1, 28, 28), 5)


def test_patchify_values_tiny_grid():
    """4×4 image, 2×2 patches — hand-checkable row-major order."""
    image = torch.arange(16.0).reshape(1, 4, 4)
    patches = patchify(image, 2)
    assert patches.shape == (1, 4, 4)
    assert patches[0, 0].tolist() == [0.0, 1.0, 4.0, 5.0]  # top-left
    assert patches[0, 1].tolist() == [2.0, 3.0, 6.0, 7.0]  # top-right
    assert patches[0, 2].tolist() == [8.0, 9.0, 12.0, 13.0]
    assert patches[0, 3].tolist() == [10.0, 11.0, 14.0, 15.0]


def test_patchify_roundtrip():
    """Patchify is lossless: the inverse permute reconstructs exactly."""
    torch.manual_seed(0)
    images = torch.randn(3, 28, 28)
    p = 7
    patches = patchify(images, p)
    gh = gw = 28 // p
    rebuilt = (
        patches.reshape(3, gh, gw, p, p)
        .permute(0, 1, 3, 2, 4)
        .reshape(3, 28, 28)
    )
    assert torch.equal(rebuilt, images)


# ---------------------------------------------------------------------------
# PatchEmbedding
# ---------------------------------------------------------------------------


def test_patch_embedding_shape():
    torch.manual_seed(1)
    pe = PatchEmbedding(patch_size=7, embedding_dim=D)
    out = pe(torch.randn(2, 28, 28))
    assert out.shape == (2, 16, D)
    assert pe.num_patches(28, 28) == 16


def test_patch_embedding_gradient_reaches_projection():
    torch.manual_seed(2)
    pe = PatchEmbedding(patch_size=7, embedding_dim=D)
    pe(torch.randn(1, 28, 28)).sum().backward()
    assert pe.proj.W.grad is not None
    assert pe.proj.W.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# The splice
# ---------------------------------------------------------------------------


def _tiny_mm(num_layers: int = 0) -> MultimodalLM:
    lm = TransformerLM(
        vocab_size=VOCAB_SIZE,
        embedding_dim=D,
        num_layers=num_layers,
        num_heads=2,
        max_seq_len=64,
    )
    return MultimodalLM(lm, patch_size=7)


def test_splice_forward_shape():
    torch.manual_seed(3)
    mm = _tiny_mm(num_layers=1)
    images = torch.rand(2, 28, 28)
    labels = torch.tensor([3, 5])
    x, imgs, y, mask = build_caption_batch(images, labels, patch_size=7)
    logits = mm(x, imgs)
    assert logits.shape == (2, x.shape[1], VOCAB_SIZE)


def test_splice_preserves_text_positions():
    """With zero blocks there is no mixing: changing the IMAGE must not
    change TEXT-position logits, and must change placeholder logits."""
    torch.manual_seed(4)
    mm = _tiny_mm(num_layers=0)
    labels = torch.tensor([3, 5])
    x, imgs1, _, _ = build_caption_batch(torch.rand(2, 28, 28), labels, patch_size=7)
    _, imgs2, _, _ = build_caption_batch(torch.rand(2, 28, 28), labels, patch_size=7)
    logits1 = mm(x, imgs1)
    logits2 = mm(x, imgs2)
    placeholder = x == IMG_ID
    assert torch.allclose(logits1[~placeholder], logits2[~placeholder], atol=1e-5)
    assert not torch.allclose(logits1[placeholder], logits2[placeholder], atol=1e-3)


def test_splice_count_mismatch_raises():
    torch.manual_seed(5)
    mm = _tiny_mm()
    # Batch built for patch_size=7 (16 placeholders) but 14×14 images
    # supply only 4 patches at patch_size=7.
    x, _, _, _ = build_caption_batch(
        torch.rand(2, 28, 28), torch.tensor([1, 2]), patch_size=7
    )
    with pytest.raises(ValueError):
        mm(x, torch.rand(2, 14, 14))


def test_splice_per_row_mismatch_raises_when_batch_total_matches():
    """One row cannot borrow another row's placeholder capacity."""
    torch.manual_seed(6)
    mm = _tiny_mm()
    x, imgs, _, _ = build_caption_batch(
        torch.rand(2, 28, 28), torch.tensor([1, 2]), patch_size=7
    )
    # Keep the batch-wide count at 32 while changing the per-row counts
    # from [16, 16] to [15, 17]. A global-only validation misses this.
    x = x.clone()
    x[0, 0] = 3
    x[1, -1] = IMG_ID
    with pytest.raises(ValueError):
        mm(x, imgs)


def test_splice_two_images_one_sequence():
    torch.manual_seed(7)
    mm = _tiny_mm(num_layers=1)
    # Hand-built interleaved row: two images' placeholders + a caption.
    n_patches = 16
    ids = [IMG_ID] * n_patches + [IMG_ID] * n_patches + caption_ids(4)
    x = torch.tensor([ids], dtype=torch.long)
    images = torch.rand(1, 2, 28, 28)
    logits = mm(x, images)
    assert logits.shape == (1, len(ids), VOCAB_SIZE)


def test_splice_gradient_reaches_patch_projection():
    torch.manual_seed(8)
    mm = _tiny_mm(num_layers=1)
    x, imgs, y, mask = build_caption_batch(
        torch.rand(2, 28, 28), torch.tensor([3, 5]), patch_size=7
    )
    mm(x, imgs).sum().backward()
    assert mm.patch_embed.proj.W.grad is not None
    assert mm.patch_embed.proj.W.grad.abs().sum() > 0
