"""Display helpers for the Module 05 embeddings notebook.

These helpers keep PCA projection, token-label cleanup, and plotting glue out
of the notebook. The pedagogical implementations live in ``g2c.embeddings``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import matplotlib.pyplot as plt
import torch

from g2c.embeddings import SkipGramEmbeddingModel, normalized
from g2c.tokenizer import BPETokenizer

__all__ = [
    "decode_token_label",
    "frequent_learned_token_ids",
    "is_readable_learned_token",
    "learned_token_vectors",
    "plot_glove_slice",
    "plot_learned_token_embeddings_2d",
    "project_glove_query_2d",
    "project_glove_words_2d",
    "project_rows_2d",
    "token_key",
]


def project_rows_2d(weight: torch.Tensor) -> torch.Tensor:
    """Project embedding rows to 2D with a small PCA using torch.linalg.svd."""
    rows = weight.detach()
    rows = rows - rows.mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(rows, full_matrices=False)
    return rows @ vh[:2].T


def decode_token_label(
    tokenizer: BPETokenizer,
    token_id: int,
    *,
    max_chars: int = 18,
) -> str:
    text = tokenizer.vocab[token_id].decode("utf-8", errors="replace")
    text = text.replace("\n", "\\n").replace("\t", "\\t")
    if len(text) > max_chars:
        text = text[: max_chars - 1] + "..."
    return text or "<empty>"


def token_key(tokenizer: BPETokenizer, token_id: int) -> str:
    return f"{token_id}:{decode_token_label(tokenizer, token_id)!r}"


def is_readable_learned_token(
    tokenizer: BPETokenizer,
    token_id: int,
    *,
    min_chars: int = 3,
) -> bool:
    if token_id < tokenizer.base_vocab_size:
        return False
    text = tokenizer.vocab[token_id].decode("utf-8", errors="replace")
    return len(text.strip()) >= min_chars and any(ch.isalpha() for ch in text)


def frequent_learned_token_ids(
    ids: Sequence[int] | torch.Tensor,
    tokenizer: BPETokenizer,
    *,
    top_n: int = 60,
    min_chars: int = 3,
) -> list[int]:
    values = [int(token_id) for token_id in ids.tolist()] if isinstance(ids, torch.Tensor) else ids
    counts = Counter(values)
    candidates = [
        token_id
        for token_id in range(tokenizer.base_vocab_size, len(tokenizer.vocab))
        if counts[token_id] > 0
        and is_readable_learned_token(tokenizer, token_id, min_chars=min_chars)
    ]
    candidates.sort(key=lambda token_id: counts[token_id], reverse=True)
    return candidates[:top_n]


def learned_token_vectors(
    model: SkipGramEmbeddingModel,
    tokenizer: BPETokenizer,
    ids: Sequence[int] | torch.Tensor,
) -> dict[str, torch.Tensor]:
    token_ids = frequent_learned_token_ids(ids, tokenizer, top_n=120, min_chars=3)
    return {
        token_key(tokenizer, token_id): model.embedding.weight[token_id].detach()
        for token_id in token_ids
    }


def plot_learned_token_embeddings_2d(
    model: SkipGramEmbeddingModel,
    tokenizer: BPETokenizer,
    ids: Sequence[int] | torch.Tensor,
    *,
    top_n: int = 70,
    min_chars: int = 3,
) -> None:
    """Plot frequent learned-token embeddings after a 2D PCA projection."""
    coords = project_rows_2d(model.embedding.weight)
    plot_token_ids = frequent_learned_token_ids(
        ids,
        tokenizer,
        top_n=top_n,
        min_chars=min_chars,
    )

    plt.figure(figsize=(10, 7))
    plt.axhline(0, color="0.9", linewidth=1)
    plt.axvline(0, color="0.9", linewidth=1)
    plt.scatter(coords[plot_token_ids, 0], coords[plot_token_ids, 1], s=24)
    for token_id in plot_token_ids:
        plt.text(
            coords[token_id, 0].item(),
            coords[token_id, 1].item(),
            decode_token_label(tokenizer, token_id, max_chars=12),
            fontsize=8,
        )
    plt.title("TinyShakespeare learned token embeddings, projected to 2D")
    plt.xlabel("PCA dimension 1")
    plt.ylabel("PCA dimension 2")
    plt.tight_layout()
    plt.show()


def project_glove_words_2d(
    vectors: dict[str, torch.Tensor],
    words: list[str],
) -> tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project selected GloVe words to 2D with PCA on normalized vectors."""
    present_words = [word for word in words if word in vectors]
    rows = torch.stack([normalized(vectors[word]) for word in present_words])
    mean = rows.mean(dim=0)
    centered = rows - mean
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    basis = vh[:2]
    coords = centered @ basis.T
    return present_words, coords, mean, basis


def project_glove_query_2d(
    query: torch.Tensor,
    mean: torch.Tensor,
    basis: torch.Tensor,
) -> torch.Tensor:
    return (normalized(query) - mean) @ basis.T


def plot_glove_slice(vectors: dict[str, torch.Tensor]) -> None:
    """Plot a small semantic slice of pretrained GloVe vectors."""
    words = [
        "king",
        "queen",
        "prince",
        "princess",
        "man",
        "woman",
        "crown",
        "throne",
        "palace",
        "royal",
        "paris",
        "france",
        "rome",
        "italy",
        "madrid",
        "spain",
        "berlin",
        "germany",
        "cat",
        "dog",
        "kitten",
        "puppy",
        "animal",
        "pet",
    ]
    words, coords, mean, basis = project_glove_words_2d(vectors, words)
    coord_by_word = {word: coords[i] for i, word in enumerate(words)}

    plt.figure(figsize=(10, 7))
    plt.axhline(0, color="0.9", linewidth=1)
    plt.axvline(0, color="0.9", linewidth=1)
    plt.scatter(coords[:, 0], coords[:, 1], s=35)

    for word, coord in coord_by_word.items():
        plt.text(coord[0].item() + 0.01, coord[1].item() + 0.01, word, fontsize=9)

    query = vectors["king"] - vectors["man"] + vectors["woman"]
    query_coord = project_glove_query_2d(query, mean, basis)
    plt.scatter(query_coord[0], query_coord[1], marker="*", s=220, color="crimson")
    plt.text(
        query_coord[0].item() + 0.01,
        query_coord[1].item() + 0.01,
        "king - man + woman",
        color="crimson",
    )

    for start, end in [("man", "woman"), ("king", "queen")]:
        if start in coord_by_word and end in coord_by_word:
            start_coord = coord_by_word[start]
            end_coord = coord_by_word[end]
            delta = end_coord - start_coord
            plt.arrow(
                start_coord[0].item(),
                start_coord[1].item(),
                delta[0].item(),
                delta[1].item(),
                length_includes_head=True,
                head_width=0.025,
                alpha=0.55,
            )

    plt.title("Selected pretrained GloVe vectors projected to 2D")
    plt.xlabel("PCA dimension 1")
    plt.ylabel("PCA dimension 2")
    plt.show()
