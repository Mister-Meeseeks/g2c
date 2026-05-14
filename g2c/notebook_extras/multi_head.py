"""Notebook-only helpers for Module 08 multi-head attention experiments.

The conceptual work in Module 08 is the reshape/split/concat logic inside
``g2c.attention.MultiHeadAttention``. This module keeps the notebook focused
on that by moving the small language-model comparison and heatmap plotting
boilerplate out of the notebook cells.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch

from g2c.artifacts import load_tokenizer_artifact, tokenizer_artifact_exists
from g2c.attention import MultiHeadAttention
from g2c.embeddings import LearnedPositionalEmbedding, TokenEmbedding
from g2c.nn import CrossEntropyLoss, Linear, Module, resolve_device
from g2c.training import AdamW

__all__ = [
    "MultiHeadComparison",
    "MultiHeadNotebookData",
    "TinyMultiHeadLM",
    "estimate_perplexity",
    "get_sequence_batch",
    "load_mha_text",
    "plot_multi_head_attention_deviation",
    "plot_multi_head_attention_heads",
    "plot_multi_head_comparison",
    "prepare_multi_head_data",
    "sequence_loss",
    "train_mha_lm",
    "train_multi_head_comparison",
]

SHAKESPEARE_TOKENIZER = "ShakespeareTokenizer"
SHAKESPEARE_VOCAB_SIZE = 1024
SHAKESPEARE_TRAIN_CHARS = 1_000_000


@dataclass
class MultiHeadNotebookData:
    text: str
    ids: torch.Tensor
    train_ids: torch.Tensor
    val_ids: torch.Tensor
    vocab_size: int
    encode_text: Callable[[str], list[int]]
    token_label: Callable[[int], str]
    tokenizer_mode: str


@dataclass
class MultiHeadComparison:
    data: MultiHeadNotebookData
    models: dict[int, TinyMultiHeadLM]
    histories: dict[int, dict[str, list[float]]]


class TinyMultiHeadLM(Module):
    """Tiny causal LM used only for the Module 08 head-count comparison."""

    def __init__(self, vocab_size: int, seq_len: int, embedding_dim: int, num_heads: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.embed = TokenEmbedding(vocab_size, embedding_dim)
        self.pos = LearnedPositionalEmbedding(seq_len, embedding_dim)
        self.attn = MultiHeadAttention(embedding_dim, num_heads, causal=True)
        self.out = Linear(embedding_dim, vocab_size)

    def parameters(self):
        return [
            *self.embed.parameters(),
            *self.pos.parameters(),
            *self.attn.parameters(),
            *self.out.parameters(),
        ]

    def hidden_states(self, ids: torch.Tensor) -> torch.Tensor:
        _, seq_len = ids.shape
        return self.embed(ids) + self.pos(seq_len).unsqueeze(0)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.hidden_states(ids)
        x = self.attn(x)
        return self.out(x)


def load_mha_text(repo_root: Path, max_chars: int = SHAKESPEARE_TRAIN_CHARS) -> str:
    """Load a TinyShakespeare slice for the Module 08 comparison."""
    path = repo_root / "data" / "tinyshakespeare.txt"
    if path.exists():
        return path.read_text(encoding="utf-8")[:max_chars]
    base = """
    the model predicts the next token from context
    attention lets tokens choose which earlier tokens matter
    multiple heads can look for different local patterns
    the same projection size can be split into different head counts
    """
    return ("\n".join(line.strip() for line in base.strip().splitlines()) + "\n") * 200


def prepare_multi_head_data(
    *,
    repo_root: Path,
    max_chars: int = SHAKESPEARE_TRAIN_CHARS,
    vocab_size: int = SHAKESPEARE_VOCAB_SIZE,
    tokenizer_name: str = SHAKESPEARE_TOKENIZER,
    train_fraction: float = 0.9,
) -> MultiHeadNotebookData:
    """Prepare the TinyShakespeare split used by Module 08.

    The preferred path reuses the saved Shakespeare tokenizer artifact at a
    capped effective vocabulary. If that artifact is not available, the
    notebook still runs with a simple character tokenizer fallback.
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    text = load_mha_text(repo_root, max_chars=max_chars)
    if tokenizer_artifact_exists(tokenizer_name, repo_root=repo_root):
        artifact = load_tokenizer_artifact(tokenizer_name, repo_root=repo_root)
        tokenizer = artifact.tokenizer
        capped_vocab_size = min(vocab_size, len(tokenizer.vocab))
        actual_vocab_size = tokenizer.effective_vocab_size(capped_vocab_size)
        token_ids = tokenizer.encode_with_vocab_size(text, capped_vocab_size)

        def encode_text(prompt: str) -> list[int]:
            return tokenizer.encode_with_vocab_size(prompt, capped_vocab_size)

        def token_label(token_id: int) -> str:
            return _display_token_piece(tokenizer.decode([int(token_id)]))

        tokenizer_mode = f"{tokenizer_name} at vocab {actual_vocab_size}"
    else:
        chars = sorted(set(text))
        char_to_id = {char: index for index, char in enumerate(chars)}
        id_to_char = {index: char for char, index in char_to_id.items()}
        fallback_id = char_to_id.get(" ", 0)
        actual_vocab_size = len(chars)
        token_ids = [char_to_id[char] for char in text]

        def encode_text(prompt: str) -> list[int]:
            return [char_to_id.get(char, fallback_id) for char in prompt]

        def token_label(token_id: int) -> str:
            return _display_token_piece(id_to_char[int(token_id)])

        tokenizer_mode = "character tokenizer fallback"

    ids = torch.tensor(token_ids, dtype=torch.long)
    split = int(train_fraction * len(ids))
    data = MultiHeadNotebookData(
        text=text,
        ids=ids,
        train_ids=ids[:split],
        val_ids=ids[split:],
        vocab_size=actual_vocab_size,
        encode_text=encode_text,
        token_label=token_label,
        tokenizer_mode=tokenizer_mode,
    )
    print("source chars:", len(data.text))
    print("tokenizer:", data.tokenizer_mode)
    print("tokens:", len(data.ids))
    print("vocab size:", data.vocab_size)
    print("train tokens:", len(data.train_ids))
    print("val tokens:", len(data.val_ids))
    return data


def get_sequence_batch(
    ids: torch.Tensor,
    seq_len: int,
    batch_size: int,
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(0, len(ids) - seq_len - 1, (batch_size,), generator=generator)
    x = torch.stack([ids[start : start + seq_len] for start in starts])
    y = torch.stack([ids[start + 1 : start + seq_len + 1] for start in starts])
    return x, y


def sequence_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return CrossEntropyLoss()(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))


def estimate_perplexity(
    model: TinyMultiHeadLM,
    ids: torch.Tensor,
    *,
    batches: int = 8,
    batch_size: int = 64,
    seed: int = 123,
    device: str | torch.device | None = None,
) -> float:
    if device is None:
        resolved_device = getattr(model, "device", torch.device("cpu"))
    else:
        resolved_device = resolve_device(device)
    generator = torch.Generator().manual_seed(seed)
    losses = []
    with torch.no_grad():
        for _ in range(batches):
            x, y = get_sequence_batch(ids, model.seq_len, batch_size, generator=generator)
            x = x.to(resolved_device)
            y = y.to(resolved_device)
            losses.append(float(sequence_loss(model(x), y).item()))
    return math.exp(sum(losses) / len(losses))


def train_mha_lm(
    data: MultiHeadNotebookData,
    num_heads: int,
    *,
    num_steps: int = 1000,
    log_every: int = 100,
    seed: int = 0,
    device: str | torch.device = "auto",
    seq_len: int = 24,
    embedding_dim: int = 64,
    batch_size: int = 128,
    lr: float = 3e-3,
    weight_decay: float = 1e-4,
) -> tuple[TinyMultiHeadLM, dict[str, list[float]]]:
    """Train one tiny multi-head LM for a notebook comparison."""
    resolved_device = resolve_device(device)
    model = TinyMultiHeadLM(
        vocab_size=data.vocab_size,
        seq_len=seq_len,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
    ).to(resolved_device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed)
    val_perplexities = []
    train_losses = []
    steps = []

    for step in range(num_steps):
        x, y = get_sequence_batch(data.train_ids, model.seq_len, batch_size, generator=generator)
        x = x.to(resolved_device)
        y = y.to(resolved_device)
        loss = sequence_loss(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % log_every == 0 or step == num_steps - 1:
            steps.append(step)
            train_losses.append(float(loss.item()))
            val_perplexities.append(
                estimate_perplexity(
                    model,
                    data.val_ids,
                    seed=seed + step + 1,
                    device=resolved_device,
                )
            )

    return model, {
        "steps": steps,
        "train_losses": train_losses,
        "val_perplexities": val_perplexities,
    }


def train_multi_head_comparison(
    data: MultiHeadNotebookData,
    *,
    head_counts: Sequence[int] = (1, 4, 8),
    seed: int = 10,
    device: str | torch.device = "auto",
    **train_kwargs: Any,
) -> MultiHeadComparison:
    """Train matched tiny LMs at different head counts."""
    models: dict[int, TinyMultiHeadLM] = {}
    histories: dict[int, dict[str, list[float]]] = {}
    for i, num_heads in enumerate(head_counts):
        print(f"training H={num_heads}")
        model, history = train_mha_lm(
            data,
            int(num_heads),
            seed=seed + i,
            device=device,
            **train_kwargs,
        )
        models[int(num_heads)] = model
        histories[int(num_heads)] = history
        print("  final val perplexity:", history["val_perplexities"][-1])
    return MultiHeadComparison(data=data, models=models, histories=histories)


def plot_multi_head_comparison(comparison: MultiHeadComparison) -> None:
    """Plot validation perplexity curves for a head-count comparison."""
    plt.figure(figsize=(7, 4))
    for num_heads, history in comparison.histories.items():
        plt.plot(
            history["steps"],
            history["val_perplexities"],
            marker="o",
            label=f"H={num_heads}",
        )
    plt.yscale("log")
    plt.title("Validation perplexity at fixed D=64")
    plt.xlabel("step")
    plt.ylabel("perplexity")
    plt.legend()
    plt.show()


def plot_multi_head_attention_heads(
    model: TinyMultiHeadLM,
    data: MultiHeadNotebookData,
    *,
    prompt: str = "First Citizen: ",
    max_tokens: int | None = None,
    cols: int = 2,
) -> torch.Tensor:
    """Plot one raw attention heatmap per head and return ``(H, T, T)`` weights."""
    weights, labels = _attention_weights_and_labels(model, data, prompt, max_tokens=max_tokens)
    rows = math.ceil(model.num_heads / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(10, 4 * rows), constrained_layout=True)
    axes = _flat_axes(axes)

    for head in range(model.num_heads):
        ax = axes[head]
        image = ax.imshow(
            weights[head],
            vmin=0.0,
            vmax=float(weights[head].max()),
            cmap="magma",
        )
        ax.set_title(f"head {head}")
        _set_token_ticks(ax, labels)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    for ax in axes[model.num_heads :]:
        ax.axis("off")
    plt.show()
    return weights


def plot_multi_head_attention_deviation(
    model: TinyMultiHeadLM,
    data: MultiHeadNotebookData,
    *,
    prompt: str = "First Citizen: ",
    max_tokens: int | None = None,
    cols: int = 2,
) -> torch.Tensor:
    """Plot per-head attention after subtracting causal-uniform attention."""
    weights, labels = _attention_weights_and_labels(model, data, prompt, max_tokens=max_tokens)
    baseline = _causal_uniform_baseline(
        weights.shape[-1],
        device=weights.device,
        dtype=weights.dtype,
    )
    centered = weights - baseline
    allowed_positions = torch.tril(torch.ones_like(baseline, dtype=torch.bool))
    allowed_positions[0, 0] = False

    print("deviation from causal-uniform baseline")
    print("-" * 48)
    for head in range(model.num_heads):
        diffs = centered[head][allowed_positions].abs()
        print(
            f"head {head}: mean_abs_diff={diffs.mean().item():.4f}  "
            f"max_abs_diff={diffs.max().item():.4f}"
        )

    max_abs = max(float(centered.abs().max().item()), 1e-6)
    rows = math.ceil(model.num_heads / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(10, 4 * rows), constrained_layout=True)
    axes = _flat_axes(axes)
    for head in range(model.num_heads):
        ax = axes[head]
        image = ax.imshow(
            centered[head],
            vmin=-max_abs,
            vmax=max_abs,
            cmap="coolwarm",
        )
        ax.set_title(f"head {head} minus causal-uniform")
        _set_token_ticks(ax, labels)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    for ax in axes[model.num_heads :]:
        ax.axis("off")
    plt.show()
    return centered


def _attention_weights_and_labels(
    model: TinyMultiHeadLM,
    data: MultiHeadNotebookData,
    prompt: str,
    *,
    max_tokens: int | None,
) -> tuple[torch.Tensor, list[str]]:
    ids = torch.tensor(data.encode_text(prompt), dtype=torch.long)
    token_limit = model.seq_len if max_tokens is None else min(model.seq_len, max_tokens)
    ids = ids[:token_limit]
    if ids.numel() == 0:
        raise ValueError("prompt encoded to no tokens")
    if int(ids.max()) >= model.vocab_size:
        raise ValueError(
            "prompt encoded to a token outside the model vocabulary; "
            f"max token id {int(ids.max())}, model vocab {model.vocab_size}"
        )
    batch = ids.unsqueeze(0).to(model.device)
    labels = [f"{i}:{data.token_label(token_id)}" for i, token_id in enumerate(ids.tolist())]
    x = model.hidden_states(batch)
    weights = model.attn.attention_weights(x)[0].detach().cpu()
    return weights, labels


def _display_token_piece(piece: str) -> str:
    piece = piece.replace("\n", "\\n").replace("\t", "\\t")
    if piece == " ":
        return "space"
    return piece.replace(" ", "·")


def _causal_uniform_baseline(
    seq_len: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    allowed = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=dtype))
    return allowed / allowed.sum(dim=-1, keepdim=True)


def _set_token_ticks(ax: Any, labels: Sequence[str]) -> None:
    ax.set_xlabel("key position")
    ax.set_ylabel("query position")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)


def _flat_axes(axes: Any) -> list[Any]:
    if hasattr(axes, "reshape"):
        return list(axes.reshape(-1))
    return [axes]
