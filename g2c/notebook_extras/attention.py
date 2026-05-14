"""Notebook-only helpers for Module 07 attention visualizations.

The student-facing attention implementation lives in ``g2c.attention``. This
module keeps the notebook readable by handling token-label display, optional
artifact lookup, and plotting glue around the experiments.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch

from g2c.artifacts import (
    load_model_artifact_with_tokenizer,
    load_tokenizer_artifact,
    model_artifact_exists,
    tokenizer_artifact_exists,
)
from g2c.attention import SelfAttention
from g2c.embeddings import LearnedPositionalEmbedding, TokenEmbedding
from g2c.nn import CrossEntropyLoss, Linear, Module, resolve_device
from g2c.training import AdamW

__all__ = [
    "TinyShakespeareAttentionProbe",
    "TinyShakespeareProbeData",
    "attention_entropy",
    "causal_uniform_matrix",
    "choose_shakespeare_artifact_name",
    "display_token_piece",
    "load_tinyshakespeare_probe_data",
    "plot_random_sentence_attention",
    "plot_trained_shakespeare_attention_heads",
    "run_tinyshakespeare_attention_probe",
    "train_attention_probe",
    "transformer_layer_attention_weights",
]


SHAKESPEARE_PROBE_TOKENIZER = "ShakespeareTokenizer"
SHAKESPEARE_PROBE_VOCAB = 2048
SHAKESPEARE_PROBE_MAX_TOKENS = 1_000_000


@dataclass
class TinyShakespeareProbeData:
    text: str
    ids: torch.Tensor
    vocab_size: int
    encode_text: Callable[[str], list[int]]
    axis_labels: Callable[[list[int]], list[str]]
    mode: str


class TinyShakespeareAttentionProbe(Module):
    """Small single-head causal LM used only for the Module 07 visualization."""

    def __init__(self, vocab_size: int, seq_len: int, embedding_dim: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.embed = TokenEmbedding(vocab_size, embedding_dim)
        self.pos = LearnedPositionalEmbedding(seq_len, embedding_dim)
        self.attn = SelfAttention(embedding_dim, causal=True)
        self.out = Linear(embedding_dim, vocab_size)

    def parameters(self):
        return [
            *self.embed.parameters(),
            *self.pos.parameters(),
            *self.attn.parameters(),
            *self.out.parameters(),
        ]

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        seq_len = ids.shape[1]
        x = self.embed(ids) + self.pos(seq_len).unsqueeze(0)
        x = self.attn(x)
        return self.out(x)

    def attention_weights_for_ids(self, ids: torch.Tensor) -> torch.Tensor:
        seq_len = ids.shape[1]
        x = self.embed(ids) + self.pos(seq_len).unsqueeze(0)
        return self.attn.attention_weights(x)


def display_token_piece(piece: str) -> str:
    """Make a decoded token readable as an axis label."""
    piece = piece.replace("\n", "\\n").replace("\t", "\\t")
    if piece == " ":
        return "space"
    return piece.replace(" ", "·")


def plot_random_sentence_attention(
    sentences: Sequence[str],
    *,
    repo_root: Path,
    tokenizer_name: str = "G2CTokenizer",
    vocab_size: int = 2048,
    embedding_dim: int = 16,
    causal: bool = True,
) -> None:
    """Plot raw and baseline-centered random attention for short sentences."""
    if not sentences:
        raise ValueError("sentences must not be empty")

    encode, decode, actual_vocab_size, tokenizer_mode = _sentence_tokenizer(
        sentences,
        repo_root=repo_root,
        tokenizer_name=tokenizer_name,
        vocab_size=vocab_size,
    )

    token_embed = TokenEmbedding(vocab_size=actual_vocab_size, embedding_dim=embedding_dim)
    pos_embed = LearnedPositionalEmbedding(max_seq_len=128, embedding_dim=embedding_dim)
    attention = SelfAttention(embedding_dim=embedding_dim, causal=causal)
    print("tokenizer:", tokenizer_mode)
    print("causal mask:", causal)

    fig, axes = plt.subplots(
        2, len(sentences), figsize=(7 * len(sentences), 9), constrained_layout=True
    )
    if len(sentences) == 1:
        axes = axes.reshape(2, 1)

    for column, sentence in enumerate(sentences):
        ids = encode(sentence)
        ids_tensor = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
        x = token_embed(ids_tensor) + pos_embed(ids_tensor.shape[1]).unsqueeze(0)
        weights = attention.attention_weights(x)[0].detach()
        labels = _token_labels(ids, decode)

        if causal:
            baseline = causal_uniform_matrix(weights.shape[-1])
            baseline_label = "causal-uniform"
            raw_vmax = 1.0
        else:
            baseline = torch.full_like(weights, 1.0 / weights.shape[-1])
            baseline_label = "uniform"
            raw_vmax = 0.25
        deviation = weights - baseline
        dev_limit = max(float(deviation.abs().max()), 1e-6)

        print(
            f"{sentence!r}: T={weights.shape[-1]}, "
            f"max={float(weights.max()):.4f}, "
            f"max above {baseline_label}={float(deviation.max()):.4f}"
        )

        prob_ax = axes[0, column]
        prob_image = prob_ax.imshow(weights, vmin=0.0, vmax=raw_vmax, cmap="viridis")
        prob_ax.set_title(sentence)
        prob_ax.set_ylabel("query position")
        fig.colorbar(prob_image, ax=prob_ax, fraction=0.046, pad=0.04)

        dev_ax = axes[1, column]
        dev_image = dev_ax.imshow(
            deviation,
            vmin=-dev_limit,
            vmax=dev_limit,
            cmap="coolwarm",
        )
        dev_ax.set_title(f"attention minus {baseline_label} baseline")
        dev_ax.set_xlabel("key position")
        dev_ax.set_ylabel("query position")
        fig.colorbar(dev_image, ax=dev_ax, fraction=0.046, pad=0.04)

        for ax in (prob_ax, dev_ax):
            _set_token_ticks(ax, labels, fontsize=7)
    plt.show()


def load_tinyshakespeare_probe_data(
    *,
    repo_root: Path,
    max_tokens: int = SHAKESPEARE_PROBE_MAX_TOKENS,
) -> TinyShakespeareProbeData | None:
    """Load TinyShakespeare IDs for the optional attention-probe visualization."""
    path = repo_root / "data" / "datasets" / "tinyshakespeare.txt"
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")

    if tokenizer_artifact_exists(SHAKESPEARE_PROBE_TOKENIZER, repo_root=repo_root):
        tokenizer = load_tokenizer_artifact(
            SHAKESPEARE_PROBE_TOKENIZER, repo_root=repo_root
        ).tokenizer
        actual_vocab_size = tokenizer.effective_vocab_size(SHAKESPEARE_PROBE_VOCAB)
        token_ids = tokenizer.encode_with_vocab_size(text, SHAKESPEARE_PROBE_VOCAB)[:max_tokens]
        ids = torch.tensor(token_ids, dtype=torch.long)

        def encode_text(prompt: str) -> list[int]:
            return tokenizer.encode_with_vocab_size(prompt, SHAKESPEARE_PROBE_VOCAB)

        def axis_labels(token_ids: list[int]) -> list[str]:
            return [
                f"{i}:{display_token_piece(tokenizer.decode([int(token_id)]))}"
                for i, token_id in enumerate(token_ids)
            ]

        mode = f"{SHAKESPEARE_PROBE_TOKENIZER} at vocab {actual_vocab_size}"
        return TinyShakespeareProbeData(
            text, ids, actual_vocab_size, encode_text, axis_labels, mode
        )

    text = text[:max_tokens]
    chars = sorted(set(text))
    char_to_id = {ch: i for i, ch in enumerate(chars)}
    id_to_char = {i: ch for ch, i in char_to_id.items()}
    ids = torch.tensor([char_to_id[ch] for ch in text], dtype=torch.long)
    fallback_id = char_to_id.get(" ", 0)

    def encode_text(prompt: str) -> list[int]:
        return [char_to_id.get(ch, fallback_id) for ch in prompt]

    def axis_labels(token_ids: list[int]) -> list[str]:
        return [
            f"{i}:{display_token_piece(id_to_char[int(token_id)])}"
            for i, token_id in enumerate(token_ids)
        ]

    return TinyShakespeareProbeData(
        text,
        ids,
        len(chars),
        encode_text,
        axis_labels,
        "character tokenizer fallback",
    )


def train_attention_probe(
    model: TinyShakespeareAttentionProbe,
    ids: torch.Tensor,
    *,
    steps: int = 600,
    batch_size: int = 128,
    lr: float = 3e-3,
    log_every: int = 100,
    seed: int = 0,
    device: str | torch.device = "auto",
) -> tuple[list[int], list[float]]:
    """Train the optional tiny attention probe for a few notebook snapshots."""
    device = resolve_device(device)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed)
    logged_steps: list[int] = []
    losses: list[float] = []

    for step in range(steps):
        x, y = _get_token_batch(ids, model.seq_len, batch_size, generator=generator)
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = loss_fn(logits.reshape(-1, model.vocab_size), y.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % log_every == 0 or step == steps - 1:
            logged_steps.append(step)
            losses.append(float(loss.detach().cpu()))
    return logged_steps, losses


def run_tinyshakespeare_attention_probe(
    *,
    repo_root: Path,
    device: str | torch.device = "auto",
    probe_seq_len: int = 25,
    steps: int = 600,
    batch_size: int = 128,
    lr: float = 3e-3,
    log_every: int = 100,
    seed: int = 7,
) -> TinyShakespeareAttentionProbe | None:
    """Train and plot the optional TinyShakespeare positive-control probe."""
    probe_data = load_tinyshakespeare_probe_data(repo_root=repo_root)
    if probe_data is None:
        print("Skipping TinyShakespeare attention probe: run ./setup.sh first.")
        return None

    probe_prompt_text = probe_data.text[:800]
    probe_prompt_token_ids = probe_data.encode_text(probe_prompt_text)
    if len(probe_prompt_token_ids) < probe_seq_len:
        probe_prompt_token_ids = probe_data.ids[:probe_seq_len].tolist()
    else:
        probe_prompt_token_ids = probe_prompt_token_ids[:probe_seq_len]
    probe_prompt_ids = torch.tensor([probe_prompt_token_ids], dtype=torch.long)

    _ = torch.manual_seed(seed)
    model = TinyShakespeareAttentionProbe(
        vocab_size=probe_data.vocab_size,
        seq_len=probe_seq_len,
        embedding_dim=64,
    )
    resolved_device = resolve_device(device)
    model.to(resolved_device)
    before_weights = (
        model.attention_weights_for_ids(probe_prompt_ids.to(resolved_device))[0].detach().cpu()
    )

    start = time.perf_counter()
    probe_steps, probe_losses = train_attention_probe(
        model,
        probe_data.ids,
        steps=steps,
        batch_size=batch_size,
        lr=lr,
        log_every=log_every,
        seed=0,
        device=resolved_device,
    )
    elapsed = time.perf_counter() - start
    after_weights = (
        model.attention_weights_for_ids(probe_prompt_ids.to(resolved_device))[0].detach().cpu()
    )

    print("tokenizer:", probe_data.mode)
    print("training tokens:", len(probe_data.ids))
    print(f"trained for {elapsed:.1f}s on {resolved_device}")
    print("loss:", [round(x, 3) for x in probe_losses])
    print(
        "mean max attention before/after:",
        round(float(before_weights.max(dim=-1).values.mean()), 3),
        round(float(after_weights.max(dim=-1).values.mean()), 3),
    )
    print(
        "mean attention entropy before/after:",
        round(attention_entropy(before_weights), 3),
        round(attention_entropy(after_weights), 3),
    )

    _plot_probe_loss(probe_steps, probe_losses)
    labels = probe_data.axis_labels(probe_prompt_token_ids)
    _plot_probe_attention(before_weights, after_weights, labels)
    return model


def attention_entropy(weights: torch.Tensor) -> float:
    """Mean row entropy of an attention matrix."""
    return float((-(weights * (weights + 1e-9).log()).sum(dim=-1)).mean())


def causal_uniform_matrix(seq_len: int) -> torch.Tensor:
    """Return the row-wise uniform distribution over each causal prefix."""
    baseline = torch.zeros(seq_len, seq_len)
    for row in range(seq_len):
        baseline[row, : row + 1] = 1.0 / (row + 1)
    return baseline


def choose_shakespeare_artifact_name(*, repo_root: Path) -> str | None:
    """Return the strongest ShakespeareLM artifact available locally."""
    for name in ("ShakespeareLM-1M", "ShakespeareLM"):
        if model_artifact_exists(name, repo_root=repo_root):
            return name
    return None


def transformer_layer_attention_weights(
    model: Any,
    token_ids: torch.Tensor,
    *,
    layer_index: int = 0,
) -> torch.Tensor:
    """Return ``(heads, T, T)`` attention weights from one TransformerLM layer."""
    _, seq_len = token_ids.shape
    x = model.token_embed(token_ids) + model.pos_embed(seq_len).unsqueeze(0)
    for index, block in enumerate(model.blocks):
        normalized = block.ln1(x)
        if index == layer_index:
            return block.attn.attention_weights(normalized)[0].detach()
        x = x + block.attn(normalized)
        x = x + block.ffn(block.ln2(x))
    raise ValueError(f"layer_index {layer_index} is out of range")


def plot_trained_shakespeare_attention_heads(
    *,
    repo_root: Path,
    artifact_name: str | None = None,
    prompt: str = "First Citizen: ",
    layer_index: int = 0,
    max_tokens: int = 25,
) -> None:
    """Plot raw and centered head maps for a saved ShakespeareLM artifact."""
    artifact_name = artifact_name or choose_shakespeare_artifact_name(repo_root=repo_root)
    if artifact_name is None:
        print("Skipping trained attention pass: run Module 10 and save ShakespeareLM first.")
        return

    loaded = load_model_artifact_with_tokenizer(artifact_name, repo_root=repo_root)
    model = loaded.model
    tokenizer = loaded.tokenizer
    ids = tokenizer.encode_with_vocab_size(prompt, model.vocab_size)
    ids = ids[: min(len(ids), model.max_seq_len, max_tokens)]
    token_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    weights = transformer_layer_attention_weights(model, token_ids, layer_index=layer_index)
    labels = _trained_token_labels(tokenizer, ids)

    seq_len = weights.shape[-1]
    causal_uniform = causal_uniform_matrix(seq_len)
    heads_to_show = min(4, weights.shape[0])
    fig, axes = plt.subplots(
        2,
        heads_to_show,
        figsize=(4 * heads_to_show, 8),
        constrained_layout=True,
    )
    if heads_to_show == 1:
        axes = axes.reshape(2, 1)

    for head in range(heads_to_show):
        raw = weights[head]
        deviation = raw - causal_uniform
        dev_limit = max(float(deviation.abs().max()), 1e-6)

        raw_ax = axes[0, head]
        raw_image = raw_ax.imshow(raw, vmin=0.0, vmax=1.0, cmap="viridis")
        raw_ax.set_title(f"head {head}: raw")
        fig.colorbar(raw_image, ax=raw_ax, fraction=0.046, pad=0.04)

        dev_ax = axes[1, head]
        dev_image = dev_ax.imshow(
            deviation,
            vmin=-dev_limit,
            vmax=dev_limit,
            cmap="coolwarm",
        )
        dev_ax.set_title(f"head {head}: minus causal uniform")
        fig.colorbar(dev_image, ax=dev_ax, fraction=0.046, pad=0.04)

        for ax in (raw_ax, dev_ax):
            ax.set_xlabel("key position")
            ax.set_ylabel("query position")
            _set_token_ticks(ax, labels, fontsize=7)

        top = torch.topk(raw[-1], k=min(3, seq_len))
        top_items = [(labels[int(i)], round(float(v), 3)) for v, i in zip(top.values, top.indices)]
        print(f"head {head} top keys for final token:", top_items)

    plt.show()


def _sentence_tokenizer(
    sentences: Sequence[str],
    *,
    repo_root: Path,
    tokenizer_name: str,
    vocab_size: int,
) -> tuple[Callable[[str], list[int]], Callable[[int], str], int, str]:
    corpus = "\n".join(sentences)
    if tokenizer_artifact_exists(tokenizer_name, repo_root=repo_root):
        tokenizer = load_tokenizer_artifact(tokenizer_name, repo_root=repo_root).tokenizer
        actual_vocab_size = tokenizer.effective_vocab_size(vocab_size)

        def encode(text: str) -> list[int]:
            return tokenizer.encode_with_vocab_size(text, vocab_size)

        def decode(token_id: int) -> str:
            return tokenizer.decode([int(token_id)])

        return encode, decode, actual_vocab_size, f"{tokenizer_name} at vocab {actual_vocab_size}"

    chars = sorted(set(corpus))
    char_to_id = {char: index for index, char in enumerate(chars)}
    id_to_char = {index: char for char, index in char_to_id.items()}

    def encode(text: str) -> list[int]:
        return [char_to_id[char] for char in text]

    def decode(token_id: int) -> str:
        return id_to_char[int(token_id)]

    return encode, decode, len(chars), "character tokenizer fallback"


def _token_labels(ids: Sequence[int], decode: Callable[[int], str]) -> list[str]:
    return [f"{i}:{display_token_piece(decode(int(token_id)))}" for i, token_id in enumerate(ids)]


def _trained_token_labels(tokenizer: Any, ids: Sequence[int]) -> list[str]:
    labels = []
    for i, token_id in enumerate(ids):
        piece = display_token_piece(tokenizer.decode([int(token_id)]))
        if len(piece) > 12:
            piece = piece[:9] + "..."
        labels.append(f"{i}:{piece}")
    return labels


def _set_token_ticks(ax: Any, labels: Sequence[str], *, fontsize: int) -> None:
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=fontsize)
    ax.set_yticklabels(labels, fontsize=fontsize)


def _get_token_batch(
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


def _plot_probe_loss(steps: Sequence[int], losses: Sequence[float]) -> None:
    plt.figure(figsize=(6, 3))
    plt.plot(steps, losses, marker="o")
    plt.title("TinyShakespeare attention-probe loss")
    plt.xlabel("step")
    plt.ylabel("token cross-entropy")
    plt.show()


def _plot_probe_attention(
    before_weights: torch.Tensor,
    after_weights: torch.Tensor,
    labels: Sequence[str],
) -> None:
    baseline = causal_uniform_matrix(len(labels))
    panels = [
        ("before training", before_weights, "viridis", 0.0, 1.0),
        ("after training", after_weights, "viridis", 0.0, 1.0),
        ("before - causal uniform", before_weights - baseline, "coolwarm", None, None),
        ("after - causal uniform", after_weights - baseline, "coolwarm", None, None),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    for ax, (title, matrix, cmap, vmin, vmax) in zip(axes.flat, panels):
        if vmin is None or vmax is None:
            limit = max(float(matrix.abs().max()), 1e-6)
            vmin, vmax = -limit, limit
        image = ax.imshow(matrix, vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_title(title)
        ax.set_xlabel("key position")
        ax.set_ylabel("query position")
        _set_token_ticks(ax, labels, fontsize=6)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    plt.show()
