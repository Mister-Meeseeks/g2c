"""Non-pedagogical data and display helpers for the midtraining notebook."""
from __future__ import annotations

import gzip
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from IPython.display import Markdown, display

from g2c.artifacts.corpora import CorpusSource, resolve_corpus
from g2c.artifacts.paths import find_repo_root
from g2c.tokenizer.fast import FastBPEEncoder

__all__ = [
    "load_g2c_source_ids",
    "plot_domain_loss_deltas",
    "show_domain_loss_table",
]


def _read_shard_prefix(path: Path, byte_count: int) -> bytes:
    opener = gzip.open if path.suffix == ".gz" else Path.open
    if path.suffix == ".gz":
        with opener(path, "rb") as stream:
            return stream.read(byte_count)
    with opener(path, "rb") as stream:
        return stream.read(byte_count)


def _read_source_bytes(source: CorpusSource, byte_count: int) -> bytes:
    """Read up to `byte_count` uncompressed bytes, wrapping if necessary."""
    parts: list[bytes] = []
    remaining = byte_count
    shard_index = 0
    while remaining > 0:
        shard = source.shards[shard_index % len(source.shards)]
        chunk = _read_shard_prefix(shard.path, min(remaining, shard.uncompressed_bytes))
        if not chunk:
            break
        parts.append(chunk)
        remaining -= len(chunk)
        shard_index += 1
    return b"".join(parts)


def load_g2c_source_ids(
    corpus: str,
    source_names: tuple[str, ...],
    *,
    split: str,
    byte_count: int,
    tokenizer,
    vocab_size: int | None = None,
    repo_root: str | Path | None = None,
) -> torch.Tensor:
    """Load and tokenize a proportional slice of selected G2C subcorpora.

    This is notebook plumbing rather than a student deliverable. Allocation
    among selected sources follows their relative byte sizes in the manifest.
    """
    if not source_names:
        raise ValueError("source_names must be non-empty")
    if byte_count <= 0:
        raise ValueError("byte_count must be positive")
    root = find_repo_root(repo_root)
    spec = resolve_corpus(corpus, split=split, repo_root=root)
    if spec is None:
        raise FileNotFoundError(f"corpus {corpus!r} has no local {split!r} split")
    by_name = {source.name: source for source in spec.sources}
    missing = [name for name in source_names if name not in by_name]
    if missing:
        raise ValueError(f"sources not present in {corpus}/{split}: {missing}")

    selected = [by_name[name] for name in source_names]
    available = sum(source.uncompressed_bytes for source in selected)
    encoder = FastBPEEncoder(tokenizer, vocab_size=vocab_size)
    ids: list[int] = []
    assigned = 0
    eot_id = getattr(tokenizer, "special_to_id", {}).get("<|endoftext|>")
    for index, source in enumerate(selected):
        allocation = (
            byte_count - assigned
            if index == len(selected) - 1
            else round(byte_count * source.uncompressed_bytes / available)
        )
        assigned += allocation
        text = _read_source_bytes(source, allocation).decode("utf-8", errors="replace")
        ids.extend(encoder.encode(text))
        if eot_id is not None:
            ids.append(eot_id)
    return torch.tensor(ids, dtype=torch.long)


def show_domain_loss_table(results: dict[str, dict[str, float]]) -> None:
    """Render checkpoint × domain losses and deltas from the base row."""
    if not results:
        return
    checkpoint_names = list(results)
    domains = list(next(iter(results.values())))
    base = results[checkpoint_names[0]]
    header = "| checkpoint | " + " | ".join(domains) + " |\n"
    rule = "|---|" + "|".join("---:" for _ in domains) + "|\n"
    rows = []
    for checkpoint in checkpoint_names:
        cells = []
        for domain in domains:
            loss = results[checkpoint][domain]
            delta = loss - base[domain]
            suffix = "" if checkpoint == checkpoint_names[0] else f" ({delta:+.3f})"
            cells.append(f"{loss:.3f}{suffix}")
        rows.append(f"| {checkpoint} | " + " | ".join(cells) + " |")
    display(Markdown(header + rule + "\n".join(rows)))


def plot_domain_loss_deltas(results: dict[str, dict[str, float]]) -> None:
    """Grouped bars of loss change relative to the first checkpoint."""
    if len(results) < 2:
        return
    names = list(results)
    domains = list(results[names[0]])
    base = results[names[0]]
    x = torch.arange(len(domains), dtype=torch.float32).numpy()
    width = 0.8 / (len(names) - 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    for index, name in enumerate(names[1:]):
        deltas = [results[name][domain] - base[domain] for domain in domains]
        offset = (index - (len(names) - 2) / 2) * width
        ax.bar(x + offset, deltas, width=width, label=name)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(x, domains)
    ax.set_ylabel("loss change from base (lower is better)")
    ax.legend()
    ax.set_title("Adaptation versus retention")
    fig.tight_layout()
    plt.show()
