"""Notebook helpers for Module 12 scaling experiments.

These helpers keep the scaling notebook focused on the experiment rather than
on artifact-table formatting. They are notebook ergonomics only; the course
deliverable for Module 12 is the comparison and interpretation, not this file.
"""

from __future__ import annotations

import math
from typing import Any

import matplotlib.pyplot as plt


def count_parameters(model: Any) -> int:
    """Return the exact parameter count for a course model."""
    return sum(p.numel() for p in model.parameters())


def artifact_summary_row(artifact: Any) -> dict[str, Any]:
    """Return one table row for a loaded model artifact."""
    manifest = artifact.manifest
    training_config = artifact.training_config
    params = count_parameters(artifact.model)
    steps = int(manifest.get("steps_completed") or 0)
    batch_size = training_config.get("batch_size")
    context_length = training_config.get("context_length")
    tokens_per_step = (
        int(batch_size) * int(context_length)
        if isinstance(batch_size, int) and isinstance(context_length, int)
        else None
    )
    tokens_seen = steps * tokens_per_step if tokens_per_step is not None else None
    flops = 6 * params * tokens_seen if tokens_seen is not None else None
    final_val = manifest.get("final_val_loss")
    return {
        "name": artifact.canonical_name,
        "saved_as": artifact.name,
        "display": artifact.display_name,
        "params": params,
        "vocab": getattr(artifact.model, "vocab_size", None),
        "layers": getattr(artifact.model, "num_layers", None),
        "width": getattr(artifact.model, "embedding_dim", None),
        "context": getattr(artifact.model, "max_seq_len", None),
        "steps": steps,
        "tokens_seen": tokens_seen,
        "flops": flops,
        "final_train_loss": manifest.get("final_train_loss"),
        "final_val_loss": final_val,
        "final_val_ppl": math.exp(final_val) if isinstance(final_val, (int, float)) else None,
        "source": manifest.get("source"),
    }


def artifact_summary_rows(artifacts: list[Any]) -> list[dict[str, Any]]:
    """Return sorted summary rows for loaded model artifacts."""
    rows = [artifact_summary_row(artifact) for artifact in artifacts]
    return sorted(rows, key=lambda row: row["params"])


def human_count(value: int | float | None) -> str:
    """Format a large count compactly for notebook tables."""
    if value is None:
        return "-"
    value = float(value)
    if abs(value) >= 1_000_000_000_000:
        return f"{value / 1_000_000_000_000:.2f}T"
    if abs(value) >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:,.0f}"


def markdown_table(rows: list[dict[str, Any]]) -> str:
    """Return a Markdown table for scaling-artifact summaries."""
    header = (
        "| model | params | vocab | layers | width | context | steps | "
        "tokens seen | val loss | val ppl |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    lines = [header]
    for row in rows:
        val_loss = row["final_val_loss"]
        val_ppl = row["final_val_ppl"]
        val_loss_text = f"{val_loss:.3f}" if isinstance(val_loss, (int, float)) else "-"
        val_ppl_text = f"{val_ppl:.1f}" if isinstance(val_ppl, float) else "-"
        lines.append(
            "| "
            f"{row['name']} | "
            f"{human_count(row['params'])} | "
            f"{row['vocab'] if row['vocab'] is not None else '-'} | "
            f"{row['layers'] if row['layers'] is not None else '-'} | "
            f"{row['width'] if row['width'] is not None else '-'} | "
            f"{row['context'] if row['context'] is not None else '-'} | "
            f"{row['steps']:,} | "
            f"{human_count(row['tokens_seen'])} | "
            f"{val_loss_text} | "
            f"{val_ppl_text} |"
        )
    return "\n".join(lines)


def plot_loss_vs_params(rows: list[dict[str, Any]]) -> None:
    """Plot final validation loss and perplexity against parameter count."""
    usable = [
        row
        for row in rows
        if isinstance(row.get("final_val_loss"), (int, float))
        and isinstance(row.get("params"), int)
    ]
    if not usable:
        print("No rows have final validation losses yet.")
        return

    xs = [row["params"] for row in usable]
    losses = [row["final_val_loss"] for row in usable]
    ppls = [row["final_val_ppl"] for row in usable]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(xs, losses, marker="o")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("parameters")
    axes[0].set_ylabel("validation cross entropy")
    axes[0].set_title("Loss vs size")

    axes[1].plot(xs, ppls, marker="o")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("parameters")
    axes[1].set_ylabel("validation perplexity")
    axes[1].set_title("Perplexity vs size")

    for ax in axes:
        for row in usable:
            y = row["final_val_loss"] if ax is axes[0] else row["final_val_ppl"]
            ax.annotate(row["name"].replace("StoryLM-", ""), (row["params"], y))
    fig.tight_layout()
    plt.show()


def plot_compute_vs_loss(rows: list[dict[str, Any]]) -> None:
    """Plot validation loss against approximate training FLOPs."""
    usable = [
        row
        for row in rows
        if isinstance(row.get("final_val_loss"), (int, float))
        and isinstance(row.get("flops"), int)
        and row["flops"] > 0
    ]
    if not usable:
        print("No rows have enough metadata for a compute plot yet.")
        return
    xs = [row["flops"] for row in usable]
    ys = [row["final_val_loss"] for row in usable]
    plt.figure(figsize=(6, 4))
    plt.plot(xs, ys, marker="o")
    plt.xscale("log")
    plt.xlabel("approximate training FLOPs")
    plt.ylabel("validation cross entropy")
    plt.title("Loss vs compute spent")
    for row in usable:
        plt.annotate(
            row["name"].replace("StoryLM-", ""),
            (row["flops"], row["final_val_loss"]),
        )
    plt.tight_layout()
    plt.show()
