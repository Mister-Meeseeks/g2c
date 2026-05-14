"""Visualization helpers for the Module 03B training notebook."""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import matplotlib.pyplot as plt
import torch

from g2c.nn import CrossEntropyLoss, Sequential

__all__ = ["plot_optimizer_decision_snapshots"]


OptimizerRun = tuple[type, float, float]


def plot_optimizer_decision_snapshots(
    optimizer_runs: Mapping[str, OptimizerRun],
    *,
    build_model: Callable[[], Sequential],
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    steps: Sequence[int] = (50, 150, 300),
    seed: int = 0,
    grid_size: int = 160,
) -> dict[str, dict[int, dict[str, float]]]:
    """Show 2D decision boundaries for optimizer runs at fixed step counts.

    Rows are training snapshots; columns are optimizer settings. Each panel
    retrains from the same seed up to that step count, so the columns show the
    optimizer's learning trajectory on the same data rather than three
    unrelated random starts.
    """
    if not optimizer_runs:
        raise ValueError("optimizer_runs must not be empty")
    if not steps:
        raise ValueError("steps must not be empty")
    if grid_size < 20:
        raise ValueError("grid_size must be at least 20")
    if any(step < 0 for step in steps):
        raise ValueError("steps must be non-negative")

    all_x = torch.cat([x_train, x_val], dim=0)
    x_min, x_max = all_x[:, 0].min().item() - 0.35, all_x[:, 0].max().item() + 0.35
    y_min, y_max = all_x[:, 1].min().item() - 0.35, all_x[:, 1].max().item() + 0.35
    xs = torch.linspace(x_min, x_max, grid_size)
    ys = torch.linspace(y_min, y_max, grid_size)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing="xy")
    grid = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)

    rows = len(steps)
    cols = len(optimizer_runs)
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(4.1 * cols, 3.4 * rows),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    if rows == 1 and cols == 1:
        axes_grid = [[axes]]
    elif rows == 1:
        axes_grid = [list(axes)]
    elif cols == 1:
        axes_grid = [[ax] for ax in axes]
    else:
        axes_grid = axes

    summary: dict[str, dict[int, dict[str, float]]] = {}
    for col, (label, run_config) in enumerate(optimizer_runs.items()):
        optimizer_cls, lr, weight_decay = run_config
        summary[label] = {}
        for row, step in enumerate(steps):
            ax = axes_grid[row][col]
            model, train_loss, val_loss, val_accuracy = _train_optimizer_snapshot(
                build_model=build_model,
                optimizer_cls=optimizer_cls,
                lr=lr,
                weight_decay=weight_decay,
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                steps=int(step),
                seed=seed,
            )
            _plot_snapshot_panel(
                ax,
                model=model,
                grid=grid,
                grid_x=grid_x,
                grid_y=grid_y,
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
            )
            if row == 0:
                ax.set_title(label)
            ax.text(
                0.02,
                0.98,
                f"val acc {val_accuracy:.2f}\nval loss {val_loss:.3f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none"},
            )
            if col == 0:
                ax.set_ylabel(f"step {step}\nx2")
            ax.set_xlabel("x1")
            summary[label][int(step)] = {
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_loss, 4),
                "val_accuracy": round(val_accuracy, 3),
            }

    fig.suptitle("Optimizer trajectories on the XOR-style classification data")
    plt.show()
    return summary


def _train_optimizer_snapshot(
    *,
    build_model: Callable[[], Sequential],
    optimizer_cls: type,
    lr: float,
    weight_decay: float,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    steps: int,
    seed: int,
) -> tuple[Sequential, float, float, float]:
    torch.manual_seed(seed)
    model = build_model()
    loss_fn = CrossEntropyLoss()
    optimizer = optimizer_cls(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loss = loss_fn(model(x_train), y_train)
    for _ in range(steps):
        optimizer.zero_grad()
        train_loss = loss_fn(model(x_train), y_train)
        train_loss.backward()
        optimizer.step()
        if not torch.isfinite(train_loss):
            break

    with torch.no_grad():
        train_loss_value = float(loss_fn(model(x_train), y_train).item())
        val_logits = model(x_val)
        val_loss_value = float(loss_fn(val_logits, y_val).item())
        val_accuracy = float((val_logits.argmax(dim=-1) == y_val).float().mean().item())

    return model, train_loss_value, val_loss_value, val_accuracy


def _plot_snapshot_panel(
    ax: Any,
    *,
    model: Sequential,
    grid: torch.Tensor,
    grid_x: torch.Tensor,
    grid_y: torch.Tensor,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
) -> None:
    with torch.no_grad():
        logits = model(grid)
        probs = torch.softmax(logits, dim=-1)[:, 1].reshape(grid_x.shape)
        pred = logits.argmax(dim=-1).reshape(grid_x.shape)
        val_preds = model(x_val).argmax(dim=-1)

    ax.contourf(
        grid_x,
        grid_y,
        pred,
        levels=[-0.5, 0.5, 1.5],
        cmap="coolwarm",
        alpha=0.22,
    )
    if probs.min().item() < 0.5 < probs.max().item():
        ax.contour(
            grid_x,
            grid_y,
            probs,
            levels=[0.5],
            colors="black",
            linewidths=1.4,
        )
    ax.scatter(
        x_train[:, 0],
        x_train[:, 1],
        c=y_train,
        cmap="coolwarm",
        s=9,
        alpha=0.36,
        edgecolors="none",
    )
    ax.scatter(
        x_val[:, 0],
        x_val[:, 1],
        c=y_val,
        cmap="coolwarm",
        s=22,
        alpha=0.95,
        edgecolors="black",
        linewidths=0.35,
    )
    missed = val_preds != y_val
    if missed.any():
        ax.scatter(
            x_val[missed, 0],
            x_val[missed, 1],
            marker="x",
            c="black",
            s=32,
            linewidths=1.0,
        )
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15)
