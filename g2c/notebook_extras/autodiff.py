"""Visualization helpers for the Module 01 autodiff notebook."""
from __future__ import annotations

import random
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np

from g2c.autodiff import ScalarMLP, train_xor_step, xor_loss

__all__ = ["plot_xor_decision_snapshots"]


def _train_xor_model_to_step(*, step: int, seed: int, lr: float) -> ScalarMLP:
    if step < 0:
        raise ValueError("step must be non-negative")
    model = ScalarMLP(rng=random.Random(seed))
    for _ in range(step):
        train_xor_step(model, lr=lr)
    return model


def _evaluate_xor_grid(
    model: ScalarMLP,
    xs: np.ndarray,
    ys: np.ndarray,
) -> np.ndarray:
    values = np.empty((len(ys), len(xs)), dtype=np.float32)
    for row, y in enumerate(ys):
        for col, x in enumerate(xs):
            values[row, col] = model((float(x), float(y))).data
    return values


def plot_xor_decision_snapshots(
    *,
    steps: Sequence[int] = (0, 100, 300, 500),
    seed: int = 42,
    lr: float = 0.1,
    grid_size: int = 90,
    padding: float = 0.25,
) -> None:
    """Plot XOR decision boundaries for independently trained snapshots.

    The background is the model output over the 2D input plane. The black curve
    marks output 0, the natural decision boundary for tanh targets `-1` and `1`.
    Training points are overlaid as the four XOR corners.
    """
    if grid_size < 10:
        raise ValueError("grid_size must be at least 10")
    if not steps:
        raise ValueError("steps must not be empty")

    xs = np.linspace(-padding, 1.0 + padding, grid_size)
    ys = np.linspace(-padding, 1.0 + padding, grid_size)
    points = np.array(
        [
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, -1.0],
        ],
        dtype=np.float32,
    )

    fig, axes = plt.subplots(
        1,
        len(steps),
        figsize=(3.4 * len(steps), 3.5),
        sharex=True,
        sharey=True,
    )
    if len(steps) == 1:
        axes = np.array([axes])

    contour = None
    levels = np.linspace(-1.0, 1.0, 21)
    for ax, step in zip(axes, steps, strict=True):
        model = _train_xor_model_to_step(step=int(step), seed=seed, lr=lr)
        values = _evaluate_xor_grid(model, xs, ys)
        contour = ax.contourf(
            xs,
            ys,
            values,
            levels=levels,
            cmap="coolwarm",
            vmin=-1.0,
            vmax=1.0,
            alpha=0.78,
        )
        ax.contour(xs, ys, values, levels=[0.0], colors="black", linewidths=1.5)

        negative = points[:, 2] < 0
        positive = points[:, 2] > 0
        ax.scatter(
            points[negative, 0],
            points[negative, 1],
            marker="o",
            s=70,
            c="white",
            edgecolors="black",
            linewidths=1.3,
            label="-1 target",
        )
        ax.scatter(
            points[positive, 0],
            points[positive, 1],
            marker="^",
            s=80,
            c="black",
            edgecolors="white",
            linewidths=0.9,
            label="+1 target",
        )

        loss = xor_loss(model).data
        ax.set_title(f"step {step}\nloss {loss:.3f}")
        ax.set_aspect("equal")
        ax.set_xlim(xs[0], xs[-1])
        ax.set_ylim(ys[0], ys[-1])
        ax.set_xlabel("x1")
        ax.grid(True, alpha=0.18)

    axes[0].set_ylabel("x2")
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    if contour is not None:
        fig.colorbar(contour, ax=axes, shrink=0.82, label="model output")
    fig.suptitle("XOR decision boundary during scalar-MLP training", y=1.04)
    plt.show()
