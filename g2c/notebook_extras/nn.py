"""Display helpers for the Module 03 notebook.

These are not part of the course deliverable. They render the MNIST training
curves and the per-image / per-epoch prediction grid the notebook uses to make
the model's progress visible.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import torch

from g2c.nn import (
    CrossEntropyLoss,
    Linear,
    SGD,
    Sequential,
    accuracy_from_logits,
    build_2d_classifier,
    build_mnist_mlp,
    evaluate_accuracy,
    train_classifier,
    train_one_epoch,
)

__all__ = [
    "compare_with_and_without_relu",
    "plot_mnist_training_curves",
    "plot_sample_predictions",
    "run_weight_decay_experiment",
]

_MNIST_MEAN = 0.1307
_MNIST_STD = 0.3081


def plot_mnist_training_curves(
    train_losses: Sequence[float],
    test_accuracies: Sequence[float],
) -> None:
    """Plot per-epoch train loss and test accuracy on twin y axes."""
    epochs = list(range(1, len(train_losses) + 1))
    fig, ax_loss = plt.subplots(figsize=(7, 3.5))

    loss_color = "tab:red"
    ax_loss.plot(epochs, train_losses, "o-", color=loss_color, label="train loss")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("train loss", color=loss_color)
    ax_loss.tick_params(axis="y", labelcolor=loss_color)
    ax_loss.set_xticks(epochs)

    ax_acc = ax_loss.twinx()
    acc_color = "tab:blue"
    ax_acc.plot(epochs, test_accuracies, "s-", color=acc_color, label="test accuracy")
    ax_acc.set_ylabel("test accuracy", color=acc_color)
    ax_acc.tick_params(axis="y", labelcolor=acc_color)
    ax_acc.set_ylim(0.0, 1.0)

    ax_loss.set_title("MNIST training")
    ax_loss.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()


def plot_sample_predictions(
    images: torch.Tensor,
    true_labels: torch.Tensor,
    predictions_per_epoch: Sequence[torch.Tensor],
    *,
    mean: float = _MNIST_MEAN,
    std: float = _MNIST_STD,
) -> None:
    """Show MNIST samples with each epoch's predictions stacked below.

    The top row holds image thumbnails labelled with the true class. Each row
    underneath shows that image's predicted class for one epoch, coloured green
    when correct and red when wrong.

    Args:
        images: (N, 1, 28, 28) tensor in the same normalized form the model saw.
        true_labels: (N,) integer tensor of ground-truth classes.
        predictions_per_epoch: list of (N,) integer tensors, one per epoch.
        mean, std: normalization stats used to un-normalize for display.
    """
    n_samples = images.size(0)
    n_epochs = len(predictions_per_epoch)
    if n_samples == 0 or n_epochs == 0:
        return

    fig, axes = plt.subplots(
        nrows=n_epochs + 1,
        ncols=n_samples,
        figsize=(n_samples * 0.4, (n_epochs + 1) * 0.45),
        gridspec_kw={"hspace": 0.1, "wspace": 0.1},
    )
    if n_samples == 1:
        axes = axes.reshape(-1, 1)

    for j in range(n_samples):
        true = int(true_labels[j])

        img = (images[j, 0] * std + mean).clamp(0.0, 1.0)
        top = axes[0, j]
        top.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
        top.set_title(str(true), fontsize=8, pad=2)
        top.set_xticks([])
        top.set_yticks([])

        for i, preds in enumerate(predictions_per_epoch):
            ax = axes[i + 1, j]
            pred = int(preds[j])
            ax.text(
                0.5,
                0.5,
                str(pred),
                ha="center",
                va="center",
                fontsize=9,
                color="green" if pred == true else "red",
                transform=ax.transAxes,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

    axes[0, 0].set_ylabel(
        "true", rotation=0, fontsize=8, labelpad=20, ha="right", va="center"
    )
    for i in range(n_epochs):
        axes[i + 1, 0].set_ylabel(
            f"ep {i + 1}", rotation=0, fontsize=8, labelpad=20, ha="right", va="center"
        )

    fig.suptitle("predictions per epoch (green = correct, red = wrong)", fontsize=10)
    plt.show()


def _train_mnist_experiment(
    *,
    hidden: int,
    use_relu: bool,
    weight_decay: float,
    train_loader,
    test_loader,
    epochs: int,
    lr: float,
) -> dict[str, object]:
    """Train one MNIST MLP and return curves.

    This is notebook scaffolding, not a module deliverable. The conceptual
    training pieces live in `g2c.nn.train`; this wrapper just keeps repetitive
    experiment bookkeeping out of the notebook.
    """
    model = build_mnist_mlp(hidden=hidden, use_relu=use_relu)
    optimizer = SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = CrossEntropyLoss()
    train_losses: list[float] = []
    test_accuracies: list[float] = []

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn)
        test_acc = evaluate_accuracy(model, test_loader)
        train_losses.append(train_loss)
        test_accuracies.append(test_acc)
        print(
            f"epoch {epoch + 1:>2}/{epochs} | "
            f"loss {train_loss:.4f} | test acc {test_acc:.3f}"
        )

    return {
        "model": model,
        "train_losses": train_losses,
        "test_accuracies": test_accuracies,
    }


def run_weight_decay_experiment(
    train_loader,
    test_loader,
    *,
    hidden: int = 256,
    epochs: int = 5,
    lr: float = 0.08,
    weight_decay: float = 1e-4,
) -> dict[str, dict[str, object]]:
    """Compare matched MNIST MLPs with and without weight decay."""
    results: dict[str, dict[str, object]] = {}
    settings = [
        ("no_weight_decay", 0.0),
        ("weight_decay", weight_decay),
    ]

    for label, wd in settings:
        print(f"\n{label}")
        results[label] = _train_mnist_experiment(
            hidden=hidden,
            use_relu=True,
            weight_decay=wd,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=epochs,
            lr=lr,
        )

    plt.figure(figsize=(7, 3.5))
    for label, result in results.items():
        plt.plot(result["train_losses"], marker="o", label=f"{label} train loss")
    plt.xlabel("epoch")
    plt.ylabel("train loss")
    plt.title("Weight decay comparison")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 3.5))
    for label, result in results.items():
        plt.plot(result["test_accuracies"], marker="o", label=f"{label} test acc")
    plt.xlabel("epoch")
    plt.ylabel("test accuracy")
    plt.ylim(0.0, 1.0)
    plt.title("Validation accuracy with and without weight decay")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return results


def compare_with_and_without_relu(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    hidden: int = 16,
    steps: int = 1000,
    lr: float = 0.1,
) -> dict[str, dict[str, object]]:
    """Train matched 2D classifiers with and without the hidden ReLU."""
    results: dict[str, dict[str, object]] = {}
    settings = [
        ("with_relu", build_2d_classifier(hidden=hidden)),
        ("without_relu", Sequential(Linear(2, hidden), Linear(hidden, 2))),
    ]

    for label, model in settings:
        print(f"\n{label}")
        losses = train_classifier(
            model,
            x,
            y,
            steps=steps,
            lr=lr,
        )
        accuracy = accuracy_from_logits(model(x), y)
        print(f"final loss {losses[-1]:.4f} | train acc {accuracy:.3f}")
        results[label] = {
            "model": model,
            "train_losses": losses,
            "train_accuracy": accuracy,
        }

    plt.figure(figsize=(7, 3.5))
    for label, result in results.items():
        plt.plot(result["train_losses"], label=label)
    plt.xlabel("step")
    plt.ylabel("cross-entropy")
    plt.title("Effect of removing the nonlinearity")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    _plot_2d_classifier_comparison(results, x, y)

    return results


def _plot_2d_classifier_comparison(
    results: dict[str, dict[str, object]],
    x: torch.Tensor,
    y: torch.Tensor,
) -> None:
    """Show the learned 2D decision regions for a comparison result."""
    x_min, x_max = x[:, 0].min().item() - 0.3, x[:, 0].max().item() + 0.3
    y_min, y_max = x[:, 1].min().item() - 0.3, x[:, 1].max().item() + 0.3
    xs = torch.linspace(x_min, x_max, 160)
    ys = torch.linspace(y_min, y_max, 160)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing="xy")
    grid = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)

    fig, axes = plt.subplots(
        1,
        len(results),
        figsize=(5 * len(results), 4.2),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    if len(results) == 1:
        axes = [axes]

    for ax, (label, result) in zip(axes, results.items(), strict=True):
        model = result["model"]
        assert isinstance(model, Sequential)
        with torch.no_grad():
            logits = model(grid)
            probs = torch.softmax(logits, dim=-1)[:, 1].reshape(grid_x.shape)
            pred = logits.argmax(dim=-1).reshape(grid_x.shape)

        ax.contourf(grid_x, grid_y, pred, levels=1, alpha=0.22, cmap="coolwarm")
        boundary_levels = [0.5]
        ax.contour(
            grid_x,
            grid_y,
            probs,
            levels=boundary_levels,
            colors="black",
            linewidths=1.6,
        )
        ax.scatter(x[:, 0], x[:, 1], c=y, s=12, cmap="coolwarm", edgecolors="none")
        ax.set_title(f"{label}\ntrain acc {result['train_accuracy']:.3f}")
        ax.set_xlabel("x1")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.15)

    axes[0].set_ylabel("x2")
    fig.suptitle("Decision boundary with and without ReLU")
    plt.show()
