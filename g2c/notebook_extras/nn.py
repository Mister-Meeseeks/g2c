"""Display helpers for the Module 03 notebook.

These are not part of the course deliverable. They render the MNIST training
curves and the per-image / per-epoch prediction grid the notebook uses to make
the model's progress visible.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import torch

__all__ = ["plot_mnist_training_curves", "plot_sample_predictions"]

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
