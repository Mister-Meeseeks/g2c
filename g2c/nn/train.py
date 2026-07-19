"""Training helpers for Module 03 notebooks.

The layer implementations live in ``modules.py``, the losses in ``loss.py``,
and SGD in ``optim.py``. This file holds small editor-backed training routines
that used to live in the notebook. They are intentionally simple: their purpose
is to make the training loop visible while keeping student work in normal
Python files with tests.
"""
from __future__ import annotations

import torch

from .loss import CrossEntropyLoss, MSELoss  # noqa: F401 (for the student implementation)
from .modules import Linear, ReLU, Sequential  # noqa: F401 (for the student implementation)
from .optim import SGD


def train_linear_regression(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    steps: int = 500,
    lr: float = 0.1,
) -> tuple[Linear, list[float]]:
    """Train ``Linear(1, 1)`` on ``x -> y`` and return ``(model, losses)``."""
    # TODO
    raise NotImplementedError


def build_2d_classifier(hidden: int = 16) -> Sequential:
    """Return a small nonlinear classifier for inputs shaped ``(B, 2)``."""
    # TODO
    raise NotImplementedError


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Return classification accuracy as a Python float."""
    # TODO
    raise NotImplementedError


def train_classifier(
    model: Sequential,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    steps: int = 1000,
    lr: float = 0.1,
    weight_decay: float = 0.0,
) -> list[float]:
    """Train a full-batch classifier and return scalar losses."""
    # TODO
    raise NotImplementedError


def build_mnist_mlp(hidden: int = 128, use_relu: bool = True) -> Sequential:
    """Build the Module 03 MNIST MLP."""
    # TODO
    raise NotImplementedError


def train_one_epoch(
    model: Sequential,
    loader,
    optimizer: SGD,
    loss_fn: CrossEntropyLoss,
) -> float:
    """Train for one epoch and return mean loss weighted by batch size."""
    # TODO
    raise NotImplementedError


def evaluate_accuracy(model: Sequential, loader) -> float:
    """Return classification accuracy on a loader."""
    # TODO
    raise NotImplementedError
