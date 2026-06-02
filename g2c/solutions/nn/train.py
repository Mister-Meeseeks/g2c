# Hand-maintained solutions mirror — canonical, lives in `main` (see AGENTS.md).
"""Solutions for g2c.nn.train pedagogical scaffolds.

Patched onto the scaffold targets by g2c.solutions.apply().
"""
from __future__ import annotations

import torch
from g2c.nn.loss import CrossEntropyLoss, MSELoss
from g2c.nn.modules import Linear, ReLU, Sequential
from g2c.nn.optim import SGD


def train_linear_regression(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    steps: int = 500,
    lr: float = 0.1,
) -> tuple[Linear, list[float]]:
    """Train ``Linear(1, 1)`` on ``x -> y`` and return ``(model, losses)``."""
    model = Linear(1, 1)
    loss_fn = MSELoss()
    optimizer = SGD(model.parameters(), lr=lr)
    losses: list[float] = []

    # SOLUTION
    for _ in range(steps):
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    return model, losses


def build_2d_classifier(hidden: int = 16) -> Sequential:
    """Return a small nonlinear classifier for inputs shaped ``(B, 2)``."""
    # SOLUTION
    return Sequential(Linear(2, hidden), ReLU(), Linear(hidden, 2))


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Return classification accuracy as a Python float."""
    # SOLUTION
    preds = logits.argmax(dim=-1)
    correct = preds == targets
    return float(correct.float().mean().item())


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
    loss_fn = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    losses: list[float] = []

    # SOLUTION
    for _ in range(steps):
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    return losses


def build_mnist_mlp(hidden: int = 128, use_relu: bool = True) -> Sequential:
    """Build the Module 03 MNIST MLP."""
    # SOLUTION
    if use_relu:
        return Sequential(Linear(784, hidden), ReLU(), Linear(hidden, 10))
    return Sequential(Linear(784, hidden), Linear(hidden, 10))


def train_one_epoch(
    model: Sequential,
    loader,
    optimizer: SGD,
    loss_fn: CrossEntropyLoss,
) -> float:
    """Train for one epoch and return mean loss weighted by batch size."""
    total_loss = 0.0
    total_examples = 0

    # SOLUTION
    for images, labels in loader:
        x = images.reshape(images.size(0), -1)
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size

    if total_examples == 0:
        raise ValueError("loader produced no examples")
    return total_loss / total_examples


def evaluate_accuracy(model: Sequential, loader) -> float:
    """Return classification accuracy on a loader."""
    correct = 0
    total = 0

    # SOLUTION
    with torch.no_grad():
        for images, labels in loader:
            x = images.reshape(images.size(0), -1)
            logits = model(x)
            preds = logits.argmax(dim=-1)
            correct += int((preds == labels).sum().item())
            total += int(labels.numel())

    if total == 0:
        raise ValueError("loader produced no examples")
    return correct / total
