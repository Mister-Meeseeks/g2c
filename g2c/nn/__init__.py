from .device import resolve_device
from .loss import CrossEntropyLoss, MSELoss
from .modules import Linear, Module, ReLU, Sequential, Sigmoid, Tanh
from .optim import SGD
from .train import (
    accuracy_from_logits,
    build_2d_classifier,
    build_mnist_mlp,
    evaluate_accuracy,
    train_classifier,
    train_linear_regression,
    train_one_epoch,
)

__all__ = [
    "SGD",
    "CrossEntropyLoss",
    "Linear",
    "MSELoss",
    "Module",
    "ReLU",
    "Sequential",
    "Sigmoid",
    "Tanh",
    "accuracy_from_logits",
    "build_2d_classifier",
    "build_mnist_mlp",
    "evaluate_accuracy",
    "resolve_device",
    "train_classifier",
    "train_linear_regression",
    "train_one_epoch",
]
