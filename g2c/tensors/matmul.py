"""Three implementations of matrix multiplication.

The same mathematical operation, expressed at three levels of abstraction:
    1. matmul_loops  — pure Python triple-nested loop
    2. matmul_numpy  — NumPy (one line, vectorized via BLAS)
    3. matmul_torch  — PyTorch (one line, executes on whatever device the
                       input tensors live on; pass MPS tensors to use the
                       Apple Silicon GPU)

All three should produce identical outputs (up to floating-point rounding).
The point of having all three is to measure the cost difference, not because
you'd ever ship the loop version.

Search for `# TODO` to find the spots to implement.
"""
from __future__ import annotations

import numpy as np
import torch


def matmul_loops(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    """Triple-nested-loop matmul. C = A @ B.

    Args:
        A: m×k matrix as a list of lists of floats.
        B: k×n matrix as a list of lists of floats.

    Returns:
        m×n matrix as a list of lists of floats.

    The shape rule: (m, k) @ (k, n) = (m, n). The inner dimensions must
    agree (assert this) and they collapse; the outer dims become the output.

    Each output entry:
        C[i][j] = sum over p in 0..k of A[i][p] * B[p][j]
    """
    # TODO
    raise NotImplementedError

def matmul_numpy(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """NumPy matmul. C = A @ B.

    One line. Use the `@` operator or `np.matmul`. (`np.einsum("ij,jk->ik", A, B)`
    works too, and makes the dimension contraction explicit, but `@` is idiomatic.)
    """
    # TODO
    raise NotImplementedError



def matmul_torch(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """PyTorch matmul. C = A @ B. Output lives on the same device as the inputs.

    One line. Use `torch.matmul` or the `@` operator.
    """
    # TODO
    raise NotImplementedError
