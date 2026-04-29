"""Tests for Module 02: tensors and matmul.

These tests pin down the contract of `g2c.tensors`. Run with `pytest`.
As you implement each piece, the corresponding tests will start to pass.

Suggested order to implement & turn green:

  1. matmul_loops              → test_matmul_loops_*
  2. matmul_numpy              → test_matmul_numpy_*
  3. matmul_torch              → test_matmul_torch_*  (incl. MPS smoke test)
  4. broadcast_shapes          → test_broadcast_shapes_*
  5. TinyArray.__add__         → test_tinyarray_add_*
  6. TinyArray.__mul__         → test_tinyarray_mul_*
  7. linear                    → test_linear_*
  8. softmax                   → test_softmax_*

Construction tests (TinyArray's plumbing) pass from the start because the
boilerplate is already implemented — those serve as a sanity check on the
test file itself.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from g2c.tensors import (
    TinyArray,
    broadcast_shapes,
    linear,
    matmul_loops,
    matmul_numpy,
    matmul_torch,
    softmax,
)


# ----------------------------------------------------------------------
# TinyArray construction (plumbing — passes from the start)
# ----------------------------------------------------------------------

def test_tinyarray_construction_1d():
    a = TinyArray([1, 2, 3], (3,))
    assert a.shape == (3,)
    assert a.data == [1.0, 2.0, 3.0]


def test_tinyarray_construction_2d():
    a = TinyArray([1, 2, 3, 4, 5, 6], (2, 3))
    assert a.shape == (2, 3)
    assert a.to_nested() == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]


def test_tinyarray_construction_shape_mismatch():
    with pytest.raises(ValueError):
        TinyArray([1, 2, 3], (2, 2))


def test_tinyarray_repr_runs():
    repr(TinyArray([1, 2, 3], (3,)))  # should not raise


# ----------------------------------------------------------------------
# matmul_loops
# ----------------------------------------------------------------------

def test_matmul_loops_2x2():
    A = [[1.0, 2.0], [3.0, 4.0]]
    B = [[5.0, 6.0], [7.0, 8.0]]
    result = matmul_loops(A, B)
    assert result == [[19.0, 22.0], [43.0, 50.0]]


def test_matmul_loops_identity():
    """A @ I == A."""
    A = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    I = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    result = matmul_loops(A, I)
    assert result == A


def test_matmul_loops_rectangular():
    """(2x3) @ (3x4) = (2x4); compare against NumPy."""
    np.random.seed(0)
    A = np.random.randn(2, 3)
    B = np.random.randn(3, 4)
    expected = A @ B
    result = matmul_loops(A.tolist(), B.tolist())
    np.testing.assert_allclose(np.array(result), expected, rtol=1e-6)


# ----------------------------------------------------------------------
# matmul_numpy
# ----------------------------------------------------------------------

def test_matmul_numpy_basic():
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    B = np.array([[5.0, 6.0], [7.0, 8.0]])
    expected = np.array([[19.0, 22.0], [43.0, 50.0]])
    np.testing.assert_allclose(matmul_numpy(A, B), expected)


def test_matmul_numpy_matches_loops():
    np.random.seed(1)
    A = np.random.randn(4, 5)
    B = np.random.randn(5, 3)
    np_result = matmul_numpy(A, B)
    loop_result = np.array(matmul_loops(A.tolist(), B.tolist()))
    np.testing.assert_allclose(np_result, loop_result, rtol=1e-6)


# ----------------------------------------------------------------------
# matmul_torch
# ----------------------------------------------------------------------

def test_matmul_torch_basic():
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    B = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    expected = torch.tensor([[19.0, 22.0], [43.0, 50.0]])
    assert torch.allclose(matmul_torch(A, B), expected)


def test_matmul_torch_matches_numpy():
    torch.manual_seed(0)
    A = torch.randn(4, 5)
    B = torch.randn(5, 3)
    torch_result = matmul_torch(A, B).numpy()
    np_result = matmul_numpy(A.numpy(), B.numpy())
    np.testing.assert_allclose(torch_result, np_result, rtol=1e-5)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_matmul_torch_on_mps():
    """Matmul should run on whatever device the inputs live on, including MPS."""
    torch.manual_seed(0)
    A = torch.randn(64, 64).to("mps")
    B = torch.randn(64, 64).to("mps")
    C = matmul_torch(A, B)
    assert C.device.type == "mps"
    assert C.shape == (64, 64)


# ----------------------------------------------------------------------
# broadcast_shapes
# ----------------------------------------------------------------------

def test_broadcast_shapes_identical():
    assert broadcast_shapes((3,), (3,)) == (3,)
    assert broadcast_shapes((4, 5), (4, 5)) == (4, 5)


def test_broadcast_shapes_with_scalar():
    assert broadcast_shapes((), (3, 4)) == (3, 4)
    assert broadcast_shapes((3, 4), ()) == (3, 4)


def test_broadcast_shapes_left_pad():
    assert broadcast_shapes((4,), (3, 4)) == (3, 4)
    assert broadcast_shapes((3, 4), (4,)) == (3, 4)


def test_broadcast_shapes_size_one_stretch():
    assert broadcast_shapes((1, 4), (3, 4)) == (3, 4)
    assert broadcast_shapes((3, 1), (3, 4)) == (3, 4)


def test_broadcast_shapes_two_way_stretch():
    assert broadcast_shapes((1, 4), (3, 1)) == (3, 4)


def test_broadcast_shapes_three_dim():
    assert broadcast_shapes((2, 1, 4), (3, 4)) == (2, 3, 4)


def test_broadcast_shapes_incompatible():
    with pytest.raises(ValueError):
        broadcast_shapes((2, 3), (2, 4))
    with pytest.raises(ValueError):
        broadcast_shapes((3,), (4,))
    with pytest.raises(ValueError):
        broadcast_shapes((2, 3, 4), (2, 5, 4))


# ----------------------------------------------------------------------
# TinyArray.__add__ with broadcasting
# ----------------------------------------------------------------------

def test_tinyarray_add_same_shape():
    a = TinyArray([1, 2, 3, 4], (2, 2))
    b = TinyArray([10, 20, 30, 40], (2, 2))
    c = a + b
    assert c.shape == (2, 2)
    assert c.to_nested() == [[11.0, 22.0], [33.0, 44.0]]


def test_tinyarray_add_broadcast_row_vector():
    """(2, 3) + (3,) — row vector broadcasts down rows."""
    a = TinyArray([1, 2, 3, 4, 5, 6], (2, 3))
    b = TinyArray([10, 20, 30], (3,))
    c = a + b
    assert c.shape == (2, 3)
    assert c.to_nested() == [[11.0, 22.0, 33.0], [14.0, 25.0, 36.0]]


def test_tinyarray_add_broadcast_column_vector():
    """(2, 3) + (2, 1) — column vector broadcasts across columns."""
    a = TinyArray([1, 2, 3, 4, 5, 6], (2, 3))
    b = TinyArray([10, 20], (2, 1))
    c = a + b
    assert c.shape == (2, 3)
    assert c.to_nested() == [[11.0, 12.0, 13.0], [24.0, 25.0, 26.0]]


def test_tinyarray_add_two_way_broadcast():
    """(1, 3) + (2, 1) — both inputs stretch."""
    a = TinyArray([10, 20, 30], (1, 3))
    b = TinyArray([1, 2], (2, 1))
    c = a + b
    assert c.shape == (2, 3)
    assert c.to_nested() == [[11.0, 21.0, 31.0], [12.0, 22.0, 32.0]]


def test_tinyarray_add_matches_numpy():
    """Random check against numpy broadcasting."""
    np.random.seed(2)
    a_np = np.random.randn(3, 1, 4)
    b_np = np.random.randn(2, 4)
    expected = (a_np + b_np).tolist()

    a = TinyArray(a_np.flatten().tolist(), a_np.shape)
    b = TinyArray(b_np.flatten().tolist(), b_np.shape)
    c = a + b
    assert c.shape == (3, 2, 4)

    # Compare element-wise
    expected_flat = np.array(expected).flatten()
    np.testing.assert_allclose(c.data, expected_flat, rtol=1e-6)


# ----------------------------------------------------------------------
# TinyArray.__mul__ with broadcasting
# ----------------------------------------------------------------------

def test_tinyarray_mul_same_shape():
    a = TinyArray([1, 2, 3, 4], (2, 2))
    b = TinyArray([10, 20, 30, 40], (2, 2))
    c = a * b
    assert c.to_nested() == [[10.0, 40.0], [90.0, 160.0]]


def test_tinyarray_mul_broadcast():
    a = TinyArray([1, 2, 3, 4, 5, 6], (2, 3))
    b = TinyArray([10, 20, 30], (3,))
    c = a * b
    assert c.shape == (2, 3)
    assert c.to_nested() == [[10.0, 40.0, 90.0], [40.0, 100.0, 180.0]]


# ----------------------------------------------------------------------
# linear
# ----------------------------------------------------------------------

def test_linear_shape():
    x = torch.randn(8, 16)
    W = torch.randn(16, 4)
    b = torch.randn(4)
    y = linear(x, W, b)
    assert y.shape == (8, 4)


def test_linear_known():
    """Hand-computed example: y = x @ W + b."""
    x = torch.tensor([[1.0, 2.0]])
    W = torch.tensor([[3.0, 4.0], [5.0, 6.0]])
    b = torch.tensor([0.5, -0.5])
    # y = [[1*3 + 2*5, 1*4 + 2*6]] + [[0.5, -0.5]] = [[13.5, 15.5]]
    y = linear(x, W, b)
    assert torch.allclose(y, torch.tensor([[13.5, 15.5]]))


def test_linear_bias_broadcasts_across_batch():
    """The bias should add to every row of the batch."""
    x = torch.zeros(5, 3)
    W = torch.zeros(3, 2)
    b = torch.tensor([1.0, 2.0])
    y = linear(x, W, b)
    expected = b.unsqueeze(0).expand(5, 2)
    assert torch.allclose(y, expected)


# ----------------------------------------------------------------------
# softmax
# ----------------------------------------------------------------------

def test_softmax_sums_to_one():
    x = torch.randn(4, 8)
    y = softmax(x, dim=-1)
    assert torch.allclose(y.sum(dim=-1), torch.ones(4), atol=1e-6)


def test_softmax_all_positive():
    x = torch.randn(4, 8)
    y = softmax(x, dim=-1)
    assert (y > 0).all()


def test_softmax_uniform_input():
    """Equal logits produce a uniform distribution."""
    x = torch.zeros(1, 4)
    y = softmax(x, dim=-1)
    assert torch.allclose(y, torch.full((1, 4), 0.25))


def test_softmax_shift_invariance():
    """softmax(x) == softmax(x + c). The numerical-stability trick depends on this."""
    x = torch.randn(4, 8)
    y1 = softmax(x, dim=-1)
    y2 = softmax(x + 1000.0, dim=-1)
    assert torch.allclose(y1, y2, atol=1e-5)


def test_softmax_handles_large_values():
    """Large logits would overflow naive softmax. Stable softmax handles them."""
    x = torch.tensor([[1000.0, 1001.0, 1002.0]])
    y = softmax(x, dim=-1)
    # Result should be finite, sum to 1, and biased toward the largest entry.
    assert torch.isfinite(y).all()
    assert torch.allclose(y.sum(), torch.tensor(1.0), atol=1e-6)
    assert y[0, 2] > y[0, 1] > y[0, 0]


def test_softmax_matches_torch():
    """Sanity check against torch.nn.functional.softmax."""
    x = torch.randn(4, 8)
    ours = softmax(x, dim=-1)
    theirs = torch.nn.functional.softmax(x, dim=-1)
    assert torch.allclose(ours, theirs, atol=1e-6)


def test_softmax_along_first_dim():
    """Softmax along a non-last dim."""
    x = torch.randn(4, 8)
    y = softmax(x, dim=0)
    assert torch.allclose(y.sum(dim=0), torch.ones(8), atol=1e-6)
