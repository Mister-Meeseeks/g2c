"""Broadcasting from scratch on a toy class.

Broadcasting is the rule that lets `a + b` work when `a` and `b` have different
but compatible shapes — the smaller dimensions are implicitly stretched without
copying. This file implements the rule end-to-end in pure Python so the logic
is fully visible (NumPy and PyTorch both implement the same rule, much faster).

The plumbing (constructing TinyArray, converting back to nested lists for
inspection, validating shapes at construction) is implemented for you. The
broadcasting logic itself — `broadcast_shapes` and the element-wise ops —
is left to implement.

Search for `# TODO` to find the spots that need work.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Union

Number = Union[int, float]


def broadcast_shapes(
    shape_a: tuple[int, ...],
    shape_b: tuple[int, ...],
) -> tuple[int, ...]:
    """Compute the output shape when broadcasting `shape_a` and `shape_b`.

    The rules (NumPy/PyTorch standard):

    1. Right-align the shapes (compare from the trailing dimension).
    2. Pad the shorter shape on the LEFT with 1s until they match in length.
    3. For each dimension, the two sizes must be equal, OR one of them must
       be 1. Otherwise raise ValueError.
    4. The output dimension is `max(a, b)` (with 1 acting as the smaller).

    Examples:
        broadcast_shapes((3,), (3,))         == (3,)
        broadcast_shapes((), (3, 4))         == (3, 4)
        broadcast_shapes((4,), (3, 4))       == (3, 4)
        broadcast_shapes((1, 4), (3, 4))     == (3, 4)
        broadcast_shapes((1, 4), (3, 1))     == (3, 4)
        broadcast_shapes((2, 3), (2, 4))     -> ValueError
        broadcast_shapes((3,), (4,))         -> ValueError

    Args:
        shape_a, shape_b: shape tuples (possibly empty for scalars).

    Returns:
        The broadcasted output shape as a tuple.

    Raises:
        ValueError: if the shapes are not broadcast-compatible.
    """
    # TODO
    raise NotImplementedError


class TinyArray:
    """A minimal teaching array with NumPy-style broadcasting in element-wise ops.

    Storage is a flat list of floats in row-major (C) order. The shape is a
    tuple of dimension sizes. Indexing is computed by row-major flattening:

        flat_index = sum over d of indices[d] * stride[d]
        stride[d]  = product of shape[d+1:]   (rightmost dim has stride 1)

    Both `__add__` and `__mul__` broadcast their inputs together using
    `broadcast_shapes` to compute the output shape, then iterate over output
    indices and compute each output element from the appropriate (possibly
    broadcasted) input elements.
    """

    data: list[float]
    shape: tuple[int, ...]

    def __init__(self, data: Iterable[Number], shape: tuple[int, ...]) -> None:
        self.data = [float(x) for x in data]
        self.shape = tuple(shape)
        expected = 1
        for d in self.shape:
            expected *= d
        if len(self.data) != expected:
            raise ValueError(
                f"data length {len(self.data)} does not match shape {self.shape} "
                f"(expected {expected} elements)"
            )

    def __repr__(self) -> str:
        return f"TinyArray(shape={self.shape}, data={self.data})"

    def to_nested(self) -> Union[float, list]:
        """Convert to nested Python lists for inspection (helper for tests)."""

        def _build(offset: int, dims: tuple[int, ...]) -> Union[float, list]:
            if not dims:
                return self.data[offset]
            head, *rest = dims
            stride = 1
            for d in rest:
                stride *= d
            return [_build(offset + i * stride, tuple(rest)) for i in range(head)]

        return _build(0, self.shape)

    # ----- broadcasting element-wise ops — STUDENT IMPLEMENTS -----

    def __add__(self, other: "TinyArray") -> "TinyArray":
        """Element-wise add with broadcasting.

        Steps:
            1. Compute `out_shape = broadcast_shapes(self.shape, other.shape)`.
            2. Allocate an output data list of the right total length.
            3. For each output multi-dimensional index:
                 a. Map it back to an index into `self`, treating any size-1
                    or padded dim of `self` as 0 in that position.
                 b. Same for `other`.
                 c. Sum the two scalar values into the output position.
            4. Return TinyArray(out_data, out_shape).
        """
        # TODO
        raise NotImplementedError

    def __mul__(self, other: "TinyArray") -> "TinyArray":
        """Element-wise multiply with broadcasting. Same logic as __add__,
        with multiplication instead of addition."""
        # TODO
        raise NotImplementedError
