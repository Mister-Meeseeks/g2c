"""Muon — momentum orthogonalized by Newton–Schulz iteration.

Module 03's initialization argument said: a weight matrix's *scale* must
respect its geometry, or signal explodes or dies through depth. Muon is
the same argument applied to the *update*. A raw gradient (or momentum)
matrix has directions — its singular vectors — and their gains — its
singular values — and those gains can be wildly unequal: a few
directions dominate while the rest barely move. Muon equalizes them.
Before applying a matrix's momentum as an update, it pushes every
singular value toward 1, so the step explores all of the update's
directions at comparable magnitude instead of only its loudest ones.

The orthogonalization is done without ever computing an SVD. A fixed
odd polynomial, iterated a few times, does the job:

    X_0 = G / ||G||_F                      # normalize into the basin
    X_{i+1} = a·X_i + (b·A + c·A²) X_i     where A = X_i X_iᵀ

with coefficients `(a, b, c) = (3.4445, -4.7750, 2.0315)`. Each pass
applies the quintic `f(s) = a·s + b·s³ + c·s⁵` to every singular value.
`f` is steep near zero (small singular values get amplified hard) and
has a fixed point near 1 (`f(1) = a + b + c ≈ 1.001`), so five passes
squeeze the whole spectrum into a band around 1 while leaving the
singular *vectors* untouched. The result approximates `U Vᵀ` — the
nearest orthogonal(-ish) matrix to the momentum.

Two structural rules, both visible in every production deployment
(DeepSeek V4, GLM-5, Kimi K2):

  * **Muon is for matrices.** Orthogonalization is only defined for 2-D
    parameters. Vectors (biases, norm gains) and matrices whose rows
    are really a stack of unrelated per-token objects (embedding
    tables, the unembedding) stay on AdamW.
  * **Muon is therefore a hybrid.** This class composes with Module
    03B's `AdamW` rather than replacing it: you hand it two parameter
    lists, and the second is driven by an ordinary internal `AdamW`.

Boilerplate (`__init__`, validation, `zero_grad`) is implemented;
`zeropower_via_newtonschulz` and `step` are scaffolded.
"""
from __future__ import annotations

from collections.abc import Iterable

import torch

from g2c.training import AdamW

# Quintic Newton–Schulz coefficients (Jordan, 2024). Tuned so the
# iteration converges *fast* into a loose band around 1 rather than
# slowly to exactly 1 — for an optimizer step, "every direction gets
# roughly unit gain" is all that matters.
NS_COEFFS: tuple[float, float, float] = (3.4445, -4.7750, 2.0315)

# Keeps the Frobenius normalization away from dividing by zero for an
# all-zero momentum matrix (e.g. a parameter that never got a gradient).
EPS = 1e-7


def zeropower_via_newtonschulz(
    G: torch.Tensor, steps: int = 5, *, eps: float = EPS
) -> torch.Tensor:
    """Approximately orthogonalize `G` via Newton–Schulz iteration.

    Args:
        G: a 2-D tensor. If `G = U S Vᵀ` is its SVD, the result is
            approximately `U Vᵀ` — same singular vectors, singular
            values pushed into a band around 1.
        steps: number of polynomial passes. 5 is the standard choice;
            each pass is three small matmuls.
        eps: floor added to the Frobenius norm before dividing.

    Returns:
        A tensor with `G`'s shape and (approximately) unit singular
        values.

    Recipe:
        1. if G.ndim != 2: raise ValueError(...)
        2. a, b, c = NS_COEFFS
        3. # Normalize so every singular value lands in [0, 1] — the
           # polynomial's basin of attraction. Without this, large
           # spectra diverge under the iteration.
           X = G / (G.norm() + eps)
        4. # Work with the smaller Gram matrix: if G is tall (rows >
           # cols), transpose first so `A = X @ X.T` is (cols, cols),
           # then transpose back at the end.
           transposed = X.shape[0] > X.shape[1]
           if transposed: X = X.T
        5. for _ in range(steps):
               A = X @ X.T
               B = b * A + c * (A @ A)
               X = a * X + B @ X
        6. if transposed: X = X.T
           return X

    Sanity anchors (the tests pin these):
      * An orthogonal input comes back as a scalar multiple of itself:
        its singular values are all equal, so the polynomial moves the
        whole spectrum together and the singular vectors are untouched.
      * The result is scale-invariant: `NS(G)` equals `NS(7·G)`
        because step 3 normalizes first.
      * `NS(Gᵀ) = NS(G)ᵀ` — the iteration commutes with transposition.
    """
    # TODO
    raise NotImplementedError


class Muon:
    """Muon for matrix parameters, an internal AdamW for the rest.

    Args:
        muon_params: the 2-D parameters to update with orthogonalized
            momentum — attention and FFN projection matrices. Every
            entry must have `ndim == 2`; anything else is rejected,
            because orthogonalization is undefined for it.
        adamw_params: everything else — embeddings, positional tables,
            norm gains, biases, the unembedding bias. Driven by an
            internal `g2c.training.AdamW` (your Module 03B
            implementation — Muon composes with it, it does not
            replace it). May be empty.
        lr: learning rate for the Muon (matrix) side. Muon's
            orthogonalized updates have roughly unit scale regardless
            of gradient magnitude, so its useful lr range sits far from
            AdamW's — sweep it, don't copy `3e-4`. `0.02` is a common
            starting point.
        momentum: heavy-ball momentum coefficient `μ`. The buffer is
            `m ← μ·m + grad` (no `(1-μ)` dampening — the subsequent
            normalization inside Newton–Schulz absorbs the scale).
        weight_decay: decoupled shrinkage on the Muon side, applied
            exactly like AdamW's (`param *= 1 - lr·wd` before the
            update).
        ns_steps: Newton–Schulz passes per update.
        adamw_lr, betas, eps, adamw_weight_decay: forwarded to the
            internal `AdamW` for the non-matrix side.

    Public state:
        params: all parameters (muon side first, then adamw side).
        muon_params / adamw_params: the two groups.
        momentum_buffers: one buffer per Muon parameter, same shapes.
        lr: mutable Muon-side learning rate. (A schedule that mutates
            `lr` drives only the Muon side; reach through `._adamw.lr`
            to schedule the other group — the seam is real in
            production recipes too, which schedule the two sides
            separately.)
        step_count: number of `step()` calls applied so far.

    The external interface matches `SGD`/`AdamW`: `zero_grad()`,
    mutable `lr`, and `step()` — so it drops into Module 10's trainer.
    """

    muon_params: list[torch.Tensor]
    adamw_params: list[torch.Tensor]
    lr: float
    momentum: float
    weight_decay: float
    ns_steps: int
    momentum_buffers: list[torch.Tensor]
    step_count: int

    def __init__(
        self,
        muon_params: Iterable[torch.Tensor],
        adamw_params: Iterable[torch.Tensor] = (),
        lr: float = 0.02,
        *,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
        adamw_lr: float = 3e-4,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        adamw_weight_decay: float = 0.0,
    ) -> None:
        self.muon_params = list(muon_params)
        for p in self.muon_params:
            if p.ndim != 2:
                raise ValueError(
                    "muon_params must all be 2-D matrices; got a "
                    f"parameter with shape {tuple(p.shape)}. Vectors, "
                    "scalars, and embedding-like tables belong in "
                    "adamw_params."
                )
        self.adamw_params = list(adamw_params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.ns_steps = ns_steps
        self.momentum_buffers = [torch.zeros_like(p) for p in self.muon_params]
        self._adamw = AdamW(
            self.adamw_params,
            lr=adamw_lr,
            betas=betas,
            eps=eps,
            weight_decay=adamw_weight_decay,
        )
        self.step_count = 0

    @property
    def params(self) -> list[torch.Tensor]:
        return [*self.muon_params, *self.adamw_params]

    def zero_grad(self) -> None:
        """Set every parameter's gradient to zero in place."""
        for p in self.muon_params:
            if p.grad is not None:
                p.grad.zero_()
        self._adamw.zero_grad()

    def step(self) -> None:
        """Apply one hybrid update: Muon to matrices, AdamW to the rest.

        Update rule for each Muon parameter with a populated gradient:

            m = momentum * m + grad                       # in place
            O = zeropower_via_newtonschulz(m, steps=ns_steps)
            scale = sqrt(max(1, rows / cols))
            param *= (1 - lr * weight_decay)
            param -= lr * scale * O

        then one step of the internal AdamW for the other group.

        Important details:
          - Skip parameters whose `.grad is None`, but keep
            `momentum_buffers` aligned with `muon_params` by index.
          - Update the buffer IN PLACE (`m.mul_(μ).add_(grad)`), then
            pass it to `zeropower_via_newtonschulz`, which reads it and
            returns a fresh tensor — the buffer itself carries to the
            next step unorthogonalized.
          - `scale = sqrt(max(1, rows/cols))` compensates rectangular
            matrices: an orthogonalized `(rows, cols)` update has RMS
            entry size `~1/sqrt(rows)`, so tall matrices would take
            vanishing per-entry steps without it.
          - Call `self._adamw.step()` only if `self.adamw_params` is
            non-empty (an empty AdamW group has nothing to do).
          - Increment `step_count` once per call.
          - Wrap the Muon-side update in `torch.no_grad()`.
        """
        # TODO
        raise NotImplementedError
