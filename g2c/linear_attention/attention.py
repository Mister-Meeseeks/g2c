"""Linear attention — one computation, two forms.

Drop the softmax from Module 07's attention and the sum over history
reassociates:

    softmax(q Kᵀ) V     → must visit every past token
    (q Kᵀ) V = q (Kᵀ V) → the past collapses into a running state

With a positive feature map `φ` standing in for the softmax's
exponential and a per-head decay `γ` providing forgetting, the running
state is:

    S_t = γ · S_{t-1} + φ(k_t) v_tᵀ        # (head_dim, head_dim)
    z_t = γ · z_{t-1} + φ(k_t)             # (head_dim,)
    out_t = (φ(q_t) · S_t) / (φ(q_t) · z_t + ε)

The same numbers can be produced two ways, and implementing BOTH — and
proving they agree — is this module's core deliverable:

  * `forward` — the PARALLEL reference form. All positions at once,
    with the decay expressed as a distance-dependent mask
    `γ^(t-i)` on the score matrix. It is easy to read and supports
    teacher forcing, but deliberately materializes a `(T, T)` matrix;
    production linear-attention training needs a chunked/scan kernel.
  * `step` — the RECURRENT (inference) form. One token at a time,
    carrying `(S, z)` forward. O(1) memory in sequence length — the
    entire "KV cache" is one fixed-size matrix per head.

Boilerplate (`__init__`, `parameters`, head reshapes, `init_state`) is
implemented; `forward` and `step` are scaffolded.
"""
from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F

from g2c.nn import Linear, Module

# Keeps the normalizer away from zero. The feature map is positive, so
# denominators are positive; ε only matters for near-empty state.
EPS = 1e-6

# sigmoid(4.6) ≈ 0.99 — decay starts near "remember almost everything"
# so an untrained layer behaves like the ungated accumulator.
_GAMMA_LOGIT_INIT = 4.6


def feature_map(x: torch.Tensor) -> torch.Tensor:
    """The positive feature map φ(x) = elu(x) + 1.

    Stands in for the softmax's exponential: strictly positive, so
    scores and normalizers stay positive, but cheap and — crucially —
    factorizable, which is what lets `(qKᵀ)V` reassociate to `q(KᵀV)`.
    """
    return F.elu(x) + 1.0


class LinearAttention(Module):
    """Multi-head causal linear attention with a learned per-head decay.

    Args:
        embedding_dim: channel dim (`D`). Must be divisible by
            `num_heads`.
        num_heads: number of heads (`H`). Each head has
            `head_dim = D // H` and its own decay.

    Attributes:
        q_proj, k_proj, v_proj, out_proj: the same four `Linear(D, D)`
            projections as Module 08's multi-head attention.
        gamma_logit: `(H,)` learned tensor; the decay is
            `sigmoid(gamma_logit)`, keeping it in (0, 1).
    """

    embedding_dim: int
    num_heads: int
    head_dim: int

    def __init__(self, embedding_dim: int, num_heads: int) -> None:
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError(
                f"embedding_dim {embedding_dim} not divisible by "
                f"num_heads {num_heads}"
            )
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.q_proj = Linear(embedding_dim, embedding_dim)
        self.k_proj = Linear(embedding_dim, embedding_dim)
        self.v_proj = Linear(embedding_dim, embedding_dim)
        self.out_proj = Linear(embedding_dim, embedding_dim)
        gamma_logit = torch.full((num_heads,), _GAMMA_LOGIT_INIT)
        gamma_logit.requires_grad_(True)
        self.gamma_logit = gamma_logit

    def parameters(self) -> Iterable[torch.Tensor]:
        return [
            *self.q_proj.parameters(),
            *self.k_proj.parameters(),
            *self.v_proj.parameters(),
            *self.out_proj.parameters(),
            self.gamma_logit,
        ]

    @property
    def decay(self) -> torch.Tensor:
        """Per-head decay γ ∈ (0, 1), shape `(H,)`."""
        return torch.sigmoid(self.gamma_logit)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, D) → (B, H, T, head_dim)."""
        B, T, _ = x.shape
        return x.reshape(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, H, T, head_dim) → (B, T, D)."""
        B, _, T, _ = x.shape
        return x.permute(0, 2, 1, 3).reshape(B, T, self.embedding_dim)

    def init_state(
        self, batch_size: int, device: torch.device | str | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """An empty recurrent state: `(S, z)` of shapes
        `(B, H, head_dim, head_dim)` and `(B, H, head_dim)`."""
        S = torch.zeros(
            batch_size, self.num_heads, self.head_dim, self.head_dim,
            device=device,
        )
        z = torch.zeros(
            batch_size, self.num_heads, self.head_dim, device=device
        )
        return S, z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The parallel reference form — all positions at once.

        Args:
            x: `(B, T, D)` residual-stream input.

        Returns:
            `(B, T, D)` — same contract as Module 08's attention.

        Recipe:
            1. q = feature_map(self._split_heads(self.q_proj(x)))  # (B,H,T,d)
               k = feature_map(self._split_heads(self.k_proj(x)))
               v = self._split_heads(self.v_proj(x))               # (B,H,T,d)
            2. scores = q @ k.transpose(-2, -1)                    # (B,H,T,T)
            3. # The decay-mask: entry (t, i) is γ^(t-i) for i <= t,
               # zero for i > t (causality). This IS the recurrence,
               # unrolled: position t sees position i's contribution
               # decayed once per intervening step.
               idx = torch.arange(T, device=x.device)
               delta = (idx[:, None] - idx[None, :]).to(x.dtype)   # (T, T)
               causal = delta >= 0
               gamma = self.decay.view(self.num_heads, 1, 1)       # (H,1,1)
               decay_mask = torch.where(
                   causal, gamma ** delta.clamp(min=0), torch.zeros(())
               )                                                   # (H,T,T)
            4. weighted = scores * decay_mask                      # (B,H,T,T)
            5. out = weighted @ v / (weighted.sum(-1, keepdim=True) + EPS)
            6. return self.out_proj(self._merge_heads(out))

        Note `delta == 0` (a token attending to itself) gets weight
        γ⁰ = 1 — the current token is always fully visible. `step`
        matches this by updating the state BEFORE reading it.

        This implementation is an equivalence aid, not an efficient
        training kernel: steps 2-4 materialize `(B, H, T, T)` scores and
        masks. A production implementation uses associative scans or
        chunked recurrent kernels to realize linear-in-T training.
        """
        # TODO
        raise NotImplementedError

    def step(
        self,
        x_t: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """The recurrent (inference) form — one token, O(1) state.

        Args:
            x_t: `(B, D)` — the residual stream at ONE position.
            state: `(S, z)` from the previous step, or `None` to start
                from `init_state`.

        Returns:
            `(out, (S, z))` where `out` is `(B, D)` and the state now
            includes this token.

        Recipe:
            1. if state is None: state = self.init_state(B, x_t.device)
               S, z = state
            2. q = feature_map(self.q_proj(x_t).reshape(B, H, d))
               k = feature_map(self.k_proj(x_t).reshape(B, H, d))
               v = self.v_proj(x_t).reshape(B, H, d)
            3. # Decay, then write — update-first, so the current
               # token is visible to its own query (matches the
               # parallel form's γ⁰ = 1 diagonal):
               gamma = self.decay.view(1, H, 1)
               S = gamma.unsqueeze(-1) * S + k.unsqueeze(-1) @ v.unsqueeze(-2)
               z = gamma * z + k
            4. numer = (q.unsqueeze(-2) @ S).squeeze(-2)        # (B, H, d)
               denom = (q * z).sum(-1, keepdim=True) + EPS      # (B, H, 1)
               out = numer / denom
            5. return self.out_proj(out.reshape(B, D)), (S, z)

        The order in step 3 is the causality seam: write-then-read
        includes the current token (correct); read-then-write shifts
        everything by one and fails the equivalence test against
        `forward`.
        """
        # TODO
        raise NotImplementedError
