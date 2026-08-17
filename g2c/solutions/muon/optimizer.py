# Hand-maintained solutions mirror — canonical, lives in `main` (see AGENTS.md).
"""Solutions for g2c.muon.optimizer pedagogical scaffolds.

Patched onto the scaffold targets by g2c.solutions.apply().
"""
from __future__ import annotations

import torch

from g2c.muon.optimizer import EPS, NS_COEFFS


def zeropower_via_newtonschulz(
    G: torch.Tensor, steps: int = 5, *, eps: float = EPS
) -> torch.Tensor:
    if G.ndim != 2:
        raise ValueError(
            f"Newton–Schulz orthogonalization needs a 2-D tensor; got "
            f"shape {tuple(G.shape)}"
        )
    a, b, c = NS_COEFFS
    X = G / (G.norm() + eps)
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


class _MuonImpl:  # patched onto Muon by apply()
    def step(self) -> None:
        with torch.no_grad():
            for p, buf in zip(self.muon_params, self.momentum_buffers):
                if p.grad is None:
                    continue
                buf.mul_(self.momentum).add_(p.grad)
                update = zeropower_via_newtonschulz(buf, steps=self.ns_steps)
                scale = max(1.0, p.shape[0] / p.shape[1]) ** 0.5
                p.mul_(1 - self.lr * self.weight_decay)
                p.add_(update, alpha=-self.lr * scale)
        if self.adamw_params:
            self._adamw.step()
        self.step_count += 1
