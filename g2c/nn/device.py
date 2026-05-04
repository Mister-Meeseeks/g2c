"""Device helpers shared by the course's from-scratch modules."""
from __future__ import annotations

import torch


def resolve_device(device: str | torch.device | None = "auto") -> torch.device:
    """Return the concrete torch device to use.

    `device="auto"` means "use MPS when this PyTorch install can see it,
    otherwise use CPU." Explicit `device="mps"` raises if MPS is not
    available so a user does not accidentally run a long CPU job.
    """
    if device is None or device == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    resolved = torch.device(device)
    if resolved.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested, but torch.backends.mps.is_available() is False.")
    return resolved
