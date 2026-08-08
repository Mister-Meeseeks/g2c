# Hand-maintained solutions mirror — canonical, lives in `main` (see AGENTS.md).
"""Solutions for g2c.harness.failures pedagogical scaffolds.

Patched onto the scaffold targets by g2c.solutions.apply().
"""
from __future__ import annotations

from g2c.harness.events import (
    DETERMINISTIC_MARKERS,
    TRANSIENT_MARKERS,
    RetryDecision,
)
from g2c.tools import ToolResult


def classify_failure(result: ToolResult) -> RetryDecision:
    if not result.is_error:
        raise ValueError("classify_failure expects a failed ToolResult")
    text = result.output.lower()
    for marker in TRANSIENT_MARKERS:
        if marker in text:
            return RetryDecision("transient", True, f"matched {marker!r}")
    for marker in DETERMINISTIC_MARKERS:
        if marker in text:
            return RetryDecision(
                "deterministic", False, f"matched {marker!r}"
            )
    return RetryDecision(
        "deterministic", False, "unrecognized failure — surfacing to the model"
    )
