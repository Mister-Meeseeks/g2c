"""classify_failure — transient, deterministic, or out of budget.

Long-running agents meet failures constantly. The harness's job is to
keep TRANSIENT failures (a timeout, a busy resource) from becoming
terminal, and DETERMINISTIC failures (file not found, permission
denied) from becoming infinite retry loops. The classification policy
lives in two readable marker tables in `events.py`
(`TRANSIENT_MARKERS`, `DETERMINISTIC_MARKERS`); this function applies
them.

Scaffolded — the three-way decision is small, but getting its default
right (when in doubt, do NOT retry) is the lesson.
"""
from __future__ import annotations

from g2c.tools import ToolResult

from .events import (  # noqa: F401 - marker tables used by scaffold
    DETERMINISTIC_MARKERS,
    TRANSIENT_MARKERS,
    RetryDecision,
)


def classify_failure(result: ToolResult) -> RetryDecision:
    """Decide what one FAILED tool result means for the loop.

    Args:
        result: a `ToolResult` with `is_error=True`. Calling this on a
            successful result is a caller bug — raise `ValueError`.

    Returns:
        A `RetryDecision`:
          * any `TRANSIENT_MARKERS` substring in the (lowercased)
            output → `kind="transient"`, `retry=True`
          * otherwise, any `DETERMINISTIC_MARKERS` substring →
            `kind="deterministic"`, `retry=False`
          * otherwise → `kind="deterministic"`, `retry=False` — the
            DEFAULT is not to retry. An unrecognized failure gets
            surfaced to the model, which can read the message and
            change its approach; a blind retry can only repeat it.

        Check transient markers FIRST: an output matching both tables
        ("connection reset by peer: invalid state") is worth one
        retry.

    Recipe:
        1. if not result.is_error: raise ValueError(...)
        2. text = result.output.lower()
        3. for marker in TRANSIENT_MARKERS: ...
               return RetryDecision("transient", True, f"matched {marker!r}")
        4. for marker in DETERMINISTIC_MARKERS: ...
               return RetryDecision("deterministic", False, f"matched {marker!r}")
        5. return RetryDecision(
               "deterministic", False,
               "unrecognized failure — surfacing to the model",
           )
    """
    # TODO
    raise NotImplementedError
