"""Eval — a tiny regression-gating harness for the capstone assistant.

The capstone's "did this still work?" gate. The pattern:

    cases = [EvalCase(...), EvalCase(...), ...]
    report = run_evaluation(assistant, cases)
    assert report.pass_rate >= 0.8

The eval harness is intentionally minimal — three checks per case:

  * `expected_substring` — the final answer must contain this string
    (case-insensitive). Captures factual answers ("the capital is
    Paris"); fails noisy / wrong / "I don't know" responses.

  * `expected_tool` — at least one tool call during the run must have
    used this tool. Captures behavioral expectations ("the model
    should have called the calculator on 2+2"). `None` means no
    tool-call requirement.

  * Implicit: the run must have produced a `final_answer` (i.e.,
    `stopped_reason == "final_answer"`). A run that timed out or hit
    a duplicate-action loop is treated as a failure even if the
    `expected_substring` happens to appear in some scratchpad step.

This is a regression gate, not a benchmark. The point is "is the
assistant still doing the basic things right after I changed
something" — not "how good is the assistant on a research-grade
eval set." For richer evals, see Module 15's harness; this one is
the smoke test.

Fully implemented. No scaffolds.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .assistant import Assistant, AssistantTurn


@dataclass(frozen=True)
class EvalCase:
    """One eval case for the regression gate.

    Attributes:
        name: a short identifier for the case. Shown in failure
            output so the user knows which case broke.
        question: the user message to send to the assistant.
        expected_substring: a string that should appear in the
            assistant's final answer. Case-insensitive substring
            match. `None` skips the answer check.
        expected_tool: name of a tool that should have been called
            during the agent run. `None` skips the tool check.
        rag: per-case override of `rag_enabled`. `None` defers to
            the assistant's config; `True` / `False` forces RAG on
            / off for this case.
    """

    name: str
    question: str
    expected_substring: str | None = None
    expected_tool: str | None = None
    rag: bool | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("EvalCase.name must be a non-empty str")
        if not isinstance(self.question, str) or not self.question:
            raise ValueError("EvalCase.question must be a non-empty str")
        if self.expected_substring is not None and not isinstance(
            self.expected_substring, str
        ):
            raise TypeError(
                "EvalCase.expected_substring must be a str or None"
            )
        if self.expected_tool is not None and not isinstance(
            self.expected_tool, str
        ):
            raise TypeError(
                "EvalCase.expected_tool must be a str or None"
            )


@dataclass(frozen=True)
class EvalCaseResult:
    """The outcome of running one `EvalCase`.

    Attributes:
        case: the original case (for output / debugging).
        passed: True iff every applicable check passed.
        final_answer: the assistant's response, or `None` if the
            run didn't produce one.
        failure_reason: a short string describing why the case
            failed, or `None` on success. The first failing check
            wins; the runner stops checking after the first failure.
        turn: the full `AssistantTurn` from the run, for callers
            who want to inspect the agent steps after the fact.
    """

    case: EvalCase
    passed: bool
    final_answer: str | None
    failure_reason: str | None
    turn: AssistantTurn


@dataclass
class EvalReport:
    """Roll-up of an eval suite's results.

    Attributes:
        results: every case's `EvalCaseResult`, in the order they
            ran.

    Computed properties:
        n_total, n_passed, n_failed, pass_rate (0.0–1.0),
        failures (just the failed results).
    """

    results: list[EvalCaseResult] = field(default_factory=list)

    @property
    def n_total(self) -> int:
        return len(self.results)

    @property
    def n_passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def n_failed(self) -> int:
        return self.n_total - self.n_passed

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return self.n_passed / self.n_total

    @property
    def failures(self) -> list[EvalCaseResult]:
        return [r for r in self.results if not r.passed]

    def summary(self) -> str:
        """A one-line summary string."""
        return (
            f"EvalReport: {self.n_passed}/{self.n_total} passed "
            f"({self.pass_rate:.1%})"
        )


def _check_case(case: EvalCase, turn: AssistantTurn) -> tuple[bool, str | None]:
    """Run the checks against a completed turn. Returns
    `(passed, failure_reason)`. First failing check wins.
    """
    # Implicit check: the run must have produced a final answer.
    if turn.final_answer is None:
        reason = (
            f"no final_answer (stopped: "
            f"{turn.agent_run.stopped_reason})"
        )
        return False, reason

    # Substring check.
    if case.expected_substring is not None:
        needle = case.expected_substring.lower()
        haystack = turn.final_answer.lower()
        if needle not in haystack:
            reason = (
                f"final_answer did not contain expected substring "
                f"{case.expected_substring!r}"
            )
            return False, reason

    # Tool-call check.
    if case.expected_tool is not None:
        called_tools = {
            step.action.tool
            for step in turn.agent_run.steps
            if step.action is not None
        }
        if case.expected_tool not in called_tools:
            reason = (
                f"expected tool {case.expected_tool!r} not called "
                f"(called: {sorted(called_tools)})"
            )
            return False, reason

    return True, None


def run_evaluation(
    assistant: Assistant,
    cases: Iterable[EvalCase],
    *,
    reset_each: bool = True,
) -> EvalReport:
    """Run each case through the assistant and roll up the results.

    Args:
        assistant: an `Assistant` instance.
        cases: iterable of `EvalCase`. Order is preserved in the
            report.
        reset_each: if True (default), `assistant.reset()` is called
            BEFORE each case so cases can't contaminate one another.
            Set False if you want cases to share conversation state
            (e.g., a multi-turn eval suite).

    Returns:
        `EvalReport` with one `EvalCaseResult` per case.

    Each case's flow:
        1. (Optionally) reset the assistant.
        2. Call `assistant.chat(case.question, use_rag=case.rag)`.
        3. Apply the configured checks; record pass/fail.

    The runner doesn't catch exceptions from `chat` — if the
    assistant raises (bad config, broken retriever), the eval
    crashes. That's intentional: the eval is a regression gate,
    not a fault-tolerant runner. Fix the config; re-run.
    """
    cases_list = list(cases)
    results: list[EvalCaseResult] = []
    for case in cases_list:
        if reset_each:
            assistant.reset()
        turn = assistant.chat(case.question, use_rag=case.rag)
        passed, reason = _check_case(case, turn)
        results.append(
            EvalCaseResult(
                case=case,
                passed=passed,
                final_answer=turn.final_answer,
                failure_reason=reason,
                turn=turn,
            )
        )
    return EvalReport(results=results)
