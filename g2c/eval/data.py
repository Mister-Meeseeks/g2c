"""Eval-task containers: input examples and the result/report records.

Two task types are first-class in this module:

  * **Multiple choice.** A prompt plus a list of textual continuations,
    one of which is correct. The model picks the continuation it
    assigns highest sequence-log-probability to. Closed-set, fast,
    calibration-friendly.

  * **Generation.** A prompt plus a list of acceptable reference
    answers. The model generates freely; a string-matcher decides if
    the generated text matches any reference. Open-set, slower, more
    realistic.

Every other "task" in the syllabus (factual QA, arithmetic,
instruction-following) reduces to ONE of these two — the only thing
that changes is the matcher. Factual QA is generation + exact_match;
arithmetic is generation + numeric_match; instruction-following is
generation + a regex/keyword matcher. The harness is the same.

The four dataclasses here are pure containers — the boilerplate is
implemented in full; the math lives in `match.py`, `logprob.py`,
`multiple_choice.py`, `generation.py`, and `calibration.py`. The only
logic in this file is constructor validation: catch obviously broken
inputs at example-construction time so they don't surface as
"mysterious zero-accuracy result" three function calls later.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class MultipleChoiceExample:
    """One closed-set multiple-choice question.

    Attributes:
        prompt: the question text. Whatever convention you use for
            chat formatting (e.g. ChatML markers from Module 13) goes
            here. The eval harness does NOT re-render it; the caller
            renders the chat template and passes the rendered string.
        choices: the candidate continuations. AT LEAST 2. Each choice
            is appended to the prompt (with no separator) before the
            model is asked to score it. So if the prompt ends in
            "<|assistant|>\\n" and a choice is "Madrid.<|end|>", the
            scorer evaluates the sequence-log-probability of the
            "Madrid.<|end|>" tokens given the prompt.
        answer_idx: the index of the correct choice. `0` ≤ answer_idx
            < `len(choices)`.

    Frozen — examples are values, not handles, and equality works
    out of the box for test fixtures. (NamedTuple would also work but
    `dataclass(frozen=True)` is more discoverable and accepts
    `__post_init__`-style validation.)
    """

    prompt: str
    choices: list[str]
    answer_idx: int

    def __post_init__(self) -> None:
        if not isinstance(self.prompt, str):
            raise TypeError(f"prompt must be str, got {type(self.prompt).__name__}")
        if len(self.choices) < 2:
            raise ValueError(
                f"MultipleChoiceExample needs at least 2 choices, got {len(self.choices)}"
            )
        if not (0 <= self.answer_idx < len(self.choices)):
            raise ValueError(
                f"answer_idx={self.answer_idx} is out of range "
                f"for {len(self.choices)} choices"
            )


@dataclass(frozen=True)
class GenerationExample:
    """One open-set generation question.

    Attributes:
        prompt: the question text. As with `MultipleChoiceExample`,
            the caller is responsible for any chat-template rendering
            before the prompt reaches the harness.
        references: the list of acceptable answers. AT LEAST 1. The
            generation harness asks the model to produce a continuation
            of `prompt` and then runs a matcher over (prediction,
            references). A match against any reference counts as
            correct.

    Why a list of references? Many real questions admit multiple
    surface forms: "Madrid", "madrid", "Madrid.", "the capital is
    Madrid". Hand-authoring all the surface variants you accept is
    less error-prone than writing one regex per question. Use the
    matcher to handle case/punctuation; use the references list to
    handle genuinely different acceptable answers.
    """

    prompt: str
    references: list[str]

    def __post_init__(self) -> None:
        if not isinstance(self.prompt, str):
            raise TypeError(f"prompt must be str, got {type(self.prompt).__name__}")
        if len(self.references) < 1:
            raise ValueError(
                "GenerationExample needs at least 1 reference answer"
            )


@dataclass
class EvalResult:
    """The harness's per-example record.

    Attributes:
        correct: did the model get this example right? Always populated.
        prediction: what the model said. For multiple choice, this is
            the chosen choice INDEX (an int). For generation, this is
            the GENERATED TEXT (a str). The type difference is
            deliberate — it forces callers to know which task they're
            looking at.
        confidence: the model's confidence in its prediction, on
            [0, 1]. For multiple choice, this is the softmax probability
            of the selected option (over all options). For generation,
            this is currently unset (`None`) — the harness can be
            extended to re-score generated text and compute a
            length-normalized probability if calibration on generation
            is desired.
        metadata: per-task auxiliary fields. Multiple choice puts
            per-option log-probs here for diagnostic plotting;
            generation puts the matcher name and references list.

    Equality works for fixture comparison; the field is mutable so
    callers can mutate `metadata` after construction (e.g. add a
    `latency_ms` field after measurement).
    """

    correct: bool
    prediction: int | str
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalReport:
    """The harness's aggregated output.

    Attributes:
        task_name: a short label. Used purely for printing and notebook
            organization; the harness doesn't dispatch on it.
        n: number of examples scored. Equal to `len(results)`.
        accuracy: fraction of correct examples on [0, 1].
        mean_confidence: mean of `r.confidence` across results that
            have it (`None` is filtered out). For a perfectly
            calibrated model, mean_confidence should equal accuracy
            within sampling noise.
        ece: expected calibration error (Naeini et al., 2015). On
            [0, 1], lower is better-calibrated; 0 is perfectly
            calibrated. May be `None` if confidences aren't available
            (e.g. on a generation task) or if `compute_ece=False`
            was passed to the harness.
        results: the per-example records, in evaluation order.

    Two invariants worth restating:
      * `accuracy` and `mean_confidence` are NOT the same thing. A
        model that always says "I'm 99% sure" but is right half the
        time has accuracy=0.5, mean_confidence=0.99. The gap is what
        ECE measures.
      * `n == len(results)` always — no separate "errored / skipped"
        bookkeeping. If the harness can't score an example (e.g. the
        model returned NaN), it raises rather than silently dropping.
    """

    task_name: str
    n: int
    accuracy: float
    mean_confidence: float | None
    ece: float | None
    results: list[EvalResult]

    def __repr__(self) -> str:
        ece_str = f"{self.ece:.4f}" if self.ece is not None else "—"
        conf_str = (
            f"{self.mean_confidence:.4f}" if self.mean_confidence is not None else "—"
        )
        return (
            f"EvalReport(task={self.task_name!r}, n={self.n}, "
            f"accuracy={self.accuracy:.4f}, mean_confidence={conf_str}, "
            f"ece={ece_str})"
        )
