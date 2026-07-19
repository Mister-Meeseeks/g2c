"""Generation eval — open-ended scoring with a pluggable matcher.

Where multiple-choice eval works in log-prob space (closed set, every
option scored), generation eval works in TEXT space (open set, model
emits whatever it wants). The harness:

  1. For each example, calls a user-supplied `generate_fn(prompt)` to
     produce a string.
  2. Runs a matcher (`exact_match`, `normalized_match`, `numeric_match`,
     `contains_match`, or any user-supplied callable with the same
     signature) over (generated_text, references).
  3. Aggregates into an `EvalReport`.

The harness does NOT call into `g2c.sampling.generate` directly. The
`generate_fn` parameter is a user-supplied closure — typically:

    def generate_fn(prompt: str) -> str:
        prompt_ids = tokenizer.encode(prompt)
        out_ids = generate(model, torch.tensor(prompt_ids), max_new_tokens=64,
                           temperature=0.0, eos_id=eos_id)
        new_ids = out_ids[len(prompt_ids):]
        return tokenizer.decode(new_ids.tolist())

The decoupling is deliberate: it keeps the eval harness independent
of any particular model or sampling stack. You can swap in a function
that calls Ollama, MLX, an HTTP API, or your week-10 model. The
harness doesn't care.

Two pieces:

  * `score_generation_example(example, generate_fn, matcher) -> EvalResult`.
    Implemented — it's two lines (call generate_fn, run matcher).

  * `run_generation_eval(examples, generate_fn, matcher, *,
    task_name="generation") -> EvalReport`. Implemented — same
    aggregation pattern as the MC harness.

Generation eval doesn't compute ECE by default. Confidence on a
generated string requires re-scoring it under the model
(`continuation_logprob` on the generated text), which is a separate
operation. The harness leaves `confidence=None` and `ece=None`. If you
want calibrated generation eval, score the generated text after the
fact with `continuation_logprob` and post-process — see exercise 6.
"""
from __future__ import annotations

from collections.abc import Callable

from .data import EvalReport, EvalResult, GenerationExample

Matcher = Callable[[str, list[str]], bool]
GenerateFn = Callable[[str], str]


def score_generation_example(
    example: GenerationExample,
    generate_fn: GenerateFn,
    matcher: Matcher,
) -> EvalResult:
    """Generate a continuation of `example.prompt` and score it via `matcher`.

    Args:
        example: the prompt + references.
        generate_fn: callable taking a prompt string, returning the
            generated continuation (NOT including the prompt). The
            caller is responsible for any decoding / stripping of
            chat-template markers.
        matcher: callable taking (prediction, references) and
            returning bool. Use `exact_match`, `normalized_match`,
            `numeric_match`, `contains_match`, or a custom matcher.

    Returns:
        EvalResult with:
          * `correct`:   matcher(prediction, references)
          * `prediction`: the generated text (str)
          * `confidence`: None — see module docstring
          * `metadata`:  {'references': example.references,
                          'matcher': matcher.__name__}

    This function is implemented; the educational content is the
    matcher set in `match.py`.
    """
    prediction = generate_fn(example.prompt)
    correct = matcher(prediction, example.references)
    return EvalResult(
        correct=correct,
        prediction=prediction,
        confidence=None,
        metadata={
            "references": list(example.references),
            "matcher": getattr(matcher, "__name__", repr(matcher)),
        },
    )


def run_generation_eval(
    examples: list[GenerationExample],
    generate_fn: GenerateFn,
    matcher: Matcher,
    *,
    task_name: str = "generation",
) -> EvalReport:
    """Iterate `score_generation_example` over `examples` and aggregate.

    Args:
        examples: list of GenerationExample. Must be non-empty.
        generate_fn: as above.
        matcher:    as above.
        task_name:  label for the EvalReport.

    Returns:
        EvalReport. `mean_confidence` and `ece` are both `None` —
        generation eval doesn't expose a per-example confidence
        without an extra rescoring pass.
    """
    if len(examples) == 0:
        raise ValueError(
            "run_generation_eval needs at least one example"
        )

    results: list[EvalResult] = [
        score_generation_example(ex, generate_fn, matcher) for ex in examples
    ]
    n = len(results)
    accuracy = sum(1 for r in results if r.correct) / n
    return EvalReport(
        task_name=task_name,
        n=n,
        accuracy=accuracy,
        mean_confidence=None,
        ece=None,
        results=results,
    )
