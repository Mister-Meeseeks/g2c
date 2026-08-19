"""Verifiable rewards — the graders that replace labels.

A verifier is a plain function `(task, completion_text) -> float`. No
learned reward model, no human in the loop: the scorer is twenty lines
you can read, which keeps this module's failures legible. When training
goes weird, read the verifier before blaming the algorithm.

`verify_arithmetic_sloppy` is included ON PURPOSE: it is the
deliberately gap-ridden reward for the lesson's "break your own reward"
exercise. Audit its claimed score against the intended verifier and
inspect what the finite run actually discovers.

All provided plumbing — nothing here is scaffolded.
"""
from __future__ import annotations

import json
import random
import re

Task = dict[str, object]

_INT_RE = re.compile(r"-?\d+")


def arithmetic_task(rng: random.Random, *, digits: int = 2) -> Task:
    """One addition problem, e.g. {'prompt': '23+58=', 'answer': 81}."""
    lo, hi = 10 ** (digits - 1), 10**digits - 1
    a, b = rng.randint(lo, hi), rng.randint(lo, hi)
    return {"prompt": f"{a}+{b}=", "answer": a + b}


def arithmetic_choice_task(rng: random.Random, *, digits: int = 2) -> Task:
    """Two-option addition task with enough initial reward contrast for RL.

    Free-response two-digit addition is almost always wrong under the course's
    small BaseLM, so groups are usually all-zero and teach nothing. Two numeric
    options preserve a checkable arithmetic decision while giving a fresh
    policy enough successes and failures for group-relative updates.
    """
    base = arithmetic_task(rng, digits=digits)
    answer = int(base["answer"])
    offset = rng.choice([-9, -7, -3, -1, 1, 3, 7, 9])
    choices = [answer, answer + offset]
    rng.shuffle(choices)
    return {
        "prompt": (
            f'{base["prompt"]}? Options: {choices[0]}, {choices[1]}. '
            "Output only the correct number: "
        ),
        "answer": answer,
        "choices": choices,
    }


def format_task(rng: random.Random) -> Task:
    """One discoverable JSON-completion task.

    BaseLM is not instruction-tuned, so asking it to emit a complete JSON
    object from a bare instruction makes the binary reward almost always
    zero. Supplying the opening prefix keeps the format-only check narrow while
    making both successful and failed completions likely enough for a small
    group to carry contrast.
    """
    task = arithmetic_task(rng)
    return {
        "prompt": (
            f'Complete this JSON object with an integer answer for '
            f'{task["prompt"]} {{"answer":'
        ),
        "completion_prefix": '{"answer":',
        "answer": task["answer"],
    }


def verify_arithmetic(task: Task, completion: str) -> float:
    """1.0 iff the LAST integer in the completion equals the true answer.

    "Last" rather than "first" so a model that reasons before answering
    ("23+58 is 70 plus 11, so 81") is scored on its conclusion.
    """
    matches = _INT_RE.findall(completion)
    if not matches:
        return 0.0
    return 1.0 if int(matches[-1]) == task["answer"] else 0.0


def verify_arithmetic_sloppy(task: Task, completion: str) -> float:
    """The DELIBERATELY BROKEN reward for the reward-hacking exercise.

    Rewards any completion in which ANY emitted integer matches the
    true answer. The gap: a model that sprays many numbers gets
    rewarded for accidentally including the right one. Watch for
    completions that degenerate into digit salad while this "reward"
    climbs — then compare against `verify_arithmetic`.
    """
    return (
        1.0
        if any(int(m) == task["answer"] for m in _INT_RE.findall(completion))
        else 0.0
    )


def verify_format(task: Task, completion: str) -> float:
    """1.0 iff prompt prefix + completion contains an ``answer`` object.

    Format-only: the VALUE and any text after the first complete object
    are not checked. This is the smoke-test reward. `format_task` supplies
    an opening JSON prefix in the prompt so a small sampled group has a
    realistic chance of
    containing both valid and invalid completions. Hand-written tasks without
    `completion_prefix` retain the old whole-completion behavior.
    """
    candidate = str(task.get("completion_prefix", "")) + completion
    start = candidate.find("{")
    end = candidate.find("}", start + 1)
    if start == -1 or end <= start:
        return 0.0
    try:
        obj = json.loads(candidate[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return 0.0
    return 1.0 if isinstance(obj, dict) and "answer" in obj else 0.0
