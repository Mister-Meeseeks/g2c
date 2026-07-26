"""Synthetic instruction generation — Self-Instruct in miniature (Module 16B).

The modern data recipe: a strong teacher model writes training data
for a small student model. This file is the generation half of that
loop, built on the `Backend` interface from Module 16 — anything with
`complete(prompt, ...) -> InferenceResult` can be the teacher, which
in this course means `ProdLM`.

Everything here is implemented for you. Prompt text is given, not
discovered (writing prompts isn't the concept under study), and the
orchestration is plumbing around two backend calls per accepted pair:

    propose_instructions   one call -> a numbered list of candidates
    generate_response      one call -> the answer to one instruction

The interesting decisions live in `g2c/synth/filter.py` (which you
implement) and in `synthesize_dataset`'s funnel: propose → shape-check
→ dedup → answer → shape-check again. The funnel *counts* — how many
candidates die at each gate — are the module's first deliverable,
because they are what data quality looks like as numbers.
"""
from __future__ import annotations

import random
import re
from collections.abc import Callable

from .filter import dedupe_pairs, ngram_overlap, validate_pair  # noqa: F401

INSTRUCTION_PROMPT_TEMPLATE = """\
You are helping build a dataset of instruction-response pairs for \
training a small assistant model.

Here are {k} example instructions from the dataset:

{examples}

Write {count} NEW instructions in the same style: short, \
self-contained, and answerable in one or two sentences without any \
outside context. Vary the topics and the task types (questions, \
transformations, short generation). Do not repeat or trivially \
rephrase the examples.

Reply with a numbered list only, one instruction per line, no answers.
"""

RESPONSE_PROMPT_TEMPLATE = """\
You are writing training data for a small assistant model. Answer the \
instruction below directly and concisely, in one or two sentences, \
with no preamble and no follow-up questions.

Instruction: {instruction}

Answer:"""

_NUMBERED_LINE_RE = re.compile(r"^\s*\d+\s*[.)]\s*(.+?)\s*$")


def build_instruction_prompt(
    example_instructions: list[str], *, count: int = 8
) -> str:
    """Render the few-shot instruction-proposal prompt."""
    examples = "\n".join(
        f"{i + 1}. {text}" for i, text in enumerate(example_instructions)
    )
    return INSTRUCTION_PROMPT_TEMPLATE.format(
        k=len(example_instructions), examples=examples, count=count
    )


def build_response_prompt(instruction: str) -> str:
    """Render the answer-generation prompt for one instruction."""
    return RESPONSE_PROMPT_TEMPLATE.format(instruction=instruction)


def parse_numbered_list(text: str) -> list[str]:
    """Extract `1. ...`-style lines from a completion.

    Lines that don't look like numbered items (preamble, blank lines,
    the teacher's stray commentary) are dropped silently — the same
    permissive-parser posture as Module 18's tool-call parser: the
    teacher isn't punished for one odd line, and the funnel counts
    what survived.
    """
    items: list[str] = []
    for line in text.splitlines():
        match = _NUMBERED_LINE_RE.match(line)
        if match:
            item = match.group(1).strip().strip('"').strip()
            if item:
                items.append(item)
    return items


def propose_instructions(
    backend,
    seed_pool: list[str],
    *,
    count: int = 8,
    k: int = 6,
    rng: random.Random,
    temperature: float = 0.9,
    max_new_tokens: int = 400,
) -> list[str]:
    """One proposal round: sample `k` examples from the pool, ask the
    teacher for `count` new instructions, parse the list.

    The examples are re-sampled every round (Self-Instruct's trick):
    a fixed few-shot set anchors the teacher to one neighborhood, and
    the dedup gate then rejects almost everything after round two.
    """
    shown = rng.sample(seed_pool, min(k, len(seed_pool)))
    prompt = build_instruction_prompt(shown, count=count)
    result = backend.complete(
        prompt, max_new_tokens=max_new_tokens, temperature=temperature
    )
    return parse_numbered_list(result.completion)


def generate_response(
    backend,
    instruction: str,
    *,
    temperature: float = 0.3,
    max_new_tokens: int = 120,
) -> str:
    """Ask the teacher to answer one instruction, concisely.

    Cooler than the proposal call on purpose: we want *diverse
    instructions* but *reliable answers* — temperature is doing
    different jobs at the two stages.
    """
    prompt = build_response_prompt(instruction)
    result = backend.complete(
        prompt, max_new_tokens=max_new_tokens, temperature=temperature
    )
    return result.completion.strip()


def synthesize_dataset(
    backend,
    seeds: list[dict],
    *,
    target: int = 150,
    batch_count: int = 8,
    k: int = 6,
    threshold: float = 0.7,
    max_rounds: int | None = None,
    seed: int = 0,
    progress: Callable[[str], None] | None = None,
) -> tuple[list[dict], dict[str, int]]:
    """The Self-Instruct loop: grow a synthetic dataset from your seeds.

    Round structure:

        1. propose a batch of candidate instructions (few-shot from
           the seed + accepted pool)
        2. drop empty/absurd ones (shape gate on the instruction)
        3. dedup against seeds + everything accepted so far
        4. for each survivor, generate a response
        5. shape-gate the finished pair; accept or reject

    Stops at `target` accepted pairs, or after `max_rounds` (default:
    enough rounds to hit the target four times over — a teacher stuck
    rephrasing itself should exhaust the budget, not loop forever).

    Returns:
        `(pairs, funnel)` where `funnel` counts every stage:
        proposed / parsed / bad_instruction / duplicate / bad_pair /
        accepted / rounds. Print it. The funnel is the module's first
        deliverable — data quality, as numbers.
    """
    rng = random.Random(seed)
    seed_instructions = [pair["user"] for pair in seeds]
    accepted: list[dict] = []
    funnel = {
        "proposed": 0,
        "bad_instruction": 0,
        "duplicate": 0,
        "bad_pair": 0,
        "accepted": 0,
        "rounds": 0,
    }
    if max_rounds is None:
        max_rounds = max(8, (4 * target) // max(batch_count, 1))

    while len(accepted) < target and funnel["rounds"] < max_rounds:
        funnel["rounds"] += 1
        pool = seed_instructions + [pair["user"] for pair in accepted]
        candidates = propose_instructions(
            backend, pool, count=batch_count, k=k, rng=rng
        )
        funnel["proposed"] += len(candidates)

        shaped: list[dict] = []
        for instruction in candidates:
            # Instruction-side shape gate: reuse validate_pair with a
            # placeholder answer so one gate definition serves both.
            if validate_pair({"user": instruction, "assistant": "ok"}):
                funnel["bad_instruction"] += 1
            else:
                shaped.append({"user": instruction, "assistant": ""})

        novel = dedupe_pairs(shaped, threshold=threshold, against=pool)
        funnel["duplicate"] += len(shaped) - len(novel)

        for candidate in novel:
            if len(accepted) >= target:
                break
            candidate["assistant"] = generate_response(
                backend, candidate["user"]
            )
            reasons = validate_pair(candidate)
            if reasons:
                funnel["bad_pair"] += 1
                continue
            accepted.append(candidate)
            funnel["accepted"] += 1
        if progress is not None:
            progress(
                f"round {funnel['rounds']}: {len(accepted)}/{target} accepted"
            )

    return accepted, funnel
