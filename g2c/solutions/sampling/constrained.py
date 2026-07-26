# Hand-maintained solutions mirror — canonical, lives in `main` (see AGENTS.md).
"""Solutions for g2c.sampling.constrained pedagogical scaffolds.

Patched onto the scaffold targets by g2c.solutions.apply().
"""
from __future__ import annotations

import torch

from g2c.sampling.constrained import (  # noqa: F401 (target-module context)
    JsonPrefixAutomaton,
)


def allowed_token_mask(
    automaton: JsonPrefixAutomaton,
    state: tuple[str, int],
    pieces: list[str],
) -> torch.Tensor:
    """The decode-and-check sweep: which vocabulary entries keep the
    output a valid prefix of the grammar?

    Args:
        automaton: the grammar.
        state: the automaton state for the text generated so far.
        pieces: `pieces[i]` is token id `i` decoded to text (from
            `vocab_pieces`).

    Returns:
        `(len(pieces),)` bool tensor. `True` at `i` means appending
        `pieces[i]` is grammatical.

    Recipe:

        1. allowed = torch.zeros(len(pieces), dtype=torch.bool)

        2. For each piece:

               allowed[i] = bool(piece) and (
                   automaton.advance(state, piece) is not None
               )

           Two conditions, both load-bearing:

           * `bool(piece)` — a token that decodes to the EMPTY STRING
             (some special tokens do) advances nothing. Allow it and
             the sampler can emit it forever without the automaton
             ever moving: an infinite loop that produces no text.
           * `advance(...) is not None` — the piece, fed character by
             character, never left the grammar. A multi-character
             piece is checked as a unit: `'": "'` is one perfectly
             legal hop from the end of a key.

        3. return allowed

    Note what happens on a complete state: `done` accepts no
    characters, every advance returns None, and the mask comes back
    all-False. That is not an error — it is the automaton saying
    "stop," and the generation loop treats it exactly that way.
    """
    allowed = torch.zeros(len(pieces), dtype=torch.bool)
    for i, piece in enumerate(pieces):
        allowed[i] = bool(piece) and (
            automaton.advance(state, piece) is not None
        )
    return allowed


@torch.no_grad()
def generate_json(
    model,
    prompt_ids: torch.Tensor,
    pieces: list[str],
    *,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
    automaton: JsonPrefixAutomaton | None = None,
    mask_cache: dict | None = None,
) -> torch.Tensor:
    """Module 11's generation loop with the grammar mask spliced in.

    The continuation — not the prompt — is constrained: the automaton
    starts at `initial()` regardless of what the prompt says, and every
    sampled token must keep the generated text a valid prefix of the
    tool-call JSON subset. Generation stops when the root object
    closes.

    (See the scaffold in g2c/sampling/constrained.py for the full
    argument documentation and recipe.)
    """
    if prompt_ids.dim() != 1 or prompt_ids.numel() == 0:
        raise ValueError("prompt_ids must be a non-empty 1-D tensor.")
    if temperature < 0:
        raise ValueError(f"temperature must be >= 0, got {temperature}.")
    if automaton is None:
        automaton = JsonPrefixAutomaton()
    if mask_cache is None:
        mask_cache = {}
    state = automaton.initial()
    greedy = temperature == 0.0
    full_ids = prompt_ids.detach().cpu().clone()
    device = getattr(model, "device", torch.device("cpu"))

    for _ in range(max_new_tokens):
        ctx = full_ids[-model.max_seq_len :]
        logits = model(ctx.to(device).unsqueeze(0))
        last_logits = logits[:, -1, :].cpu()

        if last_logits.shape[-1] != len(pieces):
            raise ValueError(
                f"model vocab size {last_logits.shape[-1]} != len(pieces) "
                f"{len(pieces)} -- wrong tokenizer for this model?"
            )

        # THE CONSTRAINT -- before any warper. Top-k after the mask
        # operates within the grammatical set; before it, all k
        # survivors could be ungrammatical and softmax would NaN.
        # Memoized by state: the mask is a function of the automaton
        # state alone, so the O(V) sweep runs once per distinct state
        # instead of once per step.
        allowed = mask_cache.get(state)
        if allowed is None:
            allowed = allowed_token_mask(automaton, state, pieces)
            mask_cache[state] = allowed
        if not allowed.any():
            break  # the automaton says stop (complete state)
        last_logits = last_logits.clone()
        last_logits[:, ~allowed] = float("-inf")

        if greedy:
            next_id = last_logits.argmax(dim=-1)
        else:
            # No repetition penalty on purpose: JSON must repeat its
            # delimiters.
            last_logits = apply_temperature(last_logits, temperature)
            if top_k is not None:
                last_logits = top_k_filter(last_logits, top_k)
            if top_p is not None:
                last_logits = top_p_filter(last_logits, top_p)
            probs = torch.softmax(last_logits, dim=-1)
            next_id = torch.multinomial(
                probs, num_samples=1, generator=generator
            ).squeeze(-1)

        full_ids = torch.cat([full_ids, next_id], dim=0)
        state = automaton.advance(state, pieces[next_id.item()])

        if automaton.is_complete(state):
            break

    return full_ids
