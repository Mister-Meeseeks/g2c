"""Constrained decoding — sampling that cannot emit invalid output.

Built here in `g2c/sampling/` because you own this sampler; taught in
Module 18, where you've just watched a model emit tool calls that
almost parse. The parser's answer to a malformed block is to skip it
and hope the model retries. This file is the stronger answer: make
malformed output *unsampleable*.

The mechanism is one move. At every decoding step you already hold the
full next-token distribution — that's what `generate()` samples from.
A grammar tells you which tokens would keep the output well-formed.
Set every other logit to `-inf` before sampling, and invalid
continuations don't become unlikely — they become impossible:

    last_logits[~allowed] = -inf        # the entire trick

Everything else in this file is bookkeeping around that line: a
machine that answers "which continuations are legal?" (the automaton),
a sweep that asks it once per vocabulary entry (the mask), and the
Module 11 loop with the mask spliced in (the constrained generator).

The division of labor is the lesson:

    the GRAMMAR carries the syntax — parse rate goes to 100%
    the MODEL carries the semantics — content quality doesn't move

A weak model under a JSON grammar emits perfectly parseable garbage.
That is not a failure of the method; it is exactly the contract. It is
also why `ProdLM`'s `format: "json"` flag must live server-side: the
mask needs the logits, and an API that only returns text has already
thrown them away.

One deliberate omission: `repetition_penalty` is absent from the
constrained loop. It penalizes tokens that already appeared — and
well-formed JSON *requires* repeating `"`, `:`, and `}`. A warper
designed to fight prose loops actively fights the grammar here.

Production systems (llguidance, Outlines, Ollama's `format`) compile
the grammar into a token-level automaton ahead of time instead of
checking every token with string operations each step. Same idea,
heavy engineering; we pay the O(V) sweep for legibility.
"""
from __future__ import annotations

import torch

from .temperature import apply_temperature  # noqa: F401 (for the student implementation)
from .top_k import top_k_filter  # noqa: F401 (for the student implementation)
from .top_p import top_p_filter  # noqa: F401 (for the student implementation)

# ----------------------------------------------------------------------
# The grammar: a prefix acceptor for a JSON subset
# ----------------------------------------------------------------------

_WS = " \t\n\r"
_ESCAPABLE = '"\\/bfnrt'
_DIGITS = "0123456789"
_MAX_WS_RUN = 3

# The automaton state is an immutable (mode, depth, ws_run) tuple.
_State = tuple[str, int, int]


class JsonPrefixAutomaton:
    """Prefix acceptor for the tool-call JSON subset.

    Grammar: the root is an object; keys are strings; values are
    strings, numbers (JSON's rules: no leading zeros, and we omit
    exponents), or nested objects. String escapes are the
    single-character ones (`\\"` `\\\\` `\\/` `\\b` `\\f` `\\n` `\\r`
    `\\t` — no `\\uXXXX`), and nothing at all is allowed after the
    root object closes — so a constrained generator stops crisply.
    Arrays, booleans, and `null` are omitted on purpose: the course's
    tool calls (`{"name": ..., "arguments": {...}}`) never need them,
    and every production is one you can read.

    One deliberate tightening over real JSON: whitespace between
    tokens is **rationed** — at most three consecutive characters.
    Unbounded whitespace is perfectly grammatical JSON, and that is
    exactly the problem: a prose-trained model keeps much of its
    probability mass on spaces and newlines, so under an unrationed
    grammar it hides in whitespace forever and burns the whole token
    budget without ever committing to `{`. The ration means the mask
    forces a structural character every few tokens. (Production
    grammars do the same — llama.cpp's JSON grammar bounds its `ws`
    rule for this exact reason.)

    Implemented for you — a hand-rolled pushdown acceptor is classic
    CS, not an LLM concept. But *read it*: the states are named, and
    "which characters may come next" is the entire interface the
    scaffolds consume.

    The state is an immutable `(mode, depth, ws_run)` tuple — `mode`
    names the parse position, `depth` counts open objects, `ws_run`
    counts the current whitespace run. Treat it as opaque; the
    interface is `initial` / `advance` / `is_complete`. Immutability
    is what makes the per-token sweep cheap: advancing a *copy* of the
    state down each candidate piece never disturbs the real one.

        state = automaton.initial()
        state = automaton.advance(state, '{"name"')   # -> new state
        automaton.advance(state, "}")                 # -> None (rejected)
        automaton.is_complete(state)                  # root object closed?
    """

    # Inside strings, a space is content, not a separator — the ration
    # and the central whitespace handling below don't apply there.
    _STRING_MODES = frozenset({"key", "key_esc", "str", "str_esc"})

    def initial(self) -> _State:
        """The state before any characters: expecting the root `{`."""
        return ("pre_root", 0, 0)

    def is_complete(self, state: _State) -> bool:
        """True when the root object has closed. No state accepts
        further characters after this."""
        return state[0] == "done"

    def advance(self, state: _State | None, text: str) -> _State | None:
        """Feed `text` one character at a time. Returns the new state,
        or None the moment a character makes the prefix invalid.
        `advance(None, ...)` is None, so rejections propagate."""
        for ch in text:
            if state is None:
                return None
            state = self._step(state, ch)
        return state

    def _step(self, state: _State, ch: str) -> _State | None:
        mode, depth, ws_run = state

        # Numbers have no terminator character: they end when the next
        # structural character arrives. Close the number, then let that
        # character be handled by the logic below.
        if mode in ("num_int", "num_zero") and ch not in _DIGITS and ch != ".":
            mode = "post_value"
        elif mode == "num_frac" and ch not in _DIGITS:
            mode = "post_value"

        # Between-token whitespace, handled once and RATIONED (see the
        # class docstring). Every non-whitespace transition below
        # resets the run to 0.
        if ch in _WS and mode not in self._STRING_MODES:
            if mode == "done" or ws_run >= _MAX_WS_RUN:
                return None
            return (mode, depth, ws_run + 1)

        if mode == "pre_root":
            if ch == "{":
                return ("obj_open", 1, 0)
            return None

        if mode == "obj_open":  # just after '{': first key, or empty object
            if ch == '"':
                return ("key", depth, 0)
            if ch == "}":
                return self._close_object(depth)
            return None

        if mode == "key":
            if ch == '"':
                return ("post_key", depth, 0)
            if ch == "\\":
                return ("key_esc", depth, 0)
            if ord(ch) >= 0x20:
                return (mode, depth, 0)
            return None  # raw control characters are invalid JSON

        if mode == "key_esc":
            if ch in _ESCAPABLE:
                return ("key", depth, 0)
            return None

        if mode == "post_key":  # between key and ':'
            if ch == ":":
                return ("pre_value", depth, 0)
            return None

        if mode == "pre_value":
            if ch == '"':
                return ("str", depth, 0)
            if ch == "{":
                return ("obj_open", depth + 1, 0)
            if ch == "-":
                return ("num_neg", depth, 0)
            if ch == "0":
                # JSON forbids leading zeros: after 0, only '.' or the
                # end of the number. "01" must be rejected, or the
                # "constrained output always parses" guarantee breaks.
                return ("num_zero", depth, 0)
            if ch in _DIGITS:
                return ("num_int", depth, 0)
            return None

        if mode == "str":
            if ch == '"':
                return ("post_value", depth, 0)
            if ch == "\\":
                return ("str_esc", depth, 0)
            if ord(ch) >= 0x20:
                return (mode, depth, 0)
            return None

        if mode == "str_esc":
            if ch in _ESCAPABLE:
                return ("str", depth, 0)
            return None

        if mode == "num_neg":  # a bare '-' needs at least one digit
            if ch == "0":
                return ("num_zero", depth, 0)
            if ch in _DIGITS:
                return ("num_int", depth, 0)
            return None

        if mode == "num_zero":  # after a leading 0: '.' or nothing
            if ch == ".":
                return ("num_frac_first", depth, 0)
            return None  # "01" etc. — rejected by the reclassify above

        if mode == "num_int":
            if ch in _DIGITS:
                return (mode, depth, 0)
            if ch == ".":
                return ("num_frac_first", depth, 0)
            return None  # unreachable: handled by the reclassify above

        if mode == "num_frac_first":  # '.' needs at least one digit
            if ch in _DIGITS:
                return ("num_frac", depth, 0)
            return None

        if mode == "num_frac":
            if ch in _DIGITS:
                return (mode, depth, 0)
            return None  # unreachable: handled by the reclassify above

        if mode == "post_value":  # a value just closed: ',' or '}'
            if ch == ",":
                return ("obj_comma", depth, 0)
            if ch == "}":
                return self._close_object(depth)
            return None

        if mode == "obj_comma":  # after ',': the next key (no trailing comma)
            if ch == '"':
                return ("key", depth, 0)
            return None

        return None  # mode == "done": nothing is ever allowed again

    @staticmethod
    def _close_object(depth: int) -> _State:
        depth -= 1
        return ("done", 0, 0) if depth == 0 else ("post_value", depth, 0)


# ----------------------------------------------------------------------
# The vocabulary as text
# ----------------------------------------------------------------------


def vocab_pieces(tokenizer, vocab_size: int) -> list[str]:
    """Decode every token id once, so the per-step sweep is pure string
    work. Implemented for you.

    Call this ONCE and reuse the list — it is the only tokenizer work
    constrained decoding needs, and at BaseLM's 49k vocabulary it is
    the slow part.

    Assumes piecewise decoding concatenates to sequence decoding,
    which holds for the course BPE and GPT-2-style byte BPEs (BaseLM
    included). SentencePiece models with context-dependent decoding
    would need more care.
    """
    return [tokenizer.decode([i]) for i in range(vocab_size)]


# ----------------------------------------------------------------------
# Scaffolds: the mask, and the loop
# ----------------------------------------------------------------------


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
    # TODO
    raise NotImplementedError


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

    Args:
        model: anything with `forward((1, T)) -> (1, T, V)` logits and
            a `max_seq_len` attribute — same contract as `generate`.
        prompt_ids: 1-D LongTensor, the prompt. Non-empty.
        pieces: `vocab_pieces(tokenizer, V)` — must have exactly one
            entry per logit column.
        max_new_tokens: token budget. If it runs out before the root
            object closes, the output is a *valid prefix* — truncated,
            never malformed. Callers can tell the difference by
            checking `is_complete` on the re-advanced text.
        temperature / top_k / top_p: Module 11's warpers, applied
            AFTER the mask, within the allowed set. `temperature=0.0`
            is greedy mode — which bypasses the warpers, but NOT the
            mask: the grammar is a constraint, not a warper.
        generator: optional torch.Generator for reproducible sampling.
        automaton: the grammar; defaults to `JsonPrefixAutomaton()`.
        mask_cache: optional dict for memoizing masks by automaton
            state. The O(V) decode-and-check sweep is the slow part of
            every step — but the mask depends ONLY on the automaton
            state, the states are hashable tuples, and a whole
            generation visits a few dozen distinct ones. Memoizing
            turns almost every step's sweep into a dict lookup. Pass
            the SAME dict across calls that share `pieces` and the
            cost amortizes across your whole experiment. (This is the
            first rung of the ladder to production: Outlines and
            llguidance precompute the mask for *every* automaton state
            ahead of time, so sampling never sweeps at all.) `None`
            still caches, but only within this one call.

    Returns:
        1-D LongTensor — `prompt_ids` followed by the constrained
        continuation, same contract as `generate`.

    Recipe — `generate`'s loop with two insertions (mask before
    warpers, grammar-driven stop):

        1. # Validation:
           if prompt_ids.dim() != 1 or prompt_ids.numel() == 0:
               raise ValueError(...)
           if temperature < 0:
               raise ValueError(...)

        2. if automaton is None:
               automaton = JsonPrefixAutomaton()
           if mask_cache is None:
               mask_cache = {}
           state = automaton.initial()
           greedy = (temperature == 0.0)
           full_ids = prompt_ids.detach().cpu().clone()
           device = getattr(model, "device", torch.device("cpu"))

        3. for _ in range(max_new_tokens):
               # 3a. Forward pass, exactly as in `generate`:
               ctx = full_ids[-model.max_seq_len:]
               logits = model(ctx.to(device).unsqueeze(0))
               last_logits = logits[:, -1, :].cpu()          # (1, V)

               # 3b. Vocab/pieces must agree — a mismatch (wrong
               #     tokenizer for this model) would mask garbage:
               if last_logits.shape[-1] != len(pieces):
                   raise ValueError(...)

               # 3c. THE CONSTRAINT — before any warper. Memoized by
               #     state: the mask is a function of the automaton
               #     state alone, so the O(V) sweep runs once per
               #     distinct state instead of once per step.
               allowed = mask_cache.get(state)
               if allowed is None:
                   allowed = allowed_token_mask(automaton, state, pieces)
                   mask_cache[state] = allowed
               if not allowed.any():
                   break          # automaton says stop (complete state)
               last_logits = last_logits.clone()
               last_logits[:, ~allowed] = float("-inf")

               # Why mask FIRST: top-k keeps the k highest logits
               # wherever they are. Run it before the mask and all k
               # survivors can be ungrammatical — leaving every logit
               # at -inf and softmax at NaN. Constrain the set, then
               # shape the distribution within it.

               # 3d. Warp and sample, exactly as in `generate`
               #     (greedy -> argmax; otherwise temperature, top_k,
               #     top_p, softmax, multinomial). No repetition
               #     penalty — JSON must repeat its delimiters.

               # 3e. Append, and advance the grammar by the piece:
               full_ids = torch.cat([full_ids, next_id], dim=0)
               state = automaton.advance(state, pieces[next_id.item()])

               # 3f. Stop the moment the root object closes:
               if automaton.is_complete(state):
                   break

        4. return full_ids

    Notes:
      - Step 3e cannot produce `state is None`: the mask only admitted
        tokens whose advance succeeds. If it ever comes back None, the
        mask is unsound — which is precisely what the tests check.
      - A JSON prefix always has at least one legal next character,
        and byte-level vocabularies (course BPE, BaseLM) contain every
        single character — so the mask only empties at completion.
    """
    # TODO
    raise NotImplementedError
