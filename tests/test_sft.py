"""Tests for Module 13: instruction tuning (SFT).

Suggested order to implement & turn green:

  1. ChatTemplate.render            → test_chat_template_render_*
  2. ChatTemplate.render_with_mask  → test_render_with_mask_*
  3. pad_and_collate                → test_pad_and_collate_*
  4. masked_cross_entropy           → test_masked_cross_entropy_*
  5. SFTTrainer.train_step          → test_sft_trainer_train_step_*,
                                      test_sft_trainer_loss_decreases

Steps 1–4 are independent — you can implement them in any order; their
tests don't depend on each other or on a real tokenizer (a tiny
character-level fake tokenizer is used for the mask tests). Step 5
exercises the full SFT loop end-to-end against a tiny `TransformerLM`
and depends on Modules 03 / 05 / 07 / 08 / 09 / 10 being implemented.
If your Module 10 deliverable test (`test_trainer_train_runs_to_
completion`) isn't passing, finish that first.

Boilerplate tests — `ChatTemplate` constants, `SFTExample` shape,
`SFTTrainer.__init__` validation, `SFTTrainer.lr` — pass from the
start as a sanity check on the test file itself.

Most tests use very small dimensions (`vocab_size=128`,
`embedding_dim=8`, `num_layers=1`, `T=12`, batch=2) so the suite runs
in well under a second.
"""
from __future__ import annotations

import math

import pytest
import torch

from g2c.sft import (
    ChatTemplate,
    SFTExample,
    SFTTrainer,
    masked_cross_entropy,
    pad_and_collate,
)
from g2c.training import lm_cross_entropy
from g2c.transformer import TransformerLM


# ----------------------------------------------------------------------
# Test fixtures: a tiny character-level fake tokenizer
# ----------------------------------------------------------------------


class _FakeTokenizer:
    """A trivial tokenizer that maps each character to its ord() value.

    Used so the chat-template mask tests don't depend on Module 04's
    BPE implementation being filled in. The mask boundaries are
    independent of the tokenization scheme — every byte gets a mask
    bit, and the content/role-marker partition is determined by the
    template, not the tokenizer.
    """

    def encode(self, s: str) -> list[int]:
        return [ord(c) for c in s]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i) for i in ids)


# ----------------------------------------------------------------------
# ChatTemplate — boilerplate (constants implemented)
# ----------------------------------------------------------------------


def test_chat_template_constants_are_set():
    """The role markers are class-level constants accessible without
    instantiation. A drift in these strings between training and
    inference is the most-common Module 13 / Module 17 / Module 19
    bug; this test pins them down."""
    assert ChatTemplate.USER == "<|user|>"
    assert ChatTemplate.ASSISTANT == "<|assistant|>"
    assert ChatTemplate.END == "<|end|>"


def test_chat_template_can_be_instantiated():
    """Construction is parameter-free."""
    t = ChatTemplate()
    assert t.USER == "<|user|>"


# ----------------------------------------------------------------------
# ChatTemplate.render — scaffolded
# ----------------------------------------------------------------------


def test_chat_template_render_single_pair():
    """The headline format pin-down. A single (user, assistant) turn
    must produce exactly this byte string."""
    t = ChatTemplate()
    out = t.render(
        [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
    )
    assert out == "<|user|>\nHi\n<|assistant|>\nHello<|end|>"


def test_chat_template_render_user_only_ends_with_newline():
    """The inference-time prompt-assembly case: a partial conversation
    ending with a user turn. The trailing newline is what makes the
    BPE tokenization of the next role marker stay consistent with
    training-time tokenization."""
    t = ChatTemplate()
    out = t.render([{"role": "user", "content": "Hi"}])
    assert out == "<|user|>\nHi\n"


def test_chat_template_render_assistant_turn_no_trailing_newline():
    """An assistant-final turn ends with <|end|>, NOT a newline."""
    t = ChatTemplate()
    out = t.render(
        [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
    )
    assert out.endswith("<|end|>")
    assert not out.endswith("<|end|>\n")


def test_chat_template_render_multi_turn():
    """Two (user, assistant) pairs render as their concatenation."""
    t = ChatTemplate()
    out = t.render(
        [
            {"role": "user", "content": "A"},
            {"role": "assistant", "content": "B"},
            {"role": "user", "content": "C"},
            {"role": "assistant", "content": "D"},
        ]
    )
    expected = (
        "<|user|>\nA\n<|assistant|>\nB<|end|>"
        "<|user|>\nC\n<|assistant|>\nD<|end|>"
    )
    assert out == expected


def test_chat_template_render_empty_messages_raises():
    """An empty message list is a programming error — raises rather
    than silently returning empty string."""
    t = ChatTemplate()
    with pytest.raises(ValueError):
        t.render([])


def test_chat_template_render_unknown_role_raises():
    """An unknown role is a programming error."""
    t = ChatTemplate()
    with pytest.raises(ValueError):
        t.render([{"role": "system", "content": "..."}])


# ----------------------------------------------------------------------
# ChatTemplate.render_with_mask — scaffolded
# ----------------------------------------------------------------------


def test_render_with_mask_returns_sft_example():
    t = ChatTemplate()
    tok = _FakeTokenizer()
    ex = t.render_with_mask(
        [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Yo"},
        ],
        tokenizer=tok,
    )
    assert isinstance(ex, SFTExample)


def test_render_with_mask_ids_and_mask_same_length():
    t = ChatTemplate()
    tok = _FakeTokenizer()
    ex = t.render_with_mask(
        [
            {"role": "user", "content": "Hello there"},
            {"role": "assistant", "content": "Hi back"},
        ],
        tokenizer=tok,
    )
    assert len(ex.ids) == len(ex.mask)


def test_render_with_mask_ids_match_render_string():
    """The ids returned by render_with_mask must equal the
    tokenization of render(messages). If they diverge, the loss mask
    is being computed on a different sequence than the model sees."""
    t = ChatTemplate()
    tok = _FakeTokenizer()
    messages = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
    ]
    ex = t.render_with_mask(messages, tokenizer=tok)
    expected_ids = tok.encode(t.render(messages))
    assert ex.ids == expected_ids


def test_render_with_mask_user_tokens_zero():
    """Every user-turn token (role marker, content, trailing newline)
    has mask = 0."""
    t = ChatTemplate()
    tok = _FakeTokenizer()
    ex = t.render_with_mask(
        [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ],
        tokenizer=tok,
    )
    user_chunk = f"{t.USER}\nHi\n"
    user_len = len(tok.encode(user_chunk))
    assert ex.mask[:user_len] == [0] * user_len


def test_render_with_mask_assistant_role_marker_zero():
    """The `<|assistant|>\\n` role-marker prefix has mask = 0 — the
    model should NOT learn to emit the marker; it's a prompt token
    that the chat-template assembly pre-fills at inference."""
    t = ChatTemplate()
    tok = _FakeTokenizer()
    ex = t.render_with_mask(
        [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ],
        tokenizer=tok,
    )
    user_chunk = f"{t.USER}\nHi\n"
    user_len = len(tok.encode(user_chunk))
    assistant_marker = f"{t.ASSISTANT}\n"
    marker_len = len(tok.encode(assistant_marker))
    assert ex.mask[user_len : user_len + marker_len] == [0] * marker_len


def test_render_with_mask_assistant_content_one():
    """Every byte of assistant content has mask = 1 — including the
    trailing `<|end|>`. If `<|end|>` is masked out, the model never
    learns to stop, and inference loops to max_new_tokens."""
    t = ChatTemplate()
    tok = _FakeTokenizer()
    ex = t.render_with_mask(
        [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ],
        tokenizer=tok,
    )
    response = f"Hello{t.END}"
    response_len = len(tok.encode(response))
    # The response is the LAST `response_len` tokens; all should be 1.
    assert ex.mask[-response_len:] == [1] * response_len


def test_render_with_mask_end_token_position_is_masked_one():
    """A pin-down for the most common omission: every byte of
    <|end|> must be in the mask=1 region."""
    t = ChatTemplate()
    tok = _FakeTokenizer()
    ex = t.render_with_mask(
        [
            {"role": "user", "content": "X"},
            {"role": "assistant", "content": "Y"},
        ],
        tokenizer=tok,
    )
    end_len = len(tok.encode(t.END))
    assert ex.mask[-end_len:] == [1] * end_len


def test_render_with_mask_multi_turn_alternates():
    """In a multi-turn conversation, every assistant turn's content
    and trailing <|end|> is mask=1; every user turn (incl. role
    marker) and every assistant role marker is mask=0."""
    t = ChatTemplate()
    tok = _FakeTokenizer()
    ex = t.render_with_mask(
        [
            {"role": "user", "content": "A"},
            {"role": "assistant", "content": "B"},
            {"role": "user", "content": "C"},
            {"role": "assistant", "content": "D"},
        ],
        tokenizer=tok,
    )
    # Total mask sum should equal total length of (B<|end|> + D<|end|>).
    expected_one_count = (
        len(tok.encode(f"B{t.END}")) + len(tok.encode(f"D{t.END}"))
    )
    assert sum(ex.mask) == expected_one_count


# ----------------------------------------------------------------------
# SFTExample — boilerplate (NamedTuple)
# ----------------------------------------------------------------------


def test_sft_example_is_named_tuple():
    """SFTExample is a NamedTuple — accessible by .ids and .mask."""
    ex = SFTExample(ids=[1, 2, 3], mask=[0, 1, 1])
    assert ex.ids == [1, 2, 3]
    assert ex.mask == [0, 1, 1]


def test_sft_example_iterable_unpack():
    """NamedTuple unpacking works (used inside the trainer)."""
    ex = SFTExample(ids=[1, 2], mask=[0, 1])
    ids, mask = ex
    assert ids == [1, 2]
    assert mask == [0, 1]


# ----------------------------------------------------------------------
# pad_and_collate — scaffolded
# ----------------------------------------------------------------------


def test_pad_and_collate_returns_three_tensors():
    examples = [SFTExample(ids=[1, 2, 3, 4], mask=[0, 0, 1, 1])]
    x, y, mask = pad_and_collate(examples, max_seq_len=4, pad_id=0)
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert isinstance(mask, torch.Tensor)


def test_pad_and_collate_shapes():
    """Output shape is (B, max_seq_len - 1) for all three tensors —
    the shift-by-one shrinks the time dim by 1."""
    examples = [
        SFTExample(ids=[1, 2, 3, 4, 5], mask=[0, 0, 1, 1, 1]),
        SFTExample(ids=[6, 7, 8], mask=[0, 1, 1]),
    ]
    x, y, mask = pad_and_collate(examples, max_seq_len=5, pad_id=0)
    assert x.shape == (2, 4)
    assert y.shape == (2, 4)
    assert mask.shape == (2, 4)


def test_pad_and_collate_dtypes():
    examples = [SFTExample(ids=[1, 2, 3, 4], mask=[0, 0, 1, 1])]
    x, y, mask = pad_and_collate(examples, max_seq_len=4, pad_id=0)
    assert x.dtype == torch.long
    assert y.dtype == torch.long
    # mask may be long or float — int-typed is enough; the loss
    # function casts as needed.
    assert mask.dtype in (torch.long, torch.float, torch.float32)


def test_pad_and_collate_shifts_for_lm():
    """y[b, t] should equal `ids[b, t+1]` — the standard LM shift.

    A miswired shift silently trains the model on the wrong target
    distribution; pinning it down is more important than it looks."""
    examples = [SFTExample(ids=[10, 11, 12, 13, 14], mask=[0, 0, 1, 1, 1])]
    x, y, mask = pad_and_collate(examples, max_seq_len=5, pad_id=0)
    assert torch.equal(x[0], torch.tensor([10, 11, 12, 13]))
    assert torch.equal(y[0], torch.tensor([11, 12, 13, 14]))


def test_pad_and_collate_loss_mask_aligned_with_y():
    """loss_mask[b, t] must align with y[b, t] — i.e. it's the
    INPUT mask shifted by one. This is the off-by-one that, if wrong,
    trains the model to predict the prompt token instead of the
    response."""
    # ids: [10, 11, 12, 13, 14]
    # mask: [ 0,  0,  1,  1,  1]
    # After shift: y = [11, 12, 13, 14], loss_mask should be the
    # mask of those targets — which is mask[1:] = [0, 1, 1, 1].
    examples = [SFTExample(ids=[10, 11, 12, 13, 14], mask=[0, 0, 1, 1, 1])]
    x, y, mask = pad_and_collate(examples, max_seq_len=5, pad_id=0)
    assert torch.equal(mask[0].long(), torch.tensor([0, 1, 1, 1]))


def test_pad_and_collate_pads_short_examples():
    """Short examples are padded with pad_id; padded mask positions
    are 0."""
    # max_seq_len=5 but the example has length 3.
    examples = [SFTExample(ids=[10, 11, 12], mask=[0, 1, 1])]
    x, y, mask = pad_and_collate(examples, max_seq_len=5, pad_id=99)
    # ids_b padded to [10, 11, 12, 99, 99].
    # x = ids_b[:-1] = [10, 11, 12, 99]
    # y = ids_b[1:]  = [11, 12, 99, 99]
    # mask_b padded to [0, 1, 1, 0, 0]
    # loss_mask = mask_b[1:] = [1, 1, 0, 0]
    assert torch.equal(x[0], torch.tensor([10, 11, 12, 99]))
    assert torch.equal(y[0], torch.tensor([11, 12, 99, 99]))
    assert torch.equal(mask[0].long(), torch.tensor([1, 1, 0, 0]))


def test_pad_and_collate_pad_positions_have_loss_mask_zero():
    """The most important padding pin: padded positions in the
    target stream have loss_mask = 0. Otherwise the model is
    trained to predict the pad token, which is wrong."""
    examples = [SFTExample(ids=[1, 2, 3], mask=[0, 1, 1])]
    x, y, mask = pad_and_collate(examples, max_seq_len=6, pad_id=0)
    # mask_b = [0, 1, 1, 0, 0, 0]; loss_mask = mask_b[1:] = [1, 1, 0, 0, 0]
    # The last 3 positions are padding-targets.
    assert mask[0, 2:].long().tolist() == [0, 0, 0]


def test_pad_and_collate_truncates_long_examples():
    """Examples longer than max_seq_len are truncated to max_seq_len
    tokens (head-truncation — keep the start, drop the tail)."""
    examples = [
        SFTExample(ids=list(range(20)), mask=[0] * 5 + [1] * 15),
    ]
    x, y, mask = pad_and_collate(examples, max_seq_len=6, pad_id=0)
    # Truncated to 6: ids=[0,1,2,3,4,5], mask=[0,0,0,0,0,1].
    # Shifted: x=[0,1,2,3,4], y=[1,2,3,4,5], loss_mask=[0,0,0,0,1].
    assert torch.equal(x[0], torch.tensor([0, 1, 2, 3, 4]))
    assert torch.equal(y[0], torch.tensor([1, 2, 3, 4, 5]))
    assert torch.equal(mask[0].long(), torch.tensor([0, 0, 0, 0, 1]))


def test_pad_and_collate_batch_pads_to_common_length():
    """Mixed-length examples in one batch all get padded to
    max_seq_len, then shifted."""
    examples = [
        SFTExample(ids=[1, 2, 3], mask=[0, 1, 1]),
        SFTExample(ids=[4, 5, 6, 7, 8], mask=[0, 0, 1, 1, 1]),
    ]
    x, y, mask = pad_and_collate(examples, max_seq_len=5, pad_id=0)
    assert x.shape == (2, 4)
    # Second example wasn't padded (length already == max_seq_len).
    assert torch.equal(x[1], torch.tensor([4, 5, 6, 7]))


# ----------------------------------------------------------------------
# masked_cross_entropy — scaffolded
# ----------------------------------------------------------------------


def test_masked_cross_entropy_returns_scalar():
    logits = torch.randn(2, 4, 7)
    targets = torch.randint(0, 7, (2, 4))
    mask = torch.ones(2, 4, dtype=torch.long)
    loss = masked_cross_entropy(logits, targets, mask)
    assert loss.shape == ()
    assert torch.is_tensor(loss)


def test_masked_cross_entropy_full_mask_matches_lm_loss():
    """When the mask is uniformly 1 (every position counted), the
    masked loss equals the unmasked Module 10 lm_cross_entropy."""
    torch.manual_seed(42)
    logits = torch.randn(2, 5, 8)
    targets = torch.randint(0, 8, (2, 5))
    mask = torch.ones(2, 5, dtype=torch.long)
    sft_loss = masked_cross_entropy(logits, targets, mask).item()
    lm_loss = lm_cross_entropy(logits, targets).item()
    assert math.isclose(sft_loss, lm_loss, rel_tol=1e-5)


def test_masked_cross_entropy_zero_mask_returns_zero():
    """A fully-masked batch returns 0.0, not NaN. Guards the corner
    case of a batch with no assistant tokens."""
    logits = torch.randn(2, 4, 7)
    targets = torch.randint(0, 7, (2, 4))
    mask = torch.zeros(2, 4, dtype=torch.long)
    loss = masked_cross_entropy(logits, targets, mask)
    assert loss.item() == 0.0
    assert not torch.isnan(loss)


def test_masked_cross_entropy_normalizes_by_mask_count():
    """The denominator is mask.sum(), not B*T. Construct a batch
    where one half is masked: the loss should equal the average over
    the unmasked positions only.

    Concretely: take a batch where mask is 1 on the first half of
    positions and 0 on the second half. The loss should equal the
    average per-position CE over the first half — NOT half that
    value (which is what (sum / B*T) would produce)."""
    torch.manual_seed(0)
    logits = torch.randn(1, 4, 5)
    targets = torch.randint(0, 5, (1, 4))
    mask_full = torch.ones(1, 4, dtype=torch.long)
    full_loss = masked_cross_entropy(logits, targets, mask_full).item()

    # Mask out the second half; recompute over only positions 0, 1.
    mask_half = torch.tensor([[1, 1, 0, 0]])
    half_loss = masked_cross_entropy(logits, targets, mask_half).item()

    # Manually compute the per-position CE for positions 0 and 1.
    per_pos_full = torch.nn.functional.cross_entropy(
        logits.reshape(-1, 5), targets.reshape(-1), reduction="none"
    )
    expected_half = (per_pos_full[0].item() + per_pos_full[1].item()) / 2

    assert math.isclose(half_loss, expected_half, rel_tol=1e-5)
    # Sanity: the half_loss is NOT just full_loss / 2 (which would
    # mean we're averaging by B*T instead of mask.sum()).
    assert not math.isclose(half_loss, full_loss / 2, rel_tol=1e-3)


def test_masked_cross_entropy_ignores_masked_logits():
    """Changing logits at mask=0 positions must not change the loss.

    This is the strictest "is the mask actually masking" test."""
    torch.manual_seed(7)
    logits = torch.randn(1, 4, 5)
    targets = torch.randint(0, 5, (1, 4))
    mask = torch.tensor([[0, 0, 1, 1]])
    loss_before = masked_cross_entropy(logits, targets, mask).item()
    # Wildly perturb the masked-out positions.
    logits[0, 0] = 1000.0
    logits[0, 1] = -1000.0
    loss_after = masked_cross_entropy(logits, targets, mask).item()
    assert math.isclose(loss_before, loss_after, rel_tol=1e-5)


def test_masked_cross_entropy_accepts_float_mask():
    """The mask may be a float tensor (e.g. cast to logits.dtype by
    the caller). Both int and float should produce the same answer."""
    torch.manual_seed(11)
    logits = torch.randn(1, 4, 5)
    targets = torch.randint(0, 5, (1, 4))
    mask_int = torch.tensor([[0, 1, 1, 1]], dtype=torch.long)
    mask_float = mask_int.to(torch.float32)
    a = masked_cross_entropy(logits, targets, mask_int).item()
    b = masked_cross_entropy(logits, targets, mask_float).item()
    assert math.isclose(a, b, rel_tol=1e-5)


# ----------------------------------------------------------------------
# SFTTrainer — construction and lr (boilerplate, implemented)
# ----------------------------------------------------------------------


def _make_tiny_model():
    """A tiny TransformerLM for the SFTTrainer end-to-end tests.

    1 layer, embedding_dim=8, vocab=128, max_seq_len=16. Small enough
    to forward+backward in milliseconds. vocab=128 covers the ASCII
    range used by the fake tokenizer.
    """
    return TransformerLM(
        vocab_size=128,
        embedding_dim=8,
        num_layers=1,
        num_heads=2,
        max_seq_len=16,
    )


def _make_tiny_dataset(n: int = 6) -> list[SFTExample]:
    """A handful of synthetic examples with explicit ids and masks.

    Avoids depending on Module 04 BPE / Module 13 chat-template — the
    mask boundaries are pre-baked. Each example is 8 tokens with 5
    prompt tokens (mask=0) and 3 response tokens (mask=1).
    """
    examples: list[SFTExample] = []
    rng = torch.Generator().manual_seed(0)
    for _ in range(n):
        ids = torch.randint(1, 128, (8,), generator=rng).tolist()
        mask = [0, 0, 0, 0, 0, 1, 1, 1]
        examples.append(SFTExample(ids=ids, mask=mask))
    return examples


def test_sft_trainer_construction_defaults():
    model = _make_tiny_model()
    examples = _make_tiny_dataset()
    trainer = SFTTrainer(
        model,
        examples=examples,
        max_seq_len=8,
        pad_id=0,
        batch_size=2,
        max_steps=10,
        max_lr=3e-4,
    )
    assert trainer.max_steps == 10
    assert trainer.batch_size == 2
    assert trainer.step == 0
    assert trainer.optimizer is not None


def test_sft_trainer_empty_examples_raises():
    model = _make_tiny_model()
    with pytest.raises(ValueError):
        SFTTrainer(
            model,
            examples=[],
            max_seq_len=8,
            pad_id=0,
            batch_size=2,
            max_steps=10,
            max_lr=3e-4,
        )


def test_sft_trainer_bad_batch_size_raises():
    model = _make_tiny_model()
    with pytest.raises(ValueError):
        SFTTrainer(
            model,
            examples=_make_tiny_dataset(),
            max_seq_len=8,
            pad_id=0,
            batch_size=0,
            max_steps=10,
            max_lr=3e-4,
        )


def test_sft_trainer_max_seq_len_too_small_raises():
    """max_seq_len <= 1 means the shift-by-one would produce empty
    tensors. Reject at construction time."""
    model = _make_tiny_model()
    with pytest.raises(ValueError):
        SFTTrainer(
            model,
            examples=_make_tiny_dataset(),
            max_seq_len=1,
            pad_id=0,
            batch_size=2,
            max_steps=10,
            max_lr=3e-4,
        )


def test_sft_trainer_lr_uses_cosine_schedule():
    """SFTTrainer.lr should reproduce the cosine_with_warmup result
    at the current step. (Boilerplate test — passes without
    train_step being implemented.)"""
    from g2c.training import cosine_with_warmup

    model = _make_tiny_model()
    trainer = SFTTrainer(
        model,
        examples=_make_tiny_dataset(),
        max_seq_len=8,
        pad_id=0,
        batch_size=2,
        max_steps=100,
        max_lr=1e-3,
        warmup_steps=10,
    )
    assert math.isclose(
        trainer.lr(0),
        cosine_with_warmup(0, warmup_steps=10, max_steps=100, max_lr=1e-3),
    )
    assert math.isclose(
        trainer.lr(50),
        cosine_with_warmup(50, warmup_steps=10, max_steps=100, max_lr=1e-3),
    )


# ----------------------------------------------------------------------
# SFTTrainer.train_step — scaffolded
# ----------------------------------------------------------------------


def test_sft_trainer_train_step_returns_metrics():
    """One step returns a dict with loss / lr / grad_norm; advances
    self.step by 1."""
    torch.manual_seed(0)
    model = _make_tiny_model()
    examples = _make_tiny_dataset()
    trainer = SFTTrainer(
        model,
        examples=examples,
        max_seq_len=8,
        pad_id=0,
        batch_size=2,
        max_steps=10,
        max_lr=3e-4,
        grad_clip=1.0,
    )
    metrics = trainer.train_step()
    assert "loss" in metrics
    assert "lr" in metrics
    assert "grad_norm" in metrics
    assert isinstance(metrics["loss"], float)
    assert math.isfinite(metrics["loss"])
    assert trainer.step == 1


def test_sft_trainer_loss_decreases():
    """The headline end-to-end check. Running 30 SFT steps on a tiny
    synthetic dataset should drive the loss down meaningfully — the
    final loss is at least 5% lower than the initial loss.

    If this fails: data, loss, optimizer, or trainer wiring is
    broken. The smaller tests in this file will tell you which."""
    torch.manual_seed(0)
    model = _make_tiny_model()
    examples = _make_tiny_dataset(n=4)
    trainer = SFTTrainer(
        model,
        examples=examples,
        max_seq_len=8,
        pad_id=0,
        batch_size=2,
        max_steps=30,
        max_lr=1e-2,  # Higher than recommended SFT lr; we want fast
                      # signal in 30 steps for the test.
        warmup_steps=2,
        grad_clip=1.0,
    )
    history = trainer.train()
    initial_loss = history["train_loss"][0]
    final_loss = history["train_loss"][-1]
    assert final_loss < initial_loss * 0.95, (
        f"SFT loss did not decrease: initial={initial_loss}, "
        f"final={final_loss}"
    )


def test_sft_trainer_evaluate_returns_float():
    """evaluate() runs without grad and returns a Python float."""
    torch.manual_seed(0)
    model = _make_tiny_model()
    examples = _make_tiny_dataset()
    trainer = SFTTrainer(
        model,
        examples=examples,
        max_seq_len=8,
        pad_id=0,
        batch_size=2,
        max_steps=10,
        max_lr=3e-4,
        eval_iters=3,
    )
    val_loss = trainer.evaluate(examples)
    assert isinstance(val_loss, float)
    assert math.isfinite(val_loss)


def test_sft_trainer_evaluate_empty_raises():
    model = _make_tiny_model()
    trainer = SFTTrainer(
        model,
        examples=_make_tiny_dataset(),
        max_seq_len=8,
        pad_id=0,
        batch_size=2,
        max_steps=10,
        max_lr=3e-4,
    )
    with pytest.raises(ValueError):
        trainer.evaluate([])
