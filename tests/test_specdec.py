"""Tests for g2c/specdec — Beyond module: speculative decoding + MTP.

Suggested order to implement & turn green:

1. `greedy_verify` — unblocks the `test_greedy_verify_*` group. Get
   the return convention right first: the accepted count AND the free
   correction/bonus token.
2. `speculative_verify` — unblocks the stochastic tests. The
   statistical test (`test_speculative_verify_matches_target_distribution`)
   is the module's theorem check: emitted tokens must follow the
   TARGET distribution no matter how bad the drafter is.
3. `speculative_generate` — unblocks the loop tests. The anchor is
   `test_speculative_generate_matches_argmax_chain_with_any_drafter`:
   greedy speculative output must equal the target's own argmax chain,
   for ANY drafter.
4. `MTPHead.forward` + `mtp_loss` — the alignment test pins the
   two-ahead shift.

`SpecStats`, `draft_greedy`, and the `MTPHead` boilerplate are
provided, so those tests pass from the start. The tests marked
"integration" at the bottom drive real `TransformerLM`s, so they
additionally need Modules 05–09 (yours, or `G2C_APPLY_SOLUTIONS`).
"""
from __future__ import annotations

import pytest
import torch

from g2c.specdec import (
    MTPHead,
    SpecStats,
    draft_greedy,
    greedy_verify,
    mtp_loss,
    speculative_generate,
    speculative_verify,
)

V = 8  # mock vocabulary size


class _TableLM:
    """A deterministic first-order mock LM: logits depend only on the
    current token, via a `(V, V)` table. Its greedy chain is trivial to
    compute by hand, which makes exact-equality tests possible without
    any trained model."""

    max_seq_len = 32
    device = torch.device("cpu")

    def __init__(self, table: torch.Tensor) -> None:
        self.table = table

    def __call__(self, ids: torch.Tensor) -> torch.Tensor:  # (1,T) -> (1,T,V)
        return self.table[ids]


def _chain_table(seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(V, V)


def _argmax_chain(table: torch.Tensor, start: int, n: int) -> list[int]:
    out, cur = [], start
    for _ in range(n):
        cur = int(table[cur].argmax())
        out.append(cur)
    return out


# ---------------------------------------------------------------------------
# Provided plumbing — green from the start.
# ---------------------------------------------------------------------------


def test_spec_stats_accounting():
    stats = SpecStats()
    stats.record(drafted=4, accepted=3, generated=4)
    stats.record(drafted=4, accepted=1, generated=2)
    assert stats.target_passes == 2
    assert stats.drafted == 8
    assert stats.accepted == 4
    assert stats.generated == 6
    assert stats.acceptance_rate == pytest.approx(0.5)
    assert stats.tokens_per_pass == pytest.approx(3.0)


def test_spec_stats_empty_is_safe():
    stats = SpecStats()
    assert stats.acceptance_rate == 0.0
    assert stats.tokens_per_pass == 0.0


def test_draft_greedy_follows_argmax_chain():
    table = _chain_table(0)
    drafter = _TableLM(table)
    prompt = torch.tensor([3], dtype=torch.long)
    draft = draft_greedy(drafter, prompt, 5)
    assert draft.tolist() == _argmax_chain(table, 3, 5)
    assert draft.dtype == torch.long


def test_draft_greedy_crops_to_drafter_window():
    table = _chain_table(1)
    drafter = _TableLM(table)
    drafter.max_seq_len = 4
    long_prompt = torch.randint(0, V, (20,))
    draft = draft_greedy(drafter, long_prompt, 3)
    # A first-order model only looks at the last token, so cropping
    # cannot change the result — this just asserts it doesn't crash
    # and produces the chain from the prompt's last token.
    assert draft.tolist() == _argmax_chain(table, int(long_prompt[-1]), 3)


# ---------------------------------------------------------------------------
# greedy_verify
# ---------------------------------------------------------------------------


def _logits_for(rows: list[int]) -> torch.Tensor:
    """Logits whose argmax at row i is rows[i]."""
    out = torch.zeros(len(rows), V)
    for i, tok in enumerate(rows):
        out[i, tok] = 5.0
    return out


def test_greedy_verify_accepts_full_match_with_bonus():
    draft = torch.tensor([2, 4, 6])
    logits = _logits_for([2, 4, 6, 1])  # target agrees, then wants 1
    n, nxt = greedy_verify(draft, logits)
    assert (n, nxt) == (3, 1)


def test_greedy_verify_rejects_at_first_position():
    draft = torch.tensor([2, 4, 6])
    logits = _logits_for([7, 4, 6, 1])
    n, nxt = greedy_verify(draft, logits)
    assert (n, nxt) == (0, 7)  # correction, not the draft token


def test_greedy_verify_rejects_mid_block():
    draft = torch.tensor([2, 4, 6, 3])
    logits = _logits_for([2, 4, 5, 3, 0])
    n, nxt = greedy_verify(draft, logits)
    assert (n, nxt) == (2, 5)


def test_greedy_verify_validates_shapes():
    with pytest.raises(ValueError):
        greedy_verify(torch.tensor([1, 2]), _logits_for([1, 2]))  # k rows, not k+1
    with pytest.raises(ValueError):
        greedy_verify(torch.tensor([], dtype=torch.long), _logits_for([1]))


# ---------------------------------------------------------------------------
# speculative_verify — the stochastic rule
# ---------------------------------------------------------------------------


def test_speculative_verify_identical_distributions_always_accept():
    torch.manual_seed(2)
    p = torch.softmax(torch.randn(3, V), dim=-1)
    probs = torch.cat([p, torch.softmax(torch.randn(1, V), dim=-1)])
    gen = torch.Generator().manual_seed(0)
    draft = torch.tensor([int(row.argmax()) for row in p])
    for _ in range(20):
        n, nxt = speculative_verify(draft, probs, p, generator=gen)
        assert n == 3  # ratio is 1 everywhere: never rejected
        assert 0 <= nxt < V


def test_speculative_verify_zero_target_prob_always_rejects():
    # Drafter proposes token 0; target gives it zero probability.
    draft = torch.tensor([0])
    target = torch.zeros(2, V)
    target[:, 1:] = 1.0 / (V - 1)
    draft_p = torch.zeros(1, V)
    draft_p[0, 0] = 1.0
    gen = torch.Generator().manual_seed(1)
    for _ in range(20):
        n, nxt = speculative_verify(draft, target, draft_p, generator=gen)
        assert n == 0
        assert nxt != 0  # resampled from the residual: target mass only


def test_speculative_verify_matches_target_distribution():
    """The speculative sampling theorem, checked empirically: with
    draft ~ p_draft and this acceptance rule, the emitted first token
    is distributed as p_target — for a deliberately BAD drafter."""
    torch.manual_seed(3)
    p_target = torch.tensor([[0.5, 0.3, 0.15, 0.05]])
    p_draft = torch.tensor([[0.05, 0.15, 0.3, 0.5]])  # reversed: bad drafter
    probs = torch.cat([p_target, torch.full((1, 4), 0.25)])

    gen = torch.Generator().manual_seed(4)
    counts = torch.zeros(4)
    trials = 4000
    for _ in range(trials):
        d = torch.multinomial(p_draft[0], 1, generator=gen)
        n, nxt = speculative_verify(d, probs, p_draft, generator=gen)
        emitted = int(d) if n == 1 else nxt
        counts[emitted] += 1

    empirical = counts / trials
    tv = 0.5 * (empirical - p_target[0]).abs().sum()
    assert tv < 0.05, (empirical, p_target[0])


# ---------------------------------------------------------------------------
# speculative_generate — the loop
# ---------------------------------------------------------------------------


def test_speculative_generate_matches_argmax_chain_with_any_drafter():
    """The anchor: greedy speculative output equals the target's own
    greedy chain no matter how unrelated the drafter is."""
    target_table = _chain_table(10)
    drafter_table = _chain_table(99)  # a completely different model
    target, drafter = _TableLM(target_table), _TableLM(drafter_table)

    prompt = torch.tensor([5], dtype=torch.long)
    ids, stats = speculative_generate(target, drafter, prompt, 12, k=4)
    assert ids[:1].tolist() == [5]
    assert ids[1:].tolist() == _argmax_chain(target_table, 5, 12)
    assert stats.generated == 12
    assert stats.target_passes >= 1
    # A useless drafter still advances at least one token per pass.
    assert stats.tokens_per_pass >= 1.0


def test_speculative_generate_perfect_drafter_max_acceptance():
    table = _chain_table(11)
    target, drafter = _TableLM(table), _TableLM(table)
    prompt = torch.tensor([2], dtype=torch.long)
    ids, stats = speculative_generate(target, drafter, prompt, 12, k=3)
    assert ids[1:].tolist() == _argmax_chain(table, 2, 12)
    # Every draft accepted: k + 1 = 4 tokens per pass, 12 tokens in 3.
    assert stats.acceptance_rate == pytest.approx(1.0)
    assert stats.target_passes == 3


def test_speculative_generate_respects_budget():
    table = _chain_table(12)
    target, drafter = _TableLM(table), _TableLM(table)
    prompt = torch.tensor([1], dtype=torch.long)
    for budget in (1, 2, 5, 7):
        ids, stats = speculative_generate(target, drafter, prompt, budget, k=3)
        assert ids.numel() == 1 + budget
        assert stats.generated == budget


def test_speculative_generate_stops_at_eos():
    # Build a chain that hits token 0 quickly, and call 0 the EOS.
    table = torch.zeros(V, V)
    table[1, 2] = 5.0
    table[2, 0] = 5.0  # 1 -> 2 -> 0 (eos)
    table[0, 3] = 5.0
    target, drafter = _TableLM(table), _TableLM(table)
    prompt = torch.tensor([1], dtype=torch.long)
    ids, stats = speculative_generate(target, drafter, prompt, 10, k=4, eos_id=0)
    assert ids.tolist() == [1, 2, 0]
    assert stats.generated == 2


def test_speculative_generate_validates_input():
    table = _chain_table(13)
    m = _TableLM(table)
    with pytest.raises(ValueError):
        speculative_generate(m, m, torch.tensor([[1]]), 4)
    with pytest.raises(ValueError):
        speculative_generate(m, m, torch.tensor([1]), 4, k=0)


# ---------------------------------------------------------------------------
# The MTP head
# ---------------------------------------------------------------------------


def _tiny_base():
    from g2c.transformer import TransformerLM

    torch.manual_seed(20)
    return TransformerLM(
        vocab_size=V, embedding_dim=16, num_layers=2, num_heads=2,
        max_seq_len=16,
    )


def test_mtp_head_parameters_exclude_frozen_base():
    base = _tiny_base()
    head = MTPHead(base)
    base_ids = {id(p) for p in base.parameters()}
    head_ids = {id(p) for p in head.parameters()}
    assert not base_ids & head_ids
    assert len(head_ids) > 0


def test_mtp_loss_alignment():
    """One-hot logits at exactly ids[:, t+2] must give near-zero loss;
    the same logits shifted one step must not."""
    torch.manual_seed(21)
    B, T = 2, 8
    ids = torch.randint(0, V, (B, T))
    aligned = torch.full((B, T - 1, V), -10.0)
    misaligned = torch.full((B, T - 1, V), -10.0)
    for b in range(B):
        for t in range(T - 2):
            aligned[b, t, ids[b, t + 2]] = 10.0
            misaligned[b, t, ids[b, t + 1]] = 10.0
    assert mtp_loss(aligned, ids).item() < 0.01
    assert mtp_loss(misaligned, ids).item() > 1.0


# ---------------------------------------------------------------------------
# Integration — real TransformerLMs (needs Modules 05–09 implemented,
# or G2C_APPLY_SOLUTIONS).
# ---------------------------------------------------------------------------


def test_mtp_head_forward_shape():
    from g2c.specdec.mtp import hidden_states

    base = _tiny_base()
    head = MTPHead(base)
    ids = torch.randint(0, V, (2, 10))
    h = hidden_states(base, ids)
    assert h.shape == (2, 10, 16)
    logits = head(h[:, :-1], ids[:, 1:])
    assert logits.shape == (2, 9, V)


def test_speculative_generate_matches_module_11_generate():
    from g2c.sampling import generate

    torch.manual_seed(22)
    target = _tiny_base()
    torch.manual_seed(77)
    from g2c.transformer import TransformerLM

    drafter = TransformerLM(
        vocab_size=V, embedding_dim=8, num_layers=1, num_heads=2,
        max_seq_len=16,
    )
    prompt = torch.tensor([3, 1], dtype=torch.long)
    plain = generate(target, prompt, 8, temperature=0.0)
    spec, _ = speculative_generate(target, drafter, prompt, 8, k=3)
    assert spec.tolist() == plain.tolist()
