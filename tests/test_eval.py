"""Tests for `g2c.eval` — the eval harness for Module 15.

The package divides cleanly into independent layers; the tests are
grouped to follow them.

Suggested order to implement & turn green:

    1. `g2c.eval.match` — implement the four match functions
       (exact_match, normalized_match, contains_match, numeric_match).
       They are independent and small. Turns green:
         - All tests in TestExactMatch / TestNormalizedMatch /
           TestContainsMatch / TestNumericMatch.
         - End-to-end generation tests in TestRunGenerationEval (the
           harness uses these matchers).

    2. `g2c.eval.logprob.continuation_logprob`. Pure math: tokenize
       prompt + continuation, forward, log_softmax + gather + sum
       over a continuation-aligned mask. Turns green:
         - All tests in TestContinuationLogprob.

    3. `g2c.eval.calibration.expected_calibration_error`. The bucket
       loop. Turns green:
         - All tests in TestExpectedCalibrationError.
         - The ECE-aware aggregations in TestRunMultipleChoiceEval
           (once score_multiple_choice is implemented in step 5).

    4. `g2c.eval.calibration.reliability_curve`. Same binning as ECE;
       returns the per-bin (mean_confidence, accuracy, count) triples.
       Turns green:
         - All tests in TestReliabilityCurve.

    5. `g2c.eval.multiple_choice.score_multiple_choice`. Per-example
       scoring: continuation_logprob over each choice, argmax,
       softmax for confidence. Depends on step 2. Turns green:
         - All tests in TestScoreMultipleChoice.
         - All tests in TestRunMultipleChoiceEval (the harness
           is implemented but it iterates score_multiple_choice).

Boilerplate tests (TestMultipleChoiceExample, TestGenerationExample,
TestEvalResult, TestEvalReport) pass from the start — the dataclasses
are fully implemented.

Strategy for fixtures:
  * Most tests use a `_BiasedLogitsModel` — a stub model that returns
    the same `(vocab_size,)` logits at every position. This lets us
    engineer specific log-prob outcomes without any training.
  * One end-to-end test (`test_run_multiple_choice_eval_real_transformer`)
    wires a tiny `TransformerLM` to verify the harness composes
    correctly with a real PyTorch module.
  * Generation tests use a `_FakeGenerator` — a callable that returns
    a fixed string per prompt — to exercise the harness without
    invoking actual sampling.

The module's design decouples eval from any particular tokenizer or
sampling stack, so a `_CharTokenizer` (each char → its ord) is
sufficient for every fixture except the real-transformer end-to-end.
"""
from __future__ import annotations

import math

import pytest
import torch

from g2c.eval import (
    EvalReport,
    EvalResult,
    GenerationExample,
    MultipleChoiceExample,
    contains_match,
    continuation_logprob,
    exact_match,
    expected_calibration_error,
    normalized_match,
    numeric_match,
    reliability_curve,
    run_generation_eval,
    run_multiple_choice_eval,
    score_generation_example,
    score_multiple_choice,
)


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


class _CharTokenizer:
    """One token per character; vocab_size cap clamps any larger code points."""

    def __init__(self, vocab_size: int = 256) -> None:
        self.vocab_size = vocab_size

    def encode(self, text: str) -> list[int]:
        return [min(ord(c), self.vocab_size - 1) for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i) for i in ids if 0 <= i < self.vocab_size)


class _BiasedLogitsModel:
    """Returns a fixed `(vocab_size,)` logit vector at every position.

    The same logits at every (batch, time) cell — so the model's prediction
    is independent of context. Useful for engineering exact log-prob
    differences between options.
    """

    def __init__(self, vocab_size: int, biases: list[float] | None = None) -> None:
        self.vocab_size = vocab_size
        if biases is None:
            biases = [0.0] * vocab_size
        assert len(biases) == vocab_size
        self._bias = torch.tensor(biases, dtype=torch.float32)
        self.max_seq_len = 1024

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        return self._bias.view(1, 1, -1).expand(B, T, -1).clone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self(x)

    def parameters(self):
        return iter([self._bias])


def _uniform_model(vocab_size: int = 64) -> _BiasedLogitsModel:
    return _BiasedLogitsModel(vocab_size)


def _make_tiny_transformer():
    from g2c.transformer import TransformerLM

    torch.manual_seed(0)
    return TransformerLM(
        vocab_size=128,
        embedding_dim=8,
        num_layers=1,
        num_heads=2,
        max_seq_len=24,
    )


# --------------------------------------------------------------------------
# TestMultipleChoiceExample — boilerplate, pass on `main`
# --------------------------------------------------------------------------


class TestMultipleChoiceExample:
    def test_basic(self):
        ex = MultipleChoiceExample(
            prompt="Q: capital of Spain?",
            choices=["Madrid", "Lisbon"],
            answer_idx=0,
        )
        assert ex.prompt == "Q: capital of Spain?"
        assert ex.choices == ["Madrid", "Lisbon"]
        assert ex.answer_idx == 0

    def test_too_few_choices_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            MultipleChoiceExample(prompt="x", choices=["only_one"], answer_idx=0)

    def test_no_choices_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            MultipleChoiceExample(prompt="x", choices=[], answer_idx=0)

    def test_answer_idx_out_of_range_raises(self):
        with pytest.raises(ValueError, match="answer_idx"):
            MultipleChoiceExample(prompt="x", choices=["a", "b"], answer_idx=2)

    def test_negative_answer_idx_raises(self):
        with pytest.raises(ValueError, match="answer_idx"):
            MultipleChoiceExample(prompt="x", choices=["a", "b"], answer_idx=-1)

    def test_non_str_prompt_raises(self):
        with pytest.raises(TypeError, match="prompt"):
            MultipleChoiceExample(prompt=42, choices=["a", "b"], answer_idx=0)  # type: ignore[arg-type]

    def test_frozen(self):
        from dataclasses import FrozenInstanceError

        ex = MultipleChoiceExample(prompt="q", choices=["a", "b"], answer_idx=0)
        with pytest.raises(FrozenInstanceError):
            ex.answer_idx = 1  # type: ignore[misc]


# --------------------------------------------------------------------------
# TestGenerationExample — boilerplate, pass on `main`
# --------------------------------------------------------------------------


class TestGenerationExample:
    def test_basic(self):
        ex = GenerationExample(prompt="Q: capital of Spain?", references=["Madrid"])
        assert ex.prompt == "Q: capital of Spain?"
        assert ex.references == ["Madrid"]

    def test_multiple_references(self):
        ex = GenerationExample(prompt="x", references=["yes", "yeah", "yep"])
        assert len(ex.references) == 3

    def test_empty_references_raises(self):
        with pytest.raises(ValueError, match="reference"):
            GenerationExample(prompt="x", references=[])

    def test_non_str_prompt_raises(self):
        with pytest.raises(TypeError, match="prompt"):
            GenerationExample(prompt=42, references=["yes"])  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# TestEvalResult / TestEvalReport — boilerplate, pass on `main`
# --------------------------------------------------------------------------


class TestEvalResult:
    def test_basic(self):
        r = EvalResult(correct=True, prediction=0, confidence=0.9)
        assert r.correct is True
        assert r.prediction == 0
        assert r.confidence == 0.9
        assert r.metadata == {}

    def test_default_confidence_is_none(self):
        r = EvalResult(correct=False, prediction="madrid")
        assert r.confidence is None

    def test_default_metadata_is_independent(self):
        r1 = EvalResult(correct=True, prediction=0)
        r2 = EvalResult(correct=False, prediction=1)
        r1.metadata["foo"] = "bar"
        assert r2.metadata == {}


class TestEvalReport:
    def test_basic(self):
        rep = EvalReport(
            task_name="t",
            n=2,
            accuracy=0.5,
            mean_confidence=0.6,
            ece=0.1,
            results=[],
        )
        assert rep.task_name == "t"
        assert rep.n == 2
        assert rep.accuracy == 0.5
        assert rep.mean_confidence == 0.6
        assert rep.ece == 0.1

    def test_repr_includes_task_name_and_accuracy(self):
        rep = EvalReport(
            task_name="my_task",
            n=10,
            accuracy=0.8,
            mean_confidence=0.7,
            ece=0.05,
            results=[],
        )
        s = repr(rep)
        assert "my_task" in s
        assert "0.8" in s

    def test_repr_handles_none_ece(self):
        rep = EvalReport(
            task_name="t",
            n=1,
            accuracy=1.0,
            mean_confidence=None,
            ece=None,
            results=[],
        )
        s = repr(rep)
        assert "—" in s


# --------------------------------------------------------------------------
# TestExactMatch
# --------------------------------------------------------------------------


class TestExactMatch:
    def test_identity(self):
        assert exact_match("Madrid", ["Madrid"]) is True

    def test_case_sensitive(self):
        assert exact_match("Madrid", ["madrid"]) is False

    def test_no_match(self):
        assert exact_match("Madrid", ["Lisbon"]) is False

    def test_multiple_references_one_matches(self):
        assert exact_match("Madrid", ["Lisbon", "Madrid", "Paris"]) is True

    def test_multiple_references_none_match(self):
        assert exact_match("Madrid", ["Lisbon", "Paris"]) is False

    def test_empty_references_raises(self):
        with pytest.raises(ValueError):
            exact_match("Madrid", [])

    def test_returns_bool(self):
        out = exact_match("a", ["a"])
        assert isinstance(out, bool)


# --------------------------------------------------------------------------
# TestNormalizedMatch
# --------------------------------------------------------------------------


class TestNormalizedMatch:
    def test_identity(self):
        assert normalized_match("madrid", ["madrid"]) is True

    def test_case_insensitive(self):
        assert normalized_match("Madrid", ["madrid"]) is True
        assert normalized_match("MADRID", ["madrid"]) is True

    def test_strips_punctuation(self):
        assert normalized_match("Madrid.", ["madrid"]) is True
        assert normalized_match("Madrid,", ["madrid"]) is True
        assert normalized_match("'Madrid'", ["madrid"]) is True

    def test_strips_whitespace(self):
        assert normalized_match("  Madrid  ", ["madrid"]) is True

    def test_collapses_internal_whitespace(self):
        assert normalized_match("Mad   rid", ["mad rid"]) is True

    def test_genuine_difference(self):
        assert normalized_match("Madrid", ["Lisbon"]) is False

    def test_word_boundaries_matter(self):
        # "MADRID" and "MAD RID" differ in word count after normalization.
        assert normalized_match("MAD RID", ["madrid"]) is False

    def test_multiple_references(self):
        assert normalized_match(
            "Madrid.", ["Lisbon", "MADRID"]
        ) is True

    def test_empty_references_raises(self):
        with pytest.raises(ValueError):
            normalized_match("Madrid", [])


# --------------------------------------------------------------------------
# TestContainsMatch
# --------------------------------------------------------------------------


class TestContainsMatch:
    def test_substring_present(self):
        assert contains_match(
            "the answer is Madrid, capital of Spain", ["Madrid"]
        ) is True

    def test_case_insensitive(self):
        assert contains_match("MADRID is the capital", ["madrid"]) is True

    def test_substring_absent(self):
        assert contains_match("the capital is Lisbon", ["Madrid"]) is False

    def test_multiple_references_one_matches(self):
        assert contains_match(
            "the answer is yes", ["yes", "yeah", "yep"]
        ) is True

    def test_empty_references_raises(self):
        with pytest.raises(ValueError):
            contains_match("Madrid", [])


# --------------------------------------------------------------------------
# TestNumericMatch
# --------------------------------------------------------------------------


class TestNumericMatch:
    def test_integer_match(self):
        assert numeric_match("50", ["50"]) is True

    def test_integer_in_text_match(self):
        assert numeric_match("the answer is 50", ["50"]) is True

    def test_match_inside_endmarker(self):
        assert numeric_match("50<|end|>", ["50."]) is True

    def test_float_match_within_tolerance(self):
        assert numeric_match("3.14", ["3.1416"], tolerance=0.01) is True

    def test_float_match_outside_tolerance(self):
        assert numeric_match("3.14", ["3.1416"], tolerance=0.0) is False

    def test_negative_number(self):
        assert numeric_match("-273", ["-273"]) is True
        assert numeric_match("-273", ["273"]) is False

    def test_no_number_in_prediction_returns_false(self):
        assert numeric_match("no number here", ["50"]) is False

    def test_no_number_in_reference_skipped(self):
        # Reference without a number is silently skipped.
        assert numeric_match("50", ["no number"]) is False

    def test_first_number_only(self):
        # "Half of 100 is 50" — first number is 100, not 50.
        assert numeric_match("Half of 100 is 50", ["50"]) is False

    def test_empty_references_raises(self):
        with pytest.raises(ValueError):
            numeric_match("50", [])


# --------------------------------------------------------------------------
# TestContinuationLogprob
# --------------------------------------------------------------------------


class TestContinuationLogprob:
    def test_returns_float_and_int(self):
        model = _uniform_model(vocab_size=64)
        tok = _CharTokenizer(vocab_size=64)
        out = continuation_logprob(model, tok, "ab", "cd")
        assert isinstance(out, tuple)
        assert len(out) == 2
        sum_lp, n = out
        assert isinstance(sum_lp, float)
        assert isinstance(n, int)

    def test_uniform_logits_value(self):
        # Uniform logits → every token has probability 1/V → log p = -log V.
        # Sum over T_cont continuation tokens → -T_cont * log(V).
        V = 64
        model = _uniform_model(vocab_size=V)
        tok = _CharTokenizer(vocab_size=V)
        prompt = "abc"            # 3 tokens
        continuation = "defgh"    # 5 tokens
        sum_lp, n = continuation_logprob(model, tok, prompt, continuation)
        expected = -5 * math.log(V)
        assert n == 5
        assert sum_lp == pytest.approx(expected, rel=1e-5)

    def test_token_count_matches_tokenizer(self):
        V = 64
        model = _uniform_model(vocab_size=V)
        tok = _CharTokenizer(vocab_size=V)
        prompt = "x"
        continuation = "abcdefg"  # 7 chars → 7 tokens
        _, n = continuation_logprob(model, tok, prompt, continuation)
        assert n == len(tok.encode(continuation))
        assert n == 7

    def test_uniform_logits_independent_of_prompt(self):
        # With uniform logits, the log-prob depends only on the
        # continuation length and vocab size — the prompt is irrelevant.
        V = 64
        model = _uniform_model(vocab_size=V)
        tok = _CharTokenizer(vocab_size=V)
        sum_lp_a, _ = continuation_logprob(model, tok, "short", "abc")
        sum_lp_b, _ = continuation_logprob(model, tok, "much longer prompt", "abc")
        assert sum_lp_a == pytest.approx(sum_lp_b, rel=1e-5)

    def test_biased_logits_higher_for_preferred_token(self):
        V = 128
        biases = [0.0] * V
        a_id = ord("a")
        biases[a_id] = 5.0  # model strongly prefers 'a' over 'b'
        model = _BiasedLogitsModel(V, biases)
        tok = _CharTokenizer(vocab_size=V)
        sum_lp_a, _ = continuation_logprob(model, tok, "p", "a")
        sum_lp_b, _ = continuation_logprob(model, tok, "p", "b")
        assert sum_lp_a > sum_lp_b

    def test_empty_prompt_raises(self):
        model = _uniform_model(vocab_size=64)
        tok = _CharTokenizer(vocab_size=64)
        with pytest.raises(ValueError):
            continuation_logprob(model, tok, "", "abc")

    def test_empty_continuation_raises(self):
        model = _uniform_model(vocab_size=64)
        tok = _CharTokenizer(vocab_size=64)
        with pytest.raises(ValueError):
            continuation_logprob(model, tok, "abc", "")

    def test_returns_zero_for_no_grad(self):
        # The function should not accumulate gradients on the model
        # parameters — eval-time only.
        model = _uniform_model(vocab_size=64)
        tok = _CharTokenizer(vocab_size=64)
        # Reset any pre-existing grads.
        for p in model.parameters():
            if hasattr(p, "grad"):
                p.grad = None
        continuation_logprob(model, tok, "p", "abc")
        for p in model.parameters():
            assert p.grad is None or torch.all(p.grad == 0)


# --------------------------------------------------------------------------
# TestExpectedCalibrationError
# --------------------------------------------------------------------------


class TestExpectedCalibrationError:
    def test_perfect_calibration_zero(self):
        # Bin 5 (around confidence=0.5): half are correct.
        # Each prediction confidence == its bin's accuracy.
        confs = [0.5, 0.5, 0.5, 0.5]
        corr = [True, True, False, False]
        ece = expected_calibration_error(confs, corr, n_bins=10)
        assert ece == pytest.approx(0.0, abs=1e-9)

    def test_always_confident_half_correct(self):
        # Always confident=1.0; correct only half the time.
        # All in the last bin: conf=1.0, acc=0.5, |1.0-0.5|=0.5.
        confs = [1.0] * 10
        corr = [True] * 5 + [False] * 5
        ece = expected_calibration_error(confs, corr, n_bins=10)
        assert ece == pytest.approx(0.5, abs=1e-9)

    def test_always_correct_zero_confidence(self):
        # Maximally miscalibrated: confidence=0 every time, all correct.
        confs = [0.0] * 10
        corr = [True] * 10
        ece = expected_calibration_error(confs, corr, n_bins=10)
        assert ece == pytest.approx(1.0, abs=1e-9)

    def test_known_case_two_bins(self):
        # Two bins, hand-computable.
        # Bin 0 (low conf): two examples, conf=[0.0, 0.1], accuracy=0.5.
        #   mean conf = 0.05, |0.5 - 0.05| = 0.45, weight = 2/4 = 0.5.
        # Bin 1 (high conf): two examples, conf=[0.9, 1.0], accuracy=1.0.
        #   mean conf = 0.95, |1.0 - 0.95| = 0.05, weight = 2/4 = 0.5.
        # ECE = 0.5 * 0.45 + 0.5 * 0.05 = 0.25.
        confs = [0.0, 0.1, 0.9, 1.0]
        corr = [True, False, True, True]
        ece = expected_calibration_error(confs, corr, n_bins=2)
        assert ece == pytest.approx(0.25, abs=1e-9)

    def test_in_unit_interval(self):
        # On any input, ECE ∈ [0, 1].
        confs = [0.1, 0.4, 0.7, 0.95]
        corr = [True, False, True, False]
        ece = expected_calibration_error(confs, corr, n_bins=10)
        assert 0.0 <= ece <= 1.0

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            expected_calibration_error([], [], n_bins=10)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            expected_calibration_error([0.5, 0.7], [True], n_bins=10)

    def test_zero_bins_raises(self):
        with pytest.raises(ValueError):
            expected_calibration_error([0.5], [True], n_bins=0)

    def test_n_bins_affects_value(self):
        # Boundary effects: same data, different n_bins, can produce
        # different ECE values.
        confs = [0.05, 0.15, 0.55, 0.75, 0.95]
        corr = [False, False, True, True, True]
        ece_5 = expected_calibration_error(confs, corr, n_bins=5)
        ece_50 = expected_calibration_error(confs, corr, n_bins=50)
        assert isinstance(ece_5, float)
        assert isinstance(ece_50, float)


# --------------------------------------------------------------------------
# TestReliabilityCurve
# --------------------------------------------------------------------------


class TestReliabilityCurve:
    def test_lengths(self):
        confs = [0.1, 0.3, 0.7, 0.9]
        corr = [False, True, True, True]
        bc, ba, bn = reliability_curve(confs, corr, n_bins=10)
        assert len(bc) == 10
        assert len(ba) == 10
        assert len(bn) == 10

    def test_counts_sum_to_n(self):
        confs = [0.1, 0.3, 0.7, 0.9, 0.99]
        corr = [False, True, True, True, True]
        _, _, bn = reliability_curve(confs, corr, n_bins=10)
        assert sum(bn) == len(confs)

    def test_empty_bins_yield_nan(self):
        # Two examples, n_bins=10 — at least 8 bins should be empty.
        confs = [0.1, 0.9]
        corr = [True, True]
        bc, ba, bn = reliability_curve(confs, corr, n_bins=10)
        empty_bins = [i for i, c in enumerate(bn) if c == 0]
        for i in empty_bins:
            assert math.isnan(bc[i])
            assert math.isnan(ba[i])

    def test_perfect_calibration_diagonal(self):
        # All examples in one bin; conf=acc → reliability point is on
        # the diagonal.
        confs = [0.5, 0.5, 0.5, 0.5]
        corr = [True, True, False, False]
        bc, ba, bn = reliability_curve(confs, corr, n_bins=10)
        bin_with_data = [i for i, c in enumerate(bn) if c > 0]
        assert len(bin_with_data) == 1
        i = bin_with_data[0]
        assert bc[i] == pytest.approx(0.5)
        assert ba[i] == pytest.approx(0.5)
        assert bn[i] == 4


# --------------------------------------------------------------------------
# TestScoreMultipleChoice
# --------------------------------------------------------------------------


def _two_option_example(pad_choices_to_same_length: bool = True) -> MultipleChoiceExample:
    if pad_choices_to_same_length:
        return MultipleChoiceExample(
            prompt="p", choices=["a", "b"], answer_idx=0
        )
    return MultipleChoiceExample(
        prompt="p", choices=["a", "bb"], answer_idx=0
    )


class TestScoreMultipleChoice:
    def test_picks_higher_logprob_option(self):
        # Bias toward 'a' (token id ord('a')=97).
        V = 128
        biases = [0.0] * V
        biases[ord("a")] = 5.0
        model = _BiasedLogitsModel(V, biases)
        tok = _CharTokenizer(vocab_size=V)
        ex = _two_option_example()
        result = score_multiple_choice(model, tok, ex)
        assert result.prediction == 0   # picked 'a'
        assert result.correct is True   # gold is 0

    def test_picks_lower_when_biased_against_gold(self):
        V = 128
        biases = [0.0] * V
        biases[ord("b")] = 5.0  # Bias toward 'b' (gold is 'a')
        model = _BiasedLogitsModel(V, biases)
        tok = _CharTokenizer(vocab_size=V)
        ex = _two_option_example()
        result = score_multiple_choice(model, tok, ex)
        assert result.prediction == 1
        assert result.correct is False

    def test_confidence_in_unit_interval(self):
        V = 128
        biases = [0.0] * V
        biases[ord("a")] = 3.0
        model = _BiasedLogitsModel(V, biases)
        tok = _CharTokenizer(vocab_size=V)
        ex = _two_option_example()
        r = score_multiple_choice(model, tok, ex)
        assert 0.0 <= r.confidence <= 1.0

    def test_confidence_above_half_when_predicted_better(self):
        # If the model strictly prefers the predicted option, its
        # softmax-confidence should exceed 1/N = 0.5.
        V = 128
        biases = [0.0] * V
        biases[ord("a")] = 5.0
        model = _BiasedLogitsModel(V, biases)
        tok = _CharTokenizer(vocab_size=V)
        ex = _two_option_example()
        r = score_multiple_choice(model, tok, ex)
        assert r.confidence > 0.5

    def test_uniform_logits_indifferent(self):
        # With uniform logits and equal-length options, all options
        # score equally → confidence == 1/N = 0.5 within fp noise.
        V = 64
        model = _uniform_model(V)
        tok = _CharTokenizer(vocab_size=V)
        ex = MultipleChoiceExample(prompt="p", choices=["a", "b"], answer_idx=0)
        r = score_multiple_choice(model, tok, ex)
        assert r.confidence == pytest.approx(0.5, abs=1e-5)

    def test_metadata_contains_diagnostic_fields(self):
        V = 128
        biases = [0.0] * V
        biases[ord("a")] = 3.0
        model = _BiasedLogitsModel(V, biases)
        tok = _CharTokenizer(vocab_size=V)
        ex = _two_option_example()
        r = score_multiple_choice(model, tok, ex)
        assert "option_logps" in r.metadata
        assert "option_lengths" in r.metadata
        assert "option_scores" in r.metadata
        assert "gold_idx" in r.metadata
        assert len(r.metadata["option_logps"]) == 2
        assert len(r.metadata["option_lengths"]) == 2
        assert r.metadata["gold_idx"] == 0

    def test_length_normalize_changes_decision(self):
        # Engineer an example where length-normalize flips which option
        # wins. Since log-probs are negative, a LONGER continuation
        # always has a more-negative (lower) raw SUM. So:
        #   raw scoring (sum):     SHORTER option wins.
        #   normalized scoring:    whichever option has a higher
        #                          per-token mean log-prob wins.
        # If the long option's tokens are higher-probability per-token
        # than the short option's token, normalize flips the decision.
        V = 128
        biases = [0.0] * V
        # Make 'b' substantially more probable per-token than 'a'.
        biases[ord("b")] = 2.0
        model = _BiasedLogitsModel(V, biases)
        tok = _CharTokenizer(vocab_size=V)
        # Choice 0 = "a"        (1 token).  sum ≈ per-token ≈ −5.18.
        # Choice 1 = "bbbbbb"   (6 tokens). per-token ≈ −3.18, sum ≈ −19.06.
        # Raw:        A wins   (−5.18  > −19.06).
        # Normalized: B wins   (−3.18  > −5.18).
        ex = MultipleChoiceExample(
            prompt="p", choices=["a", "bbbbbb"], answer_idx=0
        )
        r_raw = score_multiple_choice(model, tok, ex, length_normalize=False)
        r_norm = score_multiple_choice(model, tok, ex, length_normalize=True)
        assert r_raw.prediction == 0   # short option wins raw sum
        assert r_norm.prediction == 1  # long option wins per-token mean


# --------------------------------------------------------------------------
# TestRunMultipleChoiceEval
# --------------------------------------------------------------------------


class TestRunMultipleChoiceEval:
    def test_smoke(self):
        V = 128
        biases = [0.0] * V
        biases[ord("a")] = 5.0
        model = _BiasedLogitsModel(V, biases)
        tok = _CharTokenizer(vocab_size=V)
        examples = [
            MultipleChoiceExample(prompt="p1", choices=["a", "b"], answer_idx=0),
            MultipleChoiceExample(prompt="p2", choices=["a", "b"], answer_idx=0),
        ]
        rep = run_multiple_choice_eval(model, tok, examples)
        assert isinstance(rep, EvalReport)
        assert rep.n == 2
        assert rep.accuracy == 1.0
        assert len(rep.results) == 2

    def test_empty_raises(self):
        V = 64
        model = _uniform_model(V)
        tok = _CharTokenizer(V)
        with pytest.raises(ValueError):
            run_multiple_choice_eval(model, tok, [])

    def test_accuracy_reflects_correctness(self):
        # Half the examples have gold=0 (model picks 'a' → correct),
        # half have gold=1 (model picks 'a' → incorrect).
        V = 128
        biases = [0.0] * V
        biases[ord("a")] = 5.0
        model = _BiasedLogitsModel(V, biases)
        tok = _CharTokenizer(V)
        examples = [
            MultipleChoiceExample(prompt="p", choices=["a", "b"], answer_idx=0),
            MultipleChoiceExample(prompt="p", choices=["a", "b"], answer_idx=0),
            MultipleChoiceExample(prompt="p", choices=["a", "b"], answer_idx=1),
            MultipleChoiceExample(prompt="p", choices=["a", "b"], answer_idx=1),
        ]
        rep = run_multiple_choice_eval(model, tok, examples)
        assert rep.accuracy == pytest.approx(0.5)

    def test_compute_ece_false_yields_none(self):
        V = 128
        biases = [0.0] * V
        biases[ord("a")] = 5.0
        model = _BiasedLogitsModel(V, biases)
        tok = _CharTokenizer(V)
        examples = [MultipleChoiceExample(
            prompt="p", choices=["a", "b"], answer_idx=0
        )]
        rep = run_multiple_choice_eval(
            model, tok, examples, compute_ece=False
        )
        assert rep.ece is None

    def test_task_name_propagates(self):
        V = 64
        model = _uniform_model(V)
        tok = _CharTokenizer(V)
        examples = [MultipleChoiceExample(
            prompt="p", choices=["a", "b"], answer_idx=0
        )]
        rep = run_multiple_choice_eval(
            model, tok, examples, task_name="my_task"
        )
        assert rep.task_name == "my_task"

    def test_real_transformer_smoke(self):
        # End-to-end with an actual TransformerLM. Just verifies the
        # harness composes — accuracy on a freshly initialized model
        # is whatever it is.
        torch.manual_seed(0)
        model = _make_tiny_transformer()
        tok = _CharTokenizer(vocab_size=128)
        examples = [
            MultipleChoiceExample(prompt="ab", choices=["c", "d"], answer_idx=0),
            MultipleChoiceExample(prompt="ef", choices=["g", "h"], answer_idx=1),
        ]
        rep = run_multiple_choice_eval(model, tok, examples)
        assert rep.n == 2
        assert 0.0 <= rep.accuracy <= 1.0
        assert all(isinstance(r.confidence, float) for r in rep.results)


# --------------------------------------------------------------------------
# TestRunGenerationEval
# --------------------------------------------------------------------------


class TestRunGenerationEval:
    def test_score_generation_example_correct(self):
        ex = GenerationExample(prompt="Q?", references=["yes"])
        result = score_generation_example(
            ex, generate_fn=lambda p: "yes", matcher=exact_match
        )
        assert result.correct is True
        assert result.prediction == "yes"
        assert result.confidence is None
        assert result.metadata["matcher"] == "exact_match"
        assert result.metadata["references"] == ["yes"]

    def test_score_generation_example_incorrect(self):
        ex = GenerationExample(prompt="Q?", references=["yes"])
        result = score_generation_example(
            ex, generate_fn=lambda p: "no", matcher=exact_match
        )
        assert result.correct is False
        assert result.prediction == "no"

    def test_run_smoke(self):
        examples = [
            GenerationExample(prompt="Q1", references=["a"]),
            GenerationExample(prompt="Q2", references=["b"]),
            GenerationExample(prompt="Q3", references=["c"]),
        ]
        prompt_to_answer = {"Q1": "a", "Q2": "wrong", "Q3": "c"}
        rep = run_generation_eval(
            examples,
            generate_fn=lambda p: prompt_to_answer[p],
            matcher=exact_match,
        )
        assert rep.n == 3
        assert rep.accuracy == pytest.approx(2 / 3)
        assert rep.mean_confidence is None
        assert rep.ece is None

    def test_run_with_normalized_match(self):
        examples = [
            GenerationExample(prompt="Q?", references=["madrid"]),
        ]
        rep = run_generation_eval(
            examples,
            generate_fn=lambda p: "Madrid.",
            matcher=normalized_match,
        )
        assert rep.accuracy == 1.0

    def test_run_with_numeric_match(self):
        examples = [
            GenerationExample(prompt="2+2?", references=["4"]),
            GenerationExample(prompt="3*3?", references=["9"]),
        ]
        rep = run_generation_eval(
            examples,
            generate_fn=lambda p: "the answer is " + ("4" if "2+2" in p else "9"),
            matcher=numeric_match,
        )
        assert rep.accuracy == 1.0

    def test_run_empty_raises(self):
        with pytest.raises(ValueError):
            run_generation_eval([], generate_fn=lambda p: "x", matcher=exact_match)

    def test_run_task_name_propagates(self):
        examples = [GenerationExample(prompt="Q?", references=["a"])]
        rep = run_generation_eval(
            examples,
            generate_fn=lambda p: "a",
            matcher=exact_match,
            task_name="qa",
        )
        assert rep.task_name == "qa"
