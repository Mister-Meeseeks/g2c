# Beyond — Speculative decoding and multi-token prediction

> **Question this module answers:** *Why does a model that predicts one token at a time not have to generate one token at a time?*

![A small drafter model racing ahead to propose a block of four tokens while a large target model checks the whole block in a single forward pass, accepting a prefix and correcting the first mistake, next to a multi-token-prediction head bolted onto the model that drafts for itself.](specdec/BeyondSpecdec-Hero.png)

Module 11's generation loop pays one full forward pass per token, and no amount of hardware changes that the passes happen one after another. Current releases all attack this serial bottleneck the same way: DeepSeek V4 ships an attached drafter module (DSpark) and reports 60–85% faster generation; GLM-5 serves through a parameter-shared multi-token-prediction layer with a reported 2.76-token acceptance length; Qwen3.8 trains an MTP head into both of its open checkpoints. This module builds the load-bearing mechanism — draft, verify in one pass, keep the agreeing prefix — and proves the property that makes it safe to deploy: the output is *exactly* what the target model would have produced alone.

> **This is a Beyond module.** Beyond modules sit outside the numbered course: nothing in Modules 00–20 depends on them, and they are not part of finishing the course. Come here in any order, whenever a model card or paper names the idea and you want the load-bearing version — built, trained, and broken on your own machine.

---
## Before you start

* *Review*
	* [06-language-models](../modules/06-language-models.md) for the fact this module cashes in: one forward pass scores every position in parallel
	* [11-sampling](../modules/11-sampling.md) for the generation loop whose serial cost is the problem
* *Finish*
	* `g2c/transformer` ([09-transformer-block](../modules/09-transformer-block.md))
	* `g2c/sampling` ([11-sampling](../modules/11-sampling.md))
* *Run*
	* `./checkpoints.sh` — the StoryLM reference ladder (1M/5M/30M share `StoryTokenizer`, which is exactly what makes them a ready-made drafter/target family)
	* `./datasets.sh --tiny` if the TinyStories corpus is missing (the MTP head trains on it)
	* `G2C_APPLY_SOLUTIONS=01-11 ./notebook.sh specdec` instead of the plain launch if you're entering without your own implementations

---
## Where this fits in

In Module 06 we learned an interesting property of autoregression. For an already extant sequence, a single forward pass gives us prediction logits for every token in the sequence. This was exploited during training to allow us to get log loss for every token in a single pass over a document. 

Unfortunately Module 11 taught us this efficiency doesn't apply for forward completion. If we have `n` tokens in the prompt, we can't generate token `n+2` (or later) until we know token `n+1`. Which means every token in the completion requires its own full forward pass. The paradox is completion is fast but only if you already know it ahead of time.

That fact doesn't seem that useful on the surface. But the one upshot is *verification* is much cheaper than generating from scratch. If we can have an educated guess for the next `n` tokens, and that verifies correct, then we turn `n` forward completions into just one.

This is the insight behind *speculative decoding*. Think of it like branch prediction from computer architecture. If we can make a cheap educated guess about the next `N` tokens in a completion, and if we're right, then we reduce the cost from `N` forward passes for generation to one forward pass for verification. 

The question is how do we make a good educated guess? In Module 12 we compared different size models in the same family (same training regime, similar shaped architectures). What we learned is smaller models often produce similar outputs to their larger cousins, especially over the next few tokens before trajectories diverge.

Speculative decoding comes down to economics. It works best for large, slow, expensive models. We pair with a smaller, faster, cheaper *drafter model* from the same family as the *target model*. For most tokens, most of the time, the large model is "overpowered". A smaller model will correctly guess the same token as the larger model. 

Therefore, the efficiency of speculative decoding comes down to two factors. One is drafter accuracy: what percentage of its tokens match the target model? The second is the ratio of the target model's cost to the drafter model's cost. Making the drafter smaller and faster improves the latter but usually trades off against the former.

## The big idea

Speculative decoding runs the following completion loop:

1. Drafter model: generate `k` candidate tokens `d₁ … d_k` from the current completion
2. Target model: score the candidate block in one forward pass, producing aligned predictions `t₁ … t_k` plus `t_(k+1)`
3. Accept the longest prefix `d₁ … d_j` for which every aligned pair `d_i = t_i`
4. Discard the remaining candidate suffix `d_(j+1) … d_k`
5. Emit `t_(j+1)` — the correction at the first mismatch, or the bonus token `t_(k+1)` when all `k` drafts match
6. Repeat loop for the full completion

The overall idea is simple, but there are a few subtleties in that loop that are important to notice.

First, carefully note that there's path dependency in the sequence. The first candidate token that's wrong invalidates all the subsequent tokens, even if they verify. The problem is they're only "correct" for a sequence that we won't be using.

```
            cand:     verify:    match:     action:
token 0:    the       the        correct    emit
token 1:    quick     quick      correct    emit
token 2:    brown     racer      wrong      discard
token 3:    fox       fox        correct    discard (correct on wrong sequence)
```

Second, the forward verification pass will always give us one extra token "for free". If one of the candidate tokens is wrong, then the verification token at that slot is what the target model would generate instead. If all of the candidate tokens are correct, then the forward pass finishes with the final token out of sequence. Either way, we always use this "free token" to emit one additional token at the end. This is what keeps the loop moving forward on the completion, even if the draft model gets zero tokens correct.

```
   drafter (cheap):   proposes  d_1 d_2 d_3 d_4          k serial passes, but cheap
   target (1 pass):   predicts  t_1 t_2 t_3 t_4 t_5      logits at every draft position
                                                          + one past the block
   verify:            d_1 = t_1 ✓   d_2 = t_2 ✓   d_3 ≠ t_3 ✗
   emit:              d_1  d_2  t_3          ← accepted prefix + free correction
```

![A position-alignment diagram shows the last prompt token and four draft inputs producing five target predictions in one pass, then accepts only the matching prefix and emits either the first correction or a bonus token after full acceptance.](specdec/BeyondSpecdec-VerificationAlignment.png)
*The target's logits are shifted by one input position: the last prompt position predicts `t₁`, while the final draft position predicts the free `t₅`. This is the alignment pinned by the module's anchor test.*

### Sampling Distribution

An important property of speculative decoding is that it is purely an *inference optimization*. Its presence or the specific drafter model should never change the distribution of output relative to the target model.

Recall that models themselves only produce logits. Actual token output applies a *sampling policy* to those logits. In the previous section we simplified verification to a pairwise comparison of tokens. In actuality there is a specific *verification algorithm* that 1) considers logits of both target and draft models; 2) may be stochastic; and 3) is specific to the target sampling policy. All three levers are necessary, for at least some target sampling policies, to ensure the output distribution is identical. There are two broad classes:

* *Greedy verification* accepts a draft token only where it equals the target's argmax. The emitted sequence is therefore token-for-token identical to plain greedy decoding.

* *Stochastic verification* accepts the next draft token with probability `min(1, p_target / p_draft)`. If rejected, it resamples from the normalized leftover mass of `p_target - p_draft`.

![Four target-probability bars are decomposed into accepted common mass and correction mass, while a stochastic verification gate either emits the sampled draft token or draws from the normalized positive target residual.](specdec/BeyondSpecdec-StochasticVerification.png)
*For each token, accepted mass `min(p,q)` plus correction mass `(p−q)₊` reconstructs the target mass exactly. Draft-only excess is rejected, so the drafter can change work but not the target distribution.*

In both cases, a simple mathematical derivation as well as empirical tests (which we run in the test suite) will show that the speculative decoder is tokenwise identical in distribution to the target model. 

Once we've established pure inference optimization, drafter quality therefore reduces to exactly one metric: the *acceptance length*. Model cards will report a headline like "2.76 average acceptance". With output distribution off the table, acceptance length becomes the whole story of drafter model quality.

### Multi-token prediction

Up until now, we've assumed the drafter is a separate model running in parallel. The alternative is *multi-token prediction (MTP)*. Instead of a whole new model, it works by attaching a small head to the underlying target model (*the trunk model*).

The head learns to predict the token at `t+2` using as inputs the trunk model's just predicted embedding vector at `t+1` as well as the trunk's hidden state at position `t`.  The same approach can be extended to `t+3`, `t+4`, and so on. The head is trained off the same corpus used for training the underlying model. Only with the target of predicting `t+2` from `t+1`. The head is trained purely on top of the trunk, and its training gradient never flows back to any weights in the underlying model.

At decode time, the head simply rides the trunk's forward pass. Each forward pass proposes `N` tokens ahead, which can be used for the next forward pass. The speculative decoding loop is identical to the one above, the only difference being candidate tokens come from the MTP head instead of a separate drafter model. No second model to load, and near-zero drafting cost because the trunk pass happens anyway. 

### The accounting

The clean way to assess speculative decoding efficiency is *target passes per token*:

```
   plain:        1 target pass per token
   
   speculative:  1 target pass per (E[accepted] + 1) tokens
                 + k drafter passes per block          (cheap, but not free)
```

The main knob to turn (besides drafter model selection itself) is block size. The longer the block size, the higher the cumulative probability that verification will reject a candidate token. However a shorter block may leave possibly correct draft tokens on the table by ending the sequence too early.

Longer blocks will never reduce target passes per token, but their overhead cost per token goes up linear, while the target passes per token is subject to decreasing returns on the margin. Whether fewer target passes translate into faster wall-clock efficiency is an engineering question. In this module, like everything else in the course, we measure rather than assume.

### Verification is also a state transaction

The losslessness claim has a second half that a token-only implementation can hide. After an iteration, both of these must match ordinary target decoding:

```text
1. emitted tokens
2. committed inference state after those tokens
```

“Match” here means that the state represents the same accepted prefix and produces the same future target outputs, not that every implementation must use the same byte layout. A KV cache and a checkpoint-plus-replay buffer can be semantically equivalent while storing different objects.

For a Transformer KV cache, speculative verification appends `k` candidate rows. If only the first `j` drafts are accepted, rollback can usually move a sequence pointer to the accepted prefix; the rejected K/V rows remain allocated or are overwritten later, but they are no longer visible. The cache is an append-only token history, so every speculative position has an address.

A recurrent layer has a different problem. Its history has already been folded into one state:

```text
S_t = update(S_(t-1), input_t)
```

After applying four draft inputs, `S_(t+4)` does not retain four rows that can be truncated back to `S_(t+2)`. The compression that made state independent of context length also removed cheap undo. Saving a full state snapshot after every draft makes rollback possible, but multiplies traffic for a decode path that is often already memory-bound.

The durable systems pattern is to treat verification like a transaction:

```text
committed checkpoint state S₀ + recent accepted-input buffer
                         │
                         ├─ append k speculative inputs to a branch
                         ├─ compute outputs needed for verification
                         └─ verifier accepts j
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
             commit first j inputs      discard rejected suffix
                    │                    by moving a buffer pointer
                    └─ later flush: replay buffered inputs into S₀
```

ReplaySSM is one production realization. It keeps a checkpoint recurrent state and caches recent recurrence inputs rather than materializing and storing a full new state at every token. Outputs can be computed from the checkpoint plus that input buffer; when the buffer fills, replay folds the inputs into a new checkpoint. During speculation, accepting or rejecting a suffix becomes a buffer-pointer operation instead of restoration from `k` full-state snapshots.

![Three state-commit strategies accept two of four speculative inputs: a KV cache truncates addressable rows with a sequence pointer, recurrent snapshots restore an earlier full state, and checkpoint-plus-input replay advances a buffer pointer before folding accepted inputs into a new checkpoint.](specdec/BeyondSpecdec-StateTransactions.png)
*Every strategy commits the same two-token boundary, but the stored object changes the cost of rollback: token rows truncate, compressed state must be restored, and an input log can discard a suffix by pointer before replay.*

This does not change the verifier's probability rule or acceptance length. It changes how the target produces verification outputs and how its hidden state commits the accepted prefix. In a hybrid model, the transaction spans both kinds of state: full/sparse attention layers commit cache rows, while recurrent layers commit buffered inputs or a reconstructed state. Every layer must land on the same accepted token boundary.

The course loop deliberately recomputes the prefix and therefore gets correct state by construction. It does not implement ReplaySSM. The bridge matters because it reveals a hidden denominator in speculative-decoding claims: fewer target passes help only if branching, verifying, and committing the target's actual inference state are cheaper than ordinary decode.

## Concepts to internalize

- **Teacher forcing works on guesses.** One forward pass scores `k+1` positions whether the tokens came from a corpus or a drafter. Verification is Module 06's parallel scoring pointed at the future.
- **Verification is what makes speculation lossless.** Greedy: identical tokens. Stochastic: identical distribution. The drafter can only change speed. If your implementation's output differs from plain decoding, it is wrong, not "approximately faster."
- **Every iteration advances.** Accepted prefix plus one free correction/bonus token — the floor is plain decoding, never worse.
- **Acceptance length is the drafter's whole report card.** `E[accepted] + 1` tokens per pass is the number to sweep, and the number model cards quote.
- **MTP is speculation with the drafter built in.** A trained head turns one trunk pass into a draft; "MTP layer" on a model card is a serving feature, not just a training trick.
- **Drafter and target must share a tokenizer.** Draft *ids* are the interface. A vocabulary mismatch isn't degraded — it's meaningless.
- **Losslessness includes state.** After each block, the emitted tokens *and* every target-layer cache or recurrent state must match target-only decoding at the accepted boundary.
- **Rollback depends on the state representation.** A KV cache can truncate token rows; a fixed recurrent summary has no intrinsic undo. Snapshotting, replaying inputs, or reconstructing state are different systems costs around the same verifier.
- **Replay turns recent inputs into a transaction log.** A stable checkpoint plus an input buffer separates speculative work from committed history; commit and rejection can become pointer changes rather than full-state copies.

### What we don't cover

- **Tree and multi-candidate drafting.** Medusa and EAGLE verify a *tree* of candidate continuations in one pass with specialized attention masks. Our single-chain block is the mechanism; trees are its batched generalization.
- **Cached-state integration and ReplaySSM kernels.** Production verification reuses and commits KV caches, recurrent states, or replay buffers across iterations. Our loop recomputes the prefix so the algorithm stays inspectable; Module 16 has the cache this would compose with, while [Beyond Linear Attention](linear-attention.md) supplies the fixed-state recurrence whose rollback is different.
- **Hardware-aware speculation scheduling.** DSpark chooses draft lengths against measured server throughput. That is a serving-systems concern layered on top of the mechanism built here.
- **The full sampled speculative loop.** We build and verify the stochastic acceptance *rule*; the notebook's decoding loop is greedy, where losslessness is checkable by string equality rather than statistics.
- **Distribution-preservation proofs.** The tests check the theorem empirically; the two primary papers prove it in a page each.

---
## What you'll build

Package: `g2c/specdec/`

```python
# verify.py — pure functions, the module's core
def greedy_verify(draft_ids, target_logits): ...        # SCAFFOLDED
    # accepted-prefix length + the target's free correction/bonus token
def speculative_verify(draft_ids, target_probs,
                       draft_probs, *, generator): ...  # SCAFFOLDED
    # the min(1, p_t/p_d) acceptance rule + residual resampling

# generate.py — the loop and its instrument
class SpecStats: ...                                    # implemented
    # target_passes, drafted, accepted, generated;
    # acceptance_rate, tokens_per_pass
def draft_greedy(drafter, ctx_ids, k): ...              # implemented
def speculative_generate(target, drafter, prompt_ids,
                         max_new_tokens, *, k, eos_id): ...  # SCAFFOLDED

# mtp.py — the model drafts for itself
def hidden_states(model, token_ids): ...                # implemented
class MTPHead(Module):                                  # boilerplate implemented
    def forward(self, hidden, next_token_ids): ...      # SCAFFOLDED
def mtp_loss(mtp_logits, token_ids): ...                # SCAFFOLDED
def mtp_propose(base, head, ctx_ids): ...               # implemented
```

`MTPHead` freezes its base the same way Module 13B's `LoRAModel` does — by leaving it out of `parameters()` — so Module 03B's optimizer trains the head and nothing else.

## How to run the tests

Tests live in `tests/test_specdec.py`. Initial state: 5 passed (the stats instrument, provided drafting, and the head's freeze-by-omission), 15 failed. The verification and loop tests run against tiny deterministic table-lookup mock models, so they need no trained checkpoint; the two integration tests at the bottom drive real `TransformerLM`s and additionally need Modules 05–09.

```bash
source .venv/bin/activate

pytest tests/test_specdec.py              # all module tests
pytest tests/test_specdec.py -x           # stop at first failure (recommended)
pytest tests/test_specdec.py -k verify    # the two verification rules
pytest tests/test_specdec.py -k generate  # the speculative loop
pytest tests/test_specdec.py -k mtp       # the MTP head
```

The anchor test: `test_speculative_generate_matches_argmax_chain_with_any_drafter` pairs the target with a completely unrelated drafter and asserts the output still equals the target's own greedy chain. Nearly every bug — the off-by-one in the verification block's logits, a forgotten correction token, accepting one token too many — breaks that equality immediately.

## Exercises

To launch the exercise notebook run:

```bash
./notebook.sh specdec
```

If at any point you want to archive the work in your current notebook and restart fresh:

```bash
./notebook.sh specdec --fresh
```

Written exercises live in the notebook as `Question:` / `Answer:` cells; ask a coding agent for hints or grading when you're ready, and partial submissions are fine — blank answers are skipped.

1. **Prove losslessness on the real ladder.** StoryLM-1M drafts for StoryLM-30M; assert the speculative output is byte-identical to plain greedy decoding across several prompts, then read the `SpecStats`.
2. **Sweep the acceptance curve.** Tokens-per-pass across `k ∈ {1, 2, 4, 8}` for both drafters (1M and 5M). Where does a bigger block stop paying, and which drafter earns its extra cost?
3. **The wall-clock honesty benchmark.** Time plain versus speculative decoding at your best configuration, with MPS synchronization. Report tokens/sec next to tokens/pass, including "the accounting improved and the clock didn't," if that is what you measure.
4. **Train the MTP head.** Bolt `MTPHead` onto frozen StoryLM-5M, train it on TinyStories, and plot the two-ahead loss and top-1 accuracy. How much worse is predicting `x_{t+2}` than `x_{t+1}`?
5. **Self-speculation.** Decode with `mtp_propose` + `greedy_verify` (a five-line loop) and compare its acceptance and tokens-per-pass against the separate-drafter runs from Exercise 2.
6. **Written: the cost model.** With drafter/target cost ratio `c` and measured acceptance, derive tokens per unit of target-compute for your best run, state the break-even acceptance for `k = 4`, and interpret GLM-5's reported 2.76 acceptance length in these terms.

## Pitfalls to expect

- **The verification block's off-by-one.** The `(k+1)` logits rows start at the position *predicting* the first draft token — one before the draft begins. Slicing the last `k` rows instead of the last `k+1` silently verifies against shifted predictions; the anchor equality test fails immediately and the error reads as "everything gets rejected."
- **Forgetting the free token.** On full acceptance the bonus row is a real emission; on rejection the correction is. Drop either and the loop can stall at zero progress or emit one-short blocks that break the equality.
- **Comparing across different context crops.** Plain `generate` crops to `max_seq_len`; the speculative loop must leave room for the draft block. Inside the window the outputs match exactly; if your equality check strays near the window edge, the divergence is crop semantics, not a bug.
- **Timing MPS without synchronization.** The Python timer stops before Metal finishes. Synchronize around every measured region, and report medians of repeats.
- **Expecting wall-clock to follow tokens-per-pass at toy scale.** Our verification recomputes the full prefix and the drafter loops in Python. The accounting win is real and measured; the latency win depends on machinery (KV cache, batching) this module deliberately doesn't build.
- **The MTP shift-by-one.** Misalign `mtp_loss` and the head cheerfully learns to predict `x_{t+1}` — a job the base already does perfectly — and every draft it proposes later gets rejected. The alignment test pins the correct shift; trust it over your intuition.
- **Cross-family drafting.** A drafter with a different tokenizer produces ids the target reads as arbitrary noise. Acceptance near zero with no error message is the symptom.

## M-series notes

- **Checkpoint downloads:** the ladder is ~5MB (1M), ~22MB (5M), and ~117MB (30M). All three load comfortably on any course-spec machine.
- **Verification passes recompute the whole prefix,** so per-pass cost grows with context. Keep benchmark prompts modest (a few hundred tokens) and growth stays invisible; this module's numbers are about pass *counts*.
- **The MTP training block is the compute cost of the module** — a few hundred head-only steps on TinyStories, minutes on MPS. Plug in for it. Everything else runs in seconds.
- **Timing cells synchronize MPS** before and after each measured region; keep that pattern if you add measurements of your own.

---
## Reading

Primary:

- **Leviathan, Kalman, Matias, "Fast Inference from Transformers via Speculative Decoding" (2022).** The draft-and-verify loop and the acceptance rule — this module is this paper, built. Read it first.
- **Chen, Borgeaud, Irving, Lespiau, Sifre, Jumper, "Accelerating Large Language Model Decoding with Speculative Sampling" (2023).** Independent derivation of the same rule, with the cleanest one-page proof that the target distribution survives.
- **Gloeckle, Idrissi, Rozière, Lopez-Paz, Synnaeve, "Better & Faster Large Language Models via Multi-token Prediction" (2024).** MTP as a *training* objective, before it became a serving feature.

Secondary:

- **DeepSeek-V3 Technical Report (2024), the MTP section.** The shared-trunk sequential MTP module this module's head is a minimal version of, and the design its successors inherit.
- **Cai et al., "Medusa" (2024)** and **Li et al., "EAGLE" (2024).** Multiple heads, tree-structured drafts, and feature-level drafting — the generalizations past a single chain.
- **Dao AI Lab, ["ReplaySSM: Cache SSM Inputs, Not State"](https://dao-lab.ai/blog/2026/replayssm/) (2026).** The state-transaction bridge for recurrent and delta-rule targets: checkpoint state, buffered recurrence inputs, cheap suffix rollback, and output-only speculative verification. Read for the systems contract; this module does not build its kernels.

Optional:

- **The current release lineage.** DeepSeek V4's DSpark report, GLM-5's parameter-shared MTP (2.76 acceptance), and the Qwen3.8 model cards' MTP entries show different places a drafter can live. [GLM-5.3-Flash](https://z.ai/blog/glm-5.3-flash) adds ReplaySSM to the serving side. After this module, read those systems as a drafter, a verifier, an acceptance number, and a state-commit path—not just a headline speedup.

## Deliverable checklist

- [ ] All tests in `tests/test_specdec.py` pass — the any-drafter equality anchor especially.
- [ ] Notebook: losslessness demonstrated on the StoryLM ladder (Exercise 1), the acceptance sweep (Exercise 2), and the honest wall-clock benchmark (Exercise 3).
- [ ] The MTP head trained, and self-speculation measured against separate-drafter speculation (Exercises 4–5).
- [ ] Written answer for the cost model (Exercise 6).
- [ ] You can explain — out loud, without notes — why greedy speculative decoding cannot change the output, only the pass count.
- [ ] You can explain — out loud, without notes — what the stochastic acceptance rule guarantees, and what the drafter can and cannot influence.
- [ ] You can explain — out loud, without notes — why tokens-per-pass and wall-clock speedup are different claims.
- [ ] You can explain — out loud, without notes — what an MTP head changes about where the drafter lives, and why the two models must share a tokenizer.
- [ ] You can explain — out loud, without notes — why token equality is insufficient if rejected drafts remain folded into target state.
- [ ] You can contrast KV-cache truncation, per-draft recurrent-state snapshots, and checkpoint-plus-input replay as three ways to commit the accepted prefix.
