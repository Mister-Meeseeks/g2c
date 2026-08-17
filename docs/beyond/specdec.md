# Beyond — Speculative decoding and multi-token prediction

> **Question this module answers:** *Why does a model that predicts one token at a time not have to generate one token at a time?*

<!-- TODO(hero pipeline): asset not yet generated -->
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

Module 06 established something that looked like a training convenience: a causal LM's forward pass produces a prediction at *every* position simultaneously. Teacher forcing used it to score `T` positions per pass during training. Module 11 then built generation and quietly gave the parallelism back — each new token requires a fresh pass, because position `t+1`'s input is position `t`'s output.

Speculative decoding notices that the parallelism was never actually lost; it just needs *candidate* tokens to score. If something cheap guesses the next `k` tokens, the expensive model can check all `k` guesses in one forward pass — the same teacher forcing, applied at decode time to a guessed future instead of a recorded past. Wherever the guesses match what the target would have said, serial passes were skipped. Where they diverge, the target's own prediction is sitting right there in the verification pass, free.

Two sources of guesses matter in current systems, and this module builds both:

* **A separate drafter** — a smaller model from the same family. Your StoryLM ladder is exactly this: StoryLM-1M drafting for StoryLM-30M, legal because they share one tokenizer.
* **A multi-token-prediction head** — a small attachment that predicts one step *further* ahead from the model's own hidden state, so the model drafts for itself. This is what "MTP" means on the DeepSeek, GLM, and Qwen model cards, and its serving role is speculation with the drafter built in.

## The big idea

Generation is serial because of a data dependency, not a law of arithmetic:

```
   generate:  tok_1 → tok_2 → tok_3 → ...     each needs the previous
   score:     given [tok_1 ... tok_k],        one pass judges ALL of them
```

Speculative decoding converts generation into mostly-scoring:

```
   drafter (cheap):   proposes  d_1 d_2 d_3 d_4          k serial passes, but cheap
   target (1 pass):   predicts  t_1 t_2 t_3 t_4 t_5      logits at every draft position
                                                          + one past the block
   verify:            d_1 = t_1 ✓   d_2 = t_2 ✓   d_3 ≠ t_3 ✗
   emit:              d_1  d_2  t_3          ← accepted prefix + free correction
```

Every iteration emits `accepted + 1` tokens for one target pass: the accepted prefix, plus either the target's correction at the first disagreement or — if the whole block survived — the target's *bonus* prediction one past it. The floor is one token per pass, which is exactly plain decoding: speculation can fail to help, but it cannot do worse than the loop it replaces.

The property that makes this an *inference optimization* rather than a new sampling policy is the verification rule:

* **Greedy verification** accepts a draft token only where it equals the target's argmax. The emitted sequence is therefore token-for-token identical to plain greedy decoding — the drafter has no influence on *what* is generated, only on how many passes it took.
* **Stochastic verification** (the Leviathan/Chen rule) accepts draft token `x` with probability `min(1, p_target(x)/p_draft(x))` and, on rejection, resamples from the normalized leftover mass `(p_target − p_draft)⁺`. The theorem — which the test suite checks empirically — is that emitted tokens are distributed *exactly* according to the target, no matter how bad the drafter is.

Drafter quality therefore moves exactly one number: the acceptance length. That is why model cards can report "2.76 average acceptance" as a headline — it is the whole story of how well the drafter and target agree, and the output distribution is off the table by construction.

### Multi-token prediction: the drafter moves inside

A separate drafter costs a second model. The MTP alternative attaches a small head to the base model: given the trunk's hidden state at position `t` and the embedding of the just-predicted token `x_{t+1}`, predict `x_{t+2}`. Trained with teacher forcing (the corpus supplies `x_{t+1}` and `x_{t+2}`), it rides the trunk's forward pass at decode time and proposes one token ahead — self-speculation with `k = 1`, no second model to load, and near-zero drafting cost because the trunk pass was happening anyway. Production variants chain several such steps; the mechanism is identical.

### The accounting and the honesty line

The clean way to think about the win is *target passes per token*:

```
   plain:        1 target pass per token
   speculative:  1 target pass per (E[accepted] + 1) tokens
                 + k drafter passes per block          (cheap, but not free)
```

Whether fewer target passes become faster wall-clock is an engineering question this module measures rather than assumes. At course scale, each verification pass recomputes the whole prefix (our loop does not carry Module 16's KV cache through verification), and the drafter's `k` steps run in a Python loop. The exercises treat "tokens per target pass improved, wall-clock didn't" as a legitimate, reportable result — the same honesty line Beyond Linear Attention draws between demonstrated state savings and demonstrated speed.

## Concepts to internalize

- **Teacher forcing works on guesses.** One forward pass scores `k+1` positions whether the tokens came from a corpus or a drafter. Verification is Module 06's parallel scoring pointed at the future.
- **Verification is what makes speculation lossless.** Greedy: identical tokens. Stochastic: identical distribution. The drafter can only change speed. If your implementation's output differs from plain decoding, it is wrong, not "approximately faster."
- **Every iteration advances.** Accepted prefix plus one free correction/bonus token — the floor is plain decoding, never worse.
- **Acceptance length is the drafter's whole report card.** `E[accepted] + 1` tokens per pass is the number to sweep, and the number model cards quote.
- **MTP is speculation with the drafter built in.** A trained head turns one trunk pass into a draft; "MTP layer" on a model card is a serving feature, not just a training trick.
- **Drafter and target must share a tokenizer.** Draft *ids* are the interface. A vocabulary mismatch isn't degraded — it's meaningless.

### What we don't cover

- **Tree and multi-candidate drafting.** Medusa and EAGLE verify a *tree* of candidate continuations in one pass with specialized attention masks. Our single-chain block is the mechanism; trees are its batched generalization.
- **KV-cache integration.** Production verification reuses the cache across iterations and rolls back rejected suffixes. Our loop recomputes the prefix so the algorithm stays inspectable; Module 16 has the cache this would compose with.
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

Optional:

- **The current release lineage.** DeepSeek V4's DSpark report, GLM-5's parameter-shared MTP (2.76 acceptance), and the Qwen3.8 model cards' MTP entries (2026). After this module, their serving sections read as configuration: a drafter, a verification rule, and an acceptance number.

## Deliverable checklist

- [ ] All tests in `tests/test_specdec.py` pass — the any-drafter equality anchor especially.
- [ ] Notebook: losslessness demonstrated on the StoryLM ladder (Exercise 1), the acceptance sweep (Exercise 2), and the honest wall-clock benchmark (Exercise 3).
- [ ] The MTP head trained, and self-speculation measured against separate-drafter speculation (Exercises 4–5).
- [ ] Written answer for the cost model (Exercise 6).
- [ ] You can explain — out loud, without notes — why greedy speculative decoding cannot change the output, only the pass count.
- [ ] You can explain — out loud, without notes — what the stochastic acceptance rule guarantees, and what the drafter can and cannot influence.
- [ ] You can explain — out loud, without notes — why tokens-per-pass and wall-clock speedup are different claims.
- [ ] You can explain — out loud, without notes — what an MTP head changes about where the drafter lives, and why the two models must share a tokenizer.
