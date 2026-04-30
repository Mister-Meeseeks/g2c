# Module 15 — Hallucination and evaluation

> **Question this module answers:** *Why does the model confidently invent things, and how do we measure it?*

![Module 15 on one page: the Module 13 SFT'd / Module 14 DPO'd TransformerLM under three eval lenses arranged in vertical strips. STRIP 1 (top): "FACTUAL Q&A — generation eval." Prompt: "What is the largest city in Spain?" Model emits "Lisbon.<|end|>" with confidence "exp(mean log-prob) = 0.34". The matcher (`normalized_match`) compares "lisbon" against references ["madrid"] → False. STRIP 2 (middle): "MULTIPLE CHOICE — closed-set eval." Same prompt, but with four candidate continuations: "Madrid.<|end|>", "Lisbon.<|end|>", "Barcelona.<|end|>", "Berlin.<|end|>". The harness scores each by sequence-log-probability, takes argmax, returns the predicted index plus its softmax confidence. The model picks index 1 (Lisbon) with confidence 0.42 — wrong, but only mildly confident. The right side of the strip shows the four option_logps as a small bar chart. STRIP 3 (bottom): "CALIBRATION — across the eval set." Reliability diagram: a 10-bin histogram on [0,1] confidence; for each bin, the bar is the bin's empirical accuracy and a red dot marks its mean confidence. The diagonal "y=x" is the perfectly-calibrated line. The model's bars sit BELOW the diagonal in the high-confidence bins (over-confident) and ABOVE in the low-confidence bins (under-confident) — the canonical signature of a small, under-trained LM. ECE printed below: 0.187. A right-edge sidebar lists the four matchers (`exact`, `normalized`, `numeric`, `contains`) and which task type each is best for. A bottom strip captions the headline: "Loss curves measure training; eval harnesses measure capability. Multiple choice is cheap and calibration-friendly; generation is realistic and matcher-dependent. ECE catches over-confidence that accuracy alone hides."](15-evaluation/Module15-Hero.png)

*The whole module on one page. Module 14's loss curve told you "DPO is doing something"; Module 15 tells you **what** it's doing. The two-pronged eval — closed-set scoring for capability ranking, open-set generation for behavior — plus a calibration metric for "how well does the model know what it doesn't know" — is the minimum viable harness for any serious model work. Loss alone doesn't suffice; this is the lens.*

## Prerequisites

Module 15 closes Phase IV. After Modules 13 and 14 you have a model with shaped behavior; after Module 15 you have a *measurement tool* for that behavior. Everything in this module is "harness code" — pure evaluation, no new model architecture, no new training loss. The work product is a tiny but real eval framework that you'll keep using through the capstone.

### Math

- **Sequence log-probability as a scoring primitive.** The model's belief about a continuation `y` given a prompt `x` is `log π(y | x) = Σ_t log π(y_t | x, y_{<t})`. Module 14 established this for DPO; here it's the workhorse of multiple-choice scoring. We compute it for each candidate continuation, take the argmax, call that the model's "answer." Conceptually: the model's "score" for an option is the joint log-probability of the option's tokens conditional on the prompt — exactly the autoregressive factorization, summed over the option.

- **Length normalization.** Raw `log π(y | x)` advantages SHORTER continuations (fewer log-probs to sum, each one negative). For options of similar token count this is fine; for options of wildly different length, the raw score is misleading. Length-normalized score is `(1/T_y) · log π(y | x)` — the per-token mean log-prob. This is roughly length-invariant. Multiple choice harnesses in the wild expose both as a knob: lm-eval-harness calls them `acc` (raw) and `acc_norm` (normalized).

- **Softmax over options as confidence.** Given option-level scores `s_1, ..., s_N`, the model's "confidence in option k" is the softmax probability `exp(s_k) / Σ_i exp(s_i)`. This is calibration-friendly: it's a probability on `[0, 1]`, summable to 1 across options, and directly comparable to the empirical accuracy. A perfectly calibrated model has `confidence == empirical accuracy` averaged over its predictions.

- **Expected calibration error** (Naeini et al. 2015, Eq. 1):

  ```
      ECE  =  Σ_b  (|B_b| / N) · | acc(B_b) − conf(B_b) |
  ```

  where `B_b` indexes a confidence bin (typically 10 equal-width bins on `[0, 1]`), `acc(B_b)` is the bin's empirical accuracy, and `conf(B_b)` is the bin's mean confidence. ECE is on `[0, 1]`, lower is better. The headline scalar of "is this model honest about how confident to be."

- **Reliability curves.** ECE summarizes calibration; the reliability curve plots it. For each confidence bin, plot `(mean_confidence, accuracy)`. A perfectly calibrated model's points sit on the diagonal `y = x`. Above the diagonal: under-confident. Below: over-confident. Tiny LMs almost always sit below the diagonal — *over-confidence is the default*.

- **The accuracy / confidence / calibration triangle.** Three axes that don't reduce to each other:
  - **Accuracy**: how often is the model right?
  - **Mean confidence**: how confident is it on average?
  - **Calibration (ECE)**: do its confidences match its accuracy?

  A high-accuracy model can be miscalibrated (it's right but says it's certain when it isn't); a low-accuracy model can be calibrated (it's wrong, knows it's wrong, says so). Production practice tracks all three.

### Computer science

- **Closed-set vs open-set evaluation.** Closed-set eval (multiple choice): the answer space is a finite list, and we score each entry. Open-set eval (generation): the answer space is "all possible token sequences," and we let the model emit one and check if it matches. Closed-set is fast, deterministic, and calibration-friendly; open-set is realistic but matcher-dependent and slower (requires `generate`). Real harnesses (HELM, lm-eval-harness, Eleuther Eval) support both; we build both, in their minimum-viable forms.

- **Matchers as plug-in functions.** A matcher is just `Callable[[str, list[str]], bool]`. Four are bundled:
    - `exact_match` — character-equality.
    - `normalized_match` — case/punctuation/whitespace stripped, then equality.
    - `contains_match` — any reference appears as a substring.
    - `numeric_match` — first number in prediction matches first number in some reference, within tolerance.

  The harness takes a matcher as a parameter; the matcher does its job. This decoupling means: "factual QA," "arithmetic," and "instruction-following" are not three different harnesses — they're the same harness with three different matchers.

- **`generate_fn` as a closure.** The generation harness doesn't import `g2c.sampling.generate` directly. Instead it takes a `generate_fn: Callable[[str], str]` parameter — the user passes in any callable from "prompt string" to "generated string." This decouples eval from sampling: you can plug in your trained model + your sampling function, an Ollama HTTP call, an MLX model, or a hard-coded answer-key. The harness doesn't care.

- **`continuation_logprob` as a tokenize-aware wrapper.** The DPO module exposes `sequence_logprob(logits, targets, mask)` — a *tensor-level* primitive. Eval-time scoring needs a *text-level* primitive: `continuation_logprob(model, tokenizer, prompt, continuation) -> (sum_logp, n_tokens)`. Same math (log-softmax + gather + masked sum), but it does the tokenization, model call, and mask construction itself. The tokenizer + model + mask boundary live inside this function so callers can think in strings.

- **Bin assignment in ECE.** Bin index for confidence `c` is `min(int(c * n_bins), n_bins - 1)`. The `min` is for the edge case `c == 1.0`, which would otherwise overflow into bin `n_bins`. A common bug: omitting the `min` and crashing on confidence-exactly-1.0. The other common bug: using `floor(c * n_bins)` without the `min`, then off-by-one in the last bin.

### Programming

- **`@dataclass` and `@dataclass(frozen=True)`** — the standard way to build immutable test fixtures with constructor validation via `__post_init__`. The eval module uses both: `MultipleChoiceExample` and `GenerationExample` are frozen (you don't mutate questions); `EvalResult` and `EvalReport` are mutable (the harness fills `metadata` after construction).
- **`torch.no_grad()`** — wraps the forward in `continuation_logprob`. Eval is inference-only; skipping autograd graph cuts memory roughly in half.
- **`torch.gather`** — same idiom as DPO/SFT. Pull `log p(target_t)` out of the `(1, T, V)` log_softmax tensor at each position `t`.
- **`re.compile`** + `.search` — for `numeric_match`'s "first number" extraction. The pattern `-?\d+(?:\.\d+)?` covers integers, signed integers, and decimals; ignores scientific notation and comma-separated thousands.
- **`string.punctuation`** — all ASCII punctuation as a single string, used to strip in `normalized_match`. Combined with `str.maketrans("", "", string.punctuation)`, this is a one-line "remove all punctuation" pattern.

### What you can skip

- **HELM-style multi-metric harnesses.** HELM (Liang et al. 2022) tracks ~7 metrics per task across 30+ tasks for hundreds of models. Our harness tracks accuracy + ECE on two task types over a hand-built dataset of 50–200 examples. The architecture you'd want to scale this is a "scenario / metric / model" three-way matrix; that's out of scope. The lesson is the loop, not the matrix.
- **Model-graded ("LLM-as-judge") evaluation.** Zheng et al. 2023 argue (correctly) that for many open-ended tasks, asking a strong model to *judge* outputs is more useful than any string matcher. Our toy model isn't strong enough to be a judge, and the matchers we ship are sufficient for the kinds of probes the exercises ask for. We mention LLM-as-judge in "Reading"; you don't implement it.
- **Adversarial robustness benchmarks.** AdvGLUE, RealToxicityPrompts, etc. expose models to adversarial inputs designed to elicit failures. At toy scale these are mostly noise — our model fails on plain inputs already. Read the papers; don't build the benchmarks.
- **Full MMLU.** MMLU is 57 subjects × ~100 questions = ~14k multiple-choice questions. Running it takes hours even on a 7B model. We use a hand-built MC subset of 20–50 questions for the exercise.
- **Calibration via Platt scaling or temperature scaling.** Guo et al. 2017 propose post-hoc calibration: train a one-parameter "temperature" on a held-out set so the model's confidences align better with empirical accuracy. Useful, well-known, and a clean follow-up to ECE — but it's recalibration, not measurement, and we're focused on measurement here.
- **Perplexity-based eval at scale.** WikiText, The Pile, etc. give per-token cross-entropy on huge held-out corpora. We touched on per-token cross-entropy in Module 10; we don't reuse it here because it doesn't decompose into per-task scores. The point of this module is *per-task evaluation*, where loss curves don't help.

## Why we start here

After Modules 13 and 14 you have a behavior-shaped model. The loss curves on both told you "training is doing something" — SFT loss decreased, DPO reward margin increased. But what does "something" cash out to in terms of capability?

```
   ┌──────────────────────────────────────────────────────────────────────┐
   │  THE PROBLEM WITH LOSS CURVES                                         │
   ├──────────────────────────────────────────────────────────────────────┤
   │                                                                       │
   │   SFT loss   ────────┐                                                │
   │                       │                                               │
   │                       └──►  "is the model formatting answers?"       │
   │                                                                       │
   │   DPO margin ────────┐                                                │
   │                       │                                               │
   │                       └──►  "is the chosen-vs-rejected gap growing?" │
   │                                                                       │
   │   ────────────────────────────────────────────────────────────────   │
   │                                                                       │
   │   But neither answers:                                                │
   │                                                                       │
   │     • Does the model get *factual* questions right?                   │
   │     • Is it *calibrated* — when it says "I'm sure," is it sure?       │
   │     • What does it hallucinate, and how often?                        │
   │     • Where does its capacity end?                                    │
   │                                                                       │
   │   These are eval questions. The loss curve doesn't see them.          │
   │                                                                       │
   └──────────────────────────────────────────────────────────────────────┘
```

A 20M-param Module-13 SFT'd model on the prompt:

```
<|user|>
What is 2 + 2?
<|assistant|>
```

might emit:

```
4.<|end|>
```

A grade-school correct answer. But:

```
<|user|>
What is 13 + 28?
<|assistant|>
```

might emit:

```
35.<|end|>
```

Confidently wrong. The SFT loss never saw "13 + 28" specifically; it saw the *format*. The model learned the format but doesn't know arithmetic. To see this, you need an eval harness that:

1. **Asks specific questions** — not "is the loss low" but "is *this answer* right."
2. **Aggregates** — accuracy across many such questions, not anecdotes.
3. **Measures calibration** — when the model says "35," does it actually believe 35, or is it just emitting a number-shaped string?

That's what Module 15 builds. Two harnesses (multiple-choice + generation), four matchers (exact, normalized, numeric, contains), one calibration metric (ECE), one reliability curve. ~250 lines of code in `g2c/eval/`. The pedagogical content is small but load-bearing: every later module (capstone evals, agent task success, RAG retrieval quality) reuses these primitives.

```
   ┌──────────────────────────────────────────────────────────────────────┐
   │  TWO EVAL HARNESSES, FOUR MATCHERS, ONE CALIBRATION METRIC           │
   └──────────────────────────────────────────────────────────────────────┘

      ┌──────────────────────────────────┐        ┌────────────────────────┐
      │  CLOSED-SET (multiple choice)    │        │  OPEN-SET (generation) │
      │                                  │        │                        │
      │  prompt                          │        │  prompt                │
      │    + N candidate continuations   │        │    + reference answers │
      │                                  │        │                        │
      │  ──► score each via              │        │  ──► generate_fn(prompt)│
      │      continuation_logprob        │        │       returns one      │
      │                                  │        │       generated string │
      │  ──► argmax = prediction         │        │                        │
      │  ──► softmax = confidence        │        │  ──► matcher(pred, refs)│
      │                                  │        │       returns bool     │
      │  ──► ECE over the eval set       │        │                        │
      │      compares confidence to      │        │  (no confidence — open │
      │      empirical accuracy          │        │   set, can't normalize)│
      │                                  │        │                        │
      └──────────────────────────────────┘        └────────────────────────┘

      Matchers (independent, swappable):
        exact_match       — character-identity
        normalized_match  — case/punctuation/whitespace stripped, equality
        contains_match    — any reference is a substring (case-insensitive)
        numeric_match     — first number matches, within tolerance

      Aggregation:
        EvalReport(
            n, accuracy, mean_confidence, ece, results
        )
```

The two harnesses are 60 lines each, mostly bookkeeping. The math lives in `continuation_logprob` (text-level scoring), `score_multiple_choice` (the argmax + softmax over options), and `expected_calibration_error` (the bucket loop). Those are what you implement; the rest is plumbing.

A non-obvious framing: **multiple-choice eval is a quiet diagnostic on the model's internal probability landscape.** When you ask the model `Madrid? Lisbon? Barcelona? Berlin?` and read its softmax over options, you're querying the *implicit world model* the network learned during pretraining and behavior shaping. A confident-and-correct model has world knowledge; a confident-and-wrong model has confidently-wrong world knowledge (the textbook hallucination case); an uncertain model is honest about not knowing. ECE tells you which of these regimes the model is in.

## The big idea

### Fluency vs truth as separate objectives

The deepest pathology of small LMs is that they're trained on **fluency** but judged on **truth** (or helpfulness, or correctness, etc.). A model that emits well-formed sentences gets low cross-entropy loss. A model that emits *true* sentences gets... also low cross-entropy loss, *if* the training data was true. But the loss doesn't separate the two.

```
   ┌──────────────────────────────────────────────────────────────────────┐
   │   THE TRAINING OBJECTIVE                                              │
   ├──────────────────────────────────────────────────────────────────────┤
   │                                                                       │
   │     loss  =  − Σ_t log π(target_t | context)                         │
   │                                                                       │
   │   This rewards:                                                       │
   │     ✓ matching the surface form of the training data                 │
   │     ✓ producing fluent, well-formed text                             │
   │     ✓ following the chat template (post-SFT)                         │
   │     ✓ preferring chosen over rejected (post-DPO)                     │
   │                                                                       │
   │   This does NOT reward:                                               │
   │     ✗ being factually correct                                         │
   │     ✗ admitting uncertainty                                           │
   │     ✗ refusing to invent things                                       │
   │     ✗ stopping at the right time                                     │
   │                                                                       │
   └──────────────────────────────────────────────────────────────────────┘
```

A fluent confident wrong answer is *exactly* what the loss rewards as much as a fluent confident right one — both are "low loss" if the surface form matches conventional patterns. Hallucination is the predictable consequence: the model emits the *shape* of an answer, but the *content* is whatever its training distribution most readily produced for prompts of that shape.

The eval harness is what closes this loop. By scoring outputs against **what we wanted, not what the loss measured**, we surface the gap. Most of the model's failures are *expected* given the training objective; eval makes them *visible*.

### Calibration: the headline scalar

```
   ┌──────────────────────────────────────────────────────────────────────┐
   │   ACCURACY vs CALIBRATION                                             │
   ├──────────────────────────────────────────────────────────────────────┤
   │                                                                       │
   │     accuracy   =  fraction of predictions that are correct           │
   │     confidence =  the model's stated probability for its prediction  │
   │                                                                       │
   │     Calibrated:                                                       │
   │       on the subset of predictions where confidence ≈ p,             │
   │       the empirical accuracy is also ≈ p.                            │
   │       — for ALL p in [0, 1].                                         │
   │                                                                       │
   │     Over-confident: confidence > accuracy                             │
   │     Under-confident: confidence < accuracy                            │
   │                                                                       │
   └──────────────────────────────────────────────────────────────────────┘
```

A model can be:

- **High accuracy, well calibrated.** Best case. Production-ready.
- **High accuracy, miscalibrated.** Right often, but says "I'm sure" when it isn't. Dangerous in tool-using or agentic settings — the model takes high-stakes actions on weakly-grounded beliefs.
- **Low accuracy, well calibrated.** Wrong often, but knows it's wrong. Safer than the previous case — the user can attend to the confidence signal and fall back to verification.
- **Low accuracy, miscalibrated.** Wrong and confident. Catastrophic. This is what *small over-trained LMs default to* — exactly the regime your toy model is in.

The reliability diagram makes this visible:

```
                       Reliability diagram (toy LM, 100 MC questions)
                            ▲
                       1.0  │                                    ●
                            │                          ●
                            │                ●
                  empirical │      ●                                Diagonal
                  accuracy  │                                       (perfect
                       0.5  │  ●                                     calib)
                            │                                    ╱
                            │                              ╱
                            │                        ╱
                            │                  ╱                    Model:
                            │            ╱                          ● = bin's
                       0.0  │      ╱                                    (conf,
                            │ ╱                                          acc)
                            └────────────────────────────────────────►
                                  0.0           0.5           1.0
                                          model confidence

         Model is OVER-confident in high-confidence bins (dots BELOW the line)
         and UNDER-confident in low-confidence bins (dots ABOVE the line).

         The horizontal distance from each dot to the diagonal at the same
         y-coordinate is the bin's contribution to ECE.
```

The model's points sit on the diagonal in the middle bins (confidence ≈ accuracy ≈ 0.5) but drift below in the high-confidence bins (the model says 0.95, gets it right 0.7 of the time). That gap — across all bins, weighted by bin frequency — is ECE.

### Multiple choice: scoring continuations

Multiple-choice scoring is the most pedagogically clean eval primitive. Given a prompt and N candidate continuations, the model scores each by sequence-log-probability:

```
   prompt:                "<|user|>\nLargest city in Spain?\n<|assistant|>\n"
   choices:               ["Madrid.<|end|>",                  ← score = -2.34
                           "Lisbon.<|end|>",                  ← score = -1.82  ← argmax
                           "Barcelona.<|end|>",               ← score = -3.41
                           "Berlin.<|end|>"]                  ← score = -4.12
   answer_idx (gold):     0
                                                                       │
                                                                       ▼
   prediction = argmax(scores) = 1     (the model said "Lisbon")  ──►  WRONG
                                                                       │
   confidence = softmax(scores)[1]                                     ▼
              = exp(-1.82) / Σ_i exp(-score_i)
              = 0.42                            (mildly confident)
```

The softmax-over-options confidence is the calibration-friendly readout. A confidence-aware accuracy metric:

```
                  Σ_i 1[prediction_i correct] · weight_i
   weighted_acc = ─────────────────────────────────────
                            Σ_i weight_i
```

with `weight_i = confidence_i` recovers a "predictions weighted by certainty" measure. Useful when comparing model variants.

A useful subtlety: **multiple-choice scoring is sensitive to surface forms.** If your gold answer is "Madrid." and a typo turns it into "Madrid", the scoring pipeline scores `Madrid` (a different token sequence, different log-prob). For benchmarks like MMLU where the answer is a literal letter `(A)`, `(B)`, `(C)`, `(D)`, the scoring is robust to this; for free-form continuations, the choice of trailing punctuation is its own small confound. We use the same convention as Module 13: every choice ends with `<|end|>` so the model sees the format consistently.

### Generation: scoring open outputs

Generation eval is the more realistic of the two. Given a prompt, the model generates freely; a matcher decides if the generated string matches any reference.

```
   ┌──────────────────────────────────────────────────────────────────────┐
   │   GENERATION EVAL — ONE EXAMPLE                                       │
   ├──────────────────────────────────────────────────────────────────────┤
   │                                                                       │
   │     prompt:    "<|user|>\nWhat is 2+2?\n<|assistant|>\n"             │
   │     references: ["4", "four"]                                         │
   │                                                                       │
   │           │                                                           │
   │           ▼                                                           │
   │                                                                       │
   │     generate_fn(prompt)                                               │
   │       (under the hood: model, tokenizer, sampling settings,          │
   │        cropping, decoding — but the harness sees only the result)    │
   │           │                                                           │
   │           ▼                                                           │
   │                                                                       │
   │     prediction:  "4.<|end|>"     (or whatever the model produced)    │
   │           │                                                           │
   │           ▼                                                           │
   │                                                                       │
   │     numeric_match(prediction, references)                             │
   │       extracts "4" from prediction (first number)                    │
   │       extracts "4" from reference  "4"                               │
   │       |4 - 4| <= 0.0  ──►  True                                      │
   │           │                                                           │
   │           ▼                                                           │
   │                                                                       │
   │     EvalResult(correct=True, prediction="4.<|end|>", confidence=None)│
   │                                                                       │
   └──────────────────────────────────────────────────────────────────────┘
```

The matcher is the load-bearing decision. Choose well:

```
                                     Best matcher for...
   ┌─────────────────────────────────────────────────────────────────────┐
   │  Single-token answers, well-controlled    →  exact_match            │
   │  Factual QA, hand-authored references     →  normalized_match       │
   │  Arithmetic / numeric answers             →  numeric_match          │
   │  Long-form outputs with target keywords   →  contains_match         │
   │  Refusal / safety probes                  →  contains_match (anchor!│
   │                                              search for "I cannot",│
   │                                              not just "no")        │
   └─────────────────────────────────────────────────────────────────────┘
```

Generation eval doesn't expose a confidence by default — the harness sets `confidence=None` in `EvalResult` and `ece=None` in the report. To get a confidence on a generated string, re-score it under the model with `continuation_logprob(model, tokenizer, prompt, prediction)` and convert per-token mean log-prob to a probability via `exp`. Exercise 6 walks through this.

### Hallucination categories

When the eval harness flags a wrong answer, it's useful to classify *why* it was wrong. The categories below are loose but common:

```
   ┌──────────────────────────────────────────────────────────────────────┐
   │  FACTUAL HALLUCINATION                                                │
   │  Model emits a fact that contradicts the world.                       │
   │  Example: "The largest city in Spain is Lisbon."                     │
   │  Detection: ground-truth Q&A; matcher must catch surface variations. │
   └──────────────────────────────────────────────────────────────────────┘

   ┌──────────────────────────────────────────────────────────────────────┐
   │  CONTEXTUAL HALLUCINATION                                             │
   │  Model contradicts something STATED in the prompt.                    │
   │  Example: prompt says "Alice is 30. How old is she?" → "She's 25."   │
   │  Detection: probe with prompts that contain explicit facts.           │
   │  Symptom of poor in-context attention.                                │
   └──────────────────────────────────────────────────────────────────────┘

   ┌──────────────────────────────────────────────────────────────────────┐
   │  INTRINSIC vs EXTRINSIC                                               │
   │  Intrinsic: the wrong information was implied by, but contradicts,    │
   │    the prompt or training data.                                       │
   │  Extrinsic: the wrong information has no support anywhere — the       │
   │    model made it up entirely.                                         │
   │  Detection: look at the training corpus; is the wrong answer          │
   │    plausibly derivable from anything there?                          │
   └──────────────────────────────────────────────────────────────────────┘

   ┌──────────────────────────────────────────────────────────────────────┐
   │  REFUSAL FAILURE                                                      │
   │  Model answers a question it should refuse.                           │
   │  Example: "What's my mother's maiden name?" → "Smith."                │
   │  Detection: probes designed to be unanswerable; expected response is │
   │    "I don't know" or "I cannot determine."                            │
   │  Tracked separately from accuracy — the failure mode is "responded   │
   │    at all," not "responded incorrectly."                              │
   └──────────────────────────────────────────────────────────────────────┘

   ┌──────────────────────────────────────────────────────────────────────┐
   │  FORMAT BREAKAGE (carry-over from Modules 13–14)                      │
   │  Model emits valid content but breaks the chat-template format.       │
   │  Example: forgets the trailing <|end|>; emits a stray <|user|>.       │
   │  Detection: regex on the raw output; not a "wrong answer" — a      │
   │    "wrong shape." Track separately.                                   │
   └──────────────────────────────────────────────────────────────────────┘
```

The taxonomy isn't important to get right; the *practice* of categorizing failures is. A 20-question eval with a hand-categorized failure mode column tells you more than a 1000-question eval with just an accuracy number.

### The limits of small-model reasoning

Some failure modes are not hallucinations — they're capability gaps. A 20M-param model trained on a few hundred MB of text **cannot** do multi-step arithmetic, multi-hop reasoning, or chain-of-thought-style introspection. These aren't bugs; they're the floor. The eval harness's job is to make this floor visible:

```
   ┌──────────────────────────────────────────────────────────────────────┐
   │  CAPABILITY FLOOR (toy models)                                        │
   ├──────────────────────────────────────────────────────────────────────┤
   │                                                                       │
   │     What the model probably CAN do at toy scale:                      │
   │       ✓ pattern-match the chat format                                 │
   │       ✓ continue a sentence in the same register                      │
   │       ✓ answer a simple factual question if it appears verbatim      │
   │         (or close to it) in pretraining                               │
   │       ✓ pick the more-likely of two completions                      │
   │                                                                       │
   │     What the model probably CANNOT do:                                │
   │       ✗ multi-digit arithmetic                                        │
   │       ✗ multi-step reasoning                                          │
   │       ✗ refusal of unanswerable questions                            │
   │       ✗ recognize or correct its own errors                           │
   │       ✗ honestly admit uncertainty                                    │
   │                                                                       │
   │     The eval is what lets you tell which is which.                    │
   │                                                                       │
   └──────────────────────────────────────────────────────────────────────┘
```

The deliverable for this module is a written *characterization* of these floors, not a number. "Accuracy = 0.42" is less informative than "the model can answer city-of-country questions if the country appears in pretraining, fails on arithmetic past single digits, and never refuses any question regardless of how impossible." The latter is what a reader (or future you) needs.

## Concepts to internalize

- **Loss curves measure training; eval harnesses measure capability.** The two are not interchangeable. A model can have low loss and high error rate; a model can have moderate loss and surprisingly good capability. After this module, "the loss went down" is no longer a complete description of progress.
- **Closed-set vs open-set eval is a fundamental split.** Closed-set is fast, deterministic, and calibration-friendly; open-set is realistic but matcher-dependent. Most production harnesses do both; you should too.
- **Multiple-choice scoring = sequence_logprob over each option, take argmax.** The math is one log-softmax + gather + masked sum per option — same primitive as DPO. Confidence is softmax over option scores.
- **Length normalization changes the ranking, not the math.** Raw sum advantages shorter options; per-token mean is roughly length-invariant. Sweep both when evaluating.
- **Accuracy and confidence are independent axes.** A model can be highly accurate but miscalibrated, or barely accurate but well-calibrated. ECE is the headline scalar that pins this gap.
- **ECE = bin-weighted average |bin_acc − bin_conf|.** One careful loop over confidence bins. Lower is better; 0 is perfectly calibrated; 1 is maximally miscalibrated.
- **Reliability curve is the visual for ECE.** Bin centers on the x-axis, bin accuracies on the y-axis. Diagonal is "perfectly calibrated." Tiny LMs sit BELOW the diagonal in high-confidence bins — over-confidence is the default.
- **Generation eval = generate_fn + matcher.** The harness is decoupled from sampling; you supply a closure. Matchers are pluggable.
- **The four matchers cover most needs.** `exact`, `normalized`, `numeric`, `contains`. Pick by task, not by personal preference.
- **Hallucination categories are useful diagnostic labels, not a benchmark.** Label your failures as you go; aggregate the categories at the end. The labels help you target subsequent fine-tuning.
- **Toy models have a hard capability floor.** No amount of eval polish makes a 20M-param model do arithmetic. The deliverable is a written characterization of the floor — that's the actual learning outcome.

## Scaffolding and how to run the tests

This module ships six files in `g2c/eval/`:

- **`data.py`** — `MultipleChoiceExample`, `GenerationExample`, `EvalResult`, `EvalReport`. All four are dataclasses; all are fully implemented (boilerplate). Constructor validation catches malformed inputs.
- **`match.py`** — `exact_match`, `normalized_match`, `contains_match`, `numeric_match`. All four are scaffolded.
- **`logprob.py`** — `continuation_logprob`. The text-level analogue of DPO's `sequence_logprob`. Scaffolded.
- **`multiple_choice.py`** — `score_multiple_choice` (scaffolded) + `run_multiple_choice_eval` (implemented — pure plumbing once `score_multiple_choice` and `expected_calibration_error` are).
- **`generation.py`** — `score_generation_example` and `run_generation_eval`. Both implemented.
- **`calibration.py`** — `expected_calibration_error` and `reliability_curve`. Both scaffolded.

Tests live in `tests/test_eval.py`. Initial state: 19 tests pass (all the dataclass boilerplate plus the empty-input ValueError checks for the harnesses). 70 tests fail with `NotImplementedError` until you implement.

```bash
pytest tests/test_eval.py                          # all module-15 tests
pytest tests/test_eval.py -x                       # stop at first failure
pytest tests/test_eval.py -k Match                 # all matcher tests
pytest tests/test_eval.py -k Continuation          # log-prob tests
pytest tests/test_eval.py -k Calibration           # ECE tests
pytest tests/test_eval.py -k Reliability           # reliability tests
pytest tests/test_eval.py -k MultipleChoice        # MC scoring + harness
pytest tests/test_eval.py -k Generation            # generation harness
pytest tests/test_eval.py -v                       # verbose
```

Implementation order — the five steps are independent until step 5:

  1. **The four match functions** (`exact_match`, `normalized_match`, `contains_match`, `numeric_match`). Independent and small. Turns green: all matcher tests + `TestRunGenerationEval` (the generation harness uses these).
  2. **`continuation_logprob`**. The text-level scoring primitive. Turns green: `TestContinuationLogprob`.
  3. **`expected_calibration_error`**. The bucket loop. Turns green: `TestExpectedCalibrationError` + the ECE-aware tests in `TestRunMultipleChoiceEval` (once step 5 is done).
  4. **`reliability_curve`**. Same binning as ECE. Turns green: `TestReliabilityCurve`.
  5. **`score_multiple_choice`**. Depends on step 2. Turns green: `TestScoreMultipleChoice` + `TestRunMultipleChoiceEval` (the harness is implemented).

Steps 1–4 are independent; step 5 depends on step 2. The boilerplate tests pass from the start as a sanity check on the test file itself.

The end-to-end transformer test (`test_real_transformer_smoke`) pulls in `g2c.transformer.TransformerLM` — if Module 09 isn't filled in, this single test fails on a prerequisite. Same convention as the DPO/SFT trainer end-to-end tests.

The headline tests to watch:

- **`test_continuation_logprob_uniform_logits_value`** — pins down the log-prob math: with all-zero logits, sum over a 5-token continuation is exactly `−5 · log(V)`. If this fails, `log_softmax`, the gather, or the mask alignment is wrong.
- **`test_score_multiple_choice_uniform_logits_indifferent`** — with uniform logits and equal-length options, all options score equally; confidence is exactly `1/N`. The "indifference baseline" sanity check.
- **`test_score_multiple_choice_length_normalize_changes_decision`** — the cleanest demonstration that length normalization is doing what it claims: same model, same example, same prompt, but `length_normalize=True` flips the prediction. Hand-engineered so the math is verifiable.
- **`test_ece_perfect_calibration_zero`** — when every prediction's confidence equals its bin's empirical accuracy, ECE is exactly 0. The "what does perfect calibration look like" sanity.
- **`test_ece_always_correct_zero_confidence`** — the maximally miscalibrated case: confidence 0.0, accuracy 1.0. ECE = 1.0. Tests the upper bound of the metric.
- **`test_ece_known_case_two_bins`** — a hand-computed case (ECE = 0.25) that pins the bin-weighting formula exactly.
- **`test_run_multiple_choice_eval_real_transformer_smoke`** — the end-to-end check that the harness composes with a real `TransformerLM`. Depends on Module 09.

## What you'll build

Package: `g2c/eval/`

```python
# data.py
@dataclass(frozen=True)
class MultipleChoiceExample:
    prompt: str
    choices: list[str]                                       # ≥ 2 choices
    answer_idx: int                                          # implemented

@dataclass(frozen=True)
class GenerationExample:
    prompt: str
    references: list[str]                                    # ≥ 1 reference
                                                             # implemented

@dataclass
class EvalResult:
    correct:    bool
    prediction: int | str                                    # int for MC, str for gen
    confidence: float | None = None                          # implemented
    metadata:   dict[str, Any]

@dataclass
class EvalReport:
    task_name:       str
    n:               int
    accuracy:        float
    mean_confidence: float | None
    ece:             float | None
    results:         list[EvalResult]                        # implemented


# match.py
def exact_match(prediction: str, references: list[str]) -> bool: ...        # SCAFFOLDED
def normalized_match(prediction: str, references: list[str]) -> bool: ...   # SCAFFOLDED
def contains_match(prediction: str, references: list[str]) -> bool: ...     # SCAFFOLDED
def numeric_match(
    prediction: str, references: list[str], *, tolerance: float = 0.0,
) -> bool: ...                                                              # SCAFFOLDED


# logprob.py
def continuation_logprob(
    model, tokenizer, prompt: str, continuation: str,
) -> tuple[float, int]:                                                     # SCAFFOLDED


# multiple_choice.py
def score_multiple_choice(
    model, tokenizer, example: MultipleChoiceExample,
    *, length_normalize: bool = False,
) -> EvalResult:                                                            # SCAFFOLDED

def run_multiple_choice_eval(
    model, tokenizer, examples: list[MultipleChoiceExample],
    *, length_normalize: bool = False,
    task_name: str = "multiple_choice",
    compute_ece: bool = True, ece_n_bins: int = 10,
) -> EvalReport:                                                            # implemented


# generation.py
def score_generation_example(
    example: GenerationExample,
    generate_fn: Callable[[str], str],
    matcher: Callable[[str, list[str]], bool],
) -> EvalResult:                                                            # implemented

def run_generation_eval(
    examples: list[GenerationExample],
    generate_fn: Callable[[str], str],
    matcher: Callable[[str, list[str]], bool],
    *, task_name: str = "generation",
) -> EvalReport:                                                            # implemented


# calibration.py
def expected_calibration_error(
    confidences: list[float], correct: list[bool], *, n_bins: int = 10,
) -> float:                                                                 # SCAFFOLDED

def reliability_curve(
    confidences: list[float], correct: list[bool], *, n_bins: int = 10,
) -> tuple[list[float], list[float], list[int]]:                            # SCAFFOLDED
```

Total scaffolded code: roughly 80 lines across seven functions in three files. The math is light; the lesson is the API surface and the closed-set/open-set split.

## Exercises

1. **Hand-author a tiny multiple-choice eval set.** Write 30 `MultipleChoiceExample`s, each with 4 choices and a gold answer index. Suggested mix:

     - 10 factual questions (capitals, large-named-entity facts, simple geography).
     - 10 simple arithmetic questions ("What is 3 + 4?", choices `["6", "7", "8", "9"]`).
     - 10 instruction-following questions ("If asked about today's weather, the assistant should respond with…", choices including refusal options).

   Save as JSON. Constraints:

     - **Length-match the choices.** All four options for a question should be within 10–20% of each other in token count. If one option is dramatically shorter or longer, the harness's argmax is dominated by length not content.
     - **Plausible distractors.** Wrong answers should be wrong but reasonable — not absurd. "What is the capital of Spain?" with choices `["Madrid", "Lisbon", "Berlin", "an apple"]` is too easy because "an apple" is obviously not a city.

2. **Run the multiple-choice harness on your DPO'd model.** Load your Module 14 DPO'd checkpoint. Run `run_multiple_choice_eval` over your dataset. Report:

     - Overall accuracy (raw and length-normalized — does normalization change it?).
     - Mean confidence.
     - ECE.
     - The reliability curve as a small ASCII plot or saved figure.
     - Per-category accuracy: factual / arithmetic / instruction-following separately.

   Expected pattern: accuracy of 0.30–0.55 (random chance for 4-option is 0.25). ECE of 0.10–0.25 — a moderately overconfident toy model. Length normalization may bump accuracy by a few percent. Arithmetic accuracy is essentially random; factual is whatever happened to land in pretraining.

3. **Probe with hallucination prompts.** Hand-author 20 prompts designed to elicit hallucinations:

     - 5 obscure-fact questions (low-frequency named entities your pretraining wouldn't have seen many times).
     - 5 questions with no real answer ("What is the seventh letter of the word 'cat'?").
     - 5 prompts that contain an explicit fact that the answer should respect ("Alice is 30. How old will she be in 5 years?").
     - 5 absurd prompts ("What is the color of an idea?").

   Run `run_generation_eval` with `contains_match` against the expected behavior (the right answer for the answerable ones, "I don't know" / "cannot" for the unanswerable). Categorize each failure as factual hallucination, contextual hallucination, refusal failure, or format breakage. Report:

     - Number per category.
     - One verbatim sample per category that's egregious.
     - Whether DPO improved any of these (compare against SFT-only output if you preserved that checkpoint).

4. **Build a hand-authored arithmetic eval.** 50 simple-arithmetic prompts, in `<|user|>\n{question}\n<|assistant|>\n` format. Run `run_generation_eval` with `numeric_match` (tolerance=0). Report accuracy by digit count:

     - 1+1-digit (e.g. "What is 3 + 4?")
     - 2+2-digit (e.g. "What is 13 + 28?")
     - 3+3-digit (e.g. "What is 234 + 567?")

   Expected pattern: high accuracy on 1+1-digit (the model has seen many of these in pretraining), much lower on 2+2-digit, near-zero on 3+3-digit. The point is to localize the *capability cliff* — at what digit count does the model fall off? Document this number; it's the "this model can do X" boundary you'll cite in the deliverable.

5. **Beta-sweep eval comparison.** Take your Module-14 β sweep checkpoints (β ∈ {0.05, 0.1, 0.3, 1.0} from exercise 14.3). For each, run the multiple-choice harness from exercise 2. Plot:

     - Accuracy vs β.
     - ECE vs β.
     - Reliability curves stacked.

   Expected pattern: accuracy roughly flat across β with a small peak in the sweet spot (0.1–0.3); ECE markedly worse at extreme β (low β: model drifted, high β: model unchanged from SFT, neither is well-calibrated for this dataset). Document the β that gave the best ECE — it's not necessarily the same as the one with best accuracy.

6. **Generation eval with calibration (optional).** The generation harness sets `confidence=None` by default. Extend it: after `generate_fn` produces a prediction, re-score it under the model with `continuation_logprob(model, tokenizer, prompt, prediction)` and convert the per-token mean log-prob to a probability via `exp(mean_logp)`. Wire this through to `EvalResult.confidence` and re-aggregate ECE.

   Run on the arithmetic eval from exercise 4. Plot the reliability curve. Expected pattern: the model is even *more* over-confident on generation than on multiple choice, because every emitted answer feels "complete and final" to the autoregressive scorer regardless of how plausible it is.

7. **Hand-built MMLU subset (optional).** Pick 3 MMLU subjects (e.g. high_school_geography, elementary_mathematics, college_biology). Hand-transcribe (or download and minimally adapt) 20 questions per subject in `MultipleChoiceExample` form. Run the harness. Report per-subject accuracy. Compare against random chance (0.25 for 4-option MC) and against your hand-authored eval from exercise 1.

   Expected pattern: low across all subjects (0.25–0.40 — barely above chance). The calibration metric is more informative than the accuracy metric — does the model *know* it's guessing?

8. **Evaluation post-mortem (the deliverable).** Write 2–3 paragraphs characterizing your model's typical failure modes. Categories to cover:

     - **What it can do.** At least 3 specific capabilities, with eval numbers.
     - **What it cannot do.** At least 3 specific capability ceilings, with eval numbers.
     - **How it fails.** Hallucination categories, refusal behavior, format breakage rates.
     - **How well it knows itself.** Calibration: ECE, the gap between accuracy and mean confidence, where the over-confidence concentrates (high-confidence bins or low?).

   This is the deliverable. Not the eval harness — the *characterization* of the model. Reading this back to yourself in six months should let you remember what you built and what its limits were.

## Pitfalls to expect

- **Multiple-choice scoring depends on the choices' surface form.** "Madrid" and "Madrid." score differently because they're different token sequences. Be consistent: end every choice with the same trailing format markers (e.g. `<|end|>` in the chat-template setup).

- **Length-bias in raw multiple-choice scoring.** A long incorrect option can score lower than a short incorrect option *just because it's longer*. If accuracy seems suspiciously biased toward shorter choices, switch on `length_normalize=True`. lm-eval-harness exposes both `acc` and `acc_norm` for exactly this reason.

- **ECE without `min(int(c * n_bins), n_bins - 1)`.** A confidence of exactly 1.0 (which happens — softmax max can hit 1.0 in fp32 when one logit dominates) maps to bin index `n_bins`, which is out of range. Symptom: `IndexError` or silently-wrong ECE. The `min` clamps it to the last bin.

- **ECE on a tiny eval set.** With N < 20 examples and n_bins=10, most bins are empty or contain 1–2 examples. The metric is noisy. Reduce `n_bins` to 5 (or 3) for tiny sets.

- **Forgetting `torch.no_grad()` in `continuation_logprob`.** Eval is inference-only; without `no_grad`, every score builds an autograd graph and roughly doubles peak memory. The harness will run, but slowly and memory-hungry.

- **The mask in `continuation_logprob` is off by one.** The mask aligns with the *targets* (`y = full_ids[1:]`), not the inputs. Continuation tokens in `y` start at index `len(prompt_ids) - 1`, NOT `len(prompt_ids)`. Same shift-by-one as Modules 13–14. A mask off by one silently changes the score by including or excluding one prompt boundary token's log-prob.

- **`numeric_match` matching the wrong number.** "Half of 100 is 50" extracts `100` as the first number, not `50`. If your model's answer-shape is "the answer is X", strip the prefix before matching, OR write your references to expect numbers in particular positions, OR use a stricter regex.

- **`contains_match` matching unintended substrings.** Searching for `"no"` matches `"Norway"`, `"snow"`, `"annoy"`, etc. For refusal probes, use anchored references like `"I cannot"`, `"I don't know"`, or word-boundary regex instead of bare `"no"`.

- **The matcher silently does the wrong thing.** `normalized_match("Madrid.", ["Madrid"])` returns True (correct). But `normalized_match("Madrid is the capital", ["Madrid"])` returns False — they normalize to different strings. If you want partial-match semantics, use `contains_match`, not `normalized_match`. Pick the matcher to match the question.

- **Asymmetric matcher errors.** `normalized_match("madrid", ["Madrid."])` returns True; `normalized_match("Madrid.", ["madrid"])` also returns True. Symmetric. But `contains_match("the capital is Madrid", ["Madrid"])` returns True; `contains_match("Madrid", ["the capital is Madrid"])` returns False. Asymmetric: predictions are searched for references, not vice versa. If you cross prediction and reference roles by accident, your accuracy plummets silently.

- **Mean confidence != accuracy on a perfectly calibrated model.** Only on average — for any specific model, `mean_confidence` and `accuracy` can differ even with ECE=0. ECE is an integral over bins, not a point comparison. Don't read `mean_confidence ≠ accuracy` as "miscalibrated"; that's what ECE is for.

- **Generation eval with `temperature > 0` is non-deterministic.** Same input, different output across runs. If you're comparing models, fix the seed (`torch.Generator().manual_seed(seed)`) and pass it through your `generate_fn`. Or use temperature=0 (greedy) for deterministic eval. Production benchmarks always use deterministic decoding.

- **Re-running the eval mutates a stale checkpoint.** A subtle bug: load checkpoint → run eval → train one more step → run eval again, expecting "same model, same eval, same numbers." But the optimizer modified the model's parameters between runs. Always reload the checkpoint freshly before each eval pass, or freeze with `model.eval()` and `torch.no_grad()` (the harness already does the latter inside `continuation_logprob`).

- **The eval is too small to be statistically meaningful.** 20 examples × 4-option MC has a standard error of ~0.10 on accuracy at p=0.5. A 5-percentage-point difference between models is noise at this scale. Use 50–200 examples for any comparison you want to take seriously, even at toy scale.

## Reading

Primary:

- **Naeini, Cooper, Hauskrecht, "Obtaining Well Calibrated Probabilities Using Bayesian Binning" (2015).** The ECE paper. Eq. 1 is the formula we implement; §3 has the reliability-diagram construction. The original "ECE" is in the medical-AI literature; this paper is what brought it to ML.
- **Guo, Pleiss, Sun, Weinberger, "On Calibration of Modern Neural Networks" (2017).** The deep-learning calibration paper. Demonstrates that modern (deep, residual, batch-normed) networks are *systematically* miscalibrated — overconfident — and proposes temperature scaling as a one-parameter fix. Read §3 for the diagnosis; §4 for the fix. ECE intuitions are best built here.
- **Liang, Bommasani, Lee et al., "Holistic Evaluation of Language Models" (HELM, 2022).** The benchmark-of-benchmarks. 30 scenarios × 7 metrics × dozens of models. Read §2 for the framework; you don't need the appendix. The scenario / metric / model decomposition is the conceptual backbone of all production eval.

Secondary:

- **Ji, Lee, Frieske et al., "Survey of Hallucination in Natural Language Generation" (2023).** The hallucination taxonomy (factual / contextual, intrinsic / extrinsic, etc.) is from §3. Read just §3 — it's all you need for the categorization framework used in this module.
- **Zheng, Chiang, Sheng et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena" (2023).** Argues that for many open-ended tasks, asking a strong model to judge is more useful than any string matcher. Read §3 for the analysis of judge bias (position bias, verbosity bias, self-enhancement bias). Useful conceptual frame for "what's wrong with the matchers we shipped."
- **Lin, Hilton, Evans, "TruthfulQA" (2022).** A benchmark specifically designed to elicit hallucinations on common misconceptions. Read §3 for the question construction methodology — useful inspiration for exercise 3's adversarial dataset.

Optional:

- **Hendrycks et al., "Measuring Massive Multitask Language Understanding" (MMLU, 2021).** The de facto multiple-choice eval. 57 subjects × ~100 questions. Read just §2 (the construction methodology) — you don't need to memorize the subjects.
- **Kadavath, Conerly, Askell et al., "Language Models (Mostly) Know What They Know" (2022).** Anthropic-internal study of whether models can self-rate confidence in their answers. Methodologically interesting; the headline result (some models can, somewhat) is dataset-dependent.
- **Wang, Wei, Schuurmans et al., "Self-Consistency Improves Chain of Thought Reasoning" (2022).** Generation-time decoding strategy that tracks marginal confidence by sampling multiple chains. Out of scope here, but a clean pointer for "how confidence-aware decoding can shift accuracy."
- **Yang, Cao, Yan, "Hallucinations in Large Language Models: A Survey" (2024).** A more recent survey than Ji et al. Skim §2 (taxonomy refinements) for context.

## Deliverable checklist

- [ ] All tests in `tests/test_eval.py` pass (88 tests if Module 09's `TransformerLM.forward` is implemented; otherwise 87 — the real-transformer end-to-end test depends on it).
- [ ] Hand-authored multiple-choice eval set of 30+ questions in `data/eval/multiple_choice.json`, with the structure described in exercise 1. Each question has 4 length-matched choices.
- [ ] Hand-authored generation eval set of 20+ questions in `data/eval/generation.json` (exercise 3), with categorized expected-failure modes.
- [ ] Hand-authored arithmetic eval set of 50 questions in `data/eval/arithmetic.json` (exercise 4), spanning 1+1-digit through 3+3-digit.
- [ ] Notebook: `notebooks/15-evaluation.ipynb`. Runs the multiple-choice and generation harnesses on your DPO'd model. Plots the reliability curve. Reports accuracy, mean confidence, and ECE per task type. Commit with outputs visible.
- [ ] **Written characterization** of your model's typical failure modes (exercise 8). 2–3 paragraphs in `docs/eval-postmortem.md` (or in the notebook's final markdown cell). Cover: what it can do (with eval numbers), what it cannot do (with eval numbers), how it fails (categorized), and how well-calibrated it is.
- [ ] You can explain — out loud, without notes — the difference between accuracy and calibration, and why a model can be highly accurate yet poorly calibrated.
- [ ] You can explain — out loud, without notes — the multiple-choice scoring procedure: tokenize each option, sequence_logprob conditional on the prompt, argmax for prediction, softmax for confidence.
- [ ] You can explain — out loud, without notes — how ECE is computed (bin by confidence, weight by bin frequency, sum the per-bin |acc − conf| differences) and what its `[0, 1]` extremes mean.
- [ ] You can explain — out loud, without notes — when to use each of the four matchers and what kinds of bug each silently introduces if used inappropriately.

## M-series notes

Module 15 is mostly inference + bookkeeping. Comfortable on every M-series machine.

- **Per-eval cost.** A multiple-choice example with 4 choices costs 4 forward passes through the model. At our scale (20M params, ~50-token prompts), each forward is milliseconds. A 100-question MC eval is on the order of seconds.

- **Generation eval cost.** Dominated by generation, not by scoring. Each example requires running `generate_fn` once with up to `max_new_tokens` autoregressive steps. At 64 max_new_tokens, a 50-question generation eval is ~30–60 seconds for a 20M model.

- **Memory.** The harness holds the model + a few small Python lists of `EvalResult`s. Even a 1000-question eval over a 20M-param model fits comfortably in 8GB.

- **Reproducibility.** For multiple-choice: deterministic by construction (no sampling). For generation: pass a `torch.Generator().manual_seed(seed)` through your `generate_fn` if you sample with `temperature > 0`, or use greedy (`temperature=0`).

- **No `model.eval()` / dropout caveat.** Our `g2c.transformer.TransformerLM` doesn't use dropout in its current scaffold, so toggling between train and eval mode doesn't matter. If you add dropout later, remember `model.eval()` before the eval harness.

- **`continuation_logprob` and MPS.** All tensor operations in `continuation_logprob` are well-supported on MPS. The model forward, log_softmax, gather, and masked sum all run natively. No CPU fallback warnings expected.

- **Storing eval reports.** The `EvalReport` is JSON-serializable except for `metadata`'s `option_logps` lists (which can contain nans). For checkpointed eval results across training runs, serialize the per-example `correct`, `prediction`, `confidence`, and `metadata['option_logps']` to a CSV or JSON line per example. The reliability curve and ECE can always be recomputed from these.

- **Baseline numbers** to expect from a freshly DPO'd 20M model on the deliverable eval sets:

  ```
     ┌─────────────────────────────────┬──────────┬──────────┬──────────┐
     │ Task                            │ Accuracy │ Mean Conf│ ECE      │
     ├─────────────────────────────────┼──────────┼──────────┼──────────┤
     │ Hand-built 4-option MC (n=30)   │  0.30–   │  0.45–   │  0.10–   │
     │                                 │  0.55    │  0.65    │  0.25    │
     │ Hand-built generation (n=20)    │  0.10–   │  N/A     │  N/A     │
     │                                 │  0.40    │          │          │
     │ Arithmetic 1+1 digit (n=15)     │  0.30–   │  N/A     │  N/A     │
     │                                 │  0.70    │          │          │
     │ Arithmetic 2+2 digit (n=15)     │  0.0–    │  N/A     │  N/A     │
     │                                 │  0.20    │          │          │
     │ Arithmetic 3+3 digit (n=15)     │  0.0     │  N/A     │  N/A     │
     └─────────────────────────────────┴──────────┴──────────┴──────────┘
  ```

  These are rough; your numbers will vary with checkpoint quality, eval set difficulty, and matcher strictness. The *shape* of the gradient — high on simple, near-zero on hard — is what matters; the absolute numbers don't.
