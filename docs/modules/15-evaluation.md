# Module 15 — Hallucination and evaluation

> **Question this module answers:** *Why does the model confidently invent things, and how do we measure it?*

![Hero](15-evaluation/Module15-Hero.png)

The two weeks have been about tuning a base model's behavior. This week is about measuring how well we did. It's a three-pronged evaluation: closed-set scoring for capability ranking, open-set for behavior, and a calibration metric for "how well does the model know what it doesn't know". The minimum viable harness for any serious model work. A single loss number alone doesn't cut it for assistant grade models.

---
## Before you start

* *Review*
	* [[11-sampling]] For the difference from logits and text completion
	* [[13-sft]] For formatting targeting
	* [[14-dpo]] For the framework of how to score comparison
	* [[PyTorch Primer]] if any PyTorch code is unfamiliar or confusing
* *Finish*
	* `g2c/sampling` from [[11-sampling]]
	* [[14-dpo]] The eval harness will use the post-trained model generated and saved in this notebook
* *Run* `.venv/bin/python scripts/artifact_status.py --module 15` if you are deciding between a self-trained model and the BaseLM path. If you need BaseLM, run `./baselm.sh`.

---
## Where this fits in

After Modules 13 and 14 you have a behavior-shaped model. The loss curves gave you a single coarse-grained metric — SFT loss decreased, DPO reward margin increased. But what does that cash out to in terms of capability?

```
   ┌───────────────────────────────────────────────────────────────────────┐
   │  THE PROBLEM WITH LOSS CURVES                                         │
   ├───────────────────────────────────────────────────────────────────────┤
   │                                                                       │
   │   SFT loss   ─────────┐                                               │
   │                       │                                               │
   │                       └──►  "is the model formatting answers?"        │
   │                                                                       │
   │   DPO margin ─────────┐                                               │
   │                       │                                               │
   │                       └──►  "is the chosen-vs-rejected gap growing?"  │
   │                                                                       │
   │   ────────────────────────────────────────────────────────────────    │
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
   └───────────────────────────────────────────────────────────────────────┘
```

A small model after post-training might emit:

```
<|user|>
What is 2 + 2?
<|assistant|>
4.<|end|>
```

A grade-school correct answer. But when you try a slightly more complex prompt:

```
<|user|>
What is 13 + 28?
<|assistant|>
35.<|end|>
```

Confidently wrong. The model learned the format but doesn't know arithmetic.

The deepest pathology of language models is that they're trained on **fluency** but judged on **truth** (or helpfulness, or correctness, etc.). A model that emits well-formed sentences gets low cross-entropy loss. A model that emits *true* sentences gets... also low cross-entropy loss. But the loss curve doesn't separate the two.

```
   ┌──────────────────────────────────────────────────────────────────────┐
   │   THE TRAINING OBJECTIVE                                             │
   ├──────────────────────────────────────────────────────────────────────┤
   │                                                                      │
   │     loss  =  − Σ_t log π(target_t | context)                         │
   │                                                                      │
   │   This rewards:                                                      │
   │     ✓ matching the surface form of the training data                 │
   │     ✓ producing fluent, well-formed text                             │
   │     ✓ following the chat template (post-SFT)                         │
   │     ✓ preferring chosen over rejected (post-DPO)                     │
   │                                                                      │
   │   This does NOT reward:                                              │
   │     ✗ being factually correct                                        │
   │     ✗ admitting uncertainty                                          │
   │     ✗ refusing to invent things                                      │
   │     ✗ stopping at the right time                                     │
   │                                                                      │
   └──────────────────────────────────────────────────────────────────────┘
```

A confident fluent wrong answer is rewarded as much as a confident fluent right one. Both are "low loss" if the surface form matches conventional patterns. The predictable consequence is **hallucination**: the model emits the *shape* of an answer, but the *content* is whatever its training distribution most readily produced for prompts of that shape.

## The big idea

The **eval harness** closes this loop. By scoring outputs against what we want, not what the loss measures, we surface the gap. Most of the model's failures are expected given the training objective; eval makes them *visible*.

An eval harness is a repeatable procedure to evaluate and score model behavior against specific questions or challenges. The harness abstracts and standardizes the grading procedure against arbitrary questions. Standardized scoring mean models can be cross-compared in an objective way. 

The key steps in an evaluation harness are:

1. Asks specific questions — not "is the loss low" but "is *this answer* right."
2. Aggregates — accuracy across many such questions, not anecdotes.
3. Measures calibration (optional) — when the model says "35," does it actually believe 35, or is it just emitting a number-shaped string?

Evaluation harnesses come in two flavors:

1. **Closed-set eval.** Asks a question, then evaluates against a pre-determined set of multiple-choice answers.
2. **Open-set eval** Asks a question, then lets the model complete text normally. Grades the text response using a **matcher**. 

![Closed-set (multiple choice) vs open-set (generation) eval. Two complementary harnesses, side by side. Closed-set: the model chooses from a finite list of candidate continuations; we score each candidate by sequence-log-probability, take argmax for the prediction, and softmax over scores for confidence. Open-set: the model generates freely, and a matcher decides whether the generated string matches any reference answer. A "key differences at a glance" panel pins the trade-off: closed-set is fast and calibration-friendly but bounded by the candidate list; open-set is realistic and uncalibrated by default. Use both — they expose different aspects of the same model.](15-evaluation/Module15-Closed.png)
*The two halves of the harness. Closed-set scoring is what builds calibration; open-set generation is what surfaces hallucination.*

### Closed-set eval

Multiple-choice scoring is the cleanest eval primitive. Given a prompt and N candidate continuations, the model scores each based on sequence-log-probability. Up until now we've used models by *sampling*: enter the prompt, generate logits for next token, sample the next token, and repeat until the completion finishes. Multiple choice inverts this procedure. 

![MultiChoice](15-evaluation/Module15-MultiChoice.png)
*Tokenize each option, score, argmax for prediction, softmax for confidence.*

We still generate logits for next token prediction. However instead of randomly sampling the next token, we already "know" the next tokens based on the pre-written answer. Let's say the question is "Who was the first president of the United States?". We start by generating the next token logits from the prompt, apply softmax, and find the probability assigned to the "George " token.

We then append "Goerge " to the prompt, then repeat looking for the "Washington" token in the outputs. And repeat until the we reach the end of the answer. That gives us a series of log probabilities, the sum of which is the model's implied probability for the answer candidate:

We do the same for all the pre-written answer candidates ("Thomas Jefferson", "George Bush", etc.). At the end we have an implied probability for each answers in the candidate set. Finally we derive relative probabilities for each choice by normalizing against the sum of all the answers in the candidate set:

```
score_i = sum_t(log softmax(logits_i_t))
    
P(candidate_i | candidate set)
  = exp(score_i) / sum_j exp(score_j)
```

This recovers a "predictions weighted by certainty" measure. Not only do we know if the model chooses the right answer, but we have a metric of how *confident* it was.

### Calibration

![Calibration](15-evaluation/Module15-Calibration.png)
*Calibration uses the model's internal logits to distinguish when the model is confidently wrong and when it's guessing*

A non-obvious framing: multiple-choice is a diagnostic on the model's internal probability landscape. When you ask the model `Madrid? Lisbon? Barcelona? Berlin?` and read its softmax over options, you're querying the *implicit world model* the network has learned. A confident-and-correct model has world knowledge; a confident-and-wrong model has confidently-wrong world knowledge (the textbook hallucination case); an uncertain model is honest about not knowing. 

**Calibration** is the process by which we quantify the model's confidence in its answer. This is only possible because closed-set eval probes the internal logits generated by the model. A model that picked the wrong choice but with low confidence should look like it's "guessing". The wrong choice selected with only marginally higher probability than the right answer. A confident hallucination not only picks the wrong choice, but picks it by a large probability margin over the right answer.

A model can be:

- **High accuracy, well calibrated.** Best case. Production-ready.
- **High accuracy, miscalibrated.** Right often, but says "I'm sure" when it isn't. Dangerous in tool-using or agentic settings — the model takes high-stakes actions on weak beliefs.
- **Low accuracy, well calibrated.** Wrong often, but knows it's wrong. Safer than the previous case — the user can attend to the confidence signal and fall back to verification.
- **Low accuracy, miscalibrated.** Wrong and confident. Catastrophic. This is what *small over-trained LMs default to* — exactly the regime your toy model is in.

```
   ┌───────────────────────────────────────────────────────────────────────┐
   │   ACCURACY vs CALIBRATION                                             │
   ├───────────────────────────────────────────────────────────────────────┤
   │                                                                       │
   │     accuracy   =  fraction of predictions that are correct            │
   │     confidence =  the model's stated probability for its prediction   │
   │                                                                       │
   │     Calibrated:                                                       │
   │       on the subset of predictions where confidence ≈ p,              │
   │       the empirical accuracy is also ≈ p.                             │
   │       — for ALL p in [0, 1].                                          │
   │                                                                       │
   │     Over-confident: confidence > accuracy                             │
   │     Under-confident: confidence < accuracy                            │
   │                                                                       │
   └───────────────────────────────────────────────────────────────────────┘
```

The metric we use to measure calibration is **expected calibration error (ECE)**. ECE measures how well confidence matches reality. If a model says “80% confident” on a group of examples, it should be right about 80% of the time; ECE averages the gap between confidence and observed accuracy across bins:

```
B_m       = set of examples whose predicted confidence falls in bin m
|B_m|     = number of examples in that bin
n         = total number of examples

acc(B_m)  = average correctness in bin m
          = (1 / |B_m|) * sum_{i in B_m} 1[pred_i = y_i]

conf(B_m) = average predicted confidence in bin m
          = (1 / |B_m|) * sum_{i in B_m} confidence_i
```

A **reliability diagram** visualizes model calibration by grouping predictions into confidence bins and plotting each bin’s average confidence against its **empirical accuracy**. A perfectly calibrated model lies on the diagonal line: when it says 70% confidence, it is correct about 70% of the time. Bars below the diagonal indicate **overconfidence**; bars above it indicate **underconfidence**.

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

         The horizontal distance from each dot to the diagonal at the same
         y-coordinate is the bin's contribution to ECE.
```

### Open-set eval

If closed-set eval is a multiple choice, open-set eval is the essay section of the test. Open-set eval starts with a prompt, then generates a a normal completion in the same way we've been doing since Module 11. Because it doesn't have an internal accuracy metric like closed-set eval, we rely on external matchers to grade the answer. You can think of matchers as the "rubric" used to grade the test response.

```
   ┌───────────────────────────────────────────────────────────────────────┐
   │   GENERATION EVAL — ONE EXAMPLE                                       │
   ├───────────────────────────────────────────────────────────────────────┤
   │                                                                       │
   │     prompt:    "<|user|>\nWhat is 2+2?\n<|assistant|>\n"              │
   │     references: ["4", "four"]                                         │
   │                                                                       │
   │           │                                                           │
   │           ▼                                                           │
   │                                                                       │
   │     generate_fn(prompt)                                               │
   │       (under the hood: model, tokenizer, sampling settings,           │
   │        cropping, decoding — but the harness sees only the result)     │
   │           │                                                           │
   │           ▼                                                           │
   │                                                                       │
   │     prediction:  "4.<|end|>"                                          │
   │           │                                                           │
   │           ▼                                                           │
   │                                                                       │
   │     numeric_match(prediction, references)                             │
   │       extracts "4" from prediction (first number)                     │
   │       extracts "4" from reference  "4"                                │
   │       |4 - 4| <= 0.0  ──►  True                                       │
   │           │                                                           │
   │           ▼                                                           │
   │                                                                       │
   │     EvalResult(correct=True, prediction="4.<|end|>")                  │
   │                                                                       │
   └───────────────────────────────────────────────────────────────────────┘
```

Selecting the right matcher is the load-bearing decision. Choose well:

![Matcher zoo — pick the right tool for the job. Four matchers, side by side. `exact_match`: character-equality, use for single-token answers and trailing-space sensitivity. `normalized_match`: case/punctuation/whitespace stripped, then equality — best for factual QA where surface form varies but content matches. `numeric_match`: extract first number from prediction, compare to first number in references within tolerance — best for arithmetic and quantitative answers. `contains_match`: any reference appears as a substring of the prediction (case-insensitive) — best for refusal probes and long-form outputs with target keywords. Each matcher has an "example: prediction vs references" panel showing one representative case with True/False; an "important" panel calls out asymmetric matchers like `contains_match` (search direction matters) and the silent-failure mode of using `numeric_match` with a regex that catches the wrong number.](15-evaluation/Module15-Matcher.png)
*The lookup table for choosing a matcher. Picking the right matcher is half the eval design problem — the wrong matcher silently produces the wrong accuracy.*

```
                                     Best matcher for...
   ┌─────────────────────────────────────────────────────────────────────┐
   │  Single-token answers, well-controlled    →  exact_match            │
   │  Factual QA, hand-authored references     →  normalized_match       │
   │  Arithmetic / numeric answers             →  numeric_match          │
   │  Long-form outputs with target keywords   →  contains_match         │
   │  Refusal / safety probes                  →  contains_match         │
   └─────────────────────────────────────────────────────────────────────┘
```

By default open-set eval doesn't generate a confidence. However it is possible to back out a proxy for confidence. First let the freeform completion generate, then generating a log-probability for the response using the same approach we used for multiple choice answers. However unlike multiple choice there is no way to normalize this value against the set of all possible answers. While it can be useful to record, it is nowhere near as load bearing as ECE.

### Hallucination categories

![Hallucination taxonomy — not all wrongs are the same. Five columns. (1) Factual hallucination: the model states a fact that contradicts the real world ("the largest city in Spain is Lisbon"). Cause: gaps in pretraining knowledge, over-generalization. Detect with: ground-truth Q&A. (2) Contextual hallucination: the model contradicts a fact stated explicitly in the prompt ("Alice is 30. How old is Alice? → 25"). Cause: weak in-context attention, recency bias, insufficient reasoning capacity. Detect with: prompts containing explicit facts. (3) Intrinsic vs extrinsic: intrinsic — the wrong information was implied by but contradicts the prompt or training data; extrinsic — the wrong information has no support anywhere, the model invented it. (4) Refusal failure: the model answers a question it should refuse ("What's my mother's maiden name? → Smith"). Cause: training data lacks refusals; refusal behavior was never reinforced. (5) Format breakage: the model emits valid content but breaks the chat-template format (forgets `<|end|>`, emits stray `<|user|>`). A "key takeaway" panel below: a wrong answer is not one thing — labeling failure modes turns error analysis into a roadmap for better data, training, and evaluation.](15-evaluation/Module15-Hallucination.png)
*Naming the failure mode is more useful than reporting raw accuracy: a 30% accuracy with 90% factual hallucinations needs different fixes than a 30% accuracy with 90% refusal failures.*

When the eval harness flags a wrong answer, it's useful to classify *why* it was wrong. The categories below are loose but common:

```
   ┌──────────────────────────────────────────────────────────────────────┐
   │  FACTUAL HALLUCINATION                                               │
   │  Model emits a fact that contradicts the world.                      │
   │  Example: "The largest city in Spain is Lisbon."                     │
   │  Detection: ground-truth Q&A; matcher must catch surface variations. │
   └──────────────────────────────────────────────────────────────────────┘

   ┌──────────────────────────────────────────────────────────────────────┐
   │  CONTEXTUAL HALLUCINATION                                            │
   │  Model contradicts something STATED in the prompt.                   │
   │  Example: prompt says "Alice is 30. How old is she?" → "She's 25."   │
   │  Detection: probe with prompts that contain explicit facts.          │
   │  Symptom of poor in-context attention.                               │
   └──────────────────────────────────────────────────────────────────────┘

   ┌──────────────────────────────────────────────────────────────────────┐
   │  INTRINSIC vs EXTRINSIC                                              │
   │  Intrinsic: the wrong information was implied by, but contradicts,   │
   │    the prompt or training data.                                      │
   │  Extrinsic: the wrong information has no support anywhere — the      │
   │    model made it up entirely.                                        │
   │  Detection: look at the training corpus; is the wrong answer         │
   │    plausibly derivable from anything there?                          │
   └──────────────────────────────────────────────────────────────────────┘

   ┌──────────────────────────────────────────────────────────────────────┐
   │  REFUSAL FAILURE                                                     │
   │  Model answers a question it should refuse.                          │
   │  Example: "What's my mother's maiden name?" → "Smith."               │
   │  Detection: probes designed to be unanswerable; expected response is │
   │    "I don't know" or "I cannot determine."                           │
   │  Tracked separately from accuracy — the failure mode is "responded   │
   │    at all," not "responded incorrectly."                             │
   └──────────────────────────────────────────────────────────────────────┘

   ┌──────────────────────────────────────────────────────────────────────┐
   │  FORMAT BREAKAGE (carry-over from Modules 13–14)                     │
   │  Model emits valid content but breaks the chat-template format.      │
   │  Example: forgets the trailing <|end|>; emits a stray <|user|>.      │
   │  Detection: regex on the raw output; not a "wrong answer" — a        │
   │    "wrong shape." Track separately.                                  │
   └──────────────────────────────────────────────────────────────────────┘
```

The taxonomy isn't important to get right; the *practice* of categorizing failures is. A 20-question eval with a hand-categorized failure mode column tells you more than a 1000-question eval with just an accuracy number.

### The limits of small-model reasoning

Some failure modes are not hallucinations — they're capability gaps. A 20M-param model trained on a few hundred MB of text **cannot** do multi-step arithmetic, multi-hop reasoning, or chain-of-thought-style introspection. These aren't bugs; they're the floor. The eval harness's job is to make this floor visible:

```
   ┌───────────────────────────────────────────────────────────────────────┐
   │  CAPABILITY FLOOR (toy models)                                        │
   ├───────────────────────────────────────────────────────────────────────┤
   │                                                                       │
   │     What the model probably CAN do at toy scale:                      │
   │       ✓ pattern-match the chat format                                 │
   │       ✓ continue a sentence in the same register                      │
   │       ✓ answer a simple factual question if it appears verbatim       │
   │         (or close to it) in pretraining                               │
   │       ✓ pick the more-likely of two completions                       │
   │                                                                       │
   │     What the model probably CANNOT do:                                │
   │       ✗ multi-digit arithmetic                                        │
   │       ✗ multi-step reasoning                                          │
   │       ✗ refusal of unanswerable questions                             │
   │       ✗ recognize or correct its own errors                           │
   │       ✗ honestly admit uncertainty                                    │
   │                                                                       │
   │     The eval is what lets you tell which is which.                    │
   │                                                                       │
   └───────────────────────────────────────────────────────────────────────┘
```

"Accuracy = 0.42" is less informative than "the model can answer city-of-country questions if the country appears in pretraining, fails on arithmetic past single digits, and never refuses any question regardless of how impossible."

## Concepts to internalize

- **Loss curves measure training; eval harnesses measure capability.** The two are not interchangeable. A model can have low loss and high error rate; a model can have moderate loss and surprisingly good capability. 
- **Accuracy and confidence are independent.** A model can be highly accurate but miscalibrated, or barely accurate but well-calibrated. ECE is the headline metric.
- **Closed-set vs open-set eval.** Closed-set is deterministic and calibration-friendly. Open-set is realistic but matcher-dependent. 
- **Multiple-choice scoring**. Sum log probabilities for each option. Confidence is softmax over option scores.
- **ECE: bin-weighted average of calibration.** One careful loop over confidence bins. Lower is better; 0 is perfectly calibrated; 1 is maximally miscalibrated.
- **Generation eval = generate function + matcher.** The harness is decoupled from sampling. Matchers are pluggable.
- **The four matchers cover most needs.** `exact`, `normalized`, `numeric`, `contains`. Pick by task.
- **Hallucination categories are diagnostics, not benchmarks.** The labels help you target subsequent fine-tuning.
- **Models have a hard capability floor.** No amount of eval polish makes a 30M-param model do arithmetic. 

### What we don't cover

- **HELM multi-metric harnesses.** HELM tracks ~7 metrics per task across 30+ tasks for hundreds of models. Our harness tracks accuracy + ECE on two task types over a hand-built dataset of 50–200 examples. HELM is the architecture you'd want to scale a "scenario / metric / model" three-way matrix. The lesson is the loop, not the matrix.
- **Model-graded ("LLM-as-judge") evaluation.** For many open-ended tasks, asking a strong model to *judge* outputs is more useful than any string matcher. Our toy model isn't strong enough to be a judge.
- **Adversarial robustness benchmarks.** AdvGLUE, RealToxicityPrompts, etc. expose models to adversarial inputs designed to elicit failures. At our scale these are mostly noise. Read the papers; don't build the benchmarks.
- **Full MMLU.** MMLU is 57 subjects × ~100 questions = ~14k multiple-choice questions. Running it takes hours even on a 7B model. We use a hand-built MC subset of 20–50 questions for the exercise.
- **Calibration via temperature scaling.** Post-hoc calibration: train a one-parameter "temperature" on a held-out set so the model's confidences align better with empirical accuracy. Useful, well-known, and a clean follow-up to ECE — but it's recalibration, not measurement.

---
## What you'll build

Package: `g2c/eval/`

```python
@dataclass(frozen=True)
class MultipleChoiceExample:
    prompt: str
    choices: list[str]
    answer_idx: int

@dataclass(frozen=True)
class GenerationExample:
    prompt: str
    references: list[str]

@dataclass
class EvalResult:
    correct:    bool
    prediction: int | str
    confidence: float | None = None                 
    metadata:   dict[str, Any]

@dataclass
class EvalReport:
    task_name:       str
    n:               int
    accuracy:        float
    mean_confidence: float | None
    ece:             float | None
    results:         list[EvalResult]                     


# match.py
def exact_match(prediction: str, references: list[str]) -> bool: ...  
def normalized_match(prediction: str, references: list[str]) -> bool: ... 
def contains_match(prediction: str, references: list[str]) -> bool: ...   
def numeric_match(
    prediction: str, references: list[str], *, tolerance: float = 0.0,
) -> bool: 


# logprob.py
def continuation_logprob(
    model, tokenizer, prompt: str, continuation: str,
) -> tuple[float, int]:              

# multiple_choice.py
def score_multiple_choice(
    model, tokenizer, example: MultipleChoiceExample,
    *, length_normalize: bool = False,
) -> EvalResult:                                 

def run_multiple_choice_eval(
    model, tokenizer, examples: list[MultipleChoiceExample],
    *, ...) -> EvalReport:    # implemented

def run_generation_eval(
    examples: list[GenerationExample],
    generate_fn: Callable[[str], str],
    matcher: Callable[[str, list[str]], bool], *, ...,
) -> EvalReport:              # implemented

# calibration.py
def expected_calibration_error(
    confidences: list[float], correct: list[bool], *, ...) -> float
def reliability_curve(
    confidences: list[float], correct: list[bool], *, ...)
```

Roughly 80 lines across seven functions.

## How to run the tests

Tests live in `tests/test_eval.py`. Initial state: 19 tests pass, 70 tests fail

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


## Exercises

1. **Hand-author a tiny multiple-choice eval set.** Write 30 `MultipleChoiceExample`s, each with 4 choices and a gold answer index. Suggested mix:

     - 10 factual questions (capitals, large-named-entity facts, simple geography).
     - 10 simple arithmetic questions ("What is 3 + 4?", choices `["6", "7", "8", "9"]`).
     - 10 instruction-following questions ("If asked about today's weather, the assistant should respond with…", choices including refusal options).

   Constraints:

     - **Length-match the choices.** All four options for a question should be within 10–20% of each other in token count. If one option is dramatically shorter or longer, the harness's argmax is dominated by length not content.
     - **Plausible distractors.** Wrong answers should be wrong but reasonable — not absurd. "What is the capital of Spain?" with choices `["Madrid", "Lisbon", "Berlin", "an apple"]` is too easy because "an apple" is obviously not a city.

2. **Run the multiple-choice harness on your DPO'd model.** Load your Module 14 DPO'd checkpoint. Run `run_multiple_choice_eval` over your dataset. Report:

     - Overall accuracy (raw and length-normalized — does normalization change it?).
     - Mean confidence.
     - ECE.
     - The reliability curve as a small ASCII plot or saved figure.
     - Per-category accuracy: factual / arithmetic / instruction-following separately.

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

5. **Beta-sweep eval comparison.** Take your Module-14 β sweep checkpoints (β ∈ {0.05, 0.1, 0.3, 1.0} from exercise 14.3). For each, run the multiple-choice harness from exercise 2. Plot:

     - Accuracy vs β.
     - ECE vs β.
     - Reliability curves stacked.

   Expected pattern: accuracy roughly flat across β with a small peak in the sweet spot (0.1–0.3); ECE markedly worse at extreme β (low β: model drifted, high β: model unchanged from SFT, neither is well-calibrated for this dataset). Document the β that gave the best ECE — it's not necessarily the same as the one with best accuracy.

6. **Generation eval with calibration (optional).** The generation harness sets `confidence=None` by default. Extend it: after `generate_fn` produces a prediction, re-score it under the model with `continuation_logprob(model, tokenizer, prompt, prediction)` and convert the per-token mean log-prob to a probability via `exp(mean_logp)`. Wire this through to `EvalResult.confidence` and re-aggregate ECE.

   Run on the arithmetic eval from exercise 4. Plot the reliability curve. Expected pattern: the model is even *more* over-confident on generation than on multiple choice, because every emitted answer feels "complete and final" to the autoregressive scorer regardless of how plausible it is.

7. **Hand-built MMLU subset (optional).** Pick 3 MMLU subjects (e.g. high_school_geography, elementary_mathematics, college_biology). Hand-transcribe (or download and minimally adapt) 20 questions per subject in `MultipleChoiceExample` form. Run the harness. Report per-subject accuracy. Compare against random chance (0.25 for 4-option MC) and against your hand-authored eval from exercise 1.

8. **Evaluation post-mortem (the deliverable).** Write 2–3 paragraphs characterizing your model's typical failure modes. Categories to cover:

     - **What it can do.** At least 3 specific capabilities, with eval numbers.
     - **What it cannot do.** At least 3 specific capability ceilings, with eval numbers.
     - **How it fails.** Hallucination categories, refusal behavior, format breakage rates.
     - **How well it knows itself.** Calibration: ECE, the gap between accuracy and mean confidence, where the over-confidence concentrates (high-confidence bins or low?).

   This is the deliverable. Not the eval harness — the *characterization* of the model. Reading this back to yourself in six months should let you remember what you built and what its limits were.

## Pitfalls to expect

- **The eval is too small to be statistically meaningful.** 20 examples × 4-option MC has a standard error of ~0.10 on accuracy at p=0.5. A 5-percentage-point difference between models is noise at this scale. Use 50–200 examples for any comparison you want to take seriously, even at toy scale.

- **Multiple-choice scoring depends on the choices' surface form.** "Madrid" and "Madrid." score differently because they're different token sequences. Be consistent: end every choice with the same trailing format markers (e.g. `<|end|>` in the chat-template setup).

- **Length-bias in raw multiple-choice scoring.** A long incorrect option can score lower than a short incorrect option *just because it's longer*. If accuracy seems suspiciously biased toward shorter choices, switch on `length_normalize=True`. lm-eval-harness exposes both `acc` and `acc_norm` for exactly this reason.

- **ECE without `min(int(c * n_bins), n_bins - 1)`.** A confidence of exactly 1.0 (which happens) maps to bin index `n_bins`, which is out of range. Symptom: `IndexError`

- **ECE on a tiny eval set.** With N < 20 examples and n_bins=10, most bins are empty or contain 1–2 examples. The metric is noisy. Reduce `n_bins` to 5 (or 3) for tiny sets.

- **Mean confidence != accuracy on a perfectly calibrated model.** Only on average — for any specific model, `mean_confidence` and `accuracy` can differ even with ECE=0. . Don't read `mean_confidence ≠ accuracy` as "miscalibrated".

- **The mask is off by one.**  A mask off by one silently changes the score by including or excluding one prompt boundary token's log-prob.

- **`numeric_match` matching the wrong number.** "Half of 100 is 50" extracts `100` as the first number, not `50`. Strip the prefix before matching, OR write your references to expect numbers in particular positions, OR use a stricter regex.

- **`contains_match` matching unintended substrings.** Searching for `"no"` matches `"Norway"`, `"snow"`, `"annoy"`, etc. 

- **The matcher silently does the wrong thing.** If you want partial-match semantics, use `contains_match`, not `normalized_match`. Pick the matcher to match the question.

- **Asymmetric matcher errors.** `normalized_match()` is Symmetric. `contains_match()` is not. Predictions are searched for references, not vice versa. If you cross prediction and reference roles by accident, your accuracy plummets silently.

- **Generation eval is non-deterministic.** Same input, different output across runs. If you're comparing models, fix the seed (`torch.Generator().manual_seed(seed)`) or use temperature=0 . Production benchmarks always use deterministic decoding.

- **Forgetting `torch.no_grad()`** Eval is inference-only. The harness will run, but slowly and memory-hungry.

## M-series notes

Module 15 is mostly inference + bookkeeping. Comfortable on every M-series machine.

- **Per-eval cost.** A multiple-choice example with 4 choices costs 4 forward passes through the model. At our scale (20M params, ~50-token prompts), each forward is milliseconds. A 100-question MC eval is on the order of seconds.

- **Generation eval cost.** Dominated by generation, not by scoring. Each example requires running `generate_fn` once with up to `max_new_tokens` autoregressive steps. At 64 max_new_tokens, a 50-question generation eval is ~30–60 seconds for a 20M model.

- **Memory.** The harness holds the model + a few small Python lists of `EvalResult`s. Even a 1000-question eval over a 20M-param model fits comfortably in 8GB.

---
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

- [ ] All tests in `tests/test_eval.py` pass
- [ ] Hand-authored multiple-choice eval set of 30+ questions in `data/eval/multiple_choice.json`. 
- [ ] Hand-authored generation eval set of 20+ questions in `data/eval/generation.json`, with categorized expected-failure modes.
- [ ] Hand-authored arithmetic eval set of 50 questions in `data/eval/arithmetic.json`, spanning 1+1-digit through 3+3-digit.
- [ ] Notebook: `notebooks/15-evaluation.ipynb`. Runs the multiple-choice and generation harnesses on your DPO'd model. 
- [ ] **Written characterization** of your model's typical failure modes. 2–3 paragraphs. Cover: what it can do (with eval numbers), what it cannot do (with eval numbers), how it fails (categorized), and how well-calibrated it is.
- [ ] You can explain — out loud, without notes — the difference between accuracy and calibration, and why a model can be highly accurate yet poorly calibrated.
- [ ] You can explain — out loud, without notes — the multiple-choice scoring procedure.
- [ ] You can explain — out loud, without notes — how ECE is computed and what its extremes mean.
- [ ] You can explain — out loud, without notes — when to use each of the four matchers and what kinds of bug each introduces if used inappropriately.
