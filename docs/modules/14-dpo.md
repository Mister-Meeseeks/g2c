# Module 14 — Preference tuning (DPO)

> **Question this module answers:** *Why is the model helpful, polite, or stylistically consistent?*

![Module 14 on one page: the Module 13 SFT'd TransformerLM confronted with a prompt where format alone is not enough. Top half: the SFT model on the prompt "<|user|>\\nWhat is 2+2?\\n<|assistant|>\\n" produces "4." — correct. But on a second prompt "<|user|>\\nWhat is the largest city in Spain?\\n<|assistant|>\\n", it produces a confidently-wrong "Lisbon.<|end|>" — because Lisbon-the-string is in its training corpus as a city name. The SFT loss has no way to penalize this; every assistant turn that's syntactically valid is treated as equally good. Bottom half: after DPO with 100 (prompt, chosen, rejected) preference pairs — where "Madrid." was preferred over "Lisbon." for the Spain prompt — the model now produces "Madrid.<|end|>" Crucially, the SAME model on prompts NOT in the preference set is also better calibrated: it answers shorter, refuses more often, drifts off-topic less. A central panel shows the DPO mechanism: the policy and reference model both forward through the same prompt+chosen and prompt+rejected sequences; their per-sequence log-probabilities feed the closed-form sigmoid loss `-log σ(β * (log[π(c)/π_ref(c)] - log[π(r)/π_ref(r)]))`. A right-edge panel highlights the three behavioral shifts: (1) implicit reward margin grows over training (chosen log-probs rise, rejected log-probs fall), (2) the reference model is FROZEN — only the policy updates, (3) at β=0 the loss is constant log(2) and nothing learns. A bottom strip captions the headline: "DPO is the RLHF objective in closed form — same gradient, no separate reward model, no PPO loop, no rollouts. The whole alignment pipeline collapses to a one-line classification loss against a frozen reference."](14-dpo/Module14-Hero.png)

The whole module on one page. Module 13 taught the model to **format** an answer; Module 14 teaches it to **prefer** one answer over another. The mechanics are: pair up (prompt, chosen, rejected), forward through both policy and reference, compute four per-sequence log-probabilities, and minimize a sigmoid loss on the implicit-reward margin. No new architecture, no reward model, no PPO. The 50–200 preference pairs are the entire training set; everything else is bookkeeping.

---
## Before you start

* *Review* [[13-sft]] for the chat template, masking, and `Trainer` shape — DPO reuses all three with two-model bookkeeping
* *Finish* `g2c/nn` from [[03-nn]] and `g2c/training` from [[03b-training]] — the DPO trainer imports `SGD`/`Module`/`resolve_device` and `clip_grad_norm_`/`cosine_with_warmup` from them
* *Finish* a Module 13 SFT'd checkpoint — DPO loads it twice (trainable policy + frozen reference)
* *Finish* Module 04's tokenizer artifact — preference triples encode through it

---
## Where this fits in

After Module 13 your SFT'd model produces well-formatted assistant turns. Probe it:

```
<|user|>
What is the largest city in Spain?
<|assistant|>
```

The 30M SFT'd model produces:

```
Lisbon.<|end|>
```

Confident. Wrong. Format-perfect. The SFT loss has no way to penalize this output — every syntactically valid `{single sentence}<|end|>` response is equally good under cross-entropy as long as the format markers are in the right place. The model "knows" the prompt is asking about a Spanish city; it picks one of the city-shaped tokens that pretraining over-represented in city contexts (probably from corpus passages where "Spain" and "Lisbon" co-occur even though one is the wrong answer). SFT taught it the *shape* of the answer; nothing taught it which answers are better.

The behavior we want — *prefer Madrid over Lisbon as the answer to "largest city in Spain"* — is intrinsically **comparative**. There's no single ground-truth target you write down; what you write down is "this answer is better than that one." Training on comparisons is the DNA of all human-feedback methods, from the original Christiano et al. 2017 deep-RL-from-preferences paper through InstructGPT (Ouyang et al. 2022) to modern DPO (Rafailov et al. 2023).

DPO is the closed-form simplification of the RLHF objective. The pipeline:

1. Collect or hand-author a few hundred `(prompt, chosen, rejected)` preference triples.
2. Hold a frozen copy of the SFT'd model as the "reference."
3. Train a separate "policy" copy by minimizing `−log σ(β · ( (log π(c|x) − log π_ref(c|x)) − (log π(r|x) − log π_ref(r|x)) ))`, averaged over the preference dataset.

That's the whole thing. There is no separate reward model, no PPO loop, no rollouts at training time, no learning-from-on-policy-samples; just a frozen reference, a trainable policy, and a sigmoid on the log-ratio difference.

```
   ┌──────────────────────────────────────────────────────────────────────┐
   │  ONE DPO STEP — start to finish                                      │
   └──────────────────────────────────────────────────────────────────────┘

      preference example:
         prompt  : "<|user|>\nWhat is the largest city in Spain?\n<|assistant|>\n"
         chosen  : "Madrid.<|end|>"
         rejected: "Lisbon.<|end|>"
                                  │
                                  ▼  build two parallel sequences
        chosen_full   = prompt_ids + chosen_ids       (mask 1 on chosen tokens)
        rejected_full = prompt_ids + rejected_ids     (mask 1 on rejected tokens)
                                  │
                                  ▼  forward (policy and reference, both)
        logits_pi_c, logits_pi_r   = policy.forward(chosen_full),  policy.forward(rejected_full)
        with torch.no_grad():
            logits_ref_c, logits_ref_r = ref.forward(chosen_full), ref.forward(rejected_full)
                                  │
                                  ▼  per-sequence log-probabilities
        logp_pi_c   = Σ_t mask[t] · log_softmax(logits_pi_c)[t, target_t]
        logp_pi_r   = ...                                       (each shape (B,))
        logp_ref_c  = ...
        logp_ref_r  = ...
                                  │
                                  ▼  the DPO loss (averaged over batch)
        L = −log σ( β · ( (logp_pi_c  − logp_pi_r)
                         − (logp_ref_c − logp_ref_r) ) )
                                  │
                                  ▼  backward + step (same Trainer recipe)
                            standard SGD, standard clip, standard cosine
                            — but ONLY policy params get updated.
```

The qualitative shift you'll see in exercise 2: a DPO'd model gets factual questions right *more often* than the SFT model, even though no factual targets were written into the dataset — only **comparative judgments** like "Madrid > Lisbon for that prompt" and "an apology beats a denial when the user is upset." The model generalizes from the comparisons. Not perfectly, not robustly at toy scale; but visibly, on prompts of the same shape as the training pairs.

A non-obvious framing: **DPO turns the policy itself into the reward model.** The quantity `r̂(x, y) = β · log[π(y|x) / π_ref(y|x)]` is the *implicit reward*, and after DPO training, the policy assigns higher implicit reward to chosen completions than to rejected ones. You can read the implicit reward off any (x, y) pair by forwarding both the policy and the reference and computing the log-ratio — this is how DPO papers' analyses work. At inference time the reward never appears explicitly; the policy just generates with higher probability the things it learned were rewarded.

## The big idea

### From RLHF to DPO: same gradient, different machinery

![RLHF (traditional) vs DPO (this module). The classical InstructGPT pipeline runs three stages: SFT (Module 13's deliverable), then reward modeling on preference comparisons, then PPO reinforcement-learning to optimize the SFT model against the reward model with a KL penalty. DPO collapses the second and third stages into a single supervised loss using the SFT'd model as a frozen reference. A "what RLHF needs" panel lists rollouts, value heads, KL controllers, and PPO ratio clipping. A "what DPO needs" panel lists: a frozen reference, a trainable policy, β, and a preference dataset. A "bottom line" panel pins the framing: same goal — align the model with human preferences — same gradient, but no reward model, no PPO loop, no on-policy rollouts.](14-dpo/Module14-RLHF.png)
*The headline simplification this module captures. The classical pipeline's middle and right stages are notoriously expensive and finicky at production scale; DPO's closed-form derivation makes them disappear, leaving a single forward+backward against an offline preference dataset.*

The RLHF pipeline (Ouyang et al., 2022) has three stages:

```
   ┌──────────────────────────────────────────────────────────────────────┐
   │   STAGE 1.  SFT.                                                     │
   │     Train on (prompt, response) pairs to get the format right.       │
   │     This is Module 13. Already done.                                 │
   ├──────────────────────────────────────────────────────────────────────┤
   │   STAGE 2.  REWARD MODELING.                                         │
   │     Train a separate scalar-output head r(x, y) on the preference    │
   │     pairs using the Bradley-Terry loss.                              │
   ├──────────────────────────────────────────────────────────────────────┤
   │   STAGE 3.  PPO RL.                                                  │
   │     Use the frozen reward model as the environment reward, run       │
   │     PPO with a KL penalty against the SFT model. The output is the   │
   │     "instruction-tuned + RLHF" model that ChatGPT et al. ship.       │
   └──────────────────────────────────────────────────────────────────────┘
```

Stage 2 is moderately expensive (a separate model). Stage 3 is *very* expensive — PPO is finicky, the reward model can be hacked by the policy ("reward hacking" / "Goodharting"), the KL controller has to be tuned, the rollouts dominate the wall-clock cost. At small scale Stage 3 is impractical.

DPO **collapses Stages 2 and 3 into a single supervised loss**. The trick is that the closed-form solution to the constrained-RL problem (Stage 3) lets you express the optimal policy in terms of the reference and the reward; then you invert to express the reward in terms of the optimal policy and the reference; then you substitute that into the reward-modeling loss (Stage 2) and the reward model disappears entirely.

The end result is a loss that depends only on:
  - the trainable policy `π`
  - the frozen reference `π_ref` (= the SFT'd model)
  - a hyperparameter `β` (the KL strength)
  - the preference dataset

No reward model. No rollouts. No on-policy data. The gradient of the DPO loss is *exactly the same* as the gradient you'd get from running RLHF with the optimal reward model — but you compute it from a fixed offline dataset using a single supervised forward+backward.

### The DPO loss, in detail

![DPO mechanism: four log-probs → one margin → one loss. Step 1 takes one preference example `(prompt, chosen, rejected)`. Step 2 forwards both the policy and the frozen reference on (prompt + chosen) and (prompt + rejected) — four sequences, four scalar log-probabilities `log π(c|x)`, `log π(r|x)`, `log π_ref(c|x)`, `log π_ref(r|x)`, each computed by log-softmaxing logits, gathering the target's column, and summing over the response tokens (mask = 1 only on the response, including `<|end|>`). Step 3 turns four log-probs into two log-ratios — chosen ratio `Δ_c = log π(c|x) − log π_ref(c|x)` and rejected ratio `Δ_r = log π(r|x) − log π_ref(r|x)` — and one margin `m = Δ_c − Δ_r`. Step 4 plugs the margin into the closed-form DPO loss `L = −log σ(β · m)`. A "step 0 sanity check" panel pins the canonical invariant: when the policy equals the reference, every Δ is zero, m is zero, σ(0) = 0.5, and the loss is exactly `log 2 ≈ 0.693`. An "intuition: what the loss does" panel shows three regimes — before training the policy and reference produce equal probabilities; during training the policy pushes chosen log-probs UP and rejected log-probs DOWN; bad training pushes both down (implicit-reward collapse).](14-dpo/Module14-FourLogs.png)
*The whole DPO mechanism on one page. Trace the flow once before reading the prose: (prompt, chosen, rejected) → four log-probs → two log-ratios → one margin → one scalar loss. Every test in `tests/test_dpo.py` is a check on one stage of this pipeline.*

For a single preference example `(x, y_c, y_r)` with chosen `y_c` and rejected `y_r`:

```
       L_DPO  =  − log σ( β · m̂ )

       m̂      =  ( log π(y_c | x)  −  log π_ref(y_c | x) )            ← chosen  log-ratio
                 −
                 ( log π(y_r | x)  −  log π_ref(y_r | x) )            ← rejected log-ratio
```

`m̂` is the **margin between implicit rewards** (chosen minus rejected), divided by `β`. The DPO loss is the negative log-likelihood of "chosen beat rejected under a Bradley-Terry model with implicit rewards."

```
   ┌────────────────────────────────────────────────────────────────────┐
   │                                                                     │
   │              log π(c|x)  −  log π_ref(c|x)        ┌────────────┐    │
   │              ─────────────────────────────  ──►   │  CHOSEN    │    │
   │              policy / ref ratio for chosen        │  IMPLICIT  │    │
   │                                                   │  REWARD    │    │
   │                                                   └────────────┘    │
   │                                                          │           │
   │                                                          ▼           │
   │                                                       (β · ·)        │
   │                                                          │           │
   │                                                          ▼           │
   │   ┌────────────┐                                  ┌────────────┐    │
   │   │  REJECTED  │   ◄──  ─────────────────────────  │  CHOSEN    │   │
   │   │  IMPLICIT  │       log π(r|x)  − log π_ref(r|x)│  REWARD −  │   │
   │   │  REWARD    │       (with same β · ·)            │  REJECTED  │   │
   │   └────────────┘                                   └────────────┘    │
   │                                                          │           │
   │                                                          ▼           │
   │                                          loss = − log σ(margin)      │
   │                                                                       │
   └────────────────────────────────────────────────────────────────────┘
```

**At step 0** (policy is freshly copied from reference): both ratios are zero, the margin is zero, `σ(0) = 0.5`, and `loss = log 2 ≈ 0.6931`. This is the canonical DPO sanity value. Always. Independent of architecture, dataset, β. If you implement DPO and your initial loss is anything else, the `pad_and_collate_pref` mask, the `sequence_logprob`, or the reference initialization is wrong.

**As training proceeds**: the policy pushes `log π(y_c|x)` UP and `log π(y_r|x)` DOWN, the margin grows positive, the loss decreases below `log 2`. The implicit reward margin (`chosen_reward − rejected_reward` in the metrics dict) is the headline diagnostic — it should grow monotonically across training, and it's a more informative quantity than the loss (which saturates as the sigmoid output approaches 1).

**β controls how far the policy can drift.** Mathematically, `β` is the KL coefficient in the original RLHF objective; intuitively, it's the "strength" of the policy's connection to the reference. Small β (e.g. 0.01) lets the policy diverge a long way for small preference signals — risk of mode collapse, repetition, gibberish. Large β (e.g. 1.0) pins the policy near the reference — safe but sometimes can't move enough to absorb the preference signal. The DPO paper and most follow-ups recommend `β ∈ [0.1, 0.5]` as the sweet spot. **At toy scale `β = 0.1` is a fine default**; exercise 4 sweeps it.

### The frozen reference: what it does and why it must stay frozen

![Policy vs frozen reference. Both models start from the same SFT'd checkpoint produced by Module 13 — `ref_model = copy.deepcopy(model)`. The policy is trainable: gradients flow, weights update, it learns to prefer chosen over rejected. The reference is frozen: never gradients (`with torch.no_grad():` around its forwards), only used to score the same prompts under the original SFT distribution so the log-ratios `log π / log π_ref` have a fixed denominator. A single preference triple `(prompt, chosen, rejected)` is fed to both models; both compute scalar log-probabilities `log π_θ(c|x)`, `log π_θ(r|x)`, `log π_ref(c|x)`, `log π_ref(r|x)`. The DPO loss combines these into a margin and pushes the policy to widen it without drifting too far from the reference. A "key idea" panel: we don't care about the absolute probabilities, only how the policy moves relative to the reference. An "important" panel: the reference must not change, ever — the deepest invariant test of a DPO implementation is to snapshot reference params before/after training and assert byte-for-byte equality.](14-dpo/Module14-Policy.png)
*The two-model setup. Module 13's SFT'd checkpoint plays both roles — once trainable, once frozen. The freeze invariant is what `test_dpo_trainer_ref_model_unchanged` enforces, and exercise 5 asks you to verify it on your real run.*

The reference is the **anchor** for the implicit-reward computation. The whole DPO loss is a function of *log-ratios* between the policy and the reference; if both move together (e.g. you accidentally update both), the log-ratios stay zero and nothing changes.

```
   ┌─────────────────────────────────────────────────────────────────┐
   │   POLICY (trainable)                  REFERENCE (frozen)        │
   │   π_θ                                 π_ref                     │
   │     │                                    │                       │
   │     │  forward(prompt + chosen)          │  forward(prompt + chosen)│
   │     ▼                                    ▼                       │
   │   logits_pi_c        ──┐         ┌──   logits_ref_c              │
   │     │                  │         │       │                       │
   │   log_softmax + gather │         │   log_softmax + gather        │
   │     │                  │         │       │                       │
   │     ▼                  │         │       ▼                       │
   │   logp_pi_c            │         │     logp_ref_c                │
   │     │                  │         │       │                       │
   │     │       (subtract) │         │       │       (no gradient,   │
   │     │ ◄──────────────────────────│       │        torch.no_grad) │
   │     │                  │         │       │                       │
   │     ▼                  │         │       │                       │
   │   chosen_logratio = logp_pi_c − logp_ref_c                      │
   │     │                                                            │
   │     ▼                                                            │
   │   (similar for rejected; subtract; multiply by β)               │
   │     │                                                            │
   │     ▼                                                            │
   │   −log σ(β · margin)                                             │
   │     │                                                            │
   │     ▼                                                            │
   │   backward                                                       │
   │     │                                                            │
   │     ▼                                                            │
   │   ONLY  π_θ.grad  populated;  π_ref  has no grad and no update.  │
   │                                                                   │
   └─────────────────────────────────────────────────────────────────┘
```

Two ways to enforce the freeze:

  1. **Architectural**: set `requires_grad=False` on every reference parameter. PyTorch's `model.parameters()` still returns them but `.grad` stays `None` and the optimizer can't step them.
  2. **Trainer-level**: don't include them in the optimizer, and forward them under `torch.no_grad()` so they don't accumulate `.grad` at all.

We use approach (2). It's cleaner — the optimizer signature is "only policy params," there's no risk of accidentally stepping the reference, and the `no_grad` context guarantees no autograd graph through the reference. `test_dpo_trainer_ref_model_unchanged` verifies this invariant: snapshot every reference parameter before training, snapshot after, assert byte-for-byte equality.

### Preference dataset construction: the costly half of DPO

DPO is technically simple but **operationally hard** because the dataset matters far more than the loss formula. Three construction methods at increasing scale:

  1. **Hand-authored.** You write 50–200 `(prompt, chosen, rejected)` triples. Slow but high signal — every example is a deliberate choice about what behavior you want the model to prefer. The exercises ask you to do this.

  2. **Sampled-then-judged.** Use the SFT'd model itself to generate two completions per prompt (e.g. with different sampling seeds or temperatures). Have a stronger model (GPT-4, Claude, a 70B local model) judge which is better. The "RLAIF" recipe — **reinforcement learning from AI feedback**.

  3. **Static benchmark datasets.** Anthropic's HH-RLHF (helpful + harmless), OpenAssistant Conversations, UltraFeedback, etc. Tens of thousands of pairs each. Use only with a stronger base model than ours; toy-scale models trained on production-scale preference data tend to learn the dataset's biases more than its preferences.

A quality pin for any preference dataset: **chosen and rejected should differ in the *behavior* you care about, not in length, fluency, or surface form**. If chosen is systematically longer than rejected, DPO learns "be longer." If chosen uses formal language and rejected uses informal, DPO learns "be formal." Anthropic's HH-RLHF paper (Bai et al., 2022) documents extensively how length and style biases leak into DPO/PPO models and how to detect them. At our scale: pair completions of similar length and similar style; vary only the *thing you're trying to teach*.

### Failure modes specific to DPO

Beyond the SFT failure modes inherited from Module 13:

```
   ┌────────────────────────────────────────────────────────────────┐
   │   REFERENCE DRIFT                                              │
   │   The reference model is not actually frozen — its parameters  │
   │   update along with the policy. Symptom: loss curve looks     │
   │   fine but the model behavior matches the SFT'd model exactly. │
   │   Fix: verify ref_model.parameters() are unchanged after a few │
   │   steps. Common cause: optimizer was constructed with both     │
   │   models' params concatenated.                                 │
   └────────────────────────────────────────────────────────────────┘

   ┌────────────────────────────────────────────────────────────────┐
   │   LENGTH BIAS                                                  │
   │   Chosen completions are systematically longer than rejected   │
   │   ones in the dataset. DPO learns "longer is better" instead   │
   │   of the actual preference. Symptom: model produces verbose,   │
   │   repetitive outputs after training. Fix: audit the dataset's  │
   │   chosen/rejected length distributions.                        │
   └────────────────────────────────────────────────────────────────┘

   ┌────────────────────────────────────────────────────────────────┐
   │   IMPLICIT-REWARD COLLAPSE                                     │
   │   Both chosen_reward and rejected_reward go to large negative  │
   │   numbers — the policy is making BOTH less likely than the     │
   │   reference does, but chosen drops less than rejected. The     │
   │   margin is fine; the absolute log-probs are nonsense.         │
   │   Symptom: generation is gibberish but DPO accuracy looks      │
   │   high. Fix: lower lr, raise β. (This is the single hardest    │
   │   DPO failure mode to debug because the loss curve hides it.)  │
   └────────────────────────────────────────────────────────────────┘

   ┌────────────────────────────────────────────────────────────────┐
   │   BETA TOO SMALL                                               │
   │   The policy drifts arbitrarily far from the reference. Even   │
   │   if the preference signal is good, the model loses base       │
   │   capacity. Symptom: format breaks, repetition increases,     │
   │   responses get longer or shorter monotonically. Fix: raise β. │
   │   At toy scale β=0.1 is the floor; β=0.01 is too small.        │
   └────────────────────────────────────────────────────────────────┘

   ┌────────────────────────────────────────────────────────────────┐
   │   BETA TOO LARGE                                               │
   │   The policy is pinned so close to the reference that the      │
   │   preference signal can't pull it. Symptom: loss stays near    │
   │   log(2), accuracy stays near 0.5, model behavior is           │
   │   indistinguishable from SFT. Fix: lower β. β=1.0 is usually   │
   │   too high for hand-authored datasets at small scale.          │
   └────────────────────────────────────────────────────────────────┘

   ┌────────────────────────────────────────────────────────────────┐
   │   PREFERENCE-LEAKED FORMAT BREAKAGE                            │
   │   The dataset's rejected completions have a systematic format  │
   │   problem (missing <|end|>, mid-response role marker, wrong    │
   │   newline). DPO learns "DON'T do this format thing," which is  │
   │   useful — except it also has an outsized gradient because the │
   │   format violation is rare. Symptom: model becomes overly      │
   │   conservative about format edge cases. Fix: ensure rejected   │
   │   completions are format-clean; the only difference vs chosen  │
   │   should be the BEHAVIOR you're correcting.                    │
   └────────────────────────────────────────────────────────────────┘
```

The middle three (length bias, implicit-reward collapse, β tuning) are real concerns even at production scale and have whole papers about them (KTO, IPO, length-controlled DPO). At toy scale you'll see them in caricature; the exercises ask you to *cause* the failure modes deliberately so you internalize what they look like.

## Concepts to internalize

- **DPO replaces RLHF's reward-model + PPO with a closed-form supervised loss.** The gradient is the same as RLHF-with-the-optimal-reward-model; the implementation collapses to a single forward+backward per step.
- **The policy is the reward model.** The implicit reward `r̂(x, y) = β · log[π(y|x) / π_ref(y|x)]` is what DPO trains on; you can read it off any (x, y) pair after training.
- **The reference must stay frozen.** If both policy and reference update together, the log-ratios stay zero and nothing learns. Snapshotting reference params before/after training is the deepest invariant test of a DPO implementation.
- **At step 0 the loss is exactly log(2).** Independent of architecture, β, or data. If yours isn't, something is wrong with the masking, the log-prob computation, or the reference initialization.
- **β is the KL strength.** Small β: more drift, more risk of collapse. Large β: less drift, less learning. At toy scale, β = 0.1 is a starting point; sweep [0.05, 0.1, 0.3, 1.0] in exercise 4.
- **The preference dataset format is `(prompt, chosen, rejected)` triples.** Chosen and rejected share the prompt prefix and diverge over the response. Mask is 1 on response tokens (chosen or rejected), 0 on prompt and padding. Same shift-by-one as SFT.
- **Sequence-level log-probabilities are SUMS, not means.** `log π(y|x) = Σ_t log π(y_t | x, y_{<t})`. A mean would change the objective by length-normalizing.
- **At toy scale, 50–200 preference pairs is the right order of magnitude.** Not 5; not 5000. Hand-authoring this many lets you control quality; controlling quality is more important than quantity at this scale (same lesson as Module 13's LIMA finding).
- **Length bias is the single most-studied DPO failure.** Always check chosen-vs-rejected length distributions before training.
- **DPO does not teach new knowledge.** Like SFT, it shifts behavior over what the base model already knows. A 20M model that didn't know Madrid is the capital of Spain before DPO doesn't *learn* it from a single preference pair — it learns to give *more probability* to "Madrid" relative to similarly-shaped wrong answers, but only when the base model's prior over the relevant tokens already gives Madrid nonzero mass.

### What we don't cover

- **Implementing a reward model and PPO.** The whole point of DPO is to replace this pipeline. You read the InstructGPT paper for context, but you do not implement it. PPO at toy scale is a research project on its own — the policy-gradient variance, the KL controller, the value head, the rollout buffer — none of it is necessary if DPO works, and DPO works.
- **Online preference collection.** Real DPO pipelines often iterate: train the policy, sample fresh rollouts, ask a stronger model (or human) to label them, append to the dataset, retrain. This is where most of the engineering effort goes in production. We use a **fixed offline dataset** of 50–200 hand-authored or LLM-judged pairs.
- **KTO / IPO / SLIC / ORPO and the rest of the DPO-derivative zoo.** Each is a small refinement of the DPO loss aimed at a specific failure mode (length bias, off-policy drift, the implicit-reward calibration problem). Skim the names; don't implement them. The vanilla DPO loss is the right unit of work for one week, and at toy scale the differences are within the noise.
- **Proper KL estimation.** The `β · log[π/π_ref]` term in DPO is technically the "implicit reward," which is loosely related to the per-step KL but is NOT a Monte Carlo estimate of it. Real RLHF papers compute a proper KL via a forward KL estimator on rollouts; DPO sidesteps this entirely. If you read the DPO paper carefully, the KL appears in the *derivation* but never has to be computed at training time.
- **LoRA for the policy.** At production scale, you train a low-rank update on top of the frozen reference; the reference and policy share the bulk of their weights. Saves the 2× memory cost. Out of scope here for the same reason it was in Module 13.
- **Length normalization.** The DPO loss as written is sensitive to response length: a long correct response and a long incorrect response can saturate the sigmoid in opposite directions. Some DPO variants normalize the log-prob by length. We don't — at toy scale with similar-length chosen/rejected pairs, length bias is small.

---
## What you'll build

Package: `g2c/dpo/`

```python
class PreferenceExample(NamedTuple):
    prompt_ids:   list[int]                                   # implemented (NamedTuple)
    chosen_ids:   list[int]                                   # implemented (NamedTuple)
    rejected_ids: list[int]                                   # implemented (NamedTuple)


def pad_and_collate_pref(
    examples: list[PreferenceExample],
    *,
    max_seq_len: int,
    pad_id: int,
) -> tuple[
    Tensor, Tensor, Tensor,   # chosen   x, y, mask
    Tensor, Tensor, Tensor,   # rejected x, y, mask
]:                                                            # SCAFFOLDED


def sequence_logprob(
    logits: Tensor,        # (B, T, V)
    targets: Tensor,       # (B, T)
    mask: Tensor,          # (B, T) — 1 where loss applies, 0 elsewhere
) -> Tensor:               # (B,) — per-example log-prob SUM over masked positions
                                                              # SCAFFOLDED


def dpo_loss(
    policy_chosen_logp:   Tensor,   # (B,) WITH gradient
    policy_rejected_logp: Tensor,   # (B,) WITH gradient
    ref_chosen_logp:      Tensor,   # (B,) NO gradient
    ref_rejected_logp:    Tensor,   # (B,) NO gradient
    *,
    beta: float,
) -> tuple[Tensor, dict[str, Tensor]]:                        # SCAFFOLDED


class DPOTrainer:
    def __init__(
        self,
        model,                                # the trainable POLICY
        *,
        ref_model,                            # the FROZEN reference
        examples: list[PreferenceExample],
        max_seq_len: int,
        pad_id: int,
        beta: float,
        batch_size: int,
        max_steps: int,
        max_lr: float,
        min_lr: float = 0.0,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        grad_clip: float | None = None,
        eval_every: int = 100,
        eval_iters: int = 20,
        log_every: int = 10,
        generator: torch.Generator | None = None,
        device: str | torch.device | None = "auto",
    ) -> None:                                                # implemented

    def lr(self, step: int | None = None) -> float:           # implemented

    def train_step(self) -> dict[str, float]:                 # SCAFFOLDED

    def evaluate(self, eval_examples) -> dict[str, float]:    # implemented

    def train(self, eval_examples=None) -> dict[str, list]:   # implemented
```

Total scaffolded code: roughly 60 lines across four locations. The math is light; the lesson is the closed-form derivation, the two-model bookkeeping, and the freeze invariant. The package ships three files: `data.py` (`PreferenceExample` boilerplate plus the scaffolded `pad_and_collate_pref`), `loss.py` (scaffolded `sequence_logprob` and `dpo_loss`), and `trainer.py` (`DPOTrainer` with constructor, `lr`, `_sample_batch`, `_logp_under`, `evaluate`, and `train` already implemented; `train_step` is the one scaffolded method).

## How to run the tests

Tests live in `tests/test_dpo.py`. Initial state: a few boilerplate tests pass (`PreferenceExample` shape, `DPOTrainer.__init__` validation, `optimizer_only_on_policy`, `evaluate_empty_raises`). The rest fail with `NotImplementedError`.

```bash
pytest tests/test_dpo.py                       # all module-14 tests
pytest tests/test_dpo.py -x                    # stop at first failure
pytest tests/test_dpo.py -k pad_and_collate    # collator tests only
pytest tests/test_dpo.py -k sequence_logprob   # log-prob tests only
pytest tests/test_dpo.py -k dpo_loss           # DPO formula tests only
pytest tests/test_dpo.py -k trainer            # trainer tests only
pytest tests/test_dpo.py -v                    # verbose
```

Implementation order — earlier scaffolds unblock later tests:

  1. **`pad_and_collate_pref`** → unblocks the collator tests.
  2. **`sequence_logprob`** → unblocks the log-prob tests.
  3. **`dpo_loss`** → unblocks the DPO-formula tests (independent of step 2 — the loss takes `(B,)` log-prob tensors directly).
  4. **`DPOTrainer.train_step`** → unblocks the end-to-end trainer tests.

Steps 1, 2, and 3 are independent — implement and test them in any order. Step 4 depends on all three.

The end-to-end trainer tests pull in your full Module 03 / 05 / 07 / 08 / 09 / 10 / 13 stack via `TransformerLM` and Module 13's optimizer / clip / schedule. If any of those scaffolds aren't filled in, the trainer tests will fail at the prerequisite layer. The Module 13 deliverable test (`test_sft_trainer_loss_decreases`) is a good gate — if it passes, your prerequisites are in order.

The headline tests to watch:

- **`test_pad_and_collate_pref_chosen_mask_only_on_response`** and **`test_pad_and_collate_pref_rejected_mask_only_on_response`** — pin down that the mask covers EXACTLY the response tokens (chosen or rejected, including trailing `<|end|>`) and nothing else. Same shift-by-one as Module 13.
- **`test_sequence_logprob_uniform_logits_full_mask`** — pin down the log-prob math: with all-zero logits and a full mask, every example's sequence log-prob is exactly `−T · log(V)`.
- **`test_dpo_loss_initial_state_is_log2`** — the canonical sanity value. When policy log-probs equal reference log-probs, the loss is exactly `log 2`. Always.
- **`test_dpo_loss_initial_state_log2_at_various_betas`** — the same invariant holds at any β. β scales the *gradient*, not the initial loss.
- **`test_dpo_loss_strict_preference_below_log2`** and **`test_dpo_loss_anti_preference_above_log2`** — directional sanity. When the policy prefers chosen more than the reference does, the loss drops below log(2); when it prefers rejected more, the loss rises above log(2).
- **`test_dpo_loss_gradient_flows_to_policy_only`** — pins the sign of the gradient: `dL/d(logp_chosen) < 0` (push chosen UP) and `dL/d(logp_rejected) > 0` (push rejected DOWN).
- **`test_dpo_trainer_initial_loss_is_log2`** — the end-to-end version of the step-0 invariant. Wires `DPOTrainer` to a tiny `TransformerLM`, runs one step at `lr=0`, asserts loss is `log 2`. If this fails the data path, the masking, or the reference initialization is broken.
- **`test_dpo_trainer_ref_model_unchanged`** — the load-bearing freeze invariant. Snapshot reference params before training; assert byte-for-byte equality after. If this fails the optimizer is mis-attached or some other update is leaking through.
- **`test_dpo_trainer_loss_decreases`** — the headline end-to-end check. 30 DPO steps on a tiny synthetic dataset must drive the loss meaningfully below `log 2`.

## Exercises

1. **Hand-author a tiny preference dataset.** Write 100 `(prompt, chosen, rejected)` triples. Suggested mix:

     - 40 factual Q&A pairs where chosen is the correct answer and rejected is a plausible-but-wrong answer (`"What is the capital of Spain?"`: chosen=`"Madrid."`, rejected=`"Lisbon."`).
     - 30 style pairs where chosen is concise and rejected is verbose or rambling.
     - 20 helpfulness pairs where chosen directly addresses the prompt and rejected sidesteps it.
     - 10 refusal pairs where chosen politely declines a problematic prompt and rejected complies.

   Save them as JSON. Constraints:

     - **Length-match.** Chosen and rejected should be within ~2× length of each other in tokens. Length bias is the single most-studied DPO failure; control it.
     - **Style-match.** Both completions should use the same surface style (terse, no preamble, ends with `<|end|>`). The only difference should be the *behavior* you're correcting.
     - **One axis at a time.** Don't pair a verbose-incorrect against a terse-correct; the model can't tell which axis you're rewarding.

2. **Run DPO and compare SFT vs DPO outputs.** Load your Module 13 SFT'd checkpoint into TWO `TransformerLM` instances — `model` (policy) and `ref_model` (reference). Wrap your 100 preference examples in `DPOTrainer` and train for **300 steps** at `max_lr=1e-4`, `batch_size=4`, `warmup_steps=10`, `grad_clip=1.0`, `max_seq_len=128`, `beta=0.1`. Save the DPO'd checkpoint.

   Generate from both — same prompt, same seed, same sampling settings. Compare on:

     - **5 prompts where the SFT model gets the answer wrong** (e.g. `"largest city in Spain"` if your SFT model says Lisbon). Does DPO improve any?
     - **5 prompts NOT in the training set** with the same shape as the chosen behaviors. Does DPO transfer?
     - **5 OOD prompts** (poetry continuation, prose completion). Has DPO degraded base capacity?

   Report:
     - Win-rate over the SFT model on the 5 in-distribution prompts.
     - Reward margin (`chosen_reward − rejected_reward`) on the held-out eval set across training steps. Should grow monotonically.
     - One sample per prompt-class where DPO clearly helped, and one where it clearly hurt.

3. **β sweep.** Train at `β ∈ {0.01, 0.05, 0.1, 0.3, 1.0}` (five runs). For each, plot:
     - Final training loss.
     - Final reward margin.
     - One sample on a held-out prompt.

   Expected pattern:
     - `β = 0.01`: aggressive drift, loss drops fast, but samples are gibberish or repetitive (capacity damage).
     - `β = 0.05–0.3`: the sweet spot, samples look better than SFT.
     - `β = 1.0`: policy barely moves, loss stays near `log 2`, samples look like SFT.

   Headline observation: the qualitative gap between `β = 0.05` and `β = 1.0` is much wider than the loss-curve gap. A loss-only plot under-states how much capacity damage low-β runs are doing. Document one extreme example.

4. **Length-bias injection.** Take your dataset; for half of the examples, *deliberately* make the chosen completion noticeably longer than the rejected one. Retrain. Generate from the resulting model on the held-out test prompts. Observe:
     - Are responses systematically longer than the SFT model's?
     - Are they systematically longer than the un-biased DPO model's?
     - Does the DPO accuracy metric (% of pairs with chosen reward > rejected reward) hide the length bias? (Hint: it does.)

   The expected pattern: 50% length-biased data is enough to make the DPO'd model demonstrably verbose. The accuracy metric stays high because the model "correctly" prefers chosen — but it learned the wrong feature. Document one prompt where the length bias is visible.

5. **Reference drift verification.** Snapshot every parameter of `ref_model` before training. Train. Snapshot after. Verify byte-for-byte equality. (This is `test_dpo_trainer_ref_model_unchanged` formalized — but doing it on your real run is reassurance that your training pipeline isn't subtly mutating the reference.)

   If they do match: congratulations, the freeze invariant holds. If they don't: find what's mutating them. Common causes: optimizer constructed with `model.parameters() + ref_model.parameters()`; somebody's using `requires_grad=False` to control freezing but a stray `.zero_grad()` is still corrupting the gradient state.

6. **Implicit-reward calibration.** After DPO, the implicit reward `r̂(x, y) = β · log[π(y|x) / π_ref(y|x)]` is supposed to be a useful "reward function" — better completions should score higher. Test this:
     - Generate 20 random completions from the DPO'd model on 10 held-out prompts.
     - For each (x, y) pair, compute the implicit reward (you have policy and reference; just forward both and compute the log-ratio).
     - Rank the 20 completions per prompt by implicit reward.
     - Read the top-1 and bottom-1 completion. Does the ranking match your intuition? (At toy scale: probably not perfectly, but better than random.)

   Report: across the 10 prompts, what fraction of the time does the top-1 implicit-reward completion match what *you* would have ranked first? This is the "is DPO actually doing what it claims" sanity check.

7. **Optional: DPO without the SFT step.** Run DPO directly from a Module 10 (pretraining) checkpoint, skipping the SFT stage. Both policy and reference start from the base model. Observe:
     - Does DPO still learn the preferences?
     - Does the model emit the chat format correctly?
     - Compare against the SFT-then-DPO model qualitatively.

   The expected pattern: DPO can learn preferences without SFT, but the format collapses (the base model never learned `<|user|>`/`<|assistant|>` markers). The InstructGPT pipeline goes SFT-then-RLHF for a reason: SFT is teaching format; DPO is teaching preference. Conflating the two stages is possible but inefficient. Document one prompt where the format breakage is visible.

8. **Optional: a single bad rejected example.** Take your dataset; in one example, deliberately make the rejected completion contain malformed format (`"Madrid"` without trailing `<|end|>`, or a rogue `<|user|>` mid-response). Train. Observe:
     - Does the gradient through that one example damage the format model on prompts unrelated to it?
     - At toy scale, how many "good" examples does it take to balance one "bad" rejected one?

   This calibrates intuition for what a "noisy" preference dataset costs. At toy scale, ratios of 50:1 (good:bad) can still be visible in outputs.

## Pitfalls to expect

- **Forgetting `torch.no_grad()` on the reference forwards.** Doesn't break correctness — the optimizer's params don't include the reference, so its `.grad` accumulating is just wasted memory. But it does grow the autograd graph through the reference parameters and roughly doubles peak memory. Always wrap reference forwards in `with torch.no_grad():`.

- **Reference and policy share the same model object.** A subtle bug: `ref_model = model` (assignment, not copy). Now both names point to the same params and updating `model` updates `ref_model` in lockstep. Symptom: log-ratios stay zero, loss stays at log(2). Fix: `ref_model = copy.deepcopy(model)`.

- **The mask covers prompt tokens.** If the loss-mask is `1` on prompt tokens (because of a Module 13 cut-and-paste error), the chosen/rejected log-prob "sums" include the prompt's log-probability. Since the prompt is identical for chosen and rejected, this *cancels in the log-ratio* — the DPO loss is exactly the same. So a mask bug here is silent at the loss level but wastes compute.

- **The mask doesn't cover `<|end|>`.** Symmetric to Module 13: if `<|end|>` is masked out, the model never has gradient signal to push its probability up (on chosen) or down (on rejected). Symptom: model never learns to stop — runs to `max_new_tokens` after DPO. Fix: ensure mask covers every response token *including* `<|end|>`.

- **β = 0.** The DPO logits are identically zero, the loss is constant `log 2`, and the gradient is zero. Training is a complete no-op. The trainer's input validation rejects β=0 explicitly, but if you sweep over β values programmatically and accidentally include 0, you'll see "perfectly flat loss curve" — which is its own diagnostic.

- **Length bias.** If your dataset's chosen completions are systematically longer than rejected, DPO learns "be longer." This is the headline DPO failure mode at every scale. Audit before training: compute mean(len(chosen)) vs mean(len(rejected)) over the dataset; they should be within 10–20% of each other.

- **`sequence_logprob` returns a scalar instead of `(B,)`.** Common copy-paste error from Module 13's `masked_cross_entropy` (which DOES return a scalar). DPO needs per-example log-probs because the dpo_loss formula does *per-example* log-ratios before averaging. Symptom: `dpo_loss` raises a shape error or, worse, broadcasts silently and produces nonsense.

- **Mean-pooling instead of sum.** `sequence_logprob` is `Σ_t mask[t] · log p(target_t)`. NOT `Σ_t mask[t] · log p(target_t) / mask.sum()`. The latter is length-normalized log-prob, which is fine for some applications but is a different objective from DPO. The DPO derivation depends on log-probabilities being SUMS — the joint log-probability of the response under the autoregressive factorization. Length-normalizing changes the objective.

- **Reference forgetting between sessions.** If you save and reload checkpoints between training sessions, make sure you re-establish the reference correctly. A common mistake: load the SFT'd checkpoint into `model`, train for 100 steps, save `model`, restart, load saved `model` into both `model` and `ref_model`. Now your reference is the *partially-DPO'd* model, not the SFT'd one. The implicit reward is computed against the wrong baseline. Always reload the original SFT'd checkpoint for the reference, even when resuming.

- **lr too high.** DPO's gradient signal is sequence-level (one log-ratio per example) so the effective magnitude per step is larger than SFT's per-token signal. Default to `max_lr=1e-4` for DPO — about 10× lower than SFT's `3e-4`, and 100× lower than pretraining's `3e-3`. Higher than 1e-3 produces visible drift in 50 steps even at modest β.

- **Implicit-reward collapse.** Both `chosen_reward` and `rejected_reward` go to large negatives; the policy is making both completions much less likely than the reference does. The margin still grows positive (chosen drops less than rejected), so the loss looks fine. But generation is gibberish — the policy has lost most of its base capacity. This is the hardest DPO failure to debug because the loss curve hides it. Always log both `chosen_reward` and `rejected_reward` separately, not just the margin; if both are very negative, lower lr or raise β.

- **Comparing pre/post DPO loss directly.** The DPO loss is on a different scale than the SFT loss — they're not comparable as numbers. The right comparison metrics are: (a) reward margin, (b) accuracy, (c) qualitative samples on held-out prompts. Don't read "DPO loss went from 0.69 to 0.42" as "the model got 39% better."

## M-series notes

DPO is more compute-hungry than SFT but still tractable on M-series:

- **Per-step cost.** Each step does 4 forward passes (policy/ref × chosen/rejected) and 1 backward (only through the policy forwards). Roughly **3× the wall-clock of an SFT step** at the same `(B, T)`. Memory peak is ~2× SFT — both models in memory at once.

- **Total wall-clock estimates** at `max_steps=300`, on a Module-13-SFT'd checkpoint:

  ```
     ┌────────┬────────────┬────────────┬────────────┐
     │ size   │ M1/M2 8GB  │ M2 Pro 32GB│ M3 Max 64GB│
     ├────────┼────────────┼────────────┼────────────┤
     │  1M    │   ~5 min   │   ~3 min   │   ~2 min   │
     │  5M    │  ~12 min   │   ~7 min   │   ~5 min   │
     │  20M   │  ~30 min   │  ~15 min   │  ~10 min   │
     └────────┴────────────┴────────────┴────────────┘
  ```

  The 20M model's DPO comfortably fits in a long-coffee-break window. As with SFT, run on the largest checkpoint you have — quality scales with base-model size, and DPO on a 1M-param base is mostly a debugging exercise (the base capacity isn't there).

- **Memory.** The 2× model footprint matters. On 8GB M1: a 20M model is fine; a 50M model would push the limits. On 32GB+: anything you can pretrain is comfortable.

- **Device.** `DPOTrainer(..., device="auto")` moves both the policy and reference model to MPS when available, then moves each chosen/rejected batch to that same device. Use `device="cpu"` when debugging the two-model bookkeeping.

- **Dataset size.** 100–500 hand-authored preference pairs occupy KB to low MB on disk. The tokenized representation fits comfortably in memory.

- **Evaluation cost.** Generating from the DPO'd model + computing implicit rewards over 100 (x, y) pairs takes 1–2 minutes at our model sizes. The deliverable comparison notebook is fast to iterate.

- **Checkpoint sizes.** Same as Modules 12/13: 1M ≈ 4MB, 5M ≈ 20MB, 20M ≈ 80MB. Storing pretraining + SFT'd + DPO'd versions of the 20M model is ~240MB — comfortable.

- **Mixed precision.** Same caveats as Modules 12/13 — stay in fp32 for DPO. The numerical tolerance for log-softmax + log-sigmoid is tight; fp16 has been a documented source of NaN losses in DPO implementations at production scale, and there's no compute reason to push it at our scale.

- **Reproducibility.** Pass `torch.Generator().manual_seed(seed)` to `DPOTrainer`. Especially important for the β sweep: variance across runs at the same β is moderate at toy scale (1.5–2× the variance across β values), so you want to control the seed when comparing.

- **Two-model debugging.** When debugging DPO, the most useful single trick is to call `_logp_under(self.model, ...)` and `_logp_under(self.ref_model, ...)` on the same `(x, y, mask)` and print both. At step 0 they should be equal (because the policy was just copied from the reference); after training they should diverge in the predicted direction (chosen log-prob diverges UP, rejected log-prob diverges DOWN). If you see them diverging together (both up or both down), that's implicit-reward collapse — lower lr or raise β.

---
## Reading

Primary:

- **Rafailov, Sharma, Mitchell et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" (2023).** The DPO paper. §4 has the full derivation; the appendix has the gradient analysis. Reading the proof that the closed-form solution to the constrained-RL objective gives the DPO loss is the canonical "aha" moment of this module.
- **Ouyang, Wu, Jiang et al., "Training language models to follow instructions with human feedback" (InstructGPT, 2022).** The RLHF paper that DPO is a closed-form simplification of. Read §3.4–3.6 for the reward modeling and PPO stages — knowing what DPO replaces makes the simplification feel more dramatic.
- **Christiano, Leike, Brown et al., "Deep reinforcement learning from human preferences" (2017).** The original deep-RL-from-preferences paper. Predates language models entirely (it's about Atari and Mujoco). Read §2 to see the Bradley-Terry preference model in its pre-LLM form.

Secondary:

- **Bai, Jones, Ndousse et al., "Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback" (Anthropic HH, 2022).** The HH-RLHF paper. Has the most-cited preference dataset (Anthropic HH) and an extensive discussion of length bias, format bias, and helpfulness-vs-harmlessness tradeoffs.
- **Casper, Davies, Shi et al., "Open Problems and Fundamental Limitations of Reinforcement Learning from Human Feedback" (2023).** A survey-style critique of RLHF. Read §3 for the failure modes that DPO inherits from RLHF (reward hacking, distributional shift, hard prompts).
- **Singhal, Goyal, Xu et al., "A Long Way to Go: Investigating Length Correlations in RLHF" (2023).** The length-bias paper. Demonstrates that >50% of the helpfulness improvement RLHF achieves on standard benchmarks comes from length increases alone. Sobering.

Optional:

- **Azar, Rowland, Piot et al., "A General Theoretical Paradigm to Understand Learning from Human Preferences" (IPO, 2023).** A small refinement of the DPO loss that mitigates an over-fitting failure mode. Read §3 for the analysis of where vanilla DPO breaks; the fix itself is two lines of code.
- **Ethayarajh, Xu, Muennighoff et al., "Model Alignment as Prospect Theoretic Optimization" (KTO, 2024).** Drops the pairwise structure entirely — train on individual `(prompt, response, label∈{good,bad})` examples instead of preference pairs. Useful when pairwise preference data is hard to collect.
- **Schulman, Wolski, Dhariwal et al., "Proximal Policy Optimization Algorithms" (2017).** The PPO paper. RLHF's third stage. We don't implement PPO; reading it once gives you the depth chart of "what DPO is replacing."
- **Hejna, Knox, Stone et al., "Inverse Preference Learning" (2023).** A different angle on the same closed-form derivation: instead of optimizing the policy directly, recover the reward implied by the policy. Mostly of interpretability interest.

## Deliverable checklist

- [ ] All tests in `tests/test_dpo.py` pass.
- [ ] Hand-authored preference dataset of 100+ `(prompt, chosen, rejected)` triples in `data/dpo/preferences.json` (or similar). Length distribution audited; chosen and rejected within 2× of each other.
- [ ] DPO'd checkpoint saved to disk, separate from the SFT'd checkpoint. The SFT'd checkpoint is preserved for re-runs and ablations.
- [ ] Notebook: `notebooks/14-dpo-sft-vs-dpo.ipynb`. Loads SFT'd and DPO'd checkpoints, runs `generate` from each on the same 15 prompts (5 in-distribution, 5 transfer, 5 OOD), prints all comparisons side by side. Plots reward margin and accuracy across training. Commit with the outputs visible.
- [ ] One paragraph on what DPO *did* (visible behavioral shift on in-distribution and transfer prompts, calibration of overconfident wrong answers) and what it *didn't* (give the model new factual knowledge, fix capability gaps, prevent format breakage on OOD prompts). The two-list framing is the deliverable; both lists should have at least three items.
- [ ] You can explain — out loud, without notes — the closed-form derivation of the DPO loss from the RLHF objective: KL-constrained max-reward → optimal policy is `π_ref · exp(r/β) / Z` → invert for `r` → substitute into Bradley-Terry → DPO.
- [ ] You can explain — out loud, without notes — why the initial DPO loss is exactly `log 2`, regardless of architecture, data, or β.
- [ ] You can explain — out loud, without notes — what the implicit reward is, why it equals `β · log[π/π_ref]`, and why the prompt's own log-probability cancels.
- [ ] You can explain — out loud, without notes — why the reference model must stay frozen, what happens if it isn't, and how to verify the freeze invariant.
- [ ] You can explain — out loud, without notes — the length bias failure mode and how to detect it before training.
