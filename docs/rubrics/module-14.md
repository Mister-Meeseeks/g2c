# Module 14 Rubric

Use this rubric to grade submitted answers in the student's working notebook (`notebooks/solutions/14-dpo.ipynb`, falling back to `notebooks/clean/14-dpo.ipynb` if no solutions copy exists yet). Each exercise has `Question:` / `Answer:` string cells; treat a blank `"Answer: "` as not submitted (skip it). If the student wrote a hint or help request inside the answer string, tutor first before grading and avoid giving away the full solution unless they explicitly ask. Give feedback exercise by exercise and avoid replacing the student's work with a full solution unless explicitly asked.

## Exercise 14.01 — Preference-dataset bias check

A correct answer should include:

- A named bias actually checked — length bias is the canonical one (the notebook computes average chosen vs rejected token counts and the max length ratio); category balance or surface-form/formality differences also count.
- The finding, with numbers cited from the audit cell (in the starter set, chosen and rejected lengths are approximately matched by design — that's the expected result).
- (Ideally) why it matters: if chosen completions are systematically longer/more formal, DPO learns "be longer"/"be formal" instead of the intended preference.

Common issues:

- Naming a bias without reporting what the check actually found.
- Treating a matched-length result as "the check was pointless" — a clean audit is the point; the bias would silently redefine the training signal.
- Checking only one pair by eye instead of the dataset-level averages/ratios.

## Exercise 14.02 — Why the collator mask covers only response targets

A correct answer should include:

- The mask makes `sequence_logprob` return exactly `log π(response | prompt)`: mask 1 on chosen/rejected response targets (including `<|end|>`), mask 0 on prompt tokens and padding.
- The prompt is the *shared* prefix of both sequences, so it carries no preference signal — for a causal model its token log-probs are identical for the chosen and rejected branches; scoring it would mean comparing full prompt+response likelihoods, diluting the response distinction the objective is supposed to isolate.
- Padding must be mask 0 because pad positions are meaningless filler; scoring them makes the implicit reward depend on the pad token's log-probability — and chosen/rejected have different lengths, so pad contributions would not even cancel.

Common issues:

- Confusing the loss mask with the causal attention mask (prompt tokens still participate in the forward pass as context).
- Explaining only the prompt half and missing padding (or vice versa) — the two exclusions fail differently.
- Forgetting the mask is aligned with the shifted targets `y` (same shift-by-one as Module 13) and that `<|end|>` is a supervised target.

## Exercise 14.03 — Why the initial DPO loss is log(2)

A correct answer should include:

- At step 0 the policy is an exact copy of the reference, so both log-ratios (`log π − log π_ref` for chosen and for rejected) are zero, the margin is zero, `σ(0) = 0.5`, and `loss = −log 0.5 = log 2 ≈ 0.693`.
- That this holds regardless of architecture, data, or β — it's the canonical DPO sanity check; a different step-0 value means the data path, masking, or reference copy is wrong.

Common issues:

- Attributing the value to properties of the preference data or to a 50/50 label distribution — it comes purely from policy == reference.
- Treating log 2 as an approximate/empirical observation instead of an exact identity.
- Missing the sigmoid step (jumping from "margin is zero" straight to the number without `σ(0) = 0.5`).

## Exercise 14.04 — Why the reference must stay frozen

A correct answer should include:

- The reference is the anchor of the implicit reward: the loss is a function of log-ratios `log(π/π_ref)`. If the reference updated along with the policy, the ratios stay near zero, the margin never grows, and there is no learning signal — the loss no longer measures improvement over the SFT starting point.
- The reference is also the drift tether: β penalizes divergence *from the reference*; a moving reference removes the anchor the KL-style regularization is defined against.
- (Ideally) how it's enforced/verified: reference forwards run under `torch.no_grad()`, and the freeze invariant can be checked by snapshotting reference parameters before and after training and asserting equality.

Common issues:

- Saying the reference is frozen "to save memory/compute" — it's a correctness requirement, not an optimization.
- Confusing the reference model with a reward model (DPO has no separate reward model; the policy/reference log-ratio *is* the implicit reward).
- No mention of what concretely goes wrong if it trains (ratios pinned at ~0, no signal).

## Exercise 14.05 — Reward margins vs free samples

A correct answer should include:

- What their score table showed relative to the free samples (run-dependent; typically yes — margins move visibly even when sampled answers look unchanged), with at least one cited row or sample.
- The mechanism: DPO optimizes the *relative* score of specific chosen-vs-rejected completion pairs. It can raise `log π(chosen)` relative to `log π(rejected)` without flipping which token is argmax at each step of an open-ended generation — so the pairwise margin is the direct view of the objective, and free samples are an indirect one.

Common issues:

- Concluding "DPO did nothing" from unchanged free samples while the margin table clearly moved.
- Reading the DPO loss value as a quality score instead of looking at margins/accuracy.
- Not citing any evidence from either the table or the samples.

## Exercise 14.06 — Where DPO helped and where it hurt

A correct answer should include:

- Specific improvements cited from the targeted-failure probes (behaviors the preference set directly covered: factual pairs, format, honesty axes) and specific failures or regressions — typically weak generalization on heldout/OOD prompts, or degraded fluency where the policy drifted. Details are run-dependent; grade the cited evidence from both probe groups.
- The framing that DPO shifts probability toward what the chosen examples share and only where the base model already has capacity — it does not install new knowledge, so coverage ends where the preference data ends.

Common issues:

- Listing only wins or only failures — the two-sided read is the exercise.
- Grading the model against instruct-model expectations instead of describing the shift.
- Attributing every failure to under-training rather than data coverage/model capacity.

## Exercise 14.07 — Choosing beta

A correct answer should include:

- A specific β from their sweep (values 0.05/0.1/0.3; the winner is run-dependent, often the middle) justified with evidence from both the sweep curves (reward margin movement) and samples (base behavior intact vs degraded).
- The correct reading of the knob: β is the KL-regularization/drift coefficient — small β lets the policy drift far from the reference (more movement, more risk of collapse/gibberish); large β pins the policy near the reference (safe but may under-move). It is *not* the optimizer learning rate — AdamW's step size (`max_lr`) is a separate knob in the same config.

Common issues:

- Calling β the learning rate, or reasoning about it as if it scaled gradient step size rather than the reference tether.
- Picking a β from margin movement alone without checking whether samples degraded (drift damage), or vice versa.
- Expecting a monotone "bigger β is better/worse" — the point of the sweep is the tradeoff shape.
