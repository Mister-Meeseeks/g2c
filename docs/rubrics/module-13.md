# Module 13 Rubric

Use this rubric to grade submitted answers in the student's working notebook (`notebooks/solutions/13-sft.ipynb`, falling back to `notebooks/clean/13-sft.ipynb` if no solutions copy exists yet). Each exercise has `Question:` / `Answer:` string cells; treat a blank `"Answer: "` as not submitted (skip it). If the student wrote a hint or help request inside the answer string, tutor first before grading and avoid giving away the full solution unless they explicitly ask. Give feedback exercise by exercise and avoid replacing the student's work with a full solution unless explicitly asked.

## Exercise 13.01 — Read the mask table

A correct answer should include:

- Loss lands only on assistant-response positions (mask 1); user-prompt tokens and padding are mask 0.
- `<|end|>` gets mask 1 because stopping is learned behavior: the model must be supervised on *emitting* the stop token or it never learns when to terminate.
- (Implicitly or explicitly) masked tokens still participate in the forward pass as context — they are only excluded from the loss.

Common issues:

- Saying prompt tokens are "ignored by the model" — they are ignored by the *loss*, not the forward pass.
- Confusing the loss mask with the causal attention mask.
- Treating `<|end|>` as a tokenizer artifact rather than a supervised target.

## Exercise 13.02 — Same-pattern heldout sweep

A correct answer should include:

- Specific skill categories that improved (typically formatting, echo/transform patterns close to the training data) and ones that still fail (arithmetic, factual recall).
- The conclusion that at this scale much of the gain is answer *shape* — SFT reliably teaches format; task semantics improve only where pretraining already had some capacity.
- At least one cited sample from the base-vs-SFT comparison.

Common issues:

- Overclaiming semantic learning because answers are format-perfect.
- Judging from a single prompt instead of the category sweep.
- Not distinguishing "follows the template" from "answers correctly."

## Exercise 13.03 — Out-of-distribution prompts

A correct answer should include:

- Compositional behavior breaks down on prompts combining patterns the SFT set never paired; the model falls back to the nearest trained pattern or produces confident nonsense.
- The framing that SFT generalizes style, not new skills — coverage ends where the data ends.
- A concrete surprising sample cited from the run.

Common issues:

- Expecting instruct-model-like generalization from a toy model and grading the model rather than describing it.
- Attributing failure to under-training (more epochs) when the cause is data coverage and model capacity.

## Exercise 13.04 — Loss-mask ablation

A correct answer should include:

- Without the mask the model usually still learns the template (it appears in every example), but it is now also trained to imitate *user* text — symptoms include echoing or continuing the user turn before answering.
- The loss comparison is not apples-to-apples: the unmasked loss averages over a different (larger) set of token positions, so its numeric value sits on a different scale than the masked run.

Common issues:

- Comparing raw loss numbers across mask settings as if lower means better.
- Concluding masking is unnecessary because the format survived the ablation.
- Missing the user-echo failure mode entirely.

## Exercise 13.05 — Format collapse and the poem continuation

A correct answer should include:

- After adding multi-sentence creative examples, the prefilled poem continuation gets longer / multi-sentence where before it stopped after one short sentence.
- The interpretation that response length and stopping behavior are learned from the *distribution of the dataset's answers* — the answer style in the data tightly controls output shape.

Common issues:

- Attributing longer outputs to "more creativity" rather than a shift in the trained answer-length distribution.
- Not rerunning the dataset/training cells after editing `sft_pairs`, so the comparison is against a stale model.

## Exercise 13.06 — Data-size sweep

A correct answer should include:

- The 10-example model roughly learns the template but memorizes/parrots content; the 50-example model covers more patterns with fewer wrong-pattern collisions.
- Format arrives before correctness as data grows.

Common issues:

- Expecting a clean monotone quality curve from a tiny sweep.
- Not recognizing memorization at 10 examples (training answers resurfacing verbatim on unrelated prompts).

## Exercise 13.07 — Pretraining loss after SFT

A correct answer should include:

- Yes: plain-text cross-entropy on held-out prose typically *rises* after SFT.
- The reason: behavior shaping shifts the model's distribution toward chat-formatted text and away from generic prose — a capacity trade-off, expected and not a bug.

Common issues:

- Treating any base-loss increase as evidence that SFT failed.
- Measuring on chat-formatted text and concluding base loss did not move.

## Exercise 13.08 — Stopping on `<|end|>`

A correct answer should include:

- Evidence from samples: generations terminate with `<|end|>` shortly after a complete answer rather than running to the `max_new_tokens` cap.
- The tie-back to 13.01/13.05: the model stops because `<|end|>` was a supervised target, and *where* it stops mirrors the answer lengths in the dataset.

Common issues:

- Citing the generation loop's stop-token cutoff as the evidence, without noting the model must first assign `<|end|>` top probability for the cutoff to trigger early.
- No cited sample at all.
