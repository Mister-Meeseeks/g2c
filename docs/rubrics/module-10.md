# Module 10 Rubric

Use this rubric to grade submitted answers in the student's working notebook (`notebooks/solutions/10-tinyllm.ipynb`, falling back to `notebooks/clean/10-tinyllm.ipynb` if no solutions copy exists yet). Each exercise has `Question:` / `Answer:` string cells; treat a blank `"Answer: "` as not submitted (skip it). If the student wrote a hint or help request inside the answer string, tutor first before grading and avoid giving away the full solution unless they explicitly ask. Give feedback exercise by exercise and avoid replacing the student's work with a full solution unless explicitly asked.

## Exercise 10.01 — Implement `Trainer.train_step`

A correct answer should include:

- `pytest tests/test_pretraining.py` passes.
- The step order is zero gradients, batch, device move, forward, loss, backward, optional clip, scheduled lr, optimizer step, increment counter.
- Metrics include loss, lr, and grad norm.

Common issues:

- Clipping after `optimizer.step`.
- Forgetting `zero_grad`.
- Incrementing the step counter before computing the step's lr.
- Leaving sampled batches on CPU while the model is on MPS.

## Exercise 10.02 — Prepare the first corpus

A correct answer should include:

- Token count, train token count, validation token count, and vocab size.
- `log(V)` baseline.
- A verified `(B, T)` input batch and `(B, T)` target batch.
- A note that this is plumbing, while Module 09B owns the conceptual setup.

Common issues:

- Training and validating on exactly the same tensor without noting it.
- Shuffling individual token IDs.
- Not checking that the corpus is long enough for the chosen context length.

## Exercise 10.03 — Train a tiny TransformerLM

A correct answer should include:

- Model shape and training hyperparameters.
- Train and validation loss curves or clear logged values.
- Loss falls below the `log(V)` baseline.
- Validation loss is reported, not only training loss.

Common issues:

- Judging the run only from one generated sample.
- Changing many knobs before establishing a small baseline.
- Ignoring validation loss.

## Exercise 10.04 — Sample from checkpoints or training milestones

A correct answer should include:

- At least one initial sample and one trained sample.
- A concrete qualitative comparison.
- Recognition that tiny models learn local form before global meaning.

Common issues:

- Expecting coherent assistant-like behavior.
- Treating one bad sample as proof training failed despite falling validation loss.
- Sampling with wildly different settings across checkpoints without noting it.

## Exercise 10.05 — Prove gradient accumulation

A correct answer should include:

Question 1 (why identical):

- Gradients are linear: the gradient of a sum of losses is the sum of the gradients, and `.grad` accumulates across `backward()` calls — the same `+=` rule from the Module 01 engine.
- Because each micro-loss is divided by K, the accumulated sum equals the mean-loss gradient of the full batch; no optimizer steps happen between micro-batches, so the weights are identical throughout.

Question 2 (the missing /K):

- Gradients come out exactly K× too big, which is indistinguishable from multiplying the learning rate by K.
- On a real run this looks like instability or divergence, and the natural-but-wrong response is to blame and lower the LR (or blame clipping) rather than fix the scaling.

Question 3 (clip ordering):

- Clipping is a nonlinear operation (a conditional rescale by `max_norm / total_norm`), so it does not commute with summation — clipping micro-gradients and summing is not the same as summing then clipping.
- Stronger answers note the in-loop clip also rescales the already-accumulated gradient each iteration, compounding the distortion.

Question 4 (the practical config):

- Halve `batch_size` to 4 **and** accumulate over 2 micro-batches per step (effective batch stays 8), dividing each micro-loss by 2 and clipping once before the step.

Common issues:

- Calling the equivalence "approximate" — it is exact up to float summation order (the check passes at `atol=1e-6`).
- Saying accumulation saves time (it saves activation memory; wall clock is unchanged or slightly worse).
- Fixing the /K bug by lowering the learning rate.

## Exercise 10.06 — Run one controlled scale-up

A correct answer should include:

- Exactly one main knob changed, or a clear explanation if more changed.
- Validation loss and sample comparison against the baseline.
- A short interpretation of whether the change helped.

Common issues:

- Changing model size, data size, context length, LR, and steps all at once.
- Comparing runs with different random seeds without noting the variance.

## Exercise 10.07 — Diagnose the run

A correct answer should include:

- Final train loss.
- Final validation loss.
- Final validation perplexity.
- A statement about whether validation tracked training.
- One specific next experiment.

Common issues:

- Reporting only samples.
- Ignoring a train/validation gap.
- Proposing a next experiment that changes several variables at once.
