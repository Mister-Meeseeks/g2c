# Module 10 Rubric

Use this rubric to grade submitted answers in the student's working notebook (`notebooks/solutions/10-your-first-llm.ipynb`, falling back to `notebooks/clean/10-your-first-llm.ipynb` if no solutions copy exists yet). Each exercise has `Question:` / `Answer:` string cells; treat a blank `"Answer: "` as not submitted (skip it). If the student wrote a hint or help request inside the answer string, tutor first before grading and avoid giving away the full solution unless they explicitly ask. Give feedback exercise by exercise and avoid replacing the student's work with a full solution unless explicitly asked.

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

## Exercise 10.05 — Run one controlled scale-up

A correct answer should include:

- Exactly one main knob changed, or a clear explanation if more changed.
- Validation loss and sample comparison against the baseline.
- A short interpretation of whether the change helped.

Common issues:

- Changing model size, data size, context length, LR, and steps all at once.
- Comparing runs with different random seeds without noting the variance.

## Exercise 10.06 — Diagnose the run

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
