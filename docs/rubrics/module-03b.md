# Module 03B Rubric

Use this rubric to grade submitted answers in `answers/module-03b.md`. If `Help request / hint request` is filled, tutor before grading and avoid giving away the full solution unless explicitly asked. Blank `Student answer` sections are not wrong; skip them unless the student asks for a completeness check. Give feedback by exercise and avoid replacing the student's work with a full solution unless explicitly asked.

## Exercise 03B.01 — Learning-rate sweep

A correct answer should include:

- A too-small LR crawls: loss decreases slowly or barely moves within the budget.
- A useful LR lowers loss quickly early and then slows as it approaches a better solution.
- A too-large LR is unstable: loss spikes, oscillates, or becomes NaN.
- The student compares runs with model, data, seed, and step count held fixed.

Common issues:

- Comparing runs with multiple changed knobs.
- Treating the largest non-diverging LR as automatically best.
- Reading only the final point and ignoring curve shape.

## Exercise 03B.02 — AdamW by hand

A correct answer should include:

- First moment after one step: `m = 0.01`.
- Second moment after one step: `v = 0.00001`.
- Bias-corrected values: `m_hat = 0.1`, `v_hat = 0.01`.
- Adaptive update approximately `0.001 * 0.1 / (sqrt(0.01) + 1e-8)`, or about `0.001`.
- New parameter approximately `0.999` with no weight decay.

Common issues:

- Forgetting bias correction.
- Taking `sqrt(v)` before bias correction and getting the wrong scale.
- Applying weight decay even though the exercise says none.

## Exercise 03B.03 — Implement AdamW

A correct answer should include:

- `pytest tests/test_optim.py` passes.
- `step_count` increments once per optimizer step.
- `m` and `v` update in place and stay aligned with `params`.
- Parameters with `grad is None` are skipped.
- The update runs under `torch.no_grad()`.

Common issues:

- Incrementing `step_count` once per parameter.
- Reinitializing `m` and `v` every step.
- Applying Adam-style coupled weight decay instead of AdamW decoupled decay.

## Exercise 03B.04 — SGD vs AdamW

A correct answer should include:

- Both optimizers train the same architecture on the same data for the same number of steps.
- A loss-curve plot or clear numerical comparison.
- A statement about sensitivity to LR: SGD is usually more sensitive to the global LR; AdamW adapts per-parameter step scales.
- A caveat that AdamW does not change the model or objective.

Common issues:

- Changing model size or data between optimizer runs.
- Comparing SGD and AdamW at the same nominal LR without noting that typical LR ranges differ.
- Claiming AdamW is always better in every setting.

## Exercise 03B.05 — Gradient clipping demo

A correct answer should include:

- The global norm of gradients `[3]` and `[4]` is `sqrt(3^2 + 4^2) = 5`.
- Clipping to `max_norm=1` scales all populated gradients by `1/5`.
- New gradients are `[0.6]` and `[0.8]`.
- The direction of the full gradient vector is preserved.

Common issues:

- Clipping each gradient independently to `1`.
- Returning the post-clip norm when the function contract asks for pre-clip norm.
- Treating clipping as a sign change or thresholding operation.

## Exercise 03B.06 — Warmup/cosine schedule

A correct answer should include:

- During warmup, LR increases linearly toward `max_lr`.
- At the last warmup step, LR reaches `max_lr` under this course's convention.
- At the first cosine step, LR is still near or equal to `max_lr`.
- During cosine decay, LR decreases smoothly toward `min_lr`.
- At or after `max_steps`, LR is held at `min_lr`.

Common issues:

- Starting warmup at `min_lr` rather than near zero.
- Decaying before warmup finishes.
- Letting LR become negative after `max_steps`.

## Exercise 03B.07 — Curve diagnosis

A correct answer should include:

- Both train and val high/flat: likely under-optimized, LR too low, optimizer weak, model too small, or data/label issue.
- Train low and val high: overfitting; try more data, smaller model, regularization, or earlier stopping.
- Both decreasing smoothly: training is healthy; continue or decay LR as planned.
- Recommendations should change one or two knobs at a time.

Common issues:

- Calling every bad curve overfitting.
- Ignoring validation loss.
- Changing model, data, optimizer, LR, and schedule simultaneously.
