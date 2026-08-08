# Rubric — Beyond: Midtraining

Student answers live in `notebooks/solutions/midtraining-continued-pretraining.ipynb`
(fall back to `notebooks/clean/midtraining-continued-pretraining.ipynb`). Grade
each submitted `Question:` / `Answer:` independently; skip blank answers.

## Q1 — Objective boundary

- **Correct**: identifies that pretraining and this module's midtraining both
  apply causal next-token loss across raw token streams, while SFT uses
  prompt/response examples and masks prompt tokens. Explains that changing all
  weights does not by itself distinguish the stages.
- **Partially correct**: says midtraining is “between” pretraining and SFT but
  does not identify the data/loss distinction.

## Q2 — Mixture and fairness

- **Correct**: explains source-first window sampling, 80/20 in expectation,
  and why equal total tokens controls approximate compute while intentionally
  giving replay fewer Python tokens. Notes that equal Python exposure would
  give replay extra total tokens.
- **Partially correct**: understands the ratio but not the fairness control.
- **Needs revision**: proposes concatenating sources and allowing windows to
  cross their boundary.

## Q3 — Schedule and branch isolation

- **Correct**: both branches start from the identical base checkpoint, use a
  fresh optimizer and the same lower-LR rewarm/cosine schedule, and consume
  307,200 tokens. Explains that starting replay from the Python-only model
  would confound branch and training duration.
- **Partially correct**: reports matching steps without connecting batch and
  context length to the token budget.

## Q4 — Adaptation versus retention

- **Correct**: interprets all nine loss cells or their deltas. Identifies
  Python improvement as adaptation, positive general/story deltas as
  forgetting, and the replay branch as a tradeoff rather than an unconditional
  winner. Reports a null or reversed result honestly if observed.
- **Partially correct**: discusses only Python loss or only final training
  loss.
- **Needs revision**: treats lower in-domain training loss as evidence that no
  capabilities were lost.

## Q5 — Behavioral evidence

- **Correct**: compares fixed prompts across all checkpoints, uses samples as
  secondary evidence alongside held-out loss, and labels syntax-validity
  experimental if it is sparse or degenerate at 30M scale.
- **Partially correct**: provides samples but changes prompts or decoding
  settings between checkpoints.

## Q6 — Optional ratio sweep

- **Correct**: if attempted, holds total tokens and schedule fixed, plots or
  tabulates both target and retention outcomes, and describes a frontier
  rather than selecting by Python loss alone.
- This question is optional; a blank answer is not incomplete work.
