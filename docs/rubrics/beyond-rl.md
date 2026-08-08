# Rubric — Beyond: Reinforcement learning for LLMs

Student answers live in `notebooks/solutions/rl-grpo.ipynb` (fall back
to `notebooks/clean/rl-grpo.ipynb`). Grade each submitted `Question:` /
`Answer:` independently; skip blank answers.

## Q1 — Reading the format smoke test

- **Correct**: reports the measured held-out generated pass rates before and
  after training, then describes the actual reward and skip curves without
  assuming their direction. Explains that an all-failure and an all-success
  group are both degenerate, so a skip must be interpreted alongside held-out
  generations and pass rate.
- **Mostly correct**: reports the measurements and identifies degenerate
  groups, but treats the training reward as interchangeable with held-out
  evaluation.
- **Needs revision**: claims that a rising skip fraction necessarily proves
  success, or reports sampled training reward as the final result.

## Q2 — The held-out arithmetic-choice result

- **Correct**: reports the measured pre/post greedy pass rates on prompts not
  used for training and cites generated answers. Uses training reward, KL,
  sampled-token entropy, and skip rate to qualify—not replace—the held-out
  result. Distinguishes evidence of task learning from policy drift and is
  willing to conclude that the finite run showed no clear improvement.
- **Mostly correct**: reports held-out rates and samples with a plausible
  interpretation, but omits or misreads one diagnostic.
- **Needs revision**: infers arithmetic improvement from training reward
  alone, evaluates on the training prompt pool, or treats sampled-token
  entropy as an exact vocabulary-wide entropy calculation.

## Q3 — Auditing the sloppy verifier

- **Correct**: reports both the sloppy verifier's claimed held-out pass rate
  and the honest verifier's pass rate, then cites the student's own generated
  samples. Explains that the sloppy verifier rewards membership anywhere in
  the completion rather than the final answer. If an exploit appeared,
  identifies it from evidence; if none appeared, says so and distinguishes
  an exploitable specification from successful discovery in this finite run.
  Patches the gap by scoring only the last integer or a delimited answer slot.
- **Mostly correct**: identifies the gap and proposes a sound patch but does
  not support the behavioral claim with both held-out scores and samples.
- **Needs revision**: announces reward hacking solely because training reward
  rose, describes a canned number-spraying result that the samples did not
  show, or proposes lowering the learning rate instead of fixing the verifier.

## Optional extensions

- **No-KL run**: accept any measured outcome that compares held-out behavior,
  KL, sampled entropy, and generations. A short run need not collapse.
- **Group-size sweep**: accept any empirical result at fixed total rollout
  budget that discusses estimator variance and degenerate-group frequency;
  there is no predetermined best `K`.
