# Rubric — Beyond: Linear attention

Student answers live in `notebooks/solutions/linear-attention-hybrids.ipynb`
(fall back to `notebooks/clean/linear-attention-hybrids.ipynb`). Grade each
submitted `Question:` / `Answer:` independently; skip blank answers.

## Q1 — Reading the cached-decoding benchmark

- **Correct**: reports the measured crossover, including an honest “none
  in the tested range,” and notes noisy or non-monotonic wall-clock results
  if present. Explains that cached full attention retains and scores against
  `T` key/value rows, so state memory and asymptotic per-token work grow with
  history even when short benchmark timings are noisy, while recurrent
  linear attention carries fixed-size `(S, z)`. Also states that this
  notebook's parallel training reference materializes `T × T` scores and
  therefore does not demonstrate efficient linear-attention training.
- **Mostly correct**: explains the inference asymptotics and observed timing
  but omits the training-implementation caveat.
- **Partially correct**: predicts a crossover from theory without reporting
  the measurement, or compares a full training pass with one decode step.

## Q2 — Reading the TinyStories comparison

- **Correct**: reports the actual validation curves and generated-story
  differences. If gaps are small, plausibly explains that average
  next-token loss can underweight relatively rare exact-retrieval events.
  If gaps are large, treats that as evidence about this simple feature map,
  fixed decay, optimization, or scale rather than forcing the expected
  story. Does not claim equal parameter count without acknowledging the
  few additional decay scalars.
- **Partially correct**: gives the expected locality argument without
  engaging with the observed curves or samples.
- **Needs revision**: claims the architectures are equivalent because their
  parameter counts are near-matched.

## Q3 — Reading the recall plot

- **Correct**: describes all measured curves, including surprising ordering
  or lack of separation. Connects plausible distance degradation to
  interference and decay in a fixed-size state, or plausible hybrid recovery
  to an exactly addressable full-attention layer. Proposes a useful control
  such as another seed, longer training, training-loss comparison, an easier
  probe, or a larger model to distinguish architecture from optimization.
- **Mostly correct**: explains the observed pattern mechanistically but does
  not propose a discriminating control.
- **Partially correct**: repeats the anticipated full/linear/hybrid ordering
  when the student's plot does not support it.

## Q4 — Fixed decay versus an input-dependent gate

- **Correct**: reports how the state norm changed with decay and where the
  learned head values moved from their approximately `0.99` initialization.
  Explains that this module's `γ_h` is constant across all tokens handled by
  a head, while an input-dependent gate can retain or forget differently
  according to token content or context.
- **Partially correct**: identifies geometric forgetting but calls the fixed
  scalar content-selective.

## Q5 — Four versions of the one-million-token bill

- **Correct**: at `T=1,000,000`, `D=4096`, `H=32`, fp16:
  - MHA KV per layer is `2·T·H·d_h·2 bytes = 16.384 GB`
    (about `15.26 GiB`);
  - GQA with `H_kv=8` is `2·T·H_kv·d_h·2 bytes = 4.096 GB`
    (about `3.81 GiB`);
  - MHA with a fixed live window `W=4096` is
    `2·W·H·d_h·2 bytes = 67,108,864 bytes` (`64 MiB`);
  - linear `(S,z)` per layer is
    `(H·(D/H)^2 + D)·2 = 1,056,768 bytes` (about `1.01 MiB`);
  - four MHA layers are about `65.5 GB`, while `[L,L,L,F]` is about
    `16.4 GB` plus roughly `3.2 MB` of recurrent state.
  Identifies MHA and GQA as linear in `T`, the fixed-window cache as bounded
  by `W`, and recurrent state as independent of `T` but quadratic in head
  dimension. Concludes that the arithmetic alone proves neither quality nor
  wall-clock speed.
- **Mostly correct**: right formulas and order of magnitude with a minor
  decimal-GB/GiB slip.
- **Partially correct**: treats GQA as fixed-size, lets a fixed window grow
  with full history, omits `z` or the stack totals, or turns memory arithmetic
  into an unsupported quality/latency claim.
