# Rubric — Beyond: Agent harness engineering

Student answers live in `notebooks/solutions/harness-reliability.ipynb`
(fall back to `notebooks/clean/harness-reliability.ipynb`). Grade each
submitted `Question:` / `Answer:` independently; skip blank answers.

## Q1 — What the event log adds

- **Correct**: names two concrete records, such as call ids,
  intent-before-outcome ordering, result status, or stop reason, and
  pairs each with a post-hoc question it answers.
- **Partially correct**: names records without explaining their audit or
  recovery value.

## Q2 — Resume and unknown outcome

- **Correct**: distinguishes a later backend crash, where completed
  results are replayable, from an unresolved `tool_call`. Explains that
  the identical unresolved log can mean either “not executed” or
  “effect landed before result logging”; call-id dedupe alone cannot
  distinguish them. The teaching runner conservatively does not rerun
  and records `unknown_outcome_after_crash`. Mentions reconciliation,
  a tool-level idempotency key, or a transaction as the stronger fix.
- **Partially correct**: locates the execute/result-log window but calls
  the outcome definitely aborted or claims the call id creates
  exactly-once behavior.
- **Needs revision**: says unresolved calls are automatically safe to
  retry.

## Q3 — Live context drift

- **Correct**: uses observed behavior from both live runs. The naive
  policy loses the task and the prompt-sensitive backend takes a drift
  action; the extractive policy preserves the task and reaches the
  expected state. Distinguishes the narrow implemented invariants
  (task and recent event) from richer production commitments that
  would need structured state or tested summarization. Places resolved
  instructions such as applicable `AGENTS.md` rules in a retained
  instruction layer rather than compactable trajectory history, and
  notes that discovery, scope, and precedence must be deterministic.
- **Partially correct**: describes only the rendered context rather
  than the resulting behavior.
- **Needs revision**: treats context loss as a process crash.

## Q4 — Failure and permission policy

- **Correct**: transient retry can help because the world may change;
  deterministic retry cannot help until the request changes; denied
  calls are policy refusals, not transient failures; repeat budgets
  halt rather than solve a loop. Also states that the permission table
  is not OS-level containment.
- **Partially correct**: gets transient versus deterministic right but
  omits permission or repetition behavior.

## Q5 — Controlled comparison

- **Correct**: reports exact-verifier outcomes and operational metrics
  for both harnesses under the same scripted scenarios. Separates
  clean-run parity from faulted differences, identifies which harness
  behavior caused each difference, distinguishes process resumes from
  tool retries, and reports null rows honestly.
- **Partially correct**: reports only final-answer presence or success
  without operational evidence.
- **Needs revision**: changes the backend, task, or fault schedule
  between harnesses and attributes the difference to the harness.

## Q6 — Optional ProdLM transfer

- **Correct**: if attempted, separates model-format/tool-selection
  failures from harness-policy failures and avoids generalizing from a
  tiny optional sample. Recognizes why a weak backend can swamp the
  harness comparison.
- This question is optional; a blank answer is not incomplete work.
