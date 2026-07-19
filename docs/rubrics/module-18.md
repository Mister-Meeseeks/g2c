# Module 18 Rubric

Use this rubric to grade submitted answers in the student's working notebook (`notebooks/solutions/18-tools.ipynb`, falling back to `notebooks/clean/18-tools.ipynb` if no solutions copy exists yet). Each exercise has `Question:` / `Answer:` string cells; treat a blank `"Answer: "` as not submitted (skip it). If the student wrote a hint or help request inside the answer string, tutor first before grading and avoid giving away the full solution unless they explicitly ask. Give feedback exercise by exercise and avoid replacing the student's work with a full solution unless explicitly asked.

The live exercises run against the student's chosen backend, so tool choices and failures vary. Grade cited transcript evidence and correct layer attribution (parse vs validate vs dispatch vs model choice), not specific outputs.

## Exercise 18.01 — Errors as ToolResult, not exceptions

A correct answer should include:

- An uncaught exception would crash the harness loop mid-run — transcript lost, no chance to recover. Wrapping as `ToolResult(is_error=True)` turns the failure into conversation: the error text is formatted as a `<tool_error>` block and fed back so the model can correct its next attempt.
- The framing that model output is untrusted and noisy — unknown tools and bad arguments are *expected* inputs to dispatch, not exceptional states.

Common issues:

- Framing the choice as code cleanliness rather than loop survival plus model recovery.
- Conflating the parse-layer policy (malformed blocks silently skipped) with the dispatch-layer policy (loud error result).
- Thinking the model sees a Python traceback — it sees the error message as text in its next prompt.

## Exercise 18.02 — Why AST whitelist beats eval()

A correct answer should include:

- `eval()` with restricted globals is a denylist: you try to remove dangerous capabilities, but documented escapes (e.g., dunder chains like `().__class__.__base__.__subclasses__()`) climb back to dangerous objects anyway.
- The AST walk is a whitelist: parse first, admit only explicitly allowed node types (Constant, BinOp, UnaryOp, allowed operators), reject everything else *by node type* before anything evaluates. The reachable surface is structurally bounded to what you admit, and nothing executes until the whole tree validates.

Common issues:

- "AST is safer because it parses the input" without the whitelist-vs-denylist distinction, which is the actual point.
- Believing `eval(expr, {"__builtins__": {}})` is safe.
- Describing the defense as pattern-matching the source string — rejection happens on node types, not regexes.

## Exercise 18.03 — The recovery feedback

A correct answer should include:

- What was fed back: the bad call (`{"expr": ...}`) failed validation, and the loop appended a `<tool_error name="calculator" id=...>` block containing the specific validation message (missing required `expression` / unknown key `expr`) to the transcript.
- Why that suffices: the next prompt shows the model its own failing call, the exact error, and the calculator's schema (still in the system prompt) side by side — everything needed to emit `{"expression": ...}` on the next turn.

Common issues:

- Saying the harness fixed the arguments — the harness only reflects the error; the model must produce the corrected call.
- Missing that the error travels as text in the prompt, not as an exception or API signal.
- Omitting the schema in the system prompt as the reference the model corrects against.

## Exercise 18.04 — Live read-and-compute chain

A correct answer should include:

- The observed chain from their step table — typically `read_file("numbers.txt")` then `calculator` (or `run_python`) — in order.
- The stopping behavior, tied to `stopped_reason`: a clean stop is a final completion with zero tool calls (`no_more_calls`); kept-going shows as redundant calls or a `max_steps` stop.

Common issues:

- Describing the expected chain instead of the one actually observed.
- Not checking `stopped_reason` before characterizing the stop.
- Calling the run a failure without checking whether a mid-chain tool call errored and was recovered.

## Exercise 18.05 — run_python vs calculator for sales.csv

A correct answer should include:

- Which tool(s) the model actually used, from their run.
- Why `run_python` is more reliable for row-by-row arithmetic: the program reads the file and computes revenue in a loop, so the numbers never pass through the model's token stream. The calculator path forces the model to transcribe every value from the `read_file` observation into one expression — each transcription is an error opportunity, and the calculator has no loops or variables.

Common issues:

- Preferring the calculator "because it's simpler" without engaging the transcription-error argument.
- Missing that the calculator path still requires reading the file first — the fragile step is the copy, not the read.
- Ignoring the safety asymmetry: `run_python` has a far larger blast radius, which is why sandboxing matters.

## Exercise 18.06 — Tool ablation

A correct answer should include:

- Both results cited: the direct completion for 8437 * 29 (correct answer 244673) — which a 3B model often gets confidently wrong — versus the exact tool-assisted answer.
- The conclusion: the tool path earns its overhead where exact computation (or fresh external data) is required and unaided reliability is poor; it is pure latency overhead for questions inside the model's competence. Reliability across runs, not one sample, is the criterion.

Common issues:

- Concluding from a single lucky direct answer that tools are unnecessary.
- Not checking either answer against the true product.
- Ignoring the cost side (extra steps and latency) entirely.

## Exercise 18.07 — Where tool use broke

A correct answer should include:

- The most common observed failure named with at least one concrete example from their transcripts, in one of the offered categories (malformed JSON, wrong tool choice, looping, stopping too early, or another named mode such as quoting collisions in `run_python` code).
- Correct layer attribution: parse failure (block never extracted) vs validation failure (`<tool_error>`) vs model-choice failure (wrong tool, premature stop) are different layers.

Common issues:

- A generic answer with no transcript evidence.
- Misclassification — e.g., calling a validation rejection "malformed JSON".
- Blaming the harness for a model-choice failure, or the model for a parser/format mismatch.

## Exercise 18.08 — One change before Module 19

A correct answer should include:

- Exactly one proposed change (formatting, stop criteria, stricter parsing, more examples, or richer evals) explicitly tied to a failure they observed — the causal link is the graded content.
- A plausible mechanism for how the change prevents that failure (e.g., wrong-tool choices → richer tool descriptions/examples in the prompt; premature stops → stop-criteria or prompt changes; unmeasured reliability → a small eval suite).

Common issues:

- A shotgun list of changes instead of one.
- A change that addresses no failure seen in their runs.
- Reaching for model retraining when a harness or prompt fix addresses the observed failure.
