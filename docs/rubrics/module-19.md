# Module 19 Rubric

Use this rubric to grade submitted answers in the student's working notebook (`notebooks/solutions/19-agent.ipynb`, falling back to `notebooks/clean/19-agent.ipynb` if no solutions copy exists yet). Each exercise has `Question:` / `Answer:` string cells; treat a blank `"Answer: "` as not submitted (skip it). If the student wrote a hint or help request inside the answer string, tutor first before grading and avoid giving away the full solution unless they explicitly ask. Give feedback exercise by exercise and avoid replacing the student's work with a full solution unless explicitly asked.

Live agent runs vary by backend and model. Grade cited step tables / trajectories from the student's runs and correct attribution (parse vs tool choice vs loop control vs model capability), not specific outputs.

## Exercise 19.01 — Final Answer wins

A correct answer should include:

- Final Answer wins: the parser sets `final_answer` and drops the action (the `ParsedStep` invariant — if `final_answer` is set, `action` is `None`).
- Why that is right for loop control: Final Answer is the model's termination signal, and honoring it guarantees the loop halts when the model says it is done. Executing the trailing action instead would take an action the model claims not to need, produce a new observation, and can keep the loop alive — burning steps or discarding the committed answer.

Common issues:

- Arguing action-should-win "because the tool result might improve the answer" without addressing termination at all.
- Not knowing what their own parser does — the notebook's "Final answer wins" demo prints it.
- Treating any deterministic precedence as equally good; the asymmetry (final answer = stop signal) is the point.

## Exercise 19.02 — Why thoughts re-render

A correct answer should include:

- The model is stateless; re-rendering its own past Thought lines gives the next call its chain of reasoning — *why* each action was taken — so it can build on prior conclusions instead of re-deriving intent from raw observations. Observations alone are uninterpreted data: a file dump does not say what it was read for.
- The format effect: each rendered Thought/Action/Observation block is an in-context example of the marker protocol, anchoring the model in-format for the next step.

Common issues:

- "So the model remembers" without noting the prompt is doing the remembering — the model itself retains nothing.
- Claiming thoughts are only for human debugging.
- Missing the format-anchoring effect entirely.

## Exercise 19.03 — Runaway loop vs legitimate retry

A correct answer should include:

- The distinguishing feature: whether a repeat can produce new information. A runaway repeats an identical call after a successful *deterministic* observation — the result cannot differ; the model failed to absorb what it saw. A legitimate retry follows an error or transient result, or changes arguments/tool.
- The heuristic (same tool + same args, consecutive) is a proxy that cannot read intent — which is why it is a flag, not a hard rule.
- A shipped default with a rationale. `loop_detection=True` (the course default) is well defended: deterministic tools never return different outputs, and each wasted step burns latency and context; disable per task for transient-failure tools (network search) or polling patterns.

Common issues:

- Picking a default with no rationale.
- Claiming the heuristic can distinguish retry intent.
- Missing that an identical call to a deterministic tool is provably information-free.

## Exercise 19.04 — Plan contribution

A correct answer should include:

- A comparison of the two trajectories cited from their runs (tool sequence and step count with `plan=False` vs `plan=True`); the typical effect is a more direct read → compute → answer sequence with fewer wandering steps when planned.
- The cost-benefit: planning is one extra backend call (seconds of latency) — worth it for multi-step tasks with obvious structure, pure overhead for one-shot tool calls.

Common issues:

- A verdict with no cited sequences from either run.
- Expecting the plan to be enforced — it is a soft prior in the prompt; deviation is not failure.
- "Planning is good" in general, without the latency side.

## Exercise 19.05 — Tool loop vs agent

A correct answer should include:

- Which harness was easier for the model on their task, with transcript evidence (steps, stop reason, answer quality).
- Where ReAct earns its overhead: the Thought slot improves tool selection on multi-step tasks, the scratchpad carries reasoning across steps, and the layered stop conditions plus `[error]` observations give structured recovery. The Module 18 loop is leaner and fine for single-call tasks.

Common issues:

- Attributing the difference to wire format — in the notebook's comparison both paths may use the native channel; this exercise is about loop structure, not format (that is 12b).
- A verdict with no evidence from the runs.
- Treating ReAct as strictly better while ignoring its token and latency cost.

## Exercise 19.06 — ReAct vs native channel

A correct answer should include:

- Which channel produced the cleaner trajectory (usually `NativeAgent`), with the specific failure mode named on the worse channel — for ReAct typically a bad parse or format wobble (missing markers, JSON mid-prose, Action without Action Input), premature prose answers, or a `no_progress`/parse-error stop.
- The tie to post-training: structured tool calling is in the model's fine-tuning distribution (the server translates to the model's own trained format), while ReAct markers appear mostly in pretraining data — so small models wobble on ReAct, not because ReAct is intrinsically harder.

Common issues:

- Attributing ReAct failures to model weakness in general rather than format familiarity.
- No concrete failure mode named from the worse channel.
- Claiming the native channel "has no parser" — the inference server still parses the model's native delimiters; the parsing moved, it did not vanish.

## Exercise 19.07 — Most informative failure

A correct answer should include:

- One concrete live failure with its transcript evidence, placed in one of the offered categories (bad parse, wrong tool, invalid arguments, duplicate action, premature final answer, missing context).
- A category that matches the evidence: `parse_error` set → bad parse; error observation naming an unknown tool → wrong tool; validation error → invalid arguments; `duplicate_action` stop → looping.
- What the failure revealed — about the prompt, the parser, the tools, or the model — since "informative" is the ask.

Common issues:

- A category that does not match the cited evidence.
- Reporting a deterministic fake-backend cell as the "live" failure.
- Naming a failure but skipping what it taught.

## Exercise 19.08 — One loop change

A correct answer should include:

- One targeted change (prompt, parser, tools, scratchpad cap, planning toggle, or stop conditions) explicitly tied to a failure observed in their runs, with a plausible mechanism — e.g., parser tolerance for the exact wobble seen; a better tool description for a wrong-tool choice; `halt_on_stuck` or `loop_detection` tuning for their stop-condition issue.

Common issues:

- A list of changes instead of one.
- A change targeting a failure never observed.
- Model-scale answers (retrain, bigger model) when a harness knob addresses the observed failure.
