# Module 20 Rubric

Use this rubric to grade submitted answers in the student's working notebook (`notebooks/solutions/20-capstone.ipynb`, falling back to `notebooks/clean/20-capstone.ipynb` if no solutions copy exists yet). Each exercise has `Question:` / `Answer:` string cells; treat a blank `"Answer: "` as not submitted (skip it). If the student wrote a hint or help request inside the answer string, tutor first before grading and avoid giving away the full solution unless they explicitly ask. Give feedback exercise by exercise and avoid replacing the student's work with a full solution unless explicitly asked.

The capstone questions are about the integrated system. Grade correct layer attribution (conversation vs retrieval vs tools vs agent loop vs backend) and cited evidence from the student's own turns, not specific model outputs.

## Exercise 20.01 — Scratchpad vs Conversation

A correct answer should include:

- The two structures have different lifecycles and bandwidth: the scratchpad is per-task, high-bandwidth (every thought/action/observation), and dropped when `Agent.run` ends; the Conversation is per-session, low-bandwidth (only user messages and final answers).
- What breaks if merged: stale tool traces from finished tasks re-render into later turns, so the model re-tries old tool calls and reasons about dead sub-goals; the context bloats with intra-task noise; and the per-turn scratchpad reset becomes impossible without also erasing session memory.

Common issues:

- Treating them as "the same thing at different sizes".
- Citing only context-length growth and missing the stale-replay failure, which is the load-bearing one.
- Believing the conversation should store agent reasoning for later turns.

## Exercise 20.02 — History before current message

A correct answer should include:

- The rule: render `format_for_prompt()` over *prior* turns first, then append the current user message separately, outside the history block.
- The symptom when wrong: the current question appears twice in the contextualized message — once inside "Previous conversation:" and once as the live question (the notebook asserts its count is exactly 1). The model then cannot distinguish prior context from the question being asked, and may treat it as already answered or answer the stale copy.

Common issues:

- "Order matters" with no concrete duplication symptom.
- Confusing this ordering with where the retrieval context block goes.
- Not connecting the answer to the pinned assertion in the multi-turn cell.

## Exercise 20.03 — Prefix RAG vs tool RAG

A correct answer should include:

- The tradeoff: prefix RAG is predictable (retrieval always fires, exactly once), cheaper in agent steps, and independent of model judgment — but it retrieves on turns that don't need it (noise injection) and cannot re-query or skip. Tool RAG lets the model decide when and what to search, at the cost of extra agent steps and the failure mode of forgetting to search.
- When the model should decide: when turns vary in whether the corpus is relevant, when queries need iterative or reformulated search, or when injected irrelevant context measurably hurts answers.

Common issues:

- Framing one style as strictly better instead of a tradeoff.
- Missing "the model can forget to retrieve" as tool RAG's characteristic failure.
- Ignoring the noise cost of always-on prefix retrieval on irrelevant turns.

## Exercise 20.04 — Integration evals

A correct answer should include:

- Module 20 changes wiring, not the model: prompts, tools, retriever, config. The eval gate must detect regressions in what can actually change — and answer text alone can look fine while the path regressed (e.g., the model answers the arithmetic from its weights without calling the calculator: the substring passes, but `expected_tool` catches the routing regression).
- Base-model factual recall tests the frozen backend, which the student's changes never touch — it measures the wrong layer.

Common issues:

- "Integration tests are more thorough" hand-waving without the right-answer-wrong-path example.
- Treating the gate as a model-quality benchmark rather than a fast regression check for config/prompt/tool changes.
- Missing that `rag=`/`expected_tool` assertions pin *which layer fired*, which is the whole point.

## Exercise 20.05 — Where the from-scratch model stops

A correct answer should include:

- The boundary: the course-trained artifact fails at instruction following and tool/format protocols — it emits corpus-style prose, never a valid structured tool call or ReAct step, so agent turns end with no action and no final answer. The failure is format compliance and instruction-following before it is knowledge.
- One concrete task with the contrast, e.g. the calculator task: ProdLM emits a well-formed calculator call and returns the exact result; the from-scratch model can neither produce a parseable call nor do the arithmetic.

Common issues:

- Framing the gap as missing knowledge only — the deeper failure is that the toy model was never post-trained on tool-calling or ReAct formats.
- No concrete task cited.
- Expecting the SFT'd toy model to tool-call — its SFT data contained no tool-call examples.

## Exercise 20.06 — Channel comparison at the assistant layer

A correct answer should include:

- Which channel produced the cleaner turn on the calculator task (expected: native) with the failure mode named on the worse channel — for ReAct, typically a bad parse, JSON mid-prose, or a runaway/no-progress turn; native failures are rarer and look like skipped tool use or empty content.
- Consistency with Module 19's finding: the model was post-trained for the structured tool-calling format; ReAct markers are mostly pretraining-era, so the small model wobbles there. Only the wire format changed — `Assistant.chat` and the turn semantics are identical.

Common issues:

- No named failure mode from the worse channel.
- Claiming the assistant behaves differently at the surface — `use_native` changes bytes on the wire, not the chat contract.
- A verdict without having run (or cited) both channels.

## Exercise 20.07 — Localize one failure

A correct answer should include:

- One live failure localized to a layer, with the `AssistantTurn` evidence trail that pointed there: `retrieved_context` (wrong/empty chunks → retrieval), `contextualized_message` (missing or duplicated history → conversation), `agent_run.steps` (unknown tool name → tool selection; validation errors → argument formatting; `duplicate_action`/`max_steps` stop → loop control; no parseable action at all → backend capability).
- Attribution to the *earliest* failing layer — downstream layers behaving reasonably on bad upstream input are not the fault.

Common issues:

- Blaming the agent loop or model for an upstream failure (e.g., generation faithfully answering from wrongly retrieved chunks).
- Evidence that does not support the named layer.
- Guessing a layer without walking the `AssistantTurn` fields in order.

## Exercise 20.08 — Strongest and weakest layer

A correct answer should include:

- A strongest layer (typically the ProdLM backend for raw answer quality, or tools for exactness) and a weakest link, each backed by evidence from the eval gate or live turns — common weakest links: hash-based retrieval, ReAct parsing on a small model, or loop control.
- A first improvement that targets the named weakest link and is proportionate: semantic embedder for retrieval, better tool descriptions, native channel, more eval cases — not a rebuild.

Common issues:

- Verdicts with no cited runs or eval results.
- An improvement that does not address the layer named weakest.
- "Improve the model" when the observed failures were harness-level (retrieval, prompts, parsing, routing).
