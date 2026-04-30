# Module 19 — Agent loops

> **Question this module answers:** *How does the model pursue multi-step goals?*

![Module 19 on one page: a four-panel circus map of the ReAct agent. PANEL 1 (top-left, "PLAN"): the user's task ("Read foo.txt and tell me the average") enters a planning booth. The backend emits "Goal: compute average of numbers in foo.txt\n1. read the file\n2. parse numbers\n3. average them" — the planner extracts a Plan(goal=..., steps=[...]). PANEL 2 (top-right, "PROMPT TIME"): the system prompt + plan block + user question are assembled. A trailing "Thought:" marker nudges the model into the ReAct format. PANEL 3 (bottom-left, "OBSERVE → THINK → ACT"): the model emits "Thought: I should read the file\nAction: read_file\nAction Input: {\"path\": \"foo.txt\"}". `parse_react_step` extracts the structured fields. The dispatcher routes through Module 18's `dispatch_tool_call`. The Observation is appended to the scratchpad. The next prompt regrows with the new Thought / Action / Observation block, and the loop continues. PANEL 4 (bottom-right, "STOP"): the model eventually emits "Thought: I have the answer\nFinal Answer: 42". The loop returns an AgentRunResult with `final_answer`, `steps`, `stopped_reason="final_answer"`, and the original `plan`. A right-edge sidebar lists key concepts: ReAct format (Thought/Action/Observation/Final Answer), the scratchpad as growing memory, planning as soft prior, error feedback as `[error]` observations, max_steps + duplicate-action loop detection as safety nets. Bottom caption: "Module 18 gave the model HANDS (tools); Module 19 gives it INTENT (planning + memory + recovery)."](19-agent/Module19-Hero.png)

*The whole module on one page. The agent loop is a thin wrapper around Module 18's tool dispatch: it adds explicit reasoning (Thought lines), persistent memory (the scratchpad), an optional plan, smarter stop conditions (duplicate-action detection, halt-on-stuck), and graceful recovery from tool errors. The model still does all the cognitive work; the agent loop just keeps it on rails.*

## Prerequisites

Module 19 is the fourth leg of Phase V (assistant systems). Module 18 built the tool-call substrate — registry, schema validation, parse → validate → dispatch contract. Module 19 wraps that substrate in a ReAct-style observe / think / act loop so the model can pursue tasks that require *several* tool calls in sequence, with recovery when one of them goes wrong.

This module is short on math and long on protocol. The whole content is:

- A small dataclass set (`Action`, `Observation`, `AgentStep`, `AgentRunResult`, `Plan`) describing one ReAct turn and one full run.
- A regex parser that pulls `Thought:` / `Action:` / `Action Input:` / `Final Answer:` out of free-form model text.
- A `Scratchpad` that accumulates per-step records and renders them back into the next prompt.
- An optional planning phase that asks the backend for a numbered plan up front.
- An `Agent` class whose `.run(user_message)` ties everything together: planning, looping, dispatching, error recovery, stop-condition checking.

There are four scaffolded methods. Three are short (each ~10–30 lines); the fourth is the loop (~70 lines including all the branches). The lesson is in the *contract* between them — what each piece is responsible for, and how a robust agent is built by composing small, well-behaved pieces.

### Math

There isn't really any math in this module. The closest things are:

- **The implicit Markov-ish property of the scratchpad.** Each turn's prompt is a function of the last turn's output and the previous turn's prompt: `prompt_{n+1} = render(history[:n+1])`. The model treats this as a state-transition problem, where its job is to pick the best next action given the current state. This is one of the conceptual reasons ReAct works: the model is good at "given history, pick action," and ReAct's prompt structure exactly matches that.

- **The "dispatch loop is a fixed point search."** When the model emits Final Answer, we've reached a fixed point — applying the loop again would just emit the same Final Answer (or restart). The whole loop is a search for that fixed point. Convergence is not guaranteed; `max_steps` is the safety net.

### Computer science

- **ReAct as a control-flow pattern.** Yao et al. 2022 showed that interleaving "Thought" and "Action" turns improves both reasoning and tool selection over either alone. The intuition: forcing the model to write a Thought line forces it to commit to *why* it's calling a tool before it picks the tool. This compresses the "what should I do" question into a fixed slot; without it, models conflate reasoning and action and pick worse tools. The empirical result is on HotpotQA, FEVER, ALFWorld, and WebShop — small models with ReAct beat larger models without.

- **The scratchpad is a working memory.** Every step appends a (thought, action, observation) record. Before the next call, all records are rendered back into the prompt. This is *short-term* memory in the cognitive-science sense — full history visible during the current task, dropped at task end. Long-term memory (across conversations) would be a separate system; the capstone in Module 20 builds that.

- **Planning vs reactive control.** A reactive agent picks the next action based on the last observation only. A planning agent commits to a sequence of subgoals up front and references them as it goes. Pure reactive often loses the thread on long tasks ("what was I doing again?"); pure planning struggles when the world doesn't match the plan. The Module 19 default is *planning followed by reactive* — produce a plan once, then run the ReAct loop with the plan visible but not enforced. This matches the Plan-and-Execute pattern from LangChain and the "Plan-and-Solve" prompting paper (Wang et al. 2023).

- **Stop conditions are policy, not mechanism.** The Module 18 loop had one stop condition: "no more `<tool_call>` blocks." Module 19 adds three more:
    - `final_answer` — model emitted Final Answer (clean exit, the canonical signal)
    - `duplicate_action` — same action with same args two steps in a row (loop detection)
    - `no_progress` — model emitted neither action nor final answer (stuck step) AND `halt_on_stuck=True`
  Plus the `max_steps` safety net inherited from Module 18. The agent's robustness depends on having *several* exit paths that handle different failure modes.

- **Error recovery as a conversation pattern.** The agent loop NEVER raises on model wobble. Bad parse → record as `parse_error` step, render to scratchpad as `Observation: [parse error] ...`. Unknown tool → `is_error=True` Observation. Tool runtime exception → also `is_error=True`. The model reads the next prompt, sees its own bad action followed by `[error] ...`, and decides what to do differently. Errors aren't bugs; they're conversation, exactly as in Module 18 — but now compounded across multiple turns.

- **The `[error]` prefix matters.** Without an explicit error marker, the model often parrots the error string back as if it were a successful answer ("The calculator returned: missing required arguments"). With `[error] ...`, instruction-tuned models reliably treat it as a recovery signal and try a different approach.

- **Duplicate-action loop detection is a heuristic, not a proof.** "Same tool, same args, two steps in a row" catches the most common loop pattern (model didn't understand the observation, retries identical call). It misses subtler loops (different args but no progress) and false-positives on legitimate retries (paginated reads, idempotent checks). The flag `loop_detection=True/False` lets you opt out per-task.

- **The `\s*` newline-gobbling regex bug.** A common subtle bug: a regex like `r"Action\s*:\s*([^\n]+)"` looks safe — `[^\n]+` won't cross newlines, right? But `\s*` matches *all* whitespace including `\n`, so given input `"Action:\nAction Input: ..."` the regex consumes the newline and captures the next line's content as the action name. The fix: post-colon whitespace must be `[ \t]*` (horizontal only), not `\s*`. This is one of the most common ways agent parsers silently produce garbage; the parser tests pin it.

### Programming

- **`re` for the ReAct parser.** Per-marker regexes (one for `Thought:`, one for `Action:`, one for `Action Input:`, one for `Final Answer:`). Each works independently; the parser combines their results. `re.IGNORECASE` for case-insensitive markers (some models lowercase). `re.DOTALL` on the body captures so JSON can span multiple lines.
- **`json.loads` with `try/except`.** The Action Input is JSON; the parser tolerates malformed JSON by setting a `parse_error` instead of raising. Same pattern as Module 18.
- **`json.dumps(..., sort_keys=True)` for action keys.** Loop detection compares `(tool_name, json.dumps(args, sort_keys=True))` across steps; sorted keys make `{"a": 1, "b": 2}` and `{"b": 2, "a": 1}` compare equal.
- **`@dataclass(frozen=True)` for `Action`, `Observation`, `ParsedStep`** — values, not handles. `@dataclass` (mutable) for `AgentStep`, `Plan`, `AgentRunResult` — built up incrementally during a run.
- **Composition with Module 18.** The agent calls `dispatch_tool_call(registry, ToolCall(...))` from Module 18 for tool execution. Module 19 doesn't reimplement validation, schema lookup, or error wrapping; it sits on top of Module 18's contracts.

### What you can skip

- **Multi-turn conversation memory.** Module 19's `Agent.run` is single-turn: one user message → one agent run → one final answer. Real assistants need conversation history that survives across runs. The capstone in Module 20 layers that on top.

- **Tree-of-thoughts and other branching strategies.** Yao et al. 2023 ("Tree of Thoughts") generalize ReAct to a search over multiple reasoning paths with backtracking. We do straight-line ReAct only; the branching mechanics are a separate module of work.

- **Reflection / self-critique loops.** Some agentic frameworks (Reflexion, Self-Refine) add a "review your last attempt" step. Useful when verification is cheaper than generation. Not built here; conceptually orthogonal to ReAct's loop.

- **Streaming output during a step.** A real production agent streams the model's tokens as they're generated and starts dispatching as soon as a complete `Action Input:` block is seen. We do synchronous calls. Same conceptual shape, ~3× more code.

- **Async / parallel agents.** A "swarm" of agents working in parallel, coordinated by a supervisor, is its own design space (Park et al.'s Generative Agents, Anthropic's orchestrator-worker pattern). We do one agent at a time.

- **Token-aware context management.** When the scratchpad grows past the model's context window, we truncate by character count. Production agents would token-count, summarize old steps, or use a vector store for long-term memory. Out of scope.

- **Production sandboxing.** All Module 18 caveats apply — the `run_python` tool runs in an unsandboxed subprocess. Fine for local pedagogy, NOT fine for a hosted agent.

## Why we start here

Module 18 fixed the "model needs to call a function" problem with one tool call per turn. But many real tasks need a *sequence*:

```
   ┌──────────────────────────────────────────────────────────────────────┐
   │  TASKS THAT NEED MORE THAN ONE TOOL CALL                             │
   ├──────────────────────────────────────────────────────────────────────┤
   │                                                                       │
   │   • "Summarize the longest file in this directory":                   │
   │       1. list files                                                   │
   │       2. read each, find the longest                                  │
   │       3. read that one fully                                          │
   │       4. write the summary                                            │
   │                                                                       │
   │   • "What's the average of the numbers in foo.txt?":                  │
   │       1. read foo.txt                                                 │
   │       2. parse numbers                                                │
   │       3. compute average                                              │
   │                                                                       │
   │   • "Find a Python error in this snippet":                            │
   │       1. read the file                                                │
   │       2. run it                                                       │
   │       3. read the error                                               │
   │       4. propose a fix                                                │
   │                                                                       │
   │   The Module 18 loop technically supports multi-call sequences, but   │
   │   it has no protocol for the model to *reason* between calls. The    │
   │   model has to decide its next action with no place to think out     │
   │   loud first. ReAct fixes this with explicit Thought lines.          │
   │                                                                       │
   └──────────────────────────────────────────────────────────────────────┘
```

The headline finding from the ReAct paper is empirical: small models (PaLM-540B in 2022; the equivalent today is a 7-13B Llama / Qwen) do measurably better on multi-step tool-use tasks when the prompt forces an explicit `Thought:` line before each `Action:`. The Thought line gives the model a place to commit to *why* it's about to call a tool; without it, models conflate reasoning and action and the tool selection gets noisy.

```
   ┌──────────────────────────────────────────────────────────────────────┐
   │  THE LOOP — REACT VERSION                                             │
   └──────────────────────────────────────────────────────────────────────┘

      plan = make_plan(backend, user_message, registry)   # optional
      transcript = system + plan_block + user_msg + "\nThought:"
      steps = []

      for step in range(max_steps):
          completion = backend.complete(transcript)
          parsed = parse_react_step(completion)

          if parsed.final_answer is not None:
              return AgentRunResult(final_answer=...)        # ← exit clean

          if parsed.action is not None:
              if duplicate(parsed.action, last_action):
                  return AgentRunResult(stopped="duplicate_action")
              obs = dispatch(parsed.action)
              steps.append((parsed.thought, parsed.action, obs))
              transcript = render(system, plan, user_msg, steps)
              continue

          # Stuck — no action, no final answer.
          if halt_on_stuck:
              return AgentRunResult(stopped="no_progress")
          steps.append((thought, parse_error_obs))
          transcript = render(...)

      return AgentRunResult(stopped="max_steps")           # ← safety net
```

Compared to Module 18's loop, the new pieces are:

  1. **The optional planning phase** (one extra backend call before the main loop).
  2. **The explicit Thought line** (the model writes its reasoning before each action).
  3. **The scratchpad** (every step's record is rendered into every subsequent prompt).
  4. **Loop detection** (stop on duplicate actions).
  5. **Stuck-step handling** (parse failure becomes an error observation, not a crash).

Each is small. The whole loop is about 70 lines. The lesson is the contract between the pieces — the parser is forgiving in the right places and strict in the right places, the scratchpad renders consistently so the model knows what it's seeing, the dispatch never raises, the stop conditions are layered.

## The big idea

### A turn is a (thought, action, observation) triple

```
   ┌──────────────────────────────────────────────────────────────────────┐
   │   ONE REACT TURN                                                      │
   ├──────────────────────────────────────────────────────────────────────┤
   │                                                                       │
   │     model emits:                                                     │
   │         Thought: I need to add 21 and 21 to get the answer.          │
   │         Action: calculator                                           │
   │         Action Input: {"expression": "21 + 21"}                      │
   │                                                                       │
   │     parser extracts:                                                 │
   │         thought  = "I need to add 21 and 21 to get the answer."      │
   │         action   = Action(tool="calculator",                         │
   │                           arguments={"expression": "21 + 21"})       │
   │                                                                       │
   │     dispatcher (Module 18) executes:                                 │
   │         result = ToolResult(output="42", is_error=False)             │
   │                                                                       │
   │     agent records:                                                   │
   │         observation = Observation(output="42", is_error=False)       │
   │         AgentStep(thought, action, observation, ...)                 │
   │                                                                       │
   │     scratchpad renders for next turn:                                │
   │         Thought: I need to add 21 and 21 to get the answer.          │
   │         Action: calculator                                           │
   │         Action Input: {"expression": "21 + 21"}                      │
   │         Observation: 42                                              │
   │                                                                       │
   └──────────────────────────────────────────────────────────────────────┘
```

The model sees its own past Thought / Action / Action Input on every subsequent turn — it gets to *read its own reasoning history* and decide what to do next. This is the scratchpad's whole job: turn the model's stateless `complete(prompt)` call into something that feels like working memory.

### The scratchpad is what makes ReAct ReAct

```
   ┌──────────────────────────────────────────────────────────────────────┐
   │   PROMPT GROWTH — TURN BY TURN                                        │
   ├──────────────────────────────────────────────────────────────────────┤
   │                                                                       │
   │   Turn 1 prompt:                                                     │
   │       <system>                                                       │
   │       <plan>                                                          │
   │       Question: <user msg>                                           │
   │       Thought:                  ← model continues from here          │
   │                                                                       │
   │   Turn 2 prompt (turn 1's [thought, action, obs] now visible):       │
   │       <system>                                                       │
   │       <plan>                                                          │
   │       Question: <user msg>                                           │
   │                                                                       │
   │       Thought: <turn 1 thought>                                      │
   │       Action: <turn 1 action.tool>                                   │
   │       Action Input: <turn 1 action.arguments>                        │
   │       Observation: <turn 1 observation>                              │
   │                                                                       │
   │       Thought:                  ← model continues from here          │
   │                                                                       │
   │   Turn 3 prompt (turn 1 + turn 2 visible):                           │
   │       ...                                                             │
   │       <turn 1 block>                                                 │
   │                                                                       │
   │       <turn 2 block>                                                 │
   │                                                                       │
   │       Thought:                  ← model continues from here          │
   │                                                                       │
   └──────────────────────────────────────────────────────────────────────┘
```

The trailing `Thought:` is the agent's nudge — it tells the model "your turn, in the ReAct format." Without it, instruction-tuned models often start the next turn with prose ("I think we should..."), which the parser then has to recover from. With it, the model continues the pattern.

### The four-marker contract

ReAct's wire format has four named markers:

```
Thought:        ← reasoning text, free-form
Action:         ← tool name (must match registry)
Action Input:   ← JSON object, the tool's arguments
Observation:    ← tool result, INJECTED BY RUNTIME (not by model)
Final Answer:   ← user-facing answer, ends the loop
```

The model emits `Thought` / `Action` / `Action Input` (and stops); the runtime appends `Observation:` and asks for the next turn. When the model has enough info, it emits `Thought` / `Final Answer:` and the loop exits.

Why these markers and not, say, `<thought>...</thought>` XML?

- **It's the format the ReAct paper used.** Yao et al. 2022 named these specific markers and demonstrated empirically that they work. Instruction-tuning datasets later picked it up; LangChain made it a de-facto standard. Models pretrained on the open internet have seen this format extensively.
- **It's regex-friendly.** Each marker is a fixed string at line-start with a colon. The parser is a few `re.search` calls.
- **It separates reasoning from action.** Distinct markers mean the parser can extract structured action data (the tool call) without tripping on the free-form reasoning.

### Errors are observations, not crashes

```
   ┌──────────────────────────────────────────────────────────────────────┐
   │  WHAT HAPPENS WHEN A TOOL CALL GOES WRONG                            │
   ├──────────────────────────────────────────────────────────────────────┤
   │                                                                       │
   │   Model emits:                                                       │
   │       Action: nonexistent_tool                                       │
   │       Action Input: {"x": "y"}                                       │
   │                                                                       │
   │   Module 18's dispatch_tool_call returns:                            │
   │       ToolResult(output="no tool named 'nonexistent_tool'; ...",     │
   │                  is_error=True)                                      │
   │                                                                       │
   │   The agent wraps:                                                   │
   │       Observation(output="no tool named ...", is_error=True)         │
   │                                                                       │
   │   The scratchpad renders for the next turn:                          │
   │       Thought: <model's thought>                                     │
   │       Action: nonexistent_tool                                       │
   │       Action Input: {"x": "y"}                                       │
   │       Observation: [error] no tool named 'nonexistent_tool'; ...     │
   │                          ^^^^^^^                                     │
   │                          this prefix is the recovery signal         │
   │                                                                       │
   │   Model sees its own bad action followed by [error] ..., decides    │
   │   what to do differently, emits the corrected action on the next    │
   │   turn. NO STACK TRACE. NO LOOP CRASH. NO LOST CONVERSATION.         │
   │                                                                       │
   └──────────────────────────────────────────────────────────────────────┘
```

The dispatcher (Module 18's `dispatch_tool_call`) already wraps every error class as `ToolResult(is_error=True)`. The agent module just propagates that to `Observation(is_error=True)` and renders it with an `[error]` prefix. The agent loop never raises on a model action; only misuse (empty `user_message`, wrong types) triggers an exception.

### Stop conditions, layered

```
   ┌──────────────────────────────────────────────────────────────────────┐
   │  STOP CONDITION TABLE                                                 │
   ├──────────────────────────────────────────────────────────────────────┤
   │                                                                       │
   │   stopped_reason   when it fires           what it means              │
   │   ──────────────   ─────────────────       ────────────────────────   │
   │   final_answer     model emitted Final     clean exit, model is done │
   │                    Answer                                             │
   │                                                                       │
   │   duplicate_       same action + same      model didn't understand   │
   │     action         args two steps in a     its last observation, is  │
   │                    row, with               looping. Stop.            │
   │                    loop_detection=True                                │
   │                                                                       │
   │   no_progress      neither action nor      model emitted prose with  │
   │                    final answer, with      no structured content.    │
   │                    halt_on_stuck=True      Halt rather than retry.   │
   │                                                                       │
   │   max_steps        loop ran out of         the safety net. Returns   │
   │                    iterations              final_answer=None — the   │
   │                                            model never decided to    │
   │                                            stop. Tune max_steps for  │
   │                                            the task.                 │
   │                                                                       │
   └──────────────────────────────────────────────────────────────────────┘
```

The default Agent has `loop_detection=True`, `halt_on_stuck=False`, `max_steps=8`. That combination handles:
- Clean tasks: model emits Final Answer in 1-3 steps. ✓
- Tasks with a tool error: model recovers (the `[error]` observation feeds back). ✓
- Tasks where the model loops: `loop_detection` cuts it off. ✓
- Tasks where the model goes off-format: parse-error observation feeds back; it usually recovers. ✓
- Truly bad situations: `max_steps` cuts off after 8 turns. ✓

Tuning these per-task is a real consideration — `loop_detection=False` for legitimate-retry workflows; `halt_on_stuck=True` when the model has only one shot at format compliance; smaller / larger `max_steps` for one-shot vs. long-task agents.

### The unified agent interface

```
   ┌─────────────────────────────────────────────────────────────────────┐
   │   g2c/agent/  PUBLIC API                                              │
   ├─────────────────────────────────────────────────────────────────────┤
   │                                                                       │
   │   Action(tool, arguments)                                             │
   │     a parsed tool invocation                                          │
   │                                                                       │
   │   Observation(output, is_error)                                       │
   │     the result of one action                                          │
   │                                                                       │
   │   Plan(goal, steps)                                                   │
   │     an optional plan from the planning phase                          │
   │                                                                       │
   │   AgentStep(completion, thought, action, observation,                 │
   │             final_answer, parse_error, inference)                     │
   │     one iteration's record                                            │
   │                                                                       │
   │   AgentRunResult(user_message, plan, final_answer, steps,             │
   │                  stopped_reason, metadata)                            │
   │     the full run's record                                             │
   │                                                                       │
   │   AgentError                                                          │
   │     internal exception type                                           │
   │                                                                       │
   │   ParsedStep(thought, action, final_answer, parse_error)              │
   │     parse_react_step's output                                         │
   │                                                                       │
   │   parse_react_step(text) → ParsedStep                                 │
   │     extract Thought/Action/Action Input/Final Answer                  │
   │                                                                       │
   │   Scratchpad(*, max_chars=None)                                       │
   │     .append(step), .render(), .steps, len()                           │
   │                                                                       │
   │   extract_plan(text, user_message) → Plan | None                      │
   │     parse a numbered plan from a planning completion                  │
   │                                                                       │
   │   make_plan(backend, user_message, registry, **kw) → Plan | None      │
   │     run the planning phase end-to-end                                 │
   │                                                                       │
   │   render_system_prompt(tools) → str                                   │
   │   render_planning_prompt(user_message, tools) → str                   │
   │   render_plan_block(goal, steps) → str                                │
   │     prompt-template renderers                                         │
   │                                                                       │
   │   Agent(backend, registry, *, max_steps=8, plan=True,                 │
   │         loop_detection=True, halt_on_stuck=False, ...)                │
   │     .run(user_message) → AgentRunResult                               │
   │                                                                       │
   └─────────────────────────────────────────────────────────────────────┘
```

Total scaffolded code: roughly 100 lines spread across four method bodies. Everything else — the dataclasses, the registry composition, the prompt templates, the planner orchestration, the dispatch wiring — is pre-implemented because the wiring isn't the lesson; the protocol is.

## Concepts to internalize

- **A turn is a triple, not an utterance.** The model's per-turn output isn't free-form text; it's structurally `(thought, action OR final_answer)`. The parser's job is to extract that structure; the loop's job is to feed it back. Treating the model as a structured-output device is what makes everything work.

- **The scratchpad turns stateless inference into state.** Each backend.complete is a memoryless function call. The scratchpad makes the model *appear* to remember by re-rendering history into every subsequent prompt. The model isn't actually remembering; the prompt is.

- **Errors are conversation, compounded.** Module 18's "errors as data" extends to multi-step: the agent's loop survives every model wobble (bad parse, unknown tool, bad args, runtime exception) and feeds it back as the next turn's observation. After 5-7 attempts, even confused models often find their way to a working answer — exactly because the errors are conversational, not crashes.

- **Planning is a soft prior, not a hard contract.** The Plan is rendered into the prompt and helps the model stay on track, but the model is free to deviate. The plan is most useful for tasks with obvious structure ("read X, transform Y, write Z"); least useful for tasks where the model just needs to call one tool.

- **Loop detection is a heuristic, but a useful one.** Same tool + same args two steps in a row catches the most common loop pattern (model didn't understand the observation, retries identical call). It's not a proof of looping; the flag exists so you can opt out for legitimate-retry tasks.

- **The trailing `Thought:` nudge is load-bearing.** Without it, instruction-tuned models often start the next turn with prose ("I think we should..."), which the parser tolerates but which costs tokens and accuracy. The `_build_prompt` helper appends `\n\nThought:` after the scratchpad; this is what keeps the model in format.

- **Module 18's `dispatch_tool_call` is the agent's tool-execution layer.** The agent doesn't reimplement validation, schema lookup, or error wrapping; it composes with what Module 18 already provides. The agent module is *only* the loop logic + scratchpad + parser + planner.

- **`max_steps` exists because models can lose the thread.** The combination of "nothing in the prompt actually requires the model to stop" and "context length is finite" means you need a hard cap. Module 18 had `max_steps=5`; Module 19 defaults to `8` because a planned task often needs the planning step + 3-5 tool calls + the final answer. Tune per-task.

## Scaffolding and how to run the tests

This module ships seven files in `g2c/agent/`:

- **`base.py`** — `Action`, `Observation`, `AgentStep`, `Plan`, `AgentRunResult`, `AgentError` dataclasses. All boilerplate.
- **`prompts.py`** — `DEFAULT_AGENT_SYSTEM`, `DEFAULT_PLANNING_PROMPT`, `render_system_prompt`, `render_planning_prompt`, `render_plan_block`. All implemented (templates).
- **`parser.py`** — `parse_react_step` (**scaffolded**) + `ParsedStep` dataclass + helper regexes.
- **`memory.py`** — `Scratchpad` class with `append` (implemented) + `render` (**scaffolded**).
- **`planner.py`** — `extract_plan` (**scaffolded**) + `make_plan` (implemented; composes extract_plan with backend.complete).
- **`agent.py`** — `Agent` class with constructor + `_build_prompt` (implemented) + `run` (**scaffolded**).
- **`__init__.py`** — public exports.

Tests live in `tests/test_agent.py`. Initial state on `main`: 45 tests pass (boilerplate + the components that are fully implemented). 76 tests fail with `NotImplementedError` (or transitively, where they call into a scaffold) until you fill in the four scaffolded methods.

```bash
pytest tests/test_agent.py                          # all module-19 tests
pytest tests/test_agent.py -x                       # stop at first failure
pytest tests/test_agent.py -k Parse                 # parser tests
pytest tests/test_agent.py -k Plan                  # planner tests
pytest tests/test_agent.py -k Scratchpad            # scratchpad tests
pytest tests/test_agent.py -k AgentRun              # main loop tests
pytest tests/test_agent.py -k Integration           # full-pipeline smoke
pytest tests/test_agent.py -v                       # verbose
```

Implementation order — four independent steps:

  1. **`parse_react_step`** in `g2c/agent/parser.py`. Extract Thought / Action / Action Input / Final Answer from a completion. Pure logic, no external deps. Easiest place to start. Turns green: `TestParseReactStep`, `TestParseReactStepEdgeCases`. Also unblocks anything that goes through `Agent.run` (since the loop calls the parser).

  2. **`extract_plan`** in `g2c/agent/planner.py`. Parse a numbered plan from a planning-prompt completion. Independent of the loop. Turns green: `TestExtractPlan`, `TestExtractPlanEdgeCases`, `TestMakePlan`.

  3. **`Scratchpad.render`** in `g2c/agent/memory.py`. Format past steps into the next prompt's history block. Turns green: `TestScratchpadRender`, `TestScratchpadTruncation`.

  4. **`Agent.run`** in `g2c/agent/agent.py`. The orchestration loop. Once 1-3 are done, this turns green and the integration smoke tests pass: `TestAgentRun`, `TestAgentRunStopConditions`, `TestAgentRunErrorRecovery`, `TestAgentRunPlanning`, `TestIntegrationSmoke`.

The four are independent. Suggested order is "parser → planner → scratchpad → loop" because that's roughly easy → hard, but you can work in any order.

The integration tests (`TestIntegrationSmoke`) also depend on Module 18's `validate_arguments` and `calculator_evaluate` being implemented — they exercise the calculator tool through the agent. If those are still scaffolded from Module 18, the integration tests will fail transitively until you finish 18 first.

Headline tests to watch:

- **`test_thought_does_not_eat_next_marker`** — pins the regex newline-gobbling bug. Without proper post-colon whitespace handling, the Thought regex can swallow the `\n` and capture the next line's content as part of the thought.

- **`test_final_answer_wins_over_action`** — pins the parser priority rule. When the model emits both Action and Final Answer, the Final Answer is authoritative. Otherwise "model said it's done but the loop kept going" bugs.

- **`test_loop_detection_stops_on_duplicate_action`** — pins the loop-detection contract. Same tool + same args two steps in a row → `stopped_reason="duplicate_action"`. If you build loop detection differently (e.g., last 3 instead of last 1), this test will fail and tell you so.

- **`test_loop_detection_distinguishes_different_args`** — pins the *negative* case. Same tool but DIFFERENT args is NOT a duplicate; the loop continues.

- **`test_unknown_tool_surfaces_as_observation_error`** — pins the dispatch composition with Module 18. The agent must compose with `dispatch_tool_call`, not raise on unknown tools.

- **`test_loop_continues_after_tool_error`** — pins the recovery contract. After a tool error, the loop continues; the model can read the `[error]` observation and try again.

- **`test_scratchpad_grows_into_prompt`** — pins the feedback wiring. Step 1's observation must appear in step 2's prompt.

- **`test_prompt_ends_with_thought_marker`** — pins the trailing-`Thought:` nudge. Without it, models drift into prose.

- **`test_arguments_rendered_as_json_not_python_repr`** — pins the JSON serialization. `repr(args)` would give Python-dict syntax (single quotes); `json.dumps(args)` gives the format the model originally emitted. Consistency matters.

- **`test_plan_unparseable_falls_through_gracefully`** — pins the "graceful planner failure" contract. A bad plan completion doesn't crash the run; the loop runs without one.

## What you'll build

Package: `g2c/agent/`

```python
# base.py
@dataclass(frozen=True)
class Action:                                                     # implemented
    tool: str
    arguments: dict[str, Any]

@dataclass(frozen=True)
class Observation:                                                # implemented
    output: str
    is_error: bool = False

class AgentError(Exception): ...                                  # implemented

@dataclass
class Plan:                                                       # implemented
    goal: str
    steps: list[str] = field(default_factory=list)

@dataclass
class AgentStep:                                                  # implemented
    completion: str
    thought: str
    action: Action | None
    observation: Observation | None
    final_answer: str | None
    parse_error: str | None
    inference: InferenceResult

@dataclass
class AgentRunResult:                                             # implemented
    user_message: str
    plan: Plan | None
    final_answer: str | None
    steps: list[AgentStep]
    stopped_reason: str
    metadata: dict[str, Any] = field(default_factory=dict)


# parser.py
@dataclass(frozen=True)
class ParsedStep:                                                 # implemented
    thought: str
    action: Action | None
    final_answer: str | None
    parse_error: str | None

def parse_react_step(text) -> ParsedStep:                         # SCAFFOLDED
    ...


# memory.py
class Scratchpad:                                                 # implemented
    def __init__(self, *, max_chars=None): ...
    def append(self, step) -> None: ...
    @property
    def steps(self) -> list[AgentStep]: ...
    def render(self) -> str:                                      # SCAFFOLDED
        ...


# planner.py
def extract_plan(text, user_message) -> Plan | None:              # SCAFFOLDED
    ...

def make_plan(backend, user_message, registry, **kw) -> Plan | None: # implemented
    ...


# prompts.py
DEFAULT_AGENT_SYSTEM: str                                         # implemented
DEFAULT_PLANNING_PROMPT: str                                      # implemented

def render_system_prompt(tools) -> str: ...                       # implemented
def render_planning_prompt(user_message, tools) -> str: ...       # implemented
def render_plan_block(goal, steps) -> str: ...                    # implemented


# agent.py
class Agent:
    def __init__(self, backend, registry, *,                      # implemented
                 max_steps=8, plan=True, loop_detection=True,
                 halt_on_stuck=False, scratchpad_max_chars=None,
                 max_new_tokens=512, temperature=0.2,
                 top_k=None, top_p=None): ...

    def _build_prompt(self, user_message, plan, scratchpad) -> str: # implemented
        ...

    def run(self, user_message) -> AgentRunResult:                # SCAFFOLDED
        ...
```

Total scaffolded code: roughly 100 lines across four function bodies. The lesson is the contracts (parsing, scratchpad rendering, plan extraction, loop control); the orchestration is layout.

## Exercises

These exercises require Ollama running with a tool-calling-capable chat model:

```bash
ollama pull llama3.2:3b           # tool-calling enabled, fast on M1+
# or
ollama pull qwen2.5:7b            # also good; better on multi-step
ollama serve
```

Llama 3.2:3b is the fastest reasonable choice; Qwen 2.5:7b handles multi-step reasoning notably better at the cost of latency. Both follow ReAct format reliably given the prompt template.

1. **One-shot calculator.** Wire `OllamaBackend("llama3.2:3b")` + a registry containing only `make_calculator()` into an `Agent` with `plan=False`. Ask 10 single-step arithmetic questions. Most should resolve in 1 tool call + final answer (= 2 steps). Report:
    - Steps per question (mean, max).
    - Tool-call success rate.
    - Cases where the model emits Final Answer without calling the calculator (and whether the answer is right).

2. **Multi-step file task.** Wire `make_calculator()` + `make_read_file(root=Path("data/numbers/"))`. Drop a few text files into `data/numbers/`, each with a list of numbers. Ask: *"Read `nums.txt` and tell me the average."* The model should call `read_file`, then `calculator`, then answer. Run 5 such two-step tasks; report which ones the model gets right end-to-end. Compare with `plan=True` vs `plan=False` — does the plan help?

3. **Stress-test loop detection.** Construct a task where the model is likely to repeat the same call (e.g., a tool that returns ambiguous output). Run with `loop_detection=True` and `loop_detection=False`; compare. Record the cases where loop detection saves a runaway loop vs where it cuts off legitimate retries. Tabulate: when is loop detection a net positive?

4. **The plan contribution.** Take a 3-tool task ("read this file, run this Python on its contents, give me the result"). Run three configurations:
    - `plan=False` (pure ReAct)
    - `plan=True` with the default planner
    - `plan=True` with a hand-written plan injected (skip the planning phase, pass `Plan(...)` directly — you'll need to either modify `Agent.run` or call the loop pieces manually)
   
   Report success rates and step counts. Is the planner helping, hurting, or neutral?

5. **Recovery characterization.** Wire a tool that fails about 30% of the time (random `raise`). Run 20 tasks. Report:
    - Recovery rate after a tool error (does the model retry / try a different tool?).
    - Cases where the model gives up and emits Final Answer without solving the problem.
    - Cases where the model loops on the failing tool (and whether `loop_detection` caught them).

6. **Add a custom tool.** Write a tool specific to your work — query your local database, fetch from an internal API, run a domain lint. Register it in a `ToolRegistry` and exercise it through `Agent.run`. The point: you have arbitrary affordances now. What does it look like to use them under ReAct? Where does the model overcall vs undercall the tool?

7. **The scratchpad cap.** Run a 6-tool task with `scratchpad_max_chars=2000` and again with no cap. Does the truncation break the agent's reasoning? When? Where would you summarize old steps instead of dropping them?

8. **Build the deliverable CLI.** A small interactive loop:

   ```python
   while True:
       q = input("? ")
       if not q.strip():
           break
       result = agent.run(q)
       print(result.final_answer or f"(stuck — {result.stopped_reason})")
       print(f"({len(result.steps)} steps, "
             f"{result.metadata['n_tool_calls']} tool calls, "
             f"plan={'yes' if result.plan else 'no'})")
       print()
   ```
   
   Run a 10-question session covering: arithmetic, file reading, Python execution, ambiguous questions, questions with no good tool. Save the transcript.

9. **Failure-mode catalog.** From your CLI runs in Exercise 8, identify three distinct failure modes the agent exhibits. For each, write down:
    - **What happens.** A specific transcript.
    - **Why.** Your hypothesis about the cause.
    - **One mitigation.** A specific change to the prompt, the loop, the registry, or the stop conditions that would help. Don't actually implement it — articulating the mitigation precisely is the exercise.
   
   The catalog is the actual deliverable. The pipeline code is the substrate; the *characterization* is what you keep.

10. **Compare with Module 18.** Take one of your Exercise 8 tasks and run it through Module 18's `run_with_tools` (no Thought lines, no plan, no loop detection). Side-by-side: which steps does the simpler loop save? Which does it lose? Where does ReAct's overhead pay off, and where is it just tax?

## Pitfalls to expect

- **The `\s*` newline-gobbling regex bug in the parser.** A regex like `r"Action\s*:\s*([^\n]+)"` looks safe but `\s*` matches *all* whitespace including `\n`. Given input `"Action:\nAction Input: ..."` it consumes the newline and captures the next line's content as the action name. The fix: post-colon whitespace must be `[ \t]*` (horizontal only). The test `test_thought_does_not_eat_next_marker` pins this; if it fails or your action names look like "Action Input: {...}", this is the bug.

- **Final Answer not winning over Action.** When the model emits both, Final Answer must be authoritative (the model is signaling "I'm done"). If the parser returns the Action, the loop keeps going past where it should stop — burning tokens and confusing the model.

- **Empty Final Answer treated as a real answer.** The parser should treat `"Final Answer:"` followed by nothing (or whitespace) as a parse error, not a valid empty answer. Returning a blank `final_answer=""` to the user is rarely what you want.

- **`json.dumps(args)` vs `repr(args)` in the scratchpad.** Python repr uses single quotes (`{'key': 'value'}`); JSON uses double (`{"key": "value"}`). The model originally emitted JSON; rendering back as Python repr trains it to think the format changed mid-conversation. Always `json.dumps`.

- **Forgetting the `[error]` prefix on error observations.** Without it, the model often parrots the error string back as if it were a successful answer ("The calculator returned: missing required arguments"). With it, instruction-tuned models reliably treat the observation as a recovery signal.

- **Forgetting the trailing `Thought:` nudge after the scratchpad.** Without it, instruction-tuned models often start the next turn with prose ("I think we should..."), which the parser tolerates but which costs tokens and accuracy. The `_build_prompt` helper appends `\n\nThought:` after rendering everything else; if you remove it, format compliance drops.

- **Loop detection comparing args via `==` on dicts.** Two dicts with the same keys in different insertion order can compare equal in Python (since 3.7's preserved insertion order doesn't affect `==`), but the safer approach is `json.dumps(args, sort_keys=True)` — it's explicitly canonical and works even if you switch to `OrderedDict` somewhere.

- **Loop detection over too long a window.** Comparing the last action against ALL prior actions (instead of just the immediately preceding one) catches more loops but also catches legitimate retries (paginated reads, re-checks). The simple "two in a row" rule is a tradeoff; tighten it only if you know you need to.

- **Halt-on-stuck triggering on a single bad parse.** Models occasionally emit a single weird turn (a code fence around their reasoning, an extra `Observation:` they shouldn't have produced) and the next turn is fine. With `halt_on_stuck=True`, you cut them off after one strike. Default `False` is more forgiving; flip to `True` only when you have a good reason.

- **Plan rendered into the prompt but the model ignores it.** This is fine — the plan is a soft prior, not a hard contract. If the model deviates because the world doesn't match the plan, that's *good* behavior. If you want hard plan enforcement, you'd build a different system (state-machine agent, plan-verifier wrapper). Module 19 does soft planning by design.

- **`make_plan` raising and crashing the run.** The agent's `run` method wraps the planning call in `try/except` and falls through to `plan=None` on any exception. If you replace `make_plan` with a stricter version that raises, the whole run dies. The graceful-degradation contract is load-bearing.

- **Running on the from-scratch Module 10 model.** Your tiny ~10M-param model wasn't trained on tool-calling data. It will not follow ReAct format. Use Ollama or MLX with an instruction-tuned model. Module 16's caveat continues to apply: the from-scratch model is for the from-scratch phases.

- **`max_steps` calibrated wrong.** Default is 8. If the model needs to call several tools sequentially after a planning step, 8 might not be enough. If the model loops on bad calls, 8 might be too generous. Tune per-task; check `result.stopped_reason == "max_steps"` to know when you've hit the cap.

- **Forgetting Module 18's `validate_arguments` is still scaffolded.** The agent's tool dispatch goes through `dispatch_tool_call` → `validate_arguments`. If you build Module 19 without finishing Module 18's scaffold first, the integration tests will fail with `NotImplementedError` from inside the dispatcher. Finish 18 → test 18 → start 19.

- **`scratchpad_max_chars` set too low.** If the cap is smaller than a typical step's rendered size, the scratchpad ends up rendering only the very last step, and the model effectively forgets what it just tried. The cap should be at LEAST a few times the average step size; production agents usually summarize old steps instead of dropping them.

- **Confusing `Action.tool` (Module 19) with `ToolCall.name` (Module 18).** They mean the same thing but use different field names. The agent's parser produces `Action(tool=...)`; the dispatcher needs `ToolCall(name=...)`. The agent module's `run` does the conversion — if you build a custom dispatch path, watch for this.

## Reading

Primary:

- **Yao, Zhao, Yu et al., "ReAct: Synergizing Reasoning and Acting in Language Models" (ICLR 2023).** The paper this module is built around. Read §2 (the prompt format) and §3 (HotpotQA, ALFWorld results). The key empirical finding: explicit `Thought:` interleaving improves both reasoning and tool selection over either alone. Most of the prompt template choices in this module trace directly to this paper.

- **Anthropic, "Building effective agents" (Dec 2024).** A practical taxonomy of agentic patterns: prompt chains, routing, parallelization, orchestrator-workers, evaluator-optimizer, ReAct. Read it after you finish this module to see where ReAct sits in the broader landscape and which patterns are worth investing in next. The "augmented LLM = LLM + tools + memory + retrieval" framing in §1 is exactly what Modules 17-19 built.

- **Wang, Xu, Lan et al., "Plan-and-Solve Prompting" (ACL 2023).** The paper behind the planning phase. Empirically: a separate planning step, then a separate execution step, beats one-shot ReAct on multi-step reasoning tasks. Module 19's optional planning phase is the simplest possible version of this idea.

Secondary:

- **Park, O'Brien, Cai et al., "Generative Agents: Interactive Simulacra of Human Behavior" (UIST 2023).** A small town of LLM-driven agents with memory, planning, and reflection. The architecture diagram in §4 (memory stream → reflection → plan → action) is the next conceptual step beyond Module 19. Skim for the architecture overview; the simulation itself is interesting but not directly applicable.

- **Shinn, Cassano, Berman et al., "Reflexion: Language Agents with Verbal Reinforcement Learning" (NeurIPS 2023).** Adds a "review your last attempt" step where the agent self-critiques and tries again. Useful when verification is cheap (test pass/fail) and generation is expensive. Not built here, but the design space is right next to ReAct.

- **Liu, Li, Du et al., "AgentBench: Evaluating LLMs as Agents" (ICLR 2024).** A benchmark suite covering tool use, web browsing, OS interaction, and game-playing. The most-useful section is the per-task error analysis in §4 — patterns of failure (planning errors, tool selection errors, loops) that any agent builder hits. Worth reading as a "what should I be measuring" reference.

Optional:

- **Yao, Yu, Zhao et al., "Tree of Thoughts" (NeurIPS 2023).** Generalizes ReAct to a search over multiple reasoning paths with backtracking. Skim §3 — the BFS / DFS over thought trees is a real architectural step beyond straight-line ReAct, but at substantial cost (each thought node is a backend call).

- **Jimenez, Yang et al., "SWE-bench" (ICLR 2024).** Real-world software-engineering agent benchmark — actual bug fixes from real GitHub issues. Skim §3 for the task structure and §5 for the leaderboard. Useful as a "what does production agentic actually look like" reference; humbling about how far simple ReAct gets you (it doesn't).

- **Mialon, Dessì, Lomeli et al., "Augmented Language Models: a Survey" (TMLR 2023).** A survey of tool use, retrieval, reasoning chains, and agent loops as of late 2022. Good for situating ReAct in a broader context. Slightly outdated but still the best survey.

## Deliverable checklist

- [ ] All tests in `tests/test_agent.py` pass: 121 tests, all green.
- [ ] Ollama running with a tool-calling chat model. `ollama list` shows `llama3.2:3b` (or your chosen model).
- [ ] Notebook: `notebooks/19-agent.ipynb`. Wires `Agent` + a multi-tool registry, runs Exercises 1, 2, 3, 4 with output cells visible.
- [ ] **Failure-mode catalog** (Exercise 9) in `docs/agent-failure-modes.md`. Three failure modes, each with a transcript, hypothesis, and proposed mitigation. The actual deliverable.
- [ ] You can explain — out loud, without notes — why the parser must use `[ \t]*` instead of `\s*` after the markers' colons.
- [ ] You can explain — out loud, without notes — why Final Answer wins over Action when both appear in the same completion.
- [ ] You can explain — out loud, without notes — why the scratchpad renders the model's *own* past Thought lines back into the prompt (and not just observations).
- [ ] You can explain — out loud, without notes — what the four stop conditions are and when each fires.
- [ ] You can explain — out loud, without notes — why errors are propagated as `Observation(is_error=True)` rather than raised as exceptions, and what changes in the model's behavior because of the `[error]` prefix.

## M-series notes

This module is comfortable on every M-series Mac. Practical considerations:

- **Each step is one backend call.** A typical run is 1 (planning) + 3-6 (loop) = 4-7 backend calls. With Ollama + Llama 3.2 3B at ~50 tokens/sec on M1, each step is 1-3 seconds; full runs are 10-20 seconds. Qwen 2.5 7B is 2-3× slower per step but produces better reasoning on complex tasks; on M1/16GB it's borderline-comfortable, on M2+/32GB it's smooth.

- **Context length grows with step count.** A 5-step run with verbose tool results easily reaches 4-8k tokens of prompt. Llama 3.2's 128k context window is comfortable; smaller-context models would force aggressive scratchpad truncation. The default `scratchpad_max_chars=None` (unlimited) works for 8-step runs; for longer agents, set a cap and add a summarization pass.

- **Subprocess startup cost dominates `run_python`.** All Module 18 caveats apply — `subprocess.run([sys.executable, "-c", code])` pays ~50-200ms per call on M-series. If `run_python` is called frequently, the agent's wall time becomes dominated by subprocess startup. Module 18's pitfall section discusses the long-lived child-process variant; out of scope here, but worth knowing.

- **Planning latency.** The planning phase is one extra backend call (~1-3 seconds with Llama 3.2:3b). For one-shot tasks this is pure overhead; for multi-step tasks it pays off. Disable with `plan=False` when the task is obviously single-step.

- **Memory considerations are inherited from Module 16/18.** The agent loop itself is pure Python plumbing — microseconds per step. The model's inference is the memory-hungry part, and the requirements are the same as Modules 16, 17, 18.
