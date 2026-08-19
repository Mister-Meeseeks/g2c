# Beyond — Agent harness engineering

> **Question this module answers:** *How do you turn a capable tool-using model into a reliable, inspectable, resumable system?*

<!-- TODO(hero pipeline): asset not yet generated -->
![An agent loop wrapped in an event log, context policy, permissions, retry classification, and explicit stopping rules.](harness/BeyondHarness-Hero.png)

Module 19's loop works—and then the process dies mid-run, old tool output crowds the task out of context, or a side effect lands without its result reaching the transcript. This module rebuilds the loop around an append-only event log, a deterministic context view, explicit permissions, conservative crash recovery, and bounded retries. It ends with a controlled comparison: the same scripted backend and tools, under two harnesses and the same injected failures.

> **This is a Beyond module.** Beyond modules sit outside the numbered course: nothing in Modules 00–20 depends on them, and they are not part of finishing the course. Come here in any order, whenever a model card or paper names the idea and you want the load-bearing version built and broken on your own machine.

---
## Before you start

* *Review* [19-agent](../modules/19-agent.md) for the reasoning/action loop and [18-tools](../modules/18-tools.md) for its tool contract
* *Finish* `g2c/tools` and `g2c/agent`
* *Run* `G2C_APPLY_SOLUTIONS=01-19 ./notebook.sh harness` instead of the plain launch if you are entering without your own implementations

The required exercises use a deterministic scripted backend. No local model setup is required; ProdLM is an optional transfer check at the end.

---
## Where this fits in

The course's agent arc built tools that execute in Module 18, a loop that reasons and acts in Module 19, and an assistant that assembles the system in Module 20. Those short notebook runs keep state in a Python list and treat failure as an exception. Longer-running systems need an answer to different questions: What reached durable storage before dying? What should the model see next? What is a tool allowed to do? And when must the run stop?

The evaluated unit extends beyond just the model weights:

```
   model + prompt + tools + context policy + workspace
         + permissions + recovery + stopping rules
   ───────────────────────────────────────────────────
                         agent
```

In this module we hold the model itself fixed, and instead focus on the surrounding system. This is so you can see how much . We call that surrounding system the *agent harness*. 

## The big idea

One design decision organizes the module: the trajectory is an append-only event log, while context, resume state, and audit output are views of that log.**

```
   events.jsonl
   ─────────────────────────────────────────────────────────
   task
   model_turn
   tool_call(call_0)     ← intent recorded before execution
   tool_result(call_0)   ← observed outcome recorded afterward
   ...
          │
          ├── context view: what the next model call sees
          ├── resume view: what a fresh process can reconstruct
          └── audit view: what the harness recorded in order
```

Module 19's message list mixed these roles together. Separating “what happened” from “what the model sees” lets the view become lossy without deleting history and lets a new process reconstruct a run without re-executing completed effects.

This is a teaching log, not a database. Its recovery claims assume one writer, an ordinary process crash, and an intact filesystem. It does not fsync each event, recover torn records, coordinate concurrent writers, or survive filesystem loss.

### Context is a deterministic policy

If a context buffer grows until it overflows and then drops the oldest text, the task statement is the first important thing to disappear. An agent does not necessarily stop when that happens; it can keep acting on recent observations and drift.

`compact_context` is deliberately modest and testable:

```
   always preserve verbatim       compact or drop first
   ────────────────────────       ─────────────────────
   the task statement             old tool output
   the newest event               old narration and actions
```

It keeps the task, keeps recent events, shortens old tool output, then drops the oldest remaining middle events if needed. It does **not** claim to discover every constraint, commitment, or open item. Production systems usually represent those separately or use a summarizer with its own evaluation. Exercise 3 runs both policies through a live prompt-sensitive backend, so drift is observed as behavior rather than inferred from two rendered strings.

The trajectory is only one part of the prompt budget. `HarnessAgent` also counts the system prompt and rendered tool schemas and reserves space for the completion. Otherwise adding one verbose tool can overflow a context that the history policy declared safe. This module uses a cheap token estimate; a production harness uses the backend's tokenizer and model-window metadata.

### Rules are resolved instructions, not trajectory history

A coding harness usually assembles instructions from more than the task: system policy, repository rules such as `AGENTS.md`, narrower directory rules, current user constraints, and permissions may all apply. Discovery, scope, and precedence must be deterministic. Once resolved, this instruction layer stays separate from compactable history:

```
   retain as resolved instructions       compact as trajectory
   ───────────────────────────────       ─────────────────────
   system and harness policy             old narration
   applicable repository rules           old actions
   task and durable constraints           bulky tool observations
   current permissions
```

The event log should record the resolved rule paths and content hashes—or a safe snapshot—so resume can detect that its operating instructions changed. This module does not implement an `AGENTS.md` filesystem resolver because file naming and precedence are product conventions. The invariant is general: compaction must not silently discard applicable instructions.

### An unresolved tool call has an unknown outcome

The dangerous interval is between executing a side effect and logging its result:

```
   log tool_call(c3) ──► execute ──► log tool_result(c3)
                    ▲              ▲
                    │              └─ crash: effect may have landed
                    └─ crash: effect may not have begun

   after restart, both worlds look identical: call without result
```

A call id solves one important case: if `tool_result(c3)` exists, replay returns that result without executing the tool again. The id must remain bound to the same tool and arguments; reusing it for a different operation is an invariant violation, not a cache hit. It cannot tell whether a call with no result ran. This runner therefore takes a conservative posture: it records `unknown_outcome_after_crash`, refuses a blind re-execution, and asks the agent to reconcile workspace state.

That is not exactly-once execution. Exactly-once effects require cooperation below the harness: a tool-level idempotency key, a transaction coupling effect and record, or operation-specific reconciliation. The crash drill constructs both possible worlds behind the same unresolved log so the ambiguity is impossible to wave away.

### Permissions are policy, not containment

The runner's `ALLOW` / `ASK` / `DENY` table makes authorization explicit and auditable. `ASK` means a matching approval was already written to the log; this module does not build an interactive approval UI. The teaching runner defaults unlisted tools to `ALLOW` so the early drills stay small. A consequential production harness normally fails closed, grants the least capability needed, and binds approval to the exact operation, actor, and lifetime.

The table is **not a sandbox**. A Python tool still has the process's filesystem, network, and credential access. A malicious or buggy tool can bypass a harness convention. Real containment requires an OS, container, or VM boundary plus explicit network and secret controls.

### Tool output is untrusted data

A webpage, file, issue, or tool response can contain text that tells the model to ignore its task or take a consequential action. Rendering that text as an `Observation` does not give it authority. A robust harness preserves provenance, separates instructions from external content, limits tool capabilities, validates consequential actions against deterministic policy, and evaluates adversarial observations. These are layered defenses, not a claim that prompt injection has been solved.

### Classify failures before retrying

Transient failures can change when the world changes: a timeout or busy resource may succeed on a bounded retry. Deterministic failures require the request to change: retrying the same missing path or malformed arguments only makes the failure slower. `classify_failure` is a small visible substring policy for this lesson, not a production error taxonomy.

Repeat and step budgets complete the implemented safety story. They are properties of the run, so resume reconstructs consumed steps and repeat history from the log rather than granting a fresh allowance. They do not make a task succeed; they turn runaway behavior into a named stop reason that can be inspected and, with an explicitly enlarged budget, resumed deliberately.

Production resource bounds also include wall-clock deadlines, cancellation propagation, tool timeouts, token or cost ceilings, and sometimes per-tool call limits. Those controls are named here but not added to the exercise API.

### Resume and replay

For the process-crash case in scope, resume replays complete events, turns unresolved calls into explicit unknown-outcome results, reconstructs remaining budgets, and re-enters the loop. It never replays tool side effects. Hidden service state, torn JSONL records, storage loss, or concurrent writers are outside this implementation's durability boundary.

### State records and telemetry answer different questions

The event log reconstructs what the run did. Operational telemetry helps explain how it behaved across runs: run and correlation ids, timestamps and latency, backend/model/tool versions, configuration, token use, retries, and stop reasons. Prompts, tool arguments, and results may contain secrets or personal data, so observability also needs redaction, access control, and retention rules. This lesson keeps the causal log small and does not implement a telemetry backend.

### The controlled comparison

Exercise 5 compares Module 19's loop with `HarnessAgent` while holding the scripted backend, task, tools, step budget, and injected fault schedule fixed. Scenarios cover a clean run, transient failure, backend crash, context pressure, repetition, and permission denial. Exact workspace verifiers—not the mere presence of `Final Answer`—score success. The table reports success, model calls, tool executions, process resumes, tool retries, duplicate effects, and stop reason.

The clean row may show no difference; that is a useful control. Faulted rows reveal which guarantees each harness actually supplies. Exercise 6 optionally repeats a small subset with one ProdLM. A production-strength model is the useful transfer check because a much weaker local model can fail basic tool formatting so often that model incapability swamps any harness effect. A second ProdLM is stretch work, not a requirement.

## Concepts to internalize

- **The event log is the reconstruction source.** Context, resume state, and audit output are views.
- **Context is a policy over the whole prompt.** Preserve resolved instructions, budget system and tool text, and make trajectory compaction deterministic enough to test.
- **An unresolved call has an unknown outcome.** Call ids dedupe recorded results, not ambiguous effects.
- **Permissions and sandboxing are different layers.** Policy does not contain tool code.
- **Observations are data, not instructions.** Preserve provenance and constrain authority around untrusted content.
- **Failures are classified before retrying.** Transient errors may merit retries; deterministic errors need a changed request.
- **Budgets survive resume.** A process restart must not reset the run's safety limits.
- **Audit state and operational telemetry differ.** Both may contain sensitive data requiring lifecycle controls.
- **Harness claims need controlled evaluation.** Hold the backend and fault schedule fixed, then verify exact external state.

### What we don't cover

- **Exactly-once distributed effects.** They require transactional or idempotent services, not a JSONL convention.
- **Security isolation.** VM/container boundaries, egress control, and secret handling are systems-security topics.
- **A product-specific rules-file resolver.** We cover deterministic discovery, scope, precedence, and resume consistency conceptually, not one tool's file convention.
- **A complete prompt-injection defense.** We establish the trust boundary and layered mitigations; robust defense remains an active systems problem.
- **A telemetry backend.** We identify useful provenance and privacy controls without adding an observability stack.
- **Power-loss durability and concurrent writers.** The event log is intentionally single-process teaching code.
- **Multi-agent orchestration.** Delegation multiplies every state, permission, and recovery problem here.
- **Training inside the harness.** Forkable rollout sandboxes and trajectory training exceed this laptop-scale module.

---
## What you'll build

Package: `g2c/harness/`

```python
class EventLog:                                      # provided
    def append(self, type, payload, *, call_id=None) -> Event: ...
    def replay(self) -> list[Event]: ...

def compact_context(events, budget_tokens) -> list[str]:  # scaffolded
    # preserve task + newest event; compact older material

Budgets(context_tokens=2000, model_context_tokens=4096)
    # trajectory allocation plus whole-window ceiling

class ToolRunner:
    def execute(self, call: ToolCall) -> ToolResult: ...   # scaffolded
    # dedupe recorded results, surface unknown outcomes,
    # enforce policy, and log intent before execution

def classify_failure(result) -> RetryDecision: ...        # scaffolded

class HarnessAgent:
    def run(self, task) -> HarnessRunResult: ...           # provided
    def resume(self) -> HarnessRunResult: ...              # scaffolded
```

`HarnessAgent` accepts a provided `context_policy` argument so the notebook can run naive and invariant-preserving policies through the same loop. The model side is unchanged; the work is harness engineering.

## How to run the tests

```bash
source .venv/bin/activate

pytest tests/test_harness.py
pytest tests/test_harness.py -x
pytest tests/test_harness.py -k compact
pytest tests/test_harness.py -k runner
pytest tests/test_harness.py -k resume
```

Initial scaffold state: **1 passed, 26 failed**. The provided event-log round trip starts green; each implementation step turns a coherent group of failures green.

The tests use a scripted backend and deliberately misbehaving tools. No ProdLM or network is required. The crash-window tests construct both possible worlds behind the same unresolved log: no side effect yet and a side effect that already landed. The runner must not pretend it can distinguish them.

## Exercises

Open the working notebook with `./notebook.sh harness` (or `./notebook.sh harness --fresh` to reset from the clean scaffold), write answers in the `Question:` / `Answer:` cells, and ask a coding agent for hints or grading. Partial submissions are fine because blank answers are skipped.

1. **Read a trajectory.** Open `events.jsonl` and trace call ids, intent-before-outcome ordering, result statuses, and the stop record.
2. **Crash and confront ambiguity.** Resume after a backend crash, then construct an unresolved call whose side effect may already have landed. Verify the runner reports an unknown outcome without re-executing it.
3. **Make context drift live.** Run the same prompt-sensitive backend through naive drop-oldest and invariant-preserving policies. Observe the naive run drift after losing the task, then identify which resolved instructions belong outside compactable trajectory history.
4. **Break policy and tools.** Exercise a transient failure, deterministic failure, denied call, and repetition trap. Identify which layer responds.
5. **Run the controlled matrix.** Compare Module 19's loop with `HarnessAgent` under fixed scripted scenarios and exact verifiers.
6. **Optional: transfer to ProdLM.** Repeat a small clean/faulted subset with one ProdLM. A second model is stretch work.

## Pitfalls to expect

- **Persisting rendered context instead of events.** Compaction then destroys the information needed for audit and resume.
- **Logging only after execution.** Intent-before-action is what exposes the unknown-outcome window.
- **Calling call-id dedupe exactly-once.** It only suppresses re-execution when a result is present.
- **Reusing a call id for different arguments.** An idempotency key names one immutable operation.
- **Budgeting only the trajectory.** System instructions, rules, tool schemas, separators, and reserved output also occupy the model window.
- **Compacting instructions with old history.** Applicable rules and durable constraints are a separate retained layer.
- **Retrying deterministic failures.** The request, not elapsed time, must change.
- **Calling permissions a sandbox.** In-process checks do not contain tool implementations.
- **Treating observations as instructions.** External content has no authority merely because the model can read it.
- **Resetting budgets on resume.** Limits belong to the logical run, not one process lifetime.
- **Claiming durable recovery from JSONL alone.** This module covers ordinary process crashes with one writer and an intact filesystem.
- **Scoring `Final Answer` as success.** Verify exact workspace state.

## M-series notes

- **The required path is CPU-cheap.** Tests and the matrix use scripted backends and run without model weights.
- **ProdLM is optional and inference-bound.** Start with one model and two scenarios; a second model is stretch work.
- **Keep the workspace on disk.** The crash drill assumes the process is disposable while the log and workspace survive.

---
## Reading

Primary:

- **Yao, Zhao, Yu et al., “ReAct: Synergizing Reasoning and Acting in Language Models” (2022).** The loop Module 19 built and this module wraps.
- **Yang, Jimenez, Wettig et al., “SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering” (2024).** A concrete argument that the model/environment interface changes agent performance.
- **Fowler, “Event Sourcing” (2005).** The source-of-truth pattern behind `EventLog`.

Secondary:

- **Anthropic, “Building Effective Agents” (2024).** A practitioner's case for simple, inspectable loops.
- **[OWASP, “LLM01: Prompt Injection.”](https://genai.owasp.org/llmrisk/llm01-prompt-injection/)** Indirect injection, least privilege, approval, content separation, and adversarial testing.
- **[OpenTelemetry, “Generative AI semantic conventions.”](https://opentelemetry.io/docs/specs/semconv/registry/attributes/gen-ai/)** A shared vocabulary for model, agent, tool, latency, and token telemetry—with explicit sensitivity warnings.
- **Terminal-Bench / Harbor.** Reproducible terminal environments with test-based verification.
- **METR, “Measuring AI Ability to Complete Long Tasks.”** Reliability over duration as a capability axis.

## Deliverable checklist

- [ ] All tests in `tests/test_harness.py` pass.
- [ ] Notebook contains an annotated event log, the unknown-outcome drill, and both live context-policy runs.
- [ ] Exercise 4 demonstrates transient, deterministic, permission, and repetition handling.
- [ ] Exercise 5 contains the controlled comparison table and an evidence-based interpretation.
- [ ] You can explain why call-id dedupe handles recorded results but cannot close the unknown-outcome window by itself.
- [ ] You can explain why applicable rules stay outside compactable trajectory history and how resume can detect that rules changed.
- [ ] You can distinguish trusted instructions, untrusted observations, audit state, and operational telemetry.
- [ ] You can explain why the permission table is not a sandbox and where this event log's durability boundary lies.
