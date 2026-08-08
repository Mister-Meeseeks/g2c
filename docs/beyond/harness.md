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

The course's agent arc built tools that execute (18), a loop that reasons and acts (19), and an assistant that assembles the system (20). Those short notebook runs can keep state in a Python list and treat failure as an exception. Longer-running systems need an answer to different questions: what reached durable-enough storage before the process died, what should the model see next, what may a tool do, and when must the run stop?

The evaluated unit is therefore more than model weights:

```
   model + prompt + tools + context policy + workspace
         + permissions + recovery + stopping rules
   ───────────────────────────────────────────────────
                         agent
```

This module holds the backend fixed and changes the harness so you can see which differences genuinely come from the surrounding system.

## The big idea

One design decision organizes the module: **the trajectory is an append-only event log, while context, resume state, and audit output are views of that log.**

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

### An unresolved tool call has an unknown outcome

The dangerous interval is between executing a side effect and logging its result:

```
   log tool_call(c3) ──► execute ──► log tool_result(c3)
                    ▲              ▲
                    │              └─ crash: effect may have landed
                    └─ crash: effect may not have begun

   after restart, both worlds look identical: call without result
```

A call id solves one important case: if `tool_result(c3)` exists, replay returns that result without executing the tool again. It cannot tell whether a call with no result ran. This runner therefore takes a conservative posture: it records `unknown_outcome_after_crash`, refuses a blind re-execution, and asks the agent to reconcile workspace state.

That is not exactly-once execution. Exactly-once effects require cooperation below the harness: a tool-level idempotency key, a transaction coupling effect and record, or operation-specific reconciliation. The crash drill constructs both possible worlds behind the same unresolved log so the ambiguity is impossible to wave away.

### Permissions are policy, not containment

The runner's `ALLOW` / `ASK` / `DENY` table makes authorization explicit and auditable. `ASK` means a matching approval was already written to the log; this module does not build an interactive approval UI.

The table is **not a sandbox**. A Python tool still has the process's filesystem, network, and credential access. A malicious or buggy tool can bypass a harness convention. Real containment requires an OS, container, or VM boundary plus explicit network and secret controls.

### Classify failures before retrying

Transient failures can change when the world changes: a timeout or busy resource may succeed on a bounded retry. Deterministic failures require the request to change: retrying the same missing path or malformed arguments only makes the failure slower. `classify_failure` is a small visible substring policy for this lesson, not a production error taxonomy.

Repeat and step budgets complete the safety story. They do not make a task succeed; they turn runaway behavior into a named stop reason that can be inspected and resumed deliberately.

### Resume and replay

For the process-crash case in scope, resume replays complete events, turns unresolved calls into explicit unknown-outcome results, and re-enters the loop. It never replays tool side effects. Hidden service state, torn JSONL records, storage loss, or concurrent writers are outside this implementation's durability boundary.

### The controlled comparison

Exercise 5 compares Module 19's loop with `HarnessAgent` while holding the scripted backend, task, tools, step budget, and injected fault schedule fixed. Scenarios cover a clean run, transient failure, backend crash, context pressure, repetition, and permission denial. Exact workspace verifiers—not the mere presence of `Final Answer`—score success. The table reports success, model calls, tool executions, recoveries, duplicate effects, and stop reason.

The clean row may show no difference; that is a useful control. Faulted rows reveal which guarantees each harness actually supplies. Exercise 6 optionally repeats a small subset with one ProdLM. A production-strength model is the useful transfer check because a much weaker local model can fail basic tool formatting so often that model incapability swamps any harness effect. A second ProdLM is stretch work, not a requirement.

## Concepts to internalize

- **The event log is the reconstruction source.** Context, resume state, and audit output are views.
- **Context is a policy.** Preserve explicit invariants; make compaction deterministic enough to test.
- **An unresolved call has an unknown outcome.** Call ids dedupe recorded results, not ambiguous effects.
- **Permissions and sandboxing are different layers.** Policy does not contain tool code.
- **Failures are classified before retrying.** Transient errors may merit retries; deterministic errors need a changed request.
- **Harness claims need controlled evaluation.** Hold the backend and fault schedule fixed, then verify exact external state.

### What we don't cover

- **Exactly-once distributed effects.** They require transactional or idempotent services, not a JSONL convention.
- **Security isolation.** VM/container boundaries, egress control, and secret handling are systems-security topics.
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

Initial scaffold state: **1 passed, 21 failed**. The provided event-log round trip starts green; each implementation step turns a coherent group of failures green.

The tests use a scripted backend and deliberately misbehaving tools. No ProdLM or network is required. The crash-window tests construct both possible worlds behind the same unresolved log: no side effect yet and a side effect that already landed. The runner must not pretend it can distinguish them.

## Exercises

Open the working notebook with `./notebook.sh harness` (or `./notebook.sh harness --fresh` to reset from the clean scaffold), write answers in the `Question:` / `Answer:` cells, and ask a coding agent for hints or grading. Partial submissions are fine because blank answers are skipped.

1. **Read a trajectory.** Open `events.jsonl` and trace call ids, intent-before-outcome ordering, result statuses, and the stop record.
2. **Crash and confront ambiguity.** Resume after a backend crash, then construct an unresolved call whose side effect may already have landed. Verify the runner reports an unknown outcome without re-executing it.
3. **Make context drift live.** Run the same prompt-sensitive backend through naive drop-oldest and invariant-preserving policies. Observe the naive run drift after losing the task.
4. **Break policy and tools.** Exercise a transient failure, deterministic failure, denied call, and repetition trap. Identify which layer responds.
5. **Run the controlled matrix.** Compare Module 19's loop with `HarnessAgent` under fixed scripted scenarios and exact verifiers.
6. **Optional: transfer to ProdLM.** Repeat a small clean/faulted subset with one ProdLM. A second model is stretch work.

## Pitfalls to expect

- **Persisting rendered context instead of events.** Compaction then destroys the information needed for audit and resume.
- **Logging only after execution.** Intent-before-action is what exposes the unknown-outcome window.
- **Calling call-id dedupe exactly-once.** It only suppresses re-execution when a result is present.
- **Retrying deterministic failures.** The request, not elapsed time, must change.
- **Calling permissions a sandbox.** In-process checks do not contain tool implementations.
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
- **Terminal-Bench / Harbor.** Reproducible terminal environments with test-based verification.
- **METR, “Measuring AI Ability to Complete Long Tasks.”** Reliability over duration as a capability axis.

## Deliverable checklist

- [ ] All tests in `tests/test_harness.py` pass.
- [ ] Notebook contains an annotated event log, the unknown-outcome drill, and both live context-policy runs.
- [ ] Exercise 4 demonstrates transient, deterministic, permission, and repetition handling.
- [ ] Exercise 5 contains the controlled comparison table and an evidence-based interpretation.
- [ ] You can explain why call-id dedupe handles recorded results but cannot close the unknown-outcome window by itself.
- [ ] You can explain why the permission table is not a sandbox and where this event log's durability boundary lies.
