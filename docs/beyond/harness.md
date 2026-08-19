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

The evaluated unit extends beyond the model weights. The *agent harness* is the software around the model: it selects context and tools, records effects, enforces policy, handles failures, and decides when to stop.

```
   model + prompt + tools + context policy + workspace
         + permissions + recovery + stopping rules
   ───────────────────────────────────────────────────
                         agent
```
A useful design principle is that model inference is generally expensive and high-latency relative to local deterministic bookkeeping. It is often worth spending modest compute and storage to make each model call more effective, inspectable, and recoverable. Logging is not literally free, however: retention, privacy, access control, and storage growth remain real system constraints.

## The big idea

An agent harness is not another intelligence inside the system. It is an ordinary program wrapped around a repeated model call: assemble the next context, ask the model what to do, mediate any requested action, record what happened, and repeat. The model proposes the next step; the harness controls what the model sees, what may happen, and what survives.

```
   task + rules + recorded events
                 │
                 ▼
         select next context
                 │
                 ▼
            call model
                 │
                 ▼
          record response
             /       \
    final answer       proposed tool call
         │                     │
        stop            authorize action
                               │
                               ▼
                         record intent
                               │
                               ▼
                        execute and record
                         outcome ──────────► repeat
```

The hard part is not inventing a more elaborate loop. It is making each ordinary systems decision—state, context, authority, recovery, and stopping—explicit enough to inspect and test.

This module makes one organizing choice: represent the trajectory as an append-only event log. Model turns, tool calls, results, and relevant metadata enter the log in order. Context, resume state, and audit output then become views over that durable history.

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

The critical principle is to separate **what the model sees next** from **what the run recorded**. Active context is bounded and selective; the event history should preserve enough information to reconstruct and analyze the run without re-executing completed effects. Archiving an event does not mean placing it in every future prompt.

### Context is a deterministic policy

Context is bounded. Long tasks and verbose tools eventually force the harness to choose what the model sees, while unnecessarily long prompts also add cost and can make relevant information harder to use. That selection process is *compaction*.

Simple first-in, first-out compaction is weak because the oldest content may include the task and durable constraints. A better policy preserves invariants, favors recent working state, and shortens or drops older verbose observations first.

A model-generated summary can compress more intelligently, but it is still a lossy generated artifact that needs evaluation. Durable commitments, constraints, and open items are safer when represented explicitly rather than entrusted only to prose summary.

The compaction function we'll build in this module is deliberately modest and testable:

```
   always preserve verbatim       compact or drop first
   ────────────────────────       ─────────────────────
   the task statement             old tool output
   the newest event               old narration and actions
```

It keeps the task, keeps recent events, shortens old tool output, then drops the oldest remaining middle events if needed. It does **not** claim to discover every constraint, commitment, or open item. Production systems usually represent those separately or use a summarizer with its own evaluation.

The trajectory is only one part of the prompt budget. A harness also has to leave room for the system prompt and rendered tool schemas, and reserve space for the next completion. Without that reserve, one verbose tool can overflow the context window. This module uses a cheap token estimate; a production harness uses the backend's tokenizer and model-window metadata.

### Rules are resolved instructions, not trajectory history

A coding harness assembles instructions from more than just the task prompt. System policy, repository rules such as `AGENTS.md`, narrower directory rules, current user constraints, and permissions may all apply. Discovery, scope, and precedence must be deterministic. Once resolved, this instruction layer stays separate from the compactable history:

```
   retain as resolved instructions       compact as trajectory
   ───────────────────────────────       ─────────────────────
   system and harness policy             old narration
   applicable repository rules           old actions
   task and durable constraints           bulky tool observations
   current permissions
```

When rules affect a run, the log should record their resolved paths and content hashes—or an appropriately redacted snapshot—so resume can detect changes. This module does not implement an `AGENTS.md` filesystem resolver because file naming and precedence vary between products. The general invariant remains: compaction must never silently discard core instructions.

### Tool selection shapes model behavior

A tool is an interface the model must learn to operate from its name, description, schema, and observed results. Tool selection therefore affects capability even when the underlying model does not change. A good tool makes useful actions easy to express, mistakes easy to detect, and consequential authority easy to constrain.

Unix-style command-line tools are often a strong baseline for coding agents. A shell plus commands such as `grep`, `sed`, and `git` are widely documented, compose well, act directly on the workspace, and appear throughout code examples that models may have encountered during training. That familiarity can matter. It is a reason to evaluate familiar interfaces, not a guarantee that shell tools are always best.

Shell commands can expose broad authority, accept ambiguous syntax, and return large unstructured outputs. A purpose-built tool may instead validate arguments, narrow permissions, return typed results, and make failures clearer. Choose the smallest tool surface that covers the task, then compare interfaces under the same model and tasks. Favor legible inputs, bounded outputs, deterministic errors, and effects the harness can verify.

### Unresolved tool calls

With agent tool calls, the most dangerous interval is between executing a side effect and logging its result:

```
   log tool_call(call_id) ──► execute ──► log tool_result(call_id)
                    ▲              ▲
                    │              └─ crash: effect may have landed
                    └─ crash: effect may not have begun

   after restart, both worlds look identical: call without result
```

`call_id` solves one critical case: if `tool_result(call_id)` exists, replay returns that result without executing the tool again. The id must remain bound to the same tool and arguments; reusing it for a different operation is an invariant violation. It cannot tell whether a call with no result ran. This runner therefore takes a conservative posture: it records `unknown_outcome_after_crash`, refuses a blind re-execution, and asks the agent to reconcile workspace state.

That is not exactly-once execution. Exactly-once effects require cooperation below the harness: a tool-level idempotency key, a transaction coupling effect and record, or operation-specific reconciliation. The crash drill constructs both possible worlds behind the same unresolved log so the ambiguity is impossible to wave away.

### Permissions are policy, not containment

The runner's `ALLOW` / `ASK` / `DENY` table makes authorization explicit and auditable. `ASK` means a matching approval was already written to the log (this module does not build an interactive approval UI). The teaching runner defaults unlisted tools to `ALLOW` so the early drills stay small. A consequential production harness normally fails closed, grants the least capability needed, and binds approval to the exact operation, actor, and lifetime.

A permission table is **not a sandbox**. A Python tool still has the process's filesystem, network, and credential access. A malicious or buggy tool can bypass a harness convention. Real containment requires an OS, container, or VM boundary plus explicit network and secret controls.

### Tool output is untrusted data

Any tool response, webpage, file, or query result may contain adversarial instructions. Rendering that text with an `Observation` tag records provenance; it does not contain the content or make it safe. Defense is layered: separate instructions from external data, limit tool capabilities, validate consequential actions against deterministic policy, require approval where appropriate, and evaluate adversarial observations.

### Classify failures before retrying

*Transient failures* may change as the external world changes: a timeout or busy resource may succeed on a bounded retry with the same request. *Deterministic failures* require the request to change; retrying the same missing path or malformed arguments only makes the failure slower. This lesson uses a small visible substring policy, not a production error taxonomy.

*Repeat budgets* and *step budgets* belong to the logical run, so resume reconstructs them from the log instead of granting a fresh allowance. They bound damage by turning runaway behavior into an inspectable stop reason that can be resumed only through a deliberate budget change.

Production resource bounds also include wall-clock deadlines, cancellation propagation, tool timeouts, token or cost ceilings, and per-tool call limits. This module names those controls without adding them to the exercise API.

### Resume and replay

Within this module's scope—an ordinary process crash, one writer, and an intact filesystem—resume replays complete events, turns unresolved calls into explicit unknown outcomes, reconstructs the run's budgets, and re-enters the loop. It must not re-execute an unresolved effect merely because its result is absent. Safe retries require tool-level idempotency or operation-specific reconciliation.

Production storage must also account for torn records, concurrent writers, hidden service state, and data loss. Those stronger durability mechanisms sit below this teaching log and remain outside the exercise.

### State records and telemetry answer different questions

The event log reconstructs what one run did. *Operational telemetry* compares behavior across runs: run and correlation ids, timestamps and latency, backend/model/tool versions, configuration, token use, retries, and stop reasons.

Both can contain secrets or personal data from prompts, tool calls, and retrieval queries. Production observability therefore needs redaction, access control, and retention rules.

### A subagent is another bounded harness run

A subagent is usually not a new kind of intelligence. It is another agent loop invoked by a parent with a delegated task, selected context, scoped tools, and its own budget. The child returns a result and supporting evidence; the parent remains responsible for deciding how to use and verify them.

Delegation works best when work can be isolated: parallel searches over independent questions, specialist analysis, implementation followed by review, or a noisy investigation kept out of the parent's active context. It works poorly when subtasks are tightly coupled or several agents edit the same state without coordination. Parallelism may reduce elapsed time, but it usually increases total inference and introduces duplicated work, inconsistent assumptions, and merge conflicts.

The same harness invariants apply recursively. Record parent–child lineage, scope permissions and budgets per child, propagate cancellation, isolate or coordinate workspace effects, and treat a child report as a claim to verify rather than proof of success. Full orchestration adds scheduling, communication, and conflict resolution; delegation does not make those systems problems disappear.

### Harness evals

Module 15 evaluated model behavior. Here the same discipline applies to the surrounding system: context policy, tool interfaces, permissions, retries, and stopping rules can materially change the behavior of a fixed model.

A controlled comparison holds the model, task, fault schedule, and budgets fixed, then verifies exact external state. Useful cases avoid both floor and ceiling effects and emphasize long, multi-turn, or faulted work where harness decisions can matter. Simple question answering is unlikely to distinguish recovery or compaction policies.

Models may also be post-trained against particular tool schemas and transcript formats. Measured performance can therefore reflect model–harness compatibility as well as the quality of either component alone. Compare the whole system, use ablations before assigning a cause, and use human review to design and audit the metrics.

## Concepts to internalize

- **The event log is the reconstruction source.** Context, resume state, and audit output are views.
- **Context is a policy over the whole prompt.** Preserve resolved instructions, budget system and tool text, and make trajectory compaction deterministic enough to test.
- **The tool surface is part of the system.** Familiarity, composability, authority, output shape, and failure clarity all affect model performance.
- **An unresolved call has an unknown outcome.** Call ids dedupe recorded results, not ambiguous effects.
- **Permissions and sandboxing are different layers.** Policy does not contain tool code.
- **Observations are data, not instructions.** Preserve provenance and constrain authority around untrusted content.
- **Failures are classified before retrying.** Transient errors may merit retries; deterministic errors need a changed request.
- **Budgets survive resume.** A process restart must not reset the run's safety limits.
- **Audit state and operational telemetry differ.** Both may contain sensitive data requiring lifecycle controls.
- **A subagent is a scoped child run.** Delegation adds parallelism and context isolation, but also lineage, coordination, and verification obligations.
- **Harness claims need controlled evaluation.** Hold the backend and fault schedule fixed, then verify exact external state.

### What we don't cover

- **Exactly-once distributed effects.** They require transactional or idempotent services, not a JSONL convention.
- **Security isolation.** VM/container boundaries, egress control, and secret handling are systems-security topics.
- **A product-specific rules-file resolver.** We cover deterministic discovery, scope, precedence, and resume consistency conceptually, not one tool's file convention.
- **A complete prompt-injection defense.** We establish the trust boundary and layered mitigations; robust defense remains an active systems problem.
- **A telemetry backend.** We identify useful provenance and privacy controls without adding an observability stack.
- **Power-loss durability and concurrent writers.** The event log is intentionally single-process teaching code.
- **A multi-agent orchestrator.** We cover delegation patterns and invariants, not scheduling, inter-agent protocols, shared-memory design, or conflict resolution.
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
