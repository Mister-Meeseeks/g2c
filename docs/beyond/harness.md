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

The evaluated unit extends beyond just the model weights. The *agent harness* is the software that wraps everything outside the model, and is built to maximize the effectiveness of the model in terms of agent capabilities, trust and usability. 

```
   model + prompt + tools + context policy + workspace
         + permissions + recovery + stopping rules
   ───────────────────────────────────────────────────
                         agent
```


A guiding principal for agent harness engineering is that LLM inference is expensive, but deterministic software is cheap. At least at the size and scale of LLM input and output. An illustrative example is that a very fast model can generate maybe 500 tokens per second. A high performance JSON parser can process 500 *million* tokens per second. A consumer hard disk can easily store 500 *billion* tokens.

The lesson is that when it comes to any input or output going to or from the LLM, the agent harness should treat logging and indexing it as essentially "free". The deterministic software bends over backwards to accomodate the LLM, because the LLM is almost always the bottleneck. 

## The big idea

Most agent harnesses have coalesced around a guiding design decision: the trajectory is represented as an append-only event log. The basic elements of the log are model turns and tool calls. All of the input, output and metadata related to these is generously preserved in the append log. Context, resume state, and audit output become materialized views over the log. 

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

The critical principal here is to separate "what the model sees on the next turn" from "the full history of the run". A naive approach would simply be to keep durable state as whatever happens to be in model context. But remember that model context is significantly more expensive than disk storage. 

Long running context is inevitably lossy, but history should not be. Archived turns, completed and abandoned tool calls, metadata around calls, and more can all become essential to reconstruct or analyze a run without having to re-execute previously completed effects.

### Context is a deterministic policy

All models have a context window ceiling. For every single model in existence today, this window is comparatively small. On any serious long-running work stream the harness will have to deal with running out of context. Even outside of that, very long context can result in both expensive inference costs and degraded instruction following. The process agent harnesses use to manage overly long context is *compaction*.

The simplest compaction strategy is first-in first-out (FIFO). When you reach the buffer window, simply drop the oldest text first. This is often the worst approach because the start of the window contains the most critical information, like general harness context, core task statement, repository rules and similar "headlines".

Compaction strategy should be aware of which content is likely to be persistently important (e.g. task instructions), and which is likely to be ephemerally important (e.g. intermediate tool call results). Recency is also an important factor. Even if it's likely only to be ephemerally important, the last tool call is much more likley to have relevant context for the next model turn than a tool call made 37 turns ago.

Compaction can also involve using LLM intelligence to compress, consolidate or strip data. For example a model might generate a short natural language summary of the important parts of the last 20 turns. Or it might go through tool call outputs and remove purely system level diagnostic data,.

The compaction function we'll build in this module is deliberately modest and testable:

```
   always preserve verbatim       compact or drop first
   ────────────────────────       ─────────────────────
   the task statement             old tool output
   the newest event               old narration and actions
```

It keeps the task, keeps recent events, shortens old tool output, then drops the oldest remaining middle events if needed. It does **not** claim to discover every constraint, commitment, or open item. Production systems usually represent those separately or use a summarizer with its own evaluation.

The trajectory is only one part of the prompt budget. A harness also has to leave room for system prompt, rendered tool schemas and reserves space for the next completion. Without adequately buffered reserve space, one verbose tool can overflow a context. This module uses a cheap token estimate; a production harness uses the backend's tokenizer and model-window metadata.

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

The event log should always record resolved rule paths and content hashes so the harness can detect when the operating instructions have changed. This module does not implement an `AGENTS.md` filesystem resolver because file naming and precedence vary between product conventions. However the invariant is general: compaction must never silently discard core instructions.

### Unresolved tool calls

With agent tool calls, the most dangerous interval is between executing a side effect and logging its result:

```
   log tool_call(call_id) ──► execute ──► log tool_result(call_id)
                    ▲              ▲
                    │              └─ crash: effect may have landed
                    └─ crash: effect may not have begun

   after restart, both worlds look identical: call without result
```

`call_id` solves one critical case: if `tool_result(call_id)` exists, replay returns that result without executing the tool again. The id must remain bound to the same tool and arguments; reusing it ever for a different operation is an invariant violation. It cannot tell whether a call with no result ran. This runner therefore takes a conservative posture: it records `unknown_outcome_after_crash`, refuses a blind re-execution, and asks the agent to reconcile workspace state.

That is not exactly-once execution. Exactly-once effects require cooperation below the harness: a tool-level idempotency key, a transaction coupling effect and record, or operation-specific reconciliation. The crash drill constructs both possible worlds behind the same unresolved log so the ambiguity is impossible to wave away.

### Permissions are policy, not containment

The runner's `ALLOW` / `ASK` / `DENY` table makes authorization explicit and auditable. `ASK` means a matching approval was already written to the log (this module does not build an interactive approval UI). The teaching runner defaults unlisted tools to `ALLOW` so the early drills stay small. A consequential production harness normally fails on closed, grants the least capability needed, and binds approval to the exact operation, actor, and lifetime.

A permission table is **not a sandbox**. A Python tool still has the process's filesystem, network, and credential access. A malicious or buggy tool can bypass a harness convention. Real containment requires an OS, container, or VM boundary plus explicit network and secret controls.

### Tool output is untrusted data

Any tool response, webpage, file, or query result may potentially contain text with malicious instructions. Rendering that text with an `Observation` tag does not make it automatically safe. A robust harness preserves provenance, separates instructions from external content, limits tool capabilities, validates consequential actions against deterministic policy, and evaluates adversarial observations. 

These measures are about layered defense in depth. Think of stacking sliced swiss cheese. Enough slices and there's a low probability a hole passes through the whole stack.  There is no silver bullet that guarantees absolute safety. 

### Classify failures before retrying

*Transient failures* are failures that may change just because the external world changes. 
A timeout or busy resource may succeed on a bounded retry even with the same request that previously failed. *Deterministic failures* require the request to change: retrying the same missing path or malformed arguments only makes the failure slower. In this lesson we use a small visible substring policy, not a production error taxonomy.

*Repeat budgets* and *step budgets* close out the safety story. They are properties of the run, so resume always reconstructs steps and repeat history from the log (rather than granting a refreshed budget on resume). Repeat and step budgets do not work by helping a task to succeed. They mitigate blast radius by turning runaway behavior into a named stop reason. A stop reason that can be inspected and, with an explicitly enlarged budget, resumed deliberately.

Beyond step and repeat budgets, *resource bounds* in production include a menagerie of wall-clock deadlines, cancellation propagation, tool timeouts, token or cost ceilings, and per-tool call limits. Those controls are named here but not added to the exercise API. LLM behavior is unpredictable, and with any complex task failure, often in unexpected ways is inevitable. Be prepared to handle gracefully. 

### Resume and replay

*Resume* is the process of continuing a previously interrupted run from something close to the state it was in when interrupted. Resume replays the complete event history, turns any pending calls into explicit unknown-outcome failures, reconstructs remaining budgets for the run, then re-enters the agent loop. Resume should **never** replay previous tool side effects, unless absolutely known to be idempotent. 

Resume must also be prepared to handle lower level system issues around durability, including hidden service state, torn JSONL records, storage loss, and concurrent writers. These are easy to forget because they happen just infrequently enough. Like any system with durable state, a production system must have contingencies for handling corrupted storage. 

The good news is because LLM input and output is smalll, slow and high latency. The durability layer has a lot of margin to afford generous repair, replication, staging, safety checks and concurrency handling. A harness has much lower performance SLAs than something like a database, and that means the durability solutions that fit in those SLAs are often simpler.

### State records and telemetry answer different questions

The event log is what reconstructs what the run did. *Operational telemetry* explains its behavior across runs: run and correlation ids, timestamps and latency, backend/model/tool versions, configuration, token use, retries, and stop reasons. 

One thing to keep in mind is that *secrets* and personal data have a way of leaking into prompts, tool calls, and retrieval queries. So observability in production also needs to consider redaction, access control, and retention rules.

### Harness evals

In Module 15 we learned how to build evaluations to measure model behavior. This is necessary for comparing models in any objective way. Without this decisions in model training and selection reduce to low confidence vibes. The same principals apply to *harness evals*. Even with identical models, the decisions we make about context engineering, tool call policy, step budget and the like can have material impacts on agent performance. 

Like model evals, harness evals should be of moderately sufficient difficulty relative to underlying capabilities. An eval that's so hard that nothing passes or an eval that fully saturates, gives us no information. Like model evals, harness evals should carefully consider objective metrics for measuring pass rate, but carefully consider ways the objective can be gated. Human review of solutions is always a necessity in eval design.

Contrasted to model evals, harness evals lean much more heavy on long running, multi-turn agentic tasks. The best and worst harness are going to make very little difference if the prompt is a simple query response like "what's the third largest city in Norway?". But for tasks like "migrate this running database with no downtime or dataloss", harness engineering can have dramatic impact even on the same model. 

One thing to be aware of, especially with modern models, is that models are increasingly post trained on agentic workflows, which by definition need to run in an agent harness. As these tasks get more complex models become increasingly optimized for harnesses that specifically look like their training environment. It's not surprising that Opus runs better in Claude Code than Codex. Be aware that it's often difficult to evaluate a model or a harness in isolation, because of synergies from agentic post trainng. 

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
