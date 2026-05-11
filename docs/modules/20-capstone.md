# Module 20 — Capstone: a tiny ChatGPT

> **Question this module answers:** *Can I integrate everything?*

![Module 20 on one page: a five-panel circus map of the capstone assistant. PANEL 1 (top-left, "USER"): a typed question enters from a CLI prompt ("? what's 2+2?"). PANEL 2 (top-middle, "MEMORY"): the Conversation object renders prior turns ("User: hi / Assistant: hello") into a "Previous conversation:" block. PANEL 3 (top-right, "RAG"): the optional retriever pulls k chunks from a corpus and formats them as a "Context from documents:" block. PANEL 4 (center-bottom, "AGENT"): the contextualized message (history + context + current question) is handed to the Module 19 Agent, which runs the full ReAct loop with planning, scratchpad, and tool dispatch. PANEL 5 (bottom-right, "OUT"): the agent's `final_answer` is returned to the user, recorded in the conversation, and gated by the EvalSuite for regression testing. Around the edges, labels point at each upstream module: M16 (Backend) feeds the agent's inference; M17 (DenseRetriever) feeds the RAG block; M18 (ToolRegistry) feeds the agent's tool palette; M19 (Agent) is the per-turn engine. A right-edge banner reads "the lesson is the architecture." A bottom caption reads "Modules 1-19 built the parts; Module 20 is where they live together — and where the post-mortem starts."](20-capstone/Module20-Hero.png)

By the end of this module you will have a working chat assistant — and, more importantly, a characterization of where each layer earns its keep, where it breaks down, and where the from-scratch model stops being viable.

---
## Before you start

* *Finish* `g2c/agent` from [[19-agent]] — `Assistant.chat` wraps `Agent.run`, threading conversation history into each turn
* *Finish* `g2c/tools` from [[18-tools]] and `g2c/inference` from [[16-inference]] — the assistant supplies the tool registry and backend that the agent runs against
* *Finish* `g2c/rag` from [[17-rag]] (optional) — only needed if you wire prefix-style retrieval into the assistant

---
## Prerequisites

Module 20 is the capstone of Phase V (assistant systems) and the final module of the course. There is no new conceptual ground broken here — every piece is already understood. What's new is *integration*: composing the substrates from Modules 16-19 with two new primitives (`Conversation` for inter-turn memory, `Assistant` for the orchestration layer), an eval-as-regression-gate harness, and a CLI you can actually talk to.

The deliverable is the *post-mortem*, not the code. The code is the substrate that makes a thoughtful post-mortem possible. Plan to spend half your time using the assistant, breaking it, and writing about it.

### Math

There isn't really any math in this module. The closest thing is the way the conversation primitive turns a stateless inference call into something that *appears* to remember:

```
   prompt_n+1 = render(history[: n+1])
            = render(history[: n] ⊕ (user_n, assistant_n))
            = prompt_n + (one new turn)
```

This is the same Markov-ish "prompt grows monotonically" property as Module 19's scratchpad — except now the unit is "one whole agent run's output," not "one ReAct step." The model is still memoryless; the prompt does the remembering.

### Computer science

- **Layered memory: scratchpad ≠ conversation.** Module 19 introduced *intra-turn* working memory (the `Scratchpad`: thought / action / observation across one task). Module 20 introduces *inter-turn* working memory (the `Conversation`: user / assistant exchanges across many tasks). Both are short-term in the cognitive-science sense — visible during the current session, dropped at session end. Long-term memory (across sessions) would be a separate system: a vector store of past conversations, a fact-extraction pipeline, or an explicit user-profile object. We don't build that; the next conceptual step is named in the "What you can skip" section.

- **The integration layer is real engineering, not a pass-through.** It's tempting to think `Assistant.chat` is just `agent.run` with a wrapper. It isn't:
    - The conversation history must be *rendered* into the agent's input, not just stored.
    - RAG retrieval has a *placement* decision: prefix-style (what we do) or tool-style (let the agent decide when to retrieve).
    - The agent's `final_answer` may be `None`; the conversation log must stay coherent regardless.
    - The eval harness needs a stable interface that survives across config changes.
  
  Each is a small choice. Bundled, they're the architecture.

- **The eval-as-regression-gate pattern.** Module 15's eval module measured your model's properties. Module 20's eval module asks a different question: *"is the assistant still doing the basic things right after I changed something?"* The check list is short (substring match in the answer, expected tool was called, run actually finished), and the report is a single number (pass rate). The point is to run it after every config change, every prompt edit, every tool addition — like a unit test for behavior. That's a different category of eval from the research-grade benchmarks in Module 15; both have a place.

- **RAG-as-prefix vs RAG-as-tool: the design choice.** When retrieval should happen is a real architectural fork:
    - *Prefix style* (what we do): retrieve once at the start of each turn, splice chunks into the prompt as context. Simple, predictable, fast. The model never has to "decide" to retrieve.
    - *Tool style*: register the retriever as a `search_corpus` tool; let the agent decide when to call it. More flexible (the model can retrieve mid-task as needed), but adds latency (extra agent steps) and can fail (the model forgets to retrieve when it should).
  
  We default to prefix style because it's the simpler integration and pedagogically clearer. A capstone exercise (#5) walks through converting to tool style — a one-page change that's worth doing once to feel the tradeoff.

- **Why is the failed-run placeholder load-bearing?** When the agent fails to produce a `final_answer` (max_steps, duplicate action, no progress), the assistant *still* has to record something in the conversation log — otherwise the next turn's history skips the failed exchange. If the user follows up with "actually, never mind, do X instead," the model needs to see the prior failed attempt as context. The placeholder (`"(no answer — stopped: max_steps)"`) is a low-information stand-in that's at least coherent. The `AssistantTurn.final_answer` stays `None` so the eval harness can detect the failure unambiguously.

- **Conversation truncation is a heuristic, not a science.** When the chat gets long, *something* has to drop. We drop oldest messages (same as the Scratchpad). A real assistant would *summarize* old messages into a single "previously discussed: ..." line; or use a vector store for "long-term memory" that the model retrieves from when it needs old context. We don't build that. The exercise on "where does plain truncation hurt" is the path-not-taken.

- **`format_for_prompt` does NOT include the current user message.** The current message goes in separately, after the history block, so the model can tell "this is what you're being asked NOW" apart from "this is what was asked before." If you include the current message in the rendered history, you double-count it AND lose the separation. The test `test_history_does_not_double_include_current_message` pins this.

- **The eval harness doesn't catch exceptions on purpose.** A broken retriever, a bad config, a missing tool — these are *config bugs*, not model bugs. The eval is a regression gate, not a fault-tolerant runner. If `chat` raises, the eval crashes loudly. Fix the config; re-run. Tool errors and parse errors are model-level wobble (and the agent already handles them as data); config errors are dev-level wobble (and the eval surfaces them by crashing).

### Programming

- **`@dataclass(frozen=True)`** for `Message` (one role + one content string), `EvalCase`, `EvalCaseResult`. Mutable `@dataclass` for `Conversation`, `AssistantTurn`, `EvalReport` — they're built up over time.

- **`Protocol`** for the retriever's interface. The assistant accepts any object with a `retrieve(query, *, k) -> Iterable[...]` method — Module 17's `DenseRetriever` matches, but you can plug in BM25, hybrid, re-ranking, anything. Loose coupling via duck-typing rather than ABC inheritance.

- **`io.StringIO`** for testing the CLI. The CLI takes `inp: TextIO` and `out: TextIO` parameters that default to `sys.stdin` / `sys.stdout`; tests pass `StringIO` instances and assert on the captured output. Same pattern as the Module 18 dispatch tests.

- **`json.dumps` for transcript persistence.** The CLI's `/save <path>` writes a JSON dump of the conversation + per-turn metadata. The `default=str` keyword handles non-serializable values (e.g., the `BackendInfo` extras dict).

- **`shlex.split`** for parsing `/save <path>` arguments. Tiny, but the right tool for "split a command line, respecting quotes" — handles `/save "path with spaces.json"` correctly without bespoke parser code.

### What you can skip

- **Persistent multi-session memory.** The conversation lives in process memory; closing the CLI drops it. A real assistant persists across sessions (SQLite, a vector store, a JSON file in `~/.local/share`), and may distill old conversations into a "user profile" or "remembered facts." The CLI's `/save` is a half-step toward this; the full system is its own design space.

- **Streaming output.** The CLI prints the answer all at once after `chat` returns. A real chat UI streams tokens as they arrive. Same conceptual shape (the model still emits Final Answer at the end), ~3× the code.

- **A web UI.** A lightweight Flask/FastAPI server fronting `Assistant.chat` is straightforward but its own work: HTTP routing, session management, frontend rendering, accessibility, deployment. Out of scope here. The capstone deliverable allows either a CLI or a web UI; the CLI is the lower-effort path.

- **Tool-style retrieval (RAG-as-a-tool).** Exercise #5 walks through it; we don't build it as the default because prefix-style is the simpler pedagogical baseline.

- **Multi-agent orchestration.** A "researcher" agent that delegates sub-tasks to "summarizer" / "fact-checker" agents is a real production pattern (Anthropic's orchestrator-worker, AutoGen, CrewAI). Conceptually orthogonal to what we built; named in the M19 lesson page.

- **Reflection / self-critique.** "Review your last answer and try again if it was wrong" is the Reflexion pattern. Useful when verification is cheap; not built here.

- **Token-budget-aware context management.** When the conversation + scratchpad + RAG context exceed the model's context window, we truncate by character count. Production systems token-count and may summarize old turns. Out of scope.

- **Production sandboxing.** All Module 18/19 caveats apply — `run_python` runs untrusted code in an unsandboxed subprocess. Fine for local pedagogy, NOT fine for a hosted assistant.

## Where this fits in

You've built nineteen modules. Module 1 was a basic autograd engine; Module 10's pretraining was a real loss curve on a real corpus; Module 19's agent was a multi-step ReAct loop with error recovery. The components are all there. They've been tested in isolation but never *together*.

The capstone is where you find out:

```
   ┌───────────────────────────────────────────────────────────────────────┐
   │  THINGS YOU LEARN BY ASSEMBLING                                       │
   ├───────────────────────────────────────────────────────────────────────┤
   │                                                                       │
   │   • Where does each layer EARN its keep?                              │
   │       - The from-scratch model (M10) generates fluent text but        │
   │         can't follow chat instructions reliably.                      │
   │       - SFT (M13) makes it follow prompts.                            │
   │       - DPO (M14) makes it polite.                                    │
   │       - The pretrained pivot (M16) brings real-world knowledge.       │
   │       - RAG (M17) brings YOUR-corpus knowledge.                       │
   │       - Tools (M18) bring computation and side effects.               │
   │       - The agent loop (M19) brings multi-step reasoning.             │
   │       - Conversation memory (M20) makes it usable.                    │
   │                                                                       │
   │   • Where does each layer BREAK?                                      │
   │       - The model's responses get repetitive at low temperature.      │
   │       - The retriever picks the wrong chunks on ambiguous queries.    │
   │       - The agent loops on poorly-described tools.                    │
   │       - The conversation drifts when context exceeds the cap.         │
   │       - Some questions need three of these all at once and one of     │
   │         them goes wrong, masking the others.                          │
   │                                                                       │
   │   • Where does the from-scratch model STOP being viable?              │
   │       - Anywhere that depends on instruction-tuning, world            │
   │         knowledge, or multi-step reasoning. (Spoilers: most           │
   │         interesting tasks.)                                           │
   │       - The capstone uses a pretrained model (Llama / Qwen) for       │
   │         this reason. The from-scratch model lives on as a             │
   │         comparison baseline — you can swap it in via                  │
   │         `LocalTransformerBackend` and feel the gap.                   │
   │                                                                       │
   └───────────────────────────────────────────────────────────────────────┘
```

The post-mortem at the end of this module is the deliverable that compresses these observations into a permanent record. It's the actual learning outcome of the course.

```
   ┌──────────────────────────────────────────────────────────────────────┐
   │  THE ASSISTANT'S CALL FLOW                                            │
   └──────────────────────────────────────────────────────────────────────┘

      Assistant.chat(user_message)
          │
          ├── 1. format conversation history       ── Conversation
          │       "User: ...\nAssistant: ..."
          │
          ├── 2. retrieve k chunks (optional)      ── DenseRetriever (M17)
          │       "Context from documents:\n..."
          │
          ├── 3. compose contextualized message
          │       <history>\n\n<context>\n\n<user_message>
          │
          ├── 4. agent.run(contextualized)         ── Agent (M19)
          │       │
          │       ├── plan?    backend.complete    ── Backend (M16)
          │       └── ReAct loop:
          │             complete → parse → dispatch → observe ──── repeat
          │                                          │
          │                                          ToolRegistry (M18)
          │
          ├── 5. record user + assistant in conversation
          │
          └── 6. return AssistantTurn
```

Every arrow is a module you built. The integration layer is small (~100 lines of new code in `Conversation` and `Assistant`); the substrate is everything else.

![Assistant.chat call flow — how the capstone assistant integrates conversation, retrieval, and the agent loop. A six-step flow annotated with which upstream module owns each piece. (1) `Assistant.chat(user_message)` enters from the CLI. (2) FORMAT CONVERSATION HISTORY: the `Conversation` (Module 20's new primitive) renders prior turns as `User: ...\nAssistant: ...`. (3) RETRIEVE k CHUNKS (optional): if `rag_enabled`, `DenseRetriever` (Module 17) embeds the query and returns top-k chunks formatted as a "Context from documents:" block. (4) COMPOSE CONTEXTUALIZED MESSAGE: history block + context block + the actual current question, in that order. (5) `agent.run(contextualized_message)` (Module 19) — planning step, then the ReAct loop with `backend.complete` calls (Module 16) and `dispatch_tool_call` against the registry (Module 18). (6) RECORD USER + ASSISTANT in conversation, return an `AssistantTurn`. A right-edge sidebar lists what each layer contributes: Conversation = inter-turn memory; Retriever = external document context; Agent = multi-step reasoning and tool use; Backend = model inference; ToolRegistry = actions outside the model. A bottom caption: every arrow is a module you built; Module 20 is the integration layer that composes them into a usable assistant.](20-capstone/Module20-Call.png)

*The architecture diagram for `Assistant.chat`. Each step in this flow corresponds to a method or class you've already built in earlier modules; Module 20's contribution is the `Conversation` and `Assistant` primitives plus the orchestration that threads them together. Tracing one user question through this diagram is the fastest way to see where the integration layer earns its keep.*

## The big idea

### Conversation is to Module 20 what Scratchpad was to Module 19

![Two-layer memory — scratchpad handles one task; conversation spans the session. Top: the Conversation (Module 20) — a list of user/assistant Messages stretching across the whole chat session, low-bandwidth, only the final-answer outputs are stored, used for inter-turn references like "do that again with X". Middle: the Scratchpad (Module 19) — a list of (thought, action, observation) records confined to a single agent.run, high-bandwidth, every reasoning step preserved, used for multi-step reasoning within ONE task. Right side: "what happens on each chat() call" walks through the lifecycle in five steps — read Conversation; create fresh Scratchpad; run agent (lots of scratchpad activity); drop Scratchpad at the end of the turn; append user + final-answer messages to Conversation. The model sees BOTH on every backend call inside step 3: the rendered conversation as part of the user_message, and the live scratchpad in the agent's prompt. They serve different purposes; they never get mixed. A bottom panel pins the takeaway: the model is still stateless — the prompt does the remembering.](20-capstone/Module20-Memory.png)

*The picture for the load-bearing memory architecture. Module 19's scratchpad is the agent's working memory inside one task; Module 20's conversation is the assistant's working memory across tasks. Mixing them is a bug magnet — the test `test_history_does_not_double_include_current_message` and the rule "scratchpad belongs to the agent, conversation belongs to the assistant" both come out of this separation.*

```
   ┌──────────────────────────────────────────────────────────────────────┐
   │   TWO LAYERS OF MEMORY                                                │
   ├──────────────────────────────────────────────────────────────────────┤
   │                                                                       │
   │   ┌─────────────────────────────────────────┐                        │
   │   │  Conversation (Module 20)               │  inter-turn            │
   │   │  user / assistant messages, across      │                        │
   │   │  the whole session                      │                        │
   │   ├─────────────────────────────────────────┤                        │
   │   │  Scratchpad (Module 19)                 │  intra-turn            │
   │   │  thought / action / observation,        │                        │
   │   │  within ONE Agent.run                   │                        │
   │   └─────────────────────────────────────────┘                        │
   │                                                                       │
   │   Each chat() call:                                                  │
   │     1. read all of Conversation                                      │
   │     2. spin up a fresh Scratchpad                                    │
   │     3. run agent (lots of scratchpad activity)                       │
   │     4. drop the Scratchpad                                           │
   │     5. append (user, final_answer) to Conversation                   │
   │                                                                       │
   │   The model sees BOTH on every backend call inside step 3:           │
   │   the rendered conversation as part of the user_message, and the     │
   │   live scratchpad in the agent's prompt. They serve different        │
   │   purposes; they don't get mixed.                                    │
   │                                                                       │
   └──────────────────────────────────────────────────────────────────────┘
```

Why two memories instead of one unified store? Different *scope* and different *consumers*:

- The scratchpad is high-bandwidth working memory during a single multi-step task. The model needs to see its own recent reasoning AND tool observations to keep on track. Old reasoning is relevant; old observations are relevant.
- The conversation is the user-facing chat history. The model needs to know "the user said X two turns ago" but not "the agent called read_file three times during turn 2." That's low-bandwidth — only the final-answer outputs matter.

Mixing them is tempting but a bug magnet: showing the model its own prior tool calls from a finished agent run pollutes the new turn's scratchpad with stale, no-longer-relevant action records. The model is more likely to repeat the old approach instead of considering whether it still applies.

### The contextualized message: where the integration happens

```
   ┌──────────────────────────────────────────────────────────────────────┐
   │   PROMPT THE AGENT SEES                                               │
   ├──────────────────────────────────────────────────────────────────────┤
   │                                                                       │
   │   Question: Previous conversation:                                   │
   │   User: hi                                                           │
   │   Assistant: hello                                                   │
   │   User: what's the weather?                                          │
   │   Assistant: I don't have a weather tool yet.                        │
   │                                                                      │
   │   Context from documents:                                            │
   │   [1] (source: docs/weather.md)                                      │
   │   The weather tool is in g2c/tools/builtins.py and uses ...          │
   │                                                                      │
   │   how do I add a weather tool?    ← the actual current question     │
   │                                                                       │
   │   Thought:                          ← model's next turn starts here  │
   │                                                                       │
   └──────────────────────────────────────────────────────────────────────┘
```

The "Question:" prefix is the agent's standard prompt template (Module 19). Everything from "Previous conversation:" through the actual question is what the assistant assembled. The agent treats the whole block as one user message; the conversational structure is a model-friendly hint at the structure, not a parser-enforced contract.

### The eval-as-regression-gate pattern

```
   ┌──────────────────────────────────────────────────────────────────────┐
   │   THE GATE                                                            │
   ├──────────────────────────────────────────────────────────────────────┤
   │                                                                       │
   │   cases = [                                                          │
   │       EvalCase(name="arith",  question="2+2?",                       │
   │                expected_substring="4",                               │
   │                expected_tool="calculator"),                          │
   │       EvalCase(name="capital", question="capital of France?",        │
   │                expected_substring="Paris"),                          │
   │       EvalCase(name="search",  question="what's in foo.md?",         │
   │                expected_tool="read_file", rag=True),                 │
   │       ...                                                            │
   │   ]                                                                  │
   │                                                                      │
   │   report = run_evaluation(assistant, cases)                          │
   │   assert report.pass_rate >= 0.8                                     │
   │                                                                      │
   │   ───────────────────────────────────────────────────────            │
   │                                                                      │
   │   Three checks per case:                                             │
   │     ✓ run produced a final_answer (not max_steps / duplicate)        │
   │     ✓ final_answer contains expected_substring (case-insensitive)    │
   │     ✓ at least one step used expected_tool (if specified)            │
   │                                                                      │
   │   Run it after every config change. Refuses to merge a               │
   │   change that drops the rate. Cheap. Keep it short — 5-15            │
   │   cases is right.                                                    │
   │                                                                      │
   └──────────────────────────────────────────────────────────────────────┘
```

This is *not* the kind of eval Module 15 built — those are research-grade benchmarks measuring properties of the model. This is a regression gate measuring properties of the *assistant* (the integration). Both have a place; don't conflate them. The Module 15 eval tells you "the model is 73% accurate on MMLU-style questions"; the Module 20 eval tells you "your latest prompt edit didn't break the calculator path."

### The unified assistant interface

```
   ┌─────────────────────────────────────────────────────────────────────┐
   │   g2c/assistant/  PUBLIC API                                          │
   ├─────────────────────────────────────────────────────────────────────┤
   │                                                                       │
   │   AssistantConfig(name, max_steps, plan, loop_detection,              │
   │                   halt_on_stuck, max_new_tokens, temperature,         │
   │                   top_k, top_p, rag_enabled, rag_k,                   │
   │                   max_history_messages, scratchpad_max_chars)         │
   │     all the knobs in one dataclass                                    │
   │                                                                       │
   │   Message(role, content)                                              │
   │     one user or assistant utterance                                   │
   │                                                                       │
   │   Conversation(messages=None, max_messages=None)                      │
   │     .add_user(content), .add_assistant(content)                       │
   │     .clear(), .messages, .last_user_message()                         │
   │     .format_for_prompt() — multi-turn analog of Scratchpad.render     │
   │                                                                       │
   │   AssistantTurn(user_message, final_answer, agent_run,                │
   │                 retrieved_context, contextualized_message,            │
   │                 metadata)                                             │
   │     one chat() call's full record                                     │
   │                                                                       │
   │   Assistant(backend, registry, *, config, retriever,                  │
   │             conversation, agent)                                      │
   │     .chat(user_message, *, use_rag=None) → AssistantTurn              │
   │     .reset(), .conversation, .agent, .turns                           │
   │                                                                       │
   │   EvalCase(name, question, expected_substring, expected_tool, rag)    │
   │   EvalCaseResult(case, passed, final_answer, failure_reason, turn)    │
   │   EvalReport(results)                                                 │
   │     .n_total, .n_passed, .n_failed, .pass_rate, .failures, .summary() │
   │                                                                       │
   │   run_evaluation(assistant, cases, *, reset_each=True) → EvalReport   │
   │                                                                       │
   │   run_cli(assistant, *, prompt="? ", inp=None, out=None)              │
   │     interactive REPL with /help, /clear, /history, /tools, /config,   │
   │     /save <path>, /exit                                               │
   │                                                                       │
   │   AssistantError                                                      │
   │     internal exception type                                           │
   │                                                                       │
   └─────────────────────────────────────────────────────────────────────┘
```

Total scaffolded code: roughly 50 lines across two method bodies. The rest is integration plumbing — implemented because the wiring isn't the lesson; the architecture is.

## Concepts to internalize

- **The capstone is integration, not invention.** No new algorithms here. The conversation primitive is essentially Module 19's scratchpad scaled up one level. The assistant is essentially "agent.run, but with context threaded through it." The eval is essentially "pytest, but the assertions check answer-text and tool-call traces." Once you accept that the work is *composing*, the module's shape becomes obvious.

- **Memory layers must not mix.** Scratchpad is per-task. Conversation is per-session. Mixing them produces an agent that re-tries old tool calls from finished tasks. The clean separation isn't aesthetic; it prevents a category of bugs.

- **Failed runs still need a coherent log.** If `agent.run` doesn't produce a `final_answer`, the assistant records a placeholder in the conversation so the next turn's history is coherent. The `AssistantTurn.final_answer` stays `None` so machine-readable consumers (the eval harness) can detect the failure unambiguously. Two channels, two purposes.

- **The eval gate runs in seconds and refuses regressions.** The point is not to measure how good the assistant is in absolute terms; it's to ensure that the next thing you change doesn't break the things that worked. Cheap. Run it often. 5-15 cases is the right ballpark; more cases dilute the signal because you stop paying attention to which specific cases regressed.

- **Prefix-style RAG is a real architectural commitment.** It means the model never decides whether to retrieve. That's a feature (predictability, lower latency) and a constraint (the model can't "give up retrieval" when the corpus has nothing relevant). The exercise on tool-style RAG makes this tradeoff concrete.

- **The post-mortem is the actual deliverable.** All 19 prior modules had a code deliverable. This one's deliverable is a *document* — `docs/capstone-postmortem.md` — that articulates what each layer does, where it earns its keep, and where it breaks. The code is the substrate that makes a thoughtful post-mortem possible. Plan accordingly.

- **The from-scratch model isn't viable for the chat use case.** Your ~10M-param Module 10 model can write fluent-ish prose but doesn't follow chat instructions, doesn't reason multi-step, and doesn't know world facts. The capstone uses a pretrained 7-8B model (Llama, Qwen) for this reason. The from-scratch model is a *baseline*, not the production engine. Articulating this gap precisely — where exactly does it break, with concrete examples — is half the post-mortem.

- **Most "agent failures" are integration failures.** When the assistant gets a question wrong, the bug is rarely in the agent loop itself. It's almost always upstream: the wrong tool was registered, the wrong chunks were retrieved, the conversation history truncated the relevant message, the prompt template was unclear. Debug by checking `AssistantTurn` fields in order: `retrieved_context`, `contextualized_message`, `agent_run.steps`. The trail of what the assistant *saw* is more useful than what the model *did*.

## What you'll build

Package: `g2c/assistant/`

```python
# config.py
class AssistantError(Exception): ...                              # implemented

@dataclass
class AssistantConfig:                                            # implemented
    name: str = "g2c-assistant"
    max_steps: int = 8
    plan: bool = True
    loop_detection: bool = True
    halt_on_stuck: bool = False
    max_new_tokens: int = 512
    temperature: float = 0.2
    top_k: int | None = None
    top_p: float | None = None
    rag_enabled: bool = True
    rag_k: int = 5
    max_history_messages: int | None = 20
    scratchpad_max_chars: int | None = None


# conversation.py
@dataclass(frozen=True)
class Message:                                                    # implemented
    role: str
    content: str

class Conversation:                                               # implemented
    def __init__(self, messages=None, *, max_messages=None): ...
    def add_user(self, content) -> Message: ...
    def add_assistant(self, content) -> Message: ...
    def clear(self) -> None: ...
    @property
    def messages(self) -> list[Message]: ...
    def last_user_message(self) -> Message | None: ...
    def format_for_prompt(self) -> str:                           # SCAFFOLDED
        ...


# assistant.py
@dataclass
class AssistantTurn:                                              # implemented
    user_message: str
    final_answer: str | None
    agent_run: AgentRunResult
    retrieved_context: str
    contextualized_message: str
    metadata: dict

class Assistant:
    def __init__(self, backend, registry, *,                      # implemented
                 config=None, retriever=None,
                 conversation=None, agent=None): ...

    def reset(self) -> None: ...                                  # implemented
    def _maybe_retrieve(self, query, *, use_rag) -> str: ...      # implemented
    def _build_contextualized_message(...) -> str: ...            # implemented

    def chat(self, user_message, *,                               # SCAFFOLDED
             use_rag=None) -> AssistantTurn:
        ...


# eval.py
@dataclass(frozen=True)
class EvalCase:                                                   # implemented
    name: str
    question: str
    expected_substring: str | None = None
    expected_tool: str | None = None
    rag: bool | None = None

@dataclass(frozen=True)
class EvalCaseResult: ...                                         # implemented

@dataclass
class EvalReport: ...                                             # implemented

def run_evaluation(assistant, cases, *,                           # implemented
                   reset_each=True) -> EvalReport: ...


# cli.py
CLI_HELP: str                                                     # implemented
def run_cli(assistant, *,                                         # implemented
            prompt="? ", inp=None, out=None) -> None: ...
```

Total scaffolded code: roughly 50 lines across two function bodies. The lesson is the architecture (where each layer fits, what flows between them); the orchestration is layout.

## How to run the tests

Tests live in `tests/test_assistant.py`. Initial state on `main`: 70 tests pass . 54 tests fail

```bash
pytest tests/test_assistant.py                          # all module-20 tests
pytest tests/test_assistant.py -x                       # stop at first failure
pytest tests/test_assistant.py -k Conversation          # conversation tests
pytest tests/test_assistant.py -k Chat                  # the orchestration
pytest tests/test_assistant.py -k Eval                  # the regression gate
pytest tests/test_assistant.py -k CLI                   # the CLI loop
pytest tests/test_assistant.py -k Integration           # full-pipeline smoke
pytest tests/test_assistant.py -v                       # verbose
```

## Exercises

These exercises require Ollama running with a tool-calling-capable chat model:

```bash
ollama pull llama3.2:3b            # tool-calling enabled, fast on M1+
# or
ollama pull qwen2.5:7b             # also good; better on multi-step
ollama serve
```

Llama 3.2:3b is the fastest reasonable choice; Qwen 2.5:7b handles multi-step reasoning notably better at the cost of latency. Both follow ReAct format reliably given the prompt template.

1. **Wire up the assistant.** Build a simple integration script in `notebooks/20-capstone.ipynb`:

   ```python
   from g2c.assistant import Assistant, AssistantConfig
   from g2c.inference import OllamaBackend
   from g2c.tools import (
       ToolRegistry, make_calculator,
       make_read_file, make_run_python,
   )
   
   backend = OllamaBackend("llama3.2:3b")
   registry = ToolRegistry([
       make_calculator(),
       make_read_file(root=Path("data/")),
       make_run_python(),
   ])
   assistant = Assistant(backend, registry, config=AssistantConfig())
   
   turn = assistant.chat("What's 47 * 23?")
   print(turn.final_answer)
   ```

   Run 5-10 questions of varying complexity. Note where it works and where it doesn't.

2. **The eval gate.** Author 5-10 `EvalCase`s covering the assistant's main use cases (arithmetic, file reading, Python execution, a couple of corpus questions). Save them in `notebooks/20-eval-cases.py`. Run `run_evaluation(assistant, cases)` and report the pass rate. Now make a small change to the system prompt or sampling temperature, re-run, and see whether the gate detects the change. The point: this is your regression test going forward.

3. **Multi-turn calculator.** Have a 5-turn conversation:
   - "What's 7 * 8?"
   - "Multiply that by 3."
   - "Now subtract 100."
   - "Convert to a percentage of 200."
   - "Show me the final number."
   
   Does the model correctly reference "that" in turn 2? Does conversation memory let it do this? Try with `max_history_messages=2` and observe where it fails.

4. **Add RAG.** Index the course's own `docs/` directory using Module 17. Plug the retriever into the assistant. Ask 5 questions whose answers are in the docs ("what does Module 7 build?", "what's the calculator tool's interface?"). Compare with `rag_enabled=True` vs `rag_enabled=False`. Where does retrieval help? Where does it hurt (wrong chunks dragging the answer off-course)?

5. **Convert RAG to a tool (the design fork).** Write a `make_search` tool that takes a `query` argument, calls the retriever, and returns formatted chunks. Register it in the assistant's tool registry. Disable prefix-style RAG (`rag_enabled=False`). Now the model has to *decide* when to call `search`. Run the same 5 questions from Exercise 4. Compare:
    - Does the model call `search` when it should?
    - Does it skip it on questions where the answer is in its parametric memory?
    - How many extra steps does this add per turn?
   
   Write up your conclusion: when is prefix-style better, when is tool-style better?

6. **Compare backends.** Swap your tiny pretrained-from-scratch model (Module 10's checkpoint, via `LocalTransformerBackend`) into the assistant. Run the same 5-10 EvalCases from Exercise 2. Report:
    - Pass rate.
    - Where exactly does the from-scratch model fail? (Format compliance, tool selection, world knowledge?)
    - Side-by-side: a few exact transcripts where the pretrained model got it right and the tiny one got it wrong.
   
   This is the "where does from-scratch stop being viable" data point that goes in the post-mortem.

7. **Conversation truncation stress test.** Set `max_history_messages=4`. Have an 8-turn conversation where turn 8 references something from turn 1. Does the model lose the thread? At what `max_history` does the reference work reliably? Where would summarization (compress old turns into a single "previously: ..." message) help?

8. **Failure-mode catalog.** From your CLI sessions, identify five distinct failure modes the assistant exhibits. For each, write down:
    - **What happens.** A specific transcript.
    - **Where the bug is.** Was it the model? The retriever? The agent loop? The conversation history? Use the `AssistantTurn` fields (`retrieved_context`, `contextualized_message`, `agent_run.steps`) to localize.
    - **One mitigation.** A specific change to the prompt, the config, the retriever, or the tool registry that would help. Don't actually implement it — articulating the mitigation is the exercise.
   
   The catalog is direct input to the post-mortem.

9. **Build the deliverable CLI.** Add a small wrapper script `scripts/g2c-chat` (or similar) that wires everything up and calls `run_cli(assistant)`. Include:
    - Reasonable defaults loaded from a config file or constants.
    - A `--reset-conversation` flag.
    - A `--save-on-exit <path>` flag.
   
   Use the CLI for a full work session. Take notes on the friction.

10. **Write the post-mortem.** This is **the deliverable** of the course. Save as `docs/capstone-postmortem.md`. Required sections:

    - **What I built.** A short architecture description. What modules feed what. (Use the diagrams from this lesson page if useful.)
    - **What each layer does.** One paragraph each on the autograd, transformer, sampling, SFT/DPO, inference, RAG, tools, agent, and conversation layers. Plain language. The version of "what does an LLM do" you'd give a thoughtful colleague.
    - **What works well.** Three concrete capabilities the assistant has, with example transcripts.
    - **What breaks.** Three concrete failure modes, with example transcripts, hypothesized causes, and proposed mitigations (this is the catalog from Exercise 8).
    - **Where the from-scratch model stops being viable.** A precise characterization (Exercise 6's data points). Where exactly does it break? With what symptoms? Why?
    - **What I'd do next.** Three things you'd build/improve if you continued working on this. Persistent memory? Better retrieval? Multi-agent? Streaming? Production sandboxing? Pick the three with highest expected value and explain why.
    - **What I learned.** One paragraph. The candid version. The post-mortem is an honest document.
    
    Length target: 1500-3000 words. Don't pad it. The post-mortem is what you keep — the code is the substrate that produced the post-mortem.

## Pitfalls to expect

- **Adding the user message to the conversation BEFORE rendering history.** If you reorder steps in `chat`, the current question shows up twice — once as the latest line in "Previous conversation:" and once as the current question. The model gets confused and often answers the prior question instead. The test `test_history_does_not_double_include_current_message` pins this; if it fails, this is the bug.

- **Forgetting that `agent_run.final_answer` may be `None`.** When the agent times out or detects a duplicate action, it returns no answer. If you blindly do `conversation.add_assistant(agent_run.final_answer)`, you crash on a None content. The recipe records a synthetic placeholder; the `AssistantTurn` keeps `final_answer=None` so callers can detect the failure. Two channels.

- **Mixing scratchpad and conversation memory.** The scratchpad's tool-call records are *intra-task*; once the task ends, they're noise. If you accidentally feed prior agent runs' scratchpad content into the next turn's prompt, the model often re-tries old tool calls instead of doing fresh work. Keep them separate; the agent owns the scratchpad, the assistant owns the conversation.

- **Wrapping the retriever in `try/except` inside `chat`.** Tempting (the agent does this for tool errors), but wrong. A broken retriever is a config bug, not a model bug. Catching it silently means the user gets RAG-off behavior without knowing why retrieval is silently failing. Let it raise.

- **`max_history_messages` set too small.** If the cap is smaller than a typical exchange's important context, the model loses references that span more than a couple of turns. Default is 20; the right number depends on the use case. Not a hard rule; it IS a hard tradeoff (longer history = bigger prompt = slower / more expensive / closer to context cap).

- **`max_history_messages=None` (unlimited) on a long session.** Eventually the prompt exceeds the model's context window and either the backend errors or silently drops content. Production assistants either summarize old turns or use a vector store for long-term memory. We don't; the cap is the workaround.

- **RAG'ing on every turn, even when irrelevant.** The retriever fires for "hi how are you" the same as it fires for "what's in foo.md". The retrieved chunks are usually noise that the model has to ignore. Two mitigations: (a) `use_rag=False` per-call when you know retrieval doesn't apply; (b) tool-style RAG (Exercise 5) so the model decides. Both have downsides.

- **Eval cases that test the model rather than the assistant.** "What's the capital of Bolivia" tests the model's parametric knowledge. "What's the calculator tool's name" tests integration. The Module 20 eval is a regression gate for *your* code; lean toward integration questions, not factual recall ones. Module 15's eval is for the model's properties.

- **The eval substring check being too liberal.** `expected_substring="4"` matches "the answer is 4" and also matches "I gave you 4 reasons why this is hard". Pick substrings that are specific enough to be evidence of the right answer, not just any answer. `expected_substring="The answer is 4"` is overly strict; `expected_substring="42"` (for "what's 6*7") is just right.

- **The eval substring check being case-sensitive.** The assistant's response casing is non-deterministic ("Madrid" vs "MADRID" vs "madrid"). The harness lowercases both sides; if you change this, you'll get flaky failures. Don't change it.

- **CLI commands without leading slashes.** "exit" is a valid chat message; "/exit" is a command. The CLI's parser checks for a leading `/`. If you accidentally drop the slash, the CLI will dutifully send `"exit"` as a message and the model will probably respond with something polite about leaving.

- **Confusing `Message` (the conversation primitive) with `AgentStep` (Module 19's per-iteration record).** Both wrap (role/role-equivalent, content/content-equivalent), but at different levels. `Message` is one user/assistant exchange in the conversation. `AgentStep` is one ReAct iteration inside one agent run inside one chat turn. They live at different layers of the stack.

- **Not running the eval suite often enough.** The whole point of the regression gate is that it's cheap (seconds, no manual labor) and catches regressions before they pile up. If you only run it once a week, you'll see five regressions at once and not know which change caused which. Run it after every config / prompt edit. Make it part of your inner loop.

- **Conflating the assistant's `max_steps` with the conversation's length.** `max_steps` is per-chat-turn (the agent's loop cap). Conversation length is across-chat-turns (number of user/assistant exchanges in the history). Independent settings; don't unify them.

- **Forgetting to reset between eval cases.** `run_evaluation(assistant, cases, reset_each=True)` is the default; if you flip it to False, cases share conversation state and case 2's question can be "interpreted in light of" case 1's exchange. This is sometimes what you want (multi-turn eval) but is rarely what you want for a regression gate.

- **The CLI's `/save` not flushing before exit.** The CLI's standard pattern is "type `/save path`, then exit." If `/save` doesn't write synchronously, you lose the transcript on exit. The CLI uses `Path.write_text` which writes synchronously; if you replace it with anything that buffers, beware.

## Reading

The capstone has fewer required readings— most of the conceptual ground was covered earlier. The reading list here is for situating the capstone work in the broader landscape and informing the post-mortem.

Primary:

- **Re-read the original course brainstorm/syllabus.** Walk through the 20-week arc end-to-end. Each module's "Question it answers" should now have a concrete answer from your own implementation. Where do your answers diverge from the syllabus's framing? Those are the most interesting parts of your post-mortem.

- **Anthropic, "Building effective agents" (Dec 2024).** A practical taxonomy of agentic patterns: prompt chains, routing, parallelization, orchestrator-workers, evaluator-optimizer, ReAct. You read this for Module 19; re-read it now to assess where your assistant sits and which patterns are worth investing in next.

- **Karpathy, "Intro to Large Language Models" (talk; YouTube).** A 60-minute "what is an LLM" overview that maps cleanly onto the modules of this course. Watch it after finishing the course. The framing in the talk should now feel familiar — you've built each piece. Where does your version differ?

Secondary:

- **OpenAI, "GPT-4 system card" (2023).** A worked example of a real production assistant's capabilities + failure modes + safety mitigations. The structure (capability → eval → failure mode → mitigation) is roughly what your post-mortem should mirror, scaled down by ~100×.

- **Anthropic, "Claude's constitutional AI" (Dec 2022, and later iterations).** The "what we want the model to do" layer that your DPO module skirted. Worth understanding as the next conceptual step beyond preference tuning.

- **Yang et al., "SWE-bench" (ICLR 2024).** Real-world software-engineering agents. Skim §3 (the task structure) and §5 (the leaderboard). Useful as a "what does production agentic actually look like" reference; sobering about how far simple ReAct gets you (it doesn't, on hard tasks).

Optional:

- **Schick et al., "Toolformer" (2023).** The "function-calling fine-tuned" version of tool use, contrasted with the "prompt-engineered" version your assistant uses. Worth reading if you're considering whether to fine-tune for tools rather than prompt for them.

- **Park et al., "Generative Agents" (UIST 2023).** A simulation of LLM-driven agents with memory, planning, and reflection. The architecture diagram in §4 is one conceptual step beyond your assistant. Mostly interesting for the simulation; the architecture transfers.

- **Bommasani et al., "On the Opportunities and Risks of Foundation Models" (Stanford, 2021).** The "what are these things really" survey from the start of the foundation-model era. Slightly outdated but still the most thoughtful big-picture survey. Read after you've shipped your assistant; the framing will land differently than it would have at the start of the course.

## Deliverable checklist

- [ ] All tests in `tests/test_assistant.py` pass.
- [ ] Ollama running with a tool-calling chat model. `ollama list` shows `llama3.2:3b` (or your chosen model).
- [ ] Notebook: `notebooks/20-capstone.ipynb`. 
- [ ] **Eval suite**: `notebooks/20-eval-cases.py` (or similar) with 5-15 `EvalCase`s. Pass rate ≥ 80% on your assistant configuration.
- [ ] **Failure-mode catalog** (Exercise 8) in `docs/capstone-failure-modes.md`. Five failure modes, each with a transcript, localization, and proposed mitigation.
- [ ] **The post-mortem** (Exercise 10) at `docs/capstone-postmortem.md`. The actual deliverable. 1500-3000 words. The required sections are listed in Exercise 10.
- [ ] CLI wrapper script that you've actually used for a work session. Doesn't need to be polished; needs to exist.
- [ ] You can explain — out loud, without notes — why the conversation primitive is separate from the scratchpad and what would break if you merged them.
- [ ] You can explain — out loud, without notes — what `format_for_prompt` does NOT include and why.
- [ ] You can explain — out loud, without notes — why the eval harness doesn't catch retriever exceptions.
- [ ] You can explain — out loud, without notes — where exactly the from-scratch Module 10 model stops being viable as a chat backend, with one concrete example.

## M-series notes

The capstone inherits Module 16-19's compute footprint — there's no new compute work in Module 20 itself; the assistant is pure orchestration on top of components you've already characterized.

- **Per-turn latency is dominated by the agent's backend calls.** With `plan=True`, a typical turn is 1 (planning) + 1-5 (loop) + 1 (final answer) backend calls. With Ollama + Llama 3.2 3B at ~50 tokens/sec on M1, that's 5-15 seconds per turn. With Qwen 2.5 7B, 2-3× slower per call but better reasoning. On M1/16GB, Llama 3.2 3B is a comfortable default; Qwen 7B is borderline. M2+/32GB makes Qwen 7B comfortable.

- **RAG adds one embedding call per turn.** The retriever embeds the query, then searches the in-memory store. Both are sub-100ms with `OllamaEmbedder`; the retrieval overhead is negligible compared to the agent's backend calls.

- **Conversation history grows the prompt linearly.** With `max_history_messages=20` and average message length of 200 chars, you're adding ~4k chars (~1k tokens) per turn. Well within Llama 3.2's 128k context window. For longer sessions or smaller-context models, drop the cap.

- **The CLI's `/save` is essentially free.** A 100-turn transcript is well under 1 MB of JSON; `Path.write_text` finishes in milliseconds. No reason not to save aggressively.

- **The eval suite runs in seconds.** 10 cases × 5-15 seconds per case = 1-3 minutes per full eval run. Cheap enough to run after every config edit. Definitely not a "let's run this overnight" workflow — it's a "let's run this every change" workflow.

- **The from-scratch baseline is comfortable but not fast.** Your Module 10 ~10M-param model on MPS via `LocalTransformerBackend` is order-of-magnitude faster per call than Ollama (no IPC, no JSON serialization), but the responses are not useful for chat — the comparison is interesting, the substitution isn't viable. If you want the comparison data point for the post-mortem, run a single eval pass with the from-scratch backend; don't expect to use it for actual chat.
