# Module 18 — Tool use

> **Question this module answers:** *How does the model act outside itself?*

![Module 18 on one page: a four-panel circus map of the tool-use loop. PANEL 1 (top-left, "REGISTRATION"): a `ToolRegistry` shelf holds four labeled drawers — `calculator` (with a parameters schema scroll: {"expression": "string"}), `read_file` ({"path": "string", "max_chars": "integer"}), `web_search` ({"query": "string"}), `run_python` ({"code": "string"}). Each drawer's front shows the tool's description in plain English. PANEL 2 (top-right, "PROMPT TIME"): the registry's tools render into a system prompt block: "Tools available:\n- calculator: Evaluate an arithmetic expression...\n- read_file: Read a UTF-8 text file..." plus a "Call format: <tool_call>{...}</tool_call>" footer. The user message gets appended below: "User: What's the population of Madrid times 7?" The model reads it. PANEL 3 (bottom-left, "PARSE & DISPATCH"): the model's completion contains a `<tool_call>{"name": "calculator", "arguments": {"expression": "3220000 * 7"}}</tool_call>`. A `parse_tool_calls` machine extracts it; a `validate_arguments` checker stamps it OK; the dispatcher looks up "calculator" in the registry and invokes it. The result `22540000` becomes a `ToolResult`, formatted as `<tool_result name="calculator" id="call_0_xxxx">22540000</tool_result>` and appended back to the prompt. PANEL 4 (bottom-right, "FEEDBACK & STOP"): the model sees the tool result, decides it has enough information, and emits a final answer (no more `<tool_call>` blocks). The loop returns a `ToolRunResult` with `final_answer`, `steps` (every back-and-forth), `stopped_reason="no_more_calls"`. A right-edge sidebar lists key concepts: tool schemas (JSON-schema-lite), the `<tool_call>` format, the parse → dispatch → feedback loop, error surfacing as `<tool_error>` tags so the model can recover, max_steps as the safety net. Bottom caption: "Module 17 gave the model EYES (retrieval); Module 18 gives it HANDS (tools). Module 19 will give it INTENT (planning loops)."](18-tools/Module18-Hero.png)

Tool use is the smallest possible architecture for "let the model affect the outside world." Define a tool a JSON, splice the schemas into the prompt, parse tool call blocks from the model's output.

---
## Before you start

* *Finish* `g2c/inference` from [[16-inference]] — the tool-use loop calls `backend.complete(...)` to produce each model turn
* *Configure* a ProdLM backend from [[16-inference]] — tool calling works best with an instruction-tuned model that already understands structured tool-call formats
* *Refresh* JSON objects and basic regex — tool calls are JSON blocks extracted from model text
* *Skim* Python AST basics if you have not used `ast.parse` before — the calculator tool uses an AST whitelist instead of `eval`

---
## Where this fits in

Module 16 built the interface we use to interact with the model. Module 17 moved the input to the model beyond just the user prompt. This module will enhance how we process the output of the model, and therefore the capabilities of the assistant system.

Up through Module 17, we've built the model and its surrounding assistant system to be *knowledgeable*. The metric has been answering questions correctly. But knowledge isn't enough for an assistant — there are tasks that fundamentally require *action*:

```
   ┌──────────────────────────────────────────────────────────────────────┐
   │  WHAT EVEN A 7B MODEL HANDED THE RIGHT CONTEXT CAN'T DO              │
   ├──────────────────────────────────────────────────────────────────────┤
   │                                                                      │
   │   • Arithmetic on numbers it didn't memorize:                        │
   │       - "what's 3220000 * 7?" — it'll guess wrong half the time      │
   │       - even Q4-quantized 70B models get long multiplications wrong  │
   │                                                                      │
   │   • Read a file you point it at:                                     │
   │       - "summarize this 50-page PDF" — it can't *open* the PDF       │
   │       - even with the bytes pasted in, you've burned 50 pages of ctx │
   │                                                                      │
   │   • Run code:                                                        │
   │       - "compute the eigenvalues of this matrix" — needs numpy       │
   │       - "is this regex correct?" — needs to actually run the regex   │
   │                                                                      │
   │   • Look up information not in its training data:                    │
   │       - "what's today's date?" — model frozen at training cutoff     │
   │       - "current price of AAPL?" — needs the live web                │
   │                                                                      │
   │   • General assistant like tasks:                                    │
   │       - "set a reminder for tomorrow?" — no ability to schedule.     |
   │       - "book a flight for next week?" — needs external access       │
   │                                                                      │
   │   These are the gaps tool use fills. Not by training the model on    │
   │   more data, not by retrieving more text — by giving it a *button*   │
   │   it can press to make something happen.                             │
   │                                                                      │
   └──────────────────────────────────────────────────────────────────────┘
```

At the end of the day, LLMs can only output *text*. So to give an assistant the ability, and not just answer questions, we need a structured way to turn text output into action. This module introduces a way to do exactly that.

## The big idea

To go from model completions to external actions, we rely on the **tool**. The system that wraps tool usage around a model is the **tool harness**. Each tool in the harness represents a separate type of external action. For example there might be a `web_search` tool, a `schedule` tool and a `run_python` tool.

Each tool in the harness is made up of two components:

1. **Tool specification**. This is what the model sees. Includes name, plain language description, and a structured parameter description. The specification informs the model when, why and how to use the tool.
2. **Tool callable**. This is the external software that the tool harness runs when the model calls the specific tool. For example for web search the callable is responsible for actually making the queries to the search engine and returning the results.

With that framework, the assistant system has everything it needs to support external actions with arbitrary tools. While models are on the tool specification convention, the individual tools are *modular*. The model doesn't have to learn the individual tools ahead of time. All we have to do to add a new tool is conform to the specification with enough descriptiveness that the model can infer at prompt time.

The typical flow in a tool call involves a three way interaction between the tool harness, the model and the external callable. At a high level it looks something like this:

1. `weather_forecast` tool is registered with the tool harness 
2. At prompt time, the harness injects  the `weather_harness` tool specification (along with all other active tools) into the system prompt. 
3. The model sees a list of tool specifications and a user prompt: "*daily high San Francisco*"
4. The model emits a completion with a formatted tool call: `weather_forecast("San Francisco", "today")`
5. The tool harness extracts the structured call, and dispatches it to the `weather_forecast` callable.
6. The callable queries AccuWeather and returns a result.
7. The harness sends the tool call result back to the model.
8. The model now sees the user query and the tool result, and answers the user's original question: "*The high today in San Francisco is 67F*"

One important framework that's important to internalize is that this workflow means **assistant turns** are no longer synonymous with a single inference call to the model. 

### Tool call format

For the tool harness and model to coordinate, it's essential that they're aligned on the *exact* format for tool calls and results. If they're not the model will not emit tool calls in a way that's recognized by the harness, and the harness will not return tool results that are understand by the harness.

The exact format of the tool call and result doesn't actually matter. What matters is that the model is post-trained with high quality data to learn the exact format. This is essentially the same approach we used in [[13-sft]] post-training to teach the model the exact assistant role formatting.

For tool call protocols, three competing conventions have emerged:

- **JSON in tagged delimiters** (`<tool_call>{json}</tool_call>`). What this module uses. Easy to parse with a regex.
- **OpenAI-style structured output**. The model emits a special `tool_calls` array as part of its response *outside* the prose. Cleaner to parse but only available on models trained for it.
- **Python-syntax calls** (`<|python_tag|>expr`). Llama 3.2's "ipython" mode. Looks like Python. We don't use this; the JSON-block approach generalizes better.

In this course we'll use the tagged JSON convention, which will emit blocks that look like this:

```
<tool_call>{"name": "calculator", "id": 123, "arguments": {"expression": "2 + 2"}}</tool_call>
```

Why JSON inside? Because JSON has a `dict` and tools usually need named arguments. Positional arguments work for one-arg tools; but don't generalize to more complex cases. JSON is the lowest-friction option that's universally well-tokenized.

Tool calls are what the model writes and the harness reads. After the tool call completes, tool results are what the harness writes and the model reads back. They follow a similar formatting convention:

```
  <tool_result name="calculator" id="call_123">
  391
  </tool_result>

  OR

  <tool_error name="calculator" id="call_123">
  missing required argument: expression
  </tool_error>
```

A few important things to note. First tool results are freeform text, they don't have to be formatted in JSON. Because they're being read by a language model (instead of deterministic harness software), this is fine. The model will know how to interpret the text.

Second tag distinction matters. `<tool_result>` for success, `<tool_error>` for failures. Without the distinction, the model often parrots an error string back as if it were a successful answer.

Finally the tool harness returns the ID that correlates with the original tool call. This is important because completions can potentially include *multiple* tool calls. Without the ID, the model has no way of knowing which result matches to which tool call.

### The feedback contract

![From model text to tool result. A four-stage pipeline drawn left-to-right. (1) PARSE: the model's completion text contains `<tool_call>{"name": "calculator", "arguments": {"expression": "3220000 * 7"}}</tool_call>`. A regex extracts the block; `json.loads` parses the body; a shape check rejects non-dict top-level, missing `name`, empty name. Output: a `ToolCall(name, arguments, call_id)`. Permissive: malformed blocks are silently skipped. (2) VALIDATE: the JSON-schema-lite validator type-checks each argument against the tool's `parameters` schema — `string`, `integer` (rejecting `bool`), `number`, `boolean`, `array`, `object`. Required keys must be present; unknown extra keys are rejected. Strict: failures surface as `ToolError`. (3) DISPATCH: the registry looks up the tool by name; if missing, return `ToolResult(is_error=True, output="unknown tool: ...")`. If validation failed, same. If the tool's `func` raises at runtime, catch the exception and wrap as `ToolResult(is_error=True, output=f"{type(e).__name__}: {e}")`. Never raise. (4) FEEDBACK: format successful results as `<tool_result name="..." id="...">22540000</tool_result>` and errors as `<tool_error>` blocks, append back to the transcript. The model reads the feedback and either calls another tool or emits a final answer. A "common failure paths" panel below: malformed JSON → silently skipped; missing required arg → `<tool_error>`; wrong type → `<tool_error>`; tool raises → `<tool_error>`; unknown tool → `<tool_error>`. Errors are conversation, not crashes.](18-tools/Module18-Parse.png)
*The full tool call flow in one diagram.*

Each step in the pipeline has a precise responsibility, and each responsibility can fail in specific ways. A robust tool harness must gracefully handle, recover from, and surface errors:

```
   ┌─────────────────────────────────────────────────────────────────────────┐
   │   FAILURE-MODE TABLE                                                    │
   ├─────────────────────────────────────────────────────────────────────────┤
   │                                                                         │
   │   Step       What can go wrong          How we surface it               │
   │   ────       ────────────────────       ─────────────────────           │
   │   parse      malformed JSON              skip block silently            │
   │              non-dict top-level          skip block silently            │
   │              missing "name" key          skip block silently            │
   │              non-str / empty name        skip block silently            │
   │                                                                         │
   │   validate   non-dict arguments          ToolError                      │
   │              missing required key        ToolError                      │
   │              unknown extra key           ToolError                      │
   │              wrong type for value        ToolError                      │
   │                                                                         │
   │   dispatch   unknown tool name           ToolResult(is_error=True)      │
   │              validation ToolError        ToolResult(is_error=True)      │
   │              tool raised at runtime      ToolResult(is_error=True)      │
   │                                                                         │
   │   loop       no more tool calls          stopped_reason="no_more_calls" │
   │              max_steps hit               stopped_reason="max_steps"     │
   │                                                                         │
   └─────────────────────────────────────────────────────────────────────────┘
```

The model may not call the tool correctly the first time. A natural response is to **retry** with an attempted correction. This is possible because the tool harness returns errors as text that the model can process, instead of raising errors in the harness itself. Without that the harness would just crash on the first error. 

Even if the tool calls succeed, the model may want to initiate another round of tool calls based on the results of the first. For example if the user asks "*what time does the best museum in Paris open*", the first turn might call `web_search("best museum in Paris")` and based on that result the second turn might call `web_search("opening hours Louvre")`.

Therefore the tool harness has to handle multiple rounds of **tool-use steps** on a single query. Which means we need to know *when to stop.* Our harness supports two stopping conditions:

1. Zero tool_calls in the last completion. It's an oddly minimal contract — the model decides when it's done by simply not emitting another `<tool_call>`. The alternative (an explicit "DONE" sentinel) is fragile; instruction-tuned models reliably stop calling tools.
2. We reach a `max_steps` threshold. Without a cap, a confused model can loop forever, eventually overflowing the context. 

### Safe evaluation by construction

![Safe calculator — AST whitelist, not eval(). A four-stage pipeline. (1) EXPRESSION INPUT: a string like `"2 + 3 * 4"` from the model. (2) PARSE TO AST: `ast.parse(expr, mode="eval")` produces an `ast.Expression` whose `.body` is a tree of nodes — `BinOp`, `Constant`, `UnaryOp`. (3) WALK & VALIDATE (whitelist): a recursive walker visits each node and checks its type against an explicit allowlist. Allowed nodes: `Constant` (numeric only — booleans rejected), `BinOp` (with `Add`/`Sub`/`Mult`/`Div`/`Mod`/`Pow`/`FloorDiv`), `UnaryOp` (with `UAdd`/`USub`), and parenthesization. Rejected nodes: `Name` (variable references), `Call` (function calls), `Attribute` (e.g., `(1).bit_length()`), `Subscript` (e.g., `[1,2,3][0]`), `Compare`, `Lambda`, `If`, `Import` — refused by name, not by pattern matching the source string. (4) EVALUATE: only after the whole tree validates, recursively compute the numeric result. A "why not eval()?" panel below pins the rationale: `eval(expr, {"__builtins__": {}})` is famously not safe — `().__class__.__base__.__subclasses__()` and many other escapes break out of the restricted environment. Whitelist > denylist for safety. The "examples" panel shows safe expressions (`"2 + 3 * 4"`, `"-5 ** 2"`) passing and unsafe expressions (`"abs(-1)"`, `"__import__('os').system('rm -rf /')"`) being refused at the AST check, before any execution happens.](18-tools/Module18-AST.png)
*The AST-walker pattern generalizes — you'd build a safe regex evaluator, a safe template engine, a safe filter expression the same way: structurally bound the surface to exactly what you can vouch for.*

Unlike deterministic software, LLMs can behave in hard to anticipate ways. When we move from outputting text to directly acting, the blast radius of unpredictable behavior dramatically expands. When exposing tools to model generated input, we generally want to take a defensive posture and preemptively assume that anything the tool harness runs is potentially adversarial.

The specifics of defensive posutre vary widely based on specific tool. A `datetime` doesn't have much surface for abuse. A `bash` tool that executes arbitrary system commands has a huge amount of risk. The risk can also be inverted. `web_search` probably can't do much from its own callable. But it could return **prompt injections** from the public Internet. Malicious instructions in the tool result that the model might read and try to follow. 

The calculator tool we're building in this module is a good case study for tool security. While arithmetic itself isn't risky, the calculator uses the python interpreter to process the text input. There is a real risk of injection of arbitrary python code. There are three approaches to managing risk here:

- **Restricted globals + `eval()`.** Famously not safe — Many escapes documented in the safe-eval literature. Don't.
- **AST whitelist.** Parse to AST, walk the tree, refuse every node not on the allowlist. What we do for the calculator. Structurally safe; the surface is what you explicitly admit.
- **Subprocess + sandbox**. Spawn a separate process with reduced privileges (`seccomp`, `nsjail`, Docker, etc.). What real production code-runners use. 

The calculator's safe-eval relies on "walking" the AST to check for anything that's *not* arithmetic:

```
   1. parse the expression to an AST
   2. walk the tree
   3. allow only: 
      * Constant (numbers), 
      * BinOp (+ - * / % ** //),
      * UnaryOp (+ -), 
      * parenthesization
   4. reject every other node type by name
```

Every other node type is refused. The surface is what we admit; nothing else gets through. This is structurally different from `eval(expr, {"__builtins__": {}})`, which is a denylist (try to take away dangerous capabilities) and which has been famously broken many times. Whitelist > denylist for safety.

```
   The "unsafe eval" that fails the AST check, with the failing node:

   "x + 1"              → Name(id='x')                 ← reject
   "abs(-1)"            → Call(func=Name('abs'))       ← reject
   "(1).bit_length()"   → Attribute / Call             ← reject
   "[1,2,3][0]"         → List, Subscript              ← reject
   "1 < 2"              → Compare                      ← reject
   "(lambda x:x)(1)"    → Lambda, Call                 ← reject
   "__import__('os')"   → Name(id='__import__'),Call   ← reject
```

Each is rejected at the AST walker, not because we pattern-matched the source string but because the AST node type isn't on the allowlist. The technique generalizes — you'd build a safe regex evaluator the same way (allow `Concat`, `CharClass`, `Repeat`; reject everything else).

The correct philsophy for tool safety is **permisstive parser, strict validator**. What this means is we expect models are going to emit noisy imperfectly formatted text. Bad json, missing fields, and malformed tool_call blocks are not security risks. If the parser we use to extract tool calls is overly strict, we are going suffer unnecessarily high tool call failures. *But* after the inputs are parsed, then we apply strict validation to what input the tool actually runs. 

## Concepts to internalize

- **A tool is a callable, but the model only sees the schema.** The model never executes Python. It emits text describing the call. The decoupling is what makes tool use safe(ish). Execution is the runtime's job.
- **Errors are conversation.** When a tool fails, the loop feeds the error back to the model and lets it try again. This single decision is responsible for most of the "robust to model mistakes" feeling.
- **Safe eval is a whitelist, not a denylist.** Allow only the AST nodes you can vouch for; refuse everything else. `eval()` with restricted globals is famously not safe; the AST-walker pattern is structurally bounded.
- **Schemas are tighter than docstrings.** Model output is much more reliable when the prompt includes a precise JSON schema than when it includes only a prose description. The schema gives the model a *shape* to fill in; prose gives it a vibe.
- **The loop's stop condition is "no more tool calls."** Instruction-tuned models reliably stop calling tools when they have enough context.
- **`max_steps` is a safety net, not a feature.** It exists because "the model loops forever" is a real failure mode. 
- **Fine-tuned tool calling is a free lunch.** Modern instruction-tuned open models emit `<tool_call>` blocks reliably given a tool-describing prompt. Pick a model with a known tool-calling format and use it.
- **The parser is permissive; the validator is strict.** The parser tolerates malformed blocks (silent skip) so the model isn't punished for occasional weirdness. The validator rejects malformed arguments (loud error) so the tool gets clean inputs. Different layers, different policies.

### What we don't cover

- **Function-calling fine-tuning.** Models like Llama 3.2 have been fine-tuned in post-training with tool-calling data — you get reliable JSON output from a properly-formatted prompt. Training your own tool-calling fine-tune would be a separate project. We rely on the model already having tool-calling instinct.
- **Real JSON Schema.** The full spec covers conditional schemas, references, format validators, and more. We implement the corner that gets used in practice. If you outgrow it, drop in the `jsonschema` library.
- **Production sandboxing.** Real code-execution tools run in Docker or a similar isolation layer. `subprocess.run` is fine for local pedagogy; it is NOT fine for a hosted service.
- **Streaming.** Real production tool-calling streams token-by-token, parses partial JSON, and starts dispatching as soon as a complete `<tool_call>` block is seen. We do the synchronous version. Conceptually identical.
- **Parallel tool execution.** Some agentic systems dispatch all tool calls in a turn concurrently with `asyncio.gather`. We dispatch sequentially. For tools whose `func` is fast (calculator, read_file), the difference is microseconds. For slow tools (HTTP search, run_python), parallelism matters in production but not for a teaching loop.
- **Tool-result truncation by content.** A real read_file tool detects giant files and summarizes; a real run_python tool truncates large stdout. We truncate by char count only — a starting point.

## What you'll build

Package: `g2c/tools/`

```python
# base.py
@dataclass(frozen=True)
class Tool:                                                       # implemented
    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., Any]

@dataclass(frozen=True)
class ToolCall:                                                   # implemented
    name: str
    arguments: dict[str, Any]
    call_id: str

@dataclass(frozen=True)
class ToolResult:                                                 # implemented
    call_id: str
    name: str
    output: str
    is_error: bool = False

class ToolError(Exception): ...                                   # implemented

@dataclass
class ToolStep:                                                   # implemented
    completion: str
    tool_calls: list[ToolCall]
    tool_results: list[ToolResult]
    inference: InferenceResult

@dataclass
class ToolRunResult:                                              # implemented
    user_message: str
    final_answer: str | None
    steps: list[ToolStep]
    stopped_reason: str
    metadata: dict[str, Any] = field(default_factory=dict)


# registry.py
class ToolRegistry:                                               # implemented
    def __init__(self, tools=None): ...
    def register(self, tool): ...
    def tools(self) -> list[Tool]: ...
    def names(self) -> list[str]: ...


# schema.py
def validate_arguments(tool, arguments) -> dict[str, Any]: ...
def render_tools_for_prompt(tools) -> str: ...                     # implemented


# parser.py
def parse_tool_calls(text) -> list[ToolCall]: ...          
def format_tool_results(results) -> str: ...                      # implemented

# builtins.py
def calculator_evaluate(expression: str) -> float: ...
def make_calculator() -> Tool: ...                                # implemented
def make_read_file(*, root) -> Tool: ...                          # implemented
def make_web_search(*, search=None) -> Tool: ...                  # implemented
def make_run_python(*, timeout=10) -> Tool: ...                   # implemented

# loop.py
def dispatch_tool_call(registry, call) -> ToolResult: ...         # implemented
def run_with_tools(                                               
    backend, registry, user_message, *, ...,
) -> ToolRunResult: ...
```

Total scaffolded code: roughly 60 lines across four function bodies. 

## How to run the tests

```bash
pytest tests/test_tools.py                          # all module-18 tests
pytest tests/test_tools.py -x                       # stop at first failure
pytest tests/test_tools.py -k Validate              # validator tests
pytest tests/test_tools.py -k Parse                 # parser tests
pytest tests/test_tools.py -k Calculator            # calculator tests
pytest tests/test_tools.py -k RunWithTools          # loop tests
pytest tests/test_tools.py -k Integration           # full-pipeline smoke
pytest tests/test_tools.py -v                       # verbose
```

## Exercises

These exercises require Ollama running with a tool-calling-capable chat model:

```bash
ollama pull llama3.2:3b           # tool-calling enabled
# or
ollama pull qwen2.5:7b            # also tool-calling
ollama serve
```

Llama 3.2 specifically is fine-tuned on tool-call data; it emits `<tool_call>` blocks reliably given a properly-formatted prompt. Qwen 2.5 also works but needs the prompt to include a clearer call-format reminder.

1. **Get the model to use the calculator reliably.** Wire `OllamaBackend("llama3.2:3b")` + a registry containing only `make_calculator()` into `run_with_tools`. Ask 20 arithmetic questions where the answer is easy for a tool but the model gets it wrong on its own (large multiplications, square roots of non-perfect-squares, expressions with deep parenthesization). Examples:
    - `"What's 3220000 times 7?"`
    - `"What's the result of (123 * 456 + 789) / 11?"`
    - `"What's 2^32?"`
    
   Report:
    - Tool-call rate (out of 20, how many called the calculator?)
    - Correctness rate when it called the tool.
    - Failure modes when it didn't call the tool. Is it confidently guessing wrong, or close-but-wrong?

2. **Multi-tool task: read + compute.** Wire `make_calculator()` + `make_read_file(root=Path("data/numbers/"))`. Drop a few text files into `data/numbers/`, each with a list of numbers. Ask the model: *"Read `data/numbers/foo.txt` and tell me the average."* The model should call `read_file`, then `calculator`, then answer. Try 5 such two-step tasks; report which steps the model gets right.

3. **Stress-test malformed outputs.** Some prompts cause models to emit malformed `<tool_call>` blocks — missing closing tag, JSON typos, mixing positional and named args. Run a few prompts that elicit these (e.g., asking for tools with very long descriptions, or with parameters whose names clash with Python keywords). Inspect the failure modes:
    - Does `parse_tool_calls` skip them (good)?
    - Does the model recover on the next step when it sees the empty step (or does it loop)?
    - When does `max_steps` kick in?
   
   Document the recovery patterns. The goal is to characterize "where does the parser tolerate and where does it just give up."

4. **`run_python` for a small data task.** Wire `make_run_python(timeout=10)` and `make_read_file(root=Path("data/"))`. Drop a CSV in `data/`, ask the model to "compute the mean of column X." It should `read_file` the CSV, then `run_python` to parse + average. Compare to letting the model do it with raw multiplication in its head — the python-tool version should be much more reliable.

5. **Add a custom tool.** Write a tool that does something specific to your work — query your local database, fetch from an internal HTTP API, run a domain-specific lint. Register it in a `ToolRegistry` and exercise it through `run_with_tools`. The point: you now have the ability to give the model arbitrary affordances. What does it look like to use them?

6. **The Toolformer-style ablation.** Repeat Exercise 1 with three configurations:
    - **No tool prompt.** Just `"You are a helpful assistant."` and the user question. Model uses only its own arithmetic.
    - **Tool prompt + tool available.** What this module builds.
    - **Tool prompt + tool *not* available.** The system prompt mentions a calculator but the registry is empty. The model still tries to call it; we see what happens (parsed call, "unknown tool" error, model recovery).
   
   Tabulate accuracy for each configuration. The gap between (a) and (b) is the tool's net win; the gap between (b) and (c) is the prompt's contribution alone.

7. **Add citation enforcement.** In Module 17 the model produced citations against retrieved chunks; here, the model produces tool results with `call_id`s. Modify `run_with_tools` (or write a wrapper) that checks: when the model emits a final answer that references "[1]" or "as the calculator showed," the corresponding tool call must actually exist in `steps`. Surface unverified citations as a warning. This is one half of "did the model actually use the tools" — citation grounding for tool-using assistants.

8. **Build the deliverable.** Write a small CLI that loops:

   ```python
   while True:
       q = input("? ")
       if not q.strip():
           break
       result = run_with_tools(backend, registry, q, max_steps=5)
       print(result.final_answer or "(stuck — max_steps hit)")
       print()
       print(f"({len(result.steps)} steps, "
             f"{sum(len(s.tool_calls) for s in result.steps)} tool calls)")
   ```
   
   Run a 10-question session covering: arithmetic, file reading, Python execution, ambiguous questions, questions with no good tool. Save the transcript.

9. **Write the post-mortem.** 3–4 paragraphs in `docs/tools-postmortem.md`:
    - **What you wired up.** Tools, model, max_steps, system prompt.
    - **What worked.** Tasks where the loop reliably succeeded.
    - **Where it broke.** Tasks where the model failed to call the right tool, called the wrong tool, or looped. Be specific.
    - **What you'd build next.** ReAct-style explicit Thought/Action/Observation turns? A planning step? Better recovery on bad calls? Multi-tool fan-out? The next investment, in 2-3 sentences.
   
   This is the actual deliverable. The pipeline code is the substrate; the *characterization* is what you keep going into Module 19's agent loop.

## Pitfalls to expect

- **Strict-mode "no extra keys" surprises.** Our validator rejects argument keys not in the schema's `properties`. JSON Schema's default is *additional properties allowed*; ours is *additional properties disallowed*. The reason: if the model invents a parameter, we'd rather catch it (and feed back an error) than silently drop it. If you wanted lenient behavior, the change is one line.

- **`json.loads` raises on trailing content.** `json.loads('{"a":1}xxx')` raises `JSONDecodeError`. The parser handles this by skipping the block; if you implement parser parsing differently and don't catch the exception, a malformed block will crash the loop.

- **Non-greedy regex matters.** Without `?`, `<tool_call>...A...</tool_call>...<tool_call>...B...</tool_call>` matches as one giant block. Always `re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)`. The `?` is the difference between "two calls" and "one call with garbled JSON."

- **`re.DOTALL` is required.** Without it, `.` doesn't match newlines, and a multi-line JSON body wouldn't be captured. Models often emit `<tool_call>\n{...}\n</tool_call>` with the JSON on its own line.

- **`call_id` collisions.** If you generate `call_id` purely from a counter (`call_0`, `call_1`, ...), two parser invocations on the same conversation produce ids that collide across turns. The recipe uses `call_{i}_{uuid4().hex[:8]}` — uniqueness within and across parses.

- **Forgetting `start=1` doesn't apply here.** The parser's `enumerate(...)` starts at 0; `call_0_...` is the first call's id. Unlike Module 17's citation indexing (where `[0]` would confuse the model), tool ids are internal — the model never sees the index, only the id string.

- **Calculator: rejecting `bool` constants explicitly.** `True + 1` parses as `BinOp(Constant(True), Add, Constant(1))`. The walker's `Constant` branch must reject `bool` before checking for `int`/`float`, because `isinstance(True, int)` is True. Same trap as the validator's bool-vs-int issue, in a different place.

- **Calculator: `MatMult` is not on the list.** Python's `@` operator parses to `ast.MatMult`, which is NOT in `_BINOPS`. If a student copies the operator list from elsewhere and includes `MatMult`, the calculator silently accepts `5 @ 3` — which raises `TypeError` at execution. The right behavior: refuse `MatMult` at the AST check, never reach execution.

- **The tool-calling format the model emits and what the parser expects must match.** Llama 3.2 emits `<tool_call>...</tool_call>`. Llama 3.2's "ipython" mode emits `<|python_tag|>...`. Qwen 2.5 emits `<tool_call>` mostly but sometimes drops the closing tag. If you see the model emit calls but the parser sees zero, the formats don't match — pick one and align both ends.

- **Forgetting the `Assistant:` marker after each turn.** The transcript grows like "user message → assistant turn 1 → tool results → assistant turn 2 → ...". After splicing tool results, the next prompt must end with `"\nAssistant:"` so the model knows it's its turn. Without it, the model often produces a `User:` block (continuing the wrong role) and the loop breaks down.

- **Passing `chunks` instead of `arguments` from the parser.** A common student bug: the parser builds a `ToolCall` from `obj["name"]` but accidentally uses `obj` itself as the arguments dict (instead of `obj["arguments"]`). The dispatcher then passes `{"name": ..., "arguments": {...}}` to the tool, which crashes. Read the JSON path carefully — `arguments` is one level down.

- **`subprocess.run` with `shell=True`.** If you pass `["python", "-c", code]`, you're safe. If you pass `f"python -c {code}"` and `shell=True`, you've introduced shell injection — the code is interpreted by the shell first. Always use the list form.

- **`run_python` cwd surprises.** The subprocess runs from the parent's cwd. If your test runs from one directory and your real session from another, the same code can produce different results. Pass `cwd=` explicitly if you care.

- **`max_steps` budget calibration.** Default is 5. If the model needs to call several tools in sequence, 5 might not be enough. If the model loops on bad calls, 5 might be too generous. Tune per-task; Module 19's agent will replace this with goal-tracking.

- **The model emits a final answer AND a tool call.** Some models emit "the answer is 42" followed by a `<tool_call>` block they didn't quite mean to send. The current loop reads the call as authoritative ("there are tool calls — keep going"). If you want "answer present means stop," that's a different policy — a wrapper on the parser, or a post-call check on the loop.

## Reading

Primary:

- **Schick, Dwivedi-Yu, Dessì et al., "Toolformer: Language Models Can Teach Themselves to Use Tools" (NeurIPS 2023).** The paper that named the genre. The key contribution is *bootstrapping* — using the model itself to label tool-call positions in pretraining text, then fine-tuning on those labels. Read §3 (the self-supervised data construction) — that's the durable idea. The empirical results in §4 are the canonical "tool-using LM beats vanilla LM on factual tasks" demonstration.
- **Anthropic, "Tool use" docs (claude.ai docs).** Practical, current. Walks through Claude's `<invoke>` format, schema requirements, multi-turn tool use, and parallel tool calls. Read alongside the OpenAI function-calling docs to see the two industry-standard formats.
- **OpenAI, "Function calling" docs (platform.openai.com).** The other industry standard. Read for contrast to Anthropic — same concept, different shape (structured `tool_calls` array vs XML-tagged JSON in prose). Either format works; the differences are surface-level.

Secondary:

- **Patil, Zhang, Wang, Gonzalez, "Gorilla: Large Language Model Connected with Massive APIs" (NeurIPS 2024).** Specialized fine-tuning for API selection at scale. Skim §3 — the construction of an API zoo and the bench against generalist models is the interesting bit. Useful as a "what does it look like when tool selection itself becomes the bottleneck" reference.
- **Yao, Zhao, Yu et al., "ReAct: Synergizing Reasoning and Acting in Language Models" (ICLR 2023).** The "Thought / Action / Observation" interleaving that we'll build in Module 19. Read it now to see what's coming; the loop in this module is the substrate ReAct sits on.
- **Anthropic, "Building effective agents" (Dec 2024).** A practical taxonomy of agentic patterns: prompt chains, routing, parallelization, orchestrator-workers, evaluator-optimizer, and the ReAct agent. Module 18's loop is the simplest agentic pattern; Module 19 builds toward the others.

Optional:

- **Qin, Liang, Ye et al., "ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs" (ICLR 2024).** Gorilla's spiritual successor — much larger API zoo, depth-first vs breadth-first search over API combinations. Skim if you want to see how production tool-using systems handle hundreds of tools.
- **Park, O'Brien, Cai et al., "Generative Agents: Interactive Simulacra of Human Behavior" (UIST 2023).** Uses tools and memory to simulate a small town of agents. Skim for the architectural overview — the way they decompose memory, planning, and reflection generalizes well beyond the simulation use case.
- **Liu, Li, Du et al., "AgentBench: Evaluating LLMs as Agents" (ICLR 2024).** A benchmark suite covering tool use, web browsing, OS interaction, and game-playing. Useful as a "what gets evaluated" reference if you want to systematically measure your agent's capability later.

## Deliverable checklist

- [ ] All tests in `tests/test_tools.py` pass: 155 tests, all green.
- [ ] Ollama running with a tool-calling-capable chat model. `ollama list` shows your chosen model.
- [ ] Notebook: `notebooks/18-tools.ipynb`. 
- [ ] **Tool-use post-mortem** (Exercise 9) in 3-4 paragraphs. The main deliverable.
- [ ] You can explain — out loud, without notes — why errors are surfaced as `ToolResult(is_error=True)` instead of raised exceptions.
- [ ] You can explain — out loud, without notes — why AST-walking is structurally safer than `eval()` with restricted globals.
- [ ] You can explain — out loud, without notes — what the loop's stop condition is and why "no more tool calls" works as a signal.
- [ ] You can explain — out loud, without notes — why the validator must reject `bool` when expecting `int` or `number`.

## M-series notes

This module is comfortable on every M-series Mac.j Practical considerations:

- **Inference happens via `OllamaBackend`** (or `LocalTransformerBackend` for the from-scratch model — but the from-scratch model isn't trained for tool calling, so it won't follow the schema). All Module 16 caveats apply: first call is slow, steady-state matches the model's parameter count.
- **Tool execution latency.** The calculator is microseconds; `read_file` is microseconds for small files; `web_search` depends on your backend (the stub is microseconds; a real DuckDuckGo / Tavily call is ~1 second); `run_python` is the slowest (subprocess startup + Python init is ~50–200ms on M-series).
- **Subprocess startup cost.** `subprocess.run([sys.executable, "-c", ...])` pays ~50-200 ms in Python startup. If `run_python` is called frequently, the total wall time is dominated by startup. A more advanced runner reuses a long-lived Python child process via `subprocess.Popen` + line-based protocol; out of scope here, but worth knowing about.
- **Context length.** Each step appends to the transcript. A 5-step run with verbose tool results can easily reach 4–8k tokens. Llama 3.2's 128k context is comfortable; smaller-context models would need careful pruning of past turns. Module 19 will introduce conversation memory management.
- **No special memory considerations for tool execution itself.** The tool runtime is pure Python plumbing. The model's inference is the memory-hungry part, and that's the same as Modules 16/17.
