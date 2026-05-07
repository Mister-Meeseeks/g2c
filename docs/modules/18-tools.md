# Module 18 — Tool use

> **Question this module answers:** *How does the model act outside itself?*

![Module 18 on one page: a four-panel circus map of the tool-use loop. PANEL 1 (top-left, "REGISTRATION"): a `ToolRegistry` shelf holds four labeled drawers — `calculator` (with a parameters schema scroll: {"expression": "string"}), `read_file` ({"path": "string", "max_chars": "integer"}), `web_search` ({"query": "string"}), `run_python` ({"code": "string"}). Each drawer's front shows the tool's description in plain English. PANEL 2 (top-right, "PROMPT TIME"): the registry's tools render into a system prompt block: "Tools available:\n- calculator: Evaluate an arithmetic expression...\n- read_file: Read a UTF-8 text file..." plus a "Call format: <tool_call>{...}</tool_call>" footer. The user message gets appended below: "User: What's the population of Madrid times 7?" The model reads it. PANEL 3 (bottom-left, "PARSE & DISPATCH"): the model's completion contains a `<tool_call>{"name": "calculator", "arguments": {"expression": "3220000 * 7"}}</tool_call>`. A `parse_tool_calls` machine extracts it; a `validate_arguments` checker stamps it OK; the dispatcher looks up "calculator" in the registry and invokes it. The result `22540000` becomes a `ToolResult`, formatted as `<tool_result name="calculator" id="call_0_xxxx">22540000</tool_result>` and appended back to the prompt. PANEL 4 (bottom-right, "FEEDBACK & STOP"): the model sees the tool result, decides it has enough information, and emits a final answer (no more `<tool_call>` blocks). The loop returns a `ToolRunResult` with `final_answer`, `steps` (every back-and-forth), `stopped_reason="no_more_calls"`. A right-edge sidebar lists key concepts: tool schemas (JSON-schema-lite), the `<tool_call>` format, the parse → dispatch → feedback loop, error surfacing as `<tool_error>` tags so the model can recover, max_steps as the safety net. Bottom caption: "Module 17 gave the model EYES (retrieval); Module 18 gives it HANDS (tools). Module 19 will give it INTENT (planning loops)."](18-tools/Module18-Hero.png)

*The whole module on one page. Tool use is the smallest possible architecture for "let the model affect the outside world." Define a tool with a JSON schema, splice the schemas into the prompt, parse `<tool_call>` blocks out of the model's output, validate arguments, dispatch the call, format the result back into the next prompt turn, and loop until the model stops calling tools. The agent loop with planning and scratchpad memory comes in Module 19; Module 18 builds the substrate.*

---
## Before you start

* *Finish* `g2c/inference` from [[16-inference]] — the tool-use loop calls `backend.complete(...)` to produce each model turn

---
## Prerequisites

Module 18 is the third leg of Phase V (assistant systems). Module 16 built the unified `Backend` interface; Module 17 used it to wire retrieval into the prompt. Module 18 wires *outgoing* affordances — the model gets a list of tools it can invoke, a structured way to invoke them, and a feedback channel for the results.

This module is short on math and long on plumbing. The whole content is:

- A small dataclass set (`Tool`, `ToolCall`, `ToolResult`, `ToolStep`) describing the contract.
- A registry that holds tools and renders them for the prompt.
- A JSON-schema-lite validator that type-checks arguments before execution.
- A regex-based parser that extracts tool calls from model text.
- A dispatcher that catches every failure mode and converts it to a model-readable error.
- A loop that ties the whole thing together.

There are four scaffolded methods. Three are short (each ~10–30 lines); the fourth is the loop (~30 lines). The lesson is in the *contract* between them — what each piece is responsible for, and what happens when one piece fails.

### Math

There isn't really any math in this module. The closest thing is:

- **Recursive AST evaluation in the calculator.** The safe-eval pattern is "parse to AST, walk the tree, refuse every node that isn't on the allowlist." This is how every safe expression evaluator works (Excel formula engines, sandboxed config languages, code-eval cells in notebooks). Knowing the pattern means you can build similar mini-languages without reaching for `eval()`.

### Computer science

- **Tool-call protocols.** Three competing conventions have emerged:
    - **JSON in tagged delimiters** (`<tool_call>{json}</tool_call>`). What this module uses. Llama 3.x and Qwen 2.5 emit this natively; Anthropic recommends XML-tagged variants for Claude. Easy to parse with a regex; the model has seen the format during instruction tuning.
    - **OpenAI-style structured output**. The model emits a special `tool_calls` array as part of its response *outside* the prose. The API surface is `response.tool_calls`, not part of the text body. Cleaner to parse but only available on models trained for it.
    - **Python-syntax calls** (`<|python_tag|>calculator(expression="2+2")`). Llama 3.2's "ipython" mode. Looks like Python; risk is that real Python code calls and fake function calls are visually indistinguishable. We don't use this; the JSON-block approach generalizes better.

  Pick one and stay consistent. The format the model has seen most often during pretraining/fine-tuning is the format it follows most reliably.

- **JSON Schema vs JSON-schema-lite.** Real JSON Schema is a sprawling spec with `$ref`, `oneOf`, `allOf`, conditional schemas, and a hundred validation keywords. Production tool-calling almost never uses any of that — the standard pattern is "object with primitive properties and a `required` list." We implement that subset. Total validator: ~30 lines. If you outgrow it, the upgrade is one import: `import jsonschema; jsonschema.validate(args, schema)`.

- **The `bool`-vs-`int` trap.** Python's `bool` is a subclass of `int`. `isinstance(True, int)` is `True`. If a JSON schema says `{"type": "integer"}` and the model emits `true`, a naive `isinstance(value, int)` accepts it — and the tool now gets `True` as its "count" parameter. The validator must reject `bool` explicitly when expecting numeric. This is the single most common subtle bug in JSON-schema-lite implementations.

- **Safe evaluation strategies.** Three approaches to "execute code from a possibly-adversarial source":
    - **Restricted globals + `eval()`.** Famously not safe — `().__class__.__base__.__subclasses__()` and many other escapes documented in the safe-eval literature. Don't.
    - **AST whitelist.** Parse to AST, walk the tree, refuse every node not on the allowlist. What we do for the calculator. Structurally safe; the surface is what you explicitly admit.
    - **Subprocess + sandbox**. Spawn a separate process with reduced privileges (`seccomp`, `nsjail`, Docker, etc.). What real production code-runners use. We use a plain `subprocess.run` for `run_python` — adequate for *local* pedagogy, NOT for a hosted service.

- **Error surfacing as data, not exceptions.** When a tool fails (unknown name, bad args, runtime error), the dispatcher returns a `ToolResult(is_error=True)` instead of raising. The loop formats it as `<tool_error>` and feeds it back to the model. The model can read the error and try again. A raised exception in the dispatcher would crash the loop and lose the conversation.

- **Tag distinction matters.** `<tool_result>` for success, `<tool_error>` for failures. Without the distinction, the model often parrots an error string back as if it were a successful answer ("The calculator returned: missing required arguments: ['expression']"). With explicit error tags, instruction-tuned models reliably treat them as recovery signals.

- **Why the loop has `max_steps`.** Without a cap, a model that gets confused about when to stop calling tools can loop forever — each step appends to the prompt, eventually overflowing the context. `max_steps=5` is the safety net. Module 19's agent uses smarter stop conditions (goal detection, plan completion); this module's contract is "5 steps, then we cut you off."

### Programming

- **`re` for the tag parser.** A single regex (`<tool_call>(.*?)</tool_call>` with `re.DOTALL`) is enough to extract every tool call in a completion. Non-greedy capture stops at the nearest closing tag.
- **`json` for the body.** Standard JSON parsing with `json.loads`. Skip blocks that don't parse; don't crash.
- **`ast` for the calculator.** `ast.parse(expr, mode="eval")` returns an `ast.Expression` whose `.body` is the expression tree. Walk the tree with `isinstance` checks.
- **`subprocess.run` for `run_python`.** With `timeout` and `capture_output=True`. The child process is its own world; we don't need to import the user's code into our process.
- **`pathlib.Path` + `.relative_to(root)` for the read_file sandbox.** Resolve paths to absolute, check that the resolved target is under the allowed root. The `.relative_to()` ValueError is the path-escape signal.
- **`@dataclass(frozen=True)`** for `Tool`, `ToolCall`, `ToolResult` — values, not handles.
- **`abc` is NOT used here.** Unlike Module 16's `Backend` and Module 17's `Embedder`, tools don't have an ABC. A tool is a `Tool` dataclass with a callable; subclassing buys nothing.

### What you can skip

- **Function-calling fine-tuning.** Models like Llama 3.2 have been fine-tuned with tool-calling data — you get reliable JSON output from a properly-formatted prompt. Training your own tool-calling fine-tune would be a separate multi-week project. We rely on the model already having tool-calling instinct from its pretraining.
- **Real JSON Schema.** The full spec covers conditional schemas, references, format validators, and more. We implement the corner that gets used in practice. If you outgrow it, drop in the `jsonschema` library.
- **Production sandboxing.** Real code-execution tools run in Docker, gVisor, Firecracker, or a similar isolation layer. `subprocess.run` is fine for local pedagogy; it is NOT fine for a hosted service.
- **Streaming.** Real production tool-calling streams token-by-token, parses partial JSON, and starts dispatching as soon as a complete `<tool_call>` block is seen. We do the synchronous version. Conceptually identical, ~3× more code.
- **Parallel tool execution.** Some agentic systems dispatch all tool calls in a turn concurrently with `asyncio.gather`. We dispatch sequentially. For tools whose `func` is fast (calculator, read_file), the difference is microseconds. For slow tools (HTTP search, run_python), parallelism matters in production but not for a teaching loop.
- **Tool-result truncation by content.** A real read_file tool detects giant files and summarizes; a real run_python tool truncates large stdout. We truncate by char count only — a starting point.

## Why we start here

Module 17 fixed the model's *knowledge* gap by retrieving relevant text into the prompt. But knowledge isn't enough for an assistant — there are tasks that fundamentally require *action*:

```
   ┌──────────────────────────────────────────────────────────────────────┐
   │  WHAT EVEN A 7B MODEL HANDED THE RIGHT CONTEXT CAN'T DO              │
   ├──────────────────────────────────────────────────────────────────────┤
   │                                                                       │
   │   • Arithmetic on numbers it didn't memorize:                         │
   │       - "what's 3220000 * 7?" — it'll guess wrong half the time      │
   │       - even Q4-quantized 70B models get long multiplications wrong  │
   │                                                                       │
   │   • Read a file you point it at:                                      │
   │       - "summarize this 50-page PDF" — it can't *open* the PDF       │
   │       - even with the bytes pasted in, you've burned 50 pages of ctx │
   │                                                                       │
   │   • Run code:                                                         │
   │       - "compute the eigenvalues of this matrix" — needs numpy       │
   │       - "is this regex correct?" — needs to actually run the regex   │
   │                                                                       │
   │   • Look up information not in its training data:                     │
   │       - "what's today's date?" — model frozen at training cutoff     │
   │       - "current price of AAPL?" — needs the live web                │
   │                                                                       │
   │   These are the gaps tool use fills. Not by training the model on    │
   │   more data, not by retrieving more text — by giving it a *button*   │
   │   it can press to make something happen.                             │
   │                                                                       │
   └──────────────────────────────────────────────────────────────────────┘
```

Tool use is a tiny architectural change with a huge capability gain. The model still emits text; the difference is that some of the text is structured (a tool call), and the runtime around the model recognizes that structure, executes the corresponding function, and feeds the result back into the conversation.

```
   ┌──────────────────────────────────────────────────────────────────────┐
   │  THE LOOP — MINIMAL VERSION                                           │
   └──────────────────────────────────────────────────────────────────────┘

      transcript = system + tool_block + user_message + "Assistant:"

      for step in range(max_steps):
          completion = backend.complete(transcript)
          tool_calls = parse_tool_calls(completion)

          if not tool_calls:
              return final_answer = completion          # ← exit clean

          results = [dispatch(c) for c in tool_calls]
          transcript += completion + format(results) + "Assistant:"

      return None    # ← exit on timeout
```

Six lines of orchestration. Most of the complexity hides in the components: parsing the tool calls reliably, validating the arguments, executing safely, formatting the results in a way the model recognizes as feedback. Each is small; the *contract* between them is the lesson.

## The big idea

### A tool is a callable + schema + name

```
   ┌──────────────────────────────────────────────────────────────────────┐
   │   THE Tool DATACLASS                                                  │
   ├──────────────────────────────────────────────────────────────────────┤
   │                                                                       │
   │     name           "calculator"                                      │
   │       ↓ matched by the parser when the model emits <tool_call>       │
   │                                                                       │
   │     description    "Evaluate an arithmetic expression..."            │
   │       ↓ rendered into the system prompt; the model reads it          │
   │       ↓ to decide whether to use this tool                           │
   │                                                                       │
   │     parameters     {"type": "object",                                │
   │                     "properties": {                                  │
   │                       "expression": {"type": "string", ...}          │
   │                     },                                                │
   │                     "required": ["expression"]}                       │
   │       ↓ rendered into the prompt as JSON                             │
   │       ↓ validated against the model's emitted arguments              │
   │                                                                       │
   │     func           callable(expression=...) → result                 │
   │       ↓ invoked by the dispatcher with validated kwargs              │
   │       ↓ result coerced to str, wrapped as ToolResult                 │
   │                                                                       │
   └──────────────────────────────────────────────────────────────────────┘
```

The `name`, `description`, and `parameters` exist for the *model* to read; the `func` exists for the *runtime* to call. The whole tool-using interaction is the model and the runtime exchanging messages through the four fields of this dataclass.

### The `<tool_call>` format

The model emits a tool call as a tagged JSON block:

```
<tool_call>{"name": "calculator", "arguments": {"expression": "2 + 2"}}</tool_call>
```

Why this format?

1. **Pretraining + instruction-tuning have seen it.** Llama 3.2 emits `<tool_call>` blocks natively when prompted for tool use. Qwen 2.5 follows the same convention. Anthropic recommends `<invoke>...</invoke>` (XML-tagged) for Claude. The "JSON in delimiter" pattern is well-established; the model knows what to do.
2. **Easy to parse.** A non-greedy regex `<tool_call>(.*?)</tool_call>` with `re.DOTALL` extracts every block in one pass.
3. **Survives surrounding prose.** The model can write chain-of-thought before/after the call, and the parser still finds the structured part.

Why JSON inside? Because JSON has a `dict` and we need named arguments. Positional arguments would work for one-arg tools; with two or more args you need names, and JSON object syntax is the lowest-friction option that's universally well-tokenized.

### The parse → validate → dispatch → feedback contract

![From model text to tool result. A four-stage pipeline drawn left-to-right. (1) PARSE: the model's completion text contains `<tool_call>{"name": "calculator", "arguments": {"expression": "3220000 * 7"}}</tool_call>`. A regex extracts the block; `json.loads` parses the body; a shape check rejects non-dict top-level, missing `name`, empty name. Output: a `ToolCall(name, arguments, call_id)`. Permissive: malformed blocks are silently skipped. (2) VALIDATE: the JSON-schema-lite validator type-checks each argument against the tool's `parameters` schema — `string`, `integer` (rejecting `bool`), `number`, `boolean`, `array`, `object`. Required keys must be present; unknown extra keys are rejected. Strict: failures surface as `ToolError`. (3) DISPATCH: the registry looks up the tool by name; if missing, return `ToolResult(is_error=True, output="unknown tool: ...")`. If validation failed, same. If the tool's `func` raises at runtime, catch the exception and wrap as `ToolResult(is_error=True, output=f"{type(e).__name__}: {e}")`. Never raise. (4) FEEDBACK: format successful results as `<tool_result name="..." id="...">22540000</tool_result>` and errors as `<tool_error>` blocks, append back to the transcript. The model reads the feedback and either calls another tool or emits a final answer. A "common failure paths" panel below: malformed JSON → silently skipped; missing required arg → `<tool_error>`; wrong type → `<tool_error>`; tool raises → `<tool_error>`; unknown tool → `<tool_error>`. Errors are conversation, not crashes.](18-tools/Module18-Parse.png)

*The picture for the four scaffolded methods. Each box is one of `parse_tool_calls`, `validate_arguments`, `dispatch_tool_call`, and the `format_tool_results` step inside `run_with_tools`. Reading off this image: the parser is permissive (silent skip on malformed blocks); the validator is strict (loud errors); the dispatcher converts everything to model-readable feedback; the loop never crashes on a model behavior. This contract is what makes the system robust.*

Each step has a precise responsibility, and each must handle its own failures by surfacing them as data (not exceptions) so the loop can continue:

```
   ┌──────────────────────────────────────────────────────────────────────┐
   │   FAILURE-MODE TABLE                                                  │
   ├──────────────────────────────────────────────────────────────────────┤
   │                                                                       │
   │   Step       What can go wrong          How we surface it            │
   │   ────       ────────────────────       ─────────────────────         │
   │   parse      malformed JSON              skip block silently         │
   │              non-dict top-level          skip block silently         │
   │              missing "name" key          skip block silently         │
   │              non-str / empty name        skip block silently         │
   │                                                                       │
   │   validate   non-dict arguments          ToolError                    │
   │              missing required key        ToolError                    │
   │              unknown extra key           ToolError                    │
   │              wrong type for value        ToolError                    │
   │                                                                       │
   │   dispatch   unknown tool name           ToolResult(is_error=True)   │
   │              validation ToolError        ToolResult(is_error=True)   │
   │              tool raised at runtime      ToolResult(is_error=True)   │
   │                                                                       │
   │   loop       no more tool calls          stopped_reason="no_more_calls" │
   │              max_steps hit               stopped_reason="max_steps"  │
   │                                                                       │
   └──────────────────────────────────────────────────────────────────────┘
```

The parser is permissive; the validator is strict; the dispatcher converts everything to model-readable feedback; the loop never crashes on a model behavior. This contract is what makes the system robust — every category of "the model did something weird" becomes a recoverable state.

### Why surface errors as data

```
   The tempting design (don't):

      def dispatch(call):
          tool = registry.get(call.name)        # raises KeyError
          args = validate(tool, call.args)      # raises ToolError
          return tool.func(**args)              # may raise anything

   The right design (do):

      def dispatch(call):
          try:
              tool = registry.get(call.name)
          except KeyError as e:
              return ToolResult(is_error=True, output=str(e))
          try:
              args = validate(tool, call.args)
          except ToolError as e:
              return ToolResult(is_error=True, output=str(e))
          try:
              return ToolResult(output=str(tool.func(**args)))
          except Exception as e:
              return ToolResult(is_error=True, output=f"{type(e).__name__}: {e}")
```

The first design throws on every failure; the loop dies, the user sees a Python stack trace. The second design wraps every failure as a `ToolResult(is_error=True)`, which the loop formats as `<tool_error name="..." id="...">message</tool_error>` and feeds back to the model. The model reads it and tries again. Errors aren't bugs; they're conversation.

### Safe evaluation, by construction

![Safe calculator — AST whitelist, not eval(). A four-stage pipeline. (1) EXPRESSION INPUT: a string like `"2 + 3 * 4"` from the model. (2) PARSE TO AST: `ast.parse(expr, mode="eval")` produces an `ast.Expression` whose `.body` is a tree of nodes — `BinOp`, `Constant`, `UnaryOp`. (3) WALK & VALIDATE (whitelist): a recursive walker visits each node and checks its type against an explicit allowlist. Allowed nodes: `Constant` (numeric only — booleans rejected), `BinOp` (with `Add`/`Sub`/`Mult`/`Div`/`Mod`/`Pow`/`FloorDiv`), `UnaryOp` (with `UAdd`/`USub`), and parenthesization. Rejected nodes: `Name` (variable references), `Call` (function calls), `Attribute` (e.g., `(1).bit_length()`), `Subscript` (e.g., `[1,2,3][0]`), `Compare`, `Lambda`, `If`, `Import` — refused by name, not by pattern matching the source string. (4) EVALUATE: only after the whole tree validates, recursively compute the numeric result. A "why not eval()?" panel below pins the rationale: `eval(expr, {"__builtins__": {}})` is famously not safe — `().__class__.__base__.__subclasses__()` and many other escapes break out of the restricted environment. Whitelist > denylist for safety. The "examples" panel shows safe expressions (`"2 + 3 * 4"`, `"-5 ** 2"`) passing and unsafe expressions (`"abs(-1)"`, `"__import__('os').system('rm -rf /')"`) being refused at the AST check, before any execution happens.](18-tools/Module18-AST.png)

*The picture for `calculator_evaluate`. The AST-walker pattern generalizes — you'd build a safe regex evaluator, a safe template engine, a safe filter expression the same way: structurally bound the surface to exactly what you can vouch for. The test `test_rejects_dunder_import` pins the most-attempted attack vector, but the security argument is structural: nodes not on the whitelist never reach evaluation.*

The calculator's safe-eval is the only piece of "real CS" in this module:

```
   1. parse the expression to an AST
   2. walk the tree
   3. allow only: Constant (numbers), BinOp (+ - * / % ** //),
                  UnaryOp (+ -), parenthesization
   4. reject every other node type by name
```

Every other node type — `Name`, `Call`, `Attribute`, `Subscript`, `Compare`, `Lambda`, `Import`, `If`, `For`, etc. — is refused. The surface is *exactly* what we admit; nothing else gets through. This is structurally different from `eval(expr, {"__builtins__": {}})`, which is a denylist (try to take away dangerous capabilities) and which has been famously broken many times. Whitelist > denylist for safety.

```
   The "unsafe eval" that fails the AST check, with the failing node:

   "x + 1"            → Name(id='x')              ← reject
   "abs(-1)"          → Call(func=Name('abs'))    ← reject
   "(1).bit_length()" → Attribute / Call          ← reject
   "[1,2,3][0]"       → List, Subscript           ← reject
   "1 < 2"            → Compare                   ← reject
   "(lambda x:x)(1)"  → Lambda, Call              ← reject
   "__import__('os')" → Name(id='__import__'),Call ← reject
```

Each is rejected at the AST walker, not because we pattern-matched the source string but because the AST node type isn't on the allowlist. The technique generalizes — you'd build a safe regex evaluator the same way (allow `Concat`, `CharClass`, `Repeat`; reject everything else).

### The loop — one screen of code

```python
def run_with_tools(backend, registry, user_message, *, max_steps=5):
    transcript = system + tools_block + f"User: {user_message}" + "\nAssistant:"
    steps = []
    for _ in range(max_steps):
        inference = backend.complete(transcript, ...)
        completion = inference.completion
        tool_calls = parse_tool_calls(completion)

        if not tool_calls:
            steps.append(ToolStep(completion, [], [], inference))
            return ToolRunResult(final_answer=completion, steps=steps,
                                 stopped_reason="no_more_calls", ...)

        results = [dispatch_tool_call(registry, c) for c in tool_calls]
        steps.append(ToolStep(completion, tool_calls, results, inference))
        transcript += " " + completion + "\n\n" + format_tool_results(results) + "\nAssistant:"

    return ToolRunResult(final_answer=None, steps=steps,
                         stopped_reason="max_steps", ...)
```

That's the entire loop. Module 19 will replace this with a ReAct-style loop that includes explicit "Thought" turns, plan tracking, and goal completion detection. For now, the contract is "no more tool calls means we're done" and "max_steps is the safety net."

### The unified tool-use interface

```
   ┌─────────────────────────────────────────────────────────────────────┐
   │   g2c/tools/  PUBLIC API                                              │
   ├─────────────────────────────────────────────────────────────────────┤
   │                                                                       │
   │   Tool(name, description, parameters, func)                          │
   │     a callable + schema + name                                        │
   │                                                                       │
   │   ToolCall(name, arguments, call_id)                                  │
   │     a parsed invocation                                               │
   │                                                                       │
   │   ToolResult(call_id, name, output, is_error)                         │
   │     the result of one execution                                       │
   │                                                                       │
   │   ToolError                                                           │
   │     internal exception type                                           │
   │                                                                       │
   │   ToolRegistry(tools=[])                                              │
   │     .register(tool), .get(name), .tools, .names()                    │
   │                                                                       │
   │   validate_arguments(tool, args) → dict                               │
   │     JSON-schema-lite validator                                        │
   │                                                                       │
   │   parse_tool_calls(text) → list[ToolCall]                             │
   │     extract <tool_call> blocks                                        │
   │                                                                       │
   │   format_tool_results(results) → str                                  │
   │     render <tool_result> / <tool_error> blocks                        │
   │                                                                       │
   │   render_tools_for_prompt(tools) → str                                │
   │     "Tools available:\n- ..."                                        │
   │                                                                       │
   │   dispatch_tool_call(registry, call) → ToolResult                     │
   │     look up + validate + execute, never raise                         │
   │                                                                       │
   │   run_with_tools(backend, registry, user_message, *, max_steps,...)  │
   │       → ToolRunResult                                                 │
   │                                                                       │
   │   Built-in tool factories:                                            │
   │     make_calculator()                                                 │
   │     make_read_file(*, root)                                           │
   │     make_web_search(*, search=None)                                   │
   │     make_run_python(*, timeout=10)                                    │
   │                                                                       │
   └─────────────────────────────────────────────────────────────────────┘
```

Total scaffolded code: roughly 60 lines spread across four method bodies (`validate_arguments`, `parse_tool_calls`, `calculator_evaluate`, `run_with_tools`). Everything else — the registry, the dispatcher, the prompt renderer, the result formatter, and three of the four built-in tools (`read_file`, `web_search`, `run_python`) — is pre-implemented because the wiring isn't the lesson; the components are.

## Concepts to internalize

- **A tool is a callable, but the model only sees the schema.** The model never executes Python. It emits text describing what it wants to call; the runtime translates that text into a function call. This decoupling is what makes tool use safe(ish): the model's job ends at "emit JSON," and the runtime's job is the actual execution.
- **Errors are conversation.** When a tool fails, the loop feeds the error back to the model and lets it try again. A raised exception ends the conversation; a `ToolResult(is_error=True)` continues it. This single decision is responsible for most of the "robust to model mistakes" feeling.
- **Safe eval is a whitelist, not a denylist.** Allow only the AST nodes you can vouch for; refuse everything else. `eval()` with restricted globals is famously not safe; the AST-walker pattern is structurally bounded.
- **Schemas are tighter than docstrings.** Model output is much more reliable when the prompt includes a precise JSON schema than when it includes only a prose description. The schema gives the model a *shape* to fill in; prose gives it a vibe.
- **The loop's stop condition is "no more tool calls."** It's an oddly minimal contract — the model itself decides when it's done by simply not emitting another `<tool_call>`. The alternative (an explicit "DONE" sentinel) is fragile; instruction-tuned models reliably stop calling tools when they have enough context.
- **`max_steps` is a safety net, not a feature.** It exists because "the model loops forever" is a real failure mode. Module 19's agent loop has smarter stop conditions (goal detection); this module's contract is "5 steps, then we cut you off."
- **Pre-fine-tuned tool calling is a free 80%.** Modern instruction-tuned open models (Llama 3.x, Qwen 2.5, Mistral 7B Instruct) emit `<tool_call>` blocks reliably given a tool-describing prompt. You don't need to fine-tune; you need to format the prompt the way they were tuned to expect. Pick a model with a known tool-calling format and use it.
- **The parser is permissive; the validator is strict.** The parser tolerates malformed blocks (silent skip) so the model isn't punished for occasional weirdness. The validator rejects malformed arguments (loud error) so the tool gets clean inputs. Different layers, different policies.

## Scaffolding and how to run the tests

This module ships seven files in `g2c/tools/`:

- **`base.py`** — `Tool`, `ToolCall`, `ToolResult`, `ToolError`, `ToolStep`, `ToolRunResult` dataclasses. All boilerplate.
- **`registry.py`** — `ToolRegistry`. Implemented.
- **`schema.py`** — `validate_arguments` (**scaffolded**) + `render_tools_for_prompt` + `DEFAULT_SYSTEM`. Implemented except the validator.
- **`parser.py`** — `parse_tool_calls` (**scaffolded**) + `format_tool_results`. Implemented except the parser.
- **`builtins.py`** — `calculator_evaluate` (**scaffolded**) + `make_calculator` + `make_read_file` + `make_web_search` + `make_run_python`. The calculator's AST walker is the scaffold; the other three tools are fully implemented.
- **`loop.py`** — `dispatch_tool_call` + `run_with_tools` (**scaffolded**). Dispatcher implemented; loop scaffolded.
- **`__init__.py`** — public exports.

Tests live in `tests/test_tools.py`. Initial state on `main`: 59 tests pass (boilerplate + the components that are fully implemented). 96 tests fail with `NotImplementedError` (or transitively, where they call into a scaffold) until you fill in the four scaffolded methods.

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

Implementation order — four independent steps:

  1. **`validate_arguments`** in `g2c/tools/schema.py`. JSON-schema-lite type checks. Pure logic — no external deps. Easiest place to start. Turns green: `TestValidateArguments`, `TestValidateArgumentsTypeChecks`, `TestValidateArgumentsErrors`. Also unblocks `TestDispatchToolCall` (the dispatcher calls the validator).

  2. **`parse_tool_calls`** in `g2c/tools/parser.py`. Regex + JSON + shape validation. Turns green: `TestParseToolCalls`, `TestParseToolCallsEdgeCases`.

  3. **`calculator_evaluate`** in `g2c/tools/builtins.py`. AST-based safe arithmetic. Turns green: `TestCalculatorEvaluate`, `TestCalculatorRejection`, `TestMakeCalculator`.

  4. **`run_with_tools`** in `g2c/tools/loop.py`. The orchestration loop. Once 1-3 are done, this turns green and the integration smoke tests pass: `TestRunWithTools`, `TestRunWithToolsLoop`, `TestRunWithToolsErrors`, `TestIntegrationSmoke`.

The four are independent. Suggested order is "easiest → hardest" but you can work in any order — each scaffolded method has its own tests that turn green when only that method is filled in.

Headline tests to watch:

- **`test_integer_rejects_bool`** — pins the `bool`-vs-`int` trap. `isinstance(True, int)` is True in Python; the validator must reject booleans explicitly when expecting numeric.
- **`test_call_id_is_unique_within_parse`** — pins the parser's id assignment. Two adjacent tool calls must get different `call_id`s.
- **`test_malformed_json_skipped`** — pins the "permissive parser" contract. Bad blocks are silently skipped so the loop can degrade to "no calls this turn."
- **`test_rejects_dunder_import`** — pins the calculator's safe-eval. `__import__('os')` must be refused at the AST check.
- **`test_unknown_tool_feedback`** — pins the dispatcher's "errors are data" contract. The model can call a nonexistent tool and the loop continues.
- **`test_tool_result_appears_in_subsequent_prompt`** — pins the feedback wiring. The loop must splice tool results into the next prompt turn or the model never sees them.
- **`test_max_steps_stops_loop`** — pins the safety net. A model that won't stop must be cut off.

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
    def get(self, name) -> Tool: ...
    @property
    def tools(self) -> list[Tool]: ...
    def names(self) -> list[str]: ...


# schema.py
def validate_arguments(tool, arguments) -> dict[str, Any]:        # SCAFFOLDED
    ...

def render_tools_for_prompt(tools) -> str:                        # implemented
    ...


# parser.py
def parse_tool_calls(text) -> list[ToolCall]:                     # SCAFFOLDED
    ...

def format_tool_results(results) -> str:                          # implemented
    ...


# builtins.py
def calculator_evaluate(expression: str) -> float:                # SCAFFOLDED
    ...

def make_calculator() -> Tool: ...                                # implemented
def make_read_file(*, root) -> Tool: ...                          # implemented
def make_web_search(*, search=None) -> Tool: ...                  # implemented
def make_run_python(*, timeout=10) -> Tool: ...                   # implemented


# loop.py
def dispatch_tool_call(registry, call) -> ToolResult: ...         # implemented

def run_with_tools(                                               # SCAFFOLDED
    backend, registry, user_message, *,
    system=DEFAULT_SYSTEM, max_steps=5, max_new_tokens=512,
    temperature=0.2, top_k=None, top_p=None,
) -> ToolRunResult:
    ...
```

Total scaffolded code: roughly 60 lines across four function bodies. Everything else is pre-implemented because the lesson is the contracts (schema validation, structured parsing, safe evaluation, loop control), not the orchestration.

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

- **Forgetting the bool-vs-int rejection in `validate_arguments`.** `isinstance(True, int)` is `True` in Python. If the schema says `{"type": "integer"}` and the model emits `true`, a naive validator accepts it — and the tool gets `True` as its numeric arg. The test `test_integer_rejects_bool` pins this; if it fails, you're missing the explicit `isinstance(value, bool)` check.

- **Strict-mode "no extra keys" surprises.** Our validator rejects argument keys not in the schema's `properties`. JSON Schema's default is *additional properties allowed*; ours is *additional properties disallowed*. The reason: if the model invents a parameter, we'd rather catch it (and feed back an error) than silently drop it. If you wanted lenient behavior, the change is one line.

- **`json.loads` raises on trailing content.** `json.loads('{"a":1}xxx')` raises `JSONDecodeError`. The parser handles this by skipping the block; if you implement parser parsing differently and don't catch the exception, a malformed block will crash the loop.

- **Non-greedy regex matters.** Without `?`, `<tool_call>...A...</tool_call>...<tool_call>...B...</tool_call>` matches as one giant block. Always `re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)`. The `?` is the difference between "two calls" and "one call with garbled JSON."

- **`re.DOTALL` is required.** Without it, `.` doesn't match newlines, and a multi-line JSON body wouldn't be captured. Models often emit `<tool_call>\n{...}\n</tool_call>` with the JSON on its own line.

- **`call_id` collisions.** If you generate `call_id` purely from a counter (`call_0`, `call_1`, ...), two parser invocations on the same conversation produce ids that collide across turns. The recipe uses `call_{i}_{uuid4().hex[:8]}` — uniqueness within and across parses.

- **Forgetting `start=1` doesn't apply here.** The parser's `enumerate(...)` starts at 0; `call_0_...` is the first call's id. Unlike Module 17's citation indexing (where `[0]` would confuse the model), tool ids are internal — the model never sees the index, only the id string.

- **Calculator: rejecting `bool` constants explicitly.** `True + 1` parses as `BinOp(Constant(True), Add, Constant(1))`. The walker's `Constant` branch must reject `bool` before checking for `int`/`float`, because `isinstance(True, int)` is True. Same trap as the validator's bool-vs-int issue, in a different place.

- **Calculator: integer overflow on `**`.** `2 ** 1000000000` parses fine, walks fine, and computes a giant integer. We don't guard against this — known limitation. A production safe-eval would put a wall-clock timeout around the whole call.

- **Calculator: `MatMult` is not on the list.** Python's `@` operator parses to `ast.MatMult`, which is NOT in `_BINOPS`. If a student copies the operator list from elsewhere and includes `MatMult`, the calculator silently accepts `5 @ 3` — which raises `TypeError` at execution. The right behavior: refuse `MatMult` at the AST check, never reach execution.

- **The tool-calling format the model emits and what the parser expects must match.** Llama 3.2 emits `<tool_call>...</tool_call>`. Llama 3.2's "ipython" mode emits `<|python_tag|>...`. Qwen 2.5 emits `<tool_call>` mostly but sometimes drops the closing tag. If you see the model emit calls but the parser sees zero, the formats don't match — pick one and align both ends.

- **Forgetting the `Assistant:` marker after each turn.** The transcript grows like "user message → assistant turn 1 → tool results → assistant turn 2 → ...". After splicing tool results, the next prompt must end with `"\nAssistant:"` so the model knows it's its turn. Without it, the model often produces a `User:` block (continuing the wrong role) and the loop breaks down.

- **Passing `chunks` instead of `arguments` from the parser.** A common student bug: the parser builds a `ToolCall` from `obj["name"]` but accidentally uses `obj` itself as the arguments dict (instead of `obj["arguments"]`). The dispatcher then passes `{"name": ..., "arguments": {...}}` to the tool, which crashes. Read the JSON path carefully — `arguments` is one level down.

- **`subprocess.run` with `shell=True`.** If you pass `["python", "-c", code]`, you're safe. If you pass `f"python -c {code}"` and `shell=True`, you've introduced shell injection — the code is interpreted by the shell first. Always use the list form.

- **`run_python` cwd surprises.** The subprocess runs from the parent's cwd. If your test runs from one directory and your real session from another, the same code can produce different results. Pass `cwd=` explicitly if you care.

- **`read_file` path-escape via symlinks.** `Path.resolve()` follows symlinks. If a malicious user (or a confused model) places a symlink in the sandboxed root pointing outside it, `resolve()` resolves through the link, and `relative_to(root)` succeeds. Our implementation uses `resolve()` and trusts the filesystem; if you're paranoid, use `os.path.realpath` and walk the path components manually. Out of scope for course pedagogy.

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
- [ ] Ollama running with a tool-calling-capable chat model. `ollama list` shows `llama3.2:3b` (or your chosen model).
- [ ] Notebook: `notebooks/18-tools.ipynb`. Wires the registry + backend + `run_with_tools`, runs Exercises 1, 2, 3, 4 with output cells visible.
- [ ] **Tool-use post-mortem** (Exercise 9) in `docs/tools-postmortem.md`. 3-4 paragraphs. The actual deliverable.
- [ ] You can explain — out loud, without notes — why errors are surfaced as `ToolResult(is_error=True)` instead of raised exceptions.
- [ ] You can explain — out loud, without notes — why AST-walking is structurally safer than `eval()` with restricted globals.
- [ ] You can explain — out loud, without notes — what the loop's stop condition is and why "no more tool calls" works as a signal.
- [ ] You can explain — out loud, without notes — why the validator must reject `bool` when expecting `int` or `number`.

## M-series notes

This module is comfortable on every M-series Mac. Practical considerations:

- **Inference happens via `OllamaBackend`** (or `LocalTransformerBackend` for the from-scratch model — but the from-scratch model isn't trained for tool calling, so it won't follow the schema). All Module 16 caveats apply: first call is slow, steady-state matches the model's parameter count.
- **Tool execution latency.** The calculator is microseconds; `read_file` is microseconds for small files; `web_search` depends on your backend (the stub is microseconds; a real DuckDuckGo / Tavily call is ~1 second); `run_python` is the slowest (subprocess startup + Python init is ~50–200ms on M-series).
- **Subprocess startup cost.** `subprocess.run([sys.executable, "-c", ...])` pays ~50-200 ms in Python startup. If `run_python` is called frequently, the total wall time is dominated by startup. A more advanced runner reuses a long-lived Python child process via `subprocess.Popen` + line-based protocol; out of scope here, but worth knowing about.
- **Context length.** Each step appends to the transcript. A 5-step run with verbose tool results can easily reach 4–8k tokens. Llama 3.2's 128k context is comfortable; smaller-context models would need careful pruning of past turns. Module 19 will introduce conversation memory management.
- **No special memory considerations for tool execution itself.** The tool runtime is pure Python plumbing. The model's inference is the memory-hungry part, and that's the same as Modules 16/17.
