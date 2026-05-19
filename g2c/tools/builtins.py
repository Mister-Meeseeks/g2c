"""Built-in tools — calculator, read_file, web_search stub, run_python.

Four tools used by the lesson, each constructed by a `make_*` factory
function that returns a `Tool`. Why factories instead of pre-built
instances? Two reasons:

  * **Closure over state.** `make_read_file(root=Path("docs/"))` builds
    a tool whose `func` is a closure over `root`. Different agents
    sandbox to different directories without redefining the tool.
  * **Test isolation.** Tests build a fresh tool per test, with
    test-scoped state (a `tmp_path`, a fake `urlopen`, etc.). No
    cross-test pollution.

The tools, in increasing complexity:

  * **`make_calculator()`** — wraps `calculator_evaluate`, an AST-based
    safe arithmetic evaluator. The student implements
    `calculator_evaluate` (it's the lesson on safe code execution).

  * **`make_read_file(root)`** — reads UTF-8 text files under a
    sandboxed root. Path-escape-protected. Truncated to a max length.
    Fully implemented.

  * **`make_web_search(callable=None)`** — a stub by default; pass a
    callable to wire a real search backend (DuckDuckGo, Tavily, etc.).
    Fully implemented.

  * **`make_run_python(timeout)`** — runs Python code in a subprocess
    with a wall-clock timeout. Fully implemented. Note the safety
    caveat: subprocess sandboxing is "the spawned Python process can
    do anything the user can do." Adequate for local pedagogy, NOT
    for a production system.

Why is the calculator the only scaffold here? The other three are
plumbing — files, HTTP, subprocess. The calculator is the genuine
small lesson on "how to evaluate untrusted code without `eval()`."
The pattern (parse → walk AST → reject everything not on the allowed
list) is the same one you'd use to build any safe expression
evaluator. Worth implementing once.
"""
from __future__ import annotations

import ast
import operator
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .base import Tool, ToolError

# Allowed AST binary operators for the calculator. Each maps to its
# `operator` module function. `ast.Pow` is included; `ast.MatMult` is
# NOT — `@` is a matrix-multiply operator that has no business here
# and would silently accept "5 @ 3" as a type error from the model.
_BINOPS: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.FloorDiv: operator.floordiv,
}

# Allowed unary ops: just sign flips. No `not`, no `~`.
_UNARYOPS: dict[type, Callable[[Any], Any]] = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def calculator_evaluate(expression: str) -> float:
    """Evaluate an arithmetic expression safely.

    Allowed: number constants, `+ - * / % ** //`, parentheses,
    unary `+` and `-`. Anything else — names, calls, subscripts,
    attribute access, comparisons, comprehensions — raises
    `ToolError`.

    The safety mechanism is structural, not blacklist-based: we parse
    to an AST, walk the tree, and explicitly check that every node is
    a member of a small allowed set. A node we don't recognize gets
    rejected. This is the right way to do "safe eval" — `eval()` with
    cleaned-up globals is famously not safe.

    Args:
        expression: the arithmetic expression as a string. Anything
            Python's parser accepts as an expression — but only the
            arithmetic subset will pass the structural check.

    Returns:
        The numeric result. `int` if the operations preserve int;
        `float` if any division / float constants get involved.

    Raises:
        ToolError: parse failures, disallowed AST nodes, or
            disallowed operators. The error message names the
            offending node type so the student can debug.

    Recipe:

        1. # Parse in expression mode. SyntaxError → ToolError.
           try:
               tree = ast.parse(expression, mode="eval")
           except SyntaxError as e:
               raise ToolError(f"could not parse: {e.msg}") from e

        2. # Walk the tree starting at the body (Expression's child).
           return _eval_node(tree.body)

        # And then a recursive helper that handles each allowed node:

        3. def _eval_node(node):
               # Constant: must be int or float, not str/None/bool.
               if isinstance(node, ast.Constant):
                   if isinstance(node.value, bool):
                       raise ToolError("booleans are not allowed in expressions")
                   if isinstance(node.value, (int, float)):
                       return node.value
                   raise ToolError(f"constant must be a number, got {type(node.value).__name__}")

               # BinOp: <op> must be in _BINOPS.
               if isinstance(node, ast.BinOp):
                   op = _BINOPS.get(type(node.op))
                   if op is None:
                       raise ToolError(f"binary operator not allowed: {type(node.op).__name__}")
                   return op(_eval_node(node.left), _eval_node(node.right))

               # UnaryOp: <op> must be in _UNARYOPS.
               if isinstance(node, ast.UnaryOp):
                   op = _UNARYOPS.get(type(node.op))
                   if op is None:
                       raise ToolError(f"unary operator not allowed: {type(node.op).__name__}")
                   return op(_eval_node(node.operand))

               # Anything else: reject by class name.
               raise ToolError(f"AST node not allowed: {type(node).__name__}")

    Implementation notes:

      * **Why structural, not regex/blacklist.** `re.search(r"[a-z]")`
        as a "safety check" misses Unicode names, `__import__`,
        `().__class__.__base__.__subclasses__()`, and a hundred other
        attack patterns documented in the safe-eval literature. AST
        whitelisting refuses everything not explicitly allowed, which
        is the only correct posture for executing untrusted input.

      * **Why reject booleans.** `True + 1` is `2` in Python because
        `bool` is a subclass of `int`. If the model emits
        `"true + 1"` (lowercase, syntax error), we reject at parse.
        If it emits `"True + 1"`, the constant *is* a bool — we
        reject explicitly so the model gets clear feedback.

      * **Pow gotcha.** `2 ** 1000000000` is parseable, allowed by
        the AST walker, and computes a giant integer that may take
        seconds to materialize. We don't guard against this — it's
        a known limitation. A production safe-eval would put a
        wall-clock timeout around the whole call. We don't because
        the pedagogical lesson is the AST walking, not denial-of-
        service mitigation.

      * **Recursive vs iterative walking.** The recipe uses recursion
        for clarity. For arbitrary-depth expressions you could blow
        the Python stack, but a 1000-deep parenthesized expression
        is not a real input we expect from the model. If you ever
        want a robust version, `ast.NodeVisitor` is the iterative
        approach.

    Sanity values:

      * `calculator_evaluate("1 + 2")` → 3
      * `calculator_evaluate("(2 + 3) * 4")` → 20
      * `calculator_evaluate("2 ** 10")` → 1024
      * `calculator_evaluate("-5 + 3")` → -2
      * `calculator_evaluate("7 / 2")` → 3.5
      * `calculator_evaluate("7 // 2")` → 3
      * `calculator_evaluate("'hello'")` → ToolError (string constant)
      * `calculator_evaluate("os.system('ls')")` → ToolError (Call/Name)
      * `calculator_evaluate("__import__('os')")` → ToolError (Name)
      * `calculator_evaluate("True + 1")` → ToolError (bool)
    """
    if not isinstance(expression, str):
        raise ToolError(
            f"expression must be a str, got {type(expression).__name__}"
        )

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ToolError(f"could not parse: {exc.msg}") from exc

    def _eval_node(node: ast.AST):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                raise ToolError("booleans are not allowed in expressions")
            if isinstance(node.value, (int, float)):
                return node.value
            raise ToolError(
                "constant must be a number, "
                f"got {type(node.value).__name__}"
            )

        if isinstance(node, ast.BinOp):
            op = _BINOPS.get(type(node.op))
            if op is None:
                raise ToolError(
                    f"binary operator not allowed: {type(node.op).__name__}"
                )
            return op(_eval_node(node.left), _eval_node(node.right))

        if isinstance(node, ast.UnaryOp):
            op = _UNARYOPS.get(type(node.op))
            if op is None:
                raise ToolError(
                    f"unary operator not allowed: {type(node.op).__name__}"
                )
            return op(_eval_node(node.operand))

        raise ToolError(f"AST node not allowed: {type(node).__name__}")

    return _eval_node(tree.body)


def make_calculator() -> Tool:
    """Build the calculator tool.

    Wraps `calculator_evaluate` and exposes it as `calculator(expression)`.
    """
    def _func(expression: str) -> str:
        return str(calculator_evaluate(expression))

    return Tool(
        name="calculator",
        description=(
            "Evaluate an arithmetic expression. Supports + - * / % ** // "
            "and parentheses. Returns the numeric result as a string."
        ),
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": (
                        "Arithmetic expression to evaluate, "
                        "e.g. '2 + 2' or '(3 ** 4) / 7'."
                    ),
                },
            },
            "required": ["expression"],
        },
        func=_func,
    )


def make_read_file(*, root: str | os.PathLike[str]) -> Tool:
    """Build a sandboxed read_file tool.

    Args:
        root: the directory under which all reads must live. Paths
            that resolve outside `root` (via `..` or absolute paths)
            are rejected. Resolved once at construction time; if the
            directory moves later, behavior is undefined.

    The returned tool is `read_file(path, max_chars=10000)`:
      * `path` is interpreted relative to `root`.
      * The result is the file's UTF-8 contents, truncated to
        `max_chars` characters with an ellipsis marker.

    Fully implemented.
    """
    root_path = Path(root).resolve()

    def _func(path: str, max_chars: int = 10000) -> str:
        if max_chars <= 0:
            raise ToolError("max_chars must be > 0")
        # Reject absolute or empty paths up front so we don't even
        # try to resolve them.
        if not path or os.path.isabs(path):
            raise ToolError(f"path must be a relative path under root, got {path!r}")
        target = (root_path / path).resolve()
        try:
            target.relative_to(root_path)
        except ValueError as e:
            raise ToolError(
                f"path escapes the allowed root: {path!r}"
            ) from e
        if not target.exists():
            raise ToolError(f"file not found: {path!r}")
        if not target.is_file():
            raise ToolError(f"not a file: {path!r}")
        try:
            text = target.read_text(encoding="utf-8")
        except UnicodeDecodeError as e:
            raise ToolError(f"file is not valid UTF-8: {path!r}") from e
        if len(text) > max_chars:
            text = text[:max_chars] + "\n... (truncated)"
        return text

    return Tool(
        name="read_file",
        description=(
            "Read a UTF-8 text file from the project root and return its "
            "contents. The path is interpreted relative to a sandboxed "
            "root directory."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path under the project root.",
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Optional truncation length (default 10000).",
                },
            },
            "required": ["path"],
        },
        func=_func,
    )


def _stub_search(query: str) -> str:
    """Default web_search backend: a stub. Returns a fixed string."""
    return (
        f"[stub web_search result for {query!r}]\n"
        "This is a stub. Wire DuckDuckGo / SerpAPI / Tavily / Bing into "
        "make_web_search(search=...) to get real results."
    )


def make_web_search(
    *,
    search: Callable[[str], str] | None = None,
) -> Tool:
    """Build a web_search tool.

    Args:
        search: optional callable that takes a query string and
            returns a string of results. If None, uses a stub that
            returns a fixed message — useful for testing the loop
            without external dependencies.

    Fully implemented. The interesting work is choosing a real backend
    (DuckDuckGo via `duckduckgo-search`, Tavily via their HTTP API, an
    SERP-API account, etc.) and writing the wrapper. We leave that to
    the student because every choice is a deployment decision, not a
    pedagogical one.
    """
    backend = search if search is not None else _stub_search

    def _func(query: str) -> str:
        if not isinstance(query, str) or not query.strip():
            raise ToolError("query must be a non-empty string")
        return str(backend(query))

    return Tool(
        name="web_search",
        description=(
            "Search the web for the given query. (Stub by default — wire "
            "a real backend for real results.)"
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query.",
                },
            },
            "required": ["query"],
        },
        func=_func,
    )


def make_run_python(
    *,
    timeout: float = 10.0,
    cwd: str | os.PathLike[str] | None = None,
) -> Tool:
    """Build a run_python tool that executes code in a subprocess.

    Args:
        timeout: wall-clock timeout in seconds. The subprocess is
            killed if it runs longer.
        cwd: working directory for the subprocess. If `None`, the
            child inherits the parent's cwd. Pair with `make_read_file(root=...)`
            using the same directory so the model sees one consistent
            file namespace across both tools.

    Fully implemented. Caveats:

      * **Not a sandbox.** The subprocess can do anything the user can
        do — read files, open sockets, start more processes. This is
        adequate for a local single-user pedagogy tool. It is NOT
        adequate for a hosted / multi-tenant system. `cwd` is a
        convenience for path consistency, not a containment boundary;
        the child can still `os.chdir` or open absolute paths.
      * **No env scrubbing.** Environment variables (including secrets)
        are inherited. Sanitize if needed.

    Output: the child's stdout (truncated to 10000 chars). On non-zero
    exit, the stderr is returned with a `[run_python: exit N]` header.
    On timeout, a `[run_python: timed out after Ns]` message.
    """
    if timeout <= 0:
        raise ValueError(f"timeout must be > 0, got {timeout}")
    cwd_path: str | None = None
    if cwd is not None:
        resolved = Path(cwd).resolve()
        if not resolved.is_dir():
            raise ValueError(f"cwd must be an existing directory, got {cwd!r}")
        cwd_path = str(resolved)

    def _func(code: str) -> str:
        if not isinstance(code, str):
            raise ToolError(f"code must be a str, got {type(code).__name__}")
        try:
            result = subprocess.run(  # noqa: S603 — local pedagogy tool
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
                cwd=cwd_path,
            )
        except subprocess.TimeoutExpired:
            return f"[run_python: timed out after {timeout}s]"
        if result.returncode != 0:
            err = result.stderr.strip() or "(no stderr)"
            return f"[run_python: exit {result.returncode}]\n{err}"
        out = result.stdout
        if not out:
            return "(no output)"
        if len(out) > 10000:
            out = out[:10000] + "\n... (truncated)"
        return out

    return Tool(
        name="run_python",
        description=(
            "Execute a Python snippet in a subprocess and return stdout. "
            "Use print() to emit results. Has a wall-clock timeout."
        ),
        parameters={
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python source to run.",
                },
            },
            "required": ["code"],
        },
        func=_func,
    )
