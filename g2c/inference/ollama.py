"""OllamaBackend — wraps a locally-served pretrained model.

Ollama is the simplest way to run a quantized 7–8B model on a
MacBook: it bundles llama.cpp, manages model files (GGUF), and
exposes an HTTP API. See https://ollama.com/ for setup. Once Ollama
is running and a model has been pulled (`ollama pull llama3.2:3b`),
this class talks to it via the REST API.

We use stdlib `urllib.request` rather than `requests` so the project
has zero non-PyTorch dependencies. The HTTP plumbing is one POST
with a JSON body, one JSON-decoded response — small enough that
pulling in `requests` would be net negative complexity.

The endpoint we use is `POST /api/generate` with `stream: false`. A
streaming variant exists (`stream: true` returns NDJSON chunks) and
is the right call for chat UIs; for "score one prompt and return,"
streaming adds parse complexity without a real benefit. See the
Ollama docs for the streaming protocol if you want to extend.

Request/response shapes (from the Ollama API spec):

```
    POST /api/generate
    {
        "model":   "<model tag, e.g. 'llama3.2:3b'>",
        "prompt":  "<prompt string>",
        "stream":  false,
        "options": {
            "num_predict": <int>,            // max_new_tokens
            "temperature": <float>,
            "top_k":       <int>,            // omitted if None
            "top_p":       <float>           // omitted if None
        }
    }

    (response when stream=false)
    {
        "model":             "<model tag>",
        "response":          "<the generated text>",
        "done":              true,
        "prompt_eval_count": <int — input tokens>,
        "eval_count":        <int — output tokens>,
        "total_duration":    <int — nanoseconds>,
        "eval_duration":     <int — nanoseconds, just generation>,
        ... (other fields we ignore)
    }
```

The boilerplate (constructor + info property + error class):
implemented. The `complete` method: scaffolded.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

from uuid import uuid4

from .backend import Backend, BackendInfo, ChatResult, InferenceResult

DEFAULT_OLLAMA_URL = "http://localhost:11434"


# Model-tag prefixes that ship with "thinking mode" enabled by default.
# These models emit `<think>...</think>` reasoning blocks before any
# user-facing content. Useful for hard problems, fatal for tool-calling
# loops with modest `max_new_tokens` budgets: the model can spend its
# entire budget inside the thinking block and never reach an actual
# tool call or final text. `is_thinking_model` checks against this list
# so callers can default `think=False` for these models without having
# to maintain the list themselves.
#
# Membership is conservative — only listed when the model's default
# behavior is "thinking on" AND Ollama honors a `think: false` toggle
# to disable it. Adding a prefix that doesn't honor the toggle has
# no harm (the param is silently ignored).
THINKING_MODEL_PREFIXES: tuple[str, ...] = (
    "qwen3",
    "deepseek-r1",
    "deepseek-r2",
)


def is_thinking_model(model_id: str) -> bool:
    """Return True if `model_id` matches a known thinking-mode prefix.

    Matches by leading tag — e.g. `"qwen3:4b"`, `"qwen3:8b-instruct"`,
    `"deepseek-r1:7b"` all return True. Case-insensitive.

    Use this to decide whether to pass `think=False` to the backend.
    On non-thinking models the parameter is harmless; on thinking
    models it disables the `<think>` block so the model's response
    fits in modest token budgets.
    """
    if not isinstance(model_id, str) or not model_id:
        return False
    lower = model_id.lower()
    for prefix in THINKING_MODEL_PREFIXES:
        if lower.startswith(prefix):
            return True
    return False


class OllamaError(RuntimeError):
    """Raised on any HTTP / JSON / Ollama-specific failure.

    Wraps the underlying `urllib.error.URLError`, `HTTPError`, or
    `json.JSONDecodeError` with a message that names the URL we
    were trying to reach. The original error is chained via
    `raise ... from underlying` so the traceback is preserved.

    Why a custom exception instead of letting `URLError` bubble up?
    A user running `backend.complete(prompt)` shouldn't have to
    catch three different stdlib exception types to handle "Ollama
    isn't running." This is one type, with a clear name.
    """


class OllamaBackend(Backend):
    """A `Backend` over a locally-served Ollama HTTP server.

    Args:
        model_id: the Ollama model tag, e.g. `"llama3.2:3b"`,
            `"qwen2.5:7b-instruct-q4_K_M"`. Must be a model you've
            already pulled (`ollama pull <tag>` from the shell);
            this class does NOT pull on demand.
        base_url: the Ollama server URL. Defaults to
            `http://localhost:11434` — the Ollama default. Override
            if you're proxying or running on a different port.
        timeout: HTTP timeout in seconds for each `complete` call.
            Defaults to 120 — generous for a 7B model on M1 doing
            128 tokens. Bump if your model is slower or your
            `max_new_tokens` is larger.
        urlopen: optional override for `urllib.request.urlopen`. The
            default is the stdlib function. Tests inject a fake.
            Production callers leave it alone.
        name: backend kind label. Defaults to `"ollama"`. Override
            only if you're subclassing.
        extra: additional metadata for the `BackendInfo.extra` dict.

    The `info` property returns a `BackendInfo` snapshot built from
    the constructor args. The base URL is stored in `info.extra`
    automatically.

    The boilerplate (constructor + `info`) is implemented. The
    `complete` method is scaffolded — that's the lesson.
    """

    def __init__(
        self,
        model_id: str = "llama3.2:3b",
        *,
        base_url: str = DEFAULT_OLLAMA_URL,
        timeout: float = 120.0,
        urlopen=None,
        name: str = "ollama",
        extra: dict[str, Any] | None = None,
    ) -> None:
        if not isinstance(model_id, str) or not model_id:
            raise ValueError("OllamaBackend requires a non-empty model_id")
        if not isinstance(base_url, str) or not base_url:
            raise ValueError("OllamaBackend requires a non-empty base_url")
        if timeout <= 0:
            raise ValueError(f"timeout must be > 0, got {timeout}")

        self._model_id = model_id
        self._base_url = base_url.rstrip("/")
        self._timeout = float(timeout)
        self._urlopen = urlopen if urlopen is not None else urllib.request.urlopen

        merged_extra: dict[str, Any] = {"base_url": self._base_url}
        if extra is not None:
            merged_extra.update(extra)
        self._info = BackendInfo(
            name=name,
            model_id=model_id,
            extra=merged_extra,
        )

    @property
    def info(self) -> BackendInfo:
        return self._info

    @property
    def base_url(self) -> str:
        return self._base_url

    def complete(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> InferenceResult:
        """POST to `{base_url}/api/generate` and parse the response.

        Args:
            prompt: input string. Non-empty.
            max_new_tokens: forwarded as `options.num_predict`.
            temperature: forwarded as `options.temperature`.
            top_k: forwarded as `options.top_k` (omitted from the
                payload entirely if None — letting Ollama use its
                default rather than passing 0).
            top_p: forwarded as `options.top_p` (likewise).

        Returns:
            `InferenceResult` with:
              * prompt = the input string
              * completion = the response's `"response"` field
              * prompt_tokens = response's `"prompt_eval_count"`,
                or None if the server didn't return it
              * completion_tokens = response's `"eval_count"`, or None
              * latency_ms = wall-clock time around the HTTP call
              * backend = self.info
              * metadata = {
                    "server_total_duration_ms": <total_duration / 1e6>,
                    "server_eval_duration_ms":  <eval_duration / 1e6>,
                    "sampling": {... the four params ...},
                }
                (The server's reported timings are kept alongside
                wall-clock for benchmarking comparisons.)

        Raises:
            OllamaError: if Ollama isn't running, the HTTP call fails,
                the response isn't valid JSON, or the response is
                missing the `"response"` field. The chained `__cause__`
                is the underlying stdlib exception.

        Recipe:
            1. # Validate.
               if not isinstance(prompt, str) or len(prompt) == 0:
                   raise ValueError("prompt must be a non-empty str")
               if max_new_tokens <= 0:
                   raise ValueError("max_new_tokens must be > 0")

            2. # Build the request body. Only include sampling
               # options that were explicitly set — leaving an option
               # absent lets Ollama fill in its default.
               options: dict[str, Any] = {
                   "num_predict": int(max_new_tokens),
                   "temperature": float(temperature),
               }
               if top_k is not None:
                   options["top_k"] = int(top_k)
               if top_p is not None:
                   options["top_p"] = float(top_p)

               body = {
                   "model":   self._model_id,
                   "prompt":  prompt,
                   "stream":  False,
                   "options": options,
               }
               body_bytes = json.dumps(body).encode("utf-8")

            3. # Build the HTTP request.
               url = f"{self._base_url}/api/generate"
               req = urllib.request.Request(
                   url,
                   data=body_bytes,
                   headers={"Content-Type": "application/json"},
                   method="POST",
               )

            4. # Time + send. Wrap stdlib errors in OllamaError so
               # callers have one type to catch.
               t0 = time.perf_counter()
               try:
                   with self._urlopen(req, timeout=self._timeout) as resp:
                       resp_bytes = resp.read()
               except urllib.error.HTTPError as e:
                   raise OllamaError(
                       f"Ollama HTTP error {e.code} at {url}: {e.reason}"
                   ) from e
               except urllib.error.URLError as e:
                   raise OllamaError(
                       f"Could not reach Ollama at {url}: {e.reason}"
                   ) from e
               t1 = time.perf_counter()
               latency_ms = (t1 - t0) * 1000.0

            5. # Parse JSON.
               try:
                   data = json.loads(resp_bytes)
               except json.JSONDecodeError as e:
                   raise OllamaError(
                       f"Ollama returned non-JSON response from {url}"
                   ) from e
               if "response" not in data:
                   raise OllamaError(
                       f"Ollama response missing 'response' field: "
                       f"keys={list(data.keys())}"
                   )

            6. # Build result.
               total_dur_ns = data.get("total_duration")
               eval_dur_ns  = data.get("eval_duration")
               metadata = {
                   "sampling": {
                       "max_new_tokens": max_new_tokens,
                       "temperature":    temperature,
                       "top_k":          top_k,
                       "top_p":          top_p,
                   },
                   "server_total_duration_ms": (
                       float(total_dur_ns) / 1e6 if total_dur_ns is not None else None
                   ),
                   "server_eval_duration_ms": (
                       float(eval_dur_ns)  / 1e6 if eval_dur_ns  is not None else None
                   ),
               }
               return InferenceResult(
                   prompt=prompt,
                   completion=data["response"],
                   prompt_tokens=data.get("prompt_eval_count"),
                   completion_tokens=data.get("eval_count"),
                   latency_ms=latency_ms,
                   backend=self._info,
                   metadata=metadata,
               )

        Implementation notes:

          * `Content-Type: application/json` is required by Ollama;
            without it the server rejects the request with a 400.

          * `urllib.request.urlopen` raises `HTTPError` on non-2xx
            (a subclass of `URLError`), and `URLError` on network
            failure. We translate both to `OllamaError` so callers
            don't need to care about stdlib exception hierarchy.

          * Ollama's `total_duration` is in NANOSECONDS. The conversion
            to milliseconds is `/ 1e6`. (Off-by-1000 here means
            "this 7B model takes 12000 ms = 12 s" when it actually
            took 12 ms — silently absurd.)

          * The wall-clock `latency_ms` and the server's
            `total_duration_ms` typically agree to within a few ms.
            They diverge if the network is slow or the server is
            warming up the model from disk. Track both — wall-clock
            is what the user feels; server-reported is what the
            model actually spent.

          * `eval_count` is the OUTPUT token count, NOT the input.
            The input token count is `prompt_eval_count`. Misnamed
            in the API but that's the spec.

        Sanity values:

          * Ollama not running: raises OllamaError with "Could not
            reach Ollama at ...". A fast failure mode.

          * Pulled model + valid prompt + Ollama running: returns
            an InferenceResult with non-empty completion, prompt
            and completion token counts, and latency on the order
            of seconds for a 7B model.

          * `max_new_tokens=1`: completion is a short string,
            completion_tokens is exactly 1.
        """
        # TODO
        raise NotImplementedError

    def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        max_new_tokens: int = 1024,
        temperature: float = 0.2,
        top_k: int | None = None,
        top_p: float | None = None,
        think: bool | None = None,
    ) -> ChatResult:
        """POST to `{base_url}/api/chat` and return a `ChatResult`.

        This is the *native* tool-calling path: Ollama receives the
        message history and a list of tool specs, routes the tools
        through the model's own post-training format (Llama's
        `<|python_tag|>`, Qwen's `<tool_call>` block, etc.), and
        returns parsed `tool_calls` as a structured field. The caller
        does not need to parse free-form text — that's the whole point.

        This method is fully implemented (not scaffolded). The lesson
        is `complete`; this is plumbing built atop the same HTTP shape.

        Args:
            messages: an OpenAI/Ollama-style chat history. Each entry
                is a dict with at minimum a `role` (`"system"`,
                `"user"`, `"assistant"`, or `"tool"`) and either
                `content` or `tool_calls`/`tool_call_id`. Non-empty.
            tools: a list of Ollama tool spec dicts, each shaped like
                `{"type": "function", "function": {"name", "description",
                "parameters"}}`. `None` (or `[]`) is allowed — the call
                degenerates to plain chat.
            max_new_tokens: forwarded as `options.num_predict`.
            temperature: forwarded as `options.temperature`. Defaults
                to 0.2 — near-greedy is the right default for tool
                selection.
            top_k, top_p: forwarded.

        Returns:
            `ChatResult` with:
              * `messages` = the input messages (echoed)
              * `content` = the assistant message's text content (may
                be `""` when the model returned only tool calls)
              * `tool_calls` = a list of dicts, one per call. Each is
                `{"name": str, "arguments": dict, "call_id": str}`.
                Ollama's response nests these under
                `message.tool_calls[i].function` — we flatten and add
                a `call_id` (Ollama's spec allows but doesn't require
                an `id`; we generate one when missing).
              * `prompt_tokens`, `completion_tokens`, `latency_ms`,
                `backend`, `metadata`: same semantics as
                `InferenceResult`.

        Raises:
            ValueError: if `messages` is not a non-empty list, or
                `max_new_tokens <= 0`.
            OllamaError: same conditions as `complete` — connection
                failure, non-JSON response, missing `message` field.

        Notes:
          * Why one method per endpoint instead of overloading
            `complete`? `/api/generate` and `/api/chat` have genuinely
            different request shapes (`prompt: str` vs `messages: list`)
            and response shapes (`response: str` vs
            `message: {content, tool_calls}`). Cramming both into one
            method requires either a discriminator argument or two
            different return types — both ugly. Two methods, clean
            contracts.

          * Ollama's `tool_calls` schema mirrors OpenAI's. Each call
            has `function: {name, arguments}` where `arguments` may be
            a JSON object (Ollama 0.3+) or a JSON-encoded string
            (older versions / some forwarders). We accept both.

          * Errors are raised, not wrapped as tool-call errors. The
            *harness* (run_with_tools) is responsible for turning
            failed dispatches into `ToolResult(is_error=True)`. This
            method only fails for "the HTTP call itself broke."
        """
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty list")
        if max_new_tokens <= 0:
            raise ValueError(
                f"max_new_tokens must be positive, got {max_new_tokens}"
            )

        options: dict[str, Any] = {
            "num_predict": int(max_new_tokens),
            "temperature": float(temperature),
        }
        if top_k is not None:
            options["top_k"] = int(top_k)
        if top_p is not None:
            options["top_p"] = float(top_p)

        body: dict[str, Any] = {
            "model": self._model_id,
            "messages": messages,
            "stream": False,
            "options": options,
        }
        if tools:
            body["tools"] = tools
        if think is not None:
            body["think"] = bool(think)

        url = f"{self._base_url}/api/chat"
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        start = time.perf_counter()
        try:
            with self._urlopen(request, timeout=self._timeout) as response:
                response_bytes = response.read()
        except urllib.error.HTTPError as exc:
            raise OllamaError(
                f"Ollama HTTP error {exc.code} at {url}: {exc.reason}"
            ) from exc
        except urllib.error.URLError as exc:
            raise OllamaError(f"Could not reach Ollama at {url}: {exc.reason}") from exc
        latency_ms = (time.perf_counter() - start) * 1000.0

        try:
            data = json.loads(response_bytes)
        except json.JSONDecodeError as exc:
            raise OllamaError(f"Ollama returned non-JSON response from {url}") from exc

        if "message" not in data:
            raise OllamaError(
                "Ollama chat response missing 'message' field: "
                f"keys={list(data.keys())}"
            )

        message = data["message"]
        content = message.get("content", "") or ""
        raw_calls = message.get("tool_calls") or []
        parsed_calls: list[dict[str, Any]] = []
        for raw in raw_calls:
            fn = raw.get("function") or {}
            name = fn.get("name")
            if not isinstance(name, str) or not name:
                # Skip malformed entries — the harness will see an
                # empty list and fall through to "no calls."
                continue
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            if not isinstance(args, dict):
                args = {}
            call_id = raw.get("id") or f"call_{uuid4().hex[:8]}"
            parsed_calls.append(
                {"name": name, "arguments": args, "call_id": str(call_id)}
            )

        total_duration = data.get("total_duration")
        eval_duration = data.get("eval_duration")
        return ChatResult(
            messages=messages,
            content=content,
            tool_calls=parsed_calls,
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
            latency_ms=latency_ms,
            backend=self._info,
            metadata={
                "sampling": {
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "top_k": top_k,
                    "top_p": top_p,
                    "think": think,
                },
                "server_total_duration_ms": (
                    float(total_duration) / 1e6 if total_duration is not None else None
                ),
                "server_eval_duration_ms": (
                    float(eval_duration) / 1e6 if eval_duration is not None else None
                ),
                "n_tools_offered": len(tools) if tools else 0,
            },
        )

    def __repr__(self) -> str:
        return (
            f"OllamaBackend(model_id={self._model_id!r}, "
            f"base_url={self._base_url!r})"
        )
