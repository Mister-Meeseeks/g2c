"""The Backend ABC and its in/out dataclasses.

The unified interface for inference. Three types live here, all
fully implemented:

  * `Backend` — abstract base. Every concrete backend (local, Ollama,
    MLX, ...) is a subclass that implements `info` and `complete`.

  * `BackendInfo` — frozen identity card. `name` is the backend kind
    (`"local"`, `"ollama"`, ...); `model_id` is the specific model
    inside that backend (`"g2c-tiny-20m"`, `"llama3.2:3b"`, ...). The
    `extra` dict is for backend-specific labels (server URL, quant
    type, dtype) that the caller wants to log alongside results.

  * `InferenceResult` — one (prompt → completion) call's output. Pins
    the prompt and completion text, the token counts on each side,
    the wall-clock latency, and a back-pointer to the backend that
    produced it. `tokens_per_second` is a derived property — the
    canonical readout for a "is this fast enough" sanity check.

The contract is small on purpose. A `Backend` does ONE thing — given a
prompt, produce an `InferenceResult`. Everything more sophisticated
(streaming, tool calls, batched generation, async) is layered on top
in later modules; this module just nails down the lowest common
denominator so RAG (Module 17), tools (Module 18), and the agent
(Module 19) have a stable substrate to build against.

Why the back-pointer to `BackendInfo` on every result? When you run a
benchmark across two backends and dump the results to JSON, the rows
need to be self-identifying — otherwise rerunning a notebook against
a different model leaves you with un-attributable timing numbers a
week later. Carrying the info along on every result is a one-line
hedge against a class of "where did this number come from" bugs.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class BackendInfo:
    """Identity card for a backend instance.

    Attributes:
        name: the backend kind. Conventions used in this module:
            `"local"` for a `LocalTransformerBackend`, `"ollama"` for
            an `OllamaBackend`. Free-form, but a notebook reading
            results from disk uses this string to dispatch.
        model_id: the specific model inside the backend. For local,
            something like `"g2c-tiny-20m"` (whatever you decide to
            label your checkpoint). For Ollama, the literal Ollama
            tag — `"llama3.2:3b"`, `"qwen2.5:7b-instruct-q4_K_M"`, etc.
        extra: free-form dict for backend-specific metadata you want
            to keep with the identity card. The base URL of an Ollama
            server, the dtype/device of a local model, the quant
            level, etc. Not used by the harness; persisted to JSON
            alongside results.

    Frozen because the identity is supposed to be immutable for the
    backend's lifetime: you shouldn't quietly rebrand a backend mid-
    benchmark. The mutable `extra` dict is still mutable in-place
    (Python's frozen freezes attribute-assignment, not contained
    objects); for the dataclass to behave as expected, treat it as
    read-only after construction.
    """

    name: str
    model_id: str
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("BackendInfo.name must be a non-empty str")
        if not isinstance(self.model_id, str) or not self.model_id:
            raise ValueError("BackendInfo.model_id must be a non-empty str")


@dataclass
class InferenceResult:
    """One backend `complete` call's result.

    Attributes:
        prompt: the input string the backend was called with.
        completion: the generated continuation. Does NOT include the
            prompt — only the new text the model emitted. Same
            convention as `g2c.sampling.generate` (which returns the
            full sequence; the wrapper slices off the prompt).
        prompt_tokens: number of input tokens the backend reported.
            `None` if the backend doesn't expose it (some inference
            servers don't). The local backend always populates it.
        completion_tokens: number of output tokens the backend
            generated. `None` only if the backend doesn't expose it.
            For accuracy of `tokens_per_second`, this should be the
            actual token count, not a character count.
        latency_ms: wall-clock latency for the call, in milliseconds.
            Always populated. Includes everything from "we called
            complete" to "we got the result" — for HTTP backends,
            that includes network round-trip and JSON serialization.
        backend: a `BackendInfo` snapshot of the backend that produced
            this result. Self-identifying — a JSON dump of results is
            still meaningful a week later.
        metadata: per-call extras — server-side timings, raw response
            JSON, the sampling parameters that were used, etc. Free-
            form and lossy. Useful for ad-hoc diagnostics; not relied
            on by the harness.

    The `tokens_per_second` property is the headline throughput
    metric. Returns `None` when either `completion_tokens` is `None`
    or `0`, or when `latency_ms` is `0` (degenerate case — should
    never happen in practice but worth handling).
    """

    prompt: str
    completion: str
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_ms: float
    backend: BackendInfo
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.latency_ms < 0.0:
            raise ValueError(
                f"InferenceResult.latency_ms must be >= 0, got {self.latency_ms}"
            )
        if self.prompt_tokens is not None and self.prompt_tokens < 0:
            raise ValueError(
                f"InferenceResult.prompt_tokens must be >= 0 or None, "
                f"got {self.prompt_tokens}"
            )
        if self.completion_tokens is not None and self.completion_tokens < 0:
            raise ValueError(
                f"InferenceResult.completion_tokens must be >= 0 or None, "
                f"got {self.completion_tokens}"
            )

    @property
    def tokens_per_second(self) -> float | None:
        """Throughput of the completion, in tokens per second.

        Returns `None` if `completion_tokens` is missing or zero, or
        if `latency_ms` is zero (the divide-by-zero case). A well-
        formed call to a real backend always returns a float.
        """
        if self.completion_tokens is None or self.completion_tokens == 0:
            return None
        if self.latency_ms <= 0.0:
            return None
        return float(self.completion_tokens) / (self.latency_ms / 1000.0)


@dataclass
class ChatResult:
    """One backend `chat_with_tools` call's result.

    The chat/tool-call analogue of `InferenceResult`. Where the
    text-completion path treats the conversation as one growing prompt
    string and parses tool calls out of free-form text, the chat path
    keeps a structured message list and receives structured tool calls
    back from the backend (e.g. Ollama's `/api/chat`).

    Attributes:
        messages: the input messages list the backend was called with.
            Each entry is a dict with `role` and either `content` or
            `tool_calls`; the exact shape mirrors OpenAI / Ollama chat
            conventions. Echoed back so the result is self-describing.
        content: the assistant's text content. May be the empty string
            when the model returned only tool calls and no prose.
        tool_calls: the structured tool invocations the backend parsed
            out of the model's response. Each entry is a dict with
            keys `name`, `arguments`, and `call_id`. The tool harness
            converts these to `ToolCall` instances.
        prompt_tokens, completion_tokens, latency_ms, backend,
            metadata: same semantics as `InferenceResult`.

    `tool_calls` is a list of dicts rather than `ToolCall` instances
    so the inference layer has no dependency on `g2c.tools`. The loop
    layer translates between the two.
    """

    messages: list[dict[str, Any]]
    content: str
    tool_calls: list[dict[str, Any]]
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_ms: float
    backend: BackendInfo
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.latency_ms < 0.0:
            raise ValueError(
                f"ChatResult.latency_ms must be >= 0, got {self.latency_ms}"
            )
        if self.prompt_tokens is not None and self.prompt_tokens < 0:
            raise ValueError(
                f"ChatResult.prompt_tokens must be >= 0 or None, "
                f"got {self.prompt_tokens}"
            )
        if self.completion_tokens is not None and self.completion_tokens < 0:
            raise ValueError(
                f"ChatResult.completion_tokens must be >= 0 or None, "
                f"got {self.completion_tokens}"
            )


class Backend(ABC):
    """The uniform inference interface.

    Subclasses implement two things:

      * `info` — a `BackendInfo` that describes this backend instance.
        Should be cheap and idempotent (returns the same value on
        every call, no side effects). Treat as a property; the base
        class declares it as one.

      * `complete(prompt, *, max_new_tokens, temperature, top_k, top_p)
        -> InferenceResult` — the actual call. Returns the result
        synchronously. Streaming and async are out of scope for this
        module; an extension can subclass and add `complete_stream`.

    The four sampling parameters are the lowest common denominator
    across local, Ollama, and MLX backends. They map cleanly:

      * `max_new_tokens` → local: passed to `g2c.sampling.generate`;
        ollama: `options.num_predict`.
      * `temperature` → both: scales logits before softmax.
      * `top_k` → both: keep top-k tokens (None / 0 disables).
      * `top_p` → both: nucleus sampling threshold (None / 1.0
        disables).

    `temperature == 0.0` is the "greedy" convention from Module 11 —
    deterministic argmax. Concrete backends MUST honor this.

    Anything backend-specific (Ollama's `stop` strings, local's
    `eos_id`, an MLX model's quantization config) lives on the
    constructor of the concrete backend, not on `complete`. The
    interface stays small.
    """

    @property
    @abstractmethod
    def info(self) -> BackendInfo:
        """The backend's identity card. Must be cheap and idempotent."""

    @abstractmethod
    def complete(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> InferenceResult:
        """Run one prompt through the backend and return the result.

        Args:
            prompt: the input string. Non-empty.
            max_new_tokens: maximum tokens to generate. Backends are
                expected to honor this as an UPPER bound; they may
                return fewer tokens (e.g. on EOS, on stop string, on
                server-side cap).
            temperature: softmax temperature on `[0, ∞)`. `0.0` is
                greedy. `1.0` is the model's native distribution.
            top_k: keep only the k most-likely tokens at each step,
                or `None` to disable.
            top_p: nucleus sampling threshold, or `None` to disable.

        Returns:
            `InferenceResult` — see the dataclass for fields.
        """
