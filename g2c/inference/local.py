"""LocalTransformerBackend — wraps the model you trained yourself.

The simplest concrete `Backend`: a `TransformerLM` from Module 09
(or anything with the same forward shape) plus a tokenizer from
Module 04 plus the sampling primitives from Module 11. Everything is
already in your `g2c/` package; this class is just the connective
tissue.

The whole point of having ONE class for this — instead of importing
`generate` and decoding the result inline — is the uniform interface.
The Module 17 RAG pipeline doesn't want to care whether the
underlying LM is your tiny model or a 7B Ollama-served Llama. It just
calls `backend.complete(prompt, max_new_tokens=64)` and reads the
result. This file is what makes that possible.

What the class does in `complete`:

  1. Tokenizes the prompt with `self._tokenizer.encode`.
  2. Times the call with `time.perf_counter()` (high-resolution
     monotonic; the standard idiom).
  3. Calls `g2c.sampling.generate(model, prompt_ids, ...)` with the
     four uniform sampling parameters mapped through, plus the
     `eos_id` that was configured at construction.
  4. Slices off the prompt from the returned full sequence to get
     just the new tokens.
  5. Decodes the new tokens with `self._tokenizer.decode`.
  6. Builds and returns the `InferenceResult`.

Boilerplate (constructor + info property): implemented. Wiring
(`complete`): scaffolded.

Why is `generate` imported at module top instead of injected? Because
the local backend's job IS to wrap your sampling implementation; an
injected closure would let you point this at a different generation
function, but that's also what subclassing is for. Keeping the import
at module top makes the dependency obvious and lets tests
monkey-patch it (`monkeypatch.setattr('g2c.inference.local.generate',
fake)`) without threading a kwarg through.
"""
from __future__ import annotations

import time
from typing import Any

import torch

from g2c.sampling import generate

from .backend import Backend, BackendInfo, InferenceResult


class LocalTransformerBackend(Backend):
    """A `Backend` over your own `TransformerLM` + tokenizer.

    Args:
        model: any module with `forward(x: (1, T)) -> (1, T, V)` plus
            a `max_seq_len: int` attribute. `g2c.transformer.TransformerLM`
            is the prototype.
        tokenizer: any object with `encode(s: str) -> list[int]` and
            `decode(ids: list[int]) -> str`. `g2c.tokenizer.bpe.BPETokenizer`
            is the prototype.
        model_id: an identity label for the checkpoint, e.g. `"g2c-tiny-20m"`.
            Stored on `self.info` and copied into every `InferenceResult`.
        eos_id: optional end-of-sequence token id. Passed through to
            `g2c.sampling.generate`. Generation stops as soon as this
            token is sampled.
        name: backend kind label. Defaults to `"local"`. Override only
            if you're subclassing for a different kind of local backend.
        extra: additional metadata for the `BackendInfo.extra` dict.
            Useful for recording dtype, device, training run id, etc.

    The `info` property returns a `BackendInfo` snapshot built from
    the constructor args. The local backend doesn't enrich it with
    runtime info (device probe, dtype probe) by default; if you want
    that, pass `extra={"device": str(...), "dtype": str(...)}` at
    construction.

    The boilerplate (constructor + `info`) is implemented. The
    `complete` method is scaffolded — that's the lesson.
    """

    def __init__(
        self,
        model,
        tokenizer,
        *,
        model_id: str = "g2c-local",
        eos_id: int | None = None,
        name: str = "local",
        extra: dict[str, Any] | None = None,
    ) -> None:
        if model is None:
            raise ValueError("LocalTransformerBackend requires a non-None model")
        if tokenizer is None:
            raise ValueError("LocalTransformerBackend requires a non-None tokenizer")
        if not hasattr(tokenizer, "encode") or not hasattr(tokenizer, "decode"):
            raise TypeError(
                "tokenizer must have .encode and .decode methods; got "
                f"{type(tokenizer).__name__}"
            )
        if eos_id is not None and not isinstance(eos_id, int):
            raise TypeError(
                f"eos_id must be int or None, got {type(eos_id).__name__}"
            )

        self._model = model
        self._tokenizer = tokenizer
        self._eos_id = eos_id
        self._info = BackendInfo(
            name=name,
            model_id=model_id,
            extra=dict(extra) if extra is not None else {},
        )

    @property
    def info(self) -> BackendInfo:
        return self._info

    @property
    def model(self):
        """The wrapped model. Exposed for advanced callers (e.g. eval
        harnesses that need direct model access for log-prob scoring)."""
        return self._model

    @property
    def tokenizer(self):
        """The wrapped tokenizer."""
        return self._tokenizer

    def complete(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> InferenceResult:
        """Encode → time → generate → slice → decode → return.

        Args:
            prompt: the input string. Non-empty after tokenization.
            max_new_tokens: forwarded to `g2c.sampling.generate`.
            temperature: forwarded.
            top_k: forwarded.
            top_p: forwarded.

        Returns:
            `InferenceResult` with:
              * prompt = the input string
              * completion = the decoded text of the NEW tokens (does
                not include the prompt)
              * prompt_tokens = number of tokens in the encoded prompt
              * completion_tokens = number of NEW tokens generated
                (not including the prompt)
              * latency_ms = wall-clock time around the generate call
              * backend = self.info
              * metadata = {"sampling": {...the four params...},
                            "eos_id": self._eos_id}

        Recipe:
            1. # Validate.
               if not isinstance(prompt, str):
                   raise TypeError(...)
               if max_new_tokens <= 0:
                   raise ValueError(...)

            2. # Encode.
               prompt_ids: list[int] = self._tokenizer.encode(prompt)
               if len(prompt_ids) == 0:
                   raise ValueError("prompt encoded to 0 tokens")

            3. # Move to the model's device. Same idiom as
               # continuation_logprob:
               device = next(iter(self._model.parameters())).device
               prompt_tensor = torch.tensor(
                   prompt_ids, dtype=torch.long, device=device
               )

            4. # Time the generate call. Use perf_counter — high-
               # resolution and monotonic. Don't use time.time() here.
               t0 = time.perf_counter()
               full = generate(
                   self._model,
                   prompt_tensor,
                   max_new_tokens=max_new_tokens,
                   temperature=temperature,
                   top_k=top_k,
                   top_p=top_p,
                   eos_id=self._eos_id,
               )
               t1 = time.perf_counter()
               latency_ms = (t1 - t0) * 1000.0

            5. # Slice the prompt off and decode the rest.
               # `full` is 1-D shape (T_prompt + n_new,).
               completion_ids = full[len(prompt_ids):].tolist()
               completion = self._tokenizer.decode(completion_ids)

            6. # Build and return the result.
               return InferenceResult(
                   prompt=prompt,
                   completion=completion,
                   prompt_tokens=len(prompt_ids),
                   completion_tokens=len(completion_ids),
                   latency_ms=latency_ms,
                   backend=self._info,
                   metadata={
                       "sampling": {
                           "max_new_tokens": max_new_tokens,
                           "temperature": temperature,
                           "top_k": top_k,
                           "top_p": top_p,
                       },
                       "eos_id": self._eos_id,
                   },
               )

        Implementation notes:

          * `time.perf_counter()` is the right call. It's high-
            resolution (sub-millisecond on macOS), monotonic (immune
            to wall-clock NTP adjustments), and the documented
            standard for benchmarking. `time.time()` is wall clock —
            wrong tool.

          * The `(t1 - t0) * 1000.0` conversion is in float
            milliseconds, matching `InferenceResult.latency_ms`.

          * `generate` is wrapped in `@torch.no_grad()` already
            (Module 11), so there's no need to add another
            `no_grad` here. The model gets called inside no_grad;
            no autograd graph is built.

          * If `len(prompt_ids) >= model.max_seq_len` the model has
            no positional capacity for new tokens. `generate`'s
            crop step handles this for the forward pass, but the
            FIRST sampled token has no positional signal beyond
            what fit in the crop window. For pedagogical safety,
            we don't reject — we let `generate` deal with it. In
            production you'd typically refuse, log, and ask the
            caller to truncate.

          * The `metadata` dict is intentionally rich — keeps the
            sampling settings with the result so a JSON dump tells
            you "what we sampled with" alongside "what we got."

        Sanity values:

          * Greedy (temperature=0) on a deterministic model: same
            prompt always produces the same completion. The
            `completion_tokens` count equals exactly `max_new_tokens`
            unless `eos_id` was emitted earlier.

          * `max_new_tokens=1` on any prompt: `completion_tokens`
            is exactly 1 (or 0 if EOS was the first sampled token).

          * Empty prompt: raises ValueError before forwarding.
        """
        # TODO
        raise NotImplementedError

    def __repr__(self) -> str:
        return (
            f"LocalTransformerBackend(name={self._info.name!r}, "
            f"model_id={self._info.model_id!r})"
        )


def _model_device(model) -> torch.device:
    device = getattr(model, "device", None)
    if isinstance(device, torch.device):
        return device
    parameters = getattr(model, "parameters", None)
    if parameters is None:
        return torch.device("cpu")
    try:
        parameter = next(iter(parameters()))
    except (StopIteration, TypeError):
        return torch.device("cpu")
    return parameter.device
