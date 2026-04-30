"""End-to-end RAG pipeline — retrieve, assemble, generate.

`RAGPipeline` is the smallest object that turns a question into an
answer-with-citations:

  question → retriever → top-k chunks → prompt assembler → prompt
                                                             ↓
  RAGAnswer  ←  parsed completion  ←  backend.complete(prompt)

Three concrete pieces, plugged together:

  * A `DenseRetriever` (or any class with a `retrieve(query, k)` API)
    that produces `RetrievedChunk`s.
  * A `Backend` (Module 16) that turns a prompt into a completion.
  * A prompt template — defaulting to `assemble_rag_prompt`.

The pipeline class is implemented in full; like `DenseRetriever`,
it's wiring around the conceptual pieces. The lesson lives in the
components, not the orchestration.

The `RAGAnswer` dataclass keeps both halves of the result alongside
each other: the model's textual answer AND the chunks that produced
it. A user-facing UI might render the answer plus a "sources" footer;
an eval harness might check the answer for accuracy AND check whether
the cited chunks actually support the claim. Same data, different
consumers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from g2c.inference import Backend, InferenceResult

from .prompt import (
    DEFAULT_INSTRUCTION,
    DEFAULT_SYSTEM,
    RAGPrompt,
    assemble_rag_prompt,
)
from .retrieve import DenseRetriever, RetrievedChunk


@dataclass
class RAGAnswer:
    """The output of a `RAGPipeline.answer` call.

    Attributes:
        question: the user's original question.
        answer: the model's response text. Comes directly from
            `InferenceResult.completion` — no post-processing.
        retrieved: the `RetrievedChunk`s the retriever returned, in
            ranked order. Same chunks that were spliced into the
            prompt; the indices line up with the citation markers
            (`retrieved[0].chunk` ↔ citation `[1]`).
        prompt: the full assembled prompt (a `RAGPrompt`). Useful for
            debugging — "is the model failing because the prompt is
            wrong, or because the chunks are wrong?"
        inference: the underlying `InferenceResult` from the backend.
            Carries latency, token counts, and backend identity. Lets
            a `RAGPipeline` benchmark drop-in to the Module 16
            harness.
        metadata: free-form per-call extras. The pipeline puts the
            retrieval `k`, the chunk count, and the backend's name
            here.

    Mutable so callers can attach post-hoc analysis (e.g. an eval
    harness might add `metadata['eval'] = {'correct': True, ...}`).
    """

    question: str
    answer: str
    retrieved: list[RetrievedChunk]
    prompt: RAGPrompt
    inference: InferenceResult
    metadata: dict[str, Any] = field(default_factory=dict)


class RAGPipeline:
    """Compose retriever + backend + prompt template into one callable.

    Args:
        retriever: a `DenseRetriever`. Could be any object with a
            `retrieve(query: str, *, k: int) -> list[RetrievedChunk]`
            method; we type-hint `DenseRetriever` for clarity but
            don't enforce.
        backend: an inference `Backend` from Module 16 — local or
            Ollama.
        system: the system header for `assemble_rag_prompt`. Defaults
            to `DEFAULT_SYSTEM`.
        instruction: the trailing "say I don't know" instruction for
            `assemble_rag_prompt`. Defaults to `DEFAULT_INSTRUCTION`.

    `answer(question, *, k=5, max_new_tokens=256, ...)` runs the full
    pipeline and returns a `RAGAnswer`.

    The pipeline is intentionally synchronous. Streaming, async, and
    batched answering are real production features but don't change
    the conceptual shape — same as the Module 16 reasoning for keeping
    `Backend.complete` synchronous.

    Fully implemented — once the components in `chunk.py`, `embed.py`,
    `store.py`, and `prompt.py` are filled in, this works without
    further code from the student.
    """

    def __init__(
        self,
        retriever: DenseRetriever,
        backend: Backend,
        *,
        system: str = DEFAULT_SYSTEM,
        instruction: str = DEFAULT_INSTRUCTION,
    ) -> None:
        if retriever is None:
            raise ValueError("RAGPipeline requires a non-None retriever")
        if backend is None:
            raise ValueError("RAGPipeline requires a non-None backend")

        self._retriever = retriever
        self._backend = backend
        self._system = system
        self._instruction = instruction

    @property
    def retriever(self) -> DenseRetriever:
        return self._retriever

    @property
    def backend(self) -> Backend:
        return self._backend

    def answer(
        self,
        question: str,
        *,
        k: int = 5,
        max_new_tokens: int = 256,
        temperature: float = 0.2,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> RAGAnswer:
        """Run retrieval + generation for one question.

        Args:
            question: the user's question. Non-empty.
            k: number of chunks to retrieve. Defaults to 5.
            max_new_tokens: forwarded to `backend.complete`.
            temperature: forwarded. Default `0.2` — RAG benefits from
                near-greedy sampling because the answer is supposed
                to be grounded in retrieved facts, not creatively
                generated.
            top_k: forwarded.
            top_p: forwarded.

        Returns:
            `RAGAnswer` with the model's response, the retrieved
            chunks, the assembled prompt, and the underlying
            `InferenceResult`.
        """
        if not isinstance(question, str) or len(question) == 0:
            raise ValueError("answer: question must be a non-empty str")
        if k <= 0:
            raise ValueError(f"answer: k must be > 0, got {k}")

        retrieved = self._retriever.retrieve(question, k=k)
        prompt = assemble_rag_prompt(
            question,
            retrieved,
            system=self._system,
            instruction=self._instruction,
        )
        inference = self._backend.complete(
            prompt.text,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        return RAGAnswer(
            question=question,
            answer=inference.completion,
            retrieved=list(retrieved),
            prompt=prompt,
            inference=inference,
            metadata={
                "k": k,
                "n_retrieved": len(retrieved),
                "backend_name": self._backend.info.name,
                "backend_model_id": self._backend.info.model_id,
            },
        )

    def __repr__(self) -> str:
        return (
            f"RAGPipeline(retriever={self._retriever!r}, "
            f"backend={self._backend.info.name!r}/"
            f"{self._backend.info.model_id!r})"
        )
