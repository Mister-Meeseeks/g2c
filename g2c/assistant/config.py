"""AssistantConfig — the knobs for the capstone assistant.

One dataclass + one exception type. Everything that earlier modules
exposed as a per-call argument (sampling temperature, agent loop
limits, RAG retrieval depth, conversation truncation) lives here so
that swapping a config swaps the assistant's behavior end to end.

Why a config object instead of a constructor with 15 keyword args?

  * **Single source of truth.** The CLI loads config from a file, the
    eval harness loads it from the test fixture, the notebook builds
    it inline. All paths converge on one dataclass.

  * **Round-trippable to disk.** `dataclasses.asdict(config)` is one
    line; persisting a config so that "this run is reproducible" is
    free. Tests do this; the CLI's `/save` does this.

  * **Default-ful.** Every field has a sensible default. The minimum
    viable invocation is `AssistantConfig()`.

This file is fully implemented. The interesting code lives in
`assistant.py` (the orchestration) and `conversation.py` (the
multi-turn memory primitive).
"""
from __future__ import annotations

from dataclasses import dataclass


class AssistantError(Exception):
    """Raised on misconfigured assistant arguments.

    Distinct from `AgentError` (Module 19). The agent's errors are
    about the loop's internal contracts; the assistant's errors are
    about the integration layer above the loop — bad config, missing
    retriever for RAG-on, etc.
    """


@dataclass
class AssistantConfig:
    """Configuration for the capstone assistant.

    Attributes:
        name: a short human-readable label for this assistant
            instance. Shown in the CLI banner and saved with
            transcripts. Default `"g2c-assistant"`.

        max_steps: forwarded to `Agent` — cap on the ReAct loop
            iterations per turn. Default 8.

        plan: whether the agent runs the planning phase before the
            ReAct loop. Default True. Disable for one-shot tasks
            where planning latency is pure overhead.

        loop_detection: forwarded to `Agent`. Default True.

        halt_on_stuck: forwarded to `Agent`. Default False (let the
            model recover from a stuck step).

        max_new_tokens: forwarded to backend.complete. 512 is enough
            for the agent's per-step Thought/Action/Action Input
            block plus a typical Final Answer.

        temperature: forwarded. Default 0.2 — near-greedy, which
            biases toward consistent format and tool selection. The
            ReAct paper's default.

        top_k, top_p: forwarded. Default `None` (disabled).

        rag_enabled: whether to call the retriever before each turn
            and splice retrieved chunks into the prompt as context.
            Default True. Ignored if no retriever was passed to
            `Assistant`. Disable per-turn via `Assistant.chat(...,
            use_rag=False)` if needed.

        rag_k: number of chunks to retrieve per turn. Default 5.

        max_history_messages: cap on the conversation messages that
            get rendered into each turn's prompt. The conversation
            object stores everything; this just controls how much is
            shown to the model. `None` means unlimited (rely on the
            backend's context window). Default 20 — generous for a
            chat assistant; production systems often summarize older
            messages instead of dropping them.

        scratchpad_max_chars: forwarded to the agent's `Scratchpad`.
            `None` means unlimited. Default `None`. Mostly a no-op on
            the native channel — message-list memory is the relevant
            pressure there, not scratchpad text length.

        use_native: which agent channel to use under the hood.
            `True` (default) uses `NativeAgent` — structured tool
            calling via `backend.chat_with_tools`. `False` uses
            `Agent` — ReAct text format. The default tracks Module
            19's recommendation: native is what models are actually
            post-trained for. Set `False` to compare or to run on a
            backend that only implements `complete`.
    """

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
    use_native: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise AssistantError("AssistantConfig.name must be a non-empty str")
        if self.max_steps <= 0:
            raise AssistantError(
                f"AssistantConfig.max_steps must be > 0, got {self.max_steps}"
            )
        if self.max_new_tokens <= 0:
            raise AssistantError(
                f"AssistantConfig.max_new_tokens must be > 0, "
                f"got {self.max_new_tokens}"
            )
        if self.temperature < 0.0:
            raise AssistantError(
                f"AssistantConfig.temperature must be >= 0, "
                f"got {self.temperature}"
            )
        if self.rag_k <= 0:
            raise AssistantError(
                f"AssistantConfig.rag_k must be > 0, got {self.rag_k}"
            )
        if (
            self.max_history_messages is not None
            and self.max_history_messages <= 0
        ):
            raise AssistantError(
                f"AssistantConfig.max_history_messages must be > 0 or None, "
                f"got {self.max_history_messages}"
            )
        if (
            self.scratchpad_max_chars is not None
            and self.scratchpad_max_chars <= 0
        ):
            raise AssistantError(
                f"AssistantConfig.scratchpad_max_chars must be > 0 or None, "
                f"got {self.scratchpad_max_chars}"
            )
