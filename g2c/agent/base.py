"""Agent — the dataclasses that describe one ReAct-style step and one run.

A ReAct agent's per-step record carries four pieces:

  * the model's THOUGHT (free-form reasoning text),
  * the model's ACTION (tool name + arguments) — or `None` if the
    model emitted a final answer instead,
  * the OBSERVATION (the tool result, fed back by the runtime) — or
    `None` if there was no action,
  * the FINAL ANSWER (user-facing text) — populated only on the last
    step when the model decides it's done.

`AgentStep` records all four. `AgentRunResult` is the end-of-run
roll-up — every step, the final answer, the stop reason, and a
metadata dict for run-level extras.

`Plan` is the (optional) planning artifact. The agent can call the
backend once before entering the main loop to produce a structured
plan — a goal + a list of subgoals. The plan is rendered into the
prompt and gives the model a target to aim at.

`AgentError` is the internal exception type. Raised when something
unrecoverable happens (e.g., the parser is asked to validate
something that violates its contract). Loop-level errors (tool
failure, parse failure, unknown tool) are surfaced as data on the
step record, NOT raised — the agent's whole point is graceful
degradation in the face of model wobble.

This file is pure dataclasses + one exception class. The interesting
code lives in:

  * `g2c/agent/parser.py` — extracting Thought/Action/Observation/Final Answer
  * `g2c/agent/memory.py` — accumulating steps and rendering the scratchpad
  * `g2c/agent/planner.py` — the optional planning step
  * `g2c/agent/agent.py` — the main loop
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from g2c.inference import InferenceResult


class AgentError(Exception):
    """Raised when the agent's internal contract is violated.

    Examples: a Plan constructed with empty steps, a Scratchpad asked
    to render with mismatched Thought/Action/Observation triples, an
    Agent constructed with a non-Backend backend.

    NOT raised on tool failures, parse failures, or model misbehavior
    during a `run`. Those are surfaced on the `AgentStep` (with
    `parse_error` or an error-flagged Observation) so the loop can
    recover. The agent's robustness depends on the loop never crashing
    on a single weird step.
    """


@dataclass(frozen=True)
class Action:
    """A parsed tool invocation extracted from the model's completion.

    Attributes:
        tool: the tool name the model asked to invoke. Matched against
            `ToolRegistry` by exact-string lookup at dispatch time.
        arguments: the dict the model emitted as `Action Input:`. NOT
            yet validated — the dispatcher runs `validate_arguments`
            before calling the tool.

    Frozen because once the parser produces an Action, downstream code
    (the dispatcher, the scratchpad renderer) treats it as a fixed
    record of "this is what the model asked for at step N." Mutating
    in place would obscure history.
    """

    tool: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class Observation:
    """The observed result of executing one Action.

    Attributes:
        output: stringified output. For successful tool calls, this is
            `str(tool.func(**args))`. For errors (unknown tool, bad
            args, tool raised), this is the error message — wrapped as
            a string so the model can read it and try a different
            approach on the next step.
        is_error: True if the action failed. The scratchpad renders
            errors with a `[error]` prefix so the model is more likely
            to recognize them as recovery signals rather than parrot
            them back as if they were successful answers.
    """

    output: str
    is_error: bool = False


@dataclass
class AgentStep:
    """One iteration of the observe → think → act loop.

    Attributes:
        completion: the raw model output for this step. Includes the
            Thought / Action / Action Input markers, exactly as the
            backend emitted them. Useful for debugging when parsing
            went wrong.
        thought: the parsed Thought text, or empty string if the
            model didn't produce one. ReAct prompts strongly bias the
            model toward emitting Thought at every step, but
            instruction-tuned models occasionally skip it on a clear
            final answer.
        action: the parsed Action, or `None` if the model emitted a
            Final Answer instead of asking for a tool call. Exactly
            one of `action` and `final_answer` is non-None on a
            well-formed step.
        observation: the Observation produced by dispatching `action`.
            `None` when there was no action.
        final_answer: the user-facing answer text, or `None` if the
            model didn't emit a Final Answer at this step.
        parse_error: a string describing what went wrong if the parser
            couldn't extract structured content from `completion`.
            `None` on a well-formed step. When set, both `action` and
            `final_answer` are `None`; the loop treats this as a stuck
            step and either retries with a recovery prompt or stops.
        inference: the underlying `InferenceResult` from the backend.
            Carries token counts and latency. Lets a multi-step run
            report compute costs step-by-step.

    Mutable — the loop builds steps incrementally as it iterates.
    """

    completion: str
    thought: str
    action: Action | None
    observation: Observation | None
    final_answer: str | None
    parse_error: str | None
    inference: InferenceResult

    # --- Named constructors for the three legal step shapes -------------
    # A well-formed step is exactly one of: a Final Answer, an Action +
    # Observation, or a stuck step (parse failure). These factories make
    # that trichotomy explicit and fill in the mechanical fields for you:
    # the always-`None`s that distinguish the shapes, and `completion`,
    # which is always `inference.completion`. Build steps with these in
    # the loop so the loop reads as *policy* (which shape, when) rather
    # than seven-field record-keeping. Constructing `AgentStep(...)`
    # directly still works; the factories are sugar, not a gate.

    @classmethod
    def final(
        cls, inference: InferenceResult, *, thought: str, final_answer: str
    ) -> AgentStep:
        """A clean exit: the model emitted a Final Answer, no action."""
        return cls(
            completion=inference.completion,
            thought=thought,
            action=None,
            observation=None,
            final_answer=final_answer,
            parse_error=None,
            inference=inference,
        )

    @classmethod
    def act(
        cls,
        inference: InferenceResult,
        *,
        thought: str,
        action: Action,
        observation: Observation,
    ) -> AgentStep:
        """An action step: the model called a tool and got an Observation."""
        return cls(
            completion=inference.completion,
            thought=thought,
            action=action,
            observation=observation,
            final_answer=None,
            parse_error=None,
            inference=inference,
        )

    @classmethod
    def stuck(
        cls, inference: InferenceResult, *, thought: str, parse_error: str
    ) -> AgentStep:
        """A stuck step: parser found neither an Action nor a Final Answer."""
        return cls(
            completion=inference.completion,
            thought=thought,
            action=None,
            observation=None,
            final_answer=None,
            parse_error=parse_error,
            inference=inference,
        )


@dataclass(frozen=True)
class StepOutcome:
    """What the agent's per-step policy decided for one iteration.

    `Agent._decide_step` (the scaffolded Module-19 deliverable) classifies
    a parsed step and returns a `StepOutcome`. The driver (`_run_loop`,
    provided) consumes it mechanically: it always appends `step` to the
    run's step list, appends to the scratchpad iff `remember`, and stops
    the loop iff `stop_reason` is set. Keeping the decision PURE — a
    value returned, not a side effect on the loop's state — means the
    policy can be unit-tested in isolation: feed it a `ParsedStep`, assert
    the `StepOutcome`, with no scratchpad or step-list to thread.

    Attributes:
        step: the `AgentStep` record for this iteration. Always present —
            even a stuck step is recorded, so the run's history is
            complete.
        stop_reason: if set, the loop halts and this becomes the run's
            `stopped_reason` (`"final_answer"`, `"duplicate_action"`, or
            `"no_progress"`). `None` keeps the loop going.
        remember: whether the driver should append `step` to the
            scratchpad so the NEXT prompt can see it. Action steps and
            retried stuck steps are remembered; a final answer and a
            halted stuck step are not (the loop is ending — nothing reads
            the scratchpad again).
    """

    step: AgentStep
    stop_reason: str | None = None
    remember: bool = True


@dataclass
class Plan:
    """An (optional) plan produced by the planning phase.

    Attributes:
        goal: a one-line restatement of the user's task. Useful for
            the agent to keep "why am I doing this" in scope when the
            scratchpad gets long.
        steps: ordered subgoals. Each is a single sentence describing
            something to do. The agent doesn't strictly follow them —
            the model is free to deviate when an Observation suggests
            a better approach — but the plan gives the model a target
            and reduces aimless tool-calling.

    A run with `plan=None` skips the planning phase and goes straight
    into reactive ReAct mode. Both modes are valid; the plan adds
    value when the task has obvious structure ("read X, transform Y,
    write Z") and adds noise when the task is one-shot ("what's 2+2").

    The plan is mutable so callers can patch it post-construction
    (e.g., a UI lets the user edit before running). The agent doesn't
    mutate it during `run`.
    """

    goal: str
    steps: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.goal, str) or not self.goal:
            raise AgentError("Plan.goal must be a non-empty str")
        if not isinstance(self.steps, list):
            raise AgentError("Plan.steps must be a list")
        for i, s in enumerate(self.steps):
            if not isinstance(s, str) or not s:
                raise AgentError(
                    f"Plan.steps[{i}] must be a non-empty str, got {s!r}"
                )


@dataclass
class AgentRunResult:
    """The full result of one `Agent.run` call.

    Attributes:
        user_message: the user's original message.
        plan: the Plan produced by the planning phase, or `None` if
            planning was skipped or failed (in which case the run
            still proceeds without a plan).
        final_answer: the model's emitted Final Answer, or `None` if
            the loop hit `max_steps` / a stuck step without producing
            one.
        steps: every AgentStep of the run, in order. Includes any
            steps with `parse_error` set.
        stopped_reason: one of:
            * `"final_answer"` — model emitted Final Answer (clean exit)
            * `"max_steps"` — loop ran out of iterations
            * `"no_progress"` — model emitted neither action nor final
              answer (stuck step) and the agent is configured to halt
              on stuck instead of retrying
            * `"duplicate_action"` — model called the same action with
              the same args twice in a row (loop detection)
        metadata: free-form per-run extras. The loop fills in step
            counts, total tool calls, the registry's tool list, the
            backend's identity, and the planner's outputs.

    Mutable so callers can attach post-hoc analysis (e.g. an eval
    harness might add `metadata['eval'] = {'correct': True, ...}`).
    """

    user_message: str
    plan: Plan | None
    final_answer: str | None
    steps: list[AgentStep]
    stopped_reason: str
    metadata: dict[str, Any] = field(default_factory=dict)
