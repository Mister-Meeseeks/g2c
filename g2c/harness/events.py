"""The event log — the harness's source of truth — and its small types.

One design decision generates almost everything in this module: the
trajectory is an APPEND-ONLY event log, and everything else — the
context window, resume state, the audit trail — is a view of it.
`EventLog` is deliberately primitive: JSON lines on disk, append and
replay, nothing else. For the process-crash exercises in this module,
the log plus the workspace hold the state needed to resume.

This is a teaching log, not a database. It does not fsync writes,
recover torn records, coordinate concurrent writers, or survive
filesystem loss. Its recovery claims assume one writer, an ordinary
process crash, and an intact filesystem.

Event types used by the harness (free-form strings, by convention):

    task          {"content": str}            — the task statement
    model_turn    {"completion": str}         — one backend completion
    tool_call     {"tool", "arguments"}       — intent, logged BEFORE execution
    tool_result   {"output", "is_error", "status"}
    approval      {"approved": bool}          — pre-authorization for ASK tools
    final_answer  {"content": str}
    stopped       {"reason": str}

All provided plumbing — nothing in this file is scaffolded.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any


class Permission(StrEnum):
    """What the harness lets a tool do without asking."""

    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


@dataclass(frozen=True)
class Event:
    """One immutable record in the trajectory log."""

    step: int
    type: str
    payload: dict[str, Any]
    call_id: str | None = None

    def to_json(self) -> str:
        return json.dumps(
            {
                "step": self.step,
                "type": self.type,
                "payload": self.payload,
                "call_id": self.call_id,
            },
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, line: str) -> Event:
        raw = json.loads(line)
        return cls(
            step=raw["step"],
            type=raw["type"],
            payload=raw["payload"],
            call_id=raw.get("call_id"),
        )


class EventLog:
    """Append-only JSONL trajectory log.

    `append` assigns the next step number and appends it to disk.
    `replay` re-reads the file, not a cache: resume after an ordinary
    process crash trusts only complete records that reached disk.

    Scope: one writer, intact filesystem, no power-loss guarantee.
    Production durability needs stronger storage semantics.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(
        self,
        type: str,
        payload: dict[str, Any],
        *,
        call_id: str | None = None,
    ) -> Event:
        """Append one event, assigning the next step number."""
        events = self.replay()
        event = Event(
            step=len(events), type=type, payload=payload, call_id=call_id
        )
        with self.path.open("a", encoding="utf-8") as f:
            f.write(event.to_json() + "\n")
        return event

    def replay(self) -> list[Event]:
        """Every event on disk, in order. Empty list for a fresh log."""
        if not self.path.exists():
            return []
        events: list[Event] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(Event.from_json(line))
        return events


@dataclass(frozen=True)
class Budgets:
    """The stopping rules — safety equipment, not configuration trivia."""

    max_steps: int = 12
    max_retries: int = 2
    max_repeats: int = 2
    context_tokens: int = 2000


@dataclass(frozen=True)
class RetryDecision:
    """What `classify_failure` decided about one failed tool call.

    kind is one of `"transient"` (retry with backoff), `"deterministic"`
    (surface to the model — retrying the same call cannot succeed), or
    `"budget"` (halt resumably).
    """

    kind: str
    retry: bool
    reason: str


# Marker tables for failure classification. Lowercase substrings, matched
# against the error output. Deliberately visible module constants: the
# classification policy should be readable, auditable data — not logic
# buried in a function.
TRANSIENT_MARKERS: tuple[str, ...] = (
    "timeout",
    "timed out",
    "busy",
    "temporarily",
    "connection reset",
    "try again",
)
DETERMINISTIC_MARKERS: tuple[str, ...] = (
    "not found",
    "no such",
    "does not exist",
    "permission denied",
    "invalid",
    "malformed",
    "unknown tool",
)


def estimate_tokens(text: str) -> int:
    """Cheap token estimate: ~4 characters per token, minimum 1.

    Good enough for budget arithmetic — the context policy needs
    "roughly how big," not a tokenizer round-trip.
    """
    return len(text) // 4 + 1
