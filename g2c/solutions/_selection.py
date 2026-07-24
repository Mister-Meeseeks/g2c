"""Map course-module selectors onto the mirror files `apply()` should bind.

Why this exists: the course is a linear chain — every module imports the one
before it. A student whose Module 07 attention has a subtle bug finds out in
Module 12, when training silently produces garbage. Before per-module
selection, their only recovery was `G2C_APPLY_SOLUTIONS=1`, which replaces
*every* implementation across all twenty modules, including the eleven they
got right. That is a bad trade, and the moment people quit.

This module lets a student hand back only the modules they choose:

    ./notebook.sh 12 --solutions=01-07     # reference 01-07, mine from 08 on
    G2C_APPLY_SOLUTIONS=01-07 pytest       # same selector, for tests

Selectors accepted (comma-separated, in any mix):

    07              a course module id ("7", "07", "3b", "09b" all work)
    01-07           an inclusive range in course order (01, 02, 03, 03b, ... 07)
    attention       a package topic — every mirror file under it
    attention.multi_head    a single mirror file
    all             everything (what `G2C_APPLY_SOLUTIONS=1` means)

Granularity is per *file*, not per topic, because topics do not map 1:1 to
modules. Modules 07 and 08 share `g2c/attention/`; a topic-level filter would
hand a student Module 08's multi-head answer when they only asked for 07.

Two files are split at the *member* level for the same reason:
`TransformerLM.forward` is Module 09's centerpiece deliverable while
`TransformerLM.forward_cached` belongs to Module 16's cached-decoding path,
and they live in the same class. Selecting one must not leak the other.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Target:
    """One mirror file, optionally restricted to specific members.

    `mirror` is the dotted path under `g2c.solutions` (e.g.
    `"transformer.transformer_lm"`). `include` limits binding to the named
    members; `exclude` binds everything except them. A member is bound if any
    selected target allows it, so overlapping selections union rather than
    fight.
    """

    mirror: str
    include: frozenset[str] | None = None
    exclude: frozenset[str] = field(default_factory=frozenset)

    def allows(self, member: str) -> bool:
        if self.include is not None:
            return member in self.include
        return member not in self.exclude


def _t(mirror: str, *, include: Iterable[str] | None = None,
       exclude: Iterable[str] = ()) -> Target:
    return Target(
        mirror=mirror,
        include=None if include is None else frozenset(include),
        exclude=frozenset(exclude),
    )


# `TransformerLM` is built across two modules: `forward` in Module 09, the
# cached-decoding `forward_cached` in Module 16 (see docs/modules/09 —
# "leaving test_transformer_lm_forward_cached_* red for now is expected").
_TRANSFORMER_LM_09 = _t("transformer.transformer_lm", exclude={"forward_cached"})
_TRANSFORMER_LM_16 = _t("transformer.transformer_lm", include={"forward_cached"})

# Ordered by course sequence — ranges slice this. Module 12 (scaling) ships no
# package deliverable; it is experiments over Module 10's trainer, so it maps
# to nothing and exists here only to keep ranges spanning it well-defined.
MODULE_TARGETS: dict[str, tuple[Target, ...]] = {
    "01": (_t("autodiff.value"), _t("autodiff.nn"), _t("autodiff.grad_check")),
    "02": (_t("tensors.matmul"), _t("tensors.broadcasting"), _t("tensors.forward")),
    "03": (_t("nn.modules"), _t("nn.loss"), _t("nn.optim"), _t("nn.train")),
    "03b": (_t("training.optim"), _t("training.clip"), _t("training.schedule")),
    "04": (_t("tokenizer.bpe"),),
    "05": (
        _t("embeddings.token"),
        _t("embeddings.positional"),
        _t("embeddings.rotary"),
        _t("embeddings.skipgram"),
        _t("embeddings.similarity"),
    ),
    "06": (_t("lm.bigram"), _t("lm.mlp"), _t("lm.train")),
    "07": (_t("attention.self_attention"),),
    "08": (_t("attention.multi_head"),),
    "09": (
        _t("transformer.layer_norm"),
        _t("transformer.ffn"),
        _t("transformer.block"),
        _TRANSFORMER_LM_09,
    ),
    "09b": (_t("pretraining.loss"),),
    "10": (_t("pretraining.trainer"),),
    "11": (
        _t("sampling.temperature"),
        _t("sampling.top_k"),
        _t("sampling.top_p"),
        _t("sampling.repetition_penalty"),
        _t("sampling.generate"),
    ),
    "12": (),
    "13": (
        _t("sft.chat_template"),
        _t("sft.data"),
        _t("sft.loss"),
        _t("sft.trainer"),
    ),
    "14": (_t("dpo.data"), _t("dpo.loss"), _t("dpo.trainer")),
    "15": (
        _t("eval.match"),
        _t("eval.logprob"),
        _t("eval.multiple_choice"),
        _t("eval.calibration"),
    ),
    "16": (
        _t("inference.local"),
        _t("inference.ollama"),
        _t("inference.benchmark"),
        _t("transformer.kv_cache"),
        _TRANSFORMER_LM_16,
    ),
    "17": (_t("rag.chunk"), _t("rag.embed"), _t("rag.store"), _t("rag.prompt")),
    "18": (
        _t("tools.schema"),
        _t("tools.parser"),
        _t("tools.builtins"),
        _t("tools.loop"),
    ),
    "19": (
        _t("agent.parser"),
        _t("agent.memory"),
        _t("agent.planner"),
        _t("agent.agent"),
    ),
    "20": (_t("assistant.conversation"), _t("assistant.assistant")),
}

MODULE_ORDER: tuple[str, ...] = tuple(MODULE_TARGETS)

# Every topic that appears in the map, e.g. "attention", "agent".
TOPICS: frozenset[str] = frozenset(
    target.mirror.split(".")[0]
    for targets in MODULE_TARGETS.values()
    for target in targets
)

# Every mirror file, e.g. "attention.multi_head".
MIRRORS: frozenset[str] = frozenset(
    target.mirror
    for targets in MODULE_TARGETS.values()
    for target in targets
)

# Values of G2C_APPLY_SOLUTIONS that mean "everything", preserving the
# original `G2C_APPLY_SOLUTIONS=1` behavior.
_ALL_TOKENS = frozenset({"1", "all", "true", "yes", "on"})
# Explicit off switches. Without these, `G2C_APPLY_SOLUTIONS=0` would be a
# non-empty string and so apply everything — the opposite of what it reads as.
_OFF_TOKENS = frozenset({"", "0", "false", "no", "off"})


class SelectionError(ValueError):
    """Raised when a selector names nothing in the course."""


def normalize_module(token: str) -> str | None:
    """Return the canonical module id for `token`, or None if it isn't one.

    Accepts the same shapes the notebook launcher does: `7`, `07`, `3b`, `03B`.
    """
    candidate = token.strip().lower()
    if not candidate:
        return None
    # Zero-pad a leading single digit: "7" -> "07", "3b" -> "03b".
    if candidate[0].isdigit() and (len(candidate) == 1 or not candidate[1].isdigit()):
        candidate = "0" + candidate
    return candidate if candidate in MODULE_TARGETS else None


def _expand_range(token: str) -> list[str] | None:
    """Expand `01-07` into the module ids spanning it in course order."""
    if token.count("-") != 1:
        return None
    left, right = (part.strip() for part in token.split("-"))
    start, stop = normalize_module(left), normalize_module(right)
    if start is None or stop is None:
        return None
    i, j = MODULE_ORDER.index(start), MODULE_ORDER.index(stop)
    if i > j:
        i, j = j, i
    return list(MODULE_ORDER[i : j + 1])


def _targets_for_token(token: str) -> list[Target]:
    """Resolve one selector to its targets. Raises SelectionError if unknown."""
    token = token.strip()
    if not token:
        return []

    module = normalize_module(token)
    if module is not None:
        return list(MODULE_TARGETS[module])

    expanded = _expand_range(token)
    if expanded is not None:
        return [t for m in expanded for t in MODULE_TARGETS[m]]

    lowered = token.lower()
    if lowered in TOPICS:
        return [
            t
            for targets in MODULE_TARGETS.values()
            for t in targets
            if t.mirror.split(".")[0] == lowered
        ]

    if lowered in MIRRORS:
        return [
            t
            for targets in MODULE_TARGETS.values()
            for t in targets
            if t.mirror == lowered
        ]

    raise SelectionError(
        f"unknown solutions selector {token!r}.\n"
        f"  modules: {', '.join(MODULE_ORDER)}\n"
        f"  ranges:  e.g. 01-07\n"
        f"  topics:  {', '.join(sorted(TOPICS))}\n"
        f"  or 'all' for every module."
    )


def resolve(selectors: Iterable[str]) -> dict[str, list[Target]]:
    """Map selectors to `{full mirror module name: targets}`.

    Overlapping selectors union: selecting both 09 and 16 yields a
    `transformer.transformer_lm` entry holding both the exclude-side and
    include-side targets, so every member ends up bound.
    """
    resolved: dict[str, list[Target]] = {}
    for selector in selectors:
        for target in _targets_for_token(selector):
            full = f"g2c.solutions.{target.mirror}"
            resolved.setdefault(full, []).append(target)
    return resolved


def parse_selectors(value: str) -> list[str] | None:
    """Interpret a `G2C_APPLY_SOLUTIONS` value.

    Returns None for "apply everything" and [] for "apply nothing"; otherwise
    the individual selectors to resolve. Separators may be commas or spaces.
    """
    normalized = value.strip().lower()
    if normalized in _OFF_TOKENS:
        return []
    if normalized in _ALL_TOKENS:
        return None
    return [part for part in value.replace(",", " ").split() if part]
