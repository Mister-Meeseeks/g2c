"""ToolRegistry — collect tools and look them up by name.

A registry is a name → Tool dict with a few ergonomic methods. It
exists for two reasons:

  * **One source of truth.** The prompt renderer iterates over the
    registry to build the "Tools available:" block. The dispatcher
    looks up by name when a `ToolCall` arrives. Both consult the same
    object, so the model can never call a tool the prompt didn't
    advertise (and vice versa).

  * **Composition.** Different runs need different tool sets. A
    research agent might have `web_search` and `read_file`; a
    code-runner has `run_python` and `read_file`; a calculator demo
    has only `calculator`. Constructing the registry per-task is the
    seam.

The class is fully implemented — it's plumbing, not the lesson.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator

from .base import Tool


class ToolRegistry:
    """Name → Tool lookup table.

    Args:
        tools: optional iterable of tools to register up front. A
            convenience for the common case of "build a registry from
            a list literal." Equivalent to constructing empty and
            calling `register` on each.

    Raises:
        ValueError: if two tools with the same name are registered.

    Use `register(tool)` to add, `get(name)` to look up, the `tools`
    property to iterate (returns a list snapshot, safe to mutate
    without affecting the registry). `name in registry` and
    `len(registry)` work as expected.
    """

    def __init__(self, tools: Iterable[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        if tools is not None:
            for t in tools:
                self.register(t)

    def register(self, tool: Tool) -> None:
        """Add `tool` to the registry. Raises if `tool.name` is already
        in use — silent overwrite is a class of bug we'd rather catch.
        """
        if not isinstance(tool, Tool):
            raise TypeError(
                f"register expected a Tool, got {type(tool).__name__}"
            )
        if tool.name in self._tools:
            raise ValueError(
                f"tool {tool.name!r} is already registered; "
                f"de-duplicate before adding"
            )
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        """Look up a tool by name. Raises `KeyError` if not registered.

        The dispatcher catches the `KeyError` and surfaces it as an
        is_error `ToolResult` — so a model hallucinating a tool name
        produces a tool error feedback message rather than a crash.
        """
        if name not in self._tools:
            raise KeyError(
                f"no tool named {name!r}; "
                f"registered: {sorted(self._tools.keys())}"
            )
        return self._tools[name]

    @property
    def tools(self) -> list[Tool]:
        """Snapshot of registered tools. List, not a view — safe to
        iterate while modifying the registry. Order is registration
        order (Python 3.7+ dict insertion order).
        """
        return list(self._tools.values())

    def names(self) -> list[str]:
        """Just the registered tool names, in registration order."""
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._tools

    def __iter__(self) -> Iterator[Tool]:
        return iter(self._tools.values())

    def __repr__(self) -> str:
        return f"ToolRegistry({sorted(self._tools.keys())!r})"
