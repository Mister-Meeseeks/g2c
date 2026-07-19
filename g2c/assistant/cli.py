"""CLI — a tiny interactive loop for the capstone assistant.

`run_cli(assistant)` is a small REPL: the user types a question, the
assistant chats, the response is printed. A few slash commands cover
the basics:

  /help           Show command list
  /clear          Reset the conversation
  /history        Print the conversation so far
  /tools          List the registered tools
  /config         Show the assistant's config
  /save <path>    Save the conversation transcript to a file
  /exit, /quit    Exit the loop (also Ctrl-D, blank input)

Fully implemented. The CLI is plumbing; the lesson is in
`assistant.py`.

Why a CLI rather than a web UI? Two reasons:

  * **Zero dependencies.** A text REPL is `print(input())` plus a
    little glue. A web UI requires a server, a frontend stack, and a
    contract between them — none of which is the lesson of this
    course.

  * **Easier to reason about.** When the assistant misbehaves, the
    CLI's transcript is a flat text record. Debugging a misbehaving
    web UI means cross-referencing browser devtools, server logs,
    and the model's prompt — three layers thick. The CLI keeps the
    debug surface flat.

If you want a web UI later, building one on top of this is mostly
glue: an HTTP endpoint that takes `{message, conversation_id}`, calls
`assistant.chat`, and returns `{answer, transcript}`. The capstone
deliverable allows either form.
"""
from __future__ import annotations

import json
import shlex
import sys
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

if TYPE_CHECKING:
    from .assistant import Assistant

CLI_HELP = """\
Commands:
  /help              Show this help.
  /clear             Reset the conversation history.
  /history           Print the conversation so far.
  /tools             List registered tools.
  /config            Show the assistant config.
  /save <path>       Save the transcript as JSON to <path>.
  /exit, /quit       Exit the loop. (Ctrl-D and blank input also exit.)

Anything else is a message to the assistant.
"""


def _print_turn(turn, file: TextIO) -> None:
    """Render one turn for the user. Compact: answer + a metadata footer."""
    answer = turn.final_answer or f"(stuck — {turn.agent_run.stopped_reason})"
    print(answer, file=file)
    # One-line footer on a separate row for visual scannability.
    n_steps = turn.metadata.get("n_agent_steps", 0)
    n_tool_calls = turn.metadata.get("n_tool_calls", 0)
    rag_marker = "rag" if turn.metadata.get("rag_fired") else "no-rag"
    plan_marker = "plan" if turn.metadata.get("had_plan") else "no-plan"
    print(
        f"  [{n_steps} steps, {n_tool_calls} tool calls, "
        f"{rag_marker}, {plan_marker}]",
        file=file,
    )


def _cmd_help(_: Assistant, _args: str, *, out: TextIO) -> bool:
    print(CLI_HELP, file=out)
    return True


def _cmd_clear(assistant: Assistant, _args: str, *, out: TextIO) -> bool:
    assistant.reset()
    print("(conversation cleared)", file=out)
    return True


def _cmd_history(assistant: Assistant, _args: str, *, out: TextIO) -> bool:
    msgs = assistant.conversation.messages
    if not msgs:
        print("(no messages yet)", file=out)
        return True
    for m in msgs:
        print(f"{m.role}: {m.content}", file=out)
    return True


def _cmd_tools(assistant: Assistant, _args: str, *, out: TextIO) -> bool:
    tools = assistant.registry.tools
    if not tools:
        print("(no tools registered)", file=out)
        return True
    for t in tools:
        print(f"  - {t.name}: {t.description}", file=out)
    return True


def _cmd_config(assistant: Assistant, _args: str, *, out: TextIO) -> bool:
    cfg = asdict(assistant.config)
    print(json.dumps(cfg, indent=2, default=str), file=out)
    return True


def _cmd_save(assistant: Assistant, args: str, *, out: TextIO) -> bool:
    parts = shlex.split(args.strip())
    if len(parts) != 1:
        print("usage: /save <path>", file=out)
        return True
    path = Path(parts[0]).expanduser()
    transcript = {
        "name": assistant.config.name,
        "config": asdict(assistant.config),
        "messages": [
            {"role": m.role, "content": m.content}
            for m in assistant.conversation.messages
        ],
        "turns": [
            {
                "user_message": t.user_message,
                "final_answer": t.final_answer,
                "stopped_reason": t.agent_run.stopped_reason,
                "metadata": t.metadata,
            }
            for t in assistant.turns
        ],
    }
    path.write_text(json.dumps(transcript, indent=2, default=str))
    print(f"(saved to {path})", file=out)
    return True


def _cmd_exit(_: Assistant, _args: str, *, out: TextIO) -> bool:
    print("(exiting)", file=out)
    return False


_COMMANDS: dict[str, Callable[..., bool]] = {
    "/help": _cmd_help,
    "/clear": _cmd_clear,
    "/history": _cmd_history,
    "/tools": _cmd_tools,
    "/config": _cmd_config,
    "/save": _cmd_save,
    "/exit": _cmd_exit,
    "/quit": _cmd_exit,
}


def _dispatch_command(
    assistant: Assistant, line: str, *, out: TextIO
) -> bool:
    """Handle a slash-command line. Returns True to continue the loop,
    False to exit. Unknown commands print a hint and continue.
    """
    parts = line.split(maxsplit=1)
    cmd = parts[0]
    args = parts[1] if len(parts) > 1 else ""
    handler = _COMMANDS.get(cmd)
    if handler is None:
        print(
            f"(unknown command {cmd!r}; type /help for the list)",
            file=out,
        )
        return True
    return handler(assistant, args, out=out)


def run_cli(
    assistant: Assistant,
    *,
    prompt: str = "? ",
    inp: TextIO | None = None,
    out: TextIO | None = None,
) -> None:
    """Run the interactive CLI loop.

    Args:
        assistant: the `Assistant` to drive.
        prompt: the input-line prompt (default `"? "`).
        inp: input stream. Defaults to `sys.stdin`. Tests pass an
            in-memory `StringIO`.
        out: output stream. Defaults to `sys.stdout`. Tests pass an
            in-memory `StringIO`.

    The loop exits on `/exit`, `/quit`, EOF (Ctrl-D), or a blank
    line. KeyboardInterrupt (Ctrl-C) also exits cleanly.

    Implementation notes:

      * Reads with `inp.readline()` rather than `input()` so the
        function works with arbitrary streams (for tests).

      * Writes the prompt to `out` rather than `sys.stderr` so that
        a test capturing `out` sees a coherent transcript.
    """
    inp = inp if inp is not None else sys.stdin
    out = out if out is not None else sys.stdout

    print(
        f"({assistant.config.name} ready — type /help for commands, "
        f"blank line to exit)",
        file=out,
    )

    while True:
        out.write(prompt)
        out.flush()
        try:
            line = inp.readline()
        except KeyboardInterrupt:
            print("(interrupted)", file=out)
            return
        if not line:  # EOF
            print("(eof)", file=out)
            return
        line = line.rstrip("\n").strip()
        if not line:  # blank
            return
        if line.startswith("/"):
            cont = _dispatch_command(assistant, line, out=out)
            if not cont:
                return
            continue
        try:
            turn = assistant.chat(line)
        except Exception as e:
            print(f"(error: {type(e).__name__}: {e})", file=out)
            continue
        _print_turn(turn, out)
