"""Day 2 demo: a coding agent with real tools behind a security policy.

- Concept: the core tools plus a Policy turn the loop into an agent that can
  build and run code — and that cannot be talked into wrecking the machine.
- Design rules:
  1. The tools are confined to a fresh scratch directory.
  2. The policy runs in yolo mode, so only the deny list stands between the
     model and the shell — exactly what the demo shows off.
"""

import sys
import tempfile
from typing import cast

from harness import bedrock
from harness.loop import run_loop
from harness.schema.loop import AssistantEvent, EventKind, EventPayload, ToolEndEvent
from harness.schema.provider import Message
from harness.security import Policy
from harness.tools import core_tools

DEFAULT_TASK = "Create fib.py with an iterative fib(n), a __main__ printing fib(30), run it and confirm the output is 832040"
SYSTEM = (
    "You are a coding agent working in {workdir}. Use the tools to act, then "
    "verify your work by running it before you answer. If a call comes back "
    "BLOCKED, do not work around it — explain civilly why it was refused."
)
MAX_PRINTED_CHARS = 500


def print_event(kind: EventKind, payload: EventPayload) -> None:
    """Print one loop event as a transcript line.

    - kind: "assistant", "tool_start", or "tool_end".
    - payload: event data emitted by the loop.
    """
    # The kind names the payload shape; the casts make that pairing explicit.
    if kind == "assistant":
        reply = cast(AssistantEvent, payload)
        for call in reply["tool_calls"]:
            print(f"[assistant] tool call: {call['name']}({call['args']})")
        if reply["text"]:
            print(f"[assistant] {reply['text']}")
    elif kind == "tool_end":
        end = cast(ToolEndEvent, payload)
        print(f"[tool] {end['name']} -> {end['result'][:MAX_PRINTED_CHARS]}")


def main() -> None:
    """Run the task from the command line (or the fib task) in a scratch directory."""
    task = " ".join(sys.argv[1:]) or DEFAULT_TASK
    workdir = tempfile.mkdtemp(prefix="harness-day2-")
    tools = {t.name: t for t in core_tools(workdir)}
    policy = Policy("yolo")
    messages: list[Message] = [{"role": "user", "text": task}]
    print(f"[workdir] {workdir}\n[user] {task}")
    run_loop(bedrock.model_id(), SYSTEM.format(workdir=workdir), messages, tools, print_event, policy.check)


if __name__ == "__main__":
    main()
