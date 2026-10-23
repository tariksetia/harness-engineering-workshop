"""Day 1 demo: a hand-written tool driven by the agent loop.

- Concept: a tool is just an object with a JSON-schema .spec and a .run
  callable; the loop does the rest.
- Design rules:
  1. The tool validates its own input — the model sends `count` as a string.
  2. Hooks stay trivial here: print every event, allow every call.
"""

import random

from harness import bedrock
from harness.loop import run_loop
from harness.schema.loop import EventPayload
from harness.schema.provider import Message, ToolCall, ToolSpec

TASK = "Roll 3 dice and tell me whether the total beats 10"
SYSTEM = "You are a helpful assistant. Use tools when they help."


class RollDice:
    """A tool that rolls six-sided dice.

    - spec: the declaration the model sees.
    - run: rolls `count` dice and returns the list of results.
    """

    spec: ToolSpec = {
        "schema": {
            "name": "roll_dice",
            "description": "Roll count six-sided dice",
            "parameters": {
                "type": "object",
                "properties": {"count": {"type": "string", "description": "How many dice"}},
                "required": ["count"],
            },
        }
    }

    def run(self, count: str) -> list[int]:
        """Roll the dice.

        - count: how many dice to roll, as a string (the schema says so).
        """
        return [random.randint(1, 6) for _ in range(int(count))]


def print_event(kind: str, payload: EventPayload) -> None:
    """Print one loop event as a transcript line.

    - kind: "assistant", "tool_start", or "tool_end".
    - payload: event data emitted by the loop.
    """
    if kind == "assistant":
        for call in payload["tool_calls"]:
            print(f"[assistant] tool call: {call['name']}({call['args']})")
        if payload["text"]:
            print(f"[assistant] {payload['text']}")
    elif kind == "tool_start":
        print(f"[tool_start] {payload['name']}")
    elif kind == "tool_end":
        print(f"[tool] {payload['name']} -> {payload['result']}")


def allow_all(call: ToolCall) -> str | None:
    """Allow every tool call.

    - call: the tool call about to run (ignored).
    """
    return None


def main() -> None:
    """Run the dice task and print the transcript."""
    messages: list[Message] = [{"role": "user", "text": TASK}]
    print(f"[user] {TASK}")
    run_loop(bedrock.model_id(), SYSTEM, messages, {"roll_dice": RollDice()}, print_event, allow_all)


if __name__ == "__main__":
    main()
