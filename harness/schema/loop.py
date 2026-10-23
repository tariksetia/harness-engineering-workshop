"""Types for the agent loop: tools, events, and the loop's hook signatures.

- Concept: the loop depends on structural shapes (a Tool is anything with
  .spec and .run), not on concrete classes.
- Design rules:
  1. Hooks are plain callables so callers can pass lambdas.
  2. Each event kind has its own named payload shape.
"""

from collections.abc import Callable
from typing import Any, Literal, Protocol, TypedDict

from harness.schema.provider import Message, ToolArguments, ToolCall, ToolSpec, Usage


class Tool(Protocol):
    """Anything the loop can execute.

    - spec: the declaration sent to the model.
    - run: callable invoked with the model's arguments as keywords; its return
      value is stringified into the tool result.
    """

    spec: ToolSpec
    run: Callable[..., Any]


# --- Events ------------------------------------------------------------------

# Which moment of the loop an event marks.
type EventKind = Literal["assistant", "tool_start", "tool_end"]


class AssistantEvent(TypedDict):
    """Payload of an "assistant" event, fired after each model reply.

    - role: always "assistant".
    - text: the model's text, possibly empty.
    - tool_calls: tool calls the model requested.
    - usage: token counts of the call.
    """

    role: Literal["assistant"]
    text: str
    tool_calls: list[ToolCall]
    usage: Usage


class ToolStartEvent(TypedDict):
    """Payload of a "tool_start" event, fired before a tool call is handled.

    - name: tool about to run.
    - args: arguments it will receive.
    """

    name: str
    args: ToolArguments


class ToolEndEvent(TypedDict):
    """Payload of a "tool_end" event, fired after a tool call is handled.

    - name: tool that ran (or was blocked / unknown).
    - result: the tool result text sent back to the model.
    """

    name: str
    result: str


# Payload of any loop event; pair it with its EventKind.
type EventPayload = AssistantEvent | ToolStartEvent | ToolEndEvent


# --- Hooks -------------------------------------------------------------------

# Receives every loop event as (kind, payload).
type OnEvent = Callable[[EventKind, EventPayload], None]

# Reason a tool call was refused; None means the call may run.
type BlockReason = str | None

# Policy check run before each tool call.
type BeforeTool = Callable[[ToolCall], BlockReason]

# Rewrites the message list before each model call (day 3: compaction).
type BeforeTurn = Callable[[list[Message]], list[Message]]
