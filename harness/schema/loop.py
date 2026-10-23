"""Types for the agent loop: tools and the loop's hook signatures.

- Concept: the loop depends on structural shapes (a Tool is anything with
  .spec and .run), not on concrete classes.
- Design rule: hooks are plain callables so callers can pass lambdas.
"""

from collections.abc import Callable
from typing import Any, Protocol

from harness.schema.provider import Message, ToolCall, ToolSpec


class Tool(Protocol):
    """Anything the loop can execute.

    - spec: the {"schema": ...} declaration sent to the model.
    - run: callable invoked with the model's arguments as keywords.
    """

    spec: ToolSpec
    run: Callable[..., Any]


type EventPayload = dict[str, Any]
type OnEvent = Callable[[str, EventPayload], None]
type BeforeTool = Callable[[ToolCall], str | None]
type BeforeTurn = Callable[[list[Message]], list[Message]]
