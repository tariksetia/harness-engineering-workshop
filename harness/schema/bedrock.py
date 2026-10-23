"""Types for the Bedrock Converse API.

- Concept: Bedrock's Converse API is the real transport behind the
  Gemini-shaped wire format the provider speaks.
- Design rule: only the fields the harness uses are typed.
"""

from typing import Any, Literal, NotRequired, TypedDict

type ConverseBlock = dict[str, Any]


class ConverseMessage(TypedDict):
    """One Converse message.

    - role: "user" or "assistant"; Converse requires strict alternation.
    - content: text / toolUse / toolResult blocks.
    """

    role: Literal["user", "assistant"]
    content: list[ConverseBlock]


class ConverseRequest(TypedDict):
    """Converse request body.

    - system: system prompt blocks.
    - messages: alternating conversation.
    - inferenceConfig: temperature and maxTokens.
    - toolConfig: optional tool declarations.
    """

    system: list[ConverseBlock]
    messages: list[ConverseMessage]
    inferenceConfig: dict[str, Any]
    toolConfig: NotRequired[dict[str, Any]]
