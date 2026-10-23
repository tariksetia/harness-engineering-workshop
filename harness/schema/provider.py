"""Types for the provider: the neutral message format and the wire format.

- Concept: the harness speaks a small provider-neutral message format; the
  provider translates it to a Gemini-shaped wire format.
- Design rule: neutral types never leak wire details except the opaque
  tool-call `signature`, which must round-trip untouched.
"""

from typing import Any, Literal, NotRequired, TypedDict

type JsonDict = dict[str, Any]
type WirePart = dict[str, Any]


class ToolSpec(TypedDict):
    """A tool declaration handed to the provider.

    - schema: function declaration ({"name", "description", "parameters"}).
    """

    schema: JsonDict


class ToolCall(TypedDict):
    """A tool call requested by the model.

    - name: tool name the model chose.
    - args: keyword arguments for the tool.
    - signature: opaque token the provider must echo back on the next turn.
    """

    name: str
    args: JsonDict
    signature: str | None


class Usage(TypedDict):
    """Token usage of one completion.

    - input: prompt tokens.
    - output: generated tokens.
    """

    input: int
    output: int


class Completion(TypedDict):
    """Result of one provider call.

    - text: concatenated non-thought text.
    - tool_calls: tool calls in the order the model emitted them.
    - usage: token counts.
    """

    text: str
    tool_calls: list[ToolCall]
    usage: Usage


class UserMessage(TypedDict):
    """A user turn.

    - role: always "user".
    - text: what the user said.
    """

    role: Literal["user"]
    text: str


class AssistantMessage(TypedDict):
    """A model turn.

    - role: always "assistant".
    - text: the model's text, possibly empty.
    - tool_calls: tool calls the model requested, possibly empty.
    """

    role: Literal["assistant"]
    text: str
    tool_calls: list[ToolCall]


class ToolMessage(TypedDict):
    """The result of one tool execution.

    - role: always "tool".
    - name: tool that produced the result.
    - text: stringified result.
    """

    role: Literal["tool"]
    name: str
    text: str


type Message = UserMessage | AssistantMessage | ToolMessage


class WireContent(TypedDict):
    """One Gemini-shaped content entry.

    - role: "user" or "model".
    - parts: text / functionCall / functionResponse parts.
    """

    role: Literal["user", "model"]
    parts: list[WirePart]


class WireRequest(TypedDict):
    """Gemini-shaped request body.

    - systemInstruction: {"parts": [{"text": system}]}.
    - contents: conversation as wire contents.
    - generationConfig: temperature and maxOutputTokens.
    - tools: optional [{"functionDeclarations": [...]}].
    """

    systemInstruction: JsonDict
    contents: list[WireContent]
    generationConfig: JsonDict
    tools: NotRequired[list[JsonDict]]
