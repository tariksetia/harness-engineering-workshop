"""Types for the provider: the neutral message format and the wire format.

- Concept: the harness speaks a small provider-neutral message format; the
  provider translates it to a Gemini-shaped wire format.
- Design rules:
  1. Neutral types never leak wire details except the opaque tool-call
     `signature`, which must round-trip untouched.
  2. Every shape is a named TypedDict; free-form dicts are allowed only where
     the content is genuinely open (JSON Schema, tool arguments).
"""

from typing import Any, Literal, NotRequired, TypedDict

# Free-form JSON object whose keys are not known to the harness.
type JsonObject = dict[str, Any]

# A JSON Schema document describing a tool's parameters.
type JsonSchema = JsonObject

# Keyword arguments the model chose for a tool, keyed by parameter name.
type ToolArguments = JsonObject


# --- Tools -------------------------------------------------------------------


class FunctionDeclaration(TypedDict):
    """A tool as the model sees it.

    - name: identifier the model uses to call the tool.
    - description: what the tool does; guides the model's choice.
    - parameters: JSON Schema of the tool's keyword arguments.
    """

    name: str
    description: NotRequired[str]
    parameters: NotRequired[JsonSchema]


class ToolSpec(TypedDict):
    """A tool declaration handed to the provider.

    - schema: the function declaration shown to the model.
    """

    schema: FunctionDeclaration


class ToolCall(TypedDict):
    """A tool call requested by the model.

    - name: tool name the model chose.
    - args: keyword arguments for the tool.
    - signature: opaque token the provider must echo back on the next turn.
    """

    name: str
    args: ToolArguments
    signature: str | None


# --- Neutral messages --------------------------------------------------------


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


# One entry of the neutral conversation.
type Message = UserMessage | AssistantMessage | ToolMessage


# --- Completion --------------------------------------------------------------


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


# --- Wire parts --------------------------------------------------------------


class TextPart(TypedDict):
    """A wire part carrying text.

    - text: the text.
    - thought: True when the text is model reasoning, not answer text.
    """

    text: str
    thought: NotRequired[bool]


class FunctionCall(TypedDict):
    """The body of a functionCall part.

    - name: tool to call.
    - args: keyword arguments for the tool.
    """

    name: str
    args: ToolArguments


class FunctionCallPart(TypedDict):
    """A wire part in which the model calls a tool.

    - functionCall: which tool, with which arguments.
    - thoughtSignature: opaque token echoed back with the call on later turns.
    """

    functionCall: FunctionCall
    thoughtSignature: NotRequired[str]


class FunctionResult(TypedDict):
    """The payload of a functionResponse.

    - result: the tool's stringified output.
    """

    result: str


class FunctionResponse(TypedDict):
    """The body of a functionResponse part.

    - name: tool that produced the result.
    - response: the result payload.
    """

    name: str
    response: FunctionResult


class FunctionResponsePart(TypedDict):
    """A wire part returning a tool result to the model.

    - functionResponse: the tool name and its result.
    """

    functionResponse: FunctionResponse


# Any single part of a wire content entry.
type WirePart = TextPart | FunctionCallPart | FunctionResponsePart


class WireContent(TypedDict):
    """One Gemini-shaped content entry.

    - role: "user" or "model".
    - parts: text / functionCall / functionResponse parts.
    """

    role: Literal["user", "model"]
    parts: list[WirePart]


# --- Wire request ------------------------------------------------------------


class SystemInstruction(TypedDict):
    """The system prompt in wire form.

    - parts: text parts making up the prompt.
    """

    parts: list[TextPart]


class GenerationConfig(TypedDict):
    """Sampling settings for one call.

    - temperature: sampling temperature.
    - maxOutputTokens: upper bound on generated tokens.
    """

    temperature: float
    maxOutputTokens: int


class WireTools(TypedDict):
    """The tools block of a wire request.

    - functionDeclarations: every tool offered to the model.
    """

    functionDeclarations: list[FunctionDeclaration]


class WireRequest(TypedDict):
    """Gemini-shaped request body.

    - systemInstruction: the system prompt.
    - contents: conversation as wire contents.
    - generationConfig: sampling settings.
    - tools: tools offered to the model; absent for a toolless call.
    """

    systemInstruction: SystemInstruction
    contents: list[WireContent]
    generationConfig: GenerationConfig
    tools: NotRequired[list[WireTools]]


# --- Wire response -----------------------------------------------------------


class Candidate(TypedDict):
    """One candidate reply.

    - content: the model's content entry.
    """

    content: WireContent


class UsageMetadata(TypedDict):
    """Token counts reported with a wire response.

    - promptTokenCount: tokens in the request.
    - candidatesTokenCount: tokens generated.
    """

    promptTokenCount: int
    candidatesTokenCount: int


class WireResponse(TypedDict):
    """Gemini-shaped response body.

    - candidates: candidate replies; the harness reads the first.
    - usageMetadata: token counts.
    """

    candidates: list[Candidate]
    usageMetadata: UsageMetadata
