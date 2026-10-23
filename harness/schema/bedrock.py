"""Types for the Bedrock Converse API.

- Concept: Bedrock's Converse API is the real transport behind the
  Gemini-shaped wire format the provider speaks.
- Design rules:
  1. Only the fields the harness uses are typed.
  2. Request and response block unions are separate: the model never sends a
     toolResult, and the harness never sends reasoning.
"""

from typing import Literal, NotRequired, TypedDict

from harness.schema.provider import JsonSchema, ToolArguments

# --- Content blocks ----------------------------------------------------------


class TextBlock(TypedDict):
    """A content block carrying text.

    - text: the text; Converse rejects empty strings.
    """

    text: str


class ToolUse(TypedDict):
    """The body of a toolUse block.

    - toolUseId: id the matching toolResult must reference.
    - name: tool to call.
    - input: keyword arguments for the tool.
    """

    toolUseId: str
    name: str
    input: ToolArguments


class ToolUseBlock(TypedDict):
    """A content block in which the assistant calls a tool.

    - toolUse: the call.
    """

    toolUse: ToolUse


class ToolResult(TypedDict):
    """The body of a toolResult block.

    - toolUseId: id of the toolUse this result answers.
    - content: the result as text blocks.
    """

    toolUseId: str
    content: list[TextBlock]


class ToolResultBlock(TypedDict):
    """A content block returning a tool result to the model.

    - toolResult: the result.
    """

    toolResult: ToolResult


class ReasoningText(TypedDict):
    """The model's visible reasoning.

    - text: the reasoning text.
    """

    text: str


class ReasoningContent(TypedDict):
    """The body of a reasoningContent block.

    - reasoningText: the reasoning; absent when the model redacts it.
    """

    reasoningText: NotRequired[ReasoningText]


class ReasoningBlock(TypedDict):
    """A content block carrying model reasoning.

    - reasoningContent: the reasoning.
    """

    reasoningContent: ReasoningContent


# A block the harness sends to Converse.
type RequestBlock = TextBlock | ToolUseBlock | ToolResultBlock

# A block Converse may return in the assistant's reply.
type ResponseBlock = TextBlock | ToolUseBlock | ReasoningBlock


# --- Request -----------------------------------------------------------------


class ConverseMessage(TypedDict):
    """One Converse message sent by the harness.

    - role: "user" or "assistant"; Converse requires strict alternation.
    - content: text / toolUse / toolResult blocks.
    """

    role: Literal["user", "assistant"]
    content: list[RequestBlock]


class InferenceConfig(TypedDict):
    """Sampling settings for one Converse call.

    - temperature: sampling temperature.
    - maxTokens: upper bound on generated tokens; must not exceed the model's ceiling.
    """

    temperature: float
    maxTokens: int


class InputSchema(TypedDict):
    """Wrapper Converse puts around a tool's JSON Schema.

    - json: JSON Schema of the tool's keyword arguments.
    """

    json: JsonSchema


class ToolSpecification(TypedDict):
    """A tool as Converse declares it.

    - name: identifier the model uses to call the tool.
    - description: what the tool does.
    - inputSchema: the tool's parameter schema.
    """

    name: str
    description: str
    inputSchema: InputSchema


class ConverseTool(TypedDict):
    """One entry of a toolConfig.

    - toolSpec: the tool declaration.
    """

    toolSpec: ToolSpecification


class ToolConfig(TypedDict):
    """The tools offered on a Converse call.

    - tools: every tool the model may call.
    """

    tools: list[ConverseTool]


class ConverseRequest(TypedDict):
    """Converse request body.

    - system: system prompt blocks.
    - messages: alternating conversation.
    - inferenceConfig: sampling settings.
    - toolConfig: tools offered; absent for a toolless call.
    """

    system: list[TextBlock]
    messages: list[ConverseMessage]
    inferenceConfig: InferenceConfig
    toolConfig: NotRequired[ToolConfig]


# --- Response ----------------------------------------------------------------


class ReplyMessage(TypedDict):
    """The assistant message Converse returns.

    - role: always "assistant".
    - content: text / toolUse / reasoning blocks.
    """

    role: Literal["assistant"]
    content: list[ResponseBlock]


class ConverseOutput(TypedDict):
    """The output section of a Converse response.

    - message: the assistant's reply.
    """

    message: ReplyMessage


class ConverseUsage(TypedDict):
    """Token counts reported by Converse.

    - inputTokens: tokens in the request.
    - outputTokens: tokens generated.
    """

    inputTokens: int
    outputTokens: int


class ConverseResponse(TypedDict):
    """Converse response body.

    - output: the assistant's reply.
    - usage: token counts.
    """

    output: ConverseOutput
    usage: ConverseUsage
