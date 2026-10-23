"""AWS Bedrock integration: Converse API behind a Gemini-shaped wire format.

- Concept: the provider speaks one wire format; this module is the adapter that
  carries it to Anthropic and DeepSeek models on Bedrock and back.
- Design rules:
  1. Auth is a Bedrock API key (short- or long-term) sent as a bearer token —
     no SigV4, no boto3.
  2. Configuration comes from env vars (loaded via dotenv), never hard-coded.
  3. Translation is pure: typed dicts in, typed dicts out; no I/O besides env reads.
"""

import os
from urllib.parse import quote

from dotenv import load_dotenv

from harness.schema.bedrock import (
    ConverseMessage,
    ConverseRequest,
    ConverseResponse,
    ConverseTool,
    RequestBlock,
    ResponseBlock,
)
from harness.schema.provider import (
    FunctionCall,
    FunctionDeclaration,
    FunctionResponse,
    WireContent,
    WirePart,
    WireRequest,
    WireResponse,
)

load_dotenv()

DEFAULT_MODEL = "deepseek.v3.2"
DEFAULT_REGION = "us-east-1"
DSML_MARKER = "<｜DSML｜"
# Bedrock rejects maxTokens above a model's own ceiling instead of clamping,
# and Claude caps output below the provider's 65536 default.
ANTHROPIC_MAX_TOKENS = 64000


def model_id() -> str:
    """Return the configured Bedrock model id.

    - Reads HARNESS_BEDROCK_MODEL; falls back to DEFAULT_MODEL (DeepSeek).
    """
    return os.getenv("HARNESS_BEDROCK_MODEL") or DEFAULT_MODEL


def region() -> str:
    """Return the bedrock-runtime region.

    - Reads HARNESS_BEDROCK_REGION, then AWS_REGION, then DEFAULT_REGION.
    """
    return os.getenv("HARNESS_BEDROCK_REGION") or os.getenv("AWS_REGION") or DEFAULT_REGION


def api_key() -> str:
    """Return the Bedrock API key.

    - Reads HARNESS_BEDROCK_API_KEY.
    - Raises RuntimeError when it is missing, so misconfiguration fails fast.
    """
    key = os.getenv("HARNESS_BEDROCK_API_KEY")
    if not key:
        raise RuntimeError("HARNESS_BEDROCK_API_KEY is not set (see .env.example)")
    return key


def converse_url(model: str) -> str:
    """Build the Converse endpoint URL for a model.

    - model: Bedrock model or inference-profile id; URL-quoted because ids
      contain ':' and may be ARNs.
    """
    return f"https://bedrock-runtime.{region()}.amazonaws.com/model/{quote(model, safe='')}/converse"


def auth_headers() -> dict[str, str]:
    """Return the HTTP headers that authenticate a Bedrock call.

    - Uses the API key as a bearer token.
    """
    return {"Authorization": f"Bearer {api_key()}"}


def to_converse(model: str, body: WireRequest) -> ConverseRequest:
    """Translate a Gemini-shaped request into a Converse request.

    - model: Bedrock model id; decides the maxTokens ceiling.
    - body: wire request built by the provider (systemInstruction, contents,
      generationConfig, optional tools).
    """
    declarations = [d for t in body.get("tools", []) for d in t["functionDeclarations"]]
    config = body["generationConfig"]
    request: ConverseRequest = {
        "system": [{"text": p["text"]} for p in body["systemInstruction"]["parts"] if p.get("text")],
        "messages": _to_messages(body["contents"], flatten_tools=not declarations),
        "inferenceConfig": {"temperature": config["temperature"], "maxTokens": _max_tokens(model, config["maxOutputTokens"])},
    }
    if declarations:
        request["toolConfig"] = {"tools": [_tool_spec(d) for d in declarations]}
    return request


def from_converse(response: ConverseResponse) -> WireResponse:
    """Translate a Converse response into a Gemini-shaped response.

    - response: parsed JSON returned by the Converse endpoint.
    """
    blocks = response.get("output", {}).get("message", {}).get("content", [])
    parts = [part for block in blocks if (part := _to_part(block)) is not None]
    usage = response.get("usage", {})
    return {
        "candidates": [{"content": {"role": "model", "parts": parts}}],
        "usageMetadata": {
            "promptTokenCount": usage.get("inputTokens", 0),
            "candidatesTokenCount": usage.get("outputTokens", 0),
        },
    }


def _max_tokens(model: str, requested: int) -> int:
    """Clamp the requested output budget to the model's ceiling.

    - model: Bedrock model id.
    - requested: maxOutputTokens from the wire request.
    """
    return min(requested, ANTHROPIC_MAX_TOKENS) if "anthropic." in model else requested


def _tool_spec(declaration: FunctionDeclaration) -> ConverseTool:
    """Wrap one function declaration as a Converse toolSpec.

    - declaration: the tool as the model sees it.
    """
    return {
        "toolSpec": {
            "name": declaration["name"],
            "description": declaration.get("description", ""),
            "inputSchema": {"json": declaration.get("parameters", {"type": "object", "properties": {}})},
        }
    }


def _to_messages(contents: list[WireContent], flatten_tools: bool) -> list[ConverseMessage]:
    """Translate wire contents into alternating Converse messages.

    - contents: Gemini-shaped conversation.
    - flatten_tools: render tool calls/results as plain text; needed when the
      request carries no toolConfig.
    """
    messages: list[ConverseMessage] = []
    # Converse pairs results to calls by toolUseId; the wire format pairs by
    # order, so ids of the latest model turn are queued and consumed in order.
    pending_ids: list[str] = []
    for content in contents:
        role = "assistant" if content["role"] == "model" else "user"
        blocks: list[RequestBlock] = []
        for part in content["parts"]:
            if "functionCall" in part:
                tool_id = part.get("thoughtSignature") or f"call_{len(messages)}_{len(blocks)}"
                pending_ids.append(tool_id)
                blocks.append(_call_block(part["functionCall"], tool_id, flatten_tools))
            elif "functionResponse" in part:
                tool_id = pending_ids.pop(0) if pending_ids else "call_unknown"
                blocks.append(_result_block(part["functionResponse"], tool_id, flatten_tools))
            elif part.get("text"):
                blocks.append({"text": part["text"]})
        if not blocks:
            continue  # Converse rejects empty content lists.
        # Converse requires strict user/assistant alternation, so consecutive
        # same-role entries (e.g. several tool results) merge into one message.
        if messages and messages[-1]["role"] == role:
            messages[-1]["content"].extend(blocks)
        else:
            messages.append({"role": role, "content": blocks})
    return messages


def _call_block(call: FunctionCall, tool_id: str, flatten: bool) -> RequestBlock:
    """Translate a wire functionCall into a Converse block.

    - call: tool name and arguments from the wire part.
    - tool_id: toolUseId that the matching result will reference.
    - flatten: emit text instead of a toolUse block.
    """
    if flatten:
        return {"text": f"[called tool {call['name']} with {call.get('args', {})}]"}
    return {"toolUse": {"toolUseId": tool_id, "name": call["name"], "input": call.get("args", {})}}


def _result_block(response: FunctionResponse, tool_id: str, flatten: bool) -> RequestBlock:
    """Translate a wire functionResponse into a Converse block.

    - response: tool name and result from the wire part.
    - tool_id: toolUseId of the call this result answers.
    - flatten: emit text instead of a toolResult block.
    """
    result = str(response.get("response", {}).get("result", ""))
    if flatten:
        return {"text": f"[tool {response['name']} returned: {result}]"}
    return {"toolResult": {"toolUseId": tool_id, "content": [{"text": result}]}}


def _to_part(block: ResponseBlock) -> WirePart | None:
    """Translate one Converse output block into a wire part.

    - block: text, toolUse, or reasoningContent block.
    - Returns None for block types the harness ignores.
    """
    if "text" in block:
        # DeepSeek V3.2 can leak its raw DSML tool-call markup into the text
        # alongside the parsed toolUse block; everything from it on is noise.
        return {"text": block["text"].split(DSML_MARKER)[0].rstrip()}
    if "toolUse" in block:
        use = block["toolUse"]
        # The toolUseId rides in thoughtSignature so it round-trips through
        # the neutral format's opaque `signature` field.
        return {"functionCall": {"name": use["name"], "args": use.get("input") or {}}, "thoughtSignature": use["toolUseId"]}
    if "reasoningContent" in block:
        text = block["reasoningContent"].get("reasoningText", {}).get("text", "")
        return {"text": text, "thought": True}
    return None
