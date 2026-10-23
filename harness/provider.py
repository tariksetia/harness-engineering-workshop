"""The provider: one model call, neutral messages in, a Completion out.

- Concept: the agent loop never sees a wire format; the provider owns
  translation (neutral -> Gemini-shaped wire -> Bedrock) and transport.
- Design rules:
  1. complete() is stateless: everything the model needs is in its arguments.
  2. Tool-call signatures are echoed back verbatim on the next turn.
  3. Transient HTTP failures are retried with exponential backoff; others
     fail loudly with the status and a truncated body.
"""

import json
import time
import urllib.error
import urllib.request

from harness import bedrock
from harness.schema.bedrock import ConverseRequest, ConverseResponse
from harness.schema.provider import (
    Completion,
    FunctionCallPart,
    FunctionResponse,
    Message,
    ToolCall,
    ToolSpec,
    WireContent,
    WirePart,
    WireRequest,
    WireResponse,
)

RETRYABLE_STATUS = {429, 500, 502, 503}


def complete(model: str, system: str, messages: list[Message], tools: list[ToolSpec] | None = None) -> Completion:
    """Run one model call.

    - model: Bedrock model id (Anthropic or DeepSeek).
    - system: system prompt.
    - messages: conversation in the neutral format.
    - tools: optional tool specs, each {"schema": ...}; omitted from the request when empty.
    """
    body: WireRequest = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": _to_wire(messages),
        "generationConfig": {"temperature": 0.4, "maxOutputTokens": 65536},
    }
    if tools:
        body["tools"] = [{"functionDeclarations": [t["schema"] for t in tools]}]
    raw = _post(bedrock.converse_url(model), bedrock.to_converse(model, body))
    return _parse(bedrock.from_converse(raw))


def _to_wire(messages: list[Message]) -> list[WireContent]:
    """Map neutral messages to wire contents.

    - messages: conversation in the neutral format.
    """
    contents: list[WireContent] = []
    for message in messages:
        if message["role"] == "user":
            contents.append({"role": "user", "parts": [{"text": message["text"]}]})
        elif message["role"] == "assistant":
            parts: list[WirePart] = [{"text": message["text"]}] if message["text"] else []
            for call in message["tool_calls"]:
                part: FunctionCallPart = {"functionCall": {"name": call["name"], "args": call["args"]}}
                # Gemini 3 rejects a replayed functionCall without its
                # signature; Bedrock carries the toolUseId the same way.
                if call["signature"] is not None:
                    part["thoughtSignature"] = call["signature"]
                parts.append(part)
            contents.append({"role": "model", "parts": parts})
        else:
            response: FunctionResponse = {"name": message["name"], "response": {"result": message["text"]}}
            contents.append({"role": "user", "parts": [{"functionResponse": response}]})
    return contents


def _parse(response: WireResponse) -> Completion:
    """Parse a wire response into a Completion.

    - response: Gemini-shaped response with candidates and usageMetadata.
    """
    candidates = response.get("candidates") or [{}]
    parts: list[WirePart] = candidates[0].get("content", {}).get("parts", [])
    text = "".join(p["text"] for p in parts if "text" in p and not p.get("thought"))
    tool_calls: list[ToolCall] = [
        {"name": p["functionCall"]["name"], "args": p["functionCall"].get("args", {}), "signature": p.get("thoughtSignature")}
        for p in parts
        if "functionCall" in p
    ]
    usage = response.get("usageMetadata", {})
    return {
        "text": text,
        "tool_calls": tool_calls,
        "usage": {"input": usage.get("promptTokenCount", 0), "output": usage.get("candidatesTokenCount", 0)},
    }


def _post(url: str, body: ConverseRequest, retries: int = 5) -> ConverseResponse:
    """POST a Converse request and return the parsed Converse response.

    - url: Converse endpoint to call.
    - body: request body, serialized as JSON.
    - retries: total attempts before giving up on transient failures.
    """
    data = json.dumps(body).encode()
    headers = {"Content-Type": "application/json", **bedrock.auth_headers()}
    for attempt in range(retries):
        request = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as error:  # Must precede URLError: it is a subclass.
            if error.code not in RETRYABLE_STATUS or attempt == retries - 1:
                detail = error.read().decode(errors="replace")[:400]
                raise RuntimeError(f"HTTP {error.code}: {detail}") from error
        except (urllib.error.URLError, TimeoutError) as error:
            if attempt == retries - 1:
                raise RuntimeError(f"request failed after {retries} attempts: {error}") from error
        time.sleep(2**attempt * 2)
    raise RuntimeError("retries must be at least 1")
