"""The agent loop: call the model, run its tools, repeat until it answers.

- Concept: an agent is a loop around a stateless model call; the message list
  is the only state.
- Design rules:
  1. The loop never crashes because a tool did — failures become tool results.
  2. A policy hook (before_tool) can block any call before it runs.
  3. The turn budget is hard: when it runs out the model must wrap up, toolless.
"""

from harness import provider
from harness.schema.loop import BeforeTool, BeforeTurn, OnEvent, Tool
from harness.schema.provider import AssistantMessage, Message, ToolCall, ToolSpec


def run_loop(
    model: str,
    system: str,
    messages: list[Message],
    tools: dict[str, Tool],
    on_event: OnEvent,
    before_tool: BeforeTool,
    max_turns: int = 80,
    before_turn: BeforeTurn | None = None,
) -> str:
    """Run the agent loop and return the model's final text.

    - model: Bedrock model id.
    - system: system prompt.
    - messages: conversation so far; mutated in place as the loop runs.
    - tools: tool name -> Tool (anything exposing .spec and .run).
    - on_event: called as on_event(kind, payload) for "assistant", "tool_start", "tool_end".
    - before_tool: returns None to allow a call or a reason string to block it.
    - max_turns: model calls allowed before the forced wrap-up call.
    - before_turn: optional rewrite of the message list before each model call (day 3: compaction).
    """
    specs = [t.spec for t in tools.values()]
    for _ in range(max_turns):
        reply = _turn(model, system, messages, specs, on_event, before_turn)
        if not reply["tool_calls"]:
            return reply["text"]
        for call in reply["tool_calls"]:
            on_event("tool_start", {"name": call["name"], "args": call["args"]})
            result = _execute(call, tools, before_tool)
            messages.append({"role": "tool", "name": call["name"], "text": result})
            on_event("tool_end", {"name": call["name"], "result": result})
    messages.append({"role": "user", "text": "Turn limit reached; wrap up now."})
    return _turn(model, system, messages, None, on_event, before_turn)["text"]


def _turn(
    model: str,
    system: str,
    messages: list[Message],
    specs: list[ToolSpec] | None,
    on_event: OnEvent,
    before_turn: BeforeTurn | None,
) -> AssistantMessage:
    """Make one model call, record the reply, and emit the "assistant" event.

    - model: Bedrock model id.
    - system: system prompt.
    - messages: conversation; the reply is appended in place.
    - specs: tool specs to offer, or None for a toolless call.
    - on_event: event sink.
    - before_turn: optional message-list rewrite applied before the call.
    """
    if before_turn is not None:
        messages[:] = before_turn(messages)  # In place: callers hold this list.
    completion = provider.complete(model, system, messages, specs)
    reply: AssistantMessage = {"role": "assistant", "text": completion["text"], "tool_calls": completion["tool_calls"]}
    messages.append(reply)
    on_event("assistant", {**reply, "usage": completion["usage"]})
    return reply


def _execute(call: ToolCall, tools: dict[str, Tool], before_tool: BeforeTool) -> str:
    """Run one tool call and return its result as text, never raising.

    - call: the tool call requested by the model.
    - tools: tool name -> Tool.
    - before_tool: policy hook; a returned reason blocks the call.
    """
    reason = before_tool(call)
    if reason is not None:
        return f"BLOCKED: {reason}"
    tool = tools.get(call["name"])
    if tool is None:
        return f"ERROR: unknown tool {call['name']}"
    try:
        return str(tool.run(**call["args"]))
    except Exception as error:  # The loop must survive any tool failure.
        return f"ERROR: {type(error).__name__}: {error}"
