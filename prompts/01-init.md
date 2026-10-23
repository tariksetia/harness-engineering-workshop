
Bedrock Specifications:
- The provider must use and support Anthropic and deepseek models via AWS Bedrock
- Use API keys (wether short term or long term) from bedrock for calling bedrock models.
- Harness must be passed following ENV Vars:
    - HARNESS_BEDROCK_MODEL (default value is deep seek)
    - HARNESS_BEDROCK_API_KEY
- create bedrock.py to handle aws bedrock integration as specified above

Write harness/provider.py to this exact contract.
- use bedrock specification above for integration
- complete(model, system, messages, tools) returns {"text": str, "tool_calls":
  [{"name", "args", "signature"}], "usage": {"input": int, "output": int}}. It
  posts systemInstruction {"parts": [{"text": system}]}, contents from
  _to_wire(messages), generationConfig {"temperature": 0.4, "maxOutputTokens":
  65536}, and, when tools are given (a list of spec dicts, each {"schema": ...}),
  tools = [{"functionDeclarations": [t["schema"] for t in tools]}].
- The neutral message format: {"role": "user", "text"}; {"role": "assistant",
  "text", "tool_calls"}; {"role": "tool", "name", "text"}. _to_wire maps user to a
  user text part; assistant to role "model" with a text part when text is
  non-empty plus one functionCall part per tool call, echoing the stored
  signature as thoughtSignature on that part (Gemini 3 requires this round-trip);
  tool to role "user" with a functionResponse part {"name": name, "response":
  {"result": text}}.
- Response parsing: concatenate text parts, skipping parts flagged "thought";
  collect each functionCall part as {"name", "args", "signature":
  part.get("thoughtSignature")}; read usageMetadata promptTokenCount and
  candidatesTokenCount into usage.
- _post(url, body, retries=5): urllib, Content-Type application/json, timeout
  600. On HTTP 429/500/502/503: sleep 2**attempt * 2 seconds and retry. On other
  HTTP errors: raise RuntimeError with the status and the first 400 characters of
  the error body. Retry URLError/TimeoutError the same way.

Write harness/loop.py to this exact contract.
- run_loop(model, system, messages, tools, on_event, before_tool, max_turns=80,
  before_turn=None) -> str. tools is a dict name -> Tool, where a Tool exposes
  .spec (a {"schema": ...} dict) and .run (a callable taking keyword
  arguments); the loop hands the provider [t.spec for t in tools.values()]. on_event(kind, payload) fires with "assistant" after
  each model reply, and "tool_start" / "tool_end" around each execution.
  before_tool(call) returns None to allow or a reason string to block; a blocked
  call gets the tool result "BLOCKED: <reason>". before_turn, when provided, is
  called with the message list before each model call and its return value
  replaces the list in place — leave it unused today; day 3 plugs compaction in
  here.
- Each turn: provider.complete, append {"role": "assistant", "text",
  "tool_calls"}, emit the event. No tool calls: return the text. Otherwise run
  each call in order and append {"role": "tool", "name", "text": str(result)}.
- Unknown tool: result "ERROR: unknown tool <name>". Tool exception: result
  "ERROR: <ExceptionType>: <message>". The loop never crashes because a tool did.
- When max_turns runs out: append a user message "Turn limit reached; wrap up
  now.", make one final call with no tools, append and return its text.

Write ./demos/day1_dice.py: a hand-written roll_dice tool — a small object with
.spec = {"schema": {"name": "roll_dice", "description": "Roll count six-sided
dice", "parameters": {"type": "object", "properties": {"count": {"type":
"string", "description": "How many dice"}}, "required": ["count"]}}} and .run
returning the rolls — plus a printing on_event, an always-allow before_tool, and
the task "Roll 3 dice and tell me whether the total beats 10".

Verify before finishing: (1) the dice transcript shows user, assistant tool call,
tool result, assistant answer; (2) the prompt "Build a landing page for a coffee
shop" returns text and no tool calls; (3) git log shows the day's commits.