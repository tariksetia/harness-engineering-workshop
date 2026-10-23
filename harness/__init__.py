"""The harness console script: run one prompt through the agent loop.

- Concept: the smallest useful agent — a prompt, the loop, no tools.
- Design rule: the CLI only wires pieces together; behavior lives in loop and provider.
"""

import sys
from typing import cast

from harness.schema.loop import AssistantEvent, EventKind, EventPayload

SYSTEM = "You are a helpful assistant."


def main() -> None:
    """Run the prompt given on the command line and print the reply.

    - Reads the prompt from sys.argv[1:]; exits with usage when it is missing.
    """
    from harness import bedrock  # Deferred so `import harness` stays side-effect free.
    from harness.loop import run_loop

    prompt = " ".join(sys.argv[1:])
    if not prompt:
        sys.exit('usage: harness "<prompt>"')
    reply = run_loop(bedrock.model_id(), SYSTEM, [{"role": "user", "text": prompt}], {}, _print_event, lambda call: None)
    print(reply)


def _print_event(kind: EventKind, payload: EventPayload) -> None:
    """Report tool-call count and usage of each assistant reply on stderr.

    - kind: loop event kind; only "assistant" is reported.
    - payload: event data emitted by the loop.
    """
    if kind == "assistant":
        reply = cast(AssistantEvent, payload)  # The kind names the payload shape.
        print(f"[assistant] tool_calls={len(reply['tool_calls'])} usage={reply['usage']}", file=sys.stderr)
