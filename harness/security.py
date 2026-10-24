"""Security policy: decide, before each tool call, whether it may run.

- Concept: permission is a pure function of the call and the mode; the loop
  only asks, via before_tool=policy.check.
- Design rules:
  1. Deny patterns win over every mode — even yolo cannot run them.
  2. Reads are always allowed; writes and shell need the mode's consent.
  3. Fail closed: the default approver refuses, and only an explicit True approves.
"""

import re
from typing import get_args

from harness.schema.loop import BlockReason
from harness.schema.provider import ToolCall
from harness.schema.security import Approver, Mode

READ_TOOLS = {"read_file", "list_files", "grep"}

DENY_PATTERNS = [
    r"\brm\s+(-\S+\s+)+[\"']?(/|~|\$HOME|\$\{HOME\})[\"']?/?\*?(\s|;|&|\||$)",  # rm -rf / ~ $HOME
    r"\bsudo\b",
    r"\bmkfs(\.\w+)?\b",
    r"\bdd\s+if=",
    r"\bcurl\b[^|]*\|\s*(sudo\s+)?(ba|z|da)?sh\b",  # curl ... | sh
    r"\bgit\s+push\b.*(--force|\s-f\b)",
    r">\s*/dev/sd[a-z]",
]


def _refuse(call: ToolCall, reason: str) -> bool:
    """Default approver: refuse everything.

    - call: the tool call awaiting approval (ignored).
    - reason: why approval is needed (ignored).
    """
    return False


class Policy:
    """A permission mode plus the approver consulted in safe mode.

    - mode: "read-only", "safe", or "yolo".
    - approver: (call, reason) -> bool; defaults to refusing.
    """

    def __init__(self, mode: Mode = "safe", approver: Approver | None = None) -> None:
        """Validate and store the mode and approver.

        - mode: permission mode; raises ValueError when unknown.
        - approver: approval callback for safe mode; None means refuse.
        """
        if mode not in get_args(Mode.__value__):
            raise ValueError(f"unknown mode {mode!r}")
        self.mode = mode
        self.approver = approver or _refuse

    def check(self, call: ToolCall) -> BlockReason:
        """Return None to allow the call or a reason string to block it.

        - call: the tool call the model requested.
        """
        if call["name"] == "bash":
            command = str(call["args"].get("command", ""))
            for pattern in DENY_PATTERNS:
                if re.search(pattern, command):
                    return f"command matches deny pattern {pattern!r}"
        if call["name"] in READ_TOOLS or self.mode == "yolo":
            return None
        if self.mode == "read-only":
            return f"{call['name']} is not allowed in read-only mode"
        reason = f"{call['name']} needs approval in safe mode"
        # Only an explicit True approves; None, "y", or any other truthy value blocks.
        if self.approver(call, reason) is True:
            return None
        return f"approval denied: {reason}"
