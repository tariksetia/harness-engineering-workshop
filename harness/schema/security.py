"""Types for the security policy: permission modes and the approval callback.

- Concept: a policy is a mode plus a human-in-the-loop approver.
- Design rule: schemas hold types only — no logic, no I/O.
"""

from collections.abc import Callable
from typing import Literal

from harness.schema.provider import ToolCall

# How much the agent may do unasked:
# - "read-only": only read tools run.
# - "safe": read tools run; everything else needs the approver's yes.
# - "yolo": everything runs except deny-listed bash commands.
type Mode = Literal["read-only", "safe", "yolo"]

# Asked to approve a non-read call in safe mode: (call, reason) -> True to allow.
type Approver = Callable[[ToolCall, str], bool]
