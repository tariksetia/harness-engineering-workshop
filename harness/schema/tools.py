"""Types for the core tools: plain functions the decorator turns into Tools.

- Concept: a tool body is an ordinary function whose keyword arguments are
  all strings and whose result is text the model reads.
- Design rule: schemas hold types only — no logic, no I/O.
"""

from collections.abc import Callable

# A tool body: called with the model's string arguments as keywords, returns text.
type ToolFunction = Callable[..., str]

# Maps a tool parameter name to the description the model sees for it.
type ParamDescriptions = dict[str, str]

# Turns a path the model supplied into a real path inside the working directory.
type PathResolver = Callable[[str], str]
