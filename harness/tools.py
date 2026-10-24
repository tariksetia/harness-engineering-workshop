"""Core tools: the six file and shell tools a coding agent needs.

- Concept: a tool is a plain function plus a JSON-schema spec; the @tool
  decorator derives the spec from the function's signature.
- Design rules:
  1. Every parameter is string-typed on purpose — models send strings anyway,
     so each tool parses its own input.
  2. Every path goes through one resolve(); nothing escapes the working directory.
  3. Outputs are bounded (lines, characters, entries, hits) so one tool call
     cannot flood the context window.
  4. Expected failures come back as "ERROR: ..." text the model can act on.
"""

import fnmatch
import inspect
import os
import re
import subprocess
from collections.abc import Callable, Iterator
from dataclasses import dataclass

from harness.schema.provider import ToolSpec
from harness.schema.tools import ParamDescriptions, PathResolver, ToolFunction

IGNORED_DIRS = {".git", "node_modules", "__pycache__", ".venv"}
MAX_READ_LINES = 4000
MAX_BASH_CHARS = 12000
BASH_KEEP_CHARS = 6000
MAX_LIST_ENTRIES = 500
MAX_GREP_HITS = 200
MAX_GREP_LINE_CHARS = 200


@dataclass
class Tool:
    """A tool the agent loop can execute.

    - name: identifier the model uses to call the tool.
    - spec: the {"schema": ...} declaration the provider expects.
    - run: the function invoked with the model's arguments as keywords.
    """

    name: str
    spec: ToolSpec
    run: ToolFunction


def tool(description: str, **params: str) -> Callable[[ToolFunction], Tool]:
    """Build a decorator that turns a plain function into a Tool.

    - description: what the tool does, shown to the model.
    - params: parameter name -> description shown to the model; parameters
      without an entry are described by their name.
    """

    def decorate(fn: ToolFunction) -> Tool:
        """Derive the Tool's spec from fn's signature.

        - fn: the tool body; parameters with defaults become optional.
        """
        return Tool(name=fn.__name__, spec=_spec(fn, description, params), run=fn)

    return decorate


def core_tools(workdir: str) -> list[Tool]:
    """Return the six core tools, all confined to workdir.

    - workdir: directory the tools operate in; symlinks are resolved once, up front.
    """
    root = os.path.realpath(workdir)
    resolve = _resolver(root)

    @tool("Read a text file; lines are numbered N<TAB>line.", path="File path relative to the working directory")
    def read_file(path: str) -> str:
        """Return the file's contents with line numbers.

        - path: file to read.
        """
        with open(resolve(path), encoding="utf-8", errors="replace") as file:
            lines = file.read().splitlines()
        numbered = "\n".join(f"{n}\t{line}" for n, line in enumerate(lines[:MAX_READ_LINES], start=1))
        if len(lines) > MAX_READ_LINES:
            numbered += f"\n... truncated: showing {MAX_READ_LINES} of {len(lines)} lines"
        return numbered

    @tool(
        "Create or overwrite a file, creating parent directories.",
        path="File path relative to the working directory",
        content="Full new contents of the file",
    )
    def write_file(path: str, content: str) -> str:
        """Write content to path.

        - path: file to write.
        - content: the complete new contents.
        """
        real = resolve(path)
        os.makedirs(os.path.dirname(real), exist_ok=True)
        with open(real, "w", encoding="utf-8") as file:
            file.write(content)
        return f"Wrote {len(content)} chars to {path}"

    @tool(
        "Replace one exact snippet in a file. The snippet must appear exactly once.",
        path="File path relative to the working directory",
        old="Exact text to replace, copied from the file",
        new="Replacement text",
    )
    def edit_file(path: str, old: str, new: str) -> str:
        """Replace the single occurrence of old with new.

        - path: file to edit.
        - old: snippet to find; must be unique in the file.
        - new: text to put in its place.
        """
        real = resolve(path)
        with open(real, encoding="utf-8") as file:
            text = file.read()
        # Uniqueness is the safety rule: an ambiguous snippet could edit the
        # wrong spot, so the model must add context rather than guess.
        count = text.count(old)
        if count == 0:
            return "ERROR: snippet not found — read the file and copy it exactly"
        if count > 1:
            return f"ERROR: snippet appears {count} times — include more context to make it unique"
        with open(real, "w", encoding="utf-8") as file:
            file.write(text.replace(old, new, 1))
        return f"Edited {path}"

    @tool(
        "Run a shell command in the working directory; returns stdout+stderr.",
        command="Shell command to run",
        timeout="Seconds before the command is killed (default 120)",
    )
    def bash(command: str, timeout: str = "120") -> str:
        """Run command through the shell and return its combined output.

        - command: shell command line.
        - timeout: seconds to wait, as a string.
        """
        seconds = float(timeout)
        try:
            done = subprocess.run(command, shell=True, cwd=root, capture_output=True, text=True, timeout=seconds)
        except subprocess.TimeoutExpired:
            return f"ERROR: timed out after {timeout}s"
        output = done.stdout + done.stderr
        if len(output) > MAX_BASH_CHARS:
            # Head and tail carry the signal: the command's start and its final error.
            omitted = len(output) - 2 * BASH_KEEP_CHARS
            output = f"{output[:BASH_KEEP_CHARS]}\n... [{omitted} chars truncated] ...\n{output[-BASH_KEEP_CHARS:]}"
        return output or f"(exit {done.returncode}, no output)"

    @tool("List files matching a glob pattern.", pattern="Glob matched against relative path or file name (default **/*)")
    def list_files(pattern: str = "**/*") -> str:
        """Return the sorted relative paths of matching files.

        - pattern: fnmatch glob.
        """
        matches = sorted(rel for rel in _walk(root) if _matches(rel, pattern))
        listing = "\n".join(matches[:MAX_LIST_ENTRIES])
        if len(matches) > MAX_LIST_ENTRIES:
            listing += f"\n... and {len(matches) - MAX_LIST_ENTRIES} more"
        return listing or "(no files)"

    @tool(
        "Search file contents for a regular expression; returns path:lineno: text.",
        regex="Python regular expression",
        pattern="Glob restricting which files are searched (default *)",
    )
    def grep(regex: str, pattern: str = "*") -> str:
        """Return matching lines across files matching pattern.

        - regex: expression searched for on each line.
        - pattern: fnmatch glob selecting the files.
        """
        compiled = re.compile(regex)
        hits: list[str] = []
        for rel in sorted(rel for rel in _walk(root) if _matches(rel, pattern)):
            for lineno, line in _text_lines(os.path.join(root, rel)):
                if compiled.search(line):
                    hits.append(f"{rel}:{lineno}: {line[:MAX_GREP_LINE_CHARS]}")
                    if len(hits) == MAX_GREP_HITS:
                        return "\n".join(hits) + f"\n... capped at {MAX_GREP_HITS} hits"
        return "\n".join(hits) or "(no matches)"

    return [read_file, write_file, edit_file, bash, list_files, grep]


def _spec(fn: ToolFunction, description: str, params: ParamDescriptions) -> ToolSpec:
    """Build the provider spec for fn from its signature.

    - fn: the tool body.
    - description: what the tool does.
    - params: parameter name -> description.
    """
    signature = inspect.signature(fn).parameters
    return {
        "schema": {
            "name": fn.__name__,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {name: {"type": "string", "description": params.get(name, name)} for name in signature},
                "required": [name for name, p in signature.items() if p.default is inspect.Parameter.empty],
            },
        }
    }


def _resolver(root: str) -> PathResolver:
    """Return the single path gate for every tool.

    - root: real path of the working directory.
    """

    def resolve(path: str) -> str:
        """Return the real path of path, refusing anything outside root.

        - path: path from the model, relative to root (or absolute).
        """
        # realpath follows symlinks and "..", so a link pointing outside the
        # tree is caught exactly like a literal ../../ path.
        real = os.path.realpath(os.path.join(root, path))
        if os.path.commonpath([root, real]) != root:
            raise PermissionError(f"{path!r} escapes the working directory")
        return real

    return resolve


def _walk(root: str) -> Iterator[str]:
    """Yield every file under root as a relative path, skipping IGNORED_DIRS.

    - root: directory to walk.
    """
    for directory, subdirs, files in os.walk(root):
        subdirs[:] = [d for d in subdirs if d not in IGNORED_DIRS]  # Prunes in place: os.walk reads it.
        for name in files:
            yield os.path.relpath(os.path.join(directory, name), root)


def _matches(rel: str, pattern: str) -> bool:
    """Tell whether a relative path matches a glob.

    - rel: path relative to the working directory.
    - pattern: fnmatch glob, tried against the relative path and the basename.
    """
    # fnmatch has no "**": "**/*" needs a slash, so top-level files are also
    # matched against the pattern with its leading "**/" removed.
    candidates = [pattern, pattern.removeprefix("**/")]
    return any(fnmatch.fnmatch(rel, p) or fnmatch.fnmatch(os.path.basename(rel), p) for p in candidates)


def _text_lines(path: str) -> Iterator[tuple[int, str]]:
    """Yield (lineno, line) for a text file; yield nothing for binary or unreadable files.

    - path: absolute file path.
    """
    try:
        with open(path, encoding="utf-8") as file:
            yield from enumerate((line.rstrip("\n") for line in file), start=1)
    except (UnicodeDecodeError, OSError):
        return
