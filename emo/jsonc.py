"""Minimal JSONC reader (JSON + comments), stdlib-only.

The experiment configs were migrated from YAML to JSON so the package carries
no third-party parser dependency: ``pyyaml`` cannot be installed in the
offline target environment. Plain JSON has no comments, but the configs carry
meaningful explanatory comments, so we accept *JSONC*: standard JSON extended
with ``//`` line comments and ``/* ... */`` block comments. Comments are
stripped here before the text is handed to the stdlib :mod:`json` parser.

``//`` and ``/*`` that occur *inside* a string literal are preserved, and
escaped quotes inside strings are handled correctly.
"""

from __future__ import annotations

import json
from typing import Any


def strip_comments(text: str) -> str:
    """Remove // and /* */ comments, leaving string literals untouched."""
    out: list[str] = []
    i, n = 0, len(text)
    in_string = False
    while i < n:
        ch = text[i]
        if in_string:
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(text[i + 1])  # keep the escaped char verbatim
                i += 2
                continue
            if ch == '"':
                in_string = False
            i += 1
            continue
        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
            continue
        if ch == "/" and i + 1 < n and text[i + 1] == "/":
            i += 2  # line comment: skip to (but keep) the newline
            while i < n and text[i] != "\n":
                i += 1
            continue
        if ch == "/" and i + 1 < n and text[i + 1] == "*":
            i += 2  # block comment: skip through the closing */
            while i + 1 < n and not (text[i] == "*" and text[i + 1] == "/"):
                i += 1
            i += 2
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def loads(text: str) -> Any:
    return json.loads(strip_comments(text))


def load_jsonc(path: str) -> Any:
    """Parse a JSONC file from disk (comments allowed)."""
    with open(path, "r", encoding="utf-8") as f:
        return loads(f.read())
