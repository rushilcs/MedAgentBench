"""Parse agent actions (GET / POST / FINISH) from raw text."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal


@dataclass
class ParsedAction:
    kind: Literal["get", "post", "finish", "invalid"]
    url: str = ""
    payload: dict[str, Any] | None = None
    result: str = ""
    raw: str = ""


def parse_action(text: str) -> ParsedAction:
    """Parse a single agent response into a structured action."""
    raw = text.strip().replace("```tool_code", "").replace("```", "").strip()

    if raw.startswith("GET"):
        url = raw[3:].strip()
        if "&_format=json" not in url and "?_format=json" not in url:
            url += "&_format=json" if "?" in url else "?_format=json"
        return ParsedAction(kind="get", url=url, raw=raw)

    if raw.startswith("POST"):
        lines = raw.split("\n", 1)
        url = lines[0][4:].strip()
        payload = None
        if len(lines) > 1:
            try:
                payload = json.loads(lines[1])
            except json.JSONDecodeError:
                return ParsedAction(kind="invalid", raw=raw)
        return ParsedAction(kind="post", url=url, payload=payload, raw=raw)

    if raw.startswith("FINISH("):
        result = raw[len("FINISH("):-1] if raw.endswith(")") else raw[len("FINISH("):]
        return ParsedAction(kind="finish", result=result, raw=raw)

    return ParsedAction(kind="invalid", raw=raw)
