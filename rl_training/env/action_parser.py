"""Parse agent actions (GET / POST / FINISH) from raw text."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal


@dataclass
class ParsedAction:
    kind: Literal["get", "post", "finish", "invalid"]
    url: str = ""
    payload: dict[str, Any] | None = None
    result: str = ""
    raw: str = ""


# Qwen3 / DeepSeek-style reasoning models emit <think>...</think> blocks
# before the actual action. We strip the entire block plus any leading
# channel/role tokens that slip into the completion (e.g. "<|channel|>final").
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_ROLE_PREFIX_RE = re.compile(
    r"^(?:<\|[^|]*\|>[^\n]*\n)+",
)
# In case the closing </think> is truncated by max_tokens, fall back to
# cutting everything up to the last </think> OR to the first GET/POST/FINISH.
_OPEN_THINK_RE = re.compile(r"<think>.*", re.DOTALL | re.IGNORECASE)
_ACTION_ANCHOR_RE = re.compile(
    r"(?ms)^\s*(GET\s+\S|POST\s+\S|FINISH\s*\()",
)


def _strip_reasoning(text: str) -> str:
    """Remove <think>...</think> blocks and common reasoning/channel prefixes."""
    t = text
    # Drop any role-channel prefixes that appear at the very start (harmless if absent).
    t = _ROLE_PREFIX_RE.sub("", t.lstrip())
    # Drop all properly-closed think blocks.
    t = _THINK_RE.sub("", t)
    # If an unclosed <think> remains, and we can still find a GET/POST/FINISH
    # anchor later in the text, cut the prefix up to that anchor.
    if "<think>" in t.lower():
        anchor = _ACTION_ANCHOR_RE.search(t)
        if anchor:
            t = t[anchor.start():]
        else:
            # Nothing after the (possibly truncated) think block -> keep as is;
            # the caller will mark it invalid.
            t = _OPEN_THINK_RE.sub("", t)
    return t.strip()


def parse_action(text: str) -> ParsedAction:
    """Parse a single agent response into a structured action.

    Robust to Qwen3-style <think>...</think> reasoning prefixes, markdown
    code fences (```), and channel/role tokens.
    """
    t = _strip_reasoning(text)
    raw = t.replace("```tool_code", "").replace("```", "").strip()

    # If the anchor still isn't at the very start (e.g. a short lead-in remains),
    # slice from the first anchor occurrence.
    if not (raw.startswith("GET") or raw.startswith("POST") or raw.startswith("FINISH(")):
        anchor = _ACTION_ANCHOR_RE.search(raw)
        if anchor:
            raw = raw[anchor.start():].strip()

    if raw.startswith("GET"):
        url_block = raw[3:].split("\n", 1)[0].strip()
        if "&_format=json" not in url_block and "?_format=json" not in url_block:
            url_block += "&_format=json" if "?" in url_block else "?_format=json"
        return ParsedAction(kind="get", url=url_block, raw=raw)

    if raw.startswith("POST"):
        lines = raw.split("\n", 1)
        url = lines[0][4:].strip()
        payload = None
        if len(lines) > 1:
            body = lines[1].strip()
            # The body may have trailing text after JSON; parse the first
            # balanced JSON object/array.
            try:
                payload = json.loads(body)
            except json.JSONDecodeError:
                # Try to extract the first JSON blob.
                try:
                    decoder = json.JSONDecoder()
                    payload, _ = decoder.raw_decode(body)
                except json.JSONDecodeError:
                    return ParsedAction(kind="invalid", raw=raw)
        return ParsedAction(kind="post", url=url, payload=payload, raw=raw)

    if raw.startswith("FINISH("):
        # Handle multi-line FINISH(...) that may include newlines in the list.
        inner = raw[len("FINISH("):]
        depth = 1
        end_idx = None
        for i, ch in enumerate(inner):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    end_idx = i
                    break
        result = inner[:end_idx] if end_idx is not None else inner
        return ParsedAction(kind="finish", result=result, raw=raw)

    return ParsedAction(kind="invalid", raw=raw)
