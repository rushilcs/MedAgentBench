"""Action syntax verification (MedAgentBench GET/POST/FINISH grammar)."""

from __future__ import annotations

import json
from dataclasses import dataclass

from rl_training.env.action_parser import parse_action


@dataclass
class SyntaxResult:
    legal: bool
    kind: str
    error_class: str | None


def verify_syntax(raw_text: str, strict: bool = False) -> SyntaxResult:
    """Return legality and coarse error class for one agent turn.

    ``strict`` (off by default for backward compatibility) requires the
    parser to find the action at the first non-thinking line — no leading
    garbage. Used by the reward path to penalize "bury text + one anchor"
    parsing hacks; the env stays lenient so good policies aren't held back.
    """
    if not raw_text or not str(raw_text).strip():
        return SyntaxResult(False, "invalid", "EMPTY")
    p = parse_action(str(raw_text), strict=strict)
    if p.kind == "invalid":
        return SyntaxResult(False, "invalid", "INVALID_SYNTAX")
    if p.kind == "post" and p.payload is None:
        return SyntaxResult(False, "post", "INVALID_POST_JSON")
    return SyntaxResult(True, p.kind, None)


def first_assistant_text(completion: list[dict]) -> str | None:
    """First assistant / agent message content (tool_calls or text)."""
    for turn in completion:
        role = turn.get("role", "")
        if role not in ("assistant", "agent"):
            continue
        if turn.get("tool_calls"):
            # Native tool path: synthesize FINISH/GET/POST string for parse_action
            for call in turn["tool_calls"]:
                fn = call.get("function", {}) or {}
                name = (fn.get("name") or "").lower()
                args = fn.get("arguments", {}) or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                if name == "finish":
                    ans = args.get("answers", "")
                    return f"FINISH({ans})"
                if name == "get_fhir_resource":
                    url = args.get("url", "")
                    return f"GET {url}"
                if name == "post_fhir_resource":
                    url = args.get("url", "")
                    payload = args.get("payload", "")
                    if isinstance(payload, dict):
                        import json
                        payload = json.dumps(payload)
                    return f"POST {url}\n{payload}"
        content = turn.get("content")
        if isinstance(content, str) and content.strip():
            return content
    return None
