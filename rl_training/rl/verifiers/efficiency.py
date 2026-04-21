"""Trajectory efficiency signals (duplicate GET, step counts)."""

from __future__ import annotations

from typing import Any

from rl_training.env.fhir_snapshot import _canonicalize_url


def canonical_get_key(url: str) -> str:
    return _canonicalize_url(url)


def analyze_tool_log(
    tool_log: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize GET redundancy from MedAgentBenchEnv._tool_log."""
    per_key: dict[tuple[str, int], int] = {}
    redundant = 0
    novel_gets = 0
    for entry in tool_log:
        if entry.get("action") != "GET" or not entry.get("success"):
            continue
        url = entry.get("url") or ""
        key = (canonical_get_key(url), int(entry.get("response_len", 0)))
        c = per_key.get(key, 0)
        per_key[key] = c + 1
        if c == 0:
            novel_gets += 1
        else:
            redundant += 1
    return {
        "novel_gets": novel_gets,
        "redundant_gets": redundant,
        "unique_canonical_url_lens": len(per_key),
    }
