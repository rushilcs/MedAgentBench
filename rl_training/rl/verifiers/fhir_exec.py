"""FHIR execution checks for RL shaping (Layer 2 in plan §7)."""

from __future__ import annotations

from typing import Any


def get_error_class(entry: dict[str, Any]) -> str | None:
    """Map a ``MedAgentBenchEnv._tool_log`` row to a coarse executor error class."""
    act = entry.get("action", "")
    if act == "GET":
        if entry.get("success"):
            return None
        return "GET_HTTP_ERROR"
    if act == "POST":
        if entry.get("success"):
            return None
        err = str(entry.get("error", "")).lower()
        if "json" in err:
            return "INVALID_POST_JSON"
        return "POST_REJECTED"
    if act == "FINISH":
        return None
    return None


def get_json_ok_proxy(entry: dict[str, Any]) -> bool:
    """True when GET succeeded with non-empty body proxy (length > 0)."""
    if entry.get("action") != "GET" or not entry.get("success"):
        return False
    return int(entry.get("response_len", 0) or 0) > 0
