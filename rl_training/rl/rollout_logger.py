"""Append-only JSONL rollout logging for reward debugging."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from typing import Any

from rl_training.env.action_parser import parse_action
from rl_training.rl.verifiers.efficiency import analyze_tool_log, canonical_get_key
from rl_training.rl.verifiers.fhir_exec import get_error_class
from rl_training.rl.verifiers.syntax import verify_syntax
from rl_training.rl.verifiers.task_masks import is_action_family

_lock = threading.Lock()


def maybe_append_rollout(path: str | None, record: dict[str, Any]) -> None:
    """Best-effort append one JSON line; creates parent dirs."""
    if not path:
        return
    line = json.dumps(record, ensure_ascii=False, default=str)
    with _lock:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def _canonical_finish_json(obj: Any) -> str | None:
    if obj is None:
        return None
    try:
        return json.dumps(obj, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return None


def _digest(entry: dict[str, Any]) -> str:
    url = str(entry.get("url") or "")
    h = hashlib.sha256(url.encode("utf-8", errors="replace")).hexdigest()[:10]
    ln = int(entry.get("response_len") or 0)
    return f"h={h} len={ln}"


def _tool_entry_to_normalized(entry: dict[str, Any]) -> str:
    act = entry.get("action", "")
    if act == "GET":
        return f"GET {entry.get('url', '')}"
    if act == "POST":
        payload = entry.get("payload") or ""
        return f"POST {entry.get('url', '')}\n{payload}"
    if act == "FINISH":
        return f"FINISH({entry.get('answers', '')})"
    return ""


def build_rollout_record(
    *,
    env: Any,
    completion: list[dict],
    reward_total: float,
    trace: dict[str, Any],
    fhir_api_base: str,
    max_rounds: int = 8,
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One JSON object per episode (plan §3 schema, best-effort)."""
    task = getattr(env, "_task", {}) or {}
    tid = task.get("id", "")
    raw_tail: str | None = None
    for turn in reversed(completion or []):
        if turn.get("role") == "assistant":
            c = turn.get("content")
            if isinstance(c, str) and c.strip():
                raw_tail = c[:2000]
                break
    tool_log = list(getattr(env, "_tool_log", []) or [])
    finished = bool(getattr(env, "_finished", False))
    step_count = int(getattr(env, "_step_count", 0))
    eff = analyze_tool_log(tool_log)
    num_get = sum(1 for e in tool_log if e.get("action") == "GET")
    num_post = sum(1 for e in tool_log if e.get("action") == "POST")

    if trace.get("first_action_invalid"):
        term_status = "invalid_action"
    elif finished:
        term_status = "completed"
    elif step_count >= max_rounds:
        term_status = "limit_reached"
    else:
        term_status = "truncated"

    step_debug_list = list(trace.get("step_debug") or [])
    step_debug_by_step: dict[Any, dict[str, Any]] = {}
    for sd in step_debug_list:
        if "step" in sd:
            step_debug_by_step[sd["step"]] = sd

    steps: list[dict[str, Any]] = []
    seen_get_digests: set[tuple[str, int]] = set()
    for idx, entry in enumerate(tool_log):
        norm = _tool_entry_to_normalized(entry)
        syn = verify_syntax(norm) if norm else verify_syntax("")
        act = entry.get("action", "")
        parsed = parse_action(norm) if norm.strip() else parse_action("")
        pargs: dict[str, Any] = {}
        if parsed.kind == "get":
            pargs = {"url": parsed.url}
        elif parsed.kind == "post":
            pargs = {"url": parsed.url}
        elif parsed.kind == "finish":
            pargs = {"result": parsed.result}
        err_cls = syn.error_class or get_error_class(entry)
        novel = True
        if act == "GET":
            dk = (
                canonical_get_key(str(entry.get("url") or "")),
                int(entry.get("response_len", 0) or 0),
            )
            novel = dk not in seen_get_digests
            if entry.get("success"):
                seen_get_digests.add(dk)
        env_obs: dict[str, Any] = {
            "http_ok": bool(entry.get("success")),
            "observation_digest": _digest(entry) if act == "GET" else None,
            "cache_hit": None,
        }
        steps.append(
            {
                "t": entry.get("step"),
                "role": "agent",
                "normalized_text": norm,
                "parse": {
                    "kind": syn.kind,
                    "legal": syn.legal,
                    "error_class": err_cls,
                    "args": pargs,
                },
                "env": env_obs,
                "progress": {
                    "info_gain_proxy": 0.0,
                    "novel_url": novel,
                    "violates_task_constraints": False,
                },
                "reward_debug": (
                    step_debug_by_step.get(entry.get("step"))
                    or (step_debug_list[idx] if idx < len(step_debug_list) else {})
                ),
            },
        )

    reward_trace: list[dict[str, Any]] = list(trace.get("reward_trace") or [])
    if not reward_trace:
        for name, val in (trace.get("terms") or {}).items():
            if isinstance(val, (int, float)):
                reward_trace.append({"term": name, "value": float(val), "weight": 1.0})

    finish_answers = getattr(env, "_finish_result", None)
    parsed_finish = None
    if isinstance(finish_answers, str) and finish_answers.strip():
        try:
            parsed_finish = json.loads(finish_answers)
        except json.JSONDecodeError:
            parsed_finish = None

    redundant_rate = 0.0
    if num_get > 0:
        redundant_rate = float(eff.get("redundant_gets", 0)) / float(num_get)

    canon_finish = _canonical_finish_json(parsed_finish)

    return {
        "schema_version": 1,
        "task_id": tid,
        "task_family": "action" if is_action_family(tid) else "query",
        "benchmark_split": "train",
        "initial_instruction": task.get("instruction", ""),
        "context": task.get("context", ""),
        "fhir_api_base": fhir_api_base,
        "max_rounds": max_rounds,
        "policy": policy or {},
        "raw_generation_tail": raw_tail,
        "steps": steps,
        "termination": {"status": term_status, "round": step_count},
        "final": {
            "raw_finish": finish_answers,
            "parsed_finish_json": parsed_finish,
            "canonical_finish_json": canon_finish,
            "refsol_pass": bool(trace.get("refsol_pass")),
            "refsol_exception": trace.get("refsol_exception"),
            "finish_raw": finish_answers,
        },
        "aggregate": {
            "success": bool(trace.get("refsol_pass")),
            "invalid_rate": 1.0 if trace.get("first_action_invalid") else 0.0,
            "limit_rate": term_status == "limit_reached",
            "num_get": num_get,
            "num_post": num_post,
            "unique_canonical_get_urls": int(eff.get("unique_canonical_url_lens", 0)),
            "redundant_get_rate": redundant_rate,
        },
        "reward_total": reward_total,
        "reward_trace": reward_trace,
        "trace": trace,
    }
