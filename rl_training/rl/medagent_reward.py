"""Benchmark-aligned episode reward for MedAgentBench GRPO (refsol + shaping).

Implements the plan's primary signal: large terminal reward for refsol pass@1,
plus small dense penalties for illegal syntax, per-step cost, duplicate GETs,
and premature POST on action families.
"""

from __future__ import annotations

import importlib
import json
import logging
import random
from dataclasses import dataclass
from typing import Any

from rl_training.rl.rollout_logger import build_rollout_record, maybe_append_rollout
from rl_training.rl.verifiers.efficiency import analyze_tool_log, canonical_get_key
from rl_training.rl.verifiers.syntax import first_assistant_text, verify_syntax
from rl_training.rl.verifiers.task_masks import is_action_family

logger = logging.getLogger(__name__)

_RUNTIME: dict[str, Any] = {
    "max_rounds": 8,
    "r_succ": 10.0,
    "r_legal_pos": 0.15,
    "r_legal_neg": -0.6,
    "r_exec": 0.05,
    "r_redundant_get": -0.25,
    "r_step": -0.03,
    "r_premature_post": -0.8,
    "r_first_invalid": -3.0,
    # Penalty for rollouts that ran out of turns without ever calling FINISH.
    # Distinguishes "ran out of budget" from "answered wrong" so the optimizer
    # can lean into the cheap fix (call FINISH earlier).
    "r_truncated": -0.5,
    "efficiency_bonus_max": 0.4,
    "rollout_log_path": None,
    "rollout_log_fraction": 0.0,
    "canonicalize": True,
    "fsm_constrained_decode": False,
    # Strict parse mode (used by reward path only, not the env). When true,
    # the first non-thinking line of the assistant turn must *be* the action
    # — no leading garbage. Kills the "bury text + one anchor" parse hack.
    "strict_parse": False,
    # Plan §4: GET +0.05 only on novel digest; cap distinct canonical URLs.
    # Default raised to max_rounds so query-heavy tasks (5+ reads) aren't
    # discouraged after only 4 distinct GETs.
    "max_get_exec_bonus_distinct": 8,
    # Small FINISH JSON-list parse bonus when refsol fails (anneal to 0 in YAML).
    "r_finish_canon": 0.0,
    "rollout_log_sample_failures": True,
}


def configure(cfg: dict[str, Any] | None = None) -> None:
    """Merge ``benchmark_reward`` YAML section into runtime knobs."""
    if not cfg:
        return
    for k, v in cfg.items():
        if k in _RUNTIME and v is not None:
            _RUNTIME[k] = v


@dataclass
class _HistEntry:
    role: str
    content: str


@dataclass
class _ResultsProxy:
    history: list[_HistEntry]
    result: str | None


def _completion_to_refsol_history(completion: list[dict]) -> list[_HistEntry]:
    """Map TRL chat turns into refsol's expected ``agent`` / ``user`` history."""
    hist: list[_HistEntry] = []
    for turn in completion:
        role = turn.get("role", "")
        if role == "assistant":
            role = "agent"
        elif role == "tool":
            # MedAgentBench refsol expects env replies on the ``user`` channel.
            role = "user"
        if role not in ("user", "agent"):
            continue
        if turn.get("tool_calls"):
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
                    hist.append(_HistEntry("agent", f"FINISH({ans})"))
                elif name == "get_fhir_resource":
                    url = args.get("url", "")
                    hist.append(_HistEntry("agent", f"GET {url}"))
                elif name == "post_fhir_resource":
                    url = args.get("url", "")
                    payload = args.get("payload", "")
                    if isinstance(payload, dict):
                        payload = json.dumps(payload)
                    hist.append(_HistEntry("agent", f"POST {url}\n{payload}"))
            continue
        content = turn.get("content")
        if isinstance(content, str) and content.strip():
            hist.append(_HistEntry(role, content))
    return hist


def _extract_finish_result(completion: list[dict]) -> str | None:
    for turn in completion:
        if turn.get("tool_calls"):
            for call in turn["tool_calls"]:
                if (call.get("function", {}).get("name") or "").lower() == "finish":
                    args = call["function"].get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}
                    if isinstance(args, dict):
                        return str(args.get("answers", ""))
        content = turn.get("content", "")
        if isinstance(content, str) and "FINISH(" in content:
            import re
            m = re.search(r"FINISH\((.+)\)", content, re.DOTALL)
            if m:
                return m.group(1)
    return None


def _refsol_task_key(task_id: str) -> str | None:
    for part in task_id.split("_"):
        if part.startswith("task"):
            return part
    return None


def refsol_pass(
    case_data: dict[str, Any],
    completion: list[dict],
    fhir_api_base: str,
    finish_result: str | None,
    trace: dict[str, Any] | None = None,
) -> bool:
    """Run the same refsol grader as ``MedAgentEnv.grade``."""
    if trace is not None:
        trace.pop("refsol_exception", None)
    key = _refsol_task_key(case_data.get("id", ""))
    if not key:
        return False
    refsol = importlib.import_module("src.server.tasks.medagentbench.refsol")
    grader = getattr(refsol, key, None)
    if grader is None:
        return False
    hist = _completion_to_refsol_history(completion)
    results = _ResultsProxy(hist, finish_result)
    try:
        ok = grader(case_data, results, fhir_api_base) is True
        return ok
    except Exception as exc:
        logger.debug("refsol exception for %s: %s", case_data.get("id"), exc)
        if trace is not None:
            trace["refsol_exception"] = f"{type(exc).__name__}: {exc}"
        return False
    finally:
        if trace is not None:
            errs = getattr(results, "extract_posts_errors", None)
            if errs:
                trace["extract_posts_errors"] = list(errs)


def _finish_json_list_ok(s: str | None) -> bool:
    if s is None or not str(s).strip():
        return False
    try:
        v = json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return False
    return isinstance(v, list)


def _efficiency_bonus(num_tool_steps: int, max_rounds: int = 8) -> float:
    """Bonus for finishing well under the round budget.

    ``num_tool_steps`` must be the count of *interaction* steps (GET/POST)
    only — FINISH and any no-op turns must be excluded by the caller, else
    a lone FINISH with no real work would still earn a bonus.
    """
    if num_tool_steps <= 0:
        return 0.0
    m = float(_RUNTIME.get("efficiency_bonus_max", 0.4))
    spare = max(0, max_rounds - num_tool_steps)
    return m * (spare / max_rounds)


def _count_interaction_steps(tool_log: list[dict[str, Any]]) -> int:
    return sum(1 for e in tool_log if e.get("action") in {"GET", "POST"})


def compute_episode_reward(
    env: Any,
    completion: list[dict],
    fhir_api_base: str,
) -> tuple[float, dict[str, Any]]:
    """Scalar reward + structured trace for logging / debugging."""
    case_data = getattr(env, "_task", {}) or {}
    tool_log = list(getattr(env, "_tool_log", []) or [])
    finished = bool(getattr(env, "_finished", False))
    finish_env = getattr(env, "_finish_result", None)
    finish_c = _extract_finish_result(completion)
    finish_result = finish_env if finish_env is not None else finish_c
    return _score_from_extracted(
        case_data=case_data,
        tool_log=tool_log,
        finished=finished,
        finish_result=finish_result,
        completion=completion,
        fhir_api_base=fhir_api_base,
        precomputed_pass=None,
    )


def compute_episode_reward_from_extras(
    *,
    case_data: dict[str, Any],
    tool_log: list[dict[str, Any]],
    finished: bool,
    finish_result: str | None,
    correct: bool,
    fhir_api_base: str,
) -> tuple[float, dict[str, Any]]:
    """Score a plain-text rollout from the extras emitted by ``rollout_func``.

    Mirrors :func:`compute_episode_reward` but takes the env-derived fields
    directly (no environment object) and trusts the caller's ``correct`` flag
    rather than re-invoking refsol. Used by the ``rollout_func`` path for
    plain-text MedAgent GRPO; the JSON-tool environment_factory path keeps
    using ``compute_episode_reward``.
    """
    return _score_from_extracted(
        case_data=case_data,
        tool_log=tool_log,
        finished=finished,
        finish_result=finish_result,
        completion=None,
        fhir_api_base=fhir_api_base,
        precomputed_pass=bool(correct),
    )


def _score_from_extracted(
    *,
    case_data: dict[str, Any],
    tool_log: list[dict[str, Any]],
    finished: bool,
    finish_result: str | None,
    completion: list[dict] | None,
    fhir_api_base: str,
    precomputed_pass: bool | None,
) -> tuple[float, dict[str, Any]]:
    beta = float(_RUNTIME["r_step"])
    trace: dict[str, Any] = {
        "terms": {},
        "refsol_pass": False,
        "first_action_invalid": False,
    }

    if precomputed_pass is not None:
        passed = bool(precomputed_pass)
        trace["refsol_pass_source"] = "precomputed"
    else:
        passed = refsol_pass(
            case_data, completion or [], fhir_api_base, finish_result, trace=trace,
        )
    trace["refsol_pass"] = passed
    max_rounds = int(_RUNTIME.get("max_rounds", 8))
    if passed:
        bonus = _efficiency_bonus(_count_interaction_steps(tool_log), max_rounds)
        trace["terms"]["succ"] = float(_RUNTIME["r_succ"])
        trace["terms"]["efficiency_bonus"] = bonus
        total = float(_RUNTIME["r_succ"]) + bonus
        trace["total"] = total
        trace["reward_trace"] = [
            {"term": "succ", "value": float(_RUNTIME["r_succ"]), "weight": 1.0},
            {"term": "efficiency_bonus", "value": bonus, "weight": 1.0},
        ]
        return total, trace

    R = 0.0
    step_cost_total = 0.0
    if completion is not None:
        first = first_assistant_text(completion)
    elif tool_log:
        # Plain-text rollout: synthesize the first-action text from the first
        # tool_log entry so the first_invalid penalty still has a signal.
        e0 = tool_log[0]
        act0 = e0.get("action", "")
        if act0 == "GET":
            first = f"GET {e0.get('url', '')}"
        elif act0 == "POST":
            first = f"POST {e0.get('url', '')}\n{e0.get('payload') or ''}"
        elif act0 == "FINISH":
            first = f"FINISH({e0.get('answers', '')})"
        else:
            first = None
    else:
        first = None
    reward_trace: list[dict[str, Any]] = []

    def _rt(name: str, value: float, weight: float = 1.0) -> None:
        reward_trace.append({"term": name, "value": value, "weight": weight})

    if first is not None:
        from rl_training.rl.decode_postprocess import apply_decode_postprocess

        text0 = (
            apply_decode_postprocess(first)
            if _RUNTIME.get("canonicalize", True)
            else (first or "").strip()
        )
        strict = bool(_RUNTIME.get("strict_parse", False))
        syn0 = verify_syntax(text0, strict=strict)
        if not syn0.legal:
            fv = float(_RUNTIME.get("r_first_invalid", -3.0))
            R += fv
            trace["first_action_invalid"] = True
            trace["terms"]["first_invalid"] = fv
            _rt("first_invalid", fv)

    r_legal_pos = float(_RUNTIME["r_legal_pos"])
    r_legal_neg = float(_RUNTIME["r_legal_neg"])
    r_exec = float(_RUNTIME["r_exec"])
    ng_cap = int(_RUNTIME.get("max_get_exec_bonus_distinct", 4))
    seen_digest_keys: set[tuple[str, int]] = set()
    distinct_canonical: set[str] = set()

    def _norm_entry(entry: dict[str, Any]) -> str:
        act = entry.get("action", "")
        if act == "GET":
            return f"GET {entry.get('url', '')}"
        if act == "POST":
            pl = entry.get("payload") or ""
            return f"POST {entry.get('url', '')}\n{pl}"
        if act == "FINISH":
            return f"FINISH({entry.get('answers', '')})"
        return ""

    # ``seen_get_attempt`` clears premature_post on *any* GET attempt, even
    # if the snapshot/live FHIR returns an error. The previous "successful
    # GET only" rule meant cache-miss runs paid -0.8 on every POST and the
    # policy could not distinguish "tried to read first" from "skipped reads
    # entirely". (Plan §3 item 2.)
    seen_get_attempt = False
    strict_step = bool(_RUNTIME.get("strict_parse", False))
    step_debug: list[dict[str, Any]] = []
    for entry in tool_log:
        act = entry.get("action", "")
        R += beta
        step_cost_total += beta
        step_terms: dict[str, float] = {"step_cost": beta}

        norm = _norm_entry(entry)
        syn_step = (
            verify_syntax(norm, strict=strict_step)
            if norm.strip() else verify_syntax("", strict=strict_step)
        )
        leg = r_legal_pos if syn_step.legal else r_legal_neg
        R += leg
        _rt(f"legal_{act or 'unk'}", leg)
        step_terms[f"legal_{act or 'unk'}"] = leg

        if act == "GET":
            seen_get_attempt = True
            url = str(entry.get("url") or "")
            rl = int(entry.get("response_len", 0) or 0)
            ck = canonical_get_key(url)
            dkey = (ck, rl)
            novel_digest = dkey not in seen_digest_keys
            seen_digest_keys.add(dkey)
            distinct_canonical.add(ck)
            if entry.get("success"):
                if (
                    novel_digest
                    and len(distinct_canonical) <= ng_cap
                ):
                    R += r_exec
                    _rt("exec_get_novel", r_exec)
                    step_terms["exec_get_novel"] = r_exec
            else:
                pen = r_legal_neg * 0.25
                R += pen
                _rt("get_fail_soft", pen)
                step_terms["get_fail_soft"] = pen
        elif act == "POST":
            if is_action_family(case_data.get("id", "")) and not seen_get_attempt:
                pp = float(_RUNTIME["r_premature_post"])
                R += pp
                trace["terms"]["premature_post"] = trace.get("terms", {}).get(
                    "premature_post", 0.0,
                ) + pp
                _rt("premature_post", pp)
                step_terms["premature_post"] = pp
        step_debug.append({
            "step": entry.get("step"),
            "action": act,
            "terms": step_terms,
            "subtotal": sum(step_terms.values()),
            "syntax_legal": syn_step.legal,
            "syntax_error_class": syn_step.error_class,
        })
    trace["step_debug"] = step_debug

    trace["terms"]["step_cost"] = step_cost_total
    _rt("step_cost", step_cost_total)

    r_fc = float(_RUNTIME.get("r_finish_canon", 0.0))
    if (
        r_fc != 0.0
        and finished
        and _finish_json_list_ok(finish_result)
        and not passed
    ):
        R += r_fc
        trace["terms"]["finish_canon"] = r_fc
        _rt("finish_canon", r_fc)

    eff = analyze_tool_log(tool_log)
    rred = int(eff.get("redundant_gets", 0)) * float(_RUNTIME["r_redundant_get"])
    R += rred
    if rred < 0:
        trace["terms"]["redundant_get"] = rred
        _rt("redundant_get", rred)

    # Truncation penalty: rollout ended without a FINISH action. Lets the
    # optimizer separate "ran out of turns" from "answered wrong". Only
    # applies on the failure branch (refsol passed branch returned earlier).
    if not finished:
        rt = float(_RUNTIME.get("r_truncated", -0.5))
        if rt != 0.0:
            R += rt
            trace["terms"]["truncated"] = rt
            _rt("truncated", rt)

    trace["total"] = R
    trace["efficiency"] = eff
    trace["reward_trace"] = reward_trace
    return R, trace


def score_completions(
    completions: list[list[dict]],
    environments: list[Any],
    fhir_api_base: str,
) -> tuple[list[float], list[dict[str, Any]]]:
    scores: list[float] = []
    traces: list[dict[str, Any]] = []
    for i, comp in enumerate(completions):
        env = environments[i] if i < len(environments) else None
        if env is None:
            scores.append(0.0)
            traces.append({"error": "no_env"})
            continue
        total, trace = compute_episode_reward(env, comp, fhir_api_base)
        scores.append(total)
        traces.append(trace)
        path = _RUNTIME.get("rollout_log_path")
        frac = float(_RUNTIME.get("rollout_log_fraction", 0.0))
        log_fail = bool(_RUNTIME.get("rollout_log_sample_failures", True))
        should_log = path and (
            random.random() < frac
            or (log_fail and not trace.get("refsol_pass"))
        )
        if should_log:
            rec = build_rollout_record(
                env=env,
                completion=comp,
                reward_total=total,
                trace=trace,
                fhir_api_base=fhir_api_base,
                max_rounds=int(_RUNTIME.get("max_rounds", 8)),
                policy={},
            )
            maybe_append_rollout(path, rec)
    return scores, traces
