"""Clinical novelty rewards for MedAgentBench GRPO training.

These deterministic, rule-based reward functions complement the generic
``correctness/efficiency/tool_usage`` rewards in ``trl_rewards.py``. They
target the failure modes most specific to medical agent RL:

  * **temporal_grounding_reward (TCG, headline novelty)** - penalises the
    rollout for using FHIR evidence outside the task's stated time window
    ("now - 24 h" for tasks 4/5/6, "now - 1 year" for task 10, etc.). The
    timestamps are extracted by :mod:`rl_training.env.trl_env` at GET time
    and stored on ``env._tool_log``; this function reads that log.
  * **risk_calibrated_deferral_reward (RCD)** - reward correct no-op /
    escalate behaviour on the two conditional-ordering tasks (5: magnesium,
    9: potassium). Heavy-handed agents that POST regardless are penalised.
  * **decision_density_reward (DDB)** - length/structure shaping of the
    final answer, only credited above a correctness floor. Keeps answers
    parseable without destroying exploratory chain-of-thought.

All three are deterministic, inspectable, and refsol-aligned. The
reward-hacking mitigations are documented inline.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from typing import Any


# ---------------------------------------------------------------- task meta


# Task-type window configuration. "now" is the canonical "2023-11-13T10:15:00"
# that appears in every MedAgentBench task context. ``window_hours`` is the
# lookback horizon that the task instruction implies.
#
# Only task types that explicitly reference a recency window are included
# here. Other task types disable the TCG reward (returns 0.0).
TASK_NOW_ISO = "2023-11-13T10:15:00+00:00"

TIME_SCOPED_TASKS: dict[int, dict[str, Any]] = {
    4:  {"window_hours": 24,        "conditional": False},
    5:  {"window_hours": 24,        "conditional": True},
    6:  {"window_hours": 24,        "conditional": False},
    7:  {"window_hours": None,      "conditional": False},  # "most recent"
    9:  {"window_hours": None,      "conditional": True},
    10: {"window_hours": 24 * 365,  "conditional": True},  # 1 year
}

CONDITIONAL_TASKS: set[int] = {5, 9, 10}


# ---------------------------------------------------------------- utilities


def _parse_task_type(task_id: str) -> int | None:
    """Extract the integer task type from a task id like ``task4_12`` or
    ``train_task9_3``. Returns ``None`` if it cannot be parsed.
    """
    for part in task_id.split("_"):
        if part.startswith("task"):
            try:
                return int(part[len("task"):])
            except ValueError:
                return None
    return None


_ISO_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?(?:[+-]\d{2}:?\d{2}|Z)?)?)"
)


def _try_parse_iso(ts: str) -> datetime | None:
    """Best-effort ISO-8601 parser that tolerates the formats appearing in
    FHIR (``Z`` suffix, missing seconds, naive / aware)."""
    if not ts:
        return None
    s = ts.strip().replace("Z", "+00:00")
    # Pad missing seconds: 2023-11-13T10:15+00:00 -> 2023-11-13T10:15:00+00:00
    short = re.match(
        r"^(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2})(?=[+-]|$)", s,
    )
    if short:
        s = short.group(1) + ":00" + s[short.end():]
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def _extract_finish_answer(completion: list[dict]) -> str | None:
    """Return the answer string passed to ``finish(...)`` by the rollout."""
    for turn in completion:
        if turn.get("tool_calls"):
            for call in turn["tool_calls"]:
                if call.get("function", {}).get("name") == "finish":
                    args = call["function"].get("arguments", {})
                    if isinstance(args, dict):
                        return args.get("answers")
                    # Some TRL versions serialize arguments as JSON string
                    if isinstance(args, str):
                        try:
                            return json.loads(args).get("answers")
                        except Exception:
                            pass
        content = turn.get("content", "")
        if isinstance(content, str) and "FINISH(" in content:
            match = re.search(r"FINISH\((.+)\)", content, re.DOTALL)
            if match:
                return match.group(1)
    return None


def _cited_timestamps_from_text(text: str) -> list[str]:
    """Find every ISO-8601 timestamp inside a piece of assistant text."""
    if not text:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for m in _ISO_RE.finditer(text):
        ts = m.group(1)
        if ts not in seen:
            seen.add(ts)
            out.append(ts)
    return out


def _assistant_text(completion: list[dict]) -> str:
    """Join all assistant message content in a completion."""
    parts: list[str] = []
    for turn in completion:
        if turn.get("role") == "assistant":
            content = turn.get("content")
            if isinstance(content, str):
                parts.append(content)
    return "\n".join(parts)


def _env_tool_log(env: Any) -> list[dict[str, Any]]:
    return list(getattr(env, "_tool_log", []) or [])


def _env_task_id(env: Any, fallback_task_id: str = "") -> str:
    task = getattr(env, "_task", None) or {}
    return task.get("id") or fallback_task_id


# ----------------------------------------------------------- TCG (headline)


def temporal_grounding_reward(completions: list[list[dict]], **kwargs) -> list[float]:
    """Reward in-window evidence use and penalise temporal collapse.

    Logic per rollout (given a time-scoped task type):

      * Collect every ISO timestamp the environment observed in GET responses
        (``env._tool_log[*].timestamps``).
      * Collect every ISO timestamp the model *cited* in its assistant turns
        and in the ``finish(...)`` answer.
      * For each cited timestamp, classify it as:
          - *in-window*:    ``now - window_hours <= ts <= now``
          - *future*:       ``ts > now``
          - *out-of-window*: ``ts < now - window_hours``
          - *unparseable*:  reject as not a timestamp
      * Also check whether the agent cited **any** observed timestamp at all
        on a time-scoped task - "no timestamped evidence" is its own failure.

    Scoring:

        r =  +1.0   if the agent cited >=1 in-window ts and zero out-of-window
                    / future timestamps
             -1.0   if any cited ts is future or out-of-window
             -0.5   if no timestamp was cited on a time-scoped task
              0.0   if the task is not time-scoped (TCG does not apply)

    This reward is the scientific headline of the clinical novelty. Weight
    in the trainer: 0.8.

    Reward-hacking surfaces + mitigations:
      - "Cite a random in-window date": the ts must match one the environment
        actually observed; we intersect with ``env._tool_log`` timestamps.
      - "Cite every ts it sees": one bad ts causes the -1.0 branch.
      - "Never cite any timestamp": the -0.5 branch is strictly worse than
        not citing a random correct one, so the RL gradient pushes toward
        grounded citation.
    """
    environments = kwargs.get("environments", [])
    task_ids = kwargs.get("task_id", []) or []
    rewards: list[float] = []

    for i, completion in enumerate(completions):
        env = environments[i] if i < len(environments) else None
        task_id = _env_task_id(env, task_ids[i] if i < len(task_ids) else "")
        task_type = _parse_task_type(task_id)
        meta = TIME_SCOPED_TASKS.get(task_type or -1)
        if meta is None:
            rewards.append(0.0)
            continue

        now = _try_parse_iso(TASK_NOW_ISO)
        if now is None:
            rewards.append(0.0)
            continue
        window = (
            timedelta(hours=meta["window_hours"])
            if meta["window_hours"] is not None else None
        )

        # Timestamps the env actually observed in GET responses
        observed: set[str] = set()
        if env is not None:
            for entry in _env_tool_log(env):
                for ts in entry.get("timestamps", []) or []:
                    observed.add(ts)

        # Timestamps the model cited (assistant text + finish answer)
        text = _assistant_text(completion)
        finish_answer = _extract_finish_answer(completion) or ""
        if isinstance(finish_answer, str):
            text = text + "\n" + finish_answer
        cited = _cited_timestamps_from_text(text)

        if not cited:
            rewards.append(-0.5)
            continue

        in_window = 0
        bad = 0
        for ts_str in cited:
            # Restrict to timestamps we actually saw (anti-hallucination)
            if observed and ts_str not in observed:
                # Be lenient: accept exact-character matches from observed;
                # others are neutral (neither +1 nor -1) rather than penalised,
                # because the model might legitimately reformat a timestamp.
                continue
            ts = _try_parse_iso(ts_str)
            if ts is None:
                continue
            # Normalise tz: treat naive as UTC to align with the "now" anchor
            if ts.tzinfo is None and now.tzinfo is not None:
                ts = ts.replace(tzinfo=now.tzinfo)
            if ts > now + timedelta(minutes=5):
                bad += 1
                continue
            if window is not None and ts < now - window:
                bad += 1
                continue
            in_window += 1

        if bad > 0:
            rewards.append(-1.0)
        elif in_window > 0:
            rewards.append(1.0)
        else:
            # Cited timestamps, none matched the observed set or parsed
            rewards.append(-0.5)

    return rewards


# ---------------------------------------------------------- RCD (secondary)


def risk_calibrated_deferral_reward(completions: list[list[dict]], **kwargs) -> list[float]:
    """Reward correct deferral on conditional-order tasks.

    Applies only to task types in ``CONDITIONAL_TASKS`` (5, 9, 10). On those
    tasks, the refsol expects either an action (POST) or no action depending
    on the patient's labs. Agents that "always POST to look useful" are a
    well-known failure mode in agentic medical systems.

    Scoring:

      * If the rollout correctly matches the refsol-expected POST/no-POST
        pattern (see environment's ``_finished`` + ``_post_count``), reward
        is +1.0 regardless of whether the POST payload is correct (the
        ``correctness_reward`` handles payload-level grading).
      * If the rollout's POST/no-POST contradicts what refsol expects,
        reward is -1.0.
      * For non-conditional tasks, returns 0.0.

    Because this reward can only be fully computed against refsol's ground
    truth (which requires FHIR labs to be read), it is gated on the
    environment having the tool log we need. We infer "expected to POST"
    using a simple decision-theoretic check: if any observed magnesium /
    potassium / HbA1c value inside the window is below threshold, POSTing
    is expected.

    Weight in the trainer: 0.5.
    """
    environments = kwargs.get("environments", [])
    task_ids = kwargs.get("task_id", []) or []
    rewards: list[float] = []

    for i, completion in enumerate(completions):
        env = environments[i] if i < len(environments) else None
        task_id = _env_task_id(env, task_ids[i] if i < len(task_ids) else "")
        task_type = _parse_task_type(task_id)
        if task_type not in CONDITIONAL_TASKS:
            rewards.append(0.0)
            continue

        # Did the rollout POST?
        did_post = False
        if env is not None:
            did_post = getattr(env, "_post_count", 0) > 0
        else:
            for turn in completion:
                for call in turn.get("tool_calls") or []:
                    if call.get("function", {}).get("name") == "post_fhir_resource":
                        did_post = True
                        break

        # Was POSTing warranted by what the env saw?
        should_post = _should_have_posted(env, task_type)
        if should_post is None:
            # Not enough evidence to judge; neutral reward
            rewards.append(0.0)
            continue

        if did_post == should_post:
            rewards.append(1.0)
        else:
            rewards.append(-1.0)

    return rewards


def _should_have_posted(env: Any, task_type: int) -> bool | None:
    """Infer whether the task's ground-truth policy required a POST.

    Returns ``True`` (POST expected), ``False`` (no POST expected), or
    ``None`` (not enough evidence observed to judge).
    """
    if env is None:
        return None
    tool_log = _env_tool_log(env)

    now = _try_parse_iso(TASK_NOW_ISO)
    if now is None:
        return None

    # Extract observed {effectiveDateTime: value} pairs from GET responses
    # by reading payload text (cheap regex rather than FHIR parsing).
    values: list[tuple[datetime, float]] = []
    for entry in tool_log:
        if entry.get("action") != "GET" or not entry.get("success"):
            continue
        # The env's tool log doesn't keep the raw body; we rely on the
        # timestamps list + a separate value pattern. For the MG/K/A1C
        # thresholds used here, we parse the response only if attached.
        body = entry.get("response") or entry.get("payload") or ""
        if not body:
            continue
        pairs = _extract_value_time_pairs(body)
        values.extend(pairs)

    if not values:
        return None

    if task_type == 5:  # magnesium <1.9 expects POST
        window = now - timedelta(hours=24)
        in_win = [v for (t, v) in values if t >= window and t <= now]
        if not in_win:
            return False
        return min(in_win) < 1.9
    if task_type == 9:  # potassium <3.5 expects POST
        in_win = [v for (t, v) in values if t <= now]
        if not in_win:
            return False
        return min(in_win) < 3.5
    if task_type == 10:  # A1C older than 1 year expects POST
        cutoff = now - timedelta(days=365)
        if not values:
            return False
        latest_ts = max(t for (t, _v) in values)
        return latest_ts < cutoff
    return None


_VT_PAIR_RE = re.compile(
    r'"effectiveDateTime"\s*:\s*"([^"]+)"[^\{]*?'
    r'"valueQuantity"\s*:\s*\{\s*"value"\s*:\s*([0-9.eE+-]+)',
    re.DOTALL,
)


def _extract_value_time_pairs(body: str) -> list[tuple[datetime, float]]:
    """Best-effort extraction of (effectiveDateTime, value) pairs from a
    FHIR Observation bundle serialized to JSON."""
    pairs: list[tuple[datetime, float]] = []
    for match in _VT_PAIR_RE.finditer(body):
        ts = _try_parse_iso(match.group(1))
        try:
            v = float(match.group(2))
        except ValueError:
            continue
        if ts is not None:
            pairs.append((ts, v))
    return pairs


# ---------------------------------------------------------- DDB (shaping)


def decision_density_reward(completions: list[list[dict]], **kwargs) -> list[float]:
    """Reward concise, structured final answers - only above a correctness floor.

    To avoid rewarding incorrect-but-short answers, this reward is gated on
    having a parseable ``finish(...)`` answer. The trainer applies this on
    top of ``correctness_reward``; the weight (0.2) means it is smaller than
    correctness (1.0) and cannot flip a wrong answer into a positive signal.

    Scoring (only when a FINISH answer is present):

      * +0.3 if the answer parses as JSON and has length <128 tokens
      * +0.1 if the answer parses as JSON and has length 128..511 tokens
      * -0.1 if the answer length >=512 tokens (verbose "answer" is a signal
        the model stuffed reasoning into the final payload)
      *  0.0 otherwise

    Weight in the trainer: 0.2.
    """
    rewards: list[float] = []
    for completion in completions:
        ans = _extract_finish_answer(completion)
        if ans is None:
            rewards.append(0.0)
            continue

        n_tokens = max(1, len(ans.split()))

        # Does it parse as JSON (the refsol graders require this)?
        parseable = True
        try:
            json.loads(ans)
        except Exception:
            parseable = False

        if not parseable:
            # Penalise non-JSON "answers" lightly - the correctness_reward
            # already catches this, but small extra signal helps the
            # trainer converge to structured output faster.
            rewards.append(-0.1)
            continue

        if n_tokens < 128:
            rewards.append(0.3)
        elif n_tokens < 512:
            rewards.append(0.1)
        else:
            rewards.append(-0.1)

    return rewards


# ----------------------------------------------------------------- utilities


def register_rewards(
    enabled: dict[str, bool] | None = None,
    *,
    benchmark_aligned: bool = False,
):
    """Return the list of reward functions to pass to ``GRPOTrainer.reward_funcs``.

    Call sites in ``train_grpo_32b.py`` use this to keep reward composition
    config-driven without a big if/else.

    When ``benchmark_aligned`` is true, only the refsol-aligned reward is
    registered so proxy TRL signals cannot dominate the update.
    """
    enabled = enabled or {}
    if benchmark_aligned:
        from rl_training.rl.trl_benchmark_reward import benchmark_aligned_reward

        return [benchmark_aligned_reward]

    from rl_training.env.trl_rewards import (
        correctness_reward,
        efficiency_reward,
        tool_usage_reward,
    )
    funcs = [correctness_reward, efficiency_reward, tool_usage_reward]
    if enabled.get("temporal_grounding_enabled", False):
        funcs.append(temporal_grounding_reward)
    if enabled.get("risk_calibrated_deferral_enabled", False):
        funcs.append(risk_calibrated_deferral_reward)
    if enabled.get("decision_density_enabled", False):
        funcs.append(decision_density_reward)
    return funcs
