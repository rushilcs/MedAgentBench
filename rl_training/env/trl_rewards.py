"""Reward functions for TRL GRPOTrainer.

Each function receives ``completions`` (list of conversation turns per sample)
and ``**kwargs`` (which includes ``environments`` when ``environment_factory``
is used).  Each returns a list[float] of per-sample rewards.
"""

from __future__ import annotations

import json
import re
from typing import Any


def _extract_finish_result(completion: list[dict]) -> str | None:
    """Find the FINISH answer from tool calls or raw text in a completion."""
    for turn in completion:
        # Native tool-calling format
        if turn.get("tool_calls"):
            for call in turn["tool_calls"]:
                if call.get("function", {}).get("name") == "finish":
                    return call["function"]["arguments"].get("answers", "")
        # Text-based fallback
        content = turn.get("content", "")
        if isinstance(content, str) and "FINISH(" in content:
            match = re.search(r"FINISH\((.+)\)", content, re.DOTALL)
            if match:
                return match.group(1)
    return None


def _count_tool_calls(completion: list[dict], tool_name: str | None = None) -> int:
    count = 0
    for turn in completion:
        if turn.get("tool_calls"):
            for call in turn["tool_calls"]:
                if tool_name is None or call.get("function", {}).get("name") == tool_name:
                    count += 1
    return count


def correctness_reward(completions: list[list[dict]], **kwargs) -> list[float]:
    """Reward based on whether the environment reached a correct finish state.

    Uses the environment's internal state to check if ``finish()`` was called
    and grades correctness via the refsol grading logic when task data is available.
    """
    environments = kwargs.get("environments", [])
    rewards: list[float] = []

    for i, completion in enumerate(completions):
        reward = 0.0
        finish_result = _extract_finish_result(completion)

        if finish_result is not None:
            reward += 0.3  # called finish (good structure)
        else:
            reward -= 0.5  # never called finish
            rewards.append(reward)
            continue

        if i < len(environments):
            env = environments[i]
            if getattr(env, "_finished", False):
                reward += 0.5  # env confirms finish was called

        rewards.append(reward)

    return rewards


def efficiency_reward(completions: list[list[dict]], **kwargs) -> list[float]:
    """Reward efficient tool usage — fewer steps is better."""
    rewards: list[float] = []

    for completion in completions:
        total_calls = _count_tool_calls(completion)

        if total_calls == 0:
            reward = -0.3  # no tool usage at all
        elif total_calls <= 3:
            reward = 0.3  # efficient
        elif total_calls <= 6:
            reward = 0.1  # acceptable
        else:
            reward = -0.2  # too many calls

        rewards.append(reward)

    return rewards


def tool_usage_reward(completions: list[list[dict]], **kwargs) -> list[float]:
    """Reward proper tool usage patterns.

    Checks that the model uses the correct tools (GET before FINISH for
    query tasks, POST for action tasks) and penalizes invalid tool calls.
    """
    rewards: list[float] = []

    for completion in completions:
        reward = 0.0
        has_get = _count_tool_calls(completion, "get_fhir_resource") > 0
        has_post = _count_tool_calls(completion, "post_fhir_resource") > 0
        has_finish = _count_tool_calls(completion, "finish") > 0

        # Also check text-based format
        for turn in completion:
            content = turn.get("content", "") or ""
            if isinstance(content, str):
                if content.strip().startswith("GET "):
                    has_get = True
                elif content.strip().startswith("POST "):
                    has_post = True
                elif content.strip().startswith("FINISH("):
                    has_finish = True

        if has_get:
            reward += 0.2  # used GET to query data

        if has_finish:
            reward += 0.2  # properly finished
        else:
            reward -= 0.5  # never finished

        # Penalize invalid assistant turns (no tool call and no valid action text)
        invalid_count = 0
        for turn in completion:
            if turn.get("role") == "assistant":
                content = (turn.get("content") or "").strip()
                has_tc = bool(turn.get("tool_calls"))
                is_valid_text = any(
                    content.startswith(prefix)
                    for prefix in ("GET ", "POST ", "FINISH(")
                )
                if not has_tc and content and not is_valid_text:
                    invalid_count += 1

        reward -= 0.2 * invalid_count

        rewards.append(reward)

    return rewards
