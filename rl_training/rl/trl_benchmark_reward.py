"""TRL ``reward_funcs`` entry: single scalar benchmark-aligned reward."""

from __future__ import annotations

import json
import logging
import os

from rl_training.rl import medagent_reward as mr

logger = logging.getLogger(__name__)


def benchmark_aligned_reward(completions: list[list[dict]], **kwargs) -> list[float]:
    """One completion → one float; uses ``refsol`` terminal pass + env shaping.

    Dual-path:

    * **Plain-text rollout_func path.** When the rollout supplied
      ``rollout_tool_log`` / ``rollout_finish_result`` / ``rollout_correct``
      (one per generation), score directly from those extras. No live env or
      refsol re-invocation needed; ``correct`` is already the refsol pass.
    * **JSON-tool environment_factory path (legacy).** Fall back to the old
      ``score_completions(completions, environments, fhir)`` flow that reads
      ``environments[i]._tool_log`` etc.
    """
    fhir = os.environ.get("FHIR_API_BASE", "http://localhost:8080/fhir/")

    tool_logs = kwargs.get("rollout_tool_log")
    if tool_logs is not None:
        finish_results = kwargs.get("rollout_finish_result") or [None] * len(tool_logs)
        corrects = kwargs.get("rollout_correct") or [False] * len(tool_logs)
        ref_jsons = kwargs.get("rollout_ref_task_json") or ["{}"] * len(tool_logs)
        fhirs = kwargs.get("rollout_fhir_api_base") or [fhir] * len(tool_logs)
        scores: list[float] = []
        for i in range(len(completions)):
            try:
                case_data = json.loads(ref_jsons[i]) if i < len(ref_jsons) else {}
            except (TypeError, json.JSONDecodeError):
                case_data = {}
            tl = tool_logs[i] if i < len(tool_logs) else []
            fr = finish_results[i] if i < len(finish_results) else None
            cv = bool(corrects[i]) if i < len(corrects) else False
            finished = bool(tl and tl[-1].get("action") == "FINISH")
            total, trace = mr.compute_episode_reward_from_extras(
                case_data=case_data,
                tool_log=tl,
                finished=finished,
                finish_result=fr,
                correct=cv,
                fhir_api_base=fhirs[i] if i < len(fhirs) else fhir,
            )
            scores.append(total)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "benchmark_reward[%d]=%.4f correct=%s trace=%s",
                    i, total, cv, trace,
                )
        return scores

    envs = kwargs.get("environments") or []
    scores, traces = mr.score_completions(completions, envs, fhir)
    if logger.isEnabledFor(logging.DEBUG):
        for i, tr in enumerate(traces):
            if i < len(scores):
                logger.debug("benchmark_reward[%d]=%.4f trace=%s", i, scores[i], tr)
    return scores
