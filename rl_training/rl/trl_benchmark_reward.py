"""TRL ``reward_funcs`` entry: single scalar benchmark-aligned reward."""

from __future__ import annotations

import logging
import os

from rl_training.rl import medagent_reward as mr

logger = logging.getLogger(__name__)


def benchmark_aligned_reward(completions: list[list[dict]], **kwargs) -> list[float]:
    """One completion → one float; uses ``refsol`` terminal pass + env shaping."""
    envs = kwargs.get("environments") or []
    fhir = os.environ.get("FHIR_API_BASE", "http://localhost:8080/fhir/")
    scores, traces = mr.score_completions(completions, envs, fhir)
    if logger.isEnabledFor(logging.DEBUG):
        for i, tr in enumerate(traces):
            if i < len(scores):
                logger.debug("benchmark_reward[%d]=%.4f trace=%s", i, scores[i], tr)
    return scores
