"""Stress evaluation harness.

Runs the :class:`rl_training.evaluation.evaluator.Evaluator` on perturbed
copies of the benchmark tasks, producing a per-axis breakdown of how the
policy's success rate degrades under each perturbation.

Typical use:

    from rl_training.evaluation.stress_eval import run_stress_eval
    result = run_stress_eval(
        policy=my_policy,
        env=my_env,
        base_tasks=benchmark_tasks,
        perturbations=["timestamp_shuffle", "distractor_padding"],
        output_dir="rl_training/outputs/stress_eval",
    )

The harness writes:
  * ``<output_dir>/<axis>__results.json`` - full EvalResult per axis
  * ``<output_dir>/summary.csv`` - one row per axis with SR, TCG-bad,
    deferral rates
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rl_training.data.timeline_perturb import PERTURBATIONS, perturb_tasks
from rl_training.evaluation.clinical_metrics import (
    ClinicalEvalResult,
    compute_clinical_metrics,
)
from rl_training.evaluation.evaluator import Evaluator
from rl_training.evaluation.metrics import EvalResult

logger = logging.getLogger(__name__)


@dataclass
class StressAxisResult:
    axis: str
    base_sr: float
    perturbed_sr: float
    delta_sr: float
    clinical: ClinicalEvalResult


def _safe_evaluate(
    evaluator: Evaluator, policy: Any, tasks: list[dict[str, Any]],
) -> tuple[EvalResult, list[Any]]:
    """Run the policy on a given task list and also return trajectories."""
    original = evaluator.benchmark_tasks
    evaluator.benchmark_tasks = tasks
    try:
        # We need the list of trajectories too for clinical metrics.
        trajectories = []
        for task in tasks:
            try:
                traj = evaluator._rollout(policy, task)  # noqa: SLF001
                trajectories.append(traj)
            except Exception as exc:
                logger.warning("Rollout failed for %s: %s", task.get("id"), exc)
        from rl_training.evaluation.metrics import compute_metrics
        result = compute_metrics(trajectories)
    finally:
        evaluator.benchmark_tasks = original
    return result, trajectories


def run_stress_eval(
    policy: Any,
    evaluator: Evaluator,
    base_tasks: list[dict[str, Any]],
    perturbations: list[str] | None = None,
    output_dir: str = "rl_training/outputs/stress_eval",
    seed: int = 0,
) -> dict[str, StressAxisResult]:
    """Evaluate ``policy`` on ``base_tasks`` and on each perturbed copy."""
    names = perturbations or list(PERTURBATIONS.keys())
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Running baseline (no perturbation) on %d tasks", len(base_tasks))
    base_result, base_trajs = _safe_evaluate(evaluator, policy, base_tasks)
    base_clin = compute_clinical_metrics(base_trajs)
    _dump(
        Path(output_dir) / "baseline__results.json",
        base_result, base_clin,
    )

    axes: dict[str, StressAxisResult] = {}
    for name in names:
        logger.info("Perturbation axis: %s", name)
        perturbed = perturb_tasks(base_tasks, name, seed=seed)
        result, trajs = _safe_evaluate(evaluator, policy, perturbed)
        clin = compute_clinical_metrics(trajs)
        _dump(Path(output_dir) / f"{name}__results.json", result, clin)
        axes[name] = StressAxisResult(
            axis=name,
            base_sr=base_result.success_rate,
            perturbed_sr=result.success_rate,
            delta_sr=result.success_rate - base_result.success_rate,
            clinical=clin,
        )

    _write_summary(Path(output_dir) / "summary.csv", base_result, base_clin, axes)
    return axes


def _dump(
    path: Path,
    result: EvalResult,
    clinical: ClinicalEvalResult,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "success_rate": result.success_rate,
        "correct": result.correct,
        "total": result.total,
        "per_task_sr": result.per_task_sr,
        "query_sr": result.query_sr,
        "action_sr": result.action_sr,
        "invalid_action_rate": result.invalid_action_rate,
        "clinical": clinical.to_dict(),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _write_summary(
    path: Path,
    base_result: EvalResult,
    base_clin: ClinicalEvalResult,
    axes: dict[str, StressAxisResult],
) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "axis", "success_rate", "delta_sr_vs_base",
            "temporal_inconsistency", "over_deferral", "under_deferral",
            "evidence_omission", "avg_answer_tokens",
        ])
        writer.writerow([
            "baseline", base_result.success_rate, 0.0,
            base_clin.temporal_inconsistency_rate,
            base_clin.over_deferral_rate, base_clin.under_deferral_rate,
            base_clin.evidence_omission_rate, base_clin.avg_answer_tokens,
        ])
        for name, axis in axes.items():
            c = axis.clinical
            writer.writerow([
                name, axis.perturbed_sr, axis.delta_sr,
                c.temporal_inconsistency_rate, c.over_deferral_rate,
                c.under_deferral_rate, c.evidence_omission_rate,
                c.avg_answer_tokens,
            ])
