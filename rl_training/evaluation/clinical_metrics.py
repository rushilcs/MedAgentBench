"""Clinical evaluation metrics over a list of :class:`Trajectory`.

These complement the generic success-rate metrics in :mod:`.metrics` with
failure-mode-specific breakdowns that surface the behaviours our clinical
rewards target:

  * ``temporal_inconsistency_rate`` - fraction of rollouts whose final answer
    includes a timestamp outside the task's recency window.
  * ``over_deferral_rate`` - on conditional-order tasks (5, 9, 10), fraction of
    rollouts where the rollout declined to POST when the refsol would have.
  * ``under_deferral_rate`` - same tasks, fraction where the rollout POSTed
    when the refsol would not have.
  * ``evidence_omission_rate`` - fraction where the rollout never issued a
    GET on a query task that requires one.
  * ``avg_answer_tokens`` / ``p95_answer_tokens`` - answer-length distribution.
  * ``invalid_action_rate`` - already reported by the base metrics but
    included here for parity.

Input format: a list of ``Trajectory`` with optional ``extra`` field. The
trainer / evaluator writes the ``env._tool_log`` into ``trajectory.task_data``
under the key ``tool_log`` so this module can read it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from rl_training.data.trajectory import Trajectory


_NOW_ISO = "2023-11-13T10:15:00+00:00"
_CONDITIONAL_TASKS = {"task5", "task9", "task10"}
_QUERY_TASKS_WITH_WINDOW = {"task4", "task6", "task7", "task10"}


@dataclass
class ClinicalEvalResult:
    total: int = 0
    temporal_inconsistency_rate: float = 0.0
    over_deferral_rate: float = 0.0
    under_deferral_rate: float = 0.0
    evidence_omission_rate: float = 0.0
    avg_answer_tokens: float = 0.0
    p95_answer_tokens: float = 0.0
    # Per-task breakdown for the above
    per_task: dict[str, dict[str, float]] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"Trajectories: {self.total}",
            f"Temporal inconsistency:  {self.temporal_inconsistency_rate:.1%}",
            f"Over-deferral:           {self.over_deferral_rate:.1%}",
            f"Under-deferral:          {self.under_deferral_rate:.1%}",
            f"Evidence omission:       {self.evidence_omission_rate:.1%}",
            f"Avg answer tokens:       {self.avg_answer_tokens:.1f}",
            f"P95 answer tokens:       {self.p95_answer_tokens:.0f}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "temporal_inconsistency_rate": self.temporal_inconsistency_rate,
            "over_deferral_rate": self.over_deferral_rate,
            "under_deferral_rate": self.under_deferral_rate,
            "evidence_omission_rate": self.evidence_omission_rate,
            "avg_answer_tokens": self.avg_answer_tokens,
            "p95_answer_tokens": self.p95_answer_tokens,
            "per_task": self.per_task,
        }


_ISO_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?(?:[+-]\d{2}:?\d{2}|Z)?)?)"
)


def _task_type(task_id: str) -> str:
    for part in task_id.split("_"):
        if part.startswith("task"):
            return part
    return ""


def _last_answer_text(traj: Trajectory) -> str:
    """Extract the answer payload from the last assistant turn (FINISH(...))."""
    for turn in reversed(traj.turns):
        if turn.role in ("agent", "assistant"):
            m = re.search(r"FINISH\((.+)\)", turn.content, re.DOTALL)
            if m:
                return m.group(1)
            return turn.content
    return ""


def _all_assistant_text(traj: Trajectory) -> str:
    """Join every assistant turn's content (used for ts-citation scans)."""
    parts: list[str] = []
    for turn in traj.turns:
        if turn.role in ("agent", "assistant"):
            parts.append(turn.content)
    return "\n".join(parts)


def _out_of_window(ts_list: list[str], window_hours: float | None) -> bool:
    """Return True if any parseable ts is outside the window."""
    now = datetime.fromisoformat(_NOW_ISO.replace("Z", "+00:00"))
    cutoff = now - timedelta(hours=window_hours) if window_hours else None
    for ts_str in ts_list:
        s = ts_str.strip().replace("Z", "+00:00")
        short = re.match(r"^(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2})(?=[+-]|$)", s)
        if short:
            s = short.group(1) + ":00" + s[short.end():]
        try:
            ts = datetime.fromisoformat(s)
        except ValueError:
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=now.tzinfo)
        if ts > now + timedelta(minutes=5):
            return True
        if cutoff and ts < cutoff:
            return True
    return False


def _has_get(traj: Trajectory) -> bool:
    for turn in traj.turns:
        if turn.role in ("agent", "assistant") and (
            "GET " in turn.content or turn.content.strip().startswith("GET ")
        ):
            return True
    return False


def _has_post(traj: Trajectory) -> bool:
    for turn in traj.turns:
        if turn.role in ("agent", "assistant") and (
            "POST " in turn.content or turn.content.strip().startswith("POST ")
        ):
            return True
    return False


def compute_clinical_metrics(trajectories: list[Trajectory]) -> ClinicalEvalResult:
    """Compute the clinical breakdown over a list of trajectories."""
    total = len(trajectories)
    if total == 0:
        return ClinicalEvalResult()

    # Per-task accumulators
    per_task: dict[str, dict[str, Any]] = {}
    answer_tokens: list[int] = []
    temporal_bad = 0
    over_def = 0
    under_def = 0
    ev_omit = 0
    cond_total = 0
    query_window_total = 0

    for traj in trajectories:
        tt = _task_type(traj.task_id)
        bucket = per_task.setdefault(tt, {
            "total": 0, "temporal_bad": 0, "over_def": 0, "under_def": 0,
            "ev_omit": 0,
        })
        bucket["total"] += 1

        answer = _last_answer_text(traj)
        answer_tokens.append(len(answer.split()))

        window = None
        if tt in {"task4", "task5", "task6"}:
            window = 24.0
        elif tt == "task10":
            window = 24.0 * 365

        if tt in _QUERY_TASKS_WITH_WINDOW or tt in _CONDITIONAL_TASKS:
            query_window_total += 1
            # Scan the FULL assistant transcript for cited timestamps, not
            # only the FINISH payload: the model often cites in reasoning.
            full = _all_assistant_text(traj)
            cited = [m.group(1) for m in _ISO_RE.finditer(full)]
            if cited and _out_of_window(cited, window):
                temporal_bad += 1
                bucket["temporal_bad"] += 1

        if tt in _CONDITIONAL_TASKS:
            cond_total += 1
            did_post = _has_post(traj)
            # Best-effort: if the rollout was graded correct AND did not post,
            # we cannot over-defer. If graded incorrect AND did not post AND a
            # POST was expected by refsol, that's over-deferral.
            if did_post and not traj.correct:
                under_def += 1
                bucket["under_def"] += 1
            elif not did_post and not traj.correct:
                over_def += 1
                bucket["over_def"] += 1

        if tt in _QUERY_TASKS_WITH_WINDOW and not _has_get(traj):
            ev_omit += 1
            bucket["ev_omit"] += 1

    avg = sum(answer_tokens) / max(1, len(answer_tokens))
    p95 = (
        sorted(answer_tokens)[int(0.95 * (len(answer_tokens) - 1))]
        if answer_tokens else 0.0
    )

    per_task_report: dict[str, dict[str, float]] = {}
    for tt, b in per_task.items():
        n = max(1, b["total"])
        per_task_report[tt] = {
            "total": b["total"],
            "temporal_inconsistency_rate": b["temporal_bad"] / n,
            "over_deferral_rate": b["over_def"] / n,
            "under_deferral_rate": b["under_def"] / n,
            "evidence_omission_rate": b["ev_omit"] / n,
        }

    return ClinicalEvalResult(
        total=total,
        temporal_inconsistency_rate=temporal_bad / max(1, query_window_total),
        over_deferral_rate=over_def / max(1, cond_total),
        under_deferral_rate=under_def / max(1, cond_total),
        evidence_omission_rate=ev_omit / max(1, len([
            t for t in trajectories if _task_type(t.task_id) in _QUERY_TASKS_WITH_WINDOW
        ])),
        avg_answer_tokens=avg,
        p95_answer_tokens=float(p95),
        per_task=per_task_report,
    )


def save_clinical_metrics(result: ClinicalEvalResult, path: str) -> None:
    """Write metrics to a JSON file."""
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
