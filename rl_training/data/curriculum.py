"""Curriculum task samplers for GRPO.

Two surfaces:

1. ``apply_soft_curriculum_mix`` — the v1 query/action soft mix used by
   the original GRPO config.

2. ``two_phase_materialise`` — the v2 weakness-weighted two-phase sampler.
   Produces a deterministic length-N list whose first ``phase_a_size``
   items are drawn from Phase A (weakness-weighted from v1 training-time
   per-family signal) and the remaining items from Phase B (hard-mode
   focus on task9/10/2). The trainer pairs this with a ``SequentialSampler``
   monkey-patch so consecutive items map to consecutive training steps.

Per plan §0.4 the Phase A signal must come from v1's training-time
per-family mean reward (rollouts.jsonl). v1 eval.json is a logged
fallback only; uniform is the final fallback.
"""

from __future__ import annotations

import json
import logging
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rl_training.rl.verifiers.task_masks import is_action_family, task_type_from_id

logger = logging.getLogger(__name__)


def apply_soft_curriculum_mix(
    tasks: list[dict[str, Any]],
    query_target_fraction: float = 0.7,
    *,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Resample tasks with replacement to target ``query_target_fraction`` query tasks."""
    if not tasks:
        return tasks
    q = [t for t in tasks if not is_action_family(t.get("id", ""))]
    a = [t for t in tasks if is_action_family(t.get("id", ""))]
    if not q or not a:
        return list(tasks)
    rng = random.Random(seed)
    n = len(tasks)
    out: list[dict[str, Any]] = []
    for _ in range(n):
        if rng.random() < query_target_fraction:
            out.append(rng.choice(q))
        else:
            out.append(rng.choice(a))
    return out


# ---------------------------------------------------------------------------
# Two-phase materialiser


def _per_family_weakness_from_rollouts(
    rollouts_path: str | Path,
) -> dict[str, float] | None:
    """Mean refsol-pass rate per task family from a v1 rollouts.jsonl."""
    p = Path(rollouts_path)
    if not p.exists():
        return None
    counts: dict[str, list[int]] = {}
    try:
        with p.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                tid = rec.get("task_id") or rec.get("id") or ""
                tt = task_type_from_id(tid)
                if tt is None:
                    continue
                fam = f"task{tt}"
                trace = rec.get("trace", {}) or {}
                passed = bool(
                    trace.get("refsol_pass", rec.get("refsol_pass", False)),
                )
                counts.setdefault(fam, []).append(1 if passed else 0)
    except OSError:
        return None
    if not counts:
        return None
    return {k: (sum(v) / len(v) if v else 0.0) for k, v in counts.items()}


def _per_family_sr_from_eval_json(
    eval_path: str | Path,
) -> dict[str, float] | None:
    """Per-task SR (0..1) from a v1 eval.json. Logged-fallback only."""
    p = Path(eval_path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    out: dict[str, float] = {}
    per_task = data.get("per_task_success_rate") or data.get("per_task_sr") or {}
    if isinstance(per_task, dict):
        for k, v in per_task.items():
            try:
                val = float(v)
            except (TypeError, ValueError):
                continue
            if val > 1.0:
                val = val / 100.0
            out[str(k)] = val
    return out or None


def _normalise(w: dict[str, float]) -> dict[str, float]:
    s = sum(max(0.0, v) for v in w.values())
    if s <= 0:
        return {k: 1.0 / max(1, len(w)) for k in w}
    return {k: max(0.0, v) / s for k, v in w.items()}


def _phase_a_weights_from_signal(
    signal: dict[str, float] | None,
    families_present: Sequence[str],
) -> dict[str, float]:
    """``(1 - score) ** 1.5`` weighting per plan §0.4."""
    if not signal:
        return {f: 1.0 / len(families_present) for f in families_present}
    raw: dict[str, float] = {}
    for fam in families_present:
        score = float(signal.get(fam, 0.5))
        score = max(0.0, min(1.0, score))
        raw[fam] = (1.0 - score) ** 1.5
    return _normalise(raw)


def _bucket_by_family(
    tasks: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for t in tasks:
        tt = task_type_from_id(t.get("id", ""))
        if tt is None:
            continue
        out.setdefault(f"task{tt}", []).append(t)
    return out


def _draw_n(
    weights: dict[str, float],
    by_family: dict[str, list[dict[str, Any]]],
    n: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    active = [
        (f, w) for f, w in weights.items() if w > 0 and by_family.get(f)
    ]
    if not active:
        # Uniform fallback over whatever families exist.
        active = [(f, 1.0) for f in by_family if by_family[f]]
    fams, ws = zip(*active)
    out: list[dict[str, Any]] = []
    for _ in range(n):
        fam = rng.choices(fams, weights=ws, k=1)[0]
        out.append(rng.choice(by_family[fam]))
    return out


def two_phase_materialise(
    tasks: list[dict[str, Any]],
    *,
    total_prompts: int,
    phase_a_prompts: int,
    phase_a_weights: dict[str, float] | None = None,
    phase_b_weights: dict[str, float] | None = None,
    v1_rollouts_path: str | None = None,
    v1_eval_fallback_path: str | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Return a deterministic length-``total_prompts`` task list.

    First ``phase_a_prompts`` items are drawn under Phase A weights; the
    remainder under Phase B weights. Pair with a SequentialSampler in the
    trainer so step-N draws prompts [N*B : (N+1)*B].
    """
    by_family = _bucket_by_family(tasks)
    if not by_family:
        return list(tasks)
    families = sorted(by_family.keys(), key=lambda s: int(s[4:]))
    rng = random.Random(seed)

    if phase_a_weights:
        phase_a = _normalise({
            k: float(v) for k, v in phase_a_weights.items() if k in by_family
        })
        logger.info("Curriculum Phase A weights (config-supplied): %s", phase_a)
    else:
        signal = None
        source = "uniform"
        if v1_rollouts_path:
            signal = _per_family_weakness_from_rollouts(v1_rollouts_path)
            if signal is not None:
                source = f"rollouts:{v1_rollouts_path}"
        if signal is None:
            # DEFENSIBILITY: we deliberately do NOT fall back to v1 test eval
            # here. The plan §0.4 commits to a training-only curriculum signal
            # (rollouts.jsonl) or no signal at all (uniform). Phase B carries
            # the hard-task focus via the explicit phase_b_weights, which is
            # fully disclosed researcher prior — not test-set leakage.
            if v1_eval_fallback_path:
                logger.warning(
                    "Curriculum Phase A: no training-time signal found at %s "
                    "and v1_eval_fallback_path=%s is intentionally NOT used "
                    "(would leak test SR). Falling back to UNIFORM Phase A.",
                    v1_rollouts_path, v1_eval_fallback_path,
                )
            else:
                logger.info(
                    "Curriculum Phase A: no rollouts.jsonl at %s; uniform.",
                    v1_rollouts_path,
                )
        phase_a = _phase_a_weights_from_signal(signal, families)
        logger.info("Curriculum Phase A weights (source=%s): %s", source, phase_a)

    if phase_b_weights:
        phase_b = _normalise({
            k: float(v) for k, v in phase_b_weights.items() if k in by_family
        })
    else:
        default = {
            "task9": 0.35, "task10": 0.30, "task2": 0.15,
            "task4": 0.08, "task5": 0.04, "task6": 0.04, "task7": 0.04,
        }
        phase_b = _normalise({
            k: v for k, v in default.items() if k in by_family
        })
    logger.info("Curriculum Phase B weights: %s", phase_b)

    phase_a_n = max(0, min(int(phase_a_prompts), int(total_prompts)))
    phase_b_n = max(0, int(total_prompts) - phase_a_n)
    out = _draw_n(phase_a, by_family, phase_a_n, rng)
    out.extend(_draw_n(phase_b, by_family, phase_b_n, rng))
    logger.info(
        "Two-phase materialised: %d total (Phase A: %d, Phase B: %d)",
        len(out), phase_a_n, phase_b_n,
    )
    return out
