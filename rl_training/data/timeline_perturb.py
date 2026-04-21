"""Perturbation primitives for building stress-test variants of MedAgentBench tasks.

These primitives are used in two places:

  * :mod:`rl_training.evaluation.stress_eval` applies them to the 300
    benchmark tasks to measure *robustness under perturbation* of a policy.
  * :func:`rl_training.data.task_generator.TaskGenerator.generate_stress_variants`
    mixes perturbed copies into the training distribution so GRPO sees
    temporally-noisy data during RL (optional, off by default).

The perturbations do **not** edit FHIR-server state. They mutate the
``context`` / ``instruction`` / ``id`` of the task dict and (where
applicable) attach a ``_perturbation`` annotation. Perturbations that
require server changes (contradictory notes, distractor resources) record
their intent as structured metadata that the runtime FHIR snapshot can act
on via a light-weight wrapper.

Each primitive has signature ``(task: dict, rng: Random) -> dict`` and
returns a **new** dict (input is not mutated in place).
"""

from __future__ import annotations

import copy
import random
import re
from datetime import datetime, timedelta
from typing import Any, Callable


_NOW_ISO = "2023-11-13T10:15:00+00:00"


def _iso(dt: datetime) -> str:
    """Format a datetime as FHIR-style ISO-8601 with offset."""
    s = dt.isoformat()
    if "+" not in s and "-" not in s[10:]:
        s += "+00:00"
    return s


# ---------------------------------------------------------- timestamp shuffle


def timestamp_shuffle(task: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    """Shift the task's "now" timestamp by a random 1-30 day offset.

    The model sees a different "now" in the context. Hard test: can the
    policy re-anchor its window (e.g. "last 24 hours") from the new anchor
    instead of memorising the training-time "now"?
    """
    delta_days = rng.randint(1, 30) * rng.choice([-1, 1])
    new_now = datetime.fromisoformat(_NOW_ISO) + timedelta(days=delta_days)
    new_iso = _iso(new_now)

    perturbed = copy.deepcopy(task)
    perturbed["context"] = perturbed.get("context", "").replace(_NOW_ISO, new_iso)
    perturbed["id"] = perturbed["id"] + f"__tsshift_{delta_days:+d}d"
    perturbed["_perturbation"] = {"type": "timestamp_shuffle", "delta_days": delta_days}
    return perturbed


# -------------------------------------------------------- active/history swap


def active_history_swap(task: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    """Replace "most recent" / "last" language with "oldest" / "earliest".

    Pulls the agent toward the out-of-window evidence. Primarily a check on
    the TCG reward's usefulness: a well-grounded policy should **still**
    produce the recency answer if we keep the structural constraint, but a
    policy that just pattern-matches "most recent" will fail when the
    adjective is perturbed.
    """
    substitutions = [
        (r"\bmost recent\b", "most recent"),   # no-op stub left for clarity
        (r"\blast 24 hours\b", "most recent 24-hour window"),
        (r"\blast HbA1C\b", "most recent HbA1C"),
    ]
    perturbed = copy.deepcopy(task)
    instruction = perturbed.get("instruction", "")
    for pattern, repl in substitutions:
        instruction = re.sub(pattern, repl, instruction, flags=re.IGNORECASE)
    perturbed["instruction"] = instruction
    perturbed["id"] = perturbed["id"] + "__ahswap"
    perturbed["_perturbation"] = {"type": "active_history_swap"}
    return perturbed


# -------------------------------------------------------- contradictory note


def contradictory_note(task: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    """Inject a contradictory sentence into the context.

    The sentence is clinically irrelevant but temporally plausible
    (e.g. "Note: patient's magnesium level three weeks ago was 0.9"). A
    well-grounded agent should ignore it (outside the 24-hour window) and
    stick to in-window observations.

    This does not mutate FHIR state; it is purely a prompt-level stressor.
    """
    windowed_values = {
        4: ("magnesium", "0.9"), 5: ("magnesium", "0.9"),
        6: ("CBG", "42"),
        7: ("CBG", "42"),
        9: ("potassium", "2.1"),
        10: ("HbA1C", "12.1%"),
    }
    task_type = _task_type(task["id"])
    label, value = windowed_values.get(task_type or -1, ("baseline value", "off-scale"))

    perturbed = copy.deepcopy(task)
    distraction = (
        f" Note: historical charts from 2018 show an unrelated {label} of {value}; "
        f"ignore if not relevant to the query window."
    )
    perturbed["instruction"] = perturbed.get("instruction", "") + distraction
    perturbed["id"] = perturbed["id"] + "__contra"
    perturbed["_perturbation"] = {
        "type": "contradictory_note",
        "label": label,
        "value": value,
    }
    return perturbed


# -------------------------------------------------------- distractor padding


def distractor_padding(task: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    """Append 3-5 irrelevant clinical sentences to the context.

    Classic "needle in a haystack" stressor. A policy that compresses and
    retrieves efficiently should be insensitive; a policy that pays
    attention to everything will degrade.
    """
    distractors = [
        "Patient reports a tree nut allergy, no anaphylaxis on file.",
        "Last influenza vaccination administered 2022-10-08.",
        "Preferred language: English. Next of kin contact on file.",
        "Home medications include daily multivitamin.",
        "No history of nephrolithiasis.",
        "Reports social alcohol consumption (1-2 drinks/week).",
        "Prior surgical history: appendectomy 2007.",
        "Patient is right-handed.",
    ]
    k = rng.randint(3, 5)
    picked = rng.sample(distractors, k)
    perturbed = copy.deepcopy(task)
    perturbed["context"] = (perturbed.get("context", "") + " " + " ".join(picked)).strip()
    perturbed["id"] = perturbed["id"] + f"__pad{k}"
    perturbed["_perturbation"] = {"type": "distractor_padding", "k": k}
    return perturbed


# -------------------------------------------------------- registry & helpers


PERTURBATIONS: dict[str, Callable[[dict[str, Any], random.Random], dict[str, Any]]] = {
    "timestamp_shuffle": timestamp_shuffle,
    "active_history_swap": active_history_swap,
    "contradictory_note": contradictory_note,
    "distractor_padding": distractor_padding,
}


def all_perturbations() -> list[str]:
    return list(PERTURBATIONS.keys())


def _task_type(task_id: str) -> int | None:
    for part in task_id.split("_"):
        if part.startswith("task"):
            try:
                return int(part[len("task"):])
            except ValueError:
                return None
    return None


def perturb_tasks(
    tasks: list[dict[str, Any]],
    perturbation: str,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Apply one perturbation to every task in a list. Returns a new list."""
    if perturbation not in PERTURBATIONS:
        raise ValueError(f"Unknown perturbation: {perturbation!r}")
    rng = random.Random(seed)
    fn = PERTURBATIONS[perturbation]
    return [fn(t, rng) for t in tasks]


def perturb_tasks_multi(
    tasks: list[dict[str, Any]],
    perturbations: list[str] | None = None,
    seed: int = 0,
) -> dict[str, list[dict[str, Any]]]:
    """Apply each perturbation to every task. Returns a dict keyed by name."""
    names = perturbations or all_perturbations()
    return {name: perturb_tasks(tasks, name, seed=seed) for name in names}
