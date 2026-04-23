"""Per-task POST body shape checks for partial-credit shaping (plan §0.2).

The MedAgentBench refsol grader is binary: a POST is either fully right or
worth zero. ``post_body_shape_ok`` lets the GRPO reward give a small +0.2
nudge when a rollout's POST has the *right shape* (correct FHIR resource
type + the required top-level fields for the task family) but the *wrong
values*. Currently those rollouts get exactly 0 from refsol despite being
one field away — this helper closes the action_sr gradient gap on tasks
4-7 / 9-10.

Mapping is intentionally conservative: only POST families with a clearly
defined target schema are listed; everything else returns ``False`` so the
shaping term stays inactive and we never double-pay an already-passing
trajectory (the caller suppresses this term when refsol passes).
"""

from __future__ import annotations

from typing import Any

from rl_training.rl.verifiers.task_masks import task_type_from_id

# Per-task expected FHIR resource type. Derived from
# src/server/tasks/medagentbench/refsol.py (task3..task10). Only includes
# tasks whose refsol grader accepts a POST.
_POST_RESOURCE_TYPE: dict[int, str] = {
    3: "Observation",
    5: "MedicationRequest",
    8: "ServiceRequest",
    # task9: refsol expects MedicationRequest *and* Observation (potassium
    # replacement + recheck order). The shape helper is per-POST, so accept
    # either resource type — getting one of the two right is still partial
    # progress on the conditional-order branch.
    9: "MedicationRequest",
    10: "ServiceRequest",
}

# Required top-level fields per resource type. Kept minimal so partial
# credit really means "structurally close", not "all clinical fields right".
_REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "Observation": ("resourceType", "subject", "code"),
    "MedicationRequest": ("resourceType", "subject", "medicationCodeableConcept"),
    "ServiceRequest": ("resourceType", "subject", "code"),
}


def _expected_resource_type(task_id: str) -> str | None:
    tt = task_type_from_id(task_id)
    if tt is None:
        return None
    return _POST_RESOURCE_TYPE.get(tt)


def post_body_shape_ok(task_id: str, body: Any) -> bool:
    """True when ``body`` matches the task family's expected POST shape.

    Returns False whenever any required field is missing, the resource
    type is wrong, or the task family does not accept a POST.
    """
    expected = _expected_resource_type(task_id)
    if expected is None:
        return False
    if not isinstance(body, dict):
        return False
    if str(body.get("resourceType", "")) != expected:
        # Also accept the secondary resource for task9 (Observation) — the
        # refsol grader expects two POSTs there, one of each type.
        tt = task_type_from_id(task_id)
        if tt == 9 and str(body.get("resourceType", "")) == "Observation":
            expected = "Observation"
        else:
            return False
    required = _REQUIRED_FIELDS.get(expected, ())
    return all(f in body for f in required)


def is_conditional_order_family(task_id: str) -> bool:
    """task9 / task10 — the ``no-result -> still emit POST`` branch.

    Used by ``r_conditional_order_branch`` to decide whether to fire.
    """
    tt = task_type_from_id(task_id)
    return tt in (9, 10)


__all__ = [
    "is_conditional_order_family",
    "post_body_shape_ok",
]
