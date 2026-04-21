"""Sandboxed Python grader for OpenAI RFT on MedAgentBench.

The OpenAI RFT sandbox executes the ``grade`` function below against each
rollout: ``grade(sample, item) -> float in [0, 1]``. The sandbox has no network
access, so every FHIR-derived reference must come from ``item`` (populated at
dataset-build time by ``rl_training.rft.reference_builder``).

This file deliberately uses only stdlib so it runs unchanged inside the
sandbox. Keep it self-contained — it is read as a string by the RFT launcher
and shipped as ``grader.source``.
"""

from __future__ import annotations

import json
import re
from typing import Any


# ---------------------------------------------------------------------------
# Output parser: mirrors rl_training/env/action_parser.py but allows multiple
# POSTs followed by a FINISH in a single completion.
# ---------------------------------------------------------------------------


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z_]*\n?", "", text)
        if text.endswith("```"):
            text = text[: -3]
    return text.strip()


def _parse_finish(fragment: str) -> tuple[Any, bool]:
    """Parse ``FINISH(<json list>)`` → (parsed_list, ok)."""
    m = re.search(r"FINISH\s*\(", fragment)
    if m is None:
        return None, False
    start = m.end()
    depth = 1
    i = start
    while i < len(fragment) and depth > 0:
        ch = fragment[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                inner = fragment[start:i]
                try:
                    return json.loads(inner), True
                except Exception:
                    return None, False
        i += 1
    return None, False


def _parse_actions(text: str) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Return ``(posts, finish_action_or_None)``."""
    cleaned = _strip_fences(text)
    posts: list[dict[str, Any]] = []
    finish: dict[str, Any] | None = None

    cursor = 0
    n = len(cleaned)
    while cursor < n:
        post_idx = cleaned.find("POST ", cursor)
        finish_idx = cleaned.find("FINISH(", cursor)

        if finish_idx != -1 and (post_idx == -1 or finish_idx < post_idx):
            result, ok = _parse_finish(cleaned[finish_idx:])
            finish = {"kind": "finish", "result": result, "ok": ok}
            break

        if post_idx == -1:
            break

        # Parse "POST <url>\n<json>..."
        line_end = cleaned.find("\n", post_idx)
        if line_end == -1:
            break
        url = cleaned[post_idx + 5: line_end].strip()

        # Body spans from line_end+1 until the next POST or FINISH marker.
        body_start = line_end + 1
        next_post = cleaned.find("\nPOST ", body_start)
        next_finish = cleaned.find("\nFINISH(", body_start)
        candidates = [x for x in (next_post, next_finish) if x != -1]
        body_end = min(candidates) if candidates else n
        body_str = cleaned[body_start:body_end].strip()

        payload: dict[str, Any] | None
        try:
            payload = json.loads(body_str)
        except Exception:
            # Some models emit the JSON inside a code fence even mid-completion.
            try:
                payload = json.loads(_strip_fences(body_str))
            except Exception:
                payload = None
        posts.append({"kind": "post", "url": url, "payload": payload})
        cursor = body_end

    return posts, finish


# ---------------------------------------------------------------------------
# Per-task offline checks (mirror src/server/tasks/medagentbench/refsol.py
# but read pre-computed reference fields instead of calling FHIR).
# ---------------------------------------------------------------------------


def _check_finish_value(finish: dict[str, Any] | None, reference_sol: Any) -> bool:
    if finish is None or not finish.get("ok"):
        return False
    return finish["result"] == reference_sol


def _check_finish_value_or_empty(finish: dict[str, Any] | None, reference_sol: Any) -> bool:
    if finish is None or not finish.get("ok"):
        return False
    return finish["result"] == reference_sol or finish["result"] == []


def _check_finish_float(finish: dict[str, Any] | None, reference_sol: list, tol: float) -> bool:
    if finish is None or not finish.get("ok"):
        return False
    result = finish["result"]
    if not isinstance(result, list) or len(result) != 1:
        return False
    try:
        return abs(float(result[0]) - float(reference_sol[0])) <= tol
    except Exception:
        return False


def _require_no_posts(posts: list[dict[str, Any]]) -> bool:
    return len(posts) == 0


def _check_task3(posts: list[dict], finish: dict | None, params: dict) -> bool:
    if len(posts) != 1:
        return False
    p = posts[0]
    mrn = params["mrn"]
    expected_prefix = "Observation"
    if not p["url"].endswith(expected_prefix):
        return False
    payload = p["payload"]
    if not isinstance(payload, dict):
        return False
    try:
        assert payload["resourceType"] == "Observation"
        assert len(payload["category"]) == 1
        assert len(payload["category"][0]["coding"]) == 1
        assert payload["category"][0]["coding"][0] == {
            "system": "http://hl7.org/fhir/observation-category",
            "code": "vital-signs",
            "display": "Vital Signs",
        }
        assert payload["code"] == {"text": "BP"}
        assert payload["effectiveDateTime"] == "2023-11-13T10:15:00+00:00"
        assert payload["status"] == "final"
        assert payload["valueString"] == "118/77 mmHg"
        assert payload["subject"] == {"reference": f"Patient/{mrn}"}
    except (AssertionError, KeyError, TypeError):
        return False
    return True


def _check_task5_post(payload: dict, mrn: str, last_value: float) -> bool:
    if last_value < 1:
        dose, rate = 4, 4
    elif last_value < 1.5:
        dose, rate = 2, 2
    else:
        dose, rate = 1, 1
    try:
        assert payload["resourceType"] == "MedicationRequest"
        coding = payload["medicationCodeableConcept"]["coding"][0]
        assert coding["system"] == "http://hl7.org/fhir/sid/ndc"
        assert coding["code"] == "0338-1715-40"
        assert "2023-11-13T10:15" in payload["authoredOn"]
        assert payload["dosageInstruction"][0]["route"] == "IV"
        dr = payload["dosageInstruction"][0]["doseAndRate"][0]
        assert dr["doseQuantity"] == {"value": dose, "unit": "g"}
        assert dr["rateQuantity"] == {"value": rate, "unit": "h"}
        assert payload["status"] == "active"
        assert payload["intent"] == "order"
        assert payload["subject"] == {"reference": f"Patient/{mrn}"}
    except (AssertionError, KeyError, TypeError):
        return False
    return True


def _check_task8(posts: list[dict], params: dict) -> bool:
    if len(posts) != 1:
        return False
    p = posts[0]
    if not p["url"].endswith("ServiceRequest"):
        return False
    payload = p["payload"]
    if not isinstance(payload, dict):
        return False
    mrn = params["mrn"]
    comment = (
        "Situation: acute left knee injury, Background: radiology report indicates ACL tear. "
        "Assessment: ACL tear grade II. Recommendation: request for Orthopedic service to evaluate "
        "and provide management recommendations."
    )
    try:
        assert payload["resourceType"] == "ServiceRequest"
        assert payload["code"]["coding"][0]["system"] == "http://snomed.info/sct"
        assert payload["code"]["coding"][0]["code"] == "306181000000106"
        assert payload["authoredOn"] == "2023-11-13T10:15:00+00:00"
        assert payload["status"] == "active"
        assert payload["intent"] == "order"
        assert payload["priority"] == "stat"
        assert comment in payload["note"]["text"]
        assert payload["subject"] == {"reference": f"Patient/{mrn}"}
    except (AssertionError, KeyError, TypeError):
        return False
    return True


def _check_task9_posts(posts: list[dict], params: dict) -> bool:
    if len(posts) != 2:
        return False
    mrn = params["mrn"]
    last_value = params["last_value"]
    dose = round((3.5 - last_value) / 0.1 * 10, 1)
    try:
        med = posts[0]
        assert med["url"].endswith("MedicationRequest")
        mp = med["payload"]
        assert mp["resourceType"] == "MedicationRequest"
        coding = mp["medicationCodeableConcept"]["coding"][0]
        assert coding["system"] == "http://hl7.org/fhir/sid/ndc"
        assert coding["code"] == "40032-917-01"
        assert "2023-11-13T10:15" in mp["authoredOn"]
        assert mp["dosageInstruction"][0]["route"].lower().strip() == "oral"
        dq = mp["dosageInstruction"][0]["doseAndRate"][0]["doseQuantity"]
        assert abs(dq["value"] - dose) <= 0.1
        assert dq["unit"] == "mEq"
        assert mp["status"] == "active"
        assert mp["intent"] == "order"
        assert mp["subject"] == {"reference": f"Patient/{mrn}"}

        lab = posts[1]
        assert lab["url"].endswith("ServiceRequest")
        lp = lab["payload"]
        assert lp["resourceType"] == "ServiceRequest"
        assert lp["code"]["coding"][0]["system"] == "http://loinc.org"
        assert lp["code"]["coding"][0]["code"] == "2823-3"
        assert lp["authoredOn"] == "2023-11-13T10:15:00+00:00"
        assert lp["status"] == "active"
        assert lp["intent"] == "order"
        assert lp["priority"] == "stat"
        assert lp["subject"] == {"reference": f"Patient/{mrn}"}
        assert "2023-11-14T08:" in lp["occurrenceDateTime"]
    except (AssertionError, KeyError, TypeError):
        return False
    return True


def _check_task10_post(posts: list[dict], params: dict) -> bool:
    if len(posts) != 1:
        return False
    p = posts[0]
    if not p["url"].endswith("ServiceRequest"):
        return False
    payload = p["payload"]
    if not isinstance(payload, dict):
        return False
    mrn = params["mrn"]
    try:
        assert payload["resourceType"] == "ServiceRequest"
        assert payload["code"]["coding"][0]["system"] == "http://loinc.org"
        assert payload["code"]["coding"][0]["code"] == "4548-4"
        assert payload["authoredOn"] == "2023-11-13T10:15:00+00:00"
        assert payload["status"] == "active"
        assert payload["intent"] == "order"
        assert payload["priority"] == "stat"
        assert payload["subject"] == {"reference": f"Patient/{mrn}"}
    except (AssertionError, KeyError, TypeError):
        return False
    return True


# ---------------------------------------------------------------------------
# Top-level dispatch.
# ---------------------------------------------------------------------------


_PARTIAL_FINISH_ONLY = 0.4


def _score_task(
    task_type: int,
    posts: list[dict],
    finish: dict | None,
    reference_sol: Any,
    task_params: dict,
    accepts_empty_finish: bool,
) -> float:
    if task_type == 1 or task_type == 2 or task_type == 4 or task_type == 7:
        if not _require_no_posts(posts):
            return 0.0
        return 1.0 if _check_finish_value(finish, reference_sol) else 0.0

    if task_type == 6:
        if not _require_no_posts(posts):
            return 0.0
        return 1.0 if _check_finish_float(finish, reference_sol, task_params.get("tolerance", 0.1)) else 0.0

    if task_type == 3:
        return 1.0 if _check_task3(posts, finish, task_params) else 0.0

    if task_type == 8:
        return 1.0 if _check_task8(posts, task_params) else 0.0

    if task_type == 5:
        last_value = task_params.get("last_value")
        must_post = task_params.get("must_post", False)
        if last_value is None:
            return 1.0 if _require_no_posts(posts) else 0.0
        if not must_post:
            if not _require_no_posts(posts):
                return 0.0
            finish_ok = (
                _check_finish_value_or_empty(finish, reference_sol)
                if accepts_empty_finish else
                _check_finish_value(finish, reference_sol)
            )
            return 1.0 if finish_ok else 0.0
        if len(posts) != 1:
            finish_ok = (
                _check_finish_value_or_empty(finish, reference_sol)
                if accepts_empty_finish else
                _check_finish_value(finish, reference_sol)
            )
            return _PARTIAL_FINISH_ONLY if finish_ok else 0.0
        post_ok = _check_task5_post(posts[0]["payload"] or {}, task_params["mrn"], last_value)
        finish_ok = (
            _check_finish_value_or_empty(finish, reference_sol)
            if accepts_empty_finish else
            _check_finish_value(finish, reference_sol)
        )
        if post_ok and finish_ok:
            return 1.0
        if finish_ok:
            return _PARTIAL_FINISH_ONLY
        return 0.0

    if task_type == 9:
        last_value = task_params.get("last_value")
        must_post = task_params.get("must_post", False)
        if not must_post:
            if not _require_no_posts(posts):
                return 0.0
            finish_ok = _check_finish_value_or_empty(finish, reference_sol)
            return 1.0 if finish_ok else 0.0
        posts_ok = _check_task9_posts(posts, task_params)
        finish_ok = _check_finish_value_or_empty(finish, reference_sol)
        if posts_ok and finish_ok:
            return 1.0
        if finish_ok:
            return _PARTIAL_FINISH_ONLY
        return 0.0

    if task_type == 10:
        needs_order = task_params.get("needs_order", False)
        if not needs_order:
            if not _require_no_posts(posts):
                return 0.0
            finish_ok = _check_finish_value_or_empty(finish, reference_sol)
            return 1.0 if finish_ok else 0.0
        posts_ok = _check_task10_post(posts, task_params)
        finish_ok = _check_finish_value_or_empty(finish, reference_sol)
        if posts_ok and finish_ok:
            return 1.0
        if finish_ok:
            return _PARTIAL_FINISH_ONLY
        return 0.0

    return 0.0


def grade(sample: dict, item: dict) -> float:
    """OpenAI RFT grader entry point.

    ``sample``: the model rollout (uses ``output_text``).
    ``item``:   one JSONL row from ``train.jsonl`` / ``val.jsonl``.

    Returns a float in [0, 1]; 1.0 == passes refsol.task{k}, 0.4 == FINISH-only
    match on POST tasks, 0.0 == fail.
    """
    text = sample.get("output_text")
    if not isinstance(text, str):
        choices = sample.get("choices") or []
        if choices and isinstance(choices, list):
            msg = (choices[0] or {}).get("message") or {}
            text = msg.get("content") or ""
    if not isinstance(text, str) or not text.strip():
        return 0.0

    posts, finish = _parse_actions(text)
    try:
        return float(_score_task(
            task_type=int(item["task_type"]),
            posts=posts,
            finish=finish,
            reference_sol=item.get("reference_sol"),
            task_params=item.get("task_params") or {},
            accepts_empty_finish=bool(item.get("accepts_empty_finish", False)),
        ))
    except Exception:
        return 0.0
