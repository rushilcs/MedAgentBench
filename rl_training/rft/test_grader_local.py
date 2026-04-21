#!/usr/bin/env python3
"""Local smoke tests for the RFT Python grader.

Runs the sandboxed ``grade`` function against hand-crafted rollouts that
exercise every task type's success and failure paths. No OpenAI API calls.

Also, when ``OPENAI_API_KEY`` is set and ``--remote`` is passed, it additionally
submits the grader to ``client.fine_tuning.alpha.graders.validate`` (and
``.run`` on a single row) so the final JSON shape is accepted by the RFT API.

Usage:
    python rl_training/rft/test_grader_local.py
    python rl_training/rft/test_grader_local.py --remote  # also hit the API
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from rl_training.rft.medagent_grader import grade


def _sample(text: str) -> dict:
    return {"output_text": text}


# ---------------------------------------------------------------------------
# Test cases: (label, expected_score, sample, item)
# ---------------------------------------------------------------------------


def _cases() -> list[tuple[str, float, dict, dict]]:
    cases: list[tuple[str, float, dict, dict]] = []

    # Task 1: lookup MRN
    item1 = {
        "task_type": 1,
        "reference_sol": ["S1234567"],
        "task_params": {"mrn": "S1234567"},
    }
    cases.append(("task1: correct", 1.0, _sample('FINISH(["S1234567"])'), item1))
    cases.append(("task1: wrong MRN", 0.0, _sample('FINISH(["WRONG"])'), item1))
    cases.append(("task1: stray POST", 0.0,
                  _sample("POST http://fhir/Observation\n{}\nFINISH([\"S1234567\"])"), item1))

    # Task 2: age
    item2 = {"task_type": 2, "reference_sol": [45], "task_params": {"mrn": "X"}}
    cases.append(("task2: correct", 1.0, _sample("FINISH([45])"), item2))
    cases.append(("task2: off-by-one", 0.0, _sample("FINISH([46])"), item2))

    # Task 3: POST vitals
    mrn3 = "PATIENT3"
    vitals_payload = {
        "resourceType": "Observation",
        "category": [{"coding": [{"system": "http://hl7.org/fhir/observation-category", "code": "vital-signs", "display": "Vital Signs"}]}],
        "code": {"text": "BP"},
        "effectiveDateTime": "2023-11-13T10:15:00+00:00",
        "status": "final",
        "valueString": "118/77 mmHg",
        "subject": {"reference": f"Patient/{mrn3}"},
    }
    item3 = {"task_type": 3, "reference_sol": [], "task_params": {"mrn": mrn3}}
    cases.append(("task3: correct POST", 1.0,
                  _sample(f"POST http://fhir/Observation\n{json.dumps(vitals_payload)}\n\nFINISH([])"),
                  item3))
    bad = dict(vitals_payload, valueString="120/80 mmHg")
    cases.append(("task3: wrong BP value", 0.0,
                  _sample(f"POST http://fhir/Observation\n{json.dumps(bad)}\n\nFINISH([])"), item3))
    cases.append(("task3: missing POST", 0.0, _sample("FINISH([])"), item3))

    # Task 4: magnesium lookup
    item4 = {"task_type": 4, "reference_sol": [1.8], "task_params": {"mrn": "X"}}
    cases.append(("task4: correct", 1.0, _sample("FINISH([1.8])"), item4))
    cases.append(("task4: missing", 1.0,
                  _sample("FINISH([-1])"),
                  {"task_type": 4, "reference_sol": [-1], "task_params": {"mrn": "X"}}))
    cases.append(("task4: wrong", 0.0, _sample("FINISH([2.1])"), item4))

    # Task 5: conditional Mg order
    item5_low = {
        "task_type": 5,
        "reference_sol": [1.2],
        "task_params": {"mrn": "MM", "last_value": 1.2, "must_post": True},
        "accepts_empty_finish": True,
    }
    mg_payload = {
        "resourceType": "MedicationRequest",
        "medicationCodeableConcept": {"coding": [{"system": "http://hl7.org/fhir/sid/ndc", "code": "0338-1715-40"}]},
        "authoredOn": "2023-11-13T10:15:00+00:00",
        "dosageInstruction": [{"route": "IV", "doseAndRate": [{"doseQuantity": {"value": 2, "unit": "g"}, "rateQuantity": {"value": 2, "unit": "h"}}]}],
        "status": "active",
        "intent": "order",
        "subject": {"reference": "Patient/MM"},
    }
    cases.append(("task5: low + correct order + finish", 1.0,
                  _sample(f"POST http://fhir/MedicationRequest\n{json.dumps(mg_payload)}\n\nFINISH([1.2])"), item5_low))
    cases.append(("task5: low + correct order + empty finish accepted", 1.0,
                  _sample(f"POST http://fhir/MedicationRequest\n{json.dumps(mg_payload)}\n\nFINISH([])"), item5_low))
    cases.append(("task5: low but no POST → partial", 0.4,
                  _sample("FINISH([1.2])"), item5_low))
    item5_none = {
        "task_type": 5,
        "reference_sol": [-1],
        "task_params": {"mrn": "MM", "last_value": None, "must_post": False},
        "accepts_empty_finish": True,
    }
    cases.append(("task5: no Mg measurement → no-post pass", 1.0,
                  _sample("FINISH([-1])"), item5_none))
    cases.append(("task5: no Mg but stray POST", 0.0,
                  _sample(f"POST http://fhir/MedicationRequest\n{json.dumps(mg_payload)}\n\nFINISH([-1])"), item5_none))

    # Task 6: CBG average with tolerance
    item6 = {"task_type": 6, "reference_sol": [123.4], "task_params": {"tolerance": 0.1}}
    cases.append(("task6: within tolerance", 1.0, _sample("FINISH([123.45])"), item6))
    cases.append(("task6: outside tolerance", 0.0, _sample("FINISH([125.0])"), item6))

    # Task 7: last CBG
    item7 = {"task_type": 7, "reference_sol": [98], "task_params": {"mrn": "Y"}}
    cases.append(("task7: correct", 1.0, _sample("FINISH([98])"), item7))

    # Task 8: ortho referral POST
    mrn8 = "ORTHOPT"
    comment = (
        "Situation: acute left knee injury, Background: radiology report indicates ACL tear. "
        "Assessment: ACL tear grade II. Recommendation: request for Orthopedic service to evaluate "
        "and provide management recommendations."
    )
    sr_payload = {
        "resourceType": "ServiceRequest",
        "code": {"coding": [{"system": "http://snomed.info/sct", "code": "306181000000106"}]},
        "authoredOn": "2023-11-13T10:15:00+00:00",
        "status": "active",
        "intent": "order",
        "priority": "stat",
        "subject": {"reference": f"Patient/{mrn8}"},
        "note": {"text": comment},
    }
    item8 = {"task_type": 8, "reference_sol": [], "task_params": {"mrn": mrn8}}
    cases.append(("task8: correct", 1.0,
                  _sample(f"POST http://fhir/ServiceRequest\n{json.dumps(sr_payload)}\n\nFINISH([])"), item8))

    # Task 9: low K → 2 POSTs
    mrn9 = "K9"
    med_payload = {
        "resourceType": "MedicationRequest",
        "medicationCodeableConcept": {"coding": [{"system": "http://hl7.org/fhir/sid/ndc", "code": "40032-917-01"}]},
        "authoredOn": "2023-11-13T10:15:00+00:00",
        "dosageInstruction": [{"route": "oral", "doseAndRate": [{"doseQuantity": {"value": 20, "unit": "mEq"}}]}],
        "status": "active",
        "intent": "order",
        "subject": {"reference": f"Patient/{mrn9}"},
    }
    lab_payload = {
        "resourceType": "ServiceRequest",
        "code": {"coding": [{"system": "http://loinc.org", "code": "2823-3"}]},
        "authoredOn": "2023-11-13T10:15:00+00:00",
        "status": "active",
        "intent": "order",
        "priority": "stat",
        "subject": {"reference": f"Patient/{mrn9}"},
        "occurrenceDateTime": "2023-11-14T08:00:00+00:00",
    }
    item9_low = {
        "task_type": 9,
        "reference_sol": [3.3],
        "task_params": {"mrn": mrn9, "last_value": 3.3, "must_post": True},
        "accepts_empty_finish": True,
    }
    cases.append(("task9: correct low-K flow", 1.0,
                  _sample(
                      f"POST http://fhir/MedicationRequest\n{json.dumps(med_payload)}\n\n"
                      f"POST http://fhir/ServiceRequest\n{json.dumps(lab_payload)}\n\n"
                      f"FINISH([3.3])"
                  ), item9_low))

    # Task 10: old A1C → needs order
    mrn10 = "A10"
    a1c_payload = {
        "resourceType": "ServiceRequest",
        "code": {"coding": [{"system": "http://loinc.org", "code": "4548-4"}]},
        "authoredOn": "2023-11-13T10:15:00+00:00",
        "status": "active",
        "intent": "order",
        "priority": "stat",
        "subject": {"reference": f"Patient/{mrn10}"},
    }
    item10 = {
        "task_type": 10,
        "reference_sol": [7.2, "2021-01-01T00:00:00+00:00"],
        "task_params": {
            "mrn": mrn10,
            "last_value": 7.2,
            "last_time": "2021-01-01T00:00:00+00:00",
            "needs_order": True,
        },
        "accepts_empty_finish": True,
    }
    cases.append(("task10: needs order + correct finish", 1.0,
                  _sample(
                      f"POST http://fhir/ServiceRequest\n{json.dumps(a1c_payload)}\n\n"
                      f'FINISH([7.2, "2021-01-01T00:00:00+00:00"])'
                  ), item10))

    # Malformed output
    cases.append(("garbage input", 0.0, _sample("sorry, I don't know"), item1))
    cases.append(("empty output", 0.0, _sample(""), item1))

    return cases


def _run_local() -> int:
    failed = 0
    for label, expected, sample, item in _cases():
        got = grade(sample, item)
        ok = abs(got - expected) < 1e-9
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {label}: expected={expected} got={got}")
        if not ok:
            failed += 1
    print(f"\n{len(_cases()) - failed}/{len(_cases())} cases passed.")
    return 1 if failed else 0


def _run_remote() -> int:
    try:
        from openai import OpenAI
    except ImportError:
        print("openai SDK not installed; skipping --remote")
        return 2
    import os
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set; skipping --remote")
        return 2

    from rl_training.training.openai_rft import build_python_grader

    grader = build_python_grader(_ROOT + "/rl_training/rft/medagent_grader.py")
    client = OpenAI()
    try:
        client.fine_tuning.alpha.graders.validate(grader=grader)
        print("[PASS] remote grader validate")
    except Exception as exc:
        print(f"[FAIL] remote grader validate: {exc}")
        return 1
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote", action="store_true",
                        help="Additionally hit client.fine_tuning.alpha.graders.validate")
    args = parser.parse_args()

    rc = _run_local()
    if args.remote:
        rc = max(rc, _run_remote())
    sys.exit(rc)


if __name__ == "__main__":
    main()
