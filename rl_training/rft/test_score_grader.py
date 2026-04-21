#!/usr/bin/env python3
"""Smoke test the score_model RFT judge against real val.jsonl rows.

For each task type present in the validation set we build:
  * a CORRECT sample (expected reward = 1.0)
  * a WRONG sample (expected reward = 0.0)
  * for POST tasks, a FINISH-ONLY sample (expected reward = 0.4)
  * an EMPTY sample (expected reward = 0.0)

We hit the OpenAI ``fine_tuning.alpha.graders.run`` endpoint for each, report
the actual reward and flag any deviations from the expected value (``ok``
allows ±0.05 slack to absorb judge noise).

Usage:
    python rl_training/rft/test_score_grader.py \\
        --val rl_training/outputs/o4_mini_rft_run/val.jsonl \\
        --judge-model gpt-4o-mini
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from openai import OpenAI

from rl_training.training.openai_rft_score_grader import build_score_model_grader


CUTOFF = "2023-11-13T10:15:00+00:00"


def _post_block(url: str, payload: dict) -> str:
    return f"POST {url}\n{json.dumps(payload)}"


def _correct_sample(item: dict) -> str:
    """Construct a canonically-correct rollout for the row's task_type."""
    t = int(item["task_type"])
    ref = item.get("reference_sol")
    params = item.get("task_params") or {}
    mrn = params.get("mrn")

    finish = f"FINISH({json.dumps(ref)})"

    if t in (1, 2, 4, 6, 7):
        return finish

    if t == 3:
        body = {
            "resourceType": "Observation",
            "category": [{"coding": [{
                "system": "http://hl7.org/fhir/observation-category",
                "code": "vital-signs",
                "display": "Vital Signs",
            }]}],
            "code": {"text": "BP"},
            "effectiveDateTime": CUTOFF,
            "status": "final",
            "valueString": "118/77 mmHg",
            "subject": {"reference": f"Patient/{mrn}"},
        }
        return _post_block("http://localhost:8080/fhir/Observation", body) + "\n" + finish

    if t == 5:
        if not params.get("must_post"):
            return finish
        last = float(params["last_value"])
        if last < 1.0:
            dose, rate = 4, 4
        elif last < 1.5:
            dose, rate = 2, 2
        else:
            dose, rate = 1, 1
        body = {
            "resourceType": "MedicationRequest",
            "medicationCodeableConcept": {"coding": [{
                "system": "http://hl7.org/fhir/sid/ndc",
                "code": "0338-1715-40",
            }]},
            "authoredOn": CUTOFF,
            "dosageInstruction": [{
                "route": "IV",
                "doseAndRate": [{
                    "doseQuantity": {"value": dose, "unit": "g"},
                    "rateQuantity": {"value": rate, "unit": "h"},
                }],
            }],
            "status": "active",
            "intent": "order",
            "subject": {"reference": f"Patient/{mrn}"},
        }
        return _post_block("http://localhost:8080/fhir/MedicationRequest", body) + "\n" + finish

    if t == 8:
        body = {
            "resourceType": "ServiceRequest",
            "code": {"coding": [{
                "system": "http://snomed.info/sct",
                "code": "306181000000106",
            }]},
            "authoredOn": CUTOFF,
            "status": "active",
            "intent": "order",
            "priority": "stat",
            "subject": {"reference": f"Patient/{mrn}"},
            "note": {"text": (
                "Situation: acute left knee injury, Background: radiology report indicates ACL tear. "
                "Assessment: ACL tear grade II. Recommendation: request for Orthopedic service to evaluate "
                "and provide management recommendations."
            )},
        }
        return _post_block("http://localhost:8080/fhir/ServiceRequest", body) + "\n" + finish

    if t == 9:
        if not params.get("must_post"):
            return finish
        last = float(params["last_value"])
        dose = round((3.5 - last) / 0.1 * 10, 1)
        med_body = {
            "resourceType": "MedicationRequest",
            "medicationCodeableConcept": {"coding": [{
                "system": "http://hl7.org/fhir/sid/ndc",
                "code": "40032-917-01",
            }]},
            "authoredOn": CUTOFF,
            "dosageInstruction": [{
                "route": "oral",
                "doseAndRate": [{"doseQuantity": {"value": dose, "unit": "mEq"}}],
            }],
            "status": "active",
            "intent": "order",
            "subject": {"reference": f"Patient/{mrn}"},
        }
        lab_body = {
            "resourceType": "ServiceRequest",
            "code": {"coding": [{"system": "http://loinc.org", "code": "2823-3"}]},
            "authoredOn": CUTOFF,
            "status": "active",
            "intent": "order",
            "priority": "stat",
            "subject": {"reference": f"Patient/{mrn}"},
            "occurrenceDateTime": "2023-11-14T08:00:00+00:00",
        }
        return (
            _post_block("http://localhost:8080/fhir/MedicationRequest", med_body)
            + "\n"
            + _post_block("http://localhost:8080/fhir/ServiceRequest", lab_body)
            + "\n"
            + finish
        )

    if t == 10:
        if not params.get("needs_order"):
            return finish
        body = {
            "resourceType": "ServiceRequest",
            "code": {"coding": [{"system": "http://loinc.org", "code": "4548-4"}]},
            "authoredOn": CUTOFF,
            "status": "active",
            "intent": "order",
            "priority": "stat",
            "subject": {"reference": f"Patient/{mrn}"},
        }
        return _post_block("http://localhost:8080/fhir/ServiceRequest", body) + "\n" + finish

    return finish


def _wrong_sample(item: dict) -> str:
    """Force-wrong: emit FINISH(["WRONG_ANSWER"]) with no POSTs."""
    return 'FINISH(["WRONG_ANSWER_42"])'


def _finish_only_sample(item: dict) -> str:
    """For POST tasks: emit only the correct FINISH (should yield 0.4)."""
    return f"FINISH({json.dumps(item.get('reference_sol'))})"


def _empty_sample(item: dict) -> str:
    return ""


def _is_partial_credit_task(item: dict) -> bool:
    """Tasks where FINISH-correct + POST-wrong yields 0.4 (not 0.0).

    Per refsol.py: only tasks 5, 9, 10 give partial credit. Tasks 3 and 8
    require POST + FINISH (no partial credit).
    """
    t = int(item["task_type"])
    p = item.get("task_params") or {}
    if t == 5 or t == 9:
        return bool(p.get("must_post"))
    if t == 10:
        return bool(p.get("needs_order"))
    return False


def _run_one(client: OpenAI, grader: dict, sample_text: str, item: dict) -> dict:
    item_only = {k: v for k, v in item.items() if k != "messages"}
    t0 = time.time()
    resp = client.fine_tuning.alpha.graders.run(
        grader=grader,
        model_sample=sample_text,
        item=item_only,
    )
    elapsed = time.time() - t0
    md = getattr(resp, "metadata", None)
    errors = getattr(md, "errors", None) if md else None
    err_summary: list[str] = []
    if errors is not None:
        for flag in (
            "python_grader_server_error",
            "python_grader_runtime_error",
            "model_grader_server_error",
            "model_grader_refusal_error",
            "model_grader_parse_error",
            "model_grader_exceeded_max_tokens_error",
            "other_error",
        ):
            if bool(getattr(errors, flag, False)):
                err_summary.append(flag)
    return {
        "reward": getattr(resp, "reward", None),
        "errors": err_summary,
        "elapsed": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val", default="rl_training/outputs/o4_mini_rft_run/val.jsonl")
    parser.add_argument("--judge-model", default="gpt-4.1-mini-2025-04-14")
    parser.add_argument("--max-per-type", type=int, default=2,
                        help="how many distinct val rows to test per task_type")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set")

    rows: list[dict] = []
    with open(args.val) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    by_type: dict[int, list[dict]] = {}
    for r in rows:
        by_type.setdefault(int(r["task_type"]), []).append(r)
    print(f"Loaded {len(rows)} val rows; task types present: {sorted(by_type)}")

    client = OpenAI()
    grader = build_score_model_grader(judge_model=args.judge_model)

    # Build cases: (task_type, row_idx, label, sample_text, expected_min, expected_max)
    cases: list[tuple] = []
    for t, items in sorted(by_type.items()):
        for idx, item in enumerate(items[: args.max_per_type]):
            cases.append((t, idx, "correct", _correct_sample(item), 0.95, 1.01))
            cases.append((t, idx, "wrong",   _wrong_sample(item),   -0.01, 0.05))
            cases.append((t, idx, "empty",   _empty_sample(item),   -0.01, 0.05))
            if _is_partial_credit_task(item):
                cases.append((t, idx, "finish_only", _finish_only_sample(item), 0.35, 0.45))

    print(f"Running {len(cases)} grader invocations with {args.workers} workers...")
    results: list[tuple] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(_run_one, client, grader, sample_text, by_type[t][idx]):
                (t, idx, label, expected_lo, expected_hi)
            for (t, idx, label, sample_text, expected_lo, expected_hi) in cases
        }
        for fut in as_completed(futures):
            t, idx, label, lo, hi = futures[fut]
            try:
                r = fut.result()
            except Exception as e:
                r = {"reward": None, "errors": [f"EXC:{type(e).__name__}: {e}"], "elapsed": 0.0}
            results.append((t, idx, label, lo, hi, r))

    print()
    print(f"{'task':>4} {'idx':>3} {'label':>12} {'reward':>7} {'expected':>14} {'errors':<60}")
    print("-" * 110)
    failures = 0
    for (t, idx, label, lo, hi, r) in sorted(results, key=lambda x: (x[0], x[1], x[2])):
        rew = r["reward"]
        ok = (rew is not None) and (lo <= rew <= hi) and not r["errors"]
        marker = "OK " if ok else "BAD"
        if not ok:
            failures += 1
        print(f"{t:>4} {idx:>3} {label:>12} {str(rew):>7} {f'[{lo:.2f},{hi:.2f}]':>14} "
              f"{','.join(r['errors'])[:58]:<60} {marker}")

    print()
    print(f"OK: {len(results) - failures}/{len(results)}  Failures: {failures}")
    if failures > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
