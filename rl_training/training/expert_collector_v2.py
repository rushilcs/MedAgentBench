"""V2 programmatic builders that push the time-window filter into the
FHIR query itself (`&date=ge<cutoff-24h>` for tasks 4/5/6 and equivalent
for 9/10).

Why: with `enable_thinking=False` at inference (required to keep refsol's
parser happy on POST tasks 3/8), the model can no longer "reason" about
date windows internally. Pushing the filter into the GET URL lets the
FHIR server do the filtering, so the assistant only has to:

  1. issue a date-filtered GET,
  2. read off the entries (latest value or empty),
  3. emit FINISH([value]) or FINISH([-1]).

That's a one-shot pattern with no internal arithmetic.

The non-time-window builders (1, 2, 3, 7, 8) reuse the originals
unchanged.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any

from rl_training.data.trajectory import Trajectory, Turn
from rl_training.env.medagent_env import MedAgentEnv
from rl_training.training.expert_collector import (
    _BUILDERS as _BASE_BUILDERS,
    _build_initial_prompt,
    _get_json,
    _get_latest_obs,
    _make_traj,
    _CUTOFF,
)


# 24 hours before the benchmark cutoff (2023-11-13T10:15:00+00:00).
_CUTOFF_24H = (_CUTOFF - timedelta(hours=24)).isoformat()
# 1 year before the benchmark cutoff.
_CUTOFF_1Y = (_CUTOFF - timedelta(days=365)).isoformat()


def _filtered_obs(env: MedAgentEnv, mrn: str, code: str, ge: str) -> tuple[dict | None, datetime | None, Any]:
    """GET observations filtered by ``date=ge<ge>`` and return (data, latest_dt, latest_val)."""
    url = f"{env.fhir_api_base}Observation?patient={mrn}&code={code}&date=ge{ge}&_count=5000&_format=json"
    data = _get_json(url)
    last_meas, last_value = None, None
    if data:
        for entry in data.get("entry", []):
            etime = datetime.fromisoformat(entry["resource"]["effectiveDateTime"])
            value = entry["resource"]["valueQuantity"]["value"]
            if last_meas is None or etime > last_meas:
                last_meas = etime
                last_value = value
    return data, last_meas, last_value


def _build_task4_v2(task: dict, env: MedAgentEnv) -> Trajectory | None:
    prompt = _build_initial_prompt(env, task)
    mrn = task["eval_MRN"]
    data, _, last_value = _filtered_obs(env, mrn, "MG", _CUTOFF_24H)
    response_str = json.dumps(data) if data else '{"total": 0}'
    answer = [last_value if last_value is not None else -1]
    turns = [
        Turn("user", prompt),
        Turn("assistant",
             f"GET {env.fhir_api_base}Observation?patient={mrn}&code=MG&date=ge{_CUTOFF_24H}"),
        Turn("user", f"Here is the response from the GET request:\n{response_str}. Please call FINISH if you have got answers for all the questions and finished all the requested tasks"),
        Turn("assistant", f"FINISH({json.dumps(answer)})"),
    ]
    return _make_traj(task, turns)


def _build_task5_v2(task: dict, env: MedAgentEnv) -> Trajectory | None:
    prompt = _build_initial_prompt(env, task)
    mrn = task["eval_MRN"]
    data, _, last_value = _filtered_obs(env, mrn, "MG", _CUTOFF_24H)
    response_str = json.dumps(data) if data else '{"total": 0}'

    turns = [
        Turn("user", prompt),
        Turn("assistant",
             f"GET {env.fhir_api_base}Observation?patient={mrn}&code=MG&date=ge{_CUTOFF_24H}"),
        Turn("user", f"Here is the response from the GET request:\n{response_str}. Please call FINISH if you have got answers for all the questions and finished all the requested tasks"),
    ]

    answer = [last_value if last_value is not None else -1]

    if last_value is not None and last_value <= 1.9:
        if last_value < 1:
            dose, rate = 4, 4
        elif last_value < 1.5:
            dose, rate = 2, 2
        else:
            dose, rate = 1, 1
        payload = {
            "resourceType": "MedicationRequest",
            "medicationCodeableConcept": {"coding": [{"system": "http://hl7.org/fhir/sid/ndc", "code": "0338-1715-40"}]},
            "authoredOn": "2023-11-13T10:15:00+00:00",
            "dosageInstruction": [{"route": "IV", "doseAndRate": [{"doseQuantity": {"value": dose, "unit": "g"}, "rateQuantity": {"value": rate, "unit": "h"}}]}],
            "status": "active",
            "intent": "order",
            "subject": {"reference": f"Patient/{mrn}"},
        }
        turns.append(Turn("assistant", f"POST {env.fhir_api_base}MedicationRequest\n{json.dumps(payload)}"))
        turns.append(Turn("user", "POST request accepted and executed successfully. Please call FINISH if you have got answers for all the questions and finished all the requested tasks"))

    turns.append(Turn("assistant", f"FINISH({json.dumps(answer)})"))
    return _make_traj(task, turns)


def _build_task6_v2(task: dict, env: MedAgentEnv) -> Trajectory | None:
    prompt = _build_initial_prompt(env, task)
    mrn = task["eval_MRN"]
    data, _, _ = _filtered_obs(env, mrn, "GLU", _CUTOFF_24H)
    response_str = json.dumps(data) if data else '{"total": 0}'
    glu_sum, glu_count = 0.0, 0
    if data:
        for entry in data.get("entry", []):
            glu_sum += entry["resource"]["valueQuantity"]["value"]
            glu_count += 1
    avg = glu_sum / glu_count if glu_count else -1
    answer = [avg]
    turns = [
        Turn("user", prompt),
        Turn("assistant",
             f"GET {env.fhir_api_base}Observation?patient={mrn}&code=GLU&date=ge{_CUTOFF_24H}"),
        Turn("user", f"Here is the response from the GET request:\n{response_str}. Please call FINISH if you have got answers for all the questions and finished all the requested tasks"),
        Turn("assistant", f"FINISH({json.dumps(answer)})"),
    ]
    return _make_traj(task, turns)


def _build_task9_v2(task: dict, env: MedAgentEnv) -> Trajectory | None:
    """Latest K (any time) + replacement orders if K < 3.5.

    For task9 the time window is 'any time' so we keep the unfiltered GET
    but add ``&_sort=-date&_count=1`` so the server returns just the
    latest entry. That single-entry pattern is also more robust than
    asking the model to scan a long list.
    """
    prompt = _build_initial_prompt(env, task)
    mrn = task["eval_MRN"]
    url = f"{env.fhir_api_base}Observation?patient={mrn}&code=K&_sort=-date&_count=1&_format=json"
    data = _get_json(url)
    response_str = json.dumps(data) if data else '{"total": 0}'
    last_value = None
    if data and data.get("entry"):
        last_value = data["entry"][0]["resource"]["valueQuantity"]["value"]

    turns = [
        Turn("user", prompt),
        Turn("assistant",
             f"GET {env.fhir_api_base}Observation?patient={mrn}&code=K&_sort=-date&_count=1"),
        Turn("user", f"Here is the response from the GET request:\n{response_str}. Please call FINISH if you have got answers for all the questions and finished all the requested tasks"),
    ]

    answer = [last_value if last_value is not None else -1]

    if last_value is not None and last_value < 3.5:
        dose = round((3.5 - last_value) / 0.1 * 10, 1)
        med_payload = {
            "resourceType": "MedicationRequest",
            "medicationCodeableConcept": {"coding": [{"system": "http://hl7.org/fhir/sid/ndc", "code": "40032-917-01"}]},
            "authoredOn": "2023-11-13T10:15:00+00:00",
            "dosageInstruction": [{"route": "oral", "doseAndRate": [{"doseQuantity": {"value": dose, "unit": "mEq"}}]}],
            "status": "active",
            "intent": "order",
            "subject": {"reference": f"Patient/{mrn}"},
        }
        turns.append(Turn("assistant", f"POST {env.fhir_api_base}MedicationRequest\n{json.dumps(med_payload)}"))
        turns.append(Turn("user", "POST request accepted and executed successfully. Please call FINISH if you have got answers for all the questions and finished all the requested tasks"))

        lab_payload = {
            "resourceType": "ServiceRequest",
            "code": {"coding": [{"system": "http://loinc.org", "code": "2823-3"}]},
            "authoredOn": "2023-11-13T10:15:00+00:00",
            "status": "active",
            "intent": "order",
            "priority": "stat",
            "subject": {"reference": f"Patient/{mrn}"},
            "occurrenceDateTime": "2023-11-14T08:00:00+00:00",
        }
        turns.append(Turn("assistant", f"POST {env.fhir_api_base}ServiceRequest\n{json.dumps(lab_payload)}"))
        turns.append(Turn("user", "POST request accepted and executed successfully. Please call FINISH if you have got answers for all the questions and finished all the requested tasks"))

    turns.append(Turn("assistant", f"FINISH({json.dumps(answer)})"))
    return _make_traj(task, turns)


def _build_task7_v2(task: dict, env: MedAgentEnv) -> Trajectory | None:
    """Latest GLU. Same trick as task9: server-side sort + _count=1."""
    prompt = _build_initial_prompt(env, task)
    mrn = task["eval_MRN"]
    url = f"{env.fhir_api_base}Observation?patient={mrn}&code=GLU&_sort=-date&_count=1&_format=json"
    data = _get_json(url)
    response_str = json.dumps(data) if data else '{"total": 0}'
    last_value = None
    if data and data.get("entry"):
        last_value = data["entry"][0]["resource"]["valueQuantity"]["value"]
    answer = [last_value if last_value is not None else -1]
    turns = [
        Turn("user", prompt),
        Turn("assistant",
             f"GET {env.fhir_api_base}Observation?patient={mrn}&code=GLU&_sort=-date&_count=1"),
        Turn("user", f"Here is the response from the GET request:\n{response_str}. Please call FINISH if you have got answers for all the questions and finished all the requested tasks"),
        Turn("assistant", f"FINISH({json.dumps(answer)})"),
    ]
    return _make_traj(task, turns)


def _build_task10_v2(task: dict, env: MedAgentEnv) -> Trajectory | None:
    """Most recent A1C and order a stat A1C lab if the latest is >1y old.

    We split the GET into two queries: first ask for the latest A1C
    (sorted desc, _count=1). That immediately tells us value + date,
    and whether we need to place a follow-up order.
    """
    prompt = _build_initial_prompt(env, task)
    mrn = task["eval_MRN"]
    url = f"{env.fhir_api_base}Observation?patient={mrn}&code=A1C&_sort=-date&_count=1&_format=json"
    data = _get_json(url)
    response_str = json.dumps(data) if data else '{"total": 0}'

    last_value, last_time_str, last_meas = None, None, None
    if data and data.get("entry"):
        e = data["entry"][0]["resource"]
        last_value = e["valueQuantity"]["value"]
        last_time_str = e["effectiveDateTime"]
        last_meas = datetime.fromisoformat(last_time_str)

    one_year_ago = datetime.fromisoformat("2022-11-13T10:15:00+00:00")
    needs_order = (last_value is None) or (last_meas < one_year_ago)
    answer = [-1] if last_value is None else [last_value, last_time_str]

    turns = [
        Turn("user", prompt),
        Turn("assistant",
             f"GET {env.fhir_api_base}Observation?patient={mrn}&code=A1C&_sort=-date&_count=1"),
        Turn("user", f"Here is the response from the GET request:\n{response_str}. Please call FINISH if you have got answers for all the questions and finished all the requested tasks"),
    ]

    if needs_order:
        lab_payload = {
            "resourceType": "ServiceRequest",
            "code": {"coding": [{"system": "http://loinc.org", "code": "4548-4"}]},
            "authoredOn": "2023-11-13T10:15:00+00:00",
            "status": "active",
            "intent": "order",
            "priority": "stat",
            "subject": {"reference": f"Patient/{mrn}"},
        }
        turns.append(Turn("assistant", f"POST {env.fhir_api_base}ServiceRequest\n{json.dumps(lab_payload)}"))
        turns.append(Turn("user", "POST request accepted and executed successfully. Please call FINISH if you have got answers for all the questions and finished all the requested tasks"))

    turns.append(Turn("assistant", f"FINISH({json.dumps(answer)})"))
    return _make_traj(task, turns)


_BUILDERS_V2: dict[str, Any] = {
    **_BASE_BUILDERS,
    "task4": _build_task4_v2,
    "task5": _build_task5_v2,
    "task6": _build_task6_v2,
    "task7": _build_task7_v2,
    "task9": _build_task9_v2,
    "task10": _build_task10_v2,
}
