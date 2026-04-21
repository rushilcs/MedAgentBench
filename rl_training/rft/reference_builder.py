"""Pre-compute everything the grader needs (offline, no FHIR network).

For each training task, given live FHIR access, this module computes:

* ``reference_sol``: the gold ``FINISH([...])`` payload (as a JSON-ready list)
* ``task_params``: auxiliary values the offline grader needs to mirror ``refsol``
  (e.g. last-measured magnesium for task5's dose check, A1C timestamp for task10)
* ``prefetched_gets``: the verbatim GET responses to inline into the single-turn
  RFT prompt

We deliberately mirror the formulas in
[src/server/tasks/medagentbench/refsol.py](src/server/tasks/medagentbench/refsol.py)
so the offline grader returns the same verdict without network.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from src.server.tasks.medagentbench.utils import send_get_request

logger = logging.getLogger(__name__)

_CUTOFF = datetime.fromisoformat("2023-11-13T10:15:00+00:00")


@dataclass
class TaskReference:
    task_type: int
    reference_sol: list[Any] | None  # None for tasks where FINISH([]) suffices
    task_params: dict[str, Any] = field(default_factory=dict)
    prefetched_gets: list[dict[str, str]] = field(default_factory=list)
    accepts_empty_finish: bool = False  # tasks 5/9/10 accept both [] and reference_sol

    def to_json(self) -> dict[str, Any]:
        return {
            "task_type": self.task_type,
            "reference_sol": self.reference_sol,
            "task_params": self.task_params,
            "prefetched_gets": self.prefetched_gets,
            "accepts_empty_finish": self.accepts_empty_finish,
        }


def _get_json(url: str) -> dict | None:
    res = send_get_request(url)
    if "data" not in res:
        logger.warning("GET failed for %s: %s", url, res.get("error"))
        return None
    data = res["data"]
    if isinstance(data, str):
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            return None
    return data


def _task_type_int(task_id: str) -> int:
    for part in task_id.split("_"):
        if part.startswith("task") and part[4:].isdigit():
            return int(part[4:])
    raise ValueError(f"Cannot infer task type from id={task_id!r}")


def _calculate_age(dob: datetime) -> int:
    today = datetime(2023, 11, 13)
    age = today.year - dob.year
    if (today.month, today.day) < (dob.month, dob.day):
        age -= 1
    return age


def _latest_in_window(entries: list[dict], within_hours: float | None = None) -> tuple[datetime | None, Any]:
    last_meas: datetime | None = None
    last_value: Any = None
    for entry in entries:
        etime = datetime.fromisoformat(entry["resource"]["effectiveDateTime"])
        value = entry["resource"]["valueQuantity"]["value"]
        if within_hours is not None and etime < (_CUTOFF - timedelta(hours=within_hours)):
            continue
        if last_meas is None or etime > last_meas:
            last_meas = etime
            last_value = value
    return last_meas, last_value


def _fhir(fhir_api_base: str) -> str:
    return fhir_api_base.rstrip("/") + "/"


def build_reference(task: dict[str, Any], fhir_api_base: str) -> TaskReference | None:
    """Build the offline-grader reference for a single training task.

    Returns ``None`` if required FHIR data cannot be fetched.
    """
    base = _fhir(fhir_api_base)
    task_type = _task_type_int(task["id"])
    mrn = task["eval_MRN"]

    if task_type == 1:
        url = f"{base}Patient?identifier={mrn}&_format=json"
        data = _get_json(url)
        if data is None:
            return None
        return TaskReference(
            task_type=1,
            reference_sol=task.get("sol", [mrn]),
            task_params={"mrn": mrn},
            prefetched_gets=[{"url": url.replace("&_format=json", ""), "body": json.dumps(data)}],
        )

    if task_type == 2:
        url = f"{base}Patient?identifier={mrn}&_format=json"
        data = _get_json(url)
        if data is None or not data.get("entry"):
            return None
        dob_str = data["entry"][0]["resource"]["birthDate"]
        dob = datetime.strptime(dob_str, "%Y-%m-%d")
        return TaskReference(
            task_type=2,
            reference_sol=[_calculate_age(dob)],
            task_params={"mrn": mrn, "dob": dob_str},
            prefetched_gets=[{"url": url.replace("&_format=json", ""), "body": json.dumps(data)}],
        )

    if task_type == 3:
        return TaskReference(
            task_type=3,
            reference_sol=[],
            task_params={"mrn": mrn},
            prefetched_gets=[],
        )

    if task_type == 4:
        url = f"{base}Observation?patient={mrn}&code=MG&_count=5000&_format=json"
        data = _get_json(url)
        entries = data.get("entry", []) if data else []
        _, last_value = _latest_in_window(entries, within_hours=24)
        return TaskReference(
            task_type=4,
            reference_sol=[last_value if last_value is not None else -1],
            task_params={"mrn": mrn},
            prefetched_gets=[{"url": url.replace("&_format=json", ""), "body": json.dumps(data or {"total": 0})}],
        )

    if task_type == 5:
        url = f"{base}Observation?patient={mrn}&code=MG&_count=5000&_format=json"
        data = _get_json(url)
        entries = data.get("entry", []) if data else []
        _, last_value = _latest_in_window(entries, within_hours=24)
        must_post = last_value is not None and last_value <= 1.9
        return TaskReference(
            task_type=5,
            reference_sol=[last_value if last_value is not None else -1],
            task_params={
                "mrn": mrn,
                "last_value": last_value,
                "must_post": must_post,
            },
            prefetched_gets=[{"url": url.replace("&_format=json", ""), "body": json.dumps(data or {"total": 0})}],
            accepts_empty_finish=True,
        )

    if task_type == 6:
        url = f"{base}Observation?patient={mrn}&code=GLU&_count=5000&_format=json"
        data = _get_json(url)
        entries = data.get("entry", []) if data else []
        glu_sum, glu_count = 0.0, 0
        for entry in entries:
            etime = datetime.fromisoformat(entry["resource"]["effectiveDateTime"])
            if etime >= (_CUTOFF - timedelta(hours=24)):
                glu_sum += entry["resource"]["valueQuantity"]["value"]
                glu_count += 1
        avg = glu_sum / glu_count if glu_count else -1
        return TaskReference(
            task_type=6,
            reference_sol=[avg],
            task_params={"mrn": mrn, "tolerance": 0.1},
            prefetched_gets=[{"url": url.replace("&_format=json", ""), "body": json.dumps(data or {"total": 0})}],
        )

    if task_type == 7:
        url = f"{base}Observation?patient={mrn}&code=GLU&_count=5000&_format=json"
        data = _get_json(url)
        entries = data.get("entry", []) if data else []
        _, last_value = _latest_in_window(entries, within_hours=None)
        return TaskReference(
            task_type=7,
            reference_sol=[last_value if last_value is not None else -1],
            task_params={"mrn": mrn},
            prefetched_gets=[{"url": url.replace("&_format=json", ""), "body": json.dumps(data or {"total": 0})}],
        )

    if task_type == 8:
        return TaskReference(
            task_type=8,
            reference_sol=[],
            task_params={"mrn": mrn},
            prefetched_gets=[],
        )

    if task_type == 9:
        url = f"{base}Observation?patient={mrn}&code=K&_count=5000&_format=json"
        data = _get_json(url)
        entries = data.get("entry", []) if data else []
        _, last_value = _latest_in_window(entries, within_hours=None)
        must_post = last_value is not None and last_value < 3.5
        return TaskReference(
            task_type=9,
            reference_sol=[last_value if last_value is not None else -1],
            task_params={
                "mrn": mrn,
                "last_value": last_value,
                "must_post": must_post,
            },
            prefetched_gets=[{"url": url.replace("&_format=json", ""), "body": json.dumps(data or {"total": 0})}],
            accepts_empty_finish=True,
        )

    if task_type == 10:
        url = f"{base}Observation?patient={mrn}&code=A1C&_count=5000&_format=json"
        data = _get_json(url)
        entries = data.get("entry", []) if data else []
        last_meas: datetime | None = None
        last_value: Any = None
        last_time: str | None = None
        for entry in entries:
            etime = datetime.fromisoformat(entry["resource"]["effectiveDateTime"])
            value = entry["resource"]["valueQuantity"]["value"]
            if last_meas is None or etime > last_meas:
                last_meas = etime
                last_value = value
                last_time = entry["resource"]["effectiveDateTime"]
        needs_order = (last_value is None) or (
            last_meas is not None
            and last_meas < datetime.fromisoformat("2022-11-13T10:15:00+00:00")
        )
        if last_value is None:
            ref_sol: list[Any] = [-1]
        else:
            ref_sol = [last_value, last_time]
        return TaskReference(
            task_type=10,
            reference_sol=ref_sol,
            task_params={
                "mrn": mrn,
                "last_value": last_value,
                "last_time": last_time,
                "needs_order": needs_order,
            },
            prefetched_gets=[{"url": url.replace("&_format=json", ""), "body": json.dumps(data or {"total": 0})}],
            accepts_empty_finish=True,
        )

    raise ValueError(f"Unknown task type {task_type}")
