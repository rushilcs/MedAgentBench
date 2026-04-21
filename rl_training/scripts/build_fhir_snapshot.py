#!/usr/bin/env python3
"""Pre-record FHIR GET responses for training tasks.

Given a training-tasks JSON file, this script iterates each task and exercises
the GETs that a reasonable agent (or the ``refsol`` grader) would issue, caching
each response to a JSONL snapshot. At training time, ``FhirSnapshot(mode="replay")``
serves those responses from memory so rollouts don't hit the live FHIR server.

Strategy per task type (see ``src/server/tasks/medagentbench/refsol.py``):
  - task1: Patient?name, Patient?family, Patient?given, Patient?birthdate
  - task2, 3:  Patient?identifier={mrn}
  - task4, 5:  Observation?patient={mrn}&code=MG
  - task6, 7:  Observation?patient={mrn}&code=GLU
  - task8:     Patient?identifier={mrn} (for referral subject)
  - task9:     Observation?patient={mrn}&code=K
  - task10:    Observation?patient={mrn}&code=A1C

For coverage, we also warm-fetch the Patient?identifier row for every task so
the agent can always look up the patient's birthDate if it wants to.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from rl_training.env.fhir_snapshot import FhirSnapshot

logger = logging.getLogger(__name__)


_CODE_BY_TASK: dict[int, str] = {
    4: "MG", 5: "MG",
    6: "GLU", 7: "GLU",
    9: "K",
    10: "A1C",
}


def _task_urls(task: dict[str, Any], fhir_api_base: str) -> list[str]:
    """Return the list of FHIR GET URLs to prefetch for one task."""
    task_id = task["id"]
    mrn = task.get("eval_MRN", "")
    urls: list[str] = []

    # Identify the task type from the id (works for both benchmark and train_ ids)
    task_type: int | None = None
    for part in task_id.split("_"):
        if part.startswith("task"):
            try:
                task_type = int(part[len("task"):])
            except ValueError:
                pass
            break
    if task_type is None:
        return urls

    if task_type == 1:
        # The model typically queries by name + DOB. We cache a wildcard Patient
        # search so any follow-up agent query hits. task1 doesn't need the MRN
        # (that's what the model has to find).
        instruction = task.get("instruction", "")
        # Best-effort: the instruction text contains "name X and DOB of Y"
        # but we don't parse it here; the full Patient list is enough.
        urls.append(f"{fhir_api_base}Patient?_count=500")
    else:
        urls.append(f"{fhir_api_base}Patient?identifier={mrn}")

    code = _CODE_BY_TASK.get(task_type)
    if code and mrn:
        urls.append(
            f"{fhir_api_base}Observation?patient={mrn}&code={code}&_count=5000"
        )

    return urls


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FHIR snapshot for training tasks")
    parser.add_argument("--tasks", required=True,
                        help="Path to training tasks JSON (or benchmark tasks JSON)")
    parser.add_argument("--fhir-base", default="http://localhost:8080/fhir/")
    parser.add_argument("--output", default="rl_training/outputs/fhir_snapshot.jsonl")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    with open(args.tasks) as f:
        tasks = json.load(f)
    logger.info("Loaded %d tasks from %s", len(tasks), args.tasks)

    # Fresh snapshot in record mode, writing to disk as we go.
    snapshot = FhirSnapshot(mode="record", path=args.output, fallthrough=True)
    if os.path.exists(args.output):
        logger.warning("Output %s exists; appending (existing entries are kept).", args.output)
        snapshot.load(args.output)

    total_urls = 0
    unique_before = len(snapshot._cache)  # noqa: SLF001 (internal is fine, same package)
    for task in tasks:
        urls = _task_urls(task, args.fhir_base)
        for url in urls:
            snapshot.send_get_request(url)
            total_urls += 1

    unique_after = len(snapshot._cache)  # noqa: SLF001
    stats = snapshot.stats()
    logger.info(
        "Recorded %d calls; %d unique URLs (added %d new rows). Stats: %s",
        total_urls, unique_after, unique_after - unique_before, stats,
    )
    logger.info("Snapshot written to %s", args.output)


if __name__ == "__main__":
    main()
