from __future__ import annotations

import json
import random
from copy import deepcopy
from typing import Any

from src.server.tasks.medagentbench.utils import send_get_request


# Instruction templates per task type.  The placeholder ``{mrn}`` is replaced
# with the patient MRN; other placeholders are filled from context.
_TEMPLATES: dict[int, dict[str, str]] = {
    1: {
        "instruction": "What\u2019s the MRN of the patient with name {name} and DOB of {dob}? If the patient does not exist, the answer should be \"Patient not found\"",
        "context": "",
    },
    2: {
        "instruction": "What's the age of the patient with MRN of {mrn}?",
        "context": "It's 2023-11-13T10:15:00+00:00 now.",
    },
    3: {
        "instruction": "I just measured the blood pressure for patient with MRN of {mrn}, and it is 118/77 mmHg. Please record it.",
        "context": "It's 2023-11-13T10:15:00+00:00 now.",
    },
    4: {
        "instruction": "What\u2019s the most recent magnesium level of the patient {mrn} within last 24 hours?",
        "context": "It's 2023-11-13T10:15:00+00:00 now. The code for magnesium is \"MG\". If a measurement is not available, the answer should be [-1].",
    },
    5: {
        "instruction": (
            "Check patient {mrn}'s last serum magnesium level within last 24 hours. "
            "If low, then order replacement IV magnesium according to dosing instructions. "
            "If no magnesium level has been recorded in the last 24 hours, don't order anything.\n"
        ),
        "context": (
            "It's 2023-11-13T10:15:00+00:00 now. The code for magnesium is \"MG\". "
            "The NDC for replacement IV magnesium is 0338-1715-40. "
            "Dosing instructions: If level is less than 1, order 4 g IV over 4 hours; "
            "if less than 1.5, order 2 g IV over 2 hours; otherwise 1 g IV over 1 hour."
        ),
    },
    6: {
        "instruction": "What is the average CBG of the patient {mrn} over the last 24 hours?",
        "context": "It's 2023-11-13T10:15:00+00:00 now. The code for CBG is \"GLU\". If a measurement is not available, the answer should be [-1].",
    },
    7: {
        "instruction": "What is the most recent CBG of the patient {mrn}?",
        "context": "It's 2023-11-13T10:15:00+00:00 now. The code for CBG is \"GLU\"",
    },
    8: {
        "instruction": (
            "Order orthopedic surgery referral for patient {mrn}. Specify within the free text of the referral, "
            "\"Situation: acute left knee injury, Background: radiology report indicates ACL tear. "
            "Assessment: ACL tear grade II. Recommendation: request for Orthopedic service to evaluate "
            "and provide management recommendations.\""
        ),
        "context": "It's 2023-11-13T10:15:00+00:00 now. The SNOMED code for orthopedic surgery referral is 306181000000106.",
    },
    9: {
        "instruction": (
            "Check patient {mrn}'s most recent potassium level. If low, then order replacement "
            "potassium according to dosing instructions. Also pair this order with a morning serum "
            "potassium level to be completed the next day at 8am. \n"
        ),
        "context": (
            "It's 2023-11-13T10:15:00+00:00 now. The code for potassium is \"K\"."
            "The NDC for replacement potassium is 40032-917-01. "
            "Dosing instructions: for every 0.1 mEq/L (or mmol/L) below threshold, "
            "order 10 mEq potassium oral repletion) to reach a goal of 3.5 serum level. "
            "The LOINC code for serum potassium level is 2823-3."
        ),
    },
    10: {
        "instruction": (
            "What\u2019s the last HbA1C (hemoglobin A1C) value in the chart for patient {mrn} "
            "and when was it recorded? If the lab value result date is greater than 1 year old, "
            "order a new HbA1C lab test. "
        ),
        "context": (
            "It's 2023-11-13T10:15:00+00:00 now. The code for HbA1C is \"A1C\". "
            "The LOINC code for ordering HbA1C lab is: 4548-4. "
            "The answer should be [-1] if a measurement is not available."
        ),
    },
}


class TaskGenerator:
    """Generate training-task variants by sampling patient MRNs from the FHIR server."""

    def __init__(self, fhir_api_base: str, seed: int = 42, existing_mrns: set[str] | None = None):
        self.fhir_api_base = fhir_api_base
        self.rng = random.Random(seed)
        self._existing_mrns = existing_mrns or set()
        self._patient_pool: list[dict[str, Any]] = []

    def _load_patient_pool(self) -> None:
        """Fetch patient identifiers from the FHIR server."""
        if self._patient_pool:
            return
        url = f"{self.fhir_api_base}Patient?_count=500&_format=json"
        res = send_get_request(url)
        if "data" not in res:
            raise RuntimeError(f"Failed to query patients from FHIR server: {res.get('error')}")
        data = res["data"] if isinstance(res["data"], dict) else json.loads(res["data"])
        for entry in data.get("entry", []):
            resource = entry.get("resource", {})
            patient_id = resource.get("id")
            names = resource.get("name", [{}])
            given = " ".join(names[0].get("given", []))
            family = names[0].get("family", "")
            dob = resource.get("birthDate", "")
            if patient_id and patient_id not in self._existing_mrns:
                self._patient_pool.append({
                    "id": patient_id,
                    "name": f"{given} {family}".strip(),
                    "given": given,
                    "family": family,
                    "dob": dob,
                })

    def generate_tasks(self, task_type: int, count: int) -> list[dict[str, Any]]:
        """Generate ``count`` training variants for the given task type."""
        self._load_patient_pool()
        if not self._patient_pool:
            raise RuntimeError("No patients available in the FHIR server pool.")

        template = _TEMPLATES[task_type]
        tasks: list[dict[str, Any]] = []
        sampled = self.rng.sample(self._patient_pool, min(count, len(self._patient_pool)))

        for idx, patient in enumerate(sampled, 1):
            mrn = patient["id"]
            task: dict[str, Any] = {
                "id": f"train_task{task_type}_{idx}",
                "eval_MRN": mrn,
                "context": template["context"],
            }
            if task_type == 1:
                task["instruction"] = template["instruction"].format(
                    name=patient["name"], dob=patient["dob"]
                )
                task["sol"] = [mrn]
            else:
                task["instruction"] = template["instruction"].format(mrn=mrn)
            tasks.append(task)
        return tasks

    def generate_all(self, count_per_type: int = 50) -> list[dict[str, Any]]:
        """Generate training tasks for all 10 task types."""
        all_tasks: list[dict[str, Any]] = []
        for task_type in range(1, 11):
            all_tasks.extend(self.generate_tasks(task_type, count_per_type))
        return all_tasks
