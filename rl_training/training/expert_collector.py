from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import Any

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

from rl_training.env.medagent_env import MedAgentEnv
from rl_training.env.reward import compute_episode_reward
from rl_training.agent.base_policy import BasePolicy
from rl_training.data.trajectory import Trajectory, Turn
from rl_training.data.trajectory_store import TrajectoryStore
from src.server.tasks.medagentbench.utils import send_get_request

logger = logging.getLogger(__name__)

_CUTOFF = datetime.fromisoformat("2023-11-13T10:15:00+00:00")


class ExpertCollector:
    """Collect expert (correct) trajectories for supervised fine-tuning."""

    def __init__(self, env: MedAgentEnv, store: TrajectoryStore):
        self.env = env
        self.store = store

    # ------------------------------------------------------------------
    # Model-based collection
    # ------------------------------------------------------------------

    def collect(
        self,
        tasks: list[dict[str, Any]],
        policy: BasePolicy,
        trajectories_per_task: int = 1,
    ) -> list[Trajectory]:
        """Run a (strong) model on each task, keep only correct trajectories."""
        collected: list[Trajectory] = []
        with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), MofNCompleteColumn()) as prog:
            ptask = prog.add_task("Collecting expert trajectories", total=len(tasks) * trajectories_per_task)
            for task in tasks:
                for _ in range(trajectories_per_task):
                    state = self.env.reset(task)
                    while not state.done:
                        action = policy.act(state.history)
                        result = self.env.step(action)
                        state = result.state
                    correct = self.env.grade() if state.status == "completed" else False
                    traj = Trajectory.from_env_history(
                        task=task,
                        history=state.history,
                        correct=correct,
                        status=state.status,
                        step_rewards=self.env.step_rewards,
                        model_id=getattr(policy, "model_id", ""),
                    )
                    traj.reward = compute_episode_reward(traj, correct, self.env.reward_config)
                    if correct:
                        collected.append(traj)
                    prog.advance(ptask)
        self.store.save_batch(collected)
        logger.info("Collected %d correct trajectories from %d attempts", len(collected), len(tasks) * trajectories_per_task)
        return collected

    # ------------------------------------------------------------------
    # Programmatic trajectory construction
    # ------------------------------------------------------------------

    def collect_programmatic(self, tasks: list[dict[str, Any]]) -> list[Trajectory]:
        """Build guaranteed-correct trajectories by reverse-engineering refsol logic.

        For each task we know the correct sequence of API calls and the
        expected FINISH answer.  We execute the real GET queries against the
        FHIR server to obtain the actual data and then construct the exact
        POST payloads and FINISH strings that ``refsol`` would accept.
        """
        collected: list[Trajectory] = []
        for task in tasks:
            task_id = task["id"].replace("train_", "")
            task_type = task_id.split("_")[0]
            builder = _BUILDERS.get(task_type)
            if builder is None:
                logger.warning("No programmatic builder for %s", task_type)
                continue
            try:
                traj = builder(task, self.env)
                if traj is not None:
                    collected.append(traj)
            except Exception as exc:
                logger.warning("Programmatic build failed for %s: %s", task["id"], exc)
        self.store.save_batch(collected)
        logger.info("Built %d programmatic trajectories", len(collected))
        return collected


# ======================================================================
# Per-task-type programmatic builders
# ======================================================================

def _fhir(base: str) -> str:
    return base.rstrip("/") + "/"


def _get_json(url: str) -> dict | None:
    res = send_get_request(url)
    if "data" not in res:
        return None
    return res["data"] if isinstance(res["data"], dict) else json.loads(res["data"])


def _make_traj(task: dict, turns: list[Turn], status: str = "completed") -> Trajectory:
    return Trajectory(
        task_id=task["id"],
        task_data=task,
        turns=turns,
        correct=True,
        status=status,
        num_steps=sum(1 for t in turns if t.role == "assistant"),
        model_id="programmatic",
    )


def _build_initial_prompt(env: MedAgentEnv, task: dict) -> str:
    state = env.reset(task)
    return state.history[0]["content"]


def _build_task1(task: dict, env: MedAgentEnv) -> Trajectory | None:
    prompt = _build_initial_prompt(env, task)
    mrn = task["eval_MRN"]
    sol = task.get("sol", [mrn])

    data = _get_json(f"{env.fhir_api_base}Patient?identifier={mrn}&_format=json")
    response_str = json.dumps(data) if data else '{"total": 0}'

    turns = [
        Turn("user", prompt),
        Turn("assistant", f"GET {env.fhir_api_base}Patient?identifier={mrn}"),
        Turn("user", f"Here is the response from the GET request:\n{response_str}. Please call FINISH if you have got answers for all the questions and finished all the requested tasks"),
        Turn("assistant", f"FINISH({json.dumps(sol)})"),
    ]
    return _make_traj(task, turns)


def _build_task2(task: dict, env: MedAgentEnv) -> Trajectory | None:
    prompt = _build_initial_prompt(env, task)
    mrn = task["eval_MRN"]
    data = _get_json(f"{env.fhir_api_base}Patient?identifier={mrn}&_format=json")
    if not data or not data.get("entry"):
        return None
    dob_str = data["entry"][0]["resource"]["birthDate"]
    dob = datetime.strptime(dob_str, "%Y-%m-%d")
    today = datetime(2023, 11, 13)
    age = today.year - dob.year
    if (today.month, today.day) < (dob.month, dob.day):
        age -= 1
    response_str = json.dumps(data)
    turns = [
        Turn("user", prompt),
        Turn("assistant", f"GET {env.fhir_api_base}Patient?identifier={mrn}"),
        Turn("user", f"Here is the response from the GET request:\n{response_str}. Please call FINISH if you have got answers for all the questions and finished all the requested tasks"),
        Turn("assistant", f"FINISH([{age}])"),
    ]
    return _make_traj(task, turns)


def _build_task3(task: dict, env: MedAgentEnv) -> Trajectory | None:
    prompt = _build_initial_prompt(env, task)
    mrn = task["eval_MRN"]
    payload = {
        "resourceType": "Observation",
        "category": [{"coding": [{"system": "http://hl7.org/fhir/observation-category", "code": "vital-signs", "display": "Vital Signs"}]}],
        "code": {"text": "BP"},
        "effectiveDateTime": "2023-11-13T10:15:00+00:00",
        "status": "final",
        "valueString": "118/77 mmHg",
        "subject": {"reference": f"Patient/{mrn}"},
    }
    turns = [
        Turn("user", prompt),
        Turn("assistant", f"POST {env.fhir_api_base}Observation\n{json.dumps(payload)}"),
        Turn("user", "POST request accepted and executed successfully. Please call FINISH if you have got answers for all the questions and finished all the requested tasks"),
        Turn("assistant", "FINISH([])"),
    ]
    return _make_traj(task, turns)


def _get_latest_obs(fhir_base: str, mrn: str, code: str, within_hours: float | None = None) -> tuple[datetime | None, Any]:
    """Get the latest observation value for a patient and code."""
    data = _get_json(f"{fhir_base}Observation?patient={mrn}&code={code}&_count=5000&_format=json")
    if not data:
        return None, None
    last_meas, last_value = None, None
    for entry in data.get("entry", []):
        etime = datetime.fromisoformat(entry["resource"]["effectiveDateTime"])
        value = entry["resource"]["valueQuantity"]["value"]
        if within_hours is not None and etime < (_CUTOFF - timedelta(hours=within_hours)):
            continue
        if last_meas is None or etime > last_meas:
            last_meas = etime
            last_value = value
    return last_meas, last_value


def _build_task4(task: dict, env: MedAgentEnv) -> Trajectory | None:
    prompt = _build_initial_prompt(env, task)
    mrn = task["eval_MRN"]
    data = _get_json(f"{env.fhir_api_base}Observation?patient={mrn}&code=MG&_count=5000&_format=json")
    response_str = json.dumps(data) if data else '{"total": 0}'
    _, last_value = _get_latest_obs(env.fhir_api_base, mrn, "MG", 24)
    answer = [last_value if last_value is not None else -1]
    turns = [
        Turn("user", prompt),
        Turn("assistant", f"GET {env.fhir_api_base}Observation?code=MG&patient={mrn}"),
        Turn("user", f"Here is the response from the GET request:\n{response_str}. Please call FINISH if you have got answers for all the questions and finished all the requested tasks"),
        Turn("assistant", f"FINISH({json.dumps(answer)})"),
    ]
    return _make_traj(task, turns)


def _build_task5(task: dict, env: MedAgentEnv) -> Trajectory | None:
    prompt = _build_initial_prompt(env, task)
    mrn = task["eval_MRN"]
    data = _get_json(f"{env.fhir_api_base}Observation?patient={mrn}&code=MG&_count=5000&_format=json")
    response_str = json.dumps(data) if data else '{"total": 0}'
    _, last_value = _get_latest_obs(env.fhir_api_base, mrn, "MG", 24)

    turns = [
        Turn("user", prompt),
        Turn("assistant", f"GET {env.fhir_api_base}Observation?code=MG&patient={mrn}"),
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


def _build_task6(task: dict, env: MedAgentEnv) -> Trajectory | None:
    prompt = _build_initial_prompt(env, task)
    mrn = task["eval_MRN"]
    data = _get_json(f"{env.fhir_api_base}Observation?patient={mrn}&code=GLU&_count=5000&_format=json")
    response_str = json.dumps(data) if data else '{"total": 0}'

    glu_sum, glu_count = 0.0, 0
    if data:
        for entry in data.get("entry", []):
            etime = datetime.fromisoformat(entry["resource"]["effectiveDateTime"])
            if etime >= (_CUTOFF - timedelta(hours=24)):
                glu_sum += entry["resource"]["valueQuantity"]["value"]
                glu_count += 1
    answer = [glu_sum / glu_count if glu_count else -1]

    turns = [
        Turn("user", prompt),
        Turn("assistant", f"GET {env.fhir_api_base}Observation?code=GLU&patient={mrn}"),
        Turn("user", f"Here is the response from the GET request:\n{response_str}. Please call FINISH if you have got answers for all the questions and finished all the requested tasks"),
        Turn("assistant", f"FINISH({json.dumps(answer)})"),
    ]
    return _make_traj(task, turns)


def _build_task7(task: dict, env: MedAgentEnv) -> Trajectory | None:
    prompt = _build_initial_prompt(env, task)
    mrn = task["eval_MRN"]
    data = _get_json(f"{env.fhir_api_base}Observation?patient={mrn}&code=GLU&_count=5000&_format=json")
    response_str = json.dumps(data) if data else '{"total": 0}'
    _, last_value = _get_latest_obs(env.fhir_api_base, mrn, "GLU")
    answer = [last_value if last_value is not None else -1]
    turns = [
        Turn("user", prompt),
        Turn("assistant", f"GET {env.fhir_api_base}Observation?code=GLU&patient={mrn}"),
        Turn("user", f"Here is the response from the GET request:\n{response_str}. Please call FINISH if you have got answers for all the questions and finished all the requested tasks"),
        Turn("assistant", f"FINISH({json.dumps(answer)})"),
    ]
    return _make_traj(task, turns)


def _build_task8(task: dict, env: MedAgentEnv) -> Trajectory | None:
    prompt = _build_initial_prompt(env, task)
    mrn = task["eval_MRN"]
    comment = (
        "Situation: acute left knee injury, Background: radiology report indicates ACL tear. "
        "Assessment: ACL tear grade II. Recommendation: request for Orthopedic service to evaluate "
        "and provide management recommendations."
    )
    payload = {
        "resourceType": "ServiceRequest",
        "code": {"coding": [{"system": "http://snomed.info/sct", "code": "306181000000106"}]},
        "authoredOn": "2023-11-13T10:15:00+00:00",
        "status": "active",
        "intent": "order",
        "priority": "stat",
        "subject": {"reference": f"Patient/{mrn}"},
        "note": {"text": comment},
    }
    turns = [
        Turn("user", prompt),
        Turn("assistant", f"POST {env.fhir_api_base}ServiceRequest\n{json.dumps(payload)}"),
        Turn("user", "POST request accepted and executed successfully. Please call FINISH if you have got answers for all the questions and finished all the requested tasks"),
        Turn("assistant", "FINISH([])"),
    ]
    return _make_traj(task, turns)


def _build_task9(task: dict, env: MedAgentEnv) -> Trajectory | None:
    prompt = _build_initial_prompt(env, task)
    mrn = task["eval_MRN"]
    data = _get_json(f"{env.fhir_api_base}Observation?patient={mrn}&code=K&_count=5000&_format=json")
    response_str = json.dumps(data) if data else '{"total": 0}'
    _, last_value = _get_latest_obs(env.fhir_api_base, mrn, "K")

    turns = [
        Turn("user", prompt),
        Turn("assistant", f"GET {env.fhir_api_base}Observation?code=K&patient={mrn}"),
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


def _build_task10(task: dict, env: MedAgentEnv) -> Trajectory | None:
    prompt = _build_initial_prompt(env, task)
    mrn = task["eval_MRN"]
    data = _get_json(f"{env.fhir_api_base}Observation?patient={mrn}&code=A1C&_count=5000&_format=json")
    response_str = json.dumps(data) if data else '{"total": 0}'

    last_meas, last_value = None, None
    last_time_str = None
    if data:
        for entry in data.get("entry", []):
            etime = datetime.fromisoformat(entry["resource"]["effectiveDateTime"])
            value = entry["resource"]["valueQuantity"]["value"]
            if last_meas is None or etime > last_meas:
                last_meas = etime
                last_value = value
                last_time_str = entry["resource"]["effectiveDateTime"]

    if last_value is None:
        answer = [-1]
    else:
        answer = [last_value, last_time_str]

    turns = [
        Turn("user", prompt),
        Turn("assistant", f"GET {env.fhir_api_base}Observation?code=A1C&patient={mrn}"),
        Turn("user", f"Here is the response from the GET request:\n{response_str}. Please call FINISH if you have got answers for all the questions and finished all the requested tasks"),
    ]

    needs_order = (last_value is None) or (last_meas < datetime.fromisoformat("2022-11-13T10:15:00+00:00"))
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


_BUILDERS: dict[str, Any] = {
    "task1": _build_task1,
    "task2": _build_task2,
    "task3": _build_task3,
    "task4": _build_task4,
    "task5": _build_task5,
    "task6": _build_task6,
    "task7": _build_task7,
    "task8": _build_task8,
    "task9": _build_task9,
    "task10": _build_task10,
}
