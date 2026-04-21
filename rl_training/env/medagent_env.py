"""Standalone MedAgentBench environment for RL training and evaluation.

Mirrors the server-side logic in ``src/server/tasks/medagentbench/__init__.py``
but runs entirely in-process so it can be used without the AgentBench server.
"""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass, field
from typing import Any

from src.server.tasks.medagentbench.utils import send_get_request
from rl_training.env.action_parser import parse_action, ParsedAction
from rl_training.env.reward import RewardConfig, compute_step_reward

_SYSTEM_PROMPT = (
    "You are an expert in using FHIR functions to assist medical professionals. "
    "You are given a question and a set of possible functions. Based on the question, "
    "you will need to make one or more function/tool calls to achieve the purpose.\n\n"
    "1. If you decide to invoke a GET function, you MUST put it in the format of\n"
    "GET url?param_name1=param_value1&param_name2=param_value2...\n\n"
    "2. If you decide to invoke a POST function, you MUST put it in the format of\n"
    "POST url\n"
    "[your payload data in JSON format]\n\n"
    "3. If you have got answers for all the questions and finished all the requested tasks, "
    "you MUST call to finish the conversation in the format of (make sure the list is JSON loadable.)\n"
    "FINISH([answer1, answer2, ...])\n\n"
    "Your response must be in the format of one of the three cases, and you can call only "
    "one function each time. You SHOULD NOT include any other text in the response.\n\n"
    "Here is a list of functions in JSON format that you can invoke. "
    "Note that you should use {api_base} as the api_base.\n{functions}\n\n"
    "Context: {context}\nQuestion: {question}"
)


@dataclass
class EnvState:
    history: list[dict[str, str]] = field(default_factory=list)
    done: bool = False
    status: str = "running"  # running | completed | invalid_action | limit_reached
    result: str | None = None


@dataclass
class StepResult:
    state: EnvState
    action: ParsedAction
    reward: float = 0.0


class MedAgentEnv:
    """In-process MedAgentBench environment."""

    def __init__(
        self,
        fhir_api_base: str,
        funcs: list[dict[str, Any]],
        max_rounds: int = 8,
        reward_config: RewardConfig | None = None,
    ):
        self.fhir_api_base = fhir_api_base
        self.funcs = funcs
        self.max_rounds = max_rounds
        self.reward_config = reward_config or RewardConfig()

        self._state = EnvState()
        self._task: dict[str, Any] = {}
        self._round = 0
        self.step_rewards: list[float] = []

        self._refsol = importlib.import_module("src.server.tasks.medagentbench.refsol")

    def reset(self, task: dict[str, Any]) -> EnvState:
        """Start a new episode for the given task."""
        self._task = task
        self._round = 0
        self.step_rewards = []

        prompt = _SYSTEM_PROMPT.format(
            api_base=self.fhir_api_base,
            functions=json.dumps(self.funcs),
            context=task.get("context", ""),
            question=task["instruction"],
        )
        self._state = EnvState(
            history=[{"role": "user", "content": prompt}],
            done=False,
            status="running",
        )
        return self._state

    def step(self, agent_response: str) -> StepResult:
        """Execute one agent action and return the new state."""
        self._round += 1
        self._state.history.append({"role": "agent", "content": agent_response})

        action = parse_action(agent_response)

        if action.kind == "get":
            get_res = send_get_request(action.url)
            if "data" in get_res:
                data = get_res["data"]
                if not isinstance(data, str):
                    data = json.dumps(data)
                reply = (
                    f"Here is the response from the GET request:\n{data}. "
                    "Please call FINISH if you have got answers for all the questions "
                    "and finished all the requested tasks"
                )
            else:
                reply = f"Error in sending the GET request: {get_res['error']}"
            self._state.history.append({"role": "user", "content": reply})
            r = compute_step_reward("get", True)

        elif action.kind == "post":
            if action.payload is None:
                self._state.history.append({"role": "user", "content": "Invalid POST request"})
                r = compute_step_reward("post", False)
            else:
                self._state.history.append({
                    "role": "user",
                    "content": (
                        "POST request accepted and executed successfully. "
                        "Please call FINISH if you have got answers for all the questions "
                        "and finished all the requested tasks"
                    ),
                })
                r = compute_step_reward("post", True)

        elif action.kind == "finish":
            self._state.done = True
            self._state.status = "completed"
            self._state.result = action.result
            r = compute_step_reward("finish", True)

        else:
            self._state.done = True
            self._state.status = "invalid_action"
            r = compute_step_reward("invalid", False)

        self.step_rewards.append(r)

        if not self._state.done and self._round >= self.max_rounds:
            self._state.done = True
            self._state.status = "limit_reached"

        return StepResult(state=self._state, action=action, reward=r)

    def grade(self) -> bool:
        """Grade the completed episode using the reference solution."""
        if self._state.status != "completed":
            return False
        task_id = self._task["id"].split("_")[0]
        if task_id.startswith("train_"):
            task_id = task_id.replace("train_", "")
        # For train_task5_3 → task5
        for part in self._task["id"].split("_"):
            if part.startswith("task"):
                task_id = part
                break

        grader = getattr(self._refsol, task_id, None)
        if grader is None:
            return False

        # Build a results-like object that refsol expects
        results = _ResultsProxy(
            history=self._state.history,
            result=self._state.result,
        )
        try:
            return grader(self._task, results, self.fhir_api_base) is True
        except Exception:
            return False


    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "MedAgentEnv":
        """Construct from a config dict (matching ``rl_training/configs/default.yaml``)."""
        with open(config["env"]["func_file"]) as f:
            funcs = json.load(f)
        return cls(
            fhir_api_base=config["env"]["fhir_api_base"],
            funcs=funcs,
            max_rounds=config["env"].get("max_rounds", 8),
        )


class _ResultsProxy:
    """Lightweight proxy matching the interface expected by refsol grader functions.

    The graders access ``results.history`` (list of objects with ``.role`` /
    ``.content``) and ``results.result`` (the FINISH payload string).
    """

    def __init__(self, history: list[dict[str, str]], result: str | None):
        self.history = [_HistoryEntry(h["role"], h["content"]) for h in history]
        self.result = result


class _HistoryEntry:
    __slots__ = ("role", "content")

    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content
