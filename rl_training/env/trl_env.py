"""TRL-compatible environment for MedAgentBench GRPO training.

When passed as ``environment_factory`` to ``GRPOTrainer``, the trainer
instantiates one ``MedAgentBenchEnv`` per rollout slot.  Public methods
(excluding ``reset``) are automatically exposed as tools that the model
can call via native tool-calling.

``reset()`` is called before each generation batch; TRL forwards dataset row
fields as keyword arguments, so ``reset`` must accept ``**kwargs``.
"""

from __future__ import annotations

import json
import os
from typing import Any

import requests


def _get_fhir_base() -> str:
    return os.environ.get("FHIR_API_BASE", "http://localhost:8080/fhir/")


def _send_get(url: str) -> dict[str, Any]:
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        ct = resp.headers.get("Content-Type", "")
        data = resp.json() if "json" in ct else resp.text
        return {"status_code": resp.status_code, "data": data}
    except Exception as e:
        return {"error": str(e)}


class MedAgentBenchEnv:
    """Stateful FHIR environment exposed as TRL tools.

    Public methods become tools available to the model during GRPO rollouts.
    """

    def __init__(self):
        self.fhir_base = _get_fhir_base()
        self._history: list[dict[str, str]] = []
        self._finished = False
        self._finish_result: str | None = None
        self._post_count = 0
        self._get_count = 0
        self._step_count = 0

    def reset(self, **kwargs: Any) -> None:
        """Reset state between rollouts.

        TRL passes each dataset row as keyword args (``prompt``, ``task_id``,
        ``instruction``, etc.); we accept and ignore them for compatibility.
        """
        _ = kwargs  # reserved for task-conditioned resets / reward shaping
        self._history = []
        self._finished = False
        self._finish_result = None
        self._post_count = 0
        self._get_count = 0
        self._step_count = 0

    def get_fhir_resource(self, url: str) -> str:
        """Execute a GET request against the FHIR server.

        Use this to read patient data, observations, medication requests,
        or any other FHIR resource. Append query parameters as needed.

        Args:
            url: The full FHIR URL to query, e.g.
                 ``{fhir_base}Patient?identifier=S1234567&_format=json``

        Returns:
            The JSON response from the FHIR server, or an error message.
        """
        if "&_format=json" not in url and "?_format=json" not in url:
            url += "&_format=json" if "?" in url else "?_format=json"
        self._step_count += 1
        self._get_count += 1
        res = _send_get(url)
        if "data" in res:
            data = res["data"]
            if not isinstance(data, str):
                data = json.dumps(data, indent=2)
            self._history.append({"role": "tool", "action": "GET", "url": url, "success": True})
            return data
        self._history.append({"role": "tool", "action": "GET", "url": url, "success": False})
        return f"Error: {res.get('error', 'unknown error')}"

    def post_fhir_resource(self, url: str, payload: str) -> str:
        """Execute a POST request to create a FHIR resource.

        Use this to create observations (vitals), medication requests,
        or service requests (lab orders, referrals).

        Args:
            url: The FHIR endpoint URL, e.g. ``{fhir_base}Observation``
            payload: A JSON string containing the FHIR resource to create.

        Returns:
            A confirmation message or error.
        """
        self._step_count += 1
        self._post_count += 1
        try:
            json.loads(payload) if isinstance(payload, str) else payload
        except (json.JSONDecodeError, TypeError):
            self._history.append({"role": "tool", "action": "POST", "url": url, "success": False})
            return "Error: Invalid JSON payload"
        self._history.append({
            "role": "tool", "action": "POST", "url": url,
            "payload": payload if isinstance(payload, str) else json.dumps(payload),
            "success": True,
        })
        return "POST request accepted and executed successfully."

    def finish(self, answers: str) -> str:
        """Call this when you have completed all tasks and have final answers.

        Args:
            answers: A JSON-loadable list of answers, e.g. ``[42]`` or ``["Patient not found"]``

        Returns:
            Confirmation that the task is finished.
        """
        self._finished = True
        self._finish_result = answers
        self._step_count += 1
        return f"Task finished with answers: {answers}"
