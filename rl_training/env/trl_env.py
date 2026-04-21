"""TRL-compatible environment for MedAgentBench GRPO training.

When passed as ``environment_factory`` to ``GRPOTrainer``, the trainer
instantiates one ``MedAgentBenchEnv`` per rollout slot.  Public methods
(excluding ``reset``) are automatically exposed as tools that the model
can call via native tool-calling.

``reset()`` is called before each generation batch; TRL forwards dataset row
fields as keyword arguments, so ``reset`` must accept ``**kwargs``. The
forwarded fields (``task_id``, ``eval_MRN``, ``instruction``, ``context``) are
captured into ``self._task`` so the clinical reward functions can inspect them.

Snapshot mode:
    When ``FhirSnapshot`` is installed (via the module-level
    ``install_global_snapshot`` helper in ``rl_training.env.fhir_snapshot``),
    all GETs are routed through the snapshot, falling through to live FHIR only
    on cache miss (if the snapshot allows). This makes GRPO rollouts
    deterministic and independent of network latency once the snapshot is built.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import requests

from rl_training.env.fhir_snapshot import FhirSnapshot, get_global_snapshot


def _get_fhir_base() -> str:
    return os.environ.get("FHIR_API_BASE", "http://localhost:8080/fhir/")


def _send_get_live(url: str) -> dict[str, Any]:
    """Hit the FHIR server directly, ignoring any snapshot."""
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        ct = resp.headers.get("Content-Type", "")
        data = resp.json() if "json" in ct else resp.text
        return {"status_code": resp.status_code, "data": data}
    except Exception as e:
        return {"error": str(e)}


# Regex to extract ISO-8601 timestamps from FHIR response payloads. We scan
# responses once per GET and record the timestamps we find so the temporal
# reward can cheaply check in-window citation later.
_ISO_RE = re.compile(
    r"\b(\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?(?:[+-]\d{2}:?\d{2}|Z)?)?)\b"
)


class MedAgentBenchEnv:
    """Stateful FHIR environment exposed as TRL tools.

    Public methods become tools available to the model during GRPO rollouts.

    Args:
        snapshot: optional ``FhirSnapshot`` to route GETs through. When
            ``None``, falls back to the module-level global snapshot
            (if installed), otherwise hits FHIR live.
    """

    def __init__(self, snapshot: FhirSnapshot | None = None):
        self.fhir_base = _get_fhir_base()
        self._snapshot = snapshot
        self._history: list[dict[str, str]] = []
        self._tool_log: list[dict[str, Any]] = []
        self._finished = False
        self._finish_result: str | None = None
        self._post_count = 0
        self._get_count = 0
        self._step_count = 0
        self._task: dict[str, Any] = {}

    # ------------------------------------------------------------ lifecycle

    def reset(self, **kwargs: Any) -> None:
        """Reset state between rollouts.

        TRL passes each dataset row as keyword args (``prompt``, ``task_id``,
        ``instruction``, ``context``, ``eval_MRN``, etc.). We capture the
        task-identifying fields into ``self._task`` so clinical rewards can
        look them up from the ``environments`` list.
        """
        self._history = []
        self._tool_log = []
        self._finished = False
        self._finish_result = None
        self._post_count = 0
        self._get_count = 0
        self._step_count = 0

        merged: dict[str, Any] = {}
        ref_raw = kwargs.get("ref_task_json")
        if isinstance(ref_raw, str) and ref_raw.strip():
            try:
                merged = json.loads(ref_raw)
            except json.JSONDecodeError:
                merged = {}
        self._task = {
            "id": kwargs.get("task_id", merged.get("id", "")),
            "eval_MRN": kwargs.get("eval_MRN", merged.get("eval_MRN", "")),
            "instruction": kwargs.get("instruction", merged.get("instruction", "")),
            "context": kwargs.get("context", merged.get("context", "")),
        }
        for k, v in merged.items():
            self._task.setdefault(k, v)

    # -------------------------------------------------------------- helpers

    def _route_get(self, url: str) -> dict[str, Any]:
        """Route a GET through snapshot (instance > global > live)."""
        snap = self._snapshot or get_global_snapshot()
        if snap is not None:
            return snap.send_get_request(url)
        return _send_get_live(url)

    # ------------------------------------------------------------- tools

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

        res = self._route_get(url)

        if "data" in res:
            data = res["data"]
            if not isinstance(data, str):
                data_text = json.dumps(data, indent=2)
            else:
                data_text = data
            timestamps = _extract_timestamps(data_text)
            self._tool_log.append({
                "step": self._step_count,
                "action": "GET",
                "url": url,
                "success": True,
                "timestamps": timestamps,
                "response_len": len(data_text),
            })
            self._history.append({
                "role": "tool", "action": "GET", "url": url, "success": True,
            })
            return data_text

        self._tool_log.append({
            "step": self._step_count,
            "action": "GET",
            "url": url,
            "success": False,
            "timestamps": [],
            "response_len": 0,
            "error": res.get("error", ""),
        })
        self._history.append({
            "role": "tool", "action": "GET", "url": url, "success": False,
        })
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
            self._tool_log.append({
                "step": self._step_count,
                "action": "POST",
                "url": url,
                "success": False,
                "timestamps": [],
                "response_len": 0,
                "error": "invalid json payload",
            })
            self._history.append({
                "role": "tool", "action": "POST", "url": url, "success": False,
            })
            return "Error: Invalid JSON payload"
        payload_text = payload if isinstance(payload, str) else json.dumps(payload)
        self._tool_log.append({
            "step": self._step_count,
            "action": "POST",
            "url": url,
            "success": True,
            "timestamps": _extract_timestamps(payload_text),
            "response_len": len(payload_text),
            "payload": payload_text,
        })
        self._history.append({
            "role": "tool", "action": "POST", "url": url,
            "payload": payload_text,
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
        self._tool_log.append({
            "step": self._step_count,
            "action": "FINISH",
            "url": "",
            "success": True,
            "timestamps": [],
            "response_len": len(answers or ""),
            "answers": answers,
        })
        return f"Task finished with answers: {answers}"


def _extract_timestamps(text: str) -> list[str]:
    """Find ISO-8601 timestamps in a FHIR response body.

    Used by temporal_grounding_reward to check whether the evidence the
    rollout surfaced is inside the task's ``now - window`` range.
    """
    if not text:
        return []
    seen: list[str] = []
    seen_set: set[str] = set()
    for match in _ISO_RE.finditer(text):
        ts = match.group(1)
        if ts not in seen_set:
            seen_set.add(ts)
            seen.append(ts)
        if len(seen) >= 64:
            break
    return seen
