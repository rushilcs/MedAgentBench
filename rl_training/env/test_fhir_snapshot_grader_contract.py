"""Pin the contract that ``FhirSnapshot.send_get_request`` returns ``data``
as a JSON-decodable *string*, matching the original
``src.server.tasks.medagentbench.utils.send_get_request`` contract that
refsol graders rely on (``json.loads(send_get_request(url)['data'])``).

Regression context: a loose ``"json" in Content-Type`` check in
``_default_live_getter`` caused snapshot rows to be cached as parsed dicts.
Refsol then raised ``TypeError: the JSON object must be str, ...``,
silently caught by the Evaluator's bare ``except``, collapsing every
FHIR-querying task (2, 4-7, 9, 10) to 0% in eval.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from rl_training.env.fhir_snapshot import FhirSnapshot, _coerce_data_to_str


def _write_snapshot_jsonl(rows: list[dict]) -> str:
    """Write rows to a temporary JSONL and return its path."""
    fp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False,
    )
    for row in rows:
        fp.write(json.dumps(row) + "\n")
    fp.close()
    return fp.name


def test_coerce_helper_handles_dict_list_str() -> None:
    assert _coerce_data_to_str({"a": 1}) == '{"a": 1}'
    assert _coerce_data_to_str([1, 2, 3]) == "[1, 2, 3]"
    assert _coerce_data_to_str("already a string") == "already a string"
    assert _coerce_data_to_str(None) is None


def test_replay_returns_string_for_dict_cached_data() -> None:
    """Cached row stored as parsed dict must still come back as string."""
    url = "http://localhost:8080/fhir/Patient?identifier=S1"
    bundle = {
        "resourceType": "Bundle",
        "entry": [{"resource": {"resourceType": "Patient", "id": "S1"}}],
    }
    path = _write_snapshot_jsonl([
        {"url": url, "status_code": 200, "data": bundle},
    ])
    snap = FhirSnapshot(mode="replay", path=path, fallthrough=False)
    res = snap.send_get_request(url)

    assert "data" in res, f"unexpected res shape: {res}"
    assert isinstance(res["data"], str), (
        f"data must be str for refsol; got {type(res['data']).__name__}"
    )
    parsed = json.loads(res["data"])
    assert parsed == bundle

    Path(path).unlink(missing_ok=True)


def test_replay_returns_string_for_str_cached_data() -> None:
    """Already-string cached rows must pass through unchanged."""
    url = "http://localhost:8080/fhir/Patient?identifier=S2"
    serialized = json.dumps({"resourceType": "Bundle", "entry": []})
    path = _write_snapshot_jsonl([
        {"url": url, "status_code": 200, "data": serialized},
    ])
    snap = FhirSnapshot(mode="replay", path=path, fallthrough=False)
    res = snap.send_get_request(url)

    assert isinstance(res["data"], str)
    assert res["data"] == serialized

    Path(path).unlink(missing_ok=True)


def test_refsol_pattern_round_trips() -> None:
    """Reproduce the exact pattern refsol uses; must not raise TypeError."""
    # Use the canonical (alphabetically-sorted) param order: snapshot rows
    # are stored under their canonical key, and lookup re-canonicalizes the
    # input URL.
    url = "http://localhost:8080/fhir/Observation?code=MG&patient=S1"
    bundle = {
        "resourceType": "Bundle",
        "entry": [{
            "resource": {
                "resourceType": "Observation",
                "effectiveDateTime": "2023-11-13T10:00:00+00:00",
                "valueQuantity": {"value": 2.1},
            },
        }],
    }
    path = _write_snapshot_jsonl([
        {"url": url, "status_code": 200, "data": bundle},
    ])
    snap = FhirSnapshot(mode="replay", path=path, fallthrough=False)

    res = snap.send_get_request(url)
    get_res = json.loads(res["data"])
    assert get_res["entry"][0]["resource"]["valueQuantity"]["value"] == 2.1

    Path(path).unlink(missing_ok=True)


if __name__ == "__main__":
    test_coerce_helper_handles_dict_list_str()
    test_replay_returns_string_for_dict_cached_data()
    test_replay_returns_string_for_str_cached_data()
    test_refsol_pattern_round_trips()
    print("OK: all FhirSnapshot grader-contract tests passed")
