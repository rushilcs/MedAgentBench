"""Audit ``data/medagentbench/test_data_v2.json`` against the current live FHIR.

For tasks 4, 5, 6, 7, 9, 10 the refsol grader recomputes ``ref_sol`` from
live FHIR at grade time, so the *gold* itself moves with the FHIR fixture.
The drift we care about is therefore not "stored gold vs live" but
**"does this row currently exercise the branch its instruction implies?"**

Concrete examples:

  * ``task10`` rows with instructions that read like the patient *should* have
    a recent A1C ("report the last A1C and when it was recorded; if older
    than 1 year, order a new test") drift when FHIR returns *no* A1C: the
    row now silently flips to the "order needed" branch, the model is
    judged against ServiceRequest payload checks rather than against
    ``[value, time]``, and the previously-rehearsed FINISH shape becomes a
    grader-fail.

  * ``task1`` rows store an explicit ``sol`` (per-MRN expected attribute);
    drift here means live FHIR doesn't contain that MRN's expected value.

This script is **read-only** — it never POSTs and never writes to FHIR.
Output: a per-row JSON manifest plus a per-task summary, both saved under
``rl_training/outputs/qwen_pipeline_v3/phase_a/test_fixture_audit/``.

Requires the same FHIR routing as eval/training: the script reads
``FHIR_LIVE_BASE_OVERRIDE`` (e.g. a Cloudflare Tunnel pointing at the dev
FHIR) and falls back to ``http://localhost:8080/fhir`` if not set.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# Reuse the canonical live getter so cache keys, content-type handling,
# and error semantics match the trainer / eval exactly.
from rl_training.env.fhir_snapshot import _default_live_getter, SnapshotLiveUnavailable

CUTOFF = datetime.fromisoformat("2023-11-13T10:15:00+00:00")
ONE_YEAR = timedelta(days=365)
ONE_DAY = timedelta(hours=24)


# ----------------------------------------------------------------- helpers

def _get_json(url: str) -> dict | None:
    """GET ``url`` from live FHIR. Returns parsed JSON or None on infra failure."""
    try:
        res = _default_live_getter(url)
    except SnapshotLiveUnavailable as exc:
        print(f"  ! live FHIR unreachable: {exc}", file=sys.stderr)
        return None
    if "data" not in res:
        return None
    data = res["data"]
    if isinstance(data, str):
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            return None
    return data


def _latest_obs(fhir_base: str, mrn: str, code: str) -> tuple[Any, str | None]:
    """Return (value, effective_time_iso) of the latest obs, or (None, None)."""
    url = f"{fhir_base}Observation?patient={mrn}&code={code}&_count=5000&_format=json"
    data = _get_json(url)
    if not data:
        return None, None
    last_dt, last_val, last_iso = None, None, None
    for entry in data.get("entry", []):
        try:
            iso = entry["resource"]["effectiveDateTime"]
            dt = datetime.fromisoformat(iso)
            v = entry["resource"]["valueQuantity"]["value"]
        except (KeyError, ValueError):
            continue
        if last_dt is None or dt > last_dt:
            last_dt, last_val, last_iso = dt, v, iso
    return last_val, last_iso


def _latest_obs_within_24h(fhir_base: str, mrn: str, code: str) -> Any:
    """Latest obs in last 24h before cutoff, or None."""
    url = f"{fhir_base}Observation?patient={mrn}&code={code}&_count=5000&_format=json"
    data = _get_json(url)
    if not data:
        return None
    last_dt, last_val = None, None
    for entry in data.get("entry", []):
        try:
            iso = entry["resource"]["effectiveDateTime"]
            dt = datetime.fromisoformat(iso)
            v = entry["resource"]["valueQuantity"]["value"]
        except (KeyError, ValueError):
            continue
        if dt >= (CUTOFF - ONE_DAY) and (last_dt is None or dt > last_dt):
            last_dt, last_val = dt, v
    return last_val


def _avg_obs_within_24h(fhir_base: str, mrn: str, code: str) -> float | None:
    url = f"{fhir_base}Observation?patient={mrn}&code={code}&_count=5000&_format=json"
    data = _get_json(url)
    if not data:
        return None
    s, n = 0.0, 0
    for entry in data.get("entry", []):
        try:
            iso = entry["resource"]["effectiveDateTime"]
            dt = datetime.fromisoformat(iso)
            v = entry["resource"]["valueQuantity"]["value"]
        except (KeyError, ValueError):
            continue
        if dt >= (CUTOFF - ONE_DAY):
            s += v
            n += 1
    return (s / n) if n else None


# ------------------------------------------------------ per-task auditors


def audit_task1(row: dict, fhir_base: str) -> dict:
    """task1 stores a per-MRN expected attribute in row['sol']. Compare against
    live FHIR's Patient resource (whatever attribute the instruction asks for
    -- we don't try to disambiguate, just record presence)."""
    mrn = row.get("eval_MRN")
    if not mrn:
        return {"branch": "no_mrn"}
    data = _get_json(f"{fhir_base}Patient?identifier={mrn}&_format=json")
    if not data:
        return {"branch": "infra_error", "stored_sol": row.get("sol")}
    has_entry = bool(data.get("entry"))
    return {
        "branch": "patient_found" if has_entry else "patient_missing",
        "stored_sol": row.get("sol"),
    }


def audit_task4(row: dict, fhir_base: str) -> dict:
    """task4: latest MG (Mg) value within last 24h, else -1."""
    val = _latest_obs_within_24h(fhir_base, row["eval_MRN"], "MG")
    return {
        "branch": "data_absent" if val is None else "data_present",
        "live_ref_sol": [val if val is not None else -1],
    }


def audit_task5(row: dict, fhir_base: str) -> dict:
    """task5: branch on latest Mg in last 24h. None=>no order, >1.9=>no order,
    else=>MedicationRequest required."""
    val = _latest_obs_within_24h(fhir_base, row["eval_MRN"], "MG")
    if val is None:
        branch = "data_absent_no_order"
    elif val > 1.9:
        branch = "high_no_order"
    else:
        branch = "low_order_required"
    return {"branch": branch, "live_latest_24h": val}


def audit_task6(row: dict, fhir_base: str) -> dict:
    """task6: average GLU within last 24h, else -1."""
    avg = _avg_obs_within_24h(fhir_base, row["eval_MRN"], "GLU")
    return {
        "branch": "data_absent" if avg is None else "data_present",
        "live_ref_sol_approx": [avg if avg is not None else -1],
    }


def audit_task7(row: dict, fhir_base: str) -> dict:
    """task7: latest GLU value (no time window)."""
    val, _ = _latest_obs(fhir_base, row["eval_MRN"], "GLU")
    return {
        "branch": "data_absent" if val is None else "data_present",
        "live_ref_sol_approx": [val if val is not None else -1],
    }


def audit_task9(row: dict, fhir_base: str) -> dict:
    """task9: branch on latest K. None or >=3.5 => no order, else => order."""
    val, _ = _latest_obs(fhir_base, row["eval_MRN"], "K")
    if val is None:
        branch = "data_absent_no_order"
    elif val >= 3.5:
        branch = "normal_no_order"
    else:
        branch = "low_order_required"
    return {"branch": branch, "live_latest_K": val}


def audit_task10(row: dict, fhir_base: str) -> dict:
    """task10: branch on latest A1C. None => order. Older than 1y => order.
    Else => report [value, time] (no order)."""
    val, iso = _latest_obs(fhir_base, row["eval_MRN"], "A1C")
    if val is None:
        branch = "data_absent_order_required"
        ref_sol = [-1]
    else:
        last_dt = datetime.fromisoformat(iso)
        if last_dt < (CUTOFF - ONE_YEAR):
            branch = "stale_order_required"
        else:
            branch = "fresh_report_value"
        ref_sol = [val, iso]
    return {"branch": branch, "live_ref_sol": ref_sol, "live_last_iso": iso}


_AUDITORS = {
    "task1": audit_task1,
    "task4": audit_task4,
    "task5": audit_task5,
    "task6": audit_task6,
    "task7": audit_task7,
    "task9": audit_task9,
    "task10": audit_task10,
}


# --------------------------------------------------------------- entrypoint


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--test", default="data/medagentbench/test_data_v2.json", type=Path,
    )
    ap.add_argument(
        "--out-dir",
        default="rl_training/outputs/qwen_pipeline_v3/phase_a/test_fixture_audit",
        type=Path,
    )
    ap.add_argument(
        "--fhir-base", default="http://localhost:8080/fhir/",
        help="Logical FHIR base used in URLs. The FHIR_LIVE_BASE_OVERRIDE env "
             "var rewrites localhost:8080/fhir on the wire, so leave this as "
             "the canonical localhost form.",
    )
    args = ap.parse_args()

    if "FHIR_LIVE_BASE_OVERRIDE" not in os.environ:
        print(
            "WARNING: FHIR_LIVE_BASE_OVERRIDE is not set. Audit will hit "
            "http://localhost:8080/fhir directly; this only works if you "
            "have a local docker FHIR running. To audit against the dev "
            "FHIR via Cloudflare Tunnel, set FHIR_LIVE_BASE_OVERRIDE first.",
            file=sys.stderr,
        )

    rows = json.loads(args.test.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)

    per_row: list[dict] = []
    branch_counts: dict[str, Counter] = defaultdict(Counter)
    infra_errors = 0
    for i, row in enumerate(rows, 1):
        tid = row["id"].split("_")[0]
        auditor = _AUDITORS.get(tid)
        if auditor is None:
            # task2/3/8 don't compute ref_sol live; nothing meaningful to audit.
            per_row.append({"id": row["id"], "task": tid, "branch": "n/a"})
            branch_counts[tid]["n/a"] += 1
            continue
        try:
            info = auditor(row, args.fhir_base)
        except Exception as exc:
            info = {"branch": "auditor_exception", "error": repr(exc)}
        if info.get("branch") in ("infra_error", "auditor_exception"):
            infra_errors += 1
        rec = {"id": row["id"], "task": tid, **info}
        per_row.append(rec)
        branch_counts[tid][info.get("branch", "?")] += 1
        if i % 25 == 0:
            print(f"  audited {i}/{len(rows)}", flush=True)

    summary = {
        "n_rows": len(rows),
        "infra_errors": infra_errors,
        "fhir_base": args.fhir_base,
        "fhir_override": os.environ.get("FHIR_LIVE_BASE_OVERRIDE", ""),
        "branch_distribution_per_task": {
            tid: dict(c) for tid, c in sorted(branch_counts.items())
        },
        "audited_at": datetime.utcnow().isoformat() + "Z",
    }

    (args.out_dir / "audit_per_row.json").write_text(
        json.dumps(per_row, indent=2) + "\n"
    )
    (args.out_dir / "audit_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))
    if infra_errors:
        print(
            f"\nWARNING: {infra_errors} rows hit FHIR infra errors during "
            "audit. Re-run after the tunnel is healthy.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
