#!/usr/bin/env python3
"""Seed the local docker FHIR with Observations on v2-training MRNs so that
the v3 SFT corpus can include realistic trajectories for under-represented
branches.

Background
----------
The MedAgentBench fixture was hand-crafted so that the **test** patients have
the labs we want to evaluate (K for task9, A1C for task10, MG for task5),
but the rest of the patient pool is empty for those codes. The v2 corpus
was therefore trained almost exclusively on the "no data, FINISH([])"
branch, while the test set exercises the "see real data, decide" branch.

This script POSTs synthetic Observations to a configurable count of
non-test patients so that v3 trajectories can teach the missing branches.

Constraints
-----------
* Never seeds against a test MRN (read from ``data/medagentbench/test_data_v2.json``).
* Uses canonical LOINC codes (K, A1C, MG) with sensible value distributions.
* Per-patient query filters in refsol mean these seeded obs CANNOT
  contaminate test grading.
* Re-runnable: writes a manifest of (mrn, code, value, effectiveDateTime)
  so we can verify the seed is in place.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


# Cutoff used by every grader. Seeded observations are timestamped relative
# to this so the time-window logic in refsol task4/task5/task6/task10
# behaves predictably.
CUTOFF = datetime.fromisoformat("2023-11-13T10:15:00+00:00")


def _post_obs(
    fhir_base: str,
    mrn: str,
    code: str,
    value: float,
    unit: str,
    effective_dt: datetime,
    *, dry_run: bool,
) -> dict | None:
    """POST a single Observation. Returns the resource dict on success."""
    payload = {
        "resourceType": "Observation",
        "status": "final",
        "code": {
            "coding": [
                {"system": "http://loinc.org", "code": code, "display": code},
            ],
            "text": code,
        },
        "subject": {"reference": f"Patient/{mrn}"},
        "effectiveDateTime": effective_dt.isoformat().replace("+00:00", "+00:00"),
        "valueQuantity": {
            "value": value,
            "unit": unit,
            "system": "http://unitsofmeasure.org",
        },
    }
    if dry_run:
        return {"dry_run": True, **payload}
    r = requests.post(
        fhir_base.rstrip("/") + "/Observation",
        json=payload,
        headers={"Content-Type": "application/fhir+json"},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def _ensure_patient(fhir_base: str, mrn: str, *, dry_run: bool) -> bool:
    """Make sure a Patient resource exists for ``mrn``. Returns True if it now does."""
    if dry_run:
        return True
    # Check first (cheap) -- the v2 training MRNs were already created by
    # the original seed script.
    r = requests.get(
        fhir_base.rstrip("/") + f"/Patient?identifier={mrn}&_format=json",
        timeout=15,
    )
    if r.status_code == 200 and r.json().get("total", 0) > 0:
        return True
    # Create with the MRN as a logical id.
    payload = {
        "resourceType": "Patient",
        "id": mrn,
        "identifier": [{"system": "MRN", "value": mrn}],
    }
    p = requests.put(
        fhir_base.rstrip("/") + f"/Patient/{mrn}",
        json=payload,
        headers={"Content-Type": "application/fhir+json"},
        timeout=30,
    )
    p.raise_for_status()
    return True


# ---------------------------------------------------- per-task seed plans

def _seed_task9_normal_k(rng: random.Random, fhir_base: str, mrns: list[str], *, dry_run: bool) -> list[dict]:
    """Each MRN gets a single K observation in the normal range [3.6, 5.0]."""
    out = []
    for mrn in mrns:
        v = round(rng.uniform(3.6, 5.0), 1)
        # Make the obs *recent* (within last week before cutoff) so it'd be
        # the latest reading.
        eff = CUTOFF - timedelta(hours=rng.randint(2, 168))
        if not _ensure_patient(fhir_base, mrn, dry_run=dry_run):
            continue
        _post_obs(fhir_base, mrn, "K", v, "mEq/L", eff, dry_run=dry_run)
        out.append({"mrn": mrn, "code": "K", "value": v, "effectiveDateTime": eff.isoformat()})
    return out


def _seed_task9_low_k(rng: random.Random, fhir_base: str, mrns: list[str], *, dry_run: bool) -> list[dict]:
    """Each MRN gets a single K observation in the low range [2.5, 3.4]."""
    out = []
    for mrn in mrns:
        v = round(rng.uniform(2.5, 3.4), 1)
        eff = CUTOFF - timedelta(hours=rng.randint(2, 72))
        if not _ensure_patient(fhir_base, mrn, dry_run=dry_run):
            continue
        _post_obs(fhir_base, mrn, "K", v, "mEq/L", eff, dry_run=dry_run)
        out.append({"mrn": mrn, "code": "K", "value": v, "effectiveDateTime": eff.isoformat()})
    return out


def _seed_task10_fresh_a1c(rng: random.Random, fhir_base: str, mrns: list[str], *, dry_run: bool) -> list[dict]:
    """Each MRN gets an A1C observation within the last year, value 5.0-9.5%."""
    out = []
    for mrn in mrns:
        v = round(rng.uniform(5.0, 9.5), 1)
        eff = CUTOFF - timedelta(days=rng.randint(7, 350))
        if not _ensure_patient(fhir_base, mrn, dry_run=dry_run):
            continue
        _post_obs(fhir_base, mrn, "A1C", v, "%", eff, dry_run=dry_run)
        out.append({"mrn": mrn, "code": "A1C", "value": v, "effectiveDateTime": eff.isoformat()})
    return out


def _seed_task10_stale_a1c(rng: random.Random, fhir_base: str, mrns: list[str], *, dry_run: bool) -> list[dict]:
    """Each MRN gets an A1C observation > 1 year old, so order is required."""
    out = []
    for mrn in mrns:
        v = round(rng.uniform(5.0, 9.5), 1)
        eff = CUTOFF - timedelta(days=rng.randint(400, 900))
        if not _ensure_patient(fhir_base, mrn, dry_run=dry_run):
            continue
        _post_obs(fhir_base, mrn, "A1C", v, "%", eff, dry_run=dry_run)
        out.append({"mrn": mrn, "code": "A1C", "value": v, "effectiveDateTime": eff.isoformat()})
    return out


def _seed_task5_low_mg(rng: random.Random, fhir_base: str, mrns: list[str], *, dry_run: bool) -> list[dict]:
    """Each MRN gets a Mg observation in the low range [0.8, 1.49] within last 24h."""
    out = []
    for mrn in mrns:
        v = round(rng.uniform(0.8, 1.49), 2)
        eff = CUTOFF - timedelta(hours=rng.randint(1, 23))
        if not _ensure_patient(fhir_base, mrn, dry_run=dry_run):
            continue
        _post_obs(fhir_base, mrn, "MG", v, "mg/dL", eff, dry_run=dry_run)
        out.append({"mrn": mrn, "code": "MG", "value": v, "effectiveDateTime": eff.isoformat()})
    return out


def _seed_task5_high_mg(rng: random.Random, fhir_base: str, mrns: list[str], *, dry_run: bool) -> list[dict]:
    """Each MRN gets a Mg observation in the normal/high range [1.95, 2.6] within last 24h."""
    out = []
    for mrn in mrns:
        v = round(rng.uniform(1.95, 2.6), 2)
        eff = CUTOFF - timedelta(hours=rng.randint(1, 23))
        if not _ensure_patient(fhir_base, mrn, dry_run=dry_run):
            continue
        _post_obs(fhir_base, mrn, "MG", v, "mg/dL", eff, dry_run=dry_run)
        out.append({"mrn": mrn, "code": "MG", "value": v, "effectiveDateTime": eff.isoformat()})
    return out


# Map each (task, branch) to (seeder_fn, target_count) and a per-branch
# disjoint MRN slice. The slices are computed at runtime from the v2
# training task list, partitioned so no MRN is reused across branches
# (otherwise the second seed wins and the first branch silently disappears).
_SEED_PLAN: list[tuple[str, str, callable, int]] = [
    ("task9", "normal_k", _seed_task9_normal_k, 200),
    ("task9", "low_k", _seed_task9_low_k, 25),
    ("task10", "fresh_a1c", _seed_task10_fresh_a1c, 100),
    ("task10", "stale_a1c", _seed_task10_stale_a1c, 50),
    ("task5", "low_mg", _seed_task5_low_mg, 25),
    ("task5", "high_mg", _seed_task5_high_mg, 25),
]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fhir-base", default="http://localhost:8080/fhir/")
    p.add_argument(
        "--v2-tasks",
        default="rl_training/outputs/qwen_pipeline_v3/phase_a/training_tasks_v2.json",
    )
    p.add_argument(
        "--test-data", default="data/medagentbench/test_data_v2.json",
    )
    p.add_argument(
        "--manifest-out",
        default="rl_training/outputs/qwen_pipeline_v3/phase_a/seed_manifest.json",
    )
    p.add_argument("--seed", type=int, default=20260424)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    rng = random.Random(args.seed)

    test_mrns = {
        r["eval_MRN"] for r in json.loads(Path(args.test_data).read_text())
        if "eval_MRN" in r
    }
    v2_train_tasks = json.loads(Path(args.v2_tasks).read_text())
    v2_train_mrns_set = {
        t["eval_MRN"] for t in v2_train_tasks
        if "eval_MRN" in t and t["eval_MRN"] not in test_mrns
    }

    # 106 v2-train MRNs is too few once we partition K across branches
    # (200 normal + 25 low > 106). Pull the full training-eligible pool
    # from FHIR (every Patient minus test_mrns) so we get ~400 MRNs.
    logger.info("Fetching full FHIR Patient pool to expand training MRNs...")
    all_mrns: set[str] = set(v2_train_mrns_set)
    next_url = args.fhir_base.rstrip("/") + "/Patient?_count=500&_format=json&_elements=identifier"
    while next_url:
        resp = requests.get(next_url, timeout=30).json()
        for ent in resp.get("entry", []):
            res = ent.get("resource", {})
            for ident in res.get("identifier", []):
                v = ident.get("value")
                if v and v not in test_mrns:
                    all_mrns.add(v)
            # Some Patient resources only have logical id, no identifier.
            rid = res.get("id")
            if rid and rid not in test_mrns:
                all_mrns.add(rid)
        next_url = next(
            (l["url"] for l in resp.get("link", []) if l.get("relation") == "next"),
            None,
        )
    v2_train_mrns = sorted(all_mrns)
    rng.shuffle(v2_train_mrns)
    logger.info(
        "%d total training-eligible MRNs available for seeding "
        "(test MRNs excluded; v2-train subset = %d)",
        len(v2_train_mrns), len(v2_train_mrns_set),
    )

    # Partition MRNs across (task, branch) slots so no MRN gets multiple
    # observations for the same code (would silently break the seed plan).
    # Different codes on the same MRN are fine -- task graders filter by
    # code, so e.g. K + MG on one patient is independent.
    by_code_assignment: dict[str, list[str]] = {"K": [], "A1C": [], "MG": []}
    needed_per_code: dict[str, int] = {"K": 0, "A1C": 0, "MG": 0}
    code_for_branch: dict[tuple[str, str], str] = {
        ("task9", "normal_k"): "K",
        ("task9", "low_k"): "K",
        ("task10", "fresh_a1c"): "A1C",
        ("task10", "stale_a1c"): "A1C",
        ("task5", "low_mg"): "MG",
        ("task5", "high_mg"): "MG",
    }
    for tid, branch, _, n in _SEED_PLAN:
        needed_per_code[code_for_branch[(tid, branch)]] += n

    if any(needed_per_code[c] > len(v2_train_mrns) for c in needed_per_code):
        logger.warning(
            "needed per code: %s; only %d MRNs available -- some MRNs "
            "will receive multiple observations for the same code (latest "
            "wins, so the per-(task,branch) split must be disjoint).",
            needed_per_code, len(v2_train_mrns),
        )

    # Build per-(task,branch) slices. For each code, allocate a disjoint
    # range of MRNs across its branches (e.g. for K: first 200 for normal,
    # next 25 for low). Different codes can re-use the same MRNs.
    slices: dict[tuple[str, str], list[str]] = {}
    cursor_by_code: dict[str, int] = {"K": 0, "A1C": 0, "MG": 0}
    for tid, branch, _, n in _SEED_PLAN:
        code = code_for_branch[(tid, branch)]
        start = cursor_by_code[code]
        end = start + n
        if end > len(v2_train_mrns):
            logger.warning(
                "%s/%s wants %d MRNs starting at %d; only %d available, "
                "truncating", tid, branch, n, start, len(v2_train_mrns),
            )
            end = len(v2_train_mrns)
        slices[(tid, branch)] = v2_train_mrns[start:end]
        cursor_by_code[code] = end

    manifest = {
        "fhir_base": args.fhir_base,
        "seed": args.seed,
        "test_mrn_count": len(test_mrns),
        "v2_train_mrn_count": len(v2_train_mrns),
        "needed_per_code": needed_per_code,
        "seeded": {},
    }

    for tid, branch, fn, n in _SEED_PLAN:
        mrns = slices[(tid, branch)]
        logger.info("seeding %s/%s on %d MRNs", tid, branch, len(mrns))
        records = fn(rng, args.fhir_base, mrns, dry_run=args.dry_run)
        manifest["seeded"][f"{tid}/{branch}"] = {
            "count": len(records), "mrn_examples": mrns[:5],
        }
        # Sanity: any of the seeded MRNs leak into test?
        leak = [m for m in mrns if m in test_mrns]
        if leak:
            logger.error("LEAK: %s/%s contains %d test MRNs", tid, branch, len(leak))
            manifest["leak"] = leak
            return 2

    Path(args.manifest_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.manifest_out).write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
