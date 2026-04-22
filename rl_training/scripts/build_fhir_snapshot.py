#!/usr/bin/env python3
"""Pre-record FHIR GET responses for training tasks (exhaustive coverage).

Given a training-tasks JSON file, this script iterates each task and exercises
EVERY plausible FHIR GET URL the model could issue, caching each response to a
JSONL snapshot. At training time, ``FhirSnapshot(mode="replay")`` serves those
responses from memory so rollouts are deterministic and never depend on a live
server.

Why this matters
----------------
The reward signal in GRPO can only fire if the model's GETs return real data.
Any URL that misses the snapshot (when ``snapshot_fallthrough=false``, which is
required for cost-safe training) silently zero-rewards the rollout. Early runs
consistently logged ``avg_correct = 0.0`` because the snapshot only covered the
narrow URL shapes ``refsol.taskN`` happened to issue, while the model would
naturally try other valid forms (e.g. ``Patient?name=...&birthdate=...``,
``Observation?subject=Patient/...``, omitted ``_count``, etc.).

This builder enumerates every shape we have evidence the model emits, plus
canonical variants (host, query order, ``_format``) covered automatically by
``FhirSnapshot._canonicalize_url``. The result is a single ``fhir_snapshot.jsonl``
that is the complete environment.

Strategy per task type (see ``src/server/tasks/medagentbench/refsol.py``):
  - task1:  name + DOB lookup (12 query shapes per task, plus a global scan)
  - task2:  Patient identifier + age math (refsol uses Patient?identifier=...)
  - task3:  POST Observation; model often pre-verifies patient
  - task4..7,9,10: Observation queries (multiple shapes per code)
  - task8:  POST ServiceRequest; model often pre-verifies patient

All URLs canonicalize identically across host/format/param-order variants, so
recording one canonical form per logical query is sufficient.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import quote

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from rl_training.env.fhir_snapshot import FhirSnapshot

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------- per-task spec

# Maps task-type -> (simple text code, [LOINC code variants]).
# The benchmark's FHIR fixture indexes observations by simple codes ("MG", "GLU",
# ...), but the model frequently emits LOINC; recording both so a LOINC query
# returns a (typically empty) Bundle rather than a snapshot miss.
_OBSERVATION_CODES: dict[int, list[str]] = {
    4:  ["MG", "19123-9", "2601-3"],                   # Magnesium
    5:  ["MG", "19123-9", "2601-3"],
    6:  ["GLU", "2345-7", "2339-0", "41653-7"],         # Glucose / CBG
    7:  ["GLU", "2345-7", "2339-0", "41653-7"],
    9:  ["K", "2823-3", "6298-4"],                      # Potassium
    10: ["A1C", "4548-4", "17856-6", "41995-2"],        # HbA1C
}

_TASK1_RE = re.compile(
    r"name\s+([A-Za-z][A-Za-z\-'.\s]+?)\s+and\s+DOB\s+of\s+(\d{4}-\d{2}-\d{2})",
    re.IGNORECASE,
)


def _task_type(task_id: str) -> int | None:
    for part in task_id.split("_"):
        if part.startswith("task"):
            try:
                return int(part[len("task"):])
            except ValueError:
                return None
    return None


def _parse_task1(instruction: str) -> tuple[str, str, str] | None:
    """Return (full_name, first, last, dob) parsed from a task1 instruction.

    Returns ``(first_name, last_name, dob)``, or ``None`` if it doesn't match.
    """
    m = _TASK1_RE.search(instruction or "")
    if not m:
        return None
    full = m.group(1).strip()
    dob = m.group(2)
    parts = full.split()
    if len(parts) < 2:
        return None
    first = parts[0]
    last = " ".join(parts[1:])
    return (first, last, dob)


def _patient_lookup_urls(mrn: str, base: str) -> list[str]:
    """All shapes the model uses to fetch one patient by MRN."""
    if not mrn:
        return []
    return [
        f"{base}Patient?identifier={mrn}",          # refsol form
        f"{base}Patient?identifier={mrn}&_count=10",
        f"{base}Patient/{mrn}",                     # direct read
        f"{base}Patient?_id={mrn}",
        f"{base}Patient?identifier={mrn}&_count=1",
    ]


def _task1_urls(first: str, last: str, dob: str, base: str) -> list[str]:
    """Every plausible name/DOB query shape for task1."""
    fn = quote(first)
    ln = quote(last)
    full = quote(f"{first} {last}")  # encodes the space as %20
    full_plus = quote(f"{first} {last}", safe="+").replace("%20", "+")
    out = [
        f"{base}Patient?name={full}&birthdate={dob}",
        f"{base}Patient?name={full_plus}&birthdate={dob}",
        f"{base}Patient?given={fn}&family={ln}&birthdate={dob}",
        f"{base}Patient?family={ln}&given={fn}&birthdate={dob}",
        f"{base}Patient?family={ln}&birthdate={dob}",
        f"{base}Patient?given={fn}&birthdate={dob}",
        f"{base}Patient?birthdate={dob}",
        f"{base}Patient?name={full}",
        f"{base}Patient?name={full_plus}",
        f"{base}Patient?given={fn}&family={ln}",
        f"{base}Patient?family={ln}",
        f"{base}Patient?given={fn}",
        f"{base}Patient?name={ln}",
        f"{base}Patient?name={fn}",
        f"{base}Patient?name={full}&birthdate={dob}&_count=10",
        f"{base}Patient?given={fn}&family={ln}&birthdate={dob}&_count=10",
    ]
    return out


def _observation_urls(mrn: str, codes: list[str], base: str) -> list[str]:
    """Every observation-query shape for one (patient, code-family) pair."""
    if not mrn:
        return []
    urls: list[str] = []
    # Bare patient observations (no code filter).
    urls.append(f"{base}Observation?patient={mrn}")
    urls.append(f"{base}Observation?patient={mrn}&_count=5000")
    urls.append(f"{base}Observation?patient={mrn}&_count=100")
    urls.append(f"{base}Observation?subject=Patient/{mrn}")
    urls.append(f"{base}Observation?subject=Patient/{mrn}&_count=5000")
    for code in codes:
        urls.extend([
            f"{base}Observation?patient={mrn}&code={code}&_count=5000",  # refsol
            f"{base}Observation?patient={mrn}&code={code}",
            f"{base}Observation?patient={mrn}&code={code}&_count=100",
            f"{base}Observation?patient={mrn}&code={code}&_count=1000",
            f"{base}Observation?subject=Patient/{mrn}&code={code}",
            f"{base}Observation?subject=Patient/{mrn}&code={code}&_count=5000",
            f"{base}Observation?patient={mrn}&code={code}&_sort=-date",
            f"{base}Observation?patient={mrn}&code={code}&_sort=-date&_count=1",
        ])
    return urls


def _wildcard_urls(base: str) -> list[str]:
    """Cover the global Patient list scan (task1 fallback) at common page sizes."""
    return [
        f"{base}Patient",
        f"{base}Patient?_count=10",
        f"{base}Patient?_count=100",
        f"{base}Patient?_count=500",
        f"{base}Patient?_count=1000",
        f"{base}Patient?_count=5000",
    ]


def _task_urls(task: dict[str, Any], base: str) -> list[str]:
    """Compose the full URL list for one task."""
    task_id = task.get("id", "")
    mrn = task.get("eval_MRN", "")
    instr = task.get("instruction", "") or ""
    ttype = _task_type(task_id)

    urls: list[str] = []
    if ttype is None:
        return urls

    # Task1: parse name+DOB and emit the search variants.
    if ttype == 1:
        parsed = _parse_task1(instr)
        if parsed:
            first, last, dob = parsed
            urls.extend(_task1_urls(first, last, dob, base))
        # The model often falls back to a full scan; the wildcards are added
        # globally below, no need to repeat per-task.
        return urls

    # Tasks 2..10: patient lookup is universal.
    urls.extend(_patient_lookup_urls(mrn, base))

    # Tasks with observations.
    codes = _OBSERVATION_CODES.get(ttype)
    if codes:
        urls.extend(_observation_urls(mrn, codes, base))

    return urls


# ------------------------------------------------------------------------ runner


def _record_one(snap: FhirSnapshot, url: str) -> tuple[str, bool, str]:
    """Fetch one URL through the snapshot. Returns (url, ok, msg)."""
    try:
        res = snap.send_get_request(url)
        if "error" in res:
            return (url, False, res["error"])
        return (url, True, "")
    except Exception as exc:
        return (url, False, f"{type(exc).__name__}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build exhaustive FHIR snapshot")
    parser.add_argument("--tasks", required=True,
                        help="Path to training tasks JSON (or benchmark tasks JSON)")
    parser.add_argument("--fhir-base", default="http://localhost:8080/fhir/")
    parser.add_argument("--output", default="rl_training/outputs/fhir_snapshot.jsonl")
    parser.add_argument("--workers", type=int, default=8,
                        help="Concurrent HTTP workers (live FHIR can handle ~16)")
    parser.add_argument("--limit", type=int, default=None,
                        help="(Debug) Only process first N tasks")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    base = args.fhir_base
    if not base.endswith("/"):
        base += "/"

    with open(args.tasks) as f:
        tasks = json.load(f)
    if args.limit:
        tasks = tasks[: args.limit]
    logger.info("Loaded %d tasks from %s", len(tasks), args.tasks)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    snapshot = FhirSnapshot(mode="record", path=str(out_path), fallthrough=True)
    if out_path.exists():
        loaded = snapshot.load(str(out_path))
        logger.info("Existing snapshot found at %s; loaded %d entries (will append).",
                    out_path, loaded)

    # ----------------------------------------------------- collect urls
    all_urls: list[str] = []
    seen: set[str] = set()

    def _push(url: str) -> None:
        if url and url not in seen:
            seen.add(url)
            all_urls.append(url)

    for url in _wildcard_urls(base):
        _push(url)
    per_task_counts: list[int] = []
    for task in tasks:
        task_urls = _task_urls(task, base)
        per_task_counts.append(len(task_urls))
        for u in task_urls:
            _push(u)

    logger.info(
        "Generated %d unique URLs (avg %.1f per task, max %d).",
        len(all_urls),
        sum(per_task_counts) / max(1, len(per_task_counts)),
        max(per_task_counts) if per_task_counts else 0,
    )

    # ----------------------------------------------------- record
    t0 = time.time()
    ok_count = 0
    err_count = 0
    err_examples: list[tuple[str, str]] = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_record_one, snapshot, u): u for u in all_urls}
        for i, fut in enumerate(as_completed(futures), start=1):
            url, ok, msg = fut.result()
            if ok:
                ok_count += 1
            else:
                err_count += 1
                if len(err_examples) < 10:
                    err_examples.append((url, msg))
            if i % 250 == 0 or i == len(all_urls):
                rate = i / max(1e-9, time.time() - t0)
                logger.info(
                    "Recorded %d/%d (ok=%d err=%d) at %.1f req/s",
                    i, len(all_urls), ok_count, err_count, rate,
                )

    elapsed = time.time() - t0
    stats = snapshot.stats()
    logger.info(
        "Done in %.1fs: ok=%d err=%d. Cache: %d entries. Stats=%s",
        elapsed, ok_count, err_count, len(snapshot._cache), stats,  # noqa: SLF001
    )
    if err_examples:
        logger.warning("First %d errors:", len(err_examples))
        for u, m in err_examples:
            logger.warning("  %s -> %s", u, m)
    logger.info("Snapshot written to %s (%.1f MB)",
                out_path,
                out_path.stat().st_size / (1024 * 1024) if out_path.exists() else 0)


if __name__ == "__main__":
    main()
