#!/usr/bin/env python3
"""End-to-end smoke test for the GRPO snapshot + reward pipeline.

This is the gate that would have caught the 2026-04-21 wasted run: it
mirrors the import + monkeypatch order of ``train_grpo_32b.py`` and
asserts that

  1. The FHIR snapshot is the active interception (no live HTTP fires
     in ``--snapshot-only`` mode against the FHIR base host).
  2. For one task per type, ``MedAgentBenchEnv.get_fhir_resource`` and
     the ``refsol.taskN`` grader hit the same URL and both succeed.
  3. ``compute_episode_reward`` returns ~+10 for a hand-built expert
     completion and a clearly negative score for a wrong completion.

Two modes:
  * ``--snapshot-only`` (default, used by CI / smoke_test_local --mode unit):
    monkeypatches ``requests.get`` so any live HTTP attempt against the
    FHIR base raises immediately. Proves the snapshot covers the URLs
    refsol needs.
  * ``--live``: leaves ``requests.get`` alone so it can fall through to a
    real FHIR (e.g. local docker on ``http://localhost:8080/fhir/``).
    Tests the docker FHIR end-to-end.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("smoke_grpo_pipeline")


class SmokeFailure(AssertionError):
    pass


# --------------------------------------------------------------- patch wiring

def _install_snapshot_and_patches(snap_path: str, fallthrough: bool):
    """Mirror the patch order in train_grpo_32b.py exactly."""
    from rl_training.env.fhir_snapshot import (
        FhirSnapshot, install_global_snapshot,
    )

    snap = FhirSnapshot(mode="replay", path=snap_path, fallthrough=fallthrough)
    install_global_snapshot(snap)
    logger.info("Loaded snapshot %s with %d rows", snap_path, len(snap._cache))

    def _patched_send_get(url, params=None, headers=None):  # noqa: ARG001
        # refsol does ``json.loads(send_get_request(...)['data'])`` because the
        # original utils returns text for FHIR's application/fhir+json. Our
        # snapshot stores ``data`` already parsed (dict), so re-serialize so
        # refsol's json.loads works unmodified.
        res = snap.send_get_request(url)
        if "data" in res and not isinstance(res["data"], str):
            res = {**res, "data": json.dumps(res["data"])}
        return res

    import src.server.tasks.medagentbench.utils as _mb_utils
    _mb_utils.send_get_request = _patched_send_get

    import importlib
    refsol_mod = importlib.import_module("src.server.tasks.medagentbench.refsol")
    refsol_mod.send_get_request = _patched_send_get
    try:
        med_env_mod = importlib.import_module("rl_training.env.medagent_env")
        med_env_mod.send_get_request = _patched_send_get
    except ModuleNotFoundError:
        pass
    return snap, refsol_mod


def _strict_no_live(fhir_base: str):
    """Replace ``requests.get`` so any live HTTP to FHIR raises."""
    import rl_training.env.fhir_snapshot as snap_mod
    import rl_training.env.trl_env as trl_env_mod
    base_host = urlparse(fhir_base).netloc

    def _blocked_get(url, *a, **kw):  # noqa: ARG001
        host = urlparse(url).netloc
        if host == base_host or host == "" or host.startswith("localhost") or host.startswith("127."):
            raise SmokeFailure(
                f"--snapshot-only: blocked live HTTP attempt to {url!r}",
            )
        import requests as _r_real
        return _r_real.get.__wrapped__(url, *a, **kw)  # pragma: no cover

    import requests
    requests.get = _blocked_get
    snap_mod.requests.get = _blocked_get
    trl_env_mod.requests.get = _blocked_get


# ----------------------------------------------------------- expected answers

def _snap_get_json(snap, url: str) -> Any:
    res = snap.send_get_request(url)
    if "error" in res:
        raise SmokeFailure(f"snapshot miss for refsol url {url!r}: {res['error']}")
    data = res["data"]
    return json.loads(data) if isinstance(data, str) else data


def _calc_age(dob: datetime) -> int:
    today = datetime(2023, 11, 13)
    age = today.year - dob.year
    if (today.month, today.day) < (dob.month, dob.day):
        age -= 1
    return age


def _expected_answer(snap, fhir_base: str, case: dict, task_type: int) -> list:
    mrn = case["eval_MRN"]
    if task_type == 1:
        return list(case["sol"])
    if task_type == 2:
        body = _snap_get_json(snap, f"{fhir_base}Patient?identifier={mrn}&_format=json")
        dob = datetime.strptime(body["entry"][0]["resource"]["birthDate"], "%Y-%m-%d")
        return [_calc_age(dob)]
    if task_type in (4, 6, 7):
        code = {4: "MG", 6: "GLU", 7: "GLU"}[task_type]
        body = _snap_get_json(
            snap,
            f"{fhir_base}Observation?patient={mrn}&code={code}&_count=5000&_format=json",
        )
        cutoff = datetime.fromisoformat("2023-11-13T10:15:00+00:00")
        last_t, last_v, vals_in_window = None, None, []
        for ent in body.get("entry", []):
            t = datetime.fromisoformat(ent["resource"]["effectiveDateTime"])
            v = ent["resource"]["valueQuantity"]["value"]
            if task_type == 7:
                if last_t is None or t > last_t:
                    last_t, last_v = t, v
            else:
                if t >= (cutoff - timedelta(hours=24)):
                    if task_type == 6:
                        vals_in_window.append(v)
                    elif last_t is None or t > last_t:
                        last_t, last_v = t, v
        if task_type == 6:
            return [sum(vals_in_window) / len(vals_in_window)] if vals_in_window else [-1]
        return [last_v if last_v is not None else -1]
    raise SmokeFailure(f"unsupported task_type {task_type} in expected_answer")


# --------------------------------------------------------- expert completions

def _build_expert_query_completion(snap, fhir_base: str, case: dict, task_type: int):
    """Drive a single MedAgentBenchEnv to a known-good FINISH and return env+completion."""
    from rl_training.env.trl_env import MedAgentBenchEnv

    env = MedAgentBenchEnv(snapshot=snap)
    env.reset(
        task_id=case["id"], eval_MRN=case["eval_MRN"],
        instruction=case.get("instruction", ""), context=case.get("context", ""),
        ref_task_json=json.dumps(case),
    )

    mrn = case["eval_MRN"]
    if task_type == 1:
        get_url = f"{fhir_base}Patient?_count=500"
    elif task_type == 2:
        get_url = f"{fhir_base}Patient?identifier={mrn}"
    else:
        code = {4: "MG", 6: "GLU", 7: "GLU"}[task_type]
        get_url = f"{fhir_base}Observation?patient={mrn}&code={code}&_count=5000"

    env.get_fhir_resource(get_url)
    answer = _expected_answer(snap, fhir_base, case, task_type)
    answer_str = json.dumps(answer)
    env.finish(answer_str)

    completion = [
        {"role": "assistant", "content": f"GET {get_url}"},
        {"role": "tool", "content": "<fhir-response>"},
        {"role": "assistant", "content": f"FINISH({answer_str})"},
    ]
    return env, completion


def _build_wrong_completion(snap, fhir_base: str, case: dict, task_type: int):
    from rl_training.env.trl_env import MedAgentBenchEnv

    env = MedAgentBenchEnv(snapshot=snap)
    env.reset(
        task_id=case["id"], eval_MRN=case["eval_MRN"],
        instruction=case.get("instruction", ""), context=case.get("context", ""),
        ref_task_json=json.dumps(case),
    )
    env.finish('["wrong-on-purpose"]')
    completion = [{"role": "assistant", "content": 'FINISH(["wrong-on-purpose"])'}]
    return env, completion


# ------------------------------------------------------------------- runner

def _pick_one_per_type(tasks: list[dict], wanted: list[int]) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for t in tasks:
        for part in t["id"].split("_"):
            if part.startswith("task"):
                try:
                    n = int(part[len("task"):])
                except ValueError:
                    continue
                if n in wanted and n not in out:
                    out[n] = t
                break
        if len(out) == len(wanted):
            break
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot-path",
                   default="rl_training/outputs/fhir_snapshot.jsonl")
    p.add_argument("--tasks", default="rl_training/data/training_tasks_v2.json")
    p.add_argument("--fhir-base", default="http://localhost:8080/fhir/")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--snapshot-only", action="store_true", default=True,
                      help="default: forbid live HTTP")
    mode.add_argument("--live", action="store_true",
                      help="allow live HTTP fallthrough (needs docker FHIR)")
    p.add_argument("--task-types", default="1,2,4,6,7",
                   help="comma-separated task types to exercise")
    args = p.parse_args()

    if args.live:
        args.snapshot_only = False

    if not Path(args.snapshot_path).exists():
        logger.error("snapshot %s missing — run build_fhir_snapshot.py first",
                     args.snapshot_path)
        return 2
    if not Path(args.tasks).exists():
        logger.error("tasks %s missing", args.tasks)
        return 2

    import os
    os.environ["FHIR_API_BASE"] = args.fhir_base

    snap, _refsol = _install_snapshot_and_patches(
        args.snapshot_path, fallthrough=bool(args.live),
    )
    if args.snapshot_only:
        _strict_no_live(args.fhir_base)

    with open(args.tasks) as f:
        all_tasks = json.load(f)
    wanted = [int(x) for x in args.task_types.split(",") if x.strip()]
    samples = _pick_one_per_type(all_tasks, wanted)
    missing = [n for n in wanted if n not in samples]
    if missing:
        raise SmokeFailure(f"no sample task in {args.tasks} for types {missing}")

    from rl_training.rl import medagent_reward as mr
    mr.configure({"max_rounds": 8})
    r_succ = float(mr._RUNTIME["r_succ"])

    failures: list[str] = []
    for task_type, case in sorted(samples.items()):
        try:
            env, comp = _build_expert_query_completion(
                snap, args.fhir_base, case, task_type,
            )
            total, trace = mr.compute_episode_reward(env, comp, args.fhir_base)
            if not trace.get("refsol_pass"):
                failures.append(
                    f"task{task_type} id={case['id']}: refsol_pass=False "
                    f"trace={trace.get('refsol_exception') or trace.get('terms')}",
                )
                continue
            if total < r_succ - 0.01:
                failures.append(
                    f"task{task_type} id={case['id']}: pass but reward "
                    f"{total:.3f} < r_succ {r_succ}",
                )
                continue
            logger.info("OK  task%d id=%s reward=%.3f", task_type, case["id"], total)
        except SmokeFailure:
            raise
        except Exception as exc:  # noqa: BLE001
            failures.append(f"task{task_type} id={case['id']}: {type(exc).__name__}: {exc}")

    case = next(iter(samples.values()))
    env, comp = _build_wrong_completion(snap, args.fhir_base, case, 1)
    wrong_total, wrong_trace = mr.compute_episode_reward(env, comp, args.fhir_base)
    if wrong_trace.get("refsol_pass"):
        failures.append(f"wrong completion unexpectedly passed refsol on {case['id']}")
    elif wrong_total >= r_succ - 0.01:
        failures.append(
            f"wrong completion got reward {wrong_total:.3f} (>= r_succ); "
            "shaping not negative enough",
        )
    else:
        logger.info("OK  negative case id=%s reward=%.3f (refsol_pass=False)",
                    case["id"], wrong_total)

    stats = snap.stats()
    logger.info("Snapshot stats: %s", stats)
    if args.snapshot_only and stats["live_calls"] > 0:
        failures.append(
            f"--snapshot-only but snapshot performed {stats['live_calls']} live calls",
        )

    if failures:
        for f in failures:
            logger.error("FAIL %s", f)
        return 1
    logger.info("ALL %d EXPERT TASKS + NEGATIVE PASSED", len(samples))
    return 0


if __name__ == "__main__":
    sys.exit(main())
