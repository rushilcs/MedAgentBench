#!/usr/bin/env python3
"""Build the SFT v2 corpus.

Strategy (revised after Step 2 post-mortem):

  * **Drop the v1 corpus entirely for time-window tasks** (4, 5, 6, 7, 9, 10).
    The v1 trajectories teach unfiltered GETs (``?code=MG&patient=X``)
    plus an internal date-filter step that the model cannot perform with
    ``enable_thinking=False`` at inference. The post-Step-2 eval showed
    task4 SR collapse from 93% (think-on) -> 47% (think-off) for exactly
    this reason.

  * **Keep the v1 corpus for non-time-window tasks** (1, 2, 3, 8) after
    the existing pollution filter. Those tasks don't depend on date
    arithmetic so the v1 examples are still useful (and add diversity).

  * **Generate a large fresh batch of programmatic v2 trajectories** for
    all 10 task types. The v2 builders push the time filter into the
    FHIR query (``&date=ge<cutoff-24h>`` or ``&_sort=-date&_count=1``),
    so the model only has to read off the response.

Output: ``rl_training/outputs/qwen_pipeline_v3/phase_a/qwen_sft_openai.jsonl``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from pathlib import Path

import yaml

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from rl_training.data.task_generator import TaskGenerator
from rl_training.data.trajectory import Trajectory
from rl_training.data.trajectory_store import TrajectoryStore
from rl_training.env.medagent_env import MedAgentEnv
from rl_training.training.expert_collector_v2 import _BUILDERS_V2

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


_FINISH_RX = re.compile(r"FINISH\(\s*(\[[\s\S]*?\])\s*\)")
_FQDN_REF_RX = re.compile(r'"reference"\s*:\s*"https?://[^"]+/Patient/')


def _is_polluted(messages: list[dict]) -> tuple[bool, str]:
    for m in messages:
        if m.get("role") != "assistant":
            continue
        content = m.get("content", "") or ""
        if _FQDN_REF_RX.search(content):
            return True, "fqdn_subject_reference"
        for finish_match in _FINISH_RX.finditer(content):
            try:
                lst = json.loads(finish_match.group(1))
            except Exception:
                continue
            if not isinstance(lst, list):
                continue
            for el in lst:
                if isinstance(el, str) and len(el) > 40:
                    return True, "prose_in_finish"
            if (
                lst
                and all(isinstance(x, str) for x in lst)
                and lst != ["Patient not found"]
                and not all(re.match(r"^\d{4}-\d{2}-\d{2}", x) for x in lst)
                and not all(re.match(r"^S\d+$", x) for x in lst)
            ):
                return True, "string_only_finish_non_canonical"
        if content.lstrip().startswith("<think>"):
            return True, "leading_think_block"
    return False, ""


def _hash_messages(messages: list[dict]) -> str:
    payload = json.dumps(
        [{"role": m["role"], "content": m["content"]} for m in messages],
        sort_keys=False,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _classify_v1_example(messages: list[dict]) -> str:
    """Return the task type string ('task1'..'task10' or '?') for a v1 example."""
    instr = next((m["content"] for m in messages if m["role"] == "user"), "")
    if "CBG" in instr and "average" in instr: return "task6"
    if "most recent CBG" in instr: return "task7"
    if "magnesium" in instr.lower() and ("replacement" in instr.lower() or "1.9" in instr or "order" in instr.lower()): return "task5"
    if "magnesium level" in instr.lower(): return "task4"
    if "potassium" in instr.lower() or "serum K " in instr or "serum K." in instr: return "task9"
    if "A1C" in instr or "HbA1C" in instr or "a1c" in instr.lower(): return "task10"
    if "orthopedic" in instr.lower() or "ACL" in instr: return "task8"
    if "blood pressure" in instr.lower() and "record" in instr.lower(): return "task3"
    if "MRN of" in instr and "age" in instr.lower(): return "task2"
    if ("name" in instr.lower() and "DOB" in instr): return "task1"
    return "?"


# Tasks where v1 examples are KEPT (no time-window). Everything else is
# regenerated from v2 builders so the GET pattern is consistent.
_KEEP_V1_TYPES = {"task1", "task2", "task3", "task8"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="rl_training/configs/default.yaml")
    p.add_argument("--v1-sft-jsonl",
                   default="rl_training/outputs/qwen_pipeline_v2/phase_a/qwen_sft_openai.jsonl")
    p.add_argument("--output-dir",
                   default="rl_training/outputs/qwen_pipeline_v3/phase_a")
    p.add_argument("--count-per-type", type=int, default=200)
    p.add_argument("--seed", type=int, default=4242)
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audit: dict = {}

    # ----- 1. Filter v1 (only tasks 1/2/3/8 are kept) -----
    v1_path = Path(args.v1_sft_jsonl)
    kept: list[dict] = []
    seen_hashes: set[str] = set()
    drops: dict[str, int] = {}
    kept_by_type: dict[str, int] = {}
    with v1_path.open() as f:
        for line in f:
            ex = json.loads(line)
            polluted, reason = _is_polluted(ex["messages"])
            if polluted:
                drops[reason] = drops.get(reason, 0) + 1
                continue
            tt = _classify_v1_example(ex["messages"])
            if tt not in _KEEP_V1_TYPES:
                drops[f"drop_v1_{tt}"] = drops.get(f"drop_v1_{tt}", 0) + 1
                continue
            h = _hash_messages(ex["messages"])
            if h in seen_hashes:
                drops["dup"] = drops.get("dup", 0) + 1
                continue
            seen_hashes.add(h)
            kept.append(ex)
            kept_by_type[tt] = kept_by_type.get(tt, 0) + 1
    logger.info("v1 filter: kept %d (%s); drops: %s",
                len(kept), kept_by_type, drops)
    audit["v1_kept"] = len(kept)
    audit["v1_kept_by_type"] = kept_by_type
    audit["v1_drops"] = drops

    # ----- 2. Generate fresh synthetic tasks -----
    with open(cfg["env"]["data_file"]) as f:
        benchmark_tasks = json.load(f)
    existing_mrns = {t["eval_MRN"] for t in benchmark_tasks if "eval_MRN" in t}
    v1_tasks_path = Path("rl_training/outputs/qwen_pipeline_v2/phase_a/training_tasks.json")
    if v1_tasks_path.exists():
        for t in json.loads(v1_tasks_path.read_text()):
            if "eval_MRN" in t:
                existing_mrns.add(t["eval_MRN"])

    task_gen = TaskGenerator(
        fhir_api_base=cfg["env"]["fhir_api_base"],
        seed=args.seed,
        existing_mrns=existing_mrns,
    )
    new_tasks = task_gen.generate_all(count_per_type=args.count_per_type)
    (out_dir / "training_tasks_v2.json").write_text(json.dumps(new_tasks, indent=2))
    logger.info("Generated %d synthetic training tasks (seed=%d)",
                len(new_tasks), args.seed)

    # ----- 3. Build programmatic trajectories with V2 builders -----
    env = MedAgentEnv.from_config(cfg)
    traj_path = out_dir / "expert_trajectories_v2.jsonl"
    store = TrajectoryStore(traj_path)
    built: list[Trajectory] = []
    failed_per_type: dict[str, int] = {}
    for task in new_tasks:
        task_id = task["id"].replace("train_", "")
        task_type = task_id.split("_")[0]
        builder = _BUILDERS_V2.get(task_type)
        if builder is None:
            continue
        try:
            traj = builder(task, env)
            if traj is not None and traj.correct:
                built.append(traj)
            else:
                failed_per_type[task_type] = failed_per_type.get(task_type, 0) + 1
        except Exception as exc:
            failed_per_type[task_type] = failed_per_type.get(task_type, 0) + 1
            logger.warning("Build failed for %s: %s", task["id"], exc)
    store.save_batch(built)
    logger.info("Built %d v2 programmatic trajectories (failed: %s)",
                len(built), failed_per_type)

    # ----- 4. Combine -----
    new_examples: list[dict] = []
    new_by_type: dict[str, int] = {}
    for tr in built:
        msgs = tr.to_openai_messages()
        polluted, _ = _is_polluted(msgs)
        if polluted:
            continue
        h = _hash_messages(msgs)
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        new_examples.append({"messages": msgs})
        ttype = tr.task_id.replace("train_", "").split("_")[0]
        new_by_type[ttype] = new_by_type.get(ttype, 0) + 1

    audit["v2_added"] = len(new_examples)
    audit["v2_added_by_type"] = new_by_type

    combined = kept + new_examples
    out_jsonl = out_dir / "qwen_sft_openai.jsonl"
    with out_jsonl.open("w") as f:
        for ex in combined:
            f.write(json.dumps(ex) + "\n")
    audit["combined_total"] = len(combined)

    summary_path = out_dir / "build_v2_summary.json"
    summary_path.write_text(json.dumps(audit, indent=2))
    logger.info("Wrote %d combined examples to %s", len(combined), out_jsonl)
    logger.info("Audit: %s", json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
