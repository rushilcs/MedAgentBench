#!/usr/bin/env python3
"""Build the train/val JSONL files for OpenAI Reinforcement Fine-Tuning of o4-mini.

Steps:
    1. Read benchmark tasks (``test_data_v2.json``) to harvest MRNs we must hold out.
    2. Use ``TaskGenerator`` to synthesize training tasks per task type with the
       same template families as the benchmark (different patients).
    3. For each task, call out to live FHIR to pre-compute the grader reference
       (via ``rl_training.rft.reference_builder.build_reference``).
    4. Fold prefetched GET responses into a single-turn prompt that mirrors the
       benchmark's system prompt so behavior transfers at eval time.
    5. Stratified-split into train/val and emit OpenAI RFT JSONL:
         {"messages": [{"role": "user", ...}], "task_type": ..., "reference": {...}, ...}

Usage:
    python rl_training/scripts/build_rft_dataset.py \
        --config rl_training/configs/o4_mini_rft.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from rl_training.data.task_generator import TaskGenerator
from rl_training.env.medagent_env import MedAgentEnv
from rl_training.rft.reference_builder import TaskReference, build_reference

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


PROMPT_SUFFIX = (
    "\n\n---\n"
    "The FHIR GET queries required to answer have been pre-executed for you. "
    "Their responses are shown verbatim below. Use them as if you had issued the "
    "GETs yourself. Then produce the remaining POST(s) (if any) and the final "
    "FINISH([...]) call using the exact same text protocol.\n"
    "\nIf the task is a pure lookup, respond with only:\n"
    "FINISH([answer])\n"
    "\nIf the task requires one or more writes, respond with the POST(s) first, "
    "each on its own line exactly like:\n"
    "POST {url}\n"
    "{json body}\n"
    "and then end with FINISH([...]).\n"
    "\nPre-executed GET responses:\n"
)


def _render_prefetched(prefetched: list[dict[str, str]]) -> str:
    if not prefetched:
        return "(no GETs required for this task — proceed directly to POST/FINISH)"
    chunks: list[str] = []
    for idx, entry in enumerate(prefetched, 1):
        chunks.append(
            f"[GET {idx}] {entry['url']}\n"
            f"Response: {entry['body']}"
        )
    return "\n\n".join(chunks)


def build_prompt(env: MedAgentEnv, task: dict[str, Any], reference: TaskReference) -> str:
    """Build the single-turn RFT prompt = system prompt + inlined GET responses."""
    state = env.reset(task)
    base_prompt = state.history[0]["content"]
    return base_prompt + PROMPT_SUFFIX + _render_prefetched(reference.prefetched_gets)


def build_rft_row(task: dict[str, Any], reference: TaskReference, prompt: str) -> dict[str, Any]:
    """Construct one JSONL row in the OpenAI RFT reinforcement-input schema."""
    return {
        "messages": [{"role": "user", "content": prompt}],
        "task_id": task["id"],
        "task_type": reference.task_type,
        "eval_MRN": task["eval_MRN"],
        "reference_sol": reference.reference_sol,
        "accepts_empty_finish": reference.accepts_empty_finish,
        "task_params": reference.task_params,
    }


def _stratified_split(
    rows: list[dict[str, Any]],
    n_train: int,
    n_val: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Stratified split preserving task-type balance."""
    by_type: dict[int, list[dict[str, Any]]] = defaultdict(list)
    rng = random.Random(seed)
    for row in rows:
        by_type[row["task_type"]].append(row)
    for bucket in by_type.values():
        rng.shuffle(bucket)

    total = n_train + n_val
    types = sorted(by_type.keys())
    per_type_train = n_train // len(types)
    per_type_val = n_val // len(types)
    extra_train = n_train - per_type_train * len(types)
    extra_val = n_val - per_type_val * len(types)

    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    for i, tt in enumerate(types):
        bucket = by_type[tt]
        t_take = per_type_train + (1 if i < extra_train else 0)
        v_take = per_type_val + (1 if i < extra_val else 0)
        taken_train = bucket[:t_take]
        taken_val = bucket[t_take:t_take + v_take]
        train.extend(taken_train)
        val.extend(taken_val)

    # Top up from a shared leftover pool if any bucket was short.
    if len(train) < n_train or len(val) < n_val:
        leftover: list[dict[str, Any]] = []
        for i, tt in enumerate(types):
            bucket = by_type[tt]
            t_take = per_type_train + (1 if i < extra_train else 0)
            v_take = per_type_val + (1 if i < extra_val else 0)
            leftover.extend(bucket[t_take + v_take:])
        rng.shuffle(leftover)
        while len(train) < n_train and leftover:
            train.append(leftover.pop())
        while len(val) < n_val and leftover:
            val.append(leftover.pop())

    rng.shuffle(train)
    rng.shuffle(val)
    return train[:n_train], val[:n_val]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    logger.info("Wrote %d rows -> %s", len(rows), path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RFT JSONL from generated tasks")
    parser.add_argument("--config", default="rl_training/configs/o4_mini_rft.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--count-per-type", type=int, default=None,
                        help="Override rft.training_data-style per-type count")
    parser.add_argument("--n-train", type=int, default=None)
    parser.add_argument("--n-val", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Generate tasks but don't fetch FHIR")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    run_dir = Path(args.output_dir or config["output"]["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    count_per_type = args.count_per_type or config.get("training_data", {}).get("tasks_per_type", 40)
    seed = config.get("training_data", {}).get("seed", 42)
    n_train = args.n_train or config["rft"]["n_train"]
    n_val = args.n_val or config["rft"]["n_val"]
    fhir_api_base = config["env"]["fhir_api_base"]

    bench_file = config["env"]["data_file"]
    with open(bench_file) as f:
        bench_tasks = json.load(f)
    existing_mrns = {t["eval_MRN"] for t in bench_tasks if "eval_MRN" in t}
    logger.info("Holding out %d MRNs from %s", len(existing_mrns), bench_file)

    gen = TaskGenerator(fhir_api_base=fhir_api_base, seed=seed, existing_mrns=existing_mrns)
    train_tasks = gen.generate_all(count_per_type=count_per_type)
    logger.info("Generated %d candidate training tasks", len(train_tasks))

    tasks_path = run_dir / "training_tasks.json"
    with open(tasks_path, "w") as f:
        json.dump(train_tasks, f, indent=2)
    logger.info("Wrote candidate tasks -> %s", tasks_path)

    if args.dry_run:
        logger.info("--dry-run set; skipping FHIR pre-fetch and JSONL emission")
        return

    env = MedAgentEnv.from_config(config)

    rows: list[dict[str, Any]] = []
    skipped = 0
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), MofNCompleteColumn()) as prog:
        ptask = prog.add_task("Pre-fetching FHIR + building RFT rows", total=len(train_tasks))
        for task in train_tasks:
            try:
                reference = build_reference(task, fhir_api_base)
            except Exception as exc:
                logger.warning("reference build failed for %s: %s", task["id"], exc)
                reference = None
            if reference is None:
                skipped += 1
                prog.advance(ptask)
                continue
            prompt = build_prompt(env, task, reference)
            rows.append(build_rft_row(task, reference, prompt))
            prog.advance(ptask)

    logger.info("Built %d RFT rows (skipped %d)", len(rows), skipped)

    # Dump the full pre-split pool for debugging / grader local validation.
    pool_path = run_dir / "rft_rows_all.jsonl"
    _write_jsonl(pool_path, rows)

    train_rows, val_rows = _stratified_split(rows, n_train, n_val, seed)
    _write_jsonl(run_dir / "train.jsonl", train_rows)
    _write_jsonl(run_dir / "val.jsonl", val_rows)

    by_type_train: dict[int, int] = defaultdict(int)
    by_type_val: dict[int, int] = defaultdict(int)
    for r in train_rows:
        by_type_train[r["task_type"]] += 1
    for r in val_rows:
        by_type_val[r["task_type"]] += 1
    logger.info("train per-type: %s", dict(sorted(by_type_train.items())))
    logger.info("val   per-type: %s", dict(sorted(by_type_val.items())))


if __name__ == "__main__":
    main()
