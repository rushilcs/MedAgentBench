#!/usr/bin/env python3
"""Build the SFT v3 corpus.

Strategy
--------
v3 = v2 corpus  +  branch-balancing augmentation derived from the live FHIR
fixture audit. The audit (``audit_test_fixture.py``) showed that on the
current fixture:

  * task9 is 29/30 normal_no_order (FINISH([])); v2 SR is 13.3%, so the model
    over-POSTs on normal-K. We need many more no-order examples.
  * task10 is 13/13/4 (fresh_report_value / absent_order / stale_order); v2
    SR is 40%, so the model under-handles the fresh_report_value branch.
  * task5 has only 2 low_order_required rows in test, but the model still
    needs the low-Mg branch in training so it doesn't catastrophically forget.

Approach: over-generate training tasks per type (no_order/no_data branches
are common enough that random sampling gives plenty), classify each by
branch using the same logic as ``audit_test_fixture.py``, build a trajectory
per task with the v2 builder, then sub-select to the target counts so the
final corpus is branch-balanced for the under-served branches.

Hard constraints
----------------
* Zero MRN overlap with the v2 test set (``data/medagentbench/test_data_v2.json``).
  Enforced via ``existing_mrns`` and re-verified by the leakage audit.
* All produced rows pass the single-action-per-turn invariant (the SFT
  loader will refuse to train otherwise).

Output
------
* ``rl_training/outputs/qwen_pipeline_v3/phase_a/qwen_sft_v3_openai.jsonl`` -- final corpus
* ``rl_training/outputs/qwen_pipeline_v3/phase_a/training_tasks_v3.json`` -- new tasks added
* ``rl_training/outputs/qwen_pipeline_v3/phase_a/build_v3_summary.json`` -- audit
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from rl_training.data.task_generator import TaskGenerator
from rl_training.data.trajectory import Trajectory
from rl_training.env.medagent_env import MedAgentEnv
from rl_training.training.expert_collector_v2 import _BUILDERS_V2
from rl_training.training.single_action_invariant import violations
from rl_training.scripts.audit_test_fixture import (
    audit_task5, audit_task9, audit_task10,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


# Per-task branch quotas to ADD on top of the v2 corpus. Numbers chosen to
# roughly mirror the test set's branch distribution (so the SFT distribution
# isn't wildly different from the eval distribution) while heavily
# emphasising the branches v2 demonstrably failed.
#
# task9: 29/30 test rows are normal_no_order; v2 gets 13.3%. Add 200 extra
# normal_no_order examples and 25 low_order to keep both branches alive.
# task10: 13/30 fresh + 13/30 absent + 4/30 stale; v2 gets 40%. Heavily
# augment fresh_report_value and stale_order_required (the under-served
# branches per v2 trajectories).
# task5: 2/30 low_order test rows; v2 currently fine on no-order. Add 25
# low_order to keep the order branch alive across re-training.
TARGET_AUGMENTATIONS: dict[str, dict[str, int]] = {
    "task9": {"normal_no_order": 200, "low_order_required": 25},
    "task10": {"fresh_report_value": 100, "stale_order_required": 50, "data_absent_order_required": 50},
    "task5": {"low_order_required": 25, "high_no_order": 25, "data_absent_no_order": 25},
}


_AUDITORS = {
    "task5": audit_task5,
    "task9": audit_task9,
    "task10": audit_task10,
}


def _hash_messages(messages: list[dict]) -> str:
    payload = json.dumps(
        [{"role": m["role"], "content": m["content"]} for m in messages],
        sort_keys=False,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _classify_task(task: dict, fhir_base: str) -> str:
    """Return the audit branch label for a generated training task, or 'n/a'.

    TaskGenerator emits ids like ``train_task9_1``; we want the ``task9`` part.
    """
    tid = next(
        (part for part in task["id"].split("_") if part.startswith("task")),
        None,
    )
    if tid is None:
        return "n/a"
    auditor = _AUDITORS.get(tid)
    if auditor is None:
        return "n/a"
    try:
        info = auditor(task, fhir_base)
    except Exception as exc:
        logger.warning("auditor failed for %s: %s", task["id"], exc)
        return "auditor_exception"
    return info.get("branch", "?")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="rl_training/configs/default.yaml")
    p.add_argument(
        "--v2-corpus",
        default="rl_training/outputs/qwen_pipeline_v3/phase_a/qwen_sft_openai.jsonl",
        help="Existing SFT v2 corpus (kept verbatim as the v3 base).",
    )
    p.add_argument(
        "--v2-tasks",
        default="rl_training/outputs/qwen_pipeline_v3/phase_a/training_tasks_v2.json",
        help="Training tasks used by v2 (their MRNs are excluded from v3 sampling).",
    )
    p.add_argument(
        "--test-data",
        default="data/medagentbench/test_data_v2.json",
        help="Test set whose MRNs are excluded from v3 sampling.",
    )
    p.add_argument(
        "--output-dir",
        default="rl_training/outputs/qwen_pipeline_v3/phase_a",
    )
    p.add_argument(
        "--oversample-multiplier", type=int, default=8,
        help="For each TARGET_AUGMENTATIONS slot, generate this many candidate "
             "tasks before sub-selecting. Higher = more robust branch coverage.",
    )
    p.add_argument("--seed", type=int, default=131313)
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fhir_base = cfg["env"]["fhir_api_base"]
    summary: dict[str, Any] = {"v3_seed": args.seed, "fhir_base": fhir_base}

    # ----- 1. Read existing v2 corpus verbatim and validate it -----
    v2_rows: list[dict] = []
    seen_hashes: set[str] = set()
    with open(args.v2_corpus) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            v = violations(ex["messages"])
            if v:
                raise RuntimeError(
                    f"v2 corpus violates single-action-per-turn invariant; "
                    f"refusing to build v3 on top. Run leakage_audit first. "
                    f"First bad row counts={v[0]['counts']}"
                )
            v2_rows.append(ex)
            seen_hashes.add(_hash_messages(ex["messages"]))
    logger.info("v2 corpus: %d rows (all pass single-action invariant)", len(v2_rows))
    summary["v2_rows"] = len(v2_rows)

    # ----- 2. Collect MRNs to exclude (test + v2 train) -----
    test_rows = json.loads(Path(args.test_data).read_text())
    test_mrns = {r["eval_MRN"] for r in test_rows if "eval_MRN" in r}
    v2_train_mrns = {
        r["eval_MRN"] for r in json.loads(Path(args.v2_tasks).read_text())
        if "eval_MRN" in r
    }
    existing_mrns = test_mrns | v2_train_mrns
    logger.info(
        "Excluding %d MRNs from v3 sampling (%d test + %d v2-train)",
        len(existing_mrns), len(test_mrns), len(v2_train_mrns),
    )

    # ----- 3. Generate over-sampled candidate tasks per augmentation type -----
    task_gen = TaskGenerator(
        fhir_api_base=fhir_base, seed=args.seed,
        existing_mrns=existing_mrns,
    )
    env = MedAgentEnv.from_config(cfg)

    targets_total = sum(
        sum(b.values()) for b in TARGET_AUGMENTATIONS.values()
    )
    summary["targets_total"] = targets_total
    summary["target_augmentations"] = TARGET_AUGMENTATIONS

    candidates_by_task: dict[str, list[dict]] = defaultdict(list)
    for tid, branches in TARGET_AUGMENTATIONS.items():
        per_task_quota = sum(branches.values())
        n_to_gen = per_task_quota * args.oversample_multiplier
        type_int = int(tid.replace("task", ""))
        logger.info("generating %d candidates for %s (quota=%d, x%d)",
                    n_to_gen, tid, per_task_quota, args.oversample_multiplier)
        candidates_by_task[tid] = task_gen.generate_tasks(type_int, n_to_gen)

    # ----- 4. Classify candidates by branch, build trajectories, sub-select -----
    new_rows: list[dict] = []
    new_by_task_branch: Counter = Counter()
    skipped: Counter = Counter()
    selected_per_slot: dict[tuple[str, str], int] = defaultdict(int)
    audit_per_row: list[dict] = []

    for tid, candidates in candidates_by_task.items():
        wanted = TARGET_AUGMENTATIONS[tid]
        builder = _BUILDERS_V2.get(tid)
        if builder is None:
            logger.warning("no v2 builder for %s; skipping", tid)
            continue
        for cand in candidates:
            if all(selected_per_slot[(tid, b)] >= n for b, n in wanted.items()):
                break  # all branches for this task are full
            branch = _classify_task(cand, fhir_base)
            audit_per_row.append({"task_id": cand["id"], "task": tid, "branch": branch})
            if branch not in wanted:
                skipped[f"{tid}/{branch}/not_targeted"] += 1
                continue
            if selected_per_slot[(tid, branch)] >= wanted[branch]:
                skipped[f"{tid}/{branch}/quota_full"] += 1
                continue
            try:
                traj: Trajectory | None = builder(cand, env)
            except Exception as exc:
                skipped[f"{tid}/{branch}/builder_exception"] += 1
                logger.warning("builder failed for %s: %s", cand["id"], exc)
                continue
            if traj is None or not traj.correct:
                skipped[f"{tid}/{branch}/builder_returned_none"] += 1
                continue
            msgs = traj.to_openai_messages()
            v = violations(msgs)
            if v:
                skipped[f"{tid}/{branch}/invariant_violation"] += 1
                logger.warning("invariant violation for %s: counts=%s", cand["id"], v[0]["counts"])
                continue
            h = _hash_messages(msgs)
            if h in seen_hashes:
                skipped[f"{tid}/{branch}/duplicate"] += 1
                continue
            seen_hashes.add(h)
            new_rows.append({"messages": msgs})
            new_by_task_branch[(tid, branch)] += 1
            selected_per_slot[(tid, branch)] += 1

    logger.info("v3 augmentation: added %d rows", len(new_rows))
    logger.info("by (task, branch): %s",
                {f"{k[0]}/{k[1]}": v for k, v in new_by_task_branch.items()})
    logger.info("skipped: %s", dict(skipped))
    summary["v3_added"] = len(new_rows)
    summary["v3_added_by_task_branch"] = {
        f"{k[0]}/{k[1]}": v for k, v in new_by_task_branch.items()
    }
    summary["skip_reasons"] = dict(skipped)
    summary["target_branch_fill"] = {
        f"{k[0]}/{k[1]}": {
            "got": selected_per_slot[k],
            "wanted": TARGET_AUGMENTATIONS[k[0]][k[1]],
        }
        for k in [(t, b) for t, bs in TARGET_AUGMENTATIONS.items() for b in bs]
    }

    # ----- 5. Write outputs -----
    combined = v2_rows + new_rows
    out_jsonl = out_dir / "qwen_sft_v3_openai.jsonl"
    with out_jsonl.open("w") as f:
        for ex in combined:
            f.write(json.dumps(ex) + "\n")
    summary["combined_total"] = len(combined)

    # Persist the new training tasks for the leakage audit.
    new_task_records: list[dict] = []
    new_task_ids: set[str] = {r["task_id"] for r in audit_per_row}
    for tid_audit, cands in candidates_by_task.items():
        for cand in cands:
            if cand["id"] in new_task_ids:
                new_task_records.append(cand)
    (out_dir / "training_tasks_v3.json").write_text(
        json.dumps(new_task_records, indent=2)
    )

    (out_dir / "build_v3_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    (out_dir / "v3_audit_per_row.json").write_text(
        json.dumps(audit_per_row, indent=2)
    )

    logger.info("wrote %d combined rows to %s", len(combined), out_jsonl)
    logger.info("summary written to %s", out_dir / "build_v3_summary.json")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
