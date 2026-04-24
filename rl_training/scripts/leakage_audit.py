"""Leakage audit between training corpora and the v2 benchmark test set.

Verifies (and emits a receipt JSON) that no patient MRN and no task id appears
in both a training-corpus task list and ``data/medagentbench/test_data_v2.json``.

Acts as a CI-style gate: ``main()`` returns non-zero exit code on any overlap.

Usage::

    python -m rl_training.scripts.leakage_audit \
        --train rl_training/outputs/qwen_pipeline_v3/phase_a/training_tasks_v2.json \
        --test  data/medagentbench/test_data_v2.json \
        --out   rl_training/outputs/qwen_pipeline_v3/phase_a/leakage_audit.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _load_tasks(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return json.load(f)


def audit(train_tasks: list[dict[str, Any]], test_tasks: list[dict[str, Any]]) -> dict[str, Any]:
    """Return a structured audit dict; caller decides pass/fail on `overlap_*` keys."""
    train_mrns = {t["eval_MRN"] for t in train_tasks if "eval_MRN" in t}
    test_mrns = {t["eval_MRN"] for t in test_tasks if "eval_MRN" in t}
    train_ids = {t.get("id") for t in train_tasks if t.get("id")}
    test_ids = {t.get("id") for t in test_tasks if t.get("id")}

    mrn_overlap = sorted(train_mrns & test_mrns)
    id_overlap = sorted(train_ids & test_ids)

    return {
        "n_train_tasks": len(train_tasks),
        "n_test_tasks": len(test_tasks),
        "n_train_unique_mrns": len(train_mrns),
        "n_test_unique_mrns": len(test_mrns),
        "mrn_overlap_count": len(mrn_overlap),
        "mrn_overlap_examples": mrn_overlap[:10],
        "task_id_overlap_count": len(id_overlap),
        "task_id_overlap_examples": id_overlap[:10],
        "passed": (not mrn_overlap) and (not id_overlap),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", required=True, type=Path)
    ap.add_argument("--test", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    train_tasks = _load_tasks(args.train)
    test_tasks = _load_tasks(args.test)
    report = audit(train_tasks, test_tasks)
    report["train_path"] = str(args.train)
    report["test_path"] = str(args.test)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")

    print(json.dumps(report, indent=2))
    if not report["passed"]:
        print(
            f"\nLEAKAGE DETECTED: {report['mrn_overlap_count']} MRN overlap, "
            f"{report['task_id_overlap_count']} task-id overlap. Refusing to proceed.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
