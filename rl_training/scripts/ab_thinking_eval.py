#!/usr/bin/env python3
"""A/B eval: thinking ON vs thinking OFF, on a small slice (3 per task type).

Reuses the same MedAgentEnv + Evaluator + VLLMPolicy as run_post_train_eval.
Runs the SAME tasks twice (different policy ``extra_body``), reports
per-task SR for each arm, and dumps trajectories to two output dirs.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from rl_training.agent.vllm_policy import VLLMPolicy
from rl_training.data.trajectory import Trajectory
from rl_training.env.medagent_env import MedAgentEnv
from rl_training.evaluation.evaluator import Evaluator
from rl_training.evaluation.metrics import compute_metrics

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def select_subset(benchmark_tasks: list[dict], per_type: int = 3) -> list[dict]:
    """Pick the first ``per_type`` cases of each task type, in original order."""
    out: list[dict] = []
    counts: dict[str, int] = {}
    for t in benchmark_tasks:
        ttype = t["id"].split("_")[0]
        if counts.get(ttype, 0) >= per_type:
            continue
        counts[ttype] = counts.get(ttype, 0) + 1
        out.append(t)
    return out


def run_arm(arm_name: str,
            tasks: list[dict],
            served_model: str,
            base_url: str,
            extra_body: dict | None,
            cfg: dict,
            out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    env = MedAgentEnv.from_config(cfg)
    evaluator = Evaluator(env=env, benchmark_tasks=tasks)
    policy = VLLMPolicy(
        model_id=served_model,
        base_url=base_url,
        temperature=0.0,
        max_tokens=2048,
        max_parallel=4,
        extra_body=extra_body,
    )

    trajectories: list[Trajectory] = []
    correct = 0
    for i, task in enumerate(tasks):
        try:
            traj = evaluator._rollout(policy, task)  # noqa: SLF001
            trajectories.append(traj)
            if traj.correct:
                correct += 1
        except Exception as exc:
            logger.warning("Rollout failed for %s: %s", task.get("id"), exc)
            trajectories.append(Trajectory.from_env_history(
                task=task, history=[], correct=False, status="error",
                model_id=served_model,
            ))
        logger.info("[%s] %d/%d done, SR=%.1f%%",
                    arm_name, i + 1, len(tasks), 100 * correct / (i + 1))

    result = compute_metrics(trajectories)
    summary = {
        "arm": arm_name,
        "extra_body": extra_body,
        "total": result.total,
        "correct": result.correct,
        "success_rate": result.success_rate,
        "per_task_sr": result.per_task_sr,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "trajectories.jsonl", "w") as f:
        for tr in trajectories:
            f.write(tr.to_jsonl_line() + "\n")
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="rl_training/configs/default.yaml")
    p.add_argument("--vllm-model", required=True)
    p.add_argument("--vllm-base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--per-type", type=int, default=3)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    data_file = cfg["env"]["data_file"]
    with open(data_file) as f:
        all_tasks = json.load(f)
    subset = select_subset(all_tasks, per_type=args.per_type)
    logger.info("Selected %d tasks (%d per type x 10 types)", len(subset), args.per_type)

    base_out = Path(args.out_dir)

    # ARM A: thinking ON (default; no extra_body kwarg)
    sa = run_arm("think_ON", subset, args.vllm_model, args.vllm_base_url,
                 extra_body=None, cfg=cfg, out_dir=base_out / "think_on")

    # ARM B: thinking OFF
    sb = run_arm("think_OFF", subset, args.vllm_model, args.vllm_base_url,
                 extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                 cfg=cfg, out_dir=base_out / "think_off")

    print("\n==== A/B SUMMARY ====")
    print(f"Total tasks per arm: {len(subset)}")
    print()
    print(f"{'task':<8} {'think_ON':>10}  {'think_OFF':>10}  delta")
    keys = sorted(set(list(sa['per_task_sr'].keys()) + list(sb['per_task_sr'].keys())),
                  key=lambda s: int(s.replace("task", "")))
    for k in keys:
        a = sa['per_task_sr'].get(k, 0.0)
        b = sb['per_task_sr'].get(k, 0.0)
        delta = b - a
        print(f"{k:<8} {a*100:>9.1f}% {b*100:>9.1f}%  {delta*100:+.1f}pp")
    print()
    print(f"OVERALL  ON: {sa['correct']}/{sa['total']} ({sa['success_rate']*100:.1f}%)  "
          f"OFF: {sb['correct']}/{sb['total']} ({sb['success_rate']*100:.1f}%)  "
          f"delta {(sb['success_rate']-sa['success_rate'])*100:+.1f}pp")

    with open(base_out / "ab_summary.json", "w") as f:
        json.dump({"think_on": sa, "think_off": sb}, f, indent=2)


if __name__ == "__main__":
    main()
