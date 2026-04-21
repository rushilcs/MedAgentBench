#!/usr/bin/env python3
"""Regenerate expert trajectories for the Qwen SFT stage.

This is a Qwen-specific analog of the gpt-4o-mini phase-A flow:

1. Load (or generate) ``training_tasks.json`` - same TaskGenerator as before.
2. Run a gpt-4o policy against each task through the MedAgentBench env,
   collecting full multi-turn trajectories.
3. Filter to trajectories where ``env.grade()`` returned True.
4. Emit two files:

   * ``expert_trajectories.jsonl`` - full ``Trajectory`` records (same
     format as ``rl_training/outputs/gpt4o_pipeline/phase_a/expert_trajectories.jsonl``).
     Kept so we can audit the rollouts and re-run filtering.
   * ``qwen_sft_openai.jsonl`` - ``{"messages":[...]}`` per line, ready to
     consume with ``trl.SFTTrainer`` + Qwen's ``tokenizer.apply_chat_template``.
     This is the file ``sft_qwen3_32b.py`` reads.

The script is idempotent: if ``expert_trajectories.jsonl`` already exists,
we skip any ``task_id`` it already covers and only run rollouts for the
missing tasks. This lets you re-run after a partial failure without
re-burning OpenAI credits.

Run locally (CPU-only, OpenAI API cost only):

    export OPENAI_API_KEY=sk-...
    python rl_training/scripts/generate_qwen_sft_expert_trajectories.py \\
        --config rl_training/configs/default.yaml \\
        --output-dir rl_training/outputs/qwen_pipeline/phase_a \\
        --expert-model gpt-4o \\
        --trajectories-per-task 3

``--trajectories-per-task 3`` at temperature 0.7 is roughly what produced
the 1361 correct samples in the existing ``medagent-sft-mini_*.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import yaml

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from rl_training.agent.openai_policy import OpenAIPolicy
from rl_training.data.task_generator import TaskGenerator
from rl_training.data.trajectory import Trajectory
from rl_training.data.trajectory_store import TrajectoryStore
from rl_training.env.medagent_env import MedAgentEnv
from rl_training.env.reward import compute_episode_reward

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _load_or_generate_training_tasks(
    config: dict,
    explicit_path: str | None,
    output_dir: Path,
) -> list[dict]:
    """Either load a pre-existing ``training_tasks.json`` or generate one.

    We prefer loading an existing file so Qwen SFT uses the exact same
    task set the gpt-4o-mini SFT was trained on (keeps the comparison
    honest). If none exists, we fall back to generating fresh tasks with
    the same seed the original pipeline used.
    """
    if explicit_path and Path(explicit_path).exists():
        with open(explicit_path) as f:
            tasks = json.load(f)
        logger.info("Loaded %d training tasks from %s", len(tasks), explicit_path)
        return tasks

    # Reuse the gpt-4o-mini pipeline's tasks if present (same tasks, same MRNs).
    default_path = Path("rl_training/outputs/gpt4o_pipeline/phase_a/training_tasks.json")
    if default_path.exists():
        with open(default_path) as f:
            tasks = json.load(f)
        logger.info(
            "Reusing gpt-4o-mini pipeline's training tasks from %s (%d tasks)",
            default_path, len(tasks),
        )
        # Copy into our output dir so the SFT run is self-contained.
        out_copy = output_dir / "training_tasks.json"
        if not out_copy.exists():
            out_copy.parent.mkdir(parents=True, exist_ok=True)
            out_copy.write_text(json.dumps(tasks, indent=2))
        return tasks

    # Last resort: generate fresh tasks with the same seed as the default config.
    logger.info(
        "No existing training_tasks.json found; generating fresh tasks "
        "(this requires a live FHIR server at %s)",
        config["env"]["fhir_api_base"],
    )
    with open(config["env"]["data_file"]) as f:
        benchmark_tasks = json.load(f)
    existing_mrns = {t["eval_MRN"] for t in benchmark_tasks if "eval_MRN" in t}
    seed = config.get("training_data", {}).get("seed", 42)
    count_per_type = config.get("training_data", {}).get("tasks_per_type", 50)
    task_gen = TaskGenerator(
        fhir_api_base=config["env"]["fhir_api_base"],
        seed=seed,
        existing_mrns=existing_mrns,
    )
    tasks = task_gen.generate_all(count_per_type=count_per_type)
    out_path = output_dir / "training_tasks.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(tasks, indent=2))
    logger.info("Generated + saved %d training tasks to %s", len(tasks), out_path)
    return tasks


def _rollout_task(
    env: MedAgentEnv,
    task: dict,
    policy: OpenAIPolicy,
) -> Trajectory:
    """Run one expert rollout against the env. Mirrors ExpertCollector.collect."""
    state = env.reset(task)
    while not state.done:
        action = policy.act(state.history)
        result = env.step(action)
        state = result.state
    correct = env.grade() if state.status == "completed" else False
    traj = Trajectory.from_env_history(
        task=task,
        history=state.history,
        correct=correct,
        status=state.status,
        step_rewards=env.step_rewards,
        model_id=getattr(policy, "model_id", ""),
    )
    traj.reward = compute_episode_reward(traj, correct, env.reward_config)
    return traj


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate expert trajectories for Qwen SFT")
    parser.add_argument("--config", default="rl_training/configs/default.yaml",
                        help="Base config (env section is consumed)")
    parser.add_argument("--training-tasks", default=None,
                        help="Explicit path to training_tasks.json; if omitted we "
                             "reuse rl_training/outputs/gpt4o_pipeline/phase_a/training_tasks.json "
                             "when present, otherwise generate fresh tasks.")
    parser.add_argument("--output-dir", default="rl_training/outputs/qwen_pipeline/phase_a",
                        help="Destination for expert_trajectories.jsonl + qwen_sft_openai.jsonl")
    parser.add_argument("--expert-model", default="gpt-4o",
                        help="OpenAI chat-completions model id used as the expert policy")
    parser.add_argument("--trajectories-per-task", type=int, default=3,
                        help="Number of independent rollouts per task (filters later to correct)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature for the expert policy; 0.7 matches "
                             "the setup that produced medagent-sft-mini")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-parallel", type=int, default=5,
                        help="Concurrent OpenAI API calls per rollout batch")
    parser.add_argument("--limit-tasks", type=int, default=None,
                        help="Dev flag: only process the first N tasks")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="Skip rollouts for tasks already covered in expert_trajectories.jsonl")
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false",
                        help="Re-run even task_ids already in the store")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        logger.error(
            "OPENAI_API_KEY is not set. This script runs gpt-4o rollouts via "
            "the OpenAI API and won't work without it."
        )
        sys.exit(2)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    traj_path = out_dir / "expert_trajectories.jsonl"
    sft_path = out_dir / "qwen_sft_openai.jsonl"
    summary_path = out_dir / "generation_summary.json"

    tasks = _load_or_generate_training_tasks(config, args.training_tasks, out_dir)
    if args.limit_tasks:
        tasks = tasks[: args.limit_tasks]
        logger.info("--limit-tasks: truncated to %d tasks", len(tasks))

    store = TrajectoryStore(traj_path)
    already = {t.task_id for t in store.load_all()} if args.skip_existing else set()
    if already:
        logger.info("Skipping %d task_ids already in %s", len(already), traj_path)

    env = MedAgentEnv.from_config(config)
    policy = OpenAIPolicy(
        model_id=args.expert_model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_parallel=args.max_parallel,
    )

    from rich.progress import (
        BarColumn, MofNCompleteColumn, Progress, SpinnerColumn,
        TextColumn, TimeElapsedColumn, TimeRemainingColumn,
    )

    pending = [t for t in tasks if t.get("id") not in already]
    total_rollouts = len(pending) * args.trajectories_per_task
    correct_count = 0
    all_count = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("  correct={task.fields[correct]}"),
        TimeElapsedColumn(),
        TextColumn("ETA"),
        TimeRemainingColumn(),
    ) as progress:
        ptask = progress.add_task(
            f"Expert rollouts ({args.expert_model})",
            total=total_rollouts,
            correct=0,
        )
        for task in pending:
            for _ in range(args.trajectories_per_task):
                try:
                    traj = _rollout_task(env, task, policy)
                except Exception as exc:
                    logger.warning("Rollout for %s raised: %s", task.get("id"), exc)
                    progress.update(ptask, advance=1, correct=correct_count)
                    continue
                store.save(traj)
                all_count += 1
                if traj.correct:
                    correct_count += 1
                progress.update(ptask, advance=1, correct=correct_count)

    correct_trajs = store.filter(correct=True)
    store.export_openai_jsonl(sft_path, correct_trajs)
    logger.info(
        "Wrote %d correct trajectories to %s (in OpenAI chat format)",
        len(correct_trajs), sft_path,
    )

    summary = {
        "expert_model": args.expert_model,
        "trajectories_per_task": args.trajectories_per_task,
        "temperature": args.temperature,
        "tasks_total": len(tasks),
        "tasks_skipped_already_present": len(already),
        "new_rollouts_attempted": all_count,
        "new_rollouts_correct": correct_count,
        "correct_total_in_store": len(correct_trajs),
        "expert_trajectories_path": str(traj_path),
        "qwen_sft_openai_path": str(sft_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("Wrote generation summary to %s", summary_path)

    if correct_count == 0 and all_count > 0:
        logger.error(
            "No correct rollouts were generated this run. "
            "Check the FHIR server, task inputs, and expert model choice."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
