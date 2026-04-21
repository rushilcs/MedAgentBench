#!/usr/bin/env python3
"""Submit OpenAI fine-tuning for gpt-4o-mini-2024-07-18 from saved expert trajectories."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from rl_training.data.trajectory_store import TrajectoryStore
from rl_training.training.openai_finetune import OpenAIFineTuner

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trajectories",
        default="rl_training/outputs/gpt4o_pipeline/phase_a/expert_trajectories.jsonl",
        help="JSONL of expert trajectories (OpenAI chat format)",
    )
    parser.add_argument("--base-model", default="gpt-4o-mini-2024-07-18")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--suffix", default="medagent-sft-mini")
    parser.add_argument(
        "--output-id-file",
        default="rl_training/outputs/gpt4o_mini_user_run/finetuned_model_id.txt",
        help="Where to write the ft:... model id when done",
    )
    args = parser.parse_args()

    store = TrajectoryStore(args.trajectories)
    trajs = [t for t in store.load_all() if t.correct]
    logger.info("Loaded %d correct trajectories from %s", len(trajs), args.trajectories)
    if not trajs:
        raise SystemExit("No trajectories to fine-tune on.")

    ft = OpenAIFineTuner()
    model_id = ft.run(
        trajectories=trajs,
        base_model=args.base_model,
        suffix=args.suffix,
        n_epochs=args.epochs,
        output_dir="rl_training/outputs/ft_data",
    )
    logger.info("Fine-tuned model: %s", model_id)

    out = Path(args.output_id_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(model_id.strip() + "\n")
    meta = out.with_suffix(".json")
    meta.write_text(
        json.dumps(
            {
                "fine_tuned_model": model_id,
                "base_model": args.base_model,
                "n_epochs": args.epochs,
                "n_trajectories": len(trajs),
                "source_jsonl": str(args.trajectories),
            },
            indent=2,
        )
        + "\n"
    )
    print(model_id)


if __name__ == "__main__":
    main()
