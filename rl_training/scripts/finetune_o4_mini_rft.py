#!/usr/bin/env python3
"""Launch the OpenAI Reinforcement Fine-Tuning (RFT) job for o4-mini.

Reads ``train.jsonl`` / ``val.jsonl`` produced by ``build_rft_dataset.py``,
validates the Python grader, submits the job, polls to completion, and writes
the resulting ``ft:...`` id + metadata into the run dir.
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

from rl_training.rft.eval_common import require_openai_key
from rl_training.training.openai_rft import (
    OpenAIRFTLauncher,
    build_python_grader,
    write_ft_outputs,
)
from rl_training.training.openai_rft_score_grader import (
    JUDGE_DEFAULT_MODEL,
    build_score_model_grader,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


GRADER_SOURCE_PATH = Path(_ROOT) / "rl_training" / "rft" / "medagent_grader.py"


def _build_grader(rft_cfg: dict) -> dict:
    grader_type = (rft_cfg.get("grader_type") or "python").lower()
    if grader_type == "python":
        return build_python_grader(
            GRADER_SOURCE_PATH,
            name="medagent_refsol",
            image_tag=rft_cfg.get("grader_image_tag", "2025-05-08"),
        )
    if grader_type == "score_model":
        return build_score_model_grader(
            name="medagent_judge",
            judge_model=rft_cfg.get("judge_model", JUDGE_DEFAULT_MODEL),
            judge_temperature=float(rft_cfg.get("judge_temperature", 0.0)),
            judge_seed=rft_cfg.get("judge_seed", 0),
            max_completion_tokens=int(rft_cfg.get("judge_max_completion_tokens", 16)),
        )
    raise SystemExit(f"Unknown grader_type={grader_type!r}; expected 'python' or 'score_model'.")


def _load_first_row(path: Path) -> dict:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                return json.loads(line)
    raise RuntimeError(f"{path} is empty")


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit o4-mini RFT job")
    parser.add_argument("--config", default="rl_training/configs/o4_mini_rft.yaml")
    parser.add_argument("--train-file", default=None,
                        help="train.jsonl path (defaults to <run_dir>/train.jsonl)")
    parser.add_argument("--val-file", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--model", default=None, help="Override base model id")
    parser.add_argument("--suffix", default=None)
    parser.add_argument("--poll-interval", type=float, default=60.0)
    parser.add_argument("--skip-validate", action="store_true",
                        help="Skip the grader validate+run smoke check")
    args = parser.parse_args()

    require_openai_key()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    run_dir = Path(args.output_dir or config["output"]["run_dir"])
    train_file = Path(args.train_file or (run_dir / "train.jsonl"))
    val_file = Path(args.val_file or (run_dir / "val.jsonl"))
    if not train_file.exists() or not val_file.exists():
        raise SystemExit(
            f"Missing RFT JSONL files: {train_file} / {val_file}. "
            "Run rl_training/scripts/build_rft_dataset.py first."
        )

    model = args.model or config["rft"]["base_model"]
    suffix = args.suffix or config["rft"]["suffix"]
    reasoning_effort = config["rft"].get("reasoning_effort", "medium")

    grader = _build_grader(config["rft"])
    logger.info("Grader: type=%s name=%s",
                grader.get("type"), grader.get("name"))

    launcher = OpenAIRFTLauncher()

    sample_row = None if args.skip_validate else _load_first_row(train_file)
    result = launcher.run(
        train_jsonl=train_file,
        val_jsonl=val_file,
        grader=grader,
        model=model,
        suffix=suffix,
        reasoning_effort=reasoning_effort,
        n_epochs=config["rft"].get("n_epochs", "auto"),
        compute_multiplier=config["rft"].get("compute_multiplier", "auto"),
        eval_interval=config["rft"].get("eval_interval", "auto"),
        eval_samples=config["rft"].get("eval_samples", "auto"),
        seed=config["rft"].get("seed", 42),
        poll_interval=args.poll_interval,
        sample_row_for_validation=sample_row,
    )

    extra = {
        "n_train": sum(1 for _ in open(train_file)),
        "n_val": sum(1 for _ in open(val_file)),
        "config_path": str(args.config),
    }
    write_ft_outputs(result=result, run_dir=run_dir, extra_meta=extra)

    if result.status != "succeeded":
        raise SystemExit(f"RFT job ended with status={result.status}: {result.events_tail[-1] if result.events_tail else ''}")
    if not result.fine_tuned_model:
        raise SystemExit("RFT job succeeded but fine_tuned_model is empty.")
    print(result.fine_tuned_model)


if __name__ == "__main__":
    main()
