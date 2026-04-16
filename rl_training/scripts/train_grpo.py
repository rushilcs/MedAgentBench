#!/usr/bin/env python3
"""Phase B: GRPO training with TRL GRPOTrainer.

Runs true reinforcement learning: the model generates tool calls,
the ``MedAgentBenchEnv`` executes them against the FHIR server,
reward functions score the result, and GRPO updates the weights.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
from peft import LoraConfig
from transformers import BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer

from rl_training.env.trl_env import MedAgentBenchEnv
from rl_training.env.trl_rewards import (
    correctness_reward,
    efficiency_reward,
    tool_usage_reward,
)
from rl_training.data.prepare_dataset import tasks_to_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase B: GRPO training")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct",
                        help="Base model or SFT checkpoint path")
    parser.add_argument("--training-tasks", required=True,
                        help="Path to training_tasks.json")
    parser.add_argument("--output-dir", default="rl_training/outputs/grpo_checkpoint")
    parser.add_argument("--fhir-base", default="http://localhost:8080/fhir/")
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--num-generations", type=int, default=4,
                        help="G: completions per prompt for GRPO")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--max-completion-length", type=int, default=2048)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--use-vllm", action="store_true",
                        help="Use vLLM for faster generation (needs more VRAM)")
    parser.add_argument("--save-steps", type=int, default=50)
    args = parser.parse_args()

    os.environ["FHIR_API_BASE"] = args.fhir_base

    # Load training tasks
    with open(args.training_tasks) as f:
        tasks = json.load(f)
    dataset = tasks_to_dataset(tasks, args.fhir_base)
    print(f"Loaded {len(dataset)} training tasks")

    # QLoRA
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
        lora_dropout=0.05,
    )

    # GRPO config
    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        learning_rate=args.lr,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        save_steps=args.save_steps,
        logging_steps=1,
        log_completions=True,
        report_to="none",
        use_vllm=args.use_vllm,
    )

    if args.use_vllm:
        grpo_config.vllm_mode = "colocate"

    # Model kwargs
    model_kwargs = {}
    if not args.no_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    trainer = GRPOTrainer(
        model=args.model,
        args=grpo_config,
        peft_config=peft_config,
        train_dataset=dataset,
        reward_funcs=[correctness_reward, efficiency_reward, tool_usage_reward],
        environment_factory=MedAgentBenchEnv,
        model_init_kwargs=model_kwargs,
    )

    print(f"Starting GRPO training")
    print(f"  Model: {args.model}")
    print(f"  Steps: {args.max_steps}")
    print(f"  Generations per prompt: {args.num_generations}")
    print(f"  vLLM: {args.use_vllm}")
    print(f"  FHIR server: {args.fhir_base}")

    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"GRPO checkpoint saved to {args.output_dir}")


if __name__ == "__main__":
    main()
