#!/usr/bin/env python3
"""Phase A: Supervised Fine-Tuning with TRL SFTTrainer + QLoRA.

Trains a Qwen model on expert trajectories (correct tool-calling
conversations) before the GRPO reinforcement-learning phase.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from datasets import Dataset
from peft import LoraConfig
from transformers import BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase A: SFT on expert trajectories")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct",
                        help="Base model name or path")
    parser.add_argument("--trajectories", required=True,
                        help="Path to expert_trajectories.jsonl (from ExpertCollector)")
    parser.add_argument("--output-dir", default="rl_training/outputs/sft_checkpoint")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--no-4bit", action="store_true",
                        help="Disable 4-bit quantisation (uses more VRAM)")
    args = parser.parse_args()

    # Load expert trajectories as SFT dataset
    from rl_training.data.prepare_dataset import expert_trajectories_to_sft_dataset
    dataset = expert_trajectories_to_sft_dataset(args.trajectories)
    print(f"Loaded {len(dataset)} expert trajectories for SFT")

    # QLoRA config
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
        lora_dropout=0.05,
    )

    # Quantization config
    quant_config = None
    if not args.no_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    # SFT config
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        max_seq_length=args.max_seq_length,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
    )

    # Build trainer
    model_kwargs = {}
    if quant_config:
        model_kwargs["quantization_config"] = quant_config

    trainer = SFTTrainer(
        model=args.model,
        args=sft_config,
        train_dataset=dataset,
        peft_config=peft_config,
        model_init_kwargs=model_kwargs,
    )

    print(f"Starting SFT training: {args.model}")
    print(f"  LoRA rank: {args.lora_r}")
    print(f"  Epochs: {args.epochs}")
    print(f"  4-bit: {not args.no_4bit}")
    trainer.train()

    trainer.save_model(args.output_dir)
    print(f"SFT checkpoint saved to {args.output_dir}")


if __name__ == "__main__":
    main()
