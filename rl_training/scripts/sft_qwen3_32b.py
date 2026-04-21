#!/usr/bin/env python3
"""QLoRA SFT on Qwen/Qwen3-32B-Instruct with TRL's ``SFTTrainer``.

This script is the Qwen analog of the gpt-4o-mini SFT job: it warm-starts
the policy on expert trajectories before GRPO sees it. The important bits:

  * **Base model:** ``Qwen/Qwen3-32B-Instruct`` with a fallback to
    ``Qwen/Qwen3-32B`` (same resolver as ``train_grpo_32b.py``).
  * **QLoRA:** NF4 4-bit quantization + LoRA (``r=16``, ``alpha=32``,
    ``target_modules="all-linear"``, bf16 compute). Same shape as GRPO.
  * **Chat template:** ``SFTTrainer`` auto-applies the tokenizer's chat
    template when the dataset has a ``messages`` column. We load
    ``qwen_sft_openai.jsonl`` produced by
    ``generate_qwen_sft_expert_trajectories.py`` directly.
  * **Completion-only loss:** ``completion_only_loss=True`` masks the
    user/system turns so we only train on assistant tokens. Matches what
    OpenAI's SFT does internally.
  * **Callbacks:** reuses the same ``ProgressCallback`` /
    ``HeartbeatCallback`` / ``CloudSyncCallback`` we built for GRPO so
    progress, watchdog, and cloud-resume behave identically.

Typical on-pod invocation (from ``launch_sft.sh``):
    CUDA_VISIBLE_DEVICES=0 python rl_training/scripts/sft_qwen3_32b.py \\
        --config rl_training/configs/qwen3_32b_sft.yaml \\
        --resume-from-checkpoint auto

Dry-run / local validation (no GPU required):
    python rl_training/scripts/sft_qwen3_32b.py \\
        --config rl_training/configs/qwen3_32b_sft.yaml \\
        --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import yaml

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

logger = logging.getLogger(__name__)


def _resolve_model_id(primary: str, fallback: str) -> str:
    """Return ``primary`` if it resolves on the Hub, otherwise ``fallback``.

    Local paths (e.g. ``/workspace/qwen3_32b_sft_merged``) are passed
    through unchanged.
    """
    if os.path.isdir(primary) or primary.startswith("/"):
        return primary
    try:
        from huggingface_hub import model_info
        model_info(primary)
        return primary
    except Exception as exc:
        logger.warning(
            "Primary model %s not resolvable (%s); using fallback %s",
            primary, exc, fallback,
        )
        return fallback


def _build_peft_config(qlora_cfg: dict[str, Any]):
    from peft import LoraConfig
    return LoraConfig(
        r=qlora_cfg.get("r", 16),
        lora_alpha=qlora_cfg.get("alpha", 32),
        lora_dropout=qlora_cfg.get("dropout", 0.05),
        target_modules=qlora_cfg.get("target_modules", "all-linear"),
        task_type="CAUSAL_LM",
    )


def _build_quant_config(qlora_cfg: dict[str, Any]):
    import torch
    from transformers import BitsAndBytesConfig
    compute_dtype = torch.bfloat16
    if qlora_cfg.get("bnb_4bit_compute_dtype") == "float16":
        compute_dtype = torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=qlora_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=qlora_cfg.get("bnb_4bit_use_double_quant", True),
    )


def _read_sft_jsonl_rows(jsonl_path: str) -> list[dict[str, Any]]:
    """Parse an OpenAI-format ``{"messages":[...]}`` JSONL into plain Python.

    Split out from ``_load_sft_dataset`` so the dry-run path doesn't need
    the ``datasets`` package.
    """
    rows: list[dict[str, Any]] = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            msgs = obj.get("messages")
            if not msgs:
                continue
            rows.append({"messages": msgs})
    if not rows:
        raise RuntimeError(f"No usable rows in {jsonl_path}")
    return rows


def _load_sft_dataset(jsonl_path: str):
    """Load an OpenAI-format ``{"messages":[...]}`` JSONL as a HF Dataset.

    ``SFTTrainer`` + a tokenizer with ``chat_template`` set will apply
    the Qwen3 chat formatting automatically as long as the column is
    literally called ``messages``.
    """
    from datasets import Dataset
    rows = _read_sft_jsonl_rows(jsonl_path)
    return Dataset.from_list(rows)


def _resolve_resume_checkpoint(
    flag: str | None, cloud_cfg: dict[str, Any], output_dir: str,
) -> str | None:
    """Translate ``--resume-from-checkpoint`` into a value TRL understands.

    Same semantics as ``train_grpo_32b.py``.
    """
    if not flag:
        return None
    if flag != "auto":
        return flag

    if cloud_cfg.get("enabled"):
        try:
            from rl_training.scripts.resume_from_cloud import fetch_latest_checkpoint
            local = fetch_latest_checkpoint(
                bucket=cloud_cfg["bucket"],
                prefix=cloud_cfg["prefix"],
                output_dir=output_dir,
            )
            if local:
                logger.info("Auto-resume pulled checkpoint from cloud: %s", local)
                return local
        except Exception as exc:
            logger.warning("Auto-resume cloud pull failed: %s", exc)

    if os.path.isdir(output_dir):
        candidates = sorted(
            (d for d in Path(output_dir).glob("checkpoint-*") if d.is_dir()),
            key=lambda d: int(d.name.split("-")[-1]),
        )
        if candidates:
            logger.info("Auto-resume using local checkpoint: %s", candidates[-1])
            return str(candidates[-1])
    logger.info("Auto-resume requested but no checkpoint found; starting fresh.")
    return None


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="QLoRA SFT on Qwen3-32B")
    parser.add_argument("--config", required=True,
                        help="YAML config (qwen3_32b_sft.yaml)")
    parser.add_argument("--sft-jsonl", default=None,
                        help="Override data.sft_jsonl in config")
    parser.add_argument("--output-dir", default=None,
                        help="Override output.dir in config")
    parser.add_argument("--num-train-epochs", type=float, default=None,
                        help="Override sft.num_train_epochs")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="If >0, cap total optimizer steps (overrides epochs)")
    parser.add_argument("--resume-from-checkpoint", default=None,
                        help="Path, or 'auto' to resume from latest cloud/local ckpt")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate config + dataset + model resolution without training")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    output_dir = args.output_dir or cfg["output"]["dir"]
    sft_jsonl = args.sft_jsonl or cfg["data"]["sft_jsonl"]
    sft_cfg = cfg.get("sft", {})

    # Resolve model id (Qwen3-32B-Instruct, fallback, or local dir)
    model_id = _resolve_model_id(
        cfg["model"]["name_or_path"],
        cfg["model"].get("fallback", cfg["model"]["name_or_path"]),
    )
    logger.info("Resolved model_id: %s", model_id)

    # Dataset
    if not os.path.exists(sft_jsonl):
        parser.error(
            f"SFT jsonl not found at {sft_jsonl}. Run "
            f"generate_qwen_sft_expert_trajectories.py first."
        )

    if args.dry_run:
        rows = _read_sft_jsonl_rows(sft_jsonl)
        logger.info(
            "--dry-run: model=%s rows=%d output=%s epochs=%s max_seq_len=%s",
            model_id, len(rows), output_dir,
            args.num_train_epochs or sft_cfg.get("num_train_epochs", 3),
            sft_cfg.get("max_seq_length", 4096),
        )
        sample = rows[0]["messages"][:2]
        logger.info("First example preview: %s", json.dumps(sample)[:400])
        return

    dataset = _load_sft_dataset(sft_jsonl)
    logger.info("Loaded %d SFT examples from %s", len(dataset), sft_jsonl)

    # --- Heavy imports only past the dry-run gate ---
    from trl import SFTConfig, SFTTrainer
    from transformers import AutoTokenizer

    peft_config = _build_peft_config(cfg.get("qlora", {}))
    quant_config = _build_quant_config(cfg.get("qlora", {}))

    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=args.num_train_epochs or sft_cfg.get("num_train_epochs", 3),
        max_steps=args.max_steps if args.max_steps and args.max_steps > 0 else -1,
        per_device_train_batch_size=sft_cfg.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=sft_cfg.get("gradient_accumulation_steps", 8),
        learning_rate=sft_cfg.get("learning_rate", 1e-5),
        lr_scheduler_type=sft_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=sft_cfg.get("warmup_ratio", 0.03),
        weight_decay=sft_cfg.get("weight_decay", 0.0),
        max_length=sft_cfg.get("max_seq_length", 4096),  # trl>=0.17 renamed max_seq_length -> max_length
        packing=sft_cfg.get("packing", False),
        completion_only_loss=sft_cfg.get("completion_only_loss", True),
        bf16=sft_cfg.get("bf16", True),
        gradient_checkpointing=sft_cfg.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        save_steps=sft_cfg.get("save_steps", 100),
        save_strategy=sft_cfg.get("save_strategy", "steps"),
        logging_steps=sft_cfg.get("logging_steps", 10),
        seed=sft_cfg.get("seed", 42),
        report_to="none",
        model_init_kwargs={
            "quantization_config": quant_config,
            "torch_dtype": cfg["model"].get("torch_dtype", "bfloat16"),
            "attn_implementation": cfg["model"].get(
                "attn_implementation", "sdpa",
            ),
        },
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Force-disable Qwen3 "thinking" mode in the chat template. Otherwise
    # tokenizer.apply_chat_template (called by SFTTrainer for the messages
    # column) inserts <think>...</think> blocks; the model then learns to
    # emit thinking tokens at inference, which corrupts our action parser
    # and inflates token usage. Implemented as a wrapper so it composes
    # with whatever default chat template Qwen ships.
    if sft_cfg.get("disable_qwen_thinking", True):
        _orig_apply = tokenizer.apply_chat_template
        def _apply_no_thinking(*p_args, **p_kwargs):  # type: ignore[no-redef]
            p_kwargs.setdefault("enable_thinking", False)
            return _orig_apply(*p_args, **p_kwargs)
        tokenizer.apply_chat_template = _apply_no_thinking  # type: ignore[assignment]
        logger.info("Patched tokenizer.apply_chat_template to enable_thinking=False")
        # Sanity-format the first row so we surface any template errors now.
        try:
            sample_msgs = dataset[0]["messages"]
            sample_text = tokenizer.apply_chat_template(
                sample_msgs, tokenize=False, add_generation_prompt=False,
            )
            has_think = "<think>" in sample_text
            logger.info(
                "Chat-template sanity: len=%d chars, contains <think>=%s",
                len(sample_text), has_think,
            )
        except Exception as exc:
            logger.warning("Chat-template sanity probe failed: %s", exc)

    # For ETA display: if the user passed --max-steps we use it; otherwise
    # estimate steps = ceil(epochs * rows / (bs * grad_accum)).
    if args.max_steps and args.max_steps > 0:
        est_steps = args.max_steps
    else:
        steps_per_epoch = max(
            1,
            len(dataset) // (
                sft_cfg.get("per_device_train_batch_size", 1)
                * sft_cfg.get("gradient_accumulation_steps", 8)
            ),
        )
        est_steps = int(
            (args.num_train_epochs or sft_cfg.get("num_train_epochs", 3))
            * steps_per_epoch
        )
    logger.info("Estimated total optimizer steps: %d", est_steps)

    # Callbacks (same three as GRPO; all graceful no-ops otherwise)
    callbacks: list[Any] = []
    try:
        from rl_training.training.progress_callback import ProgressCallback
        callbacks.append(ProgressCallback(
            output_dir=output_dir, max_steps=est_steps,
        ))
    except Exception as exc:
        logger.warning("ProgressCallback unavailable: %s", exc)

    try:
        from rl_training.training.heartbeat import HeartbeatCallback
        callbacks.append(HeartbeatCallback(
            heartbeat_path=cfg.get("resilience", {}).get(
                "heartbeat_path", "/tmp/trainer_heartbeat",
            ),
            ntfy_topic=cfg.get("resilience", {}).get("ntfy_topic") or None,
        ))
    except Exception as exc:
        logger.warning("HeartbeatCallback unavailable: %s", exc)

    cloud_cfg = cfg.get("output", {}).get("cloud_sync", {})
    if cloud_cfg.get("enabled"):
        try:
            from rl_training.training.checkpoint_sync import CloudSyncCallback
            callbacks.append(CloudSyncCallback(
                backend=cloud_cfg.get("backend", "b2"),
                bucket=cloud_cfg["bucket"],
                prefix=cloud_cfg["prefix"],
                keep_last=cfg["output"].get("checkpoint_keep_last", 3),
                progress_jsonl=os.path.join(output_dir, "progress.jsonl"),
            ))
        except Exception as exc:
            logger.warning("CloudSyncCallback unavailable: %s", exc)

    resume = _resolve_resume_checkpoint(
        args.resume_from_checkpoint, cloud_cfg, output_dir,
    )

    trainer = SFTTrainer(
        model=model_id,
        args=sft_config,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        callbacks=callbacks or None,
    )

    logger.info(
        "Starting SFT on %s | epochs=%s | bs=%d | grad_accum=%d | max_seq_len=%d",
        model_id,
        args.num_train_epochs or sft_cfg.get("num_train_epochs", 3),
        sft_cfg.get("per_device_train_batch_size", 1),
        sft_cfg.get("gradient_accumulation_steps", 8),
        sft_cfg.get("max_seq_length", 4096),
    )

    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(output_dir)
    logger.info("Saved final LoRA adapter to %s", output_dir)


if __name__ == "__main__":
    main()
