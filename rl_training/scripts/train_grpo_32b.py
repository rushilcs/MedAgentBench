#!/usr/bin/env python3
"""GRPO training on Qwen/Qwen3-32B-Instruct for MedAgentBench.

This is the 32B-specific launcher. It differs from
``rl_training/scripts/train_grpo.py`` in four ways:

  1. **Model resolution.** The HF repo ``Qwen/Qwen3-32B-Instruct`` is the
     primary target; if it 404s on the Hub, the loader falls back to
     ``Qwen/Qwen3-32B`` and logs the swap. This is the one-line decision
     captured in the plan.

  2. **vLLM server-mode rollouts.** The trainer posts generation requests to a
     vLLM OpenAI-compatible server on ``cuda:1`` rather than generating
     in-process via HF. See ``launch_vllm_server.sh``.

  3. **Clinical reward toggles.** Config (`rewards.*_enabled`) decides which
     of ``temporal_grounding_reward``, ``risk_calibrated_deferral_reward``, and
     ``decision_density_reward`` are registered with the trainer.

  4. **Resilience.** ``save_steps=10`` (config-driven), plus three callbacks:
       * ``ProgressCallback`` - rich progress bar + JSONL log
       * ``CloudSyncCallback`` - uploads LoRA adapter to B2 after each save
       * ``HeartbeatCallback`` - touches /tmp/trainer_heartbeat for the watchdog

All four are off or no-ops if the underlying pre-reqs aren't present so the
script still runs in a dev environment with only a local GPU. On RunPod with
B2 credentials and ``NTFY_TOPIC`` set, everything activates.

Resume:
  Pass ``--resume-from-checkpoint auto`` to pull the latest LoRA checkpoint
  from cloud and continue from there. On first run, "auto" is a no-op.

Typical on-box invocation (from ``launch_runpod.sh``):
    With ``vllm.use_vllm=true``: GPU0 trains, GPU1 serves ``trl vllm-serve``.
    With ``vllm.use_vllm=false``: ``launch_runpod.sh`` runs 2-process DDP via
    ``python -m torch.distributed.run --nproc_per_node=2`` on ``CUDA_VISIBLE_DEVICES=0,1``
    (override with ``GRPO_TORCHRUN_PROCS=1`` for single-GPU).

    Single-GPU manual run::
        CUDA_VISIBLE_DEVICES=0 python rl_training/scripts/train_grpo_32b.py \\
            --config rl_training/configs/qwen3_32b_grpo_clinical.yaml \\
            --training-tasks rl_training/outputs/training_tasks.json \\
            --resume-from-checkpoint auto
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


def _resolve_model_id(primary: str, fallback: str, skip_hub_check: bool = False) -> str:
    """Return ``primary`` if it's usable, otherwise ``fallback``.

    Local paths (existing directories or any absolute path, e.g.
    ``/workspace/qwen3_32b_sft_merged``) are passed through without a
    Hub lookup - this is what lets the SFT->GRPO handoff work without
    the trainer trying to resolve the merged dir against huggingface.co.
    """
    if skip_hub_check or os.path.isdir(primary) or primary.startswith("/"):
        return primary
    try:
        from huggingface_hub import model_info
        model_info(primary)
        return primary
    except Exception as exc:
        logger.warning("Primary model %s not resolvable (%s); using fallback %s",
                       primary, exc, fallback)
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


def _resolve_resume_checkpoint(
    flag: str | None, cloud_cfg: dict[str, Any], output_dir: str,
) -> str | bool | None:
    """Translate ``--resume-from-checkpoint`` into a value TRL understands.

    - ``None`` -> no resume
    - ``"auto"`` -> try to pull from cloud, fall back to latest local
    - any path -> pass through literally
    """
    if not flag:
        return None
    if flag != "auto":
        return flag

    # Auto-resume: first try to sync the latest remote checkpoint down.
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

    # Fall back to the most recent local "checkpoint-*" dir.
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

    parser = argparse.ArgumentParser(description="GRPO training on Qwen3-32B")
    parser.add_argument("--config", required=True,
                        help="YAML config (qwen3_32b_grpo.yaml or _clinical.yaml)")
    parser.add_argument("--training-tasks", required=True,
                        help="Path to training_tasks.json")
    parser.add_argument("--output-dir", default=None,
                        help="Override config's output.dir")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override config's grpo.max_steps")
    parser.add_argument("--resume-from-checkpoint", default=None,
                        help="Path, or 'auto' to resume from latest cloud/local ckpt")
    parser.add_argument("--base-model", default=None,
                        help="Override model.name_or_path (local path or HF repo id). "
                             "Use this after merge_lora.py to point at the SFT-merged dir.")
    parser.add_argument("--skip-model-info-check", action="store_true",
                        help="Skip the huggingface_hub.model_info() probe. Required for "
                             "local paths like /workspace/qwen3_32b_sft_merged.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate config + dataset + rewards without training")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    output_dir = args.output_dir or cfg["output"]["dir"]
    max_steps = args.max_steps or cfg["grpo"]["max_steps"]
    os.environ["FHIR_API_BASE"] = cfg["env"]["fhir_api_base"]

    # Resolve model id (Qwen3-32B-Instruct vs fallback, or a local merged dir)
    primary = args.base_model or cfg["model"]["name_or_path"]
    model_id = _resolve_model_id(
        primary,
        cfg["model"].get("fallback", primary),
        skip_hub_check=args.skip_model_info_check,
    )
    logger.info("Resolved model_id: %s", model_id)

    # Install global FHIR snapshot if configured
    if cfg["env"].get("use_fhir_snapshot"):
        from rl_training.env.fhir_snapshot import (
            FhirSnapshot, install_global_snapshot,
        )
        snap_path = cfg["env"]["snapshot_path"]
        mode = "replay" if cfg["env"].get("snapshot_fallthrough", True) else "replay"
        snap = FhirSnapshot(
            mode=mode,
            path=snap_path,
            fallthrough=bool(cfg["env"].get("snapshot_fallthrough", True)),
        )
        install_global_snapshot(snap)
        logger.info("Installed FHIR snapshot: %s (%d cached rows)",
                    snap_path, len(snap._cache))  # noqa: SLF001

        # refsol graders call ``utils.send_get_request`` directly, which bypasses
        # ``MedAgentBenchEnv`` snapshot routing. Patch the same trio as
        # ``run_post_train_eval.py`` so terminal reward matches benchmark task
        # grading without requiring a live Docker FHIR on localhost:8080.
        def _patched_send_get(url, params=None, headers=None):  # noqa: ARG001
            # refsol does ``json.loads(send_get_request(...)['data'])`` because
            # the original utils returns text for FHIR's application/fhir+json.
            # Snapshot stores ``data`` parsed (dict); re-serialize so refsol's
            # json.loads works unmodified. Without this every refsol GET path
            # raised TypeError → silent grader failure → r_succ never fired.
            res = snap.send_get_request(url)
            if "data" in res and not isinstance(res["data"], str):
                res = {**res, "data": json.dumps(res["data"])}
            return res

        import src.server.tasks.medagentbench.utils as _mb_utils

        _mb_utils.send_get_request = _patched_send_get
        _refsol_mod = sys.modules.get("src.server.tasks.medagentbench.refsol")
        if _refsol_mod is not None:
            _refsol_mod.send_get_request = _patched_send_get
        _med_env_mod = sys.modules.get("rl_training.env.medagent_env")
        if _med_env_mod is not None:
            _med_env_mod.send_get_request = _patched_send_get
        logger.info(
            "Patched medagentbench send_get_request for refsol + medagent_env "
            "(FHIR snapshot replay, fallthrough=%s)",
            bool(cfg["env"].get("snapshot_fallthrough", True)),
        )

    # Load training tasks
    with open(args.training_tasks) as f:
        tasks = json.load(f)
    cur_cfg = (cfg.get("benchmark_reward") or {}).get("curriculum") or cfg.get(
        "curriculum",
    )
    if isinstance(cur_cfg, dict) and cur_cfg.get("enabled"):
        from rl_training.data.curriculum import apply_soft_curriculum_mix

        qfrac = float(cur_cfg.get("query_target_fraction", 0.7))
        cseed = int(cur_cfg.get("seed", cfg.get("data", {}).get("seed", 42)))
        tasks = apply_soft_curriculum_mix(tasks, qfrac, seed=cseed)
        logger.info(
            "Soft curriculum: query_target_fraction=%.2f seed=%d (len=%d)",
            qfrac, cseed, len(tasks),
        )
    from rl_training.data.prepare_dataset import tasks_to_dataset
    dataset = tasks_to_dataset(tasks, cfg["env"]["fhir_api_base"])
    logger.info("Loaded %d training tasks", len(dataset))

    from rl_training.rl import medagent_reward as mr

    br = cfg.get("benchmark_reward") or {}
    mr.configure(br)
    if br.get("fsm_constrained_decode"):
        os.environ["MEDAGENT_RL_FSM_DECODE"] = "1"
    benchmark_aligned = bool(
        br.get("enabled", False)
        or (cfg.get("rewards") or {}).get("benchmark_aligned_enabled", False),
    )

    # Build reward list from config
    from rl_training.env.trl_rewards_clinical import register_rewards
    reward_funcs = register_rewards(cfg["rewards"], benchmark_aligned=benchmark_aligned)
    logger.info("Registered %d reward functions: %s",
                len(reward_funcs), [fn.__name__ for fn in reward_funcs])

    if args.dry_run:
        logger.info("--dry-run: model=%s tasks=%d rewards=%d output=%s max_steps=%d",
                    model_id, len(dataset), len(reward_funcs), output_dir, max_steps)
        return

    # --- Heavy imports only past the dry-run gate ---
    from trl import GRPOConfig, GRPOTrainer
    from rl_training.env.trl_env import MedAgentBenchEnv

    peft_config = _build_peft_config(cfg["qlora"])
    quant_config = _build_quant_config(cfg["qlora"])

    vllm_cfg = cfg.get("vllm", {})
    grpo_config = GRPOConfig(
        output_dir=output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=cfg["grpo"]["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["grpo"]["gradient_accumulation_steps"],
        num_generations=cfg["grpo"]["num_generations"],
        # TRL 1.x removed max_prompt_length; prompt length is now bounded by
        # vllm_max_model_length minus max_completion_length, plus tokenizer
        # truncation. Read max_prompt_length from config purely for logging.
        max_completion_length=cfg["grpo"]["max_completion_length"],
        learning_rate=cfg["grpo"]["learning_rate"],
        beta=cfg["grpo"]["beta"],
        epsilon=cfg["grpo"].get("epsilon", 0.2),
        temperature=cfg["grpo"]["temperature"],
        top_p=cfg["grpo"].get("top_p", 0.95),
        bf16=cfg["grpo"].get("bf16", True),
        gradient_checkpointing=cfg["grpo"].get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        save_steps=cfg["grpo"]["save_steps"],
        logging_steps=cfg["grpo"].get("logging_steps", 1),
        log_completions=cfg["grpo"].get("log_completions", True),
        warmup_ratio=cfg["grpo"].get("warmup_ratio", 0.0),
        weight_decay=cfg["grpo"].get("weight_decay", 0.0),
        lr_scheduler_type=cfg["grpo"].get("lr_scheduler_type", "cosine"),
        seed=cfg["grpo"].get("seed", 42),
        use_vllm=vllm_cfg.get("use_vllm", True),
        report_to="none",
        model_init_kwargs={
            "quantization_config": quant_config,
            "torch_dtype": cfg["model"].get("torch_dtype", "bfloat16"),
            "attn_implementation": cfg["model"].get(
                "attn_implementation", "sdpa",
            ),
        },
    )

    if vllm_cfg.get("use_vllm", True):
        grpo_config.vllm_mode = vllm_cfg.get("mode", "server")
        host = vllm_cfg.get("host", "127.0.0.1")
        port = vllm_cfg.get("port", 8000)
        # TRL 0.19+ accepts these; older versions ignore unknown attrs safely
        setattr(grpo_config, "vllm_server_host", host)
        setattr(grpo_config, "vllm_server_port", port)

    # Callbacks (all graceful no-ops when their deps aren't available)
    callbacks: list[Any] = []
    try:
        from rl_training.training.progress_callback import ProgressCallback
        callbacks.append(ProgressCallback(
            output_dir=output_dir, max_steps=max_steps,
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

    trainer = GRPOTrainer(
        model=model_id,
        args=grpo_config,
        peft_config=peft_config,
        train_dataset=dataset,
        reward_funcs=reward_funcs,
        environment_factory=MedAgentBenchEnv,
        callbacks=callbacks or None,
    )

    logger.info(
        "Starting GRPO on %s | steps=%d | G=%d | bs=%d | grad_accum=%d | vllm=%s",
        model_id, max_steps, cfg["grpo"]["num_generations"],
        cfg["grpo"]["per_device_train_batch_size"],
        cfg["grpo"]["gradient_accumulation_steps"],
        grpo_config.use_vllm,
    )

    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(output_dir)
    logger.info("Saved final LoRA adapter to %s", output_dir)


if __name__ == "__main__":
    main()
