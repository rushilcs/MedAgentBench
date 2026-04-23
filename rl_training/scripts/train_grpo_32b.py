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
            FhirSnapshot, _default_live_getter, install_global_snapshot,
        )
        snap_path = cfg["env"]["snapshot_path"]
        mode = "replay" if cfg["env"].get("snapshot_fallthrough", True) else "replay"
        # Diagnostic miss-log: every cache miss is appended (deduped) to
        # snapshot_misses.jsonl. With fallthrough on we still record what
        # the policy hits live so we can grow the offline snapshot.
        miss_log = os.path.join(
            cfg.get("output_dir", "rl_training/outputs/qwen3_32b_grpo_v2"),
            "snapshot_misses.jsonl",
        )

        # Cache-miss fallthrough router. The model's prompt always says
        # ``http://localhost:8080/fhir/...`` so every URL the policy emits
        # is keyed by that netloc. When FHIR_LIVE_BASE_OVERRIDE is set
        # (e.g. a Cloudflare Tunnel pointing at the dev box's docker FHIR)
        # we rewrite the netloc on the *live* hop only -- cache keys stay
        # ``localhost:8080`` so the static snapshot built locally still
        # serves hits, and only true misses traverse the tunnel.
        live_override = os.environ.get("FHIR_LIVE_BASE_OVERRIDE", "").strip()

        def _live_router(url: str) -> dict:
            target = url
            if live_override and "localhost:8080/fhir" in url:
                target = url.replace("http://localhost:8080/fhir",
                                     live_override.rstrip("/") + "/fhir")
            return _default_live_getter(target)

        snap = FhirSnapshot(
            mode=mode,
            path=snap_path,
            fallthrough=bool(cfg["env"].get("snapshot_fallthrough", True)),
            miss_log_path=miss_log,
            live_getter=_live_router,
        )
        install_global_snapshot(snap)
        logger.info(
            "Installed FHIR snapshot: %s (%d cached rows); miss log: %s; live override: %s",
            snap_path, len(snap._cache), miss_log,  # noqa: SLF001
            live_override or "<none>",
        )

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

    # Plan §0.3a: exclude the held-out validation task IDs from the GRPO
    # training pool so we never train on val examples. Missing val file is
    # not fatal — just logs a warning.
    val_path = (cfg.get("data") or {}).get("validation_tasks_path")
    val_ids: set[str] = set()
    if val_path and os.path.exists(val_path):
        try:
            with open(val_path) as fh:
                val_ids = {str(t.get("id", "")) for t in json.load(fh)}
            before = len(tasks)
            tasks = [t for t in tasks if str(t.get("id", "")) not in val_ids]
            logger.info(
                "Excluded %d held-out validation IDs from training pool (%d -> %d)",
                len(val_ids), before, len(tasks),
            )
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to load validation set %s: %s", val_path, exc)
    elif val_path:
        logger.warning(
            "Validation tasks file %s not found; training pool is NOT filtered. "
            "This means mid-train val eval will overlap with training data.",
            val_path,
        )

    cur_cfg = (cfg.get("benchmark_reward") or {}).get("curriculum") or cfg.get(
        "curriculum",
    )
    used_two_phase = False
    if isinstance(cur_cfg, dict) and cur_cfg.get("enabled"):
        cur_mode = str(cur_cfg.get("mode") or "soft").lower()
        cseed = int(cur_cfg.get("seed", cfg.get("data", {}).get("seed", 42)))
        if cur_mode == "two_phase":
            from rl_training.data.curriculum import two_phase_materialise

            prompts_per_step = (
                int(cfg["grpo"]["per_device_train_batch_size"])
                * int(cfg["grpo"]["gradient_accumulation_steps"])
            )
            phase_b_start = int(cur_cfg.get("phase_b_start_step", 100))
            total_prompts = max_steps * prompts_per_step
            phase_a_prompts = min(total_prompts, phase_b_start * prompts_per_step)
            tasks = two_phase_materialise(
                tasks,
                total_prompts=total_prompts,
                phase_a_prompts=phase_a_prompts,
                phase_a_weights=cur_cfg.get("phase_a_weights"),
                phase_b_weights=cur_cfg.get("phase_b_weights"),
                v1_rollouts_path=cur_cfg.get("v1_rollouts_path"),
                v1_eval_fallback_path=cur_cfg.get("v1_eval_fallback_path"),
                seed=cseed,
            )
            used_two_phase = True
            logger.info(
                "Two-phase curriculum: max_steps=%d prompts_per_step=%d "
                "phase_b_start=%d total=%d (Phase A %d prompts)",
                max_steps, prompts_per_step, phase_b_start,
                total_prompts, phase_a_prompts,
            )
        else:
            from rl_training.data.curriculum import apply_soft_curriculum_mix

            qfrac = float(cur_cfg.get("query_target_fraction", 0.7))
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
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    # ``env.action_format: plain_text`` selects the MedAgentBench paper's
    # GET/POST/FINISH interface (matches our 75% SFT v2 baseline + the
    # Claude 3.5 Sonnet 69.67% leaderboard number). Anything else falls back
    # to the legacy JSON-tool ``MedAgentBenchEnv`` flow.
    action_format = (cfg.get("env", {}) or {}).get("action_format", "json_tool")
    use_plain_rollout = action_format == "plain_text"

    # FIX A1: Disable Qwen3 thinking mode in GRPO rollouts.
    # SFT v2 was trained with enable_thinking=False (see sft_qwen3_32b.py).
    # Without this patch, GRPO renders prompts with thinking enabled and the
    # policy spends most of its completion budget inside <think> blocks
    # without ever emitting a tool call -> 0-1 step rollouts, avg_correct=0.
    # Eval (run_post_train_eval.py --enable-thinking false) confirms SFT v2
    # only behaves correctly with thinking off, so GRPO must match.
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if cfg.get("env", {}).get("disable_qwen_thinking", True):
        _orig_apply = tokenizer.apply_chat_template

        def _apply_no_thinking(*p_args, **p_kwargs):
            p_kwargs.setdefault("enable_thinking", False)
            return _orig_apply(*p_args, **p_kwargs)

        tokenizer.apply_chat_template = _apply_no_thinking  # type: ignore[assignment]
        logger.info("Patched tokenizer.apply_chat_template to enable_thinking=False")

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
                also_sync=cloud_cfg.get("also_sync") or [],
                also_sync_root=output_dir,
            ))
        except Exception as exc:
            logger.warning("CloudSyncCallback unavailable: %s", exc)

    # Plan §0.3b: mid-train validation eval + best-checkpoint tracking.
    midrun_cfg = cfg.get("midrun_eval") or {}
    if midrun_cfg.get("enabled"):
        try:
            from rl_training.training.midrun_eval import MidrunValidationCallback
            callbacks.append(MidrunValidationCallback(
                output_dir=output_dir,
                validation_tasks_path=midrun_cfg.get(
                    "validation_tasks_path",
                    "rl_training/data/validation_tasks_v2.json",
                ),
                every_steps=int(
                    midrun_cfg.get("every_steps")
                    or cfg["grpo"].get("eval_every_steps", 25),
                ),
                fhir_api_base=cfg["env"]["fhir_api_base"],
                func_file=cfg["env"]["func_file"],
                max_rounds=int(cfg["env"].get("max_rounds", 8)),
                max_new_tokens=int(midrun_cfg.get("max_tokens", 2048)),
                enable_thinking=bool(midrun_cfg.get("enable_thinking", False)),
                abort_on_regression_pp=float(
                    midrun_cfg.get("abort_on_regression_pp", 5.0),
                ),
            ))
            logger.info("MidrunValidationCallback enabled (every_steps=%s)",
                        midrun_cfg.get("every_steps"))
        except Exception as exc:
            logger.warning("MidrunValidationCallback unavailable: %s", exc)

    resume = _resolve_resume_checkpoint(
        args.resume_from_checkpoint, cloud_cfg, output_dir,
    )

    trainer_kwargs: dict[str, Any] = dict(
        model=model_id,
        args=grpo_config,
        peft_config=peft_config,
        processing_class=tokenizer,
        train_dataset=dataset,
        reward_funcs=reward_funcs,
        callbacks=callbacks or None,
    )

    if use_plain_rollout:
        from rl_training.env.medagent_env import MedAgentEnv
        from rl_training.rl.medagent_plain_rollout import (
            _build_task_lookup,
            make_medagent_plain_rollout,
        )

        env_cfg = cfg["env"]
        with open(env_cfg["func_file"]) as fh:
            funcs = json.load(fh)
        max_rounds = int(env_cfg.get("max_rounds", 8))
        fhir_api_base = env_cfg["fhir_api_base"]

        def _env_factory():
            return MedAgentEnv(
                fhir_api_base=fhir_api_base,
                funcs=funcs,
                max_rounds=max_rounds,
            )

        task_lookup = _build_task_lookup(dataset)
        logger.info(
            "Plain-text rollout enabled (action_format=plain_text); built "
            "prompt-hash task lookup for %d tasks",
            len(task_lookup),
        )
        trainer_kwargs["rollout_func"] = make_medagent_plain_rollout(
            tokenizer=tokenizer,
            env_factory=_env_factory,
            max_rounds=max_rounds,
            max_completion_length=int(cfg["grpo"]["max_completion_length"]),
            temperature=float(cfg["grpo"].get("temperature", 0.9)),
            top_p=float(cfg["grpo"].get("top_p", 0.95)),
            task_lookup=task_lookup,
            fhir_api_base=fhir_api_base,
        )
    else:
        from rl_training.env.trl_env import MedAgentBenchEnv
        trainer_kwargs["environment_factory"] = MedAgentBenchEnv

    trainer = GRPOTrainer(**trainer_kwargs)

    # Plan §0.4: when the two-phase materialised dataset is in use, force a
    # sequential sampler so step-N draws prompts [N*B : (N+1)*B] from the
    # pre-rolled list. Without this, HF Trainer's RandomSampler would shuffle
    # Phase A and Phase B together and dilute the curriculum.
    if used_two_phase:
        from torch.utils.data import SequentialSampler

        # transformers >=4.46 calls ``self._get_train_sampler(dataset)`` (HF
        # commit 9d6c0641f); older versions called it with no args. Accept
        # both signatures so the patch works across pinned trainer versions.
        def _seq_sampler(self_t, dataset=None):  # noqa: ARG001
            ds = dataset if dataset is not None else self_t.train_dataset
            return SequentialSampler(ds)

        try:
            trainer._get_train_sampler = _seq_sampler.__get__(  # type: ignore[attr-defined]
                trainer, type(trainer),
            )
            logger.info("Patched trainer._get_train_sampler -> SequentialSampler "
                        "for two-phase curriculum")
        except Exception as exc:
            logger.warning("Failed to patch sequential sampler: %s", exc)

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
