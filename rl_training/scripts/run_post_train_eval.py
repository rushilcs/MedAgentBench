#!/usr/bin/env python3
"""Evaluate a LoRA-finetuned model on the 300-task benchmark.

Two serving modes:

  1. **Adapter mode (default):** vLLM server was launched with
     ``--enable-lora`` and the LoRA adapter name is passed via
     ``--lora-model``. The ``VLLMPolicy`` posts ``model=<lora_name>`` so
     vLLM routes to the adapter-applied model.

  2. **Merged mode (``--merge-and-serve``):** this script merges the LoRA
     into the base weights locally and writes the merged model to
     ``--merged-output-dir``. You then launch a vLLM server pointing at
     that directory and pass its path as ``--model``.

Adapter mode is preferred for quick iteration (no weight merge cost). Merged
mode is preferred for the final evaluation run (vLLM throughput is ~5-10%
higher without the LoRA hot-swap).

Trajectories, summary, and clinical metrics are written to
``--output-dir`` in the same layout as ``run_baseline_eval.py`` so the two
runs can be diffed directly.

Known model-side gaps (NOT infra bugs; tracked separately from the FHIR
snapshot dict-vs-str fix that landed in this file):

* **task9** -- the SFT v2 model emits a 3-element FINISH payload like
  ``[4.5, "No replacement potassium needed", "Order placed"]`` but
  ``refsol.task9`` only accepts ``[last_value]`` or ``[]``. This is a prompt
  / training-data shape mismatch -- the SFT corpus did not constrain the
  FINISH list shape for conditional-order tasks. Address by either
  retraining with a stricter task9 schema or post-processing the FINISH
  payload to its first element when the task is task9-shaped.
* **task10** -- when no A1C exists, ``refsol.task10`` requires a
  ``ServiceRequest`` POST in addition to ``FINISH([-1])``. The SFT v2 model
  currently only emits the FINISH. Same root cause -- training data didn't
  teach the "no result -> still order" branch. Address by augmenting the
  SFT corpus with the missing ServiceRequest expert trajectories.

Both issues are expected to leave task9/task10 SR low even after the
dict->str grader fix lands; everything else (tasks 1-8) should recover.
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

from rl_training.agent.vllm_policy import VLLMPolicy
from rl_training.data.trajectory import Trajectory
from rl_training.env.medagent_env import MedAgentEnv
from rl_training.evaluation.clinical_metrics import (
    compute_clinical_metrics,
    save_clinical_metrics,
)
from rl_training.evaluation.evaluator import Evaluator
from rl_training.evaluation.metrics import compute_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _merge_lora(base_model: str, adapter_path: str, output_dir: str) -> str:
    """Merge a LoRA adapter into the base model and save the result.

    Returns the path to the merged model directory.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading base model %s (bfloat16) for merge...", base_model)
    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    logger.info("Attaching adapter from %s...", adapter_path)
    model = PeftModel.from_pretrained(base, adapter_path)
    logger.info("Merging and unloading...")
    merged = model.merge_and_unload()
    logger.info("Writing merged model to %s", output_dir)
    merged.save_pretrained(output_dir, safe_serialization=True)
    tok.save_pretrained(output_dir)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-train eval with vLLM")
    parser.add_argument("--config", default="rl_training/configs/default.yaml")
    parser.add_argument("--base-model", default="Qwen/Qwen3-32B-Instruct",
                        help="Base model (needed for merge mode)")
    parser.add_argument("--lora-adapter", default=None,
                        help="Path to a LoRA adapter directory")
    parser.add_argument("--lora-model-name", default="medagent_clinical",
                        help="Name to advertise for the adapter in vLLM (adapter mode)")
    parser.add_argument("--merge-and-serve", action="store_true",
                        help="Merge LoRA into base and tell vLLM to load the merged dir")
    parser.add_argument("--merged-output-dir", default=None,
                        help="Where to save the merged model (merge mode only)")
    parser.add_argument("--vllm-base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--vllm-model", default=None,
                        help="Model id served by vLLM; defaults to lora-model-name "
                             "(adapter mode) or merged-output-dir (merge mode)")
    parser.add_argument("--data-file", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-parallel", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument(
        "--enable-thinking",
        choices=["true", "false"],
        default="true",
        help="Pass chat_template_kwargs.enable_thinking to vLLM (Qwen3 thinking mode). "
             "When 'false', vLLM prepends a closed <think></think> block so the model "
             "emits the answer directly (fixes refsol's first-line URL parser).",
    )
    parser.add_argument(
        "--fhir-snapshot-jsonl",
        default=None,
        help="Replay FHIR GETs from this JSONL snapshot so eval can run without "
             "Docker/live FHIR (patches send_get_request in utils, refsol, medagent_env).",
    )
    parser.add_argument(
        "--fhir-snapshot-fallthrough",
        action="store_true",
        help="With --fhir-snapshot-jsonl: on cache miss, hit live FHIR (needs reachable server).",
    )
    args = parser.parse_args()

    if args.fhir_snapshot_jsonl:
        from rl_training.env.fhir_snapshot import FhirSnapshot

        snap = FhirSnapshot(
            mode="replay",
            path=args.fhir_snapshot_jsonl,
            fallthrough=args.fhir_snapshot_fallthrough,
        )

        def _patched_send_get(url, params=None, headers=None):  # noqa: ARG001
            # Mirrors the trainer (train_grpo_32b.py): refsol does
            # ``json.loads(send_get_request(...)['data'])`` because the
            # original ``utils.send_get_request`` returns text for HAPI's
            # ``application/fhir+json``. ``FhirSnapshot`` already coerces
            # dict/list -> str in its return path; this wrapper is
            # belt-and-suspenders so a future regression in the snapshot
            # cannot silently zero out grader scores again.
            res = snap.send_get_request(url)
            if "data" in res and not isinstance(res["data"], str):
                res = {**res, "data": json.dumps(res["data"])}
            return res

        import src.server.tasks.medagentbench.utils as _mb_utils
        import src.server.tasks.medagentbench.refsol as _refsol
        import rl_training.env.medagent_env as _med_env

        _mb_utils.send_get_request = _patched_send_get
        _refsol.send_get_request = _patched_send_get
        _med_env.send_get_request = _patched_send_get
        logger.info(
            "FHIR snapshot replay installed (%d rows, fallthrough=%s)",
            len(snap._cache),  # noqa: SLF001
            args.fhir_snapshot_fallthrough,
        )

    if args.merge_and_serve:
        if not args.lora_adapter or not args.merged_output_dir:
            parser.error("--merge-and-serve requires --lora-adapter and --merged-output-dir")
        _merge_lora(args.base_model, args.lora_adapter, args.merged_output_dir)
        served_model = args.vllm_model or args.merged_output_dir
        logger.info("Merged model ready. Launch vLLM with "
                    "--model %s before running eval.", args.merged_output_dir)
    else:
        served_model = args.vllm_model or args.lora_model_name
        if not served_model:
            parser.error("--vllm-model (or --lora-model-name) is required in adapter mode")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    data_file = args.data_file or config["env"]["data_file"]
    with open(data_file) as f:
        benchmark_tasks = json.load(f)
    logger.info("Loaded %d benchmark tasks", len(benchmark_tasks))

    env = MedAgentEnv.from_config(config)
    evaluator = Evaluator(env=env, benchmark_tasks=benchmark_tasks)

    os.environ["VLLM_BASE_URL"] = args.vllm_base_url
    extra_body = None
    if args.enable_thinking == "false":
        extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
        logger.info("enable_thinking=False (Qwen3 thinking disabled at inference)")
    policy = VLLMPolicy(
        model_id=served_model,
        base_url=args.vllm_base_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_parallel=args.max_parallel,
        extra_body=extra_body,
    )

    from rich.progress import (
        BarColumn, MofNCompleteColumn, Progress, SpinnerColumn,
        TextColumn, TimeElapsedColumn, TimeRemainingColumn,
    )

    trajectories: list[Trajectory] = []
    correct_count = 0
    # Live per-task SR is logged every PROGRESS_LOG_EVERY tasks so the user
    # can monitor accuracy from the tee'd log file (the rich progress bar
    # gets clobbered by httpx INFO logs in non-TTY contexts). We also write
    # a lightweight progress.json snapshot so external watchers can poll
    # without parsing the log.
    progress_log_every = max(1, int(os.environ.get("EVAL_PROGRESS_LOG_EVERY", "10")))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "progress.json"
    per_task_running: dict[str, dict[str, int]] = {}

    def _per_task_summary() -> dict[str, float]:
        return {
            tid: (s["correct"] / s["total"]) if s["total"] else 0.0
            for tid, s in sorted(
                per_task_running.items(),
                key=lambda kv: int(kv[0].replace("task", "")),
            )
        }

    def _write_progress(i: int, total: int) -> None:
        try:
            with open(progress_path, "w") as fp:
                json.dump({
                    "completed": i,
                    "total": total,
                    "overall_sr": correct_count / max(1, i),
                    "correct": correct_count,
                    "per_task_sr": _per_task_summary(),
                    "per_task_n": {
                        k: v["total"] for k, v in per_task_running.items()
                    },
                }, fp, indent=2)
        except Exception:
            pass

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("  SR={task.fields[sr]:.1%}"),
        TimeElapsedColumn(),
        TextColumn("ETA"),
        TimeRemainingColumn(),
    ) as progress:
        pt = progress.add_task(
            "Evaluating (post-train)", total=len(benchmark_tasks), sr=0.0,
        )
        for i, task in enumerate(benchmark_tasks):
            tid = str(task.get("id", "")).split("_")[0] or "unknown"
            try:
                traj = evaluator._rollout(policy, task)  # noqa: SLF001
                trajectories.append(traj)
                ok = bool(traj.correct)
                if ok:
                    correct_count += 1
            except Exception as exc:
                logger.warning("Rollout failed for %s: %s", task.get("id"), exc)
                trajectories.append(Trajectory.from_env_history(
                    task=task, history=[], correct=False, status="error",
                    model_id=served_model,
                ))
                ok = False
            slot = per_task_running.setdefault(tid, {"correct": 0, "total": 0})
            slot["total"] += 1
            if ok:
                slot["correct"] += 1

            done = i + 1
            progress.update(pt, advance=1, sr=correct_count / max(1, done))
            if done % progress_log_every == 0 or done == len(benchmark_tasks):
                pt_summary = ", ".join(
                    f"{tid}={sr:.0%}({per_task_running[tid]['correct']}/{per_task_running[tid]['total']})"
                    for tid, sr in _per_task_summary().items()
                )
                logger.info(
                    "progress %d/%d  overall_sr=%.1f%%  per_task=[%s]",
                    done, len(benchmark_tasks),
                    100 * correct_count / done,
                    pt_summary,
                )
                _write_progress(done, len(benchmark_tasks))

    result = compute_metrics(trajectories)
    print("\n" + result.summary())

    with open(out_dir / "eval.json", "w") as f:
        json.dump({
            "model_id": served_model,
            "base_model": args.base_model,
            "lora_adapter": args.lora_adapter,
            "merged": args.merge_and_serve,
            "total": result.total,
            "correct": result.correct,
            "success_rate": result.success_rate,
            "per_task_sr": result.per_task_sr,
            "query_sr": result.query_sr,
            "action_sr": result.action_sr,
            "invalid_action_rate": result.invalid_action_rate,
            "limit_reached_rate": result.limit_reached_rate,
            "avg_steps": result.avg_steps,
        }, f, indent=2)

    clinical = compute_clinical_metrics(trajectories)
    save_clinical_metrics(clinical, str(out_dir / "clinical.json"))
    print("\n" + clinical.summary())

    with open(out_dir / "trajectories.jsonl", "w") as f:
        for traj in trajectories:
            f.write(traj.to_jsonl_line() + "\n")
    logger.info("Saved eval + clinical + trajectories to %s", out_dir)


if __name__ == "__main__":
    main()
