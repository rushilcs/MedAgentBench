#!/usr/bin/env python3
"""Deploy a 2xH100 RunPod and print GRPO (SFT v2) bootstrap steps.

Mirrors prior agent runs: uses ``.agent_run/runpod.py`` GraphQL helpers, loads
secrets from ``MEDAGENT_RUNPOD_ENV_FILE`` (default ``.agent_run/env.json``),
then waits for SSH/Web terminal.

SSH from some networks is blocked; use the RunPod **Web terminal** (Connect)
and paste the printed bootstrap block.

Usage (repo root, ``/usr/bin/python3`` recommended on macOS)::

    export RUNPOD_API_KEY=...
    /usr/bin/python3 rl_training/scripts/deploy_grpo_sftv2_runpod.py

    # Re-print instructions for an existing pod:
    /usr/bin/python3 rl_training/scripts/deploy_grpo_sftv2_runpod.py --pod-id zmqwvrx9ied5hq
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_DEFAULT_ENV = _REPO / ".agent_run" / "env.json"
_RUNPOD_MOD = _REPO / ".agent_run" / "runpod.py"
_SUMMARY = _REPO / "rl_training" / "outputs" / "last_grpo_runpod_launch.json"


def _load_runpod():
    spec = importlib.util.spec_from_file_location("runpod_local", _RUNPOD_MOD)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _pod_env(secrets: dict) -> dict[str, str]:
    skip = {"PUBLIC_KEY", "JUPYTER_PASSWORD"}
    return {k: str(v) for k, v in secrets.items() if v and k not in skip}


def _deploy(rp, env: dict[str, str], name: str) -> dict:
    image = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
    tries = [
        ("NVIDIA H100 80GB HBM3", 2, "SECURE"),
        ("NVIDIA H100 PCIe", 2, "SECURE"),
        ("NVIDIA H100 80GB HBM3", 2, "COMMUNITY"),
    ]
    last_err = None
    for gpu_id, gcount, cloud in tries:
        try:
            return rp.find_and_deploy(
                name=f"{name}-{cloud[:3].lower()}",
                gpu_type_id=gpu_id,
                gpu_count=gcount,
                container_disk_gb=200,
                volume_gb=80,
                image_name=image,
                env=env,
                ports="22/tcp,8000/http",
                min_vcpu=16,
                min_memory_gb=125,
                cloud_type=cloud,
            )
        except RuntimeError as e:
            last_err = e
            continue
    raise last_err or RuntimeError("deploy failed")


def _ssh_line(rp, pod_id: str) -> str | None:
    import subprocess

    out = subprocess.run(
        [sys.executable, str(_RUNPOD_MOD), "ssh-info", "--id", pod_id],
        capture_output=True,
        text=True,
        cwd=str(_REPO),
        env={**os.environ},
    )
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def _bootstrap_block() -> str:
    return r"""# Paste into RunPod Web Terminal (Connect → Terminal), repo root will be created.

set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq git tmux curl ca-certificates jq rsync

mkdir -p /workspace
cd /workspace
if [[ ! -d MedAgentBench/.git ]]; then
  git clone --depth 1 https://github.com/rushilcs/MedAgentBench.git MedAgentBench
fi
cd /workspace/MedAgentBench
git pull --ff-only || true

export PYTHONPATH="/workspace/MedAgentBench:${PYTHONPATH:-}"
export STAGE=grpo
export SKIP_FLASH_ATTN=1
export CONFIG="${CONFIG:-rl_training/configs/qwen3_32b_grpo_post_sft_v2.yaml}"
export TRAINING_TASKS="${TRAINING_TASKS:-rl_training/outputs/qwen_pipeline_v3/phase_a/training_tasks_v2.json}"
export OUTPUT_DIR="${OUTPUT_DIR:-rl_training/outputs/qwen3_32b_grpo_v2}"
export B2_PREFIX="${B2_PREFIX:-qwen3_32b_run3/grpo_v2_ckpts}"
export B2_PREFIX_MERGED="${B2_PREFIX_MERGED:-qwen3_32b_run3/artifacts/sft_v2_merged}"
export MERGED_MODEL_DIR="${MERGED_MODEL_DIR:-/qwen3_32b_sft_v2_merged}"
export FHIR_SNAPSHOT_LOCAL="${FHIR_SNAPSHOT_LOCAL:-rl_training/outputs/fhir_snapshot.jsonl}"
export GRPO_TORCHRUN_PROCS="${GRPO_TORCHRUN_PROCS:-2}"

# Env vars (B2_*, HF_TOKEN, RUNPOD_*) should already exist on the pod from RunPod UI.
# If not, re-add them in the RunPod console, stop pod, start pod, or export here.

bash .agent_run/remote_bootstrap_grpo.sh
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pod-id",
        help="Skip deploy; only wait (if needed) and print bootstrap for this pod",
    )
    ap.add_argument(
        "--env-file",
        default=os.environ.get("MEDAGENT_RUNPOD_ENV_FILE", str(_DEFAULT_ENV)),
        help="JSON file with HF_TOKEN, B2_*, RUNPOD_API_KEY, ...",
    )
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()

    rp = _load_runpod()
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        print("ERROR: set RUNPOD_API_KEY", file=sys.stderr)
        sys.exit(2)

    env_path = Path(args.env_file)
    if not env_path.is_file():
        print(f"ERROR: env file missing: {env_path}", file=sys.stderr)
        sys.exit(2)
    pod_env = _pod_env(json.loads(env_path.read_text()))

    if args.pod_id:
        pod_id = args.pod_id.strip()
        print(f"Using existing pod: {pod_id}", flush=True)
    else:
        name = "medagent-grpo-resume-" + time.strftime("%m%d-%H%M")
        print(f"Deploying pod ({name})...", flush=True)
        out = _deploy(rp, pod_env, name)
        pod_id = out["id"]
        print("Deployed:", json.dumps(out, indent=2), flush=True)

    print("Waiting for RUNNING + ports...", flush=True)
    rp.wait_until_running(pod_id, timeout_s=args.timeout)
    ssh = _ssh_line(rp, pod_id)
    console = f"https://www.runpod.io/console/pods/{pod_id}"

    summary = {
        "pod_id": pod_id,
        "console_url": console,
        "ssh_command": ssh,
        "config": "rl_training/configs/qwen3_32b_grpo_post_sft_v2.yaml",
        "note": (
            "If plain SSH fails from your laptop, open the Web Terminal from the "
            "console URL and run the bootstrap block."
        ),
    }
    _SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    _SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print("\n--- Web terminal bootstrap (also in repo .agent_run/remote_bootstrap_grpo.sh) ---\n")
    print(_bootstrap_block())
    print(f"\nWrote: {_SUMMARY.relative_to(_REPO)}")


if __name__ == "__main__":
    main()
