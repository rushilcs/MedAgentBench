#!/usr/bin/env bash
# Run full GRPO (SFT v2 base) setup on a RunPod from your laptop.
#
# Prereqs:
#   - Pod is RUNNING; B2/HF secrets are in the pod template env (or /etc/rp_environment).
#   - Your public key is added in RunPod (SSH gateway uses the matching private key).
#
# Usage (RunPod SSH gateway — recommended):
#   export POD_SSH_TARGET='37bwi04ubak9wm-64411faa@ssh.runpod.io'
#   export SSH_KEY="$HOME/.ssh/id_ed25519"
#   bash rl_training/scripts/grpo_pod_one_shot_ssh.sh
#
# Usage (exposed TCP):
#   export SSH_KEY="$HOME/.ssh/id_ed25519"
#   bash rl_training/scripts/grpo_pod_one_shot_ssh.sh --direct root@64.247.201.32 12781
#
set -euo pipefail

SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
POD_SSH_TARGET="${POD_SSH_TARGET:-37bwi04ubak9wm-64411faa@ssh.runpod.io}"

SSH_BASE=(ssh -i "$SSH_KEY" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=45 -o ServerAliveCountMax=30)

if [[ "${1:-}" == "--direct" ]]; then
  host="${2:?need host e.g. root@64.247.201.32}"
  port="${3:?need port e.g. 12781}"
  SSH_CMD=("${SSH_BASE[@]}" -p "$port" "$host")
else
  SSH_CMD=("${SSH_BASE[@]}" "$POD_SSH_TARGET")
fi

if [[ ! -f "$SSH_KEY" ]]; then
  echo "ERROR: private key not found: $SSH_KEY" >&2
  exit 2
fi

echo "==> Connecting: ${SSH_CMD[*]}" >&2
"${SSH_CMD[@]}" bash -s <<'REMOTE'
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
echo "[remote] $(hostname) $(date -u +%Y-%m-%dT%H:%M:%SZ)"

if ! command -v git >/dev/null 2>&1; then
  apt-get update -qq
  apt-get install -y -qq git tmux curl ca-certificates jq rsync
fi

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
export TRAINING_TASKS="${TRAINING_TASKS:-rl_training/data/training_tasks_v2.json}"
export OUTPUT_DIR="${OUTPUT_DIR:-rl_training/outputs/qwen3_32b_grpo_v2}"
export B2_PREFIX="${B2_PREFIX:-qwen3_32b_run3/grpo_v2_ckpts}"
export B2_PREFIX_MERGED="${B2_PREFIX_MERGED:-qwen3_32b_run3/artifacts/sft_v2_merged}"
export MERGED_MODEL_DIR="${MERGED_MODEL_DIR:-/qwen3_32b_sft_v2_merged}"
export FHIR_SNAPSHOT_LOCAL="${FHIR_SNAPSHOT_LOCAL:-rl_training/outputs/fhir_snapshot.jsonl}"
export GRPO_TORCHRUN_PROCS="${GRPO_TORCHRUN_PROCS:-2}"

echo "[remote] starting remote_bootstrap_grpo.sh (deps + B2 pulls + tmux trainer)..."
bash .agent_run/remote_bootstrap_grpo.sh

echo "[remote] bootstrap finished. Attach: tmux attach -t trainer"
echo "[remote] tail: tail -f ${OUTPUT_DIR}/progress.jsonl"
REMOTE
