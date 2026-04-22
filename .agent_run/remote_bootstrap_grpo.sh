#!/usr/bin/env bash
# Run once on fresh RunPod: get repo, optional merged model from B2, launch GRPO (SFT v2).
set -euo pipefail
exec > >(tee -a /tmp/remote_bootstrap_grpo.log) 2>&1
log() { echo "[$(date -u +%H:%M:%SZ)] $*"; }

set -a
source /etc/rp_environment 2>/dev/null || true
while IFS= read -r -d '' line || [[ -n "${line:-}" ]]; do
  case "$line" in B2_*|HF_TOKEN=*|RUNPOD_*|OPENAI_API_KEY=*) export "$line" ;; esac
done < /proc/1/environ
set +a

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq >/dev/null && apt-get install -y -qq git tmux curl ca-certificates jq rsync >/dev/null || true

# Install docker.io (privileged-container path) so launch_runpod.sh can
# bring up the MedAgentBench FHIR jar — the 2026-04-21 wasted run hit
# "Connection refused" on every model GET because the runpod/pytorch
# image ships without docker. We swallow failures: if docker can't be
# installed/started in this container, the snapshot+strict mode keeps
# training reward-correct (smoke gate enforces snapshot coverage).
if ! command -v docker >/dev/null 2>&1; then
  apt-get install -y -qq docker.io >/dev/null 2>&1 || true
fi
if command -v docker >/dev/null 2>&1; then
  service docker start >/dev/null 2>&1 || dockerd >/tmp/dockerd.log 2>&1 &
  sleep 3
fi

REPO="${REPO:-/workspace/MedAgentBench}"
mkdir -p /workspace
cd /workspace

if [[ ! -d "$REPO/.git" ]]; then
  log "Cloning MedAgentBench..."
  if git clone --depth 1 https://github.com/rushilcs/MedAgentBench.git MedAgentBench 2>/dev/null; then
    log "git clone ok"
  else
    log "git clone failed; trying B2 code snapshot..."
    pip install -q b2sdk
    python3 - <<'PY'
import os, sys
from pathlib import Path
from b2sdk.v2 import InMemoryAccountInfo, B2Api
info = InMemoryAccountInfo()
api = B2Api(info)
api.authorize_account(
    "production",
    os.environ["B2_APPLICATION_KEY_ID"],
    os.environ["B2_APPLICATION_KEY"],
)
bucket = api.get_bucket_by_name(os.environ["B2_BUCKET"])
prefix = (os.environ.get("B2_CODE_PREFIX") or "qwen3_32b_run3").rstrip("/") + "/bootstrap/code_snapshot.tar.gz"
df = bucket.download_file_by_name(prefix)
Path("/tmp/code.tar.gz").write_bytes(df.get_bytes())
print("downloaded", prefix, Path("/tmp/code.tar.gz").stat().st_size)
PY
    mkdir -p "$REPO"
    tar xzf /tmp/code.tar.gz -C "$REPO" --strip-components=0 2>/dev/null || tar xzf /tmp/code.tar.gz -C "$REPO"
    rm -f /tmp/code.tar.gz
  fi
fi

cd "$REPO"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"

if [[ -f /tmp/launch_runpod_overlay.sh ]]; then
  cp /tmp/launch_runpod_overlay.sh rl_training/scripts/launch_runpod.sh
  log "Applied launch_runpod_overlay.sh"
fi

MERGED="${MERGED_MODEL_DIR:-/qwen3_32b_sft_v2_merged}"
REMOTE_MERGED="${B2_PREFIX_MERGED:-qwen3_32b_run3/artifacts/sft_v2_merged}"
export MERGED REMOTE_MERGED
# ~55GB+ expected for merged 32B; avoid re-downloading partial dirs from failed runs.
if [[ ! -d "$MERGED" || $(du -sb "$MERGED" 2>/dev/null | awk '{print $1}') -lt 45000000000 ]]; then
  log "Pulling merged SFT v2 from B2: $REMOTE_MERGED -> $MERGED"
  mkdir -p "$MERGED"
  pip install -q b2sdk
  python3 - <<'PY'
import os
from pathlib import Path
from b2sdk.v2 import InMemoryAccountInfo, B2Api

remote = os.environ["REMOTE_MERGED"].rstrip("/")
prefix = remote + "/"
local = os.environ["MERGED"]
info = InMemoryAccountInfo()
api = B2Api(info)
api.authorize_account(
    "production",
    os.environ["B2_APPLICATION_KEY_ID"],
    os.environ["B2_APPLICATION_KEY"],
)
bucket = api.get_bucket_by_name(os.environ["B2_BUCKET"])
Path(local).mkdir(parents=True, exist_ok=True)
n = 0
for file_info, _ in bucket.ls(prefix, recursive=True, latest_only=True):
    rest = file_info.file_name[len(prefix) :]
    out = Path(local) / rest
    out.parent.mkdir(parents=True, exist_ok=True)
    dl = bucket.download_file_by_name(file_info.file_name)
    with open(out, "wb") as f:
        dl.save(f)
    n += 1
print("merged download done", n, "files ->", local)
PY
fi

export STAGE=grpo
export SKIP_FLASH_ATTN=1
export SKIP_TRANSFORMERS_MAIN="${SKIP_TRANSFORMERS_MAIN:-}"
export CONFIG="${CONFIG:-rl_training/configs/qwen3_32b_grpo_post_sft_v2.yaml}"
# Default must exist after ``git clone`` (tracked in repo). Override with pipeline JSON if present on pod/B2.
export TRAINING_TASKS="${TRAINING_TASKS:-rl_training/data/training_tasks_v2.json}"
export OUTPUT_DIR="${OUTPUT_DIR:-rl_training/outputs/qwen3_32b_grpo_v2}"
export B2_PREFIX="${B2_PREFIX:-qwen3_32b_run3/grpo_v2_ckpts}"
export B2_PREFIX_MERGED="${B2_PREFIX_MERGED:-qwen3_32b_run3/artifacts/sft_v2_merged}"
export MERGED_MODEL_DIR="$MERGED"
export FHIR_SNAPSHOT_LOCAL="${FHIR_SNAPSHOT_LOCAL:-rl_training/outputs/fhir_snapshot.jsonl}"
# Leave MAX_RUNTIME_HOURS unset unless caller set it — launch_runpod.sh reads
# resilience.max_runtime_hours from CONFIG (e.g. 48h for long GRPO).
[[ -n "${MAX_RUNTIME_HOURS:-}" ]] && export MAX_RUNTIME_HOURS
export GRPO_TORCHRUN_PROCS="${GRPO_TORCHRUN_PROCS:-2}"

log "Launching GRPO via launch_runpod.sh..."
bash rl_training/scripts/launch_runpod.sh
log "launch_runpod exit=$?"
