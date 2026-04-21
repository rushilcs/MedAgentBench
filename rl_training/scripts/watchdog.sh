#!/usr/bin/env bash
# Trainer watchdog for RunPod on-demand pods.
#
# Every 60s, checks:
#   1. The heartbeat file's mtime. If stale >= HEARTBEAT_STALE_SECONDS
#      (default 900 = 15 min), assume trainer is dead/hung and stop the pod.
#   2. Wall-clock elapsed. If >= MAX_RUNTIME_HOURS (default 20), stop the pod
#      to cap the bill even if training is still progressing.
#
# When the watchdog decides to stop, it:
#   a. Uploads final logs to the configured B2 bucket (best effort).
#   b. Sends a final ntfy notification if NTFY_TOPIC is set.
#   c. Calls `runpodctl stop pod $RUNPOD_POD_ID`.
#
# Run this as the third tmux session on the pod:
#   tmux new-session -d -s watchdog 'bash rl_training/scripts/watchdog.sh'
#
# Required env vars (or sourced from /etc/runpod-watchdog.env):
#   RUNPOD_POD_ID, RUNPOD_API_KEY
# Optional:
#   HEARTBEAT_PATH            default /tmp/trainer_heartbeat
#   HEARTBEAT_STALE_SECONDS   default 900 (too low for GRPO: one step can exceed
#                               15m; launch_runpod.sh sets this from CONFIG
#                               resilience.heartbeat_stale_seconds, default 10800)
#   MAX_RUNTIME_HOURS         default 20
#   NTFY_TOPIC                default empty (no notifications)
#   B2_BUCKET, B2_PREFIX      default empty (no log upload)
#   LOG_DIR                   default rl_training/outputs/qwen3_32b_grpo

set -u

HEARTBEAT_PATH="${HEARTBEAT_PATH:-/tmp/trainer_heartbeat}"
HEARTBEAT_STALE_SECONDS="${HEARTBEAT_STALE_SECONDS:-900}"
MAX_RUNTIME_HOURS="${MAX_RUNTIME_HOURS:-20}"
LOG_DIR="${LOG_DIR:-rl_training/outputs/qwen3_32b_grpo}"
NTFY_TOPIC="${NTFY_TOPIC:-}"
B2_BUCKET="${B2_BUCKET:-}"
B2_PREFIX="${B2_PREFIX:-}"
RUNPOD_POD_ID="${RUNPOD_POD_ID:-}"
RUNPOD_API_KEY="${RUNPOD_API_KEY:-}"

START_TS=$(date +%s)
MAX_RUNTIME_SECONDS=$((MAX_RUNTIME_HOURS * 3600))

log() {
  echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] watchdog: $*"
}

send_ntfy() {
  local msg="$1"
  if [[ -n "$NTFY_TOPIC" ]]; then
    curl -s -m 5 -d "$msg" "https://ntfy.sh/${NTFY_TOPIC}" >/dev/null || true
  fi
}

upload_logs() {
  if [[ -z "$B2_BUCKET" || -z "$B2_PREFIX" ]]; then
    return
  fi
  if command -v python >/dev/null 2>&1; then
    python - <<'PY' 2>/dev/null || true
import os, sys
sys.path.insert(0, os.getcwd())
from rl_training.training.checkpoint_sync import make_backend
backend = os.environ["B2_BUCKET"]
prefix = os.environ["B2_PREFIX"]
log_dir = os.environ.get("LOG_DIR", "rl_training/outputs/qwen3_32b_grpo")
try:
    b = make_backend("b2", backend)
    b.upload_directory(log_dir, prefix.rstrip("/") + "/final_logs")
    print("final logs uploaded")
except Exception as e:
    print("upload failed:", e)
PY
  fi
}

stop_pod() {
  log "Stopping RunPod pod $RUNPOD_POD_ID ..."
  if command -v runpodctl >/dev/null 2>&1 && [[ -n "$RUNPOD_POD_ID" ]]; then
    runpodctl stop pod "$RUNPOD_POD_ID" || true
  else
    # Fallback: call the RunPod GraphQL API
    if [[ -n "$RUNPOD_API_KEY" && -n "$RUNPOD_POD_ID" ]]; then
      curl -s -X POST "https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY" \
        -H 'Content-Type: application/json' \
        -d "{\"query\":\"mutation{podStop(input:{podId:\\\"$RUNPOD_POD_ID\\\"}){id}}\"}" \
        >/dev/null 2>&1 || true
    fi
  fi
}

exit_watchdog() {
  local reason="$1"
  log "EXIT reason=$reason"
  send_ntfy "medagent trainer watchdog: $reason; stopping pod $RUNPOD_POD_ID"
  upload_logs
  stop_pod
  exit 0
}

log "started; heartbeat=$HEARTBEAT_PATH stale>=${HEARTBEAT_STALE_SECONDS}s max=${MAX_RUNTIME_HOURS}h"
send_ntfy "medagent trainer watchdog started on $RUNPOD_POD_ID"

while true; do
  now=$(date +%s)

  # Wall-clock cap
  elapsed=$((now - START_TS))
  if (( elapsed >= MAX_RUNTIME_SECONDS )); then
    exit_watchdog "wall-clock exceeded ${MAX_RUNTIME_HOURS}h"
  fi

  # Heartbeat staleness
  if [[ -f "$HEARTBEAT_PATH" ]]; then
    hb_mtime=$(stat -c %Y "$HEARTBEAT_PATH" 2>/dev/null || stat -f %m "$HEARTBEAT_PATH" 2>/dev/null || echo "$now")
    age=$((now - hb_mtime))
    if (( age >= HEARTBEAT_STALE_SECONDS )); then
      exit_watchdog "heartbeat stale (${age}s >= ${HEARTBEAT_STALE_SECONDS}s)"
    fi
  else
    # No heartbeat yet - give it a grace period equal to the stale window.
    if (( elapsed >= HEARTBEAT_STALE_SECONDS )); then
      exit_watchdog "no heartbeat file after ${elapsed}s grace"
    fi
  fi

  sleep 60
done
