#!/usr/bin/env bash
# One-shot bootstrap for the Qwen3-32B SFT stage on a 1xH100 pod.
#
# Sequence:
#   1. pip install deps (+ optional flash-attn)
#   2. Pull the SFT JSONL from B2 (optional; usually the file is already
#      baked into the repo or uploaded separately)
#   3. resume_from_cloud pulls the latest SFT LoRA checkpoint (no-op on
#      first run)
#   4. Start the SFT trainer in tmux session 'sft_trainer'
#   5. Wait for the trainer to produce the final adapter
#   6. merge_lora.py writes the BF16-merged model to $MERGED_OUTPUT_DIR
#   7. Upload the merged dir (large: ~64 GB) to B2 under
#      "${B2_PREFIX_MERGED}/qwen3_32b_sft_merged/" so the GRPO pod can
#      pull it without re-running SFT
#   8. Start the watchdog in tmux session 'sft_watchdog'
#
# Monitor with:
#   tmux attach -t sft_trainer     # rich progress bar + loss/ETA
#   tmux attach -t sft_watchdog    # heartbeat polling
#   tail -f rl_training/outputs/qwen3_32b_sft/progress.jsonl
#
# Required env vars:
#   HF_TOKEN                               for Qwen3-32B download (recommended)
# Optional (enables cloud sync + resume):
#   B2_APPLICATION_KEY_ID, B2_APPLICATION_KEY
#   B2_BUCKET, B2_PREFIX_SFT, B2_PREFIX_MERGED
#   RUNPOD_POD_ID, RUNPOD_API_KEY          for watchdog self-stop
#   NTFY_TOPIC                             for push notifications
#   MAX_RUNTIME_HOURS                      default 8
#   CONFIG                                 default rl_training/configs/qwen3_32b_sft.yaml
#   BASE_MODEL                             default Qwen/Qwen3-32B-Instruct
#   SFT_OUTPUT_DIR                         default rl_training/outputs/qwen3_32b_sft
#   MERGED_OUTPUT_DIR                      default /workspace/qwen3_32b_sft_merged
#   SKIP_MERGE=1                           stop after SFT (don't merge + upload)
#   SKIP_UPLOAD_MERGED=1                   merge locally but don't upload to B2

set -euo pipefail

CONFIG="${CONFIG:-rl_training/configs/qwen3_32b_sft.yaml}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-32B-Instruct}"
SFT_OUTPUT_DIR="${SFT_OUTPUT_DIR:-rl_training/outputs/qwen3_32b_sft}"
MERGED_OUTPUT_DIR="${MERGED_OUTPUT_DIR:-/workspace/qwen3_32b_sft_merged}"
B2_BUCKET="${B2_BUCKET:-}"
B2_PREFIX_SFT="${B2_PREFIX_SFT:-qwen3_32b_sft}"
B2_PREFIX_MERGED="${B2_PREFIX_MERGED:-qwen3_32b_sft_merged}"
MAX_RUNTIME_HOURS="${MAX_RUNTIME_HOURS:-8}"
SFT_JSONL="${SFT_JSONL:-rl_training/outputs/qwen_pipeline/phase_a/qwen_sft_openai.jsonl}"

log() { echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] launch_sft: $*"; }

log "1/8 installing dependencies..."
pip install --no-cache-dir -r requirements-gpu.txt
pip install --no-cache-dir flash-attn --no-build-isolation || \
  log "flash-attn install failed; continuing with sdpa"

log "2/8 ensuring SFT jsonl is present..."
if [[ ! -f "$SFT_JSONL" ]]; then
  if [[ -n "$B2_BUCKET" ]]; then
    log "  SFT jsonl missing locally; trying B2 download..."
    mkdir -p "$(dirname "$SFT_JSONL")"
    python - <<PY || log "SFT jsonl B2 pull failed"
import os, sys
sys.path.insert(0, os.getcwd())
from rl_training.training.checkpoint_sync import make_backend
backend = make_backend("b2", os.environ["B2_BUCKET"])
# Expect the jsonl uploaded under "${B2_PREFIX_SFT}/data/"
backend.download_directory("${B2_PREFIX_SFT}/data", os.path.dirname("${SFT_JSONL}"))
PY
  else
    log "ERROR: $SFT_JSONL missing and no B2_BUCKET set."
    log "  Run: python rl_training/scripts/generate_qwen_sft_expert_trajectories.py"
    exit 1
  fi
fi

log "3/8 resuming SFT from cloud checkpoint (if any)..."
RESUME_CKPT=""
if [[ -n "$B2_BUCKET" ]]; then
  RESUME_CKPT=$(python rl_training/scripts/resume_from_cloud.py \
      --backend b2 --bucket "$B2_BUCKET" --prefix "$B2_PREFIX_SFT" \
      --output-dir "$SFT_OUTPUT_DIR" 2>/dev/null || echo "")
fi
log "resume checkpoint: ${RESUME_CKPT:-<none>}"

log "4/8 starting SFT trainer (tmux session: sft_trainer)"
SFT_CMD="CUDA_VISIBLE_DEVICES=0 python rl_training/scripts/sft_qwen3_32b.py"
SFT_CMD+=" --config $CONFIG"
SFT_CMD+=" --output-dir $SFT_OUTPUT_DIR"
SFT_CMD+=" --sft-jsonl $SFT_JSONL"
if [[ -n "$RESUME_CKPT" ]]; then
  SFT_CMD+=" --resume-from-checkpoint $RESUME_CKPT"
else
  SFT_CMD+=" --resume-from-checkpoint auto"
fi
tmux kill-session -t sft_trainer 2>/dev/null || true
tmux new-session -d -s sft_trainer "$SFT_CMD 2>&1 | tee -a rl_training/outputs/sft_trainer.log"

log "5/8 starting watchdog (tmux session: sft_watchdog)"
tmux kill-session -t sft_watchdog 2>/dev/null || true
tmux new-session -d -s sft_watchdog \
  "HEARTBEAT_PATH=/tmp/trainer_heartbeat \
   MAX_RUNTIME_HOURS=$MAX_RUNTIME_HOURS \
   B2_BUCKET=$B2_BUCKET B2_PREFIX=$B2_PREFIX_SFT \
   LOG_DIR=$SFT_OUTPUT_DIR \
   bash rl_training/scripts/watchdog.sh 2>&1 | tee -a rl_training/outputs/sft_watchdog.log"

log "6/8 waiting for SFT trainer to finish..."
# The trainer writes the final adapter to $SFT_OUTPUT_DIR on completion.
# We block here until the tmux session ends; the watchdog kills it on stall.
while tmux has-session -t sft_trainer 2>/dev/null; do
  sleep 30
done
log "SFT trainer session ended."

if [[ ! -f "$SFT_OUTPUT_DIR/adapter_config.json" ]]; then
  log "ERROR: $SFT_OUTPUT_DIR/adapter_config.json missing; SFT did not finish cleanly."
  exit 1
fi

if [[ "${SKIP_MERGE:-0}" == "1" ]]; then
  log "7/8 SKIP_MERGE=1; stopping before merge."
  log "8/8 done."
  exit 0
fi

log "7/8 merging LoRA adapter into base -> $MERGED_OUTPUT_DIR"
mkdir -p "$MERGED_OUTPUT_DIR"
python rl_training/scripts/merge_lora.py \
  --base-model "$BASE_MODEL" \
  --adapter "$SFT_OUTPUT_DIR" \
  --output-dir "$MERGED_OUTPUT_DIR"

if [[ "${SKIP_UPLOAD_MERGED:-0}" == "1" ]]; then
  log "8/8 SKIP_UPLOAD_MERGED=1; merged weights stay local at $MERGED_OUTPUT_DIR"
  exit 0
fi

if [[ -n "$B2_BUCKET" ]]; then
  log "8/8 uploading merged weights to ${B2_BUCKET}/${B2_PREFIX_MERGED}..."
  python - <<PY
import os, sys
sys.path.insert(0, os.getcwd())
from rl_training.training.checkpoint_sync import make_backend
backend = make_backend("b2", os.environ["B2_BUCKET"])
backend.upload_directory("${MERGED_OUTPUT_DIR}", "${B2_PREFIX_MERGED}")
print("uploaded:", "${B2_PREFIX_MERGED}")
PY
  log "done. Launch the GRPO pod with STAGE=grpo to continue."
else
  log "8/8 no B2_BUCKET set; merged weights at $MERGED_OUTPUT_DIR (kept locally)"
fi
