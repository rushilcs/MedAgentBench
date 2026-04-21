#!/usr/bin/env bash
# Local pipeline check: YAML + dataset + benchmark reward registration (no GPU / no TRL train).
# From repo root:
#   bash rl_training/scripts/smoke_grpo_training.sh
#
# On RunPod after git clone + HF/B2 as needed:
#   SMOKE_GRPO=1 STAGE=grpo bash rl_training/scripts/launch_runpod.sh
# Full SFT-v2 GRPO (not smoke):
#   CONFIG=rl_training/configs/qwen3_32b_grpo_post_sft_v2.yaml \
#   TRAINING_TASKS=rl_training/outputs/qwen_pipeline_v3/phase_a/training_tasks_v2.json \
#   OUTPUT_DIR=rl_training/outputs/qwen3_32b_grpo_v2 \
#   B2_PREFIX=qwen3_32b_run3/grpo_v2_ckpts \
#   B2_PREFIX_MERGED=qwen3_32b_run3/qwen3_32b_sft_v2_merged \
#   MERGED_MODEL_DIR=/workspace/qwen3_32b_sft_v2_merged \
#   STAGE=grpo bash rl_training/scripts/launch_runpod.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

echo "==> Installing minimal Python deps for dry-run (datasets, pyyaml)..."
python -m pip install -q "datasets>=3.0.0" "pyyaml>=6.0"

echo "==> GRPO dry-run (smoke config + smoke tasks)..."
python rl_training/scripts/train_grpo_32b.py \
  --config rl_training/configs/grpo_smoke_runpod.yaml \
  --training-tasks rl_training/outputs/qwen_pipeline_v2/smoke/training_tasks.json \
  --dry-run \
  --skip-model-info-check

echo "==> OK: dry-run passed. Launch on RunPod (example):"
cat <<'EOF'

  # In pod, from repo root (ensure FHIR snapshot + merged SFT v2 exist or pull from B2):
  export HF_TOKEN=...                 # if pulling Qwen from Hub
  export SMOKE_GRPO=1                 # 2-step smoke; remove for full v2 GRPO
  export STAGE=grpo
  # Optional B2 (smoke config has cloud_sync disabled):
  # export B2_BUCKET=... B2_APPLICATION_KEY_ID=... B2_APPLICATION_KEY=...

  bash rl_training/scripts/launch_runpod.sh

  tmux attach -t trainer    # watch training
  tail -f rl_training/outputs/grpo_smoke_validation/progress.jsonl

  # Full production GRPO (no smoke) — unset SMOKE_GRPO and set CONFIG / paths:
  # unset SMOKE_GRPO
  # export CONFIG=rl_training/configs/qwen3_32b_grpo_post_sft_v2.yaml
  # export TRAINING_TASKS=rl_training/outputs/qwen_pipeline_v3/phase_a/training_tasks_v2.json
  # export OUTPUT_DIR=rl_training/outputs/qwen3_32b_grpo_v2
  # export B2_PREFIX=qwen3_32b_run3/grpo_v2_ckpts
  # export MERGED_MODEL_DIR=/workspace/qwen3_32b_sft_v2_merged
EOF
