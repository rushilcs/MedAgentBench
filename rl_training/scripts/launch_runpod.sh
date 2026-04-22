#!/usr/bin/env bash
# One-shot bootstrap for a RunPod on-demand pod.
#
# Controlled by the STAGE env var:
#   STAGE=sft   : run SFT + merge on a 1xH100 pod. Wraps launch_sft.sh.
#   STAGE=grpo  : run GRPO on a 2xH100 pod against either a locally-merged
#                 SFT base (if $MERGED_MODEL_DIR exists) or the stock
#                 Qwen3-32B base. If $B2_PREFIX_MERGED is set and the dir
#                 is missing locally, we download it from B2 first.
#   STAGE=both  : run SFT + merge on GPU0 (vLLM idle), then GRPO on 2xH100.
#                 Requires a 2xH100 pod; SFT leaves GPU1 unused.
#   STAGE=grpo-baseline : original behaviour - straight GRPO on the
#                 stock Qwen3-32B base, no SFT. Kept as an escape hatch.
#   (default)   : grpo  (post-SFT run; the default use case)
#
# Common env vars:
#   HF_TOKEN                               for Qwen3-32B download
#   RUNPOD_POD_ID, RUNPOD_API_KEY          for watchdog self-stop
#   B2_APPLICATION_KEY_ID, B2_APPLICATION_KEY   for checkpoint + log sync
#   B2_BUCKET                              bucket name
#   MAX_RUNTIME_HOURS                      default 20
#   NTFY_TOPIC                             for push notifications
#
# GRPO-specific:
#   SMOKE_GRPO=1                           quick validation: tiny step count, smoke
#                                          tasks, cloud_sync off (see grpo_smoke_runpod.yaml).
#   CONFIG                                 default rl_training/configs/qwen3_32b_grpo_post_sft.yaml
#   TRAINING_TASKS                         default rl_training/outputs/training_tasks.json
#   OUTPUT_DIR                             default rl_training/outputs/qwen3_32b_grpo_post_sft
#   GRPO_TORCHRUN_PROCS                    when vllm.use_vllm=false: torchrun --nproc_per_node (default 2).
#                                          Set to 1 to force single-GPU python (same as legacy behaviour).
#   HEARTBEAT_STALE_SECONDS                optional override; else from CONFIG resilience (default 10800s).
#   B2_PREFIX                              default qwen3_32b_grpo/post_sft
#   B2_PREFIX_MERGED                       default qwen3_32b_sft_merged
#   MERGED_MODEL_DIR                       default /workspace/qwen3_32b_sft_merged
#
# SFT-specific: see launch_sft.sh header.
#
# Monitor with:
#   tmux attach -t trainer   # GRPO rich progress bar
#   tmux attach -t sft_trainer
#   tmux attach -t vllm
#   tmux attach -t watchdog  # heartbeat polling

set -euo pipefail

STAGE="${STAGE:-grpo}"

log() { echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] launch_runpod[$STAGE]: $*"; }

run_sft() {
  log "delegating to launch_sft.sh"
  bash rl_training/scripts/launch_sft.sh
}

run_grpo() {
  if [[ "${SMOKE_GRPO:-0}" == "1" ]]; then
    : "${CONFIG:=rl_training/configs/grpo_smoke_runpod.yaml}"
    : "${TRAINING_TASKS:=rl_training/outputs/qwen_pipeline_v2/smoke/training_tasks.json}"
    : "${OUTPUT_DIR:=rl_training/outputs/grpo_smoke_validation}"
    : "${B2_PREFIX:=smoke/grpo_validation}"
    log "SMOKE_GRPO=1 -> CONFIG=$CONFIG TRAINING_TASKS=$TRAINING_TASKS OUTPUT_DIR=$OUTPUT_DIR"
  fi
  # Defaults assume we're running GRPO on top of the SFT-merged base.
  CONFIG="${CONFIG:-rl_training/configs/qwen3_32b_grpo_post_sft.yaml}"
  TRAINING_TASKS="${TRAINING_TASKS:-rl_training/outputs/training_tasks.json}"
  FHIR_SNAPSHOT_LOCAL="${FHIR_SNAPSHOT_LOCAL:-rl_training/outputs/fhir_snapshot.jsonl}"
  OUTPUT_DIR="${OUTPUT_DIR:-rl_training/outputs/qwen3_32b_grpo_post_sft}"
  B2_BUCKET="${B2_BUCKET:-}"
  B2_PREFIX="${B2_PREFIX:-qwen3_32b_grpo/post_sft}"
  B2_PREFIX_MERGED="${B2_PREFIX_MERGED:-qwen3_32b_sft_merged}"
  MERGED_MODEL_DIR="${MERGED_MODEL_DIR:-/workspace/qwen3_32b_sft_merged}"
  if [[ -z "${MAX_RUNTIME_HOURS:-}" && -f "$CONFIG" ]]; then
    MAX_RUNTIME_HOURS="$(python - "$CONFIG" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
r = cfg.get("resilience") or {}
print(int(r.get("max_runtime_hours", 20)))
PY
)"
  fi
  MAX_RUNTIME_HOURS="${MAX_RUNTIME_HOURS:-20}"
  export CONFIG TRAINING_TASKS FHIR_SNAPSHOT_LOCAL OUTPUT_DIR B2_BUCKET B2_PREFIX \
    B2_PREFIX_MERGED MERGED_MODEL_DIR MAX_RUNTIME_HOURS

  log "1/9 installing dependencies..."
  pip install --no-cache-dir -r requirements-gpu.txt
  # MedAgentBench server typings (task_generator -> src.typings) need pydantic.
  pip install --no-cache-dir "pydantic>=1.10,<3" jmespath || true
  if [[ "${SKIP_FLASH_ATTN:-0}" == "1" ]]; then
    log "1b/9 SKIP_FLASH_ATTN=1; skipping flash-attn build (sdpa only)"
  else
    pip install --no-cache-dir flash-attn --no-build-isolation || \
      log "flash-attn install failed; continuing with sdpa"
  fi
  if [[ -n "${VLLM_VERSION_PIN:-}" ]]; then
    log "1c/9 pinning vLLM to ${VLLM_VERSION_PIN} (VLLM_VERSION_PIN; TRL server-mode compatibility)"
    pip install --no-cache-dir "vllm==${VLLM_VERSION_PIN}"
  fi
  # TRL GRPOTrainer + environment_factory checks for transformers>=5.2 (not on PyPI yet).
  if [[ "${SKIP_TRANSFORMERS_MAIN:-0}" != "1" ]]; then
    log "1d/9 installing transformers from Hugging Face main (GRPO environment_factory); SKIP_TRANSFORMERS_MAIN=1 to skip"
    pip install --no-cache-dir --upgrade "git+https://github.com/huggingface/transformers.git"
  fi
  # vLLM / image wheels can leave torch==2.4 even though requirements-gpu pins
  # torch>=2.6; TRL GRPOTrainer imports FSDPModule (PyTorch 2.6+).
  if [[ "${SKIP_TORCH_UPGRADE:-0}" != "1" ]]; then
    log "1e/9 aligning PyTorch>=2.6+cu124 for TRL (SKIP_TORCH_UPGRADE=1 to skip)"
    pip install --no-cache-dir --upgrade "torch>=2.6.0" torchvision torchaudio \
      --index-url https://download.pytorch.org/whl/cu124
  fi

  log "2/9 pulling FHIR snapshot..."
  mkdir -p "$(dirname "$FHIR_SNAPSHOT_LOCAL")"
  if [[ -n "$B2_BUCKET" ]]; then
    # Prefer the canonical artifact path uploaded by Phase 2 of the
    # 2026-04-22 fix (single file, rebuilt against real FHIR with widened
    # URL coverage). Fall back to the legacy per-run prefix for older pods.
    SNAPSHOT_REMOTE="${B2_PREFIX_FHIR_SNAPSHOT:-qwen3_32b_run3/artifacts/fhir_snapshot.jsonl}"
    export SNAPSHOT_REMOTE FHIR_SNAPSHOT_LOCAL
    python - <<'PY' || log "FHIR snapshot download failed (continuing with empty cache)"
import os, sys
from pathlib import Path
sys.path.insert(0, os.getcwd())
from b2sdk.v2 import InMemoryAccountInfo, B2Api
info = InMemoryAccountInfo()
api = B2Api(info)
api.authorize_account(
    "production",
    os.environ["B2_APPLICATION_KEY_ID"],
    os.environ["B2_APPLICATION_KEY"],
)
bucket = api.get_bucket_by_name(os.environ["B2_BUCKET"])
remote = os.environ["SNAPSHOT_REMOTE"]
local = os.environ["FHIR_SNAPSHOT_LOCAL"]
Path(local).parent.mkdir(parents=True, exist_ok=True)
dl = bucket.download_file_by_name(remote)
with open(local, "wb") as f:
    dl.save(f)
print("snapshot ok", remote, "->", local, Path(local).stat().st_size, "bytes")
PY
  fi
  if [[ -f "$FHIR_SNAPSHOT_LOCAL" ]]; then
    log "2a/9 snapshot ready: $(wc -l <"$FHIR_SNAPSHOT_LOCAL") rows, $(du -h "$FHIR_SNAPSHOT_LOCAL" | cut -f1)"
  fi

  # Live FHIR for snapshot fallthrough + any code paths still hitting HTTP.
  FHIR_HEALTH_URL="${FHIR_HEALTH_URL:-http://127.0.0.1:8080/fhir/metadata}"
  if curl -sf -m 6 "$FHIR_HEALTH_URL" >/dev/null 2>&1; then
    log "2b/9 FHIR metadata OK ($FHIR_HEALTH_URL)"
  elif [[ "${SKIP_FHIR_DOCKER:-0}" == "1" ]]; then
    log "2b/9 FHIR not reachable; SKIP_FHIR_DOCKER=1 (continuing; snapshot + patched refsol should suffice)"
  elif command -v docker >/dev/null 2>&1; then
    log "2b/9 FHIR not reachable; starting MedAgentBench container on :8080..."
    docker rm -f medagentbench-fhir 2>/dev/null || true
    if ! docker image inspect medagentbench >/dev/null 2>&1 \
        && ! docker image inspect jyxsu6/medagentbench:latest >/dev/null 2>&1; then
      docker pull jyxsu6/medagentbench:latest
    fi
    docker tag jyxsu6/medagentbench:latest medagentbench 2>/dev/null || true
    docker run -d --name medagentbench-fhir -p 8080:8080 medagentbench 2>/dev/null \
      || docker run -d --name medagentbench-fhir -p 8080:8080 jyxsu6/medagentbench:latest
    for _i in $(seq 1 72); do
      if curl -sf -m 6 "$FHIR_HEALTH_URL" >/dev/null 2>&1; then
        log "2b/9 FHIR healthy after ${_i} attempts (5s)"
        break
      fi
      sleep 5
    done
    if ! curl -sf -m 6 "$FHIR_HEALTH_URL" >/dev/null 2>&1; then
      log "WARN: FHIR still not healthy after ~6m; training may rely on JSONL snapshot only"
    fi
  else
    log "2b/9 FHIR not reachable and docker unavailable; relying on snapshot + refsol HTTP patch"
  fi

  # Hard gate: when snapshot_fallthrough is true (snapshot misses → live HTTP)
  # the trainer cannot survive without a reachable FHIR. Read the active
  # config and refuse to start the trainer in that combination.
  # Strict mode (snapshot_fallthrough: false, the new default) bypasses
  # this gate because misses become bounded shaping penalties.
  if [[ -n "${CONFIG:-}" && -f "$CONFIG" && "$STAGE" == "grpo" ]]; then
    SNAPSHOT_FALLTHROUGH=$(python3 - <<PY
import yaml,sys
try:
    cfg = yaml.safe_load(open("${CONFIG}"))
    print("true" if (cfg.get("env") or {}).get("snapshot_fallthrough", True) else "false")
except Exception:
    print("true")
PY
)
    if [[ "$SNAPSHOT_FALLTHROUGH" == "true" ]] \
        && ! curl -sf -m 6 "$FHIR_HEALTH_URL" >/dev/null 2>&1 \
        && [[ "${SKIP_FHIR_DOCKER:-0}" != "1" ]]; then
      log "FATAL: snapshot_fallthrough=true and FHIR is unreachable at $FHIR_HEALTH_URL."
      log "       Misses would trigger live HTTP and silently zero out reward."
      log "       Either (a) ensure docker FHIR is up, or (b) set env.snapshot_fallthrough: false in $CONFIG"
      log "       and re-verify snapshot coverage with rl_training/scripts/smoke_grpo_pipeline.py --snapshot-only."
      exit 11
    fi
  fi

  # Stage-specific: if we're running GRPO post-SFT and the merged dir
  # isn't local, pull it from B2.
  if [[ "$STAGE" != "grpo-baseline" ]]; then
    if [[ ! -d "$MERGED_MODEL_DIR" && -n "$B2_BUCKET" ]]; then
      log "3/9 pulling merged SFT weights from B2 ($B2_PREFIX_MERGED -> $MERGED_MODEL_DIR)..."
      mkdir -p "$MERGED_MODEL_DIR"
      python - <<PY || log "merged weights download failed; GRPO will use Hub fallback base"
import os, sys
sys.path.insert(0, os.getcwd())
from rl_training.training.checkpoint_sync import make_backend
backend = make_backend("b2", os.environ["B2_BUCKET"])
backend.download_directory("${B2_PREFIX_MERGED}", "${MERGED_MODEL_DIR}")
PY
    elif [[ -d "$MERGED_MODEL_DIR" ]]; then
      log "3/9 using existing merged SFT weights at $MERGED_MODEL_DIR"
    else
      log "3/9 no merged weights available; GRPO will use Hub base (qwen3_32b_grpo_post_sft.yaml fallback)"
    fi
  else
    log "3/9 grpo-baseline mode; skipping SFT merge handoff"
  fi

  log "4/9 resuming GRPO from cloud checkpoint (if any)..."
  RESUME_CKPT=""
  if [[ -n "$B2_BUCKET" ]]; then
    RESUME_CKPT=$(python rl_training/scripts/resume_from_cloud.py \
        --backend b2 --bucket "$B2_BUCKET" --prefix "$B2_PREFIX" \
        --output-dir "$OUTPUT_DIR" 2>/dev/null || echo "")
  fi
  log "resume checkpoint: ${RESUME_CKPT:-<none>}"

  # TRL vLLM sidecar: export MAX_MODEL_LEN / VLLM_MODEL_IMPL / etc. from CONFIG when unset.
  _grpo_export_vllm_sidecar_env_from_yaml() {
    [[ -f "$CONFIG" ]] || return 0
    eval "$(python - "$CONFIG" <<'PY'
import os, shlex, sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
v = cfg.get("vllm") or {}
pairs = [
    ("max_model_len", "MAX_MODEL_LEN"),
    ("max_num_seqs", "MAX_NUM_SEQS"),
    ("gpu_memory_utilization", "GPU_MEMORY_UTILIZATION"),
    ("tensor_parallel_size", "TENSOR_PARALLEL_SIZE"),
    ("vllm_model_impl", "VLLM_MODEL_IMPL"),
]
for yk, ek in pairs:
    if ek in os.environ and str(os.environ.get(ek, "")).strip():
        continue
    if yk in v and v[yk] is not None:
        print(f"export {ek}={shlex.quote(str(v[yk]))}")
PY
    )"
  }
  _grpo_export_vllm_sidecar_env_from_yaml

  GRPO_USE_VLLM_SERVER=1
  if [[ -f "$CONFIG" ]]; then
    GRPO_USE_VLLM_SERVER="$(python - "$CONFIG" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
print("0" if not (cfg.get("vllm") or {}).get("use_vllm", True) else "1")
PY
    )"
  fi

  # Watchdog defaults to 900s stale window — too short for GRPO/HF-generate where
  # a single step can be 10–20+ minutes (heartbeat only updates on_step_end).
  if [[ -z "${HEARTBEAT_STALE_SECONDS:-}" ]]; then
    if [[ -f "$CONFIG" ]]; then
      HEARTBEAT_STALE_SECONDS="$(python - "$CONFIG" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
r = cfg.get("resilience") or {}
print(int(r.get("heartbeat_stale_seconds", 10800)))
PY
      )"
    else
      HEARTBEAT_STALE_SECONDS=10800
    fi
  fi
  export HEARTBEAT_STALE_SECONDS
  log "watchdog will use HEARTBEAT_STALE_SECONDS=${HEARTBEAT_STALE_SECONDS}s"

  # Decide what vLLM should serve: the merged dir if present, else $MODEL.
  if [[ -d "$MERGED_MODEL_DIR" && "$STAGE" != "grpo-baseline" ]]; then
    # GRPOTrainer server-mode requires `trl vllm-serve` (weight sync). Plain
    # `vllm serve` makes init_communicator return 404.
    VLLM_SERVE_MODE="trl_merged"
    export SERVE_MODE="$VLLM_SERVE_MODE"
    export MERGED_PATH="$MERGED_MODEL_DIR"
    log "5/9 vLLM (TRL) will serve merged model at $MERGED_MODEL_DIR (SERVE_MODE=$SERVE_MODE)"
  else
    VLLM_SERVE_MODE="base"
    export SERVE_MODE="$VLLM_SERVE_MODE"
    log "5/9 vLLM will serve stock base ($SERVE_MODE mode)"
  fi

  if [[ "$GRPO_USE_VLLM_SERVER" == "1" ]]; then
    log "6/9 starting vLLM server on GPU1 (tmux session: vllm)"
    tmux kill-session -t vllm 2>/dev/null || true
    tmux new-session -d -s vllm "CUDA_VISIBLE_DEVICES=1 SERVE_MODE=$SERVE_MODE MERGED_PATH=${MERGED_PATH:-} bash rl_training/scripts/launch_vllm_server.sh 2>&1 | tee -a rl_training/outputs/vllm.log"

    log "7/9 waiting for vLLM to become ready (up to 15 min)..."
    _vllm_ready() {
      # Plain `vllm serve` exposes OpenAI /v1/models. `trl vllm-serve` may not;
      # it serves TRL weight-sync plus /health (often 307 -> 200).
      curl -sfL -m 3 http://127.0.0.1:8000/v1/models >/dev/null 2>&1 \
        || curl -sfL -m 3 http://127.0.0.1:8000/health >/dev/null 2>&1
    }
    for i in $(seq 1 180); do
      if _vllm_ready; then
        log "vLLM ready after ${i}x5s"
        break
      fi
      sleep 5
    done
    if ! _vllm_ready; then
      log "ERROR: vLLM did not come up within 15 min"
      exit 1
    fi
  else
    log "6/9 skipping vLLM sidecar (vllm.use_vllm=false in $CONFIG)"
    log "7/9 skipping vLLM readiness probe"
    tmux kill-session -t vllm 2>/dev/null || true
  fi

  # Export creds for tmux children (avoid embedding secrets in process argv).
  # Also push into tmux global env so new sessions inherit even if the server
  # was started earlier without these variables.
  for _ek in HF_TOKEN B2_BUCKET B2_APPLICATION_KEY_ID B2_APPLICATION_KEY OPENAI_API_KEY; do
    _ev="${!_ek:-}"
    if [[ -n "${_ev}" ]]; then
      export "${_ek}=${_ev}"
      tmux set-environment -g "$_ek" "$_ev" 2>/dev/null || true
    fi
  done

  log "8/9 starting GRPO trainer (tmux session: trainer)"
  if [[ "$GRPO_USE_VLLM_SERVER" == "1" ]]; then
    export CUDA_VISIBLE_DEVICES=0
    TRAINER_CMD="python rl_training/scripts/train_grpo_32b.py"
    log "trainer: single process on GPU0 (vLLM rollouts on GPU1)"
  else
    _nproc="${GRPO_TORCHRUN_PROCS:-2}"
    if [[ "${_nproc}" -le 1 ]]; then
      export CUDA_VISIBLE_DEVICES=0
      TRAINER_CMD="python rl_training/scripts/train_grpo_32b.py"
      log "trainer: single process on GPU0 (GRPO_TORCHRUN_PROCS=${_nproc})"
    else
      export CUDA_VISIBLE_DEVICES=0,1
      TRAINER_CMD="python -m torch.distributed.run --standalone --nproc_per_node=${_nproc} rl_training/scripts/train_grpo_32b.py"
      log "trainer: DDP via torch.distributed.run --nproc_per_node=${_nproc} on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    fi
  fi
  TRAINER_CMD+=" --config $CONFIG"
  TRAINER_CMD+=" --training-tasks $TRAINING_TASKS"
  TRAINER_CMD+=" --output-dir $OUTPUT_DIR"
  if [[ -d "$MERGED_MODEL_DIR" && "$STAGE" != "grpo-baseline" ]]; then
    TRAINER_CMD+=" --base-model $MERGED_MODEL_DIR --skip-model-info-check"
  fi
  if [[ -n "$RESUME_CKPT" ]]; then
    TRAINER_CMD+=" --resume-from-checkpoint $RESUME_CKPT"
  else
    TRAINER_CMD+=" --resume-from-checkpoint auto"
  fi
  tmux kill-session -t trainer 2>/dev/null || true
  tmux new-session -d -s trainer "$TRAINER_CMD 2>&1 | tee -a rl_training/outputs/trainer.log"

  log "9/9 starting watchdog (tmux session: watchdog)"
  tmux kill-session -t watchdog 2>/dev/null || true
  tmux new-session -d -s watchdog \
    "HEARTBEAT_PATH=/tmp/trainer_heartbeat \
     HEARTBEAT_STALE_SECONDS=$HEARTBEAT_STALE_SECONDS \
     MAX_RUNTIME_HOURS=$MAX_RUNTIME_HOURS \
     LOG_DIR=$OUTPUT_DIR \
     bash rl_training/scripts/watchdog.sh 2>&1 | tee -a rl_training/outputs/watchdog.log"

  log "all sessions live. Safe to disconnect SSH."
  log "  tmux attach -t trainer | vllm | watchdog"
  log "  tail -f $OUTPUT_DIR/progress.jsonl"
}

case "$STAGE" in
  sft)
    run_sft
    ;;
  grpo|grpo-baseline)
    run_grpo
    ;;
  both)
    run_sft
    log "SFT + merge complete; proceeding to GRPO stage in-pod."
    # Merged dir is written by launch_sft.sh to $MERGED_OUTPUT_DIR (default
    # /workspace/qwen3_32b_sft_merged). run_grpo picks it up automatically.
    run_grpo
    ;;
  *)
    echo "Unknown STAGE=$STAGE (expected: sft|grpo|grpo-baseline|both)" >&2
    exit 2
    ;;
esac
