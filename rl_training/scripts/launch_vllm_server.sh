#!/usr/bin/env bash
# Start a vLLM OpenAI-compatible server for GRPO rollouts.
#
# Intended to be launched in the `vllm` tmux session on a 2xH100 pod:
#   CUDA_VISIBLE_DEVICES=1 bash rl_training/scripts/launch_vllm_server.sh
#
# Serving modes (controlled by $SERVE_MODE):
#   base    (default): plain `vllm serve` for Hub models / baseline eval.
#   lora              : launches `vllm serve` with --enable-lora and
#                      registers the LoRA adapter at $LORA_PATH under the
#                      name $LORA_NAME (for post-train adapter-mode eval).
#   trl_merged        : `trl vllm-serve` on a local merged weights dir (same
#                      artifact as merged mode) but exposes TRL's weight-sync
#                      API required by GRPOTrainer server-mode rollouts.
#   merged            : plain `vllm serve <MERGED_PATH>` — fine for offline
#                      eval; do not use for GRPO (TRL init_communicator 404).
#                      LoRA-merged directory. Two producers:
#                        (a) run_post_train_eval.py --merge-and-serve
#                            (GRPO LoRA merged into SFT or base)
#                        (b) merge_lora.py invoked from launch_sft.sh
#                            (SFT LoRA merged into base, used as the
#                             warm-started base for GRPO rollouts)
#                      Set MERGED_PATH=/workspace/qwen3_32b_sft_merged
#                      for the SFT -> GRPO handoff flow.
#
# Required env vars (with defaults):
#   MODEL                     Qwen/Qwen3-32B-Instruct
#   DTYPE                     bfloat16
#   GPU_MEMORY_UTILIZATION    0.90
#   MAX_MODEL_LEN             8192 (4096 is too small for MedAgentBench + max_tokens=2048)
#   MAX_NUM_SEQS              8
#   TENSOR_PARALLEL_SIZE      1
#   PORT                      8000
# For trl_base / trl_merged (GRPO server mode):
#   VLLM_MODEL_IMPL           vllm | transformers (default: transformers). Use
#                             transformers so weights from QLoRA merge_adapter
#                             match vLLM's load_weights layout (avoids shape assert).
#   TRL_VLLM_TRUST_REMOTE_CODE  1 (default) adds --trust-remote-code for local Hub dirs.
# For lora mode only:
#   LORA_PATH                 (required) absolute path to adapter dir
#   LORA_NAME                 medagent_clinical
# For merged mode only:
#   MERGED_PATH               (required) absolute path to merged model dir

set -euo pipefail

SERVE_MODE="${SERVE_MODE:-base}"
MODEL="${MODEL:-Qwen/Qwen3-32B-Instruct}"
DTYPE="${DTYPE:-bfloat16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
PORT="${PORT:-8000}"
VLLM_MODEL_IMPL="${VLLM_MODEL_IMPL:-transformers}"
TRL_VLLM_TRUST_REMOTE_CODE="${TRL_VLLM_TRUST_REMOTE_CODE:-1}"
# Plain `vllm serve` on local Hub layout dirs (merged SFT, LoRA base) needs this for Qwen3.
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"

_trl_vllm_opts() {
  # Use underscore flags (matches `trl vllm-serve --help` on TRL 1.2.x).
  TRL_VLLM_OPTS=(--vllm_model_impl "$VLLM_MODEL_IMPL")
  if [[ "$TRL_VLLM_TRUST_REMOTE_CODE" == "1" ]]; then
    TRL_VLLM_OPTS+=(--trust_remote_code)
  fi
}

echo "[launch_vllm_server] mode=$SERVE_MODE model=$MODEL port=$PORT tp=$TENSOR_PARALLEL_SIZE"
echo "[launch_vllm_server] dtype=$DTYPE gpu_mem=$GPU_MEMORY_UTILIZATION max_len=$MAX_MODEL_LEN max_seqs=$MAX_NUM_SEQS vllm_model_impl=$VLLM_MODEL_IMPL"

case "$SERVE_MODE" in
  base)
    # Plain vLLM serve for baseline/SFT-merged eval. Used by run_baseline_eval.py
    # and run_post_train_eval.py. For GRPO rollouts during training we use
    # `trl vllm-serve` (mode=trl_base) which exposes a weight-sync endpoint.
    exec vllm serve "$MODEL" \
      --dtype "$DTYPE" \
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
      --max-model-len "$MAX_MODEL_LEN" \
      --max-num-seqs "$MAX_NUM_SEQS" \
      --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
      --port "$PORT"
    ;;
  trl_base)
    # Used during GRPO training on Pod B so the trainer can push updated
    # weights into the vLLM workers via trl's weight-sync endpoint.
    _trl_vllm_opts
    exec trl vllm-serve \
      --model "$MODEL" \
      --dtype "$DTYPE" \
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
      --max-model-len "$MAX_MODEL_LEN" \
      --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
      --port "$PORT" \
      "${TRL_VLLM_OPTS[@]}"
    ;;
  trl_merged)
    : "${MERGED_PATH:?MERGED_PATH is required for trl_merged mode}"
    _trl_vllm_opts
    exec trl vllm-serve \
      --model "$MERGED_PATH" \
      --dtype "$DTYPE" \
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
      --max-model-len "$MAX_MODEL_LEN" \
      --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
      --port "$PORT" \
      "${TRL_VLLM_OPTS[@]}"
    ;;
  lora)
    : "${LORA_PATH:?LORA_PATH is required for lora mode}"
    LORA_NAME="${LORA_NAME:-medagent_clinical}"
    _trc=()
    [[ "$TRUST_REMOTE_CODE" == "1" ]] && _trc+=(--trust-remote-code)
    exec vllm serve "$MODEL" \
      --dtype "$DTYPE" \
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
      --max-model-len "$MAX_MODEL_LEN" \
      --max-num-seqs "$MAX_NUM_SEQS" \
      --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
      --enable-lora \
      --lora-modules "$LORA_NAME=$LORA_PATH" \
      --port "$PORT" \
      "${_trc[@]}"
    ;;
  merged)
    : "${MERGED_PATH:?MERGED_PATH is required for merged mode}"
    _trc=()
    [[ "$TRUST_REMOTE_CODE" == "1" ]] && _trc+=(--trust-remote-code)
    exec vllm serve "$MERGED_PATH" \
      --dtype "$DTYPE" \
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
      --max-model-len "$MAX_MODEL_LEN" \
      --max-num-seqs "$MAX_NUM_SEQS" \
      --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
      --port "$PORT" \
      "${_trc[@]}"
    ;;
  *)
    echo "Unknown SERVE_MODE=$SERVE_MODE (expected: base|trl_base|trl_merged|lora|merged)" >&2
    exit 2
    ;;
esac
