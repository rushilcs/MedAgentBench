# RL stack runbook — Qwen3-32B GRPO on MedAgentBench

One-page operational guide for running the full baseline → RL → post-train eval
pipeline on a RunPod on-demand 2xH100 pod.

Scientific thesis, rewards, and research design live in the plan. This file
is purely mechanical.

---

## 0. What you're about to spend

| Step | Cost |
|------|------|
| Local smoke (`--mode unit`, `--mode small`) | $0 |
| Regenerate gpt-4o expert trajectories (OpenAI API) | ~$20 |
| Baseline Qwen3-32B eval (1xH100, ~3-4 h) | ~$8 |
| Qwen QLoRA SFT (1xH100, ~4-5 h) | ~$10 |
| LoRA merge + B2 upload (1xH100, ~15 min) | ~$0.50 |
| SFT-only benchmark eval (1xH100, ~3-4 h) | ~$8 |
| Cloud smoke (`--mode live`, 2xH100) | ~$1 |
| GRPO on SFT-warm base (2xH100, ~10-14 h) | ~$50 |
| Post-GRPO benchmark eval (1xH100, ~3-4 h) | ~$8 |
| Stress eval (baseline + SFT + SFT+GRPO × 4 axes) | ~$16 |
| B2 storage + egress | ~$2 |
| Buffer (20%) | ~$20 |
| **Total** | **~$145** (budget: $200) |

Two-pod cost-saving variant: the SFT stage doesn't need two H100s (no
rollouts). Run SFT + merge on a **1xH100** pod, push the merged weights
to B2, tear it down, then spin up a **2xH100** pod for GRPO and have it
pull the merged weights from B2 on boot. Saves ~$5-8 vs keeping two
H100s live during the SFT phase. See §4.5 and §6 for the `STAGE=sft` /
`STAGE=grpo` workflow.

Silent-billing protection: if the trainer dies, the watchdog stops the pod
within 15 min (wasted cost: ~$1).

---

## 1. Before you spend a cent (local gates)

```bash
# From the repo root on your laptop:
python rl_training/scripts/smoke_test_local.py --mode unit
# <30 s, no network, no GPU. Must pass.
# This now includes the GRPO snapshot+reward end-to-end smoke
# (smoke_grpo_pipeline.py) which was added after the 2026-04-21
# wasted run where snapshot misses + simulated POSTs produced 0%
# success and the dashboards never noticed.

# Stronger gate (recommended before any GRPO deploy): run the same
# harness against the local FHIR docker so we catch URL drift between
# the snapshot recorder and refsol's expected URLs:
#   docker run -d --name medagentbench-fhir -p 8080:8080 medagentbench
#   curl -sf http://localhost:8080/fhir/metadata >/dev/null && echo OK
#   python rl_training/scripts/smoke_grpo_pipeline.py --live

# Optional but recommended:
python rl_training/scripts/smoke_test_local.py --mode small \
    --small-model Qwen/Qwen2.5-0.5B-Instruct
# ~5-10 min on MPS/CPU. Validates the full TRL GRPO loop.
```

If `--mode unit` fails (including the new `smoke_grpo_pipeline` step),
do **not** rent a GPU. The most common cause is the snapshot at
`rl_training/outputs/fhir_snapshot.jsonl` missing URLs that
`refsol.taskN` issues — rebuild it (see §5 / `build_fhir_snapshot.py`)
against a local docker FHIR before deploying.

---

## 2. One-time setup: storage + secrets

1. Create a Backblaze B2 bucket named `medagentbench-checkpoints` (or your
   own name — update `output.cloud_sync.bucket` in the clinical YAML).
2. Create a B2 application key scoped to that bucket. Copy the key id and
   app key.
3. (Optional) Pick an ntfy.sh topic name for push notifications.
4. Generate a Hugging Face read token for Qwen3-32B.

---

## 3. Provision the RunPod

Spin up a **2xH100 PCIe 80GB on-demand** pod with:
- Image: any PyTorch 2.4+ CUDA 12.x image (the RunPod "PyTorch 2.4" template is fine).
- Exposed ports: none needed (we tunnel via SSH).
- Environment variables (set as pod secrets):

```
HF_TOKEN=hf_...
B2_APPLICATION_KEY_ID=...
B2_APPLICATION_KEY=...
B2_BUCKET=medagentbench-checkpoints
B2_PREFIX=qwen3_32b_grpo/clinical
RUNPOD_POD_ID=...          # self-reported by RunPod; check your pod page
RUNPOD_API_KEY=...
NTFY_TOPIC=medagent-run1   # optional
```

SSH in once the pod is running.

---

## 4. Bootstrap and build the FHIR snapshot (one-time per pod)

```bash
git clone <your fork of MedAgentBench>
cd MedAgentBench
# Start the HAPI FHIR server (see repo README for the exact docker-compose)
# Once the server is up and healthy at http://localhost:8080/fhir/:

# Generate training tasks (if you don't already have them)
python rl_training/scripts/run_pipeline.py --only-training-tasks \
    --output rl_training/outputs/training_tasks.json

# Build the FHIR snapshot once; all rollouts hit this instead of the live server.
python rl_training/scripts/build_fhir_snapshot.py \
    --tasks rl_training/outputs/training_tasks.json \
    --fhir-base http://localhost:8080/fhir/ \
    --output rl_training/outputs/fhir_snapshot.jsonl

# Upload the snapshot to B2 so relaunches (e.g. after preemption) skip this step
python -c "
import os, sys; sys.path.insert(0, '.')
from rl_training.training.checkpoint_sync import make_backend
b = make_backend('b2', os.environ['B2_BUCKET'])
b.upload_directory('rl_training/outputs', os.environ['B2_PREFIX'] + '/fhir_snapshot')
"
```

---

## 4.5 SFT stage (Qwen SFT → merge → GRPO handoff)

The Qwen RL run is warm-started with a QLoRA SFT on gpt-4o expert
trajectories, mirroring what we did for `gpt-4o-mini`. The SFT LoRA is
then merged into the Qwen base so GRPO trains a **fresh** adapter on the
SFT'd weights (no adapter stacking).

**Step 1 — Regenerate expert trajectories (local, ~$20 OpenAI):**

```bash
export OPENAI_API_KEY=sk-...
python rl_training/scripts/generate_qwen_sft_expert_trajectories.py \
    --config rl_training/configs/default.yaml \
    --output-dir rl_training/outputs/qwen_pipeline/phase_a \
    --expert-model gpt-4o \
    --trajectories-per-task 3
```

Produces `expert_trajectories.jsonl` (full records for audit) and
`qwen_sft_openai.jsonl` (OpenAI-format chat messages, what SFTTrainer
eats). Idempotent: re-run to pick up where a partial run left off.

**Step 2 — Run SFT + merge (1xH100 pod, ~$18):**

On a fresh 1xH100 pod with the same env vars from §2 (plus
`B2_PREFIX_SFT=qwen3_32b_sft` and
`B2_PREFIX_MERGED=qwen3_32b_sft_merged`):

```bash
STAGE=sft bash rl_training/scripts/launch_runpod.sh
# (equivalent to:  bash rl_training/scripts/launch_sft.sh)
```

The launcher runs SFT in a `sft_trainer` tmux session, then runs
`merge_lora.py` to write `/workspace/qwen3_32b_sft_merged/`, then uploads
the merged dir to `${B2_BUCKET}/${B2_PREFIX_MERGED}/`.

Monitor:

```bash
tmux attach -t sft_trainer    # rich progress bar + loss/ETA
tmux attach -t sft_watchdog   # heartbeat polling
tail -f rl_training/outputs/qwen3_32b_sft/progress.jsonl
```

**Step 3 — Optional SFT-only eval (1xH100 pod, ~$8):**

Confirms the SFT actually moved the model before committing to the $50
GRPO run:

```bash
# Still on the 1xH100 pod, after merge completes:
SERVE_MODE=merged MERGED_PATH=/workspace/qwen3_32b_sft_merged \
    CUDA_VISIBLE_DEVICES=0 \
    tmux new-session -d -s vllm "bash rl_training/scripts/launch_vllm_server.sh"
# Wait for http://127.0.0.1:8000/v1/models, then:
python rl_training/scripts/run_baseline_eval.py \
    --model /workspace/qwen3_32b_sft_merged \
    --vllm-base-url http://127.0.0.1:8000/v1 \
    --output-dir rl_training/outputs/qwen3_32b_sft_eval \
    --max-parallel 8
```

Tear down the SFT pod when done; the merged weights are already in B2.

**Step 4 — GRPO on the SFT-warm base (2xH100 pod, ~$58):**

```bash
# On a fresh 2xH100 pod:
STAGE=grpo \
    CONFIG=rl_training/configs/qwen3_32b_grpo_post_sft.yaml \
    B2_PREFIX_MERGED=qwen3_32b_sft_merged \
    bash rl_training/scripts/launch_runpod.sh
```

`launch_runpod.sh` pulls `qwen3_32b_sft_merged/` from B2 into
`/workspace/qwen3_32b_sft_merged/`, boots vLLM pointed at the merged dir
(`SERVE_MODE=merged`), and starts `train_grpo_32b.py` with
`--base-model /workspace/qwen3_32b_sft_merged --skip-model-info-check`.
The rest of the GRPO flow (progress bar, checkpoint sync, watchdog) is
identical to §6.

**Single-pod alternative:** `STAGE=both` runs SFT → merge → GRPO in one
2xH100 pod. Simplest but wastes ~$5-8 because GPU1 is idle during SFT.

---

## 5. Cloud preflight (~$0.50-$1, last gate before the $50 run)

Before kicking off the full trainer, verify the vLLM + FHIR + rollout
wiring works on this specific pod:

```bash
# Start the vLLM server in its own tmux first. Serve the SFT-merged
# weights (post-§4.5) if they're local; otherwise the stock base:
SERVE=base
MERGE=/workspace/qwen3_32b_sft_merged
[[ -d "$MERGE" ]] && SERVE=merged && MERGED_PATH=$MERGE
tmux new-session -d -s vllm \
    "SERVE_MODE=$SERVE MERGED_PATH=${MERGED_PATH:-} \
     CUDA_VISIBLE_DEVICES=1 bash rl_training/scripts/launch_vllm_server.sh"
# Wait for it to finish loading (watch `tmux attach -t vllm`).

python rl_training/scripts/smoke_test_local.py --mode live \
    --vllm-base-url http://127.0.0.1:8000/v1 \
    --model "${MERGED_PATH:-Qwen/Qwen3-32B-Instruct}" \
    --config rl_training/configs/qwen3_32b_grpo_post_sft.yaml \
    --output-dir rl_training/outputs/live_smoke
```

If this fails, do **not** run `launch_runpod.sh`. Fix the issue (usually
a wrong model id, a down FHIR server, or a missing snapshot) first.

## 6. Launch

From inside the pod. Pick the stage appropriate for the pod you're on:

```bash
# 1xH100 pod: SFT + merge + upload (stage 2 of §4.5)
STAGE=sft bash rl_training/scripts/launch_runpod.sh

# 2xH100 pod: GRPO on the SFT-warm base (stage 4 of §4.5, the typical case)
STAGE=grpo bash rl_training/scripts/launch_runpod.sh

# 2xH100 pod: SFT + merge + GRPO in one pod (simplest, wastes ~$5-8)
STAGE=both bash rl_training/scripts/launch_runpod.sh

# 2xH100 pod: plain GRPO on stock Qwen (escape hatch, skips SFT)
STAGE=grpo-baseline \
    CONFIG=rl_training/configs/qwen3_32b_grpo_clinical.yaml \
    bash rl_training/scripts/launch_runpod.sh
```

Three (or four) tmux sessions come up depending on stage:

| Session | What runs | GPU |
|---------|-----------|-----|
| `sft_trainer`  | `sft_qwen3_32b.py` (stages `sft`, `both`) | `cuda:0` |
| `sft_watchdog` | heartbeat watchdog during SFT           | n/a     |
| `vllm`         | `trl vllm-serve` or `vllm serve <merged>` | `cuda:1` |
| `trainer`      | `train_grpo_32b.py` with QLoRA + clinical rewards | `cuda:0` |
| `watchdog`     | `watchdog.sh` (15-min stale-heartbeat + 20-hour cap) | n/a |

SSH can disconnect at any time — all sessions survive.

---

## 7. Monitor

```bash
# On the pod:
tmux attach -t trainer    # live rich progress bar: step/ETA/reward/SR
tmux attach -t vllm       # vLLM request logs
tmux attach -t watchdog   # watchdog status lines

# From your laptop (rsync the progress log down every minute):
ssh pod 'tail -f MedAgentBench/rl_training/outputs/qwen3_32b_grpo_clinical/progress.jsonl'
```

Every save step (default: every 10 steps) pushes the LoRA adapter + training
state to B2. You can inspect:

```bash
# On the pod:
ls rl_training/outputs/qwen3_32b_grpo_clinical/checkpoint-*/
# Or from laptop with the B2 CLI:
b2 ls medagentbench-checkpoints qwen3_32b_grpo/clinical/
```

---

## 8. Baseline eval (runs before or after fine-tune)

Baseline uses the same vLLM server (no LoRA). In a fresh pod (or reusing the
same pod before training starts), run:

```bash
# Leave vLLM running in its tmux session, then:
python rl_training/scripts/run_baseline_eval.py \
    --model Qwen/Qwen3-32B-Instruct \
    --vllm-base-url http://127.0.0.1:8000/v1 \
    --output-dir rl_training/outputs/qwen3_32b_baseline \
    --max-parallel 8
```

Outputs: `eval.json`, `clinical.json`, `trajectories.jsonl`.

---

## 9. Post-train eval (LoRA mode)

Restart the vLLM server in LoRA mode:

```bash
tmux kill-session -t vllm
tmux new-session -d -s vllm \
  "SERVE_MODE=lora \
   LORA_PATH=$(pwd)/rl_training/outputs/qwen3_32b_grpo_clinical/checkpoint-200 \
   LORA_NAME=medagent_clinical \
   CUDA_VISIBLE_DEVICES=1 bash rl_training/scripts/launch_vllm_server.sh"
```

Wait for `curl http://127.0.0.1:8000/v1/models` to succeed, then:

```bash
python rl_training/scripts/run_post_train_eval.py \
    --base-model Qwen/Qwen3-32B-Instruct \
    --lora-adapter rl_training/outputs/qwen3_32b_grpo_clinical/checkpoint-200 \
    --lora-model-name medagent_clinical \
    --vllm-model medagent_clinical \
    --output-dir rl_training/outputs/qwen3_32b_post_train
```

For the "final" eval with slightly higher throughput, use merge mode:

```bash
python rl_training/scripts/run_post_train_eval.py \
    --merge-and-serve \
    --base-model Qwen/Qwen3-32B-Instruct \
    --lora-adapter rl_training/outputs/qwen3_32b_grpo_clinical/checkpoint-200 \
    --merged-output-dir /workspace/qwen3_32b_merged \
    --output-dir /tmp/discard
# Then restart vLLM pointing at the merged dir:
SERVE_MODE=merged MERGED_PATH=/workspace/qwen3_32b_merged \
    CUDA_VISIBLE_DEVICES=1 bash rl_training/scripts/launch_vllm_server.sh
# And re-run run_post_train_eval.py without --merge-and-serve.
```

---

## 10. Stress eval

```bash
python - <<'PY'
import sys; sys.path.insert(0, ".")
from rl_training.agent.vllm_policy import VLLMPolicy
from rl_training.env.medagent_env import MedAgentEnv
from rl_training.evaluation.evaluator import Evaluator
from rl_training.evaluation.stress_eval import run_stress_eval
import json, yaml

with open("rl_training/configs/default.yaml") as f:
    cfg = yaml.safe_load(f)
with open(cfg["env"]["data_file"]) as f:
    tasks = json.load(f)

env = MedAgentEnv.from_config(cfg)
evaluator = Evaluator(env=env, benchmark_tasks=tasks)
policy = VLLMPolicy(model_id="medagent_clinical",
                    base_url="http://127.0.0.1:8000/v1",
                    temperature=0.0, max_tokens=2048)
run_stress_eval(
    policy, evaluator, tasks,
    perturbations=["timestamp_shuffle", "active_history_swap",
                   "contradictory_note", "distractor_padding"],
    output_dir="rl_training/outputs/qwen3_32b_stress",
)
PY
```

CSV + per-axis JSON drops to `rl_training/outputs/qwen3_32b_stress/`.

---

## 11. Resume after a failure

```bash
# On a fresh pod, after launch_runpod.sh ran once:
# Just rerun the same command. resume_from_cloud.py pulls the latest
# LoRA from B2 and the trainer picks up from that step.
bash rl_training/scripts/launch_runpod.sh
```

---

## 12. Tear down

```bash
# On the pod:
tmux kill-session -a
# Then stop the pod from the RunPod web UI, or let the watchdog do it.
```

All artefacts you care about (LoRA adapters, eval results, trajectories)
are already in B2. The pod's local disk is disposable.

---

## 13. Final comparison table (5-way)

Once all five eval runs are on disk, fill this table in by reading the
`eval.json` and `clinical.json` produced by each `run_baseline_eval.py` /
`run_post_train_eval.py` invocation.

| Model | SR | Query SR | Action SR | Temporal inconsistency ↓ | Over-deferral ↓ | Under-deferral ↓ | Source |
|-------|----|----------|-----------|--------------------------|-----------------|------------------|--------|
| gpt-4o-mini (OOTB)           | ? | ? | ? | — | — | — | `rl_training/outputs/gpt4o_mini_user_run/01_baseline_gpt4o_mini_ootb.json` |
| gpt-4o-mini (SFT)            | ? | ? | ? | — | — | — | `rl_training/outputs/gpt4o_mini_user_run/02_finetuned_gpt4o_mini_benchmark.json` |
| Qwen3-32B (OOTB)             | ? | ? | ? | ? | ? | ? | `rl_training/outputs/qwen3_32b_baseline/{eval,clinical}.json` |
| Qwen3-32B (SFT)              | ? | ? | ? | ? | ? | ? | `rl_training/outputs/qwen3_32b_sft_eval/{eval,clinical}.json` |
| Qwen3-32B (SFT + GRPO-clinical) | ? | ? | ? | ? | ? | ? | `rl_training/outputs/qwen3_32b_post_train_sft/{eval,clinical}.json` |

`—` means the column isn't produced for that model (the gpt-4o-mini runs
predate the clinical metrics pass). Regenerate them by re-running the
stress eval against the already-persisted trajectories if you want
apples-to-apples.

The intended reporting line is **Qwen SFT+GRPO > Qwen SFT > Qwen OOTB**
on SR, and **Qwen SFT+GRPO ≤ Qwen SFT ≤ Qwen OOTB** on temporal
inconsistency / over-deferral (rewards are designed to pull those down).
If either ordering flips, the training config needs another look before
claiming a win.
