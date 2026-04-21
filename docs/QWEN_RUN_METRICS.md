# Qwen3-32B Run — Three Accuracy Checkpoints

This run measures MedAgentBench v2 success rate (SR) at **three** model states and
persists every result both locally on the pod and in Backblaze B2. Each eval also
writes per-task trajectories and a clinical-metrics JSON so the raw data is
recoverable and re-gradable after the run ends.

Bucket: `medagentbench-checkpoint` · Run prefix: `qwen3_32b_run1/`

## Recorded results

| Metric | model | total | correct | SR | query_SR | action_SR | invalid_action_rate | avg_steps |
|---|---|---|---|---|---|---|---|---|
| #1 baseline | `Qwen/Qwen3-32B` | 300 | 104 | **34.67%** | 68.67% | 0.67% | 1.33% | 1.77 |
| #2 after SFT | `qwen3_32b_sft_merged` | 300 | 103 | **34.33%** | 68.00% | 0.67% | 4.00% | 1.84 |
| #3 after SFT+RL | _pending_ | — | — | — | — | — | — | — |

Per-task SR snapshot (baseline → SFT):
- task1: 100% → 100%
- task2: 100% → 100%
- task3: 0%   → 0%
- task4: 96.7% → 86.7% (regression)
- task5: 3.3% → 3.3%
- task6: 33.3% → 36.7% (slight gain)
- task7: 13.3% → 16.7% (slight gain)
- task8/9/10: 0% → 0% (action-heavy tasks; SFT did not unlock them)

## Metric #1 — Baseline (stock Qwen3-32B)

- **When:** Phase 2, on Pod A, before any training.
- **Script:** `rl_training/scripts/run_baseline_eval.py --model Qwen/Qwen3-32B`
- **Local dir:** `rl_training/outputs/qwen3_32b_baseline/`
  - `eval.json` — `{model_id, total, correct, success_rate, query_sr, action_sr, per_type_sr}`
  - `clinical.json` — hallucination / oversaturation / plausible-action metrics
  - `trajectories.jsonl` — full per-task history for regrading
  - `eval.log` — console log
- **B2 mirror:** `qwen3_32b_run1/qwen3_32b_baseline/`

## Metric #2 — After SFT (stock Qwen3-32B + QLoRA SFT, merged)

- **When:** Phase 3, on Pod A, after SFT completes and LoRA is merged into the base.
- **Script:** `rl_training/scripts/run_baseline_eval.py --model /workspace/qwen3_32b_sft_merged`
  (the merged checkpoint produced by `rl_training/scripts/merge_lora.py`)
- **Local dir:** `rl_training/outputs/qwen3_32b_sft/`
  - `eval.json`, `clinical.json`, `trajectories.jsonl`, `eval.log` (same shape as #1)
- **B2 mirrors:**
  - `qwen3_32b_run1/qwen3_32b_sft/` (eval artifacts)
  - `qwen3_32b_run1/sft_merged/` (the merged model weights so Pod B can fetch them for GRPO)

## Metric #3 — After SFT + RL (SFT-merged base + GRPO LoRA adapter, best step)

- **When:** Phase 5, on Pod B, after GRPO completes.
- **Script:** `rl_training/scripts/run_post_train_eval.py --merge-and-serve \
    --base-model /workspace/qwen3_32b_sft_merged \
    --lora-ckpt <best GRPO step dir>`
- **Local dir:** `rl_training/outputs/qwen3_32b_sft_grpo/`
  - `eval.json`, `clinical.json`, `trajectories.jsonl`, `eval.log`
- **B2 mirror:** `qwen3_32b_run1/qwen3_32b_sft_grpo/`

## Phase 6 — Comparison artifact

After all three metrics are collected I will write a single comparison file:

- **Local:** `rl_training/outputs/qwen3_32b_run1/comparison.json` + `comparison.md`
- **B2:** `qwen3_32b_run1/comparison.json`, `qwen3_32b_run1/comparison.md`

Shape (`comparison.json`):
```json
{
  "run": "qwen3_32b_run1",
  "base_model": "Qwen/Qwen3-32B",
  "metrics": {
    "baseline":     { "success_rate": 0.000, "query_sr": 0.000, "action_sr": 0.000, "total": 300, "correct": 0, "clinical": {...} },
    "after_sft":    { "success_rate": 0.000, "query_sr": 0.000, "action_sr": 0.000, "total": 300, "correct": 0, "clinical": {...} },
    "after_sft_rl": { "success_rate": 0.000, "query_sr": 0.000, "action_sr": 0.000, "total": 300, "correct": 0, "clinical": {...} }
  },
  "deltas": {
    "sft_vs_baseline":    { "success_rate": 0.000, "query_sr": 0.000, "action_sr": 0.000 },
    "sft_rl_vs_sft":      { "success_rate": 0.000, "query_sr": 0.000, "action_sr": 0.000 },
    "sft_rl_vs_baseline": { "success_rate": 0.000, "query_sr": 0.000, "action_sr": 0.000 }
  },
  "acceptance": {
    "sr_monotone_non_decreasing": false,
    "clinical_metrics_stable_or_better": false
  }
}
```

`comparison.md` will contain a rendered table with the same numbers plus per-type
SR columns for manual inspection.

## Durability guarantees

1. **Local persist:** every eval script writes to disk *before* returning.
2. **B2 mirror:** each phase's orchestration script (`pod_phase2.sh`, SFT stage wrapper, GRPO eval wrapper) calls `make_backend("b2", ...).upload_directory(...)` after the eval succeeds, so results survive pod termination.
3. **Raw trajectories are kept** (`trajectories.jsonl`) so any grader bug can be re-run without re-spending on GPUs.
4. **Redundancy:** `eval.json` is also echoed into `eval.log` via the console summary.

## Recovery from loss

If a pod dies between eval completion and upload, the results are still on the
pod's `/workspace` volume until the pod is terminated. The watchdog `stop`s the
pod rather than `terminate`s it when possible, so volumes can be re-attached
and artifacts pulled. If a pod is hard-terminated mid-eval, re-running the
eval phase is the fallback (costs one eval run, ~$10-18).
