---
title: "MedAgentBench Fine-Tuning Report"
subtitle: "GPT-4o-mini baseline, supervised fine-tuning, and benchmark results"
author: "MedAgentBench project (internal run)"
date: "April 2026"
---

# Executive summary

This report documents evaluation of **GPT-4o-mini** on the **MedAgentBench** benchmark (`test_data_v2.json`, 300 tasks), a **supervised fine-tuning** run via the OpenAI API, and **post–fine-tuning** evaluation on the **identical** benchmark. Overall success rate improved from **58.0%** to **72.3%** (+14.3 percentage points), with **Action success rate** rising from **51.3%** to **78.7%**. These results support the claim that **healthcare-oriented agent benchmarks leave substantial room for domain-targeted adaptation** (e.g., API-hosted fine-tuning), beyond off-the-shelf foundation model scores.

---

# 1. Benchmark context

**MedAgentBench** benchmarks medical LLM agents against a **virtual FHIR EHR** (typically `http://localhost:8080/fhir/` via Docker). Agents must use a fixed protocol:

- **GET** — FHIR URLs with query parameters  
- **POST** — JSON payloads to FHIR endpoints  
- **FINISH** — JSON-encodable answer list  

Episodes are capped at **8 rounds** (aligned with repository defaults).

**Dataset:** `data/medagentbench/test_data_v2.json` — **300** scenarios across **10 task types** (patient lookup, vitals, labs, medication orders, service requests, etc.). Grading uses reference solutions (`refsol`) consistent with the original MedAgentBench / AgentBench implementation.

**Metrics** (from the `rl_training` evaluator): overall **success rate (SR)**; **query SR** vs **action SR** (by task family); **invalid action rate**; **average steps** per episode.

---

# 2. Published work and paper-style reference in the repo

The benchmark is described in **Jiang et al., *NEJM AI*** (MedAgentBench; see repository README).

The repository README evaluates **`gpt-4o-mini`** through the full **AgentBench** stack (`src.start_task` + `src.assigner`). Aggregated results are stored in:

`outputs/MedAgentBenchv1/gpt-4o-mini/medagentbench-std/overall.json`

The aggregate success metric appears under **`custom` → `"success rate"`** as **0.59** (**59%**) on the same 300-case file. That value is the standard **in-repo, AgentBench-harness** reference for the paper’s quick-start reproduction.

**Methodological note:** Comparisons in Sections 4–5 use the **`rl_training` in-process harness** (`MedAgentEnv` + `OpenAIPolicy` + same JSON tasks + same graders), which is **not** guaranteed to be byte-identical to the AgentBench controller path. For a manuscript, report these as **“MedAgentBench tasks under our evaluation harness”** unless both base and fine-tuned models are also re-evaluated via `src.assigner`.

---

# 3. Baseline: out-of-the-box GPT-4o-mini

| Field | Value |
|--------|--------|
| **Model** | `gpt-4o-mini` (OpenAI Chat Completions, temperature 0) |
| **Harness** | `rl_training/scripts/evaluate.py` + `rl_training/configs/gpt4o_mini_openai.yaml` |
| **Artifact** | `rl_training/outputs/gpt4o_mini_user_run/01_baseline_gpt4o_mini_ootb.json` |

| Metric | Value |
|--------|--------|
| Overall SR | **58.0%** (174/300) |
| Query SR | 64.7% |
| Action SR | 51.3% |
| Invalid action rate | 0.0% |
| Avg steps | 2.25 |

**Comparison to paper-style 59%:** The **58%** vs **59%** gap is small and consistent with run variance, API snapshot drift, and harness differences. The baseline is **aligned in spirit** with the published ballpark.

---

# 4. Fine-tuning procedure

**Goal:** Supervised fine-tuning (SFT) so the model better matches **correct MedAgentBench-style trajectories** (structured GET/POST/FINISH behavior).

| Field | Value |
|--------|--------|
| **Training examples** | 1,361 **correct** trajectories |
| **Source file** | `rl_training/outputs/gpt4o_pipeline/phase_a/expert_trajectories.jsonl` |
| **Base model (API)** | `gpt-4o-mini-2024-07-18` |
| **Job suffix** | `medagent-sft-mini` |
| **Epochs** | 3 |
| **API** | OpenAI Fine-tuning Jobs |

**Resulting model ID:**

`ft:gpt-4o-mini-2024-07-18:personal:medagent-sft-mini:DVQWrpzL`

Also recorded in `rl_training/outputs/gpt4o_mini_user_run/finetuned_model_id.txt`.

**What fine-tuning means:** OpenAI trains a **new hosted checkpoint**; inference uses the **`ft:...`** model id. Weights are updated **on OpenAI’s infrastructure**; no local weight files are produced.

---

# 5. Post–fine-tuning evaluation (same 300 tasks, same harness)

| Field | Value |
|--------|--------|
| **Model** | `ft:gpt-4o-mini-2024-07-18:personal:medagent-sft-mini:DVQWrpzL` |
| **Harness** | Same as Section 3 |
| **Artifact** | `rl_training/outputs/gpt4o_mini_user_run/02_finetuned_gpt4o_mini_benchmark.json` |

| Metric | Baseline | Fine-tuned | Delta |
|--------|----------|------------|--------|
| Overall SR | 58.0% | **72.3%** (217/300) | **+14.3 pp** |
| Query SR | 64.7% | 66.0% | +1.3 pp |
| Action SR | 51.3% | **78.7%** | **+27.4 pp** |
| Invalid rate | 0.0% | 2.3% | +2.3 pp |
| Avg steps | 2.25 | 2.06 | -0.19 |

**Per-task SR (selected):** task7 **0% to 36.7%**; task9 **0% to 50%**; task10 **0% to 66.7%**; task6 **63.3% to 23.3%** (mixed; warrants error analysis in a full paper).

---

# 6. Discussion (paper angle)

1. **Headroom for adaptation:** Off-the-shelf **GPT-4o-mini** sits near **~58–59%** SR on this benchmark; **targeted SFT** on domain-faithful trajectories yields a **large gain** on the **same** task distribution, especially **Action SR**—where prior work often stresses weakness for structured clinical actions.

2. **Practical lever:** Hosted SFT improves agent behavior **without** local GPU training, relevant to teams that cannot run open-weight RL.

3. **Limitations to disclose:** (a) harness choice when comparing to AgentBench `overall.json`; (b) provenance of expert data and any overlap with evaluation design; (c) slight **invalid-rate** increase; (d) single snapshot / single run; (e) **SFT ≠ RL**—claims should emphasize **adaptation** and **supervised alignment to demonstrations**, not policy-gradient RL unless that is actually run.

4. **Reporting norm:** Benchmark papers could standardize reporting of **base vs adapted** models on the same JSON eval to make **clinical agent adaptation** scientifically comparable.

---

# 7. Closing (one paragraph)

MedAgentBench evaluates LLM agents on **realistic FHIR tasks**. **GPT-4o-mini** achieves about **59%** under the repository’s AgentBench reference and **58%** under our **300-task** `rl_training` replay. **Supervised fine-tuning** on **1,361** correct expert trajectories produces **`ft:gpt-4o-mini-...`**, which reaches **72.3%** overall SR on the **identical** benchmark file, with **Action SR** rising from **51.3%** to **78.7%**. Together, these findings support the thesis that **healthcare agent benchmarks exhibit substantial room for fine-tuning**, and that **post-adaptation** metrics should accompany claims about clinical readiness of foundation models.

---

# Appendix: File paths

| Description | Path |
|-------------|------|
| Baseline JSON | `rl_training/outputs/gpt4o_mini_user_run/01_baseline_gpt4o_mini_ootb.json` |
| Fine-tuned eval JSON | `rl_training/outputs/gpt4o_mini_user_run/02_finetuned_gpt4o_mini_benchmark.json` |
| Fine-tuned model id | `rl_training/outputs/gpt4o_mini_user_run/finetuned_model_id.txt` |
| Eval config | `rl_training/configs/gpt4o_mini_openai.yaml` |
| Paper-style overall (59%) | `outputs/MedAgentBenchv1/gpt-4o-mini/medagentbench-std/overall.json` |
