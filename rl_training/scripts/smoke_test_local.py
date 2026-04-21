#!/usr/bin/env python3
"""Local smoke tests for the Qwen3-32B GRPO stack.

Three modes, in increasing cost and realism:

  * ``--mode unit`` - CPU-only, <30 s. Fabricates completions and task
    dicts to validate rewards, FHIR snapshot, env hooks, perturbations,
    and clinical metrics. No model load, no network. This is the gate
    that must pass before any cloud spend.

  * ``--mode small`` - Laptop GPU/CPU, <10 min. Loads a tiny Qwen (0.5B
    or 1.5B) and runs a 2-step GRPO loop against 2 synthetic tasks. This
    is the integration test for the TRL loop itself.

  * ``--mode live`` - 2xH100 cloud preflight, ~$0.50-$1. Points at a
    running vLLM server, does one full 32B rollout against one real
    benchmark task, validates the trajectory, and confirms clinical
    metrics compute cleanly. This is the last gate before committing to
    the ~$50 fine-tune run.

Usage:
    python rl_training/scripts/smoke_test_local.py --mode unit
    python rl_training/scripts/smoke_test_local.py --mode small
    python rl_training/scripts/smoke_test_local.py --mode live \\
        --vllm-base-url http://127.0.0.1:8000/v1 \\
        --model Qwen/Qwen3-32B-Instruct \\
        --config rl_training/configs/qwen3_32b_grpo.yaml \\
        --output-dir rl_training/outputs/live_smoke
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import traceback
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("smoke_test_local")


# --------------------------------------------------------- helpers for unit mode


class SmokeTestFailure(AssertionError):
    pass


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise SmokeTestFailure(msg)


def _fabricate_completion(
    finish_answer: str = "[1.2]",
    assistant_text: str = "",
    tool_calls: list[dict] | None = None,
) -> list[dict]:
    """Build a completion list in the shape TRL passes to reward functions."""
    turns: list[dict] = []
    if assistant_text:
        turns.append({"role": "assistant", "content": assistant_text})
    if tool_calls:
        turns.append({"role": "assistant", "tool_calls": tool_calls})
    if finish_answer is not None:
        turns.append({
            "role": "assistant",
            "tool_calls": [{
                "function": {
                    "name": "finish",
                    "arguments": {"answers": finish_answer},
                },
            }],
        })
    return turns


# ------------------------------------------------------------------- unit mode


def run_unit_tests() -> int:
    start = time.time()
    failures: list[str] = []

    def _run(name: str, fn):
        t0 = time.time()
        try:
            fn()
            dt = time.time() - t0
            logger.info("PASS  %-40s  (%.3fs)", name, dt)
        except Exception as exc:
            dt = time.time() - t0
            logger.error("FAIL  %-40s  (%.3fs)  %s", name, dt, exc)
            logger.debug(traceback.format_exc())
            failures.append(f"{name}: {exc}")

    # 1. FHIR snapshot
    def test_fhir_snapshot_record_replay():
        from rl_training.env.fhir_snapshot import FhirSnapshot, _canonicalize_url

        calls = []

        def fake_live(url):
            calls.append(url)
            return {"status_code": 200, "data": {"url": url}}

        snap = FhirSnapshot(mode="record", fallthrough=True, live_getter=fake_live)
        r1 = snap.send_get_request("http://x/Patient?a=1&b=2&_format=json")
        r2 = snap.send_get_request("http://x/Patient?b=2&a=1")
        _assert(r1["status_code"] == 200, "first call should hit live")
        _assert(len(calls) == 1, f"expected 1 live call after canonicalization, got {len(calls)}")
        _assert(r2["data"] == r1["data"], "replay should return same data")

        snap2 = FhirSnapshot(mode="replay", fallthrough=False, live_getter=fake_live)
        miss = snap2.send_get_request("http://x/Patient")
        _assert("error" in miss, "strict replay should error on miss")

        canon_a = _canonicalize_url("http://x/Observation?patient=1&code=MG&_format=json")
        canon_b = _canonicalize_url("http://x/Observation?code=MG&patient=1")
        _assert(canon_a == canon_b, f"canonicalization mismatch: {canon_a!r} vs {canon_b!r}")

    _run("fhir_snapshot.record_replay", test_fhir_snapshot_record_replay)

    # 2. Env hooks (_tool_log + snapshot integration)
    def test_env_hooks():
        from rl_training.env.fhir_snapshot import (
            FhirSnapshot, SnapshotEntry, _canonicalize_url,
        )
        from rl_training.env.trl_env import MedAgentBenchEnv

        obs_bundle = {
            "entry": [
                {"resource": {
                    "resourceType": "Observation",
                    "effectiveDateTime": "2023-11-13T09:00:00+00:00",
                    "valueQuantity": {"value": 1.8, "unit": "mg/dL"},
                }},
            ],
        }
        snap = FhirSnapshot(mode="replay", fallthrough=False)
        # Env appends &_format=json and then canonicalizes, so pre-seed with
        # the canonical form of the URL the env will actually look up.
        lookup_url = "http://fake/Observation?patient=P1&code=MG&_format=json"
        key = _canonicalize_url(lookup_url)
        snap._cache[key] = SnapshotEntry(  # noqa: SLF001
            url=key, status_code=200, data=obs_bundle,
        )

        env = MedAgentBenchEnv(snapshot=snap)
        env.reset(task_id="task4_1", eval_MRN="P1", instruction="q", context="")
        out = env.get_fhir_resource("http://fake/Observation?patient=P1&code=MG")
        _assert("1.8" in out, f"GET response should include value, got {out[:100]}")
        _assert(len(env._tool_log) == 1, f"expected 1 tool log entry, got {len(env._tool_log)}")
        entry = env._tool_log[0]
        _assert(entry["action"] == "GET", "action should be GET")
        _assert(entry["success"] is True, "success should be True")
        _assert(
            any("2023-11-13T09:00:00" in ts for ts in entry["timestamps"]),
            f"timestamp not extracted: {entry['timestamps']}",
        )

    _run("env_hooks.tool_log_and_snapshot", test_env_hooks)

    # 3. Reward: correctness (base reward, just confirm it's importable + callable)
    def test_base_rewards_callable():
        from rl_training.env.trl_rewards import (
            correctness_reward, efficiency_reward, tool_usage_reward,
        )
        completions = [_fabricate_completion(finish_answer="[1.5]")]
        r1 = correctness_reward(completions, task_id=["task4_1"])
        r2 = efficiency_reward(completions, task_id=["task4_1"])
        r3 = tool_usage_reward(completions, task_id=["task4_1"])
        for name, r in [("correctness", r1), ("efficiency", r2), ("tool_usage", r3)]:
            _assert(len(r) == 1, f"{name} returned wrong shape: {r}")
            _assert(isinstance(r[0], (int, float)), f"{name} returned non-numeric: {r}")

    _run("rewards.base_callable", test_base_rewards_callable)

    # 4. Reward: Temporal Clinical Grounding
    def test_reward_tcg():
        from rl_training.env.trl_rewards_clinical import temporal_grounding_reward
        from rl_training.env.trl_env import MedAgentBenchEnv

        env = MedAgentBenchEnv()
        env.reset(task_id="task4_1", eval_MRN="P1", instruction="mg", context="")
        # Simulate env having seen two timestamps: one in-window (9am), one way old
        env._tool_log = [{
            "step": 1, "action": "GET", "url": "x", "success": True,
            "timestamps": ["2023-11-13T09:00:00+00:00", "2020-01-01T00:00:00+00:00"],
            "response_len": 10,
        }]

        # Case A: model cites the in-window timestamp -> +1.0
        completion_good = _fabricate_completion(
            assistant_text="The magnesium level at 2023-11-13T09:00:00+00:00 was 1.8 mg/dL.",
            finish_answer="[1.8]",
        )
        r = temporal_grounding_reward([completion_good], environments=[env],
                                      task_id=["task4_1"])
        _assert(r[0] == 1.0, f"good TCG: expected +1.0, got {r[0]}")

        # Case B: model cites the out-of-window timestamp -> -1.0
        completion_bad = _fabricate_completion(
            assistant_text="Using 2020-01-01T00:00:00+00:00 measurement.",
            finish_answer="[1.8]",
        )
        r = temporal_grounding_reward([completion_bad], environments=[env],
                                      task_id=["task4_1"])
        _assert(r[0] == -1.0, f"bad TCG: expected -1.0, got {r[0]}")

        # Case C: no timestamp cited -> -0.5
        completion_none = _fabricate_completion(
            assistant_text="the mg level is 1.8",
            finish_answer="[1.8]",
        )
        r = temporal_grounding_reward([completion_none], environments=[env],
                                      task_id=["task4_1"])
        _assert(r[0] == -0.5, f"no-citation TCG: expected -0.5, got {r[0]}")

        # Case D: non-time-scoped task -> 0.0 regardless
        env.reset(task_id="task8_1", eval_MRN="P1", instruction="", context="")
        r = temporal_grounding_reward([completion_bad], environments=[env],
                                      task_id=["task8_1"])
        _assert(r[0] == 0.0, f"non-scoped TCG: expected 0.0, got {r[0]}")

    _run("rewards.temporal_grounding", test_reward_tcg)

    # 5. Reward: Risk-Calibrated Deferral
    def test_reward_rcd():
        from rl_training.env.trl_rewards_clinical import (
            risk_calibrated_deferral_reward,
        )
        from rl_training.env.trl_env import MedAgentBenchEnv

        # Task 5: magnesium replacement. If env observed mg < 1.9, should POST.
        env = MedAgentBenchEnv()
        env.reset(task_id="task5_1", eval_MRN="P1", instruction="", context="")
        env._tool_log = [{
            "step": 1, "action": "GET", "url": "x", "success": True,
            "timestamps": ["2023-11-13T09:00:00+00:00"],
            "response_len": 100,
            "response": (
                '{"entry":[{"resource":{"effectiveDateTime":"2023-11-13T09:00:00+00:00",'
                '"valueQuantity":{"value":1.2}}}]}'
            ),
        }]
        # Posting was correct -> +1.0
        env._post_count = 1
        r = risk_calibrated_deferral_reward(
            [_fabricate_completion(finish_answer="[1.2]")],
            environments=[env], task_id=["task5_1"],
        )
        _assert(r[0] == 1.0, f"RCD correct post: expected +1.0, got {r[0]}")

        # No POST when one was warranted -> -1.0
        env._post_count = 0
        r = risk_calibrated_deferral_reward(
            [_fabricate_completion(finish_answer="[1.2]")],
            environments=[env], task_id=["task5_1"],
        )
        _assert(r[0] == -1.0, f"RCD under-deferral: expected -1.0, got {r[0]}")

        # Non-conditional task -> 0.0
        env.reset(task_id="task4_1", eval_MRN="P1", instruction="", context="")
        r = risk_calibrated_deferral_reward(
            [_fabricate_completion(finish_answer="[1.2]")],
            environments=[env], task_id=["task4_1"],
        )
        _assert(r[0] == 0.0, f"RCD non-conditional: expected 0.0, got {r[0]}")

    _run("rewards.risk_calibrated_deferral", test_reward_rcd)

    # 6. Reward: Decision Density Bonus
    def test_reward_ddb():
        from rl_training.env.trl_rewards_clinical import decision_density_reward

        # Short, JSON-parseable answer -> +0.3
        r = decision_density_reward([_fabricate_completion(finish_answer="[1.5]")])
        _assert(r[0] == 0.3, f"DDB short: expected 0.3, got {r[0]}")

        # Medium, JSON-parseable answer -> 0.1 (approx; build one)
        medium = "[" + ", ".join([f"{i}" for i in range(150)]) + "]"
        r = decision_density_reward([_fabricate_completion(finish_answer=medium)])
        _assert(r[0] == 0.1, f"DDB medium: expected 0.1, got {r[0]}")

        # Long -> -0.1
        long_answer = "[" + ", ".join([f"{i}" for i in range(600)]) + "]"
        r = decision_density_reward([_fabricate_completion(finish_answer=long_answer)])
        _assert(r[0] == -0.1, f"DDB long: expected -0.1, got {r[0]}")

        # Non-JSON -> -0.1
        r = decision_density_reward([_fabricate_completion(finish_answer="banana")])
        _assert(r[0] == -0.1, f"DDB non-json: expected -0.1, got {r[0]}")

    _run("rewards.decision_density", test_reward_ddb)

    # 7. Timeline perturbations
    def test_timeline_perturb():
        from rl_training.data.timeline_perturb import (
            all_perturbations, perturb_tasks, perturb_tasks_multi,
        )
        tasks = [{
            "id": "task4_1",
            "eval_MRN": "P1",
            "context": "It's 2023-11-13T10:15:00+00:00 now. The code for magnesium is \"MG\".",
            "instruction": "What's the most recent magnesium level of the patient P1 within last 24 hours?",
        }]
        names = all_perturbations()
        _assert(set(names) >= {"timestamp_shuffle", "active_history_swap",
                               "contradictory_note", "distractor_padding"},
                f"missing perturbations: {names}")

        shifted = perturb_tasks(tasks, "timestamp_shuffle", seed=1)
        _assert(len(shifted) == 1, "perturb_tasks should preserve count")
        _assert(shifted[0]["id"] != tasks[0]["id"], "id should change after perturbation")
        _assert("_perturbation" in shifted[0], "perturbation metadata missing")

        multi = perturb_tasks_multi(tasks, seed=1)
        _assert(set(multi.keys()) == set(names), "multi should cover all axes")

    _run("perturb.timeline_primitives", test_timeline_perturb)

    # 8. tasks_to_dataset - requires the `datasets` package; skip if absent.
    def test_tasks_to_dataset():
        try:
            import datasets  # noqa: F401
        except ModuleNotFoundError:
            logger.warning("SKIP  data.tasks_to_dataset (datasets pkg not installed)")
            return
        from rl_training.data.prepare_dataset import tasks_to_dataset
        tasks = [{
            "id": "task4_1", "eval_MRN": "P1", "context": "C", "instruction": "I",
        }]
        ds = tasks_to_dataset(tasks, fhir_api_base="http://fake/fhir/")
        _assert(len(ds) == 1, f"dataset size wrong: {len(ds)}")
        _assert(set(ds.column_names) >= {"prompt", "task_id", "eval_MRN",
                                         "instruction", "context"},
                f"missing columns: {ds.column_names}")
        _assert(ds[0]["task_id"] == "task4_1", "task_id round-trip failed")

    _run("data.tasks_to_dataset", test_tasks_to_dataset)

    # 9. SFT: OpenAI-format JSONL round-trip + Qwen chat template shape.
    def test_sft_openai_jsonl_roundtrip():
        import json as _json
        import tempfile

        from rl_training.data.trajectory import Trajectory, Turn
        from rl_training.data.trajectory_store import TrajectoryStore

        trajs = []
        for i in range(3):
            trajs.append(Trajectory(
                task_id=f"task1_{i}",
                task_data={"id": f"task1_{i}"},
                turns=[
                    Turn(role="user", content=f"Find MRN for patient {i}"),
                    Turn(role="assistant",
                         content=f"GET http://fhir/Patient?identifier=P{i}"),
                    Turn(role="user", content='{"entry": [...]}'),
                    Turn(role="assistant", content=f'FINISH(["P{i}"])'),
                ],
                correct=True, status="completed", num_steps=2,
            ))

        with tempfile.TemporaryDirectory() as tmp:
            traj_path = Path(tmp) / "expert.jsonl"
            sft_path = Path(tmp) / "qwen_sft_openai.jsonl"
            store = TrajectoryStore(traj_path)
            store.save_batch(trajs)
            store.export_openai_jsonl(sft_path, trajs)

            lines = sft_path.read_text().strip().splitlines()
            _assert(len(lines) == 3, f"expected 3 sft lines, got {len(lines)}")
            for i, line in enumerate(lines):
                obj = _json.loads(line)
                _assert("messages" in obj,
                        f"row {i} missing 'messages' key: {list(obj)}")
                msgs = obj["messages"]
                _assert(len(msgs) == 4, f"row {i} has {len(msgs)} messages, expected 4")
                _assert(msgs[0]["role"] == "user", f"row {i}[0] role={msgs[0]['role']}")
                _assert(msgs[1]["role"] == "assistant",
                        f"row {i}[1] role={msgs[1]['role']}")

            # Qwen chat template (mock). This mirrors what
            # AutoTokenizer.from_pretrained("Qwen/Qwen3-*").apply_chat_template
            # produces and is what sft_qwen3_32b.py relies on SFTTrainer
            # invoking for us at training time. We verify the structural
            # markers so a template mismatch surfaces here, not after a $10
            # cloud run.
            def _apply_qwen_chat_template(messages: list[dict]) -> str:
                parts = []
                for m in messages:
                    parts.append(
                        f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>"
                    )
                return "\n".join(parts)

            obj0 = _json.loads(lines[0])
            rendered = _apply_qwen_chat_template(obj0["messages"])
            _assert(
                "<|im_start|>user" in rendered,
                f"expected '<|im_start|>user' marker; got: {rendered[:120]!r}",
            )
            _assert(
                "<|im_start|>assistant" in rendered,
                f"expected '<|im_start|>assistant' marker; got: {rendered[:120]!r}",
            )
            _assert(
                rendered.count("<|im_end|>") == 4,
                f"expected 4 '<|im_end|>' markers, got {rendered.count('<|im_end|>')}",
            )

            # Simulate the _load_sft_dataset path in sft_qwen3_32b.py so we
            # know the same file it will eat at training time is well-formed.
            try:
                from datasets import Dataset
            except ModuleNotFoundError:
                logger.warning(
                    "SKIP partial sft_data.openai_jsonl_roundtrip (datasets pkg not installed)"
                )
                return
            rows = []
            with open(sft_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rows.append({"messages": _json.loads(line)["messages"]})
            ds = Dataset.from_list(rows)
            _assert(len(ds) == 3, f"HF Dataset wrong size: {len(ds)}")
            _assert("messages" in ds.column_names,
                    f"messages column missing: {ds.column_names}")

    _run("sft_data.openai_jsonl_roundtrip", test_sft_openai_jsonl_roundtrip)

    # 10. SFT: export_openai_jsonl must filter out correct=False trajectories.
    def test_sft_filter_correct():
        import json as _json
        import tempfile

        from rl_training.data.trajectory import Trajectory, Turn
        from rl_training.data.trajectory_store import TrajectoryStore

        mixed = [
            Trajectory(
                task_id="task1_ok",
                task_data={"id": "task1_ok"},
                turns=[
                    Turn(role="user", content="q"),
                    Turn(role="assistant", content='FINISH(["ok"])'),
                ],
                correct=True, status="completed", num_steps=1,
            ),
            Trajectory(
                task_id="task1_bad1",
                task_data={"id": "task1_bad1"},
                turns=[
                    Turn(role="user", content="q"),
                    Turn(role="assistant", content='FINISH(["wrong"])'),
                ],
                correct=False, status="completed", num_steps=1,
            ),
            Trajectory(
                task_id="task1_ok2",
                task_data={"id": "task1_ok2"},
                turns=[
                    Turn(role="user", content="q"),
                    Turn(role="assistant", content='FINISH(["ok2"])'),
                ],
                correct=True, status="completed", num_steps=1,
            ),
            Trajectory(
                task_id="task1_bad2",
                task_data={"id": "task1_bad2"},
                turns=[
                    Turn(role="user", content="q"),
                    Turn(role="assistant", content='FINISH(["still wrong"])'),
                ],
                correct=False, status="limit_reached", num_steps=1,
            ),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            traj_path = Path(tmp) / "expert.jsonl"
            sft_path = Path(tmp) / "qwen_sft_openai.jsonl"
            store = TrajectoryStore(traj_path)
            store.save_batch(mixed)

            correct_trajs = store.filter(correct=True)
            _assert(
                len(correct_trajs) == 2,
                f"filter(correct=True) returned {len(correct_trajs)}, expected 2",
            )
            kept_ids = {t.task_id for t in correct_trajs}
            _assert(
                kept_ids == {"task1_ok", "task1_ok2"},
                f"wrong ids kept: {kept_ids}",
            )

            store.export_openai_jsonl(sft_path, correct_trajs)
            lines = sft_path.read_text().strip().splitlines()
            _assert(
                len(lines) == 2,
                f"SFT jsonl should have 2 lines (only correct), got {len(lines)}",
            )
            for line in lines:
                obj = _json.loads(line)
                contents = [m["content"] for m in obj["messages"]]
                joined = " ".join(contents)
                _assert(
                    "wrong" not in joined,
                    f"wrong trajectory leaked into SFT jsonl: {joined[:120]}",
                )

    _run("sft_data.filter_correct", test_sft_filter_correct)

    # 11. Clinical metrics on fabricated trajectories
    def test_clinical_metrics():
        from rl_training.data.trajectory import Trajectory, Turn
        from rl_training.evaluation.clinical_metrics import compute_clinical_metrics

        # One task4 trajectory that cites an out-of-window ts in the answer
        trajs = [
            Trajectory(
                task_id="task4_1",
                task_data={"id": "task4_1"},
                turns=[
                    Turn(role="user", content="query"),
                    Turn(role="assistant",
                         content="Using 2020-01-01T00:00:00+00:00 data. FINISH([1.8])"),
                ],
                correct=False, status="completed", num_steps=1,
            ),
            Trajectory(
                task_id="task4_2",
                task_data={"id": "task4_2"},
                turns=[
                    Turn(role="user", content="query"),
                    Turn(role="assistant",
                         content="GET url=foo. FINISH([1.9])"),
                ],
                correct=True, status="completed", num_steps=1,
            ),
        ]
        result = compute_clinical_metrics(trajs)
        _assert(result.total == 2, f"total wrong: {result.total}")
        _assert(
            result.temporal_inconsistency_rate > 0.4,
            f"should flag out-of-window citation: {result.temporal_inconsistency_rate}",
        )

    _run("eval.clinical_metrics", test_clinical_metrics)

    elapsed = time.time() - start
    if failures:
        logger.error("FAILED %d/%d tests in %.2fs", len(failures), 11, elapsed)
        for f in failures:
            logger.error("  - %s", f)
        return 1
    logger.info("ALL UNIT TESTS PASSED in %.2fs", elapsed)
    return 0


# ------------------------------------------------------------- small mode (stub)


def run_small_mode(args: argparse.Namespace) -> int:
    """Laptop-scale GRPO loop with a tiny Qwen.

    Validates the full TRL integration end-to-end without renting a GPU.
    Target: <10 min on M-series Mac MPS. On CPU-only it is still correct
    but can take closer to 30 min.

    What it proves:
      * The Qwen tokenizer + model load via transformers.
      * The TRL ``GRPOTrainer`` + ``GRPOConfig`` flags align with the
        pins in ``requirements-gpu.txt``.
      * The reward functions accept the TRL completion shape and return
        per-rollout scalars.
      * The ``MedAgentBenchEnv`` (tool-calling flavour) composes with
        the trainer without errors, even with the tool loop stubbed out.

    What it does NOT prove:
      * 32B memory fit (that is what ``--mode live`` checks on the cloud box).
      * vLLM server mode (skipped; ``use_vllm=False`` in this mode).
    """
    import importlib.util
    missing = [
        mod for mod in ("torch", "transformers", "trl", "peft")
        if importlib.util.find_spec(mod) is None
    ]
    if missing:
        logger.error(
            "--mode small needs: %s. Install with: pip install -r requirements-gpu.txt",
            ", ".join(missing),
        )
        return 2

    import tempfile

    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    from rl_training.env.trl_rewards import correctness_reward
    from rl_training.env.trl_rewards_clinical import (
        decision_density_reward,
    )

    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    logger.info("Device: %s  model: %s  steps: %d  tasks: %d",
                device, args.small_model, args.max_steps, args.num_tasks)

    dtype = torch.float16 if device != "cpu" else torch.float32
    tok = AutoTokenizer.from_pretrained(args.small_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.small_model, torch_dtype=dtype, trust_remote_code=True,
    )
    model.to(device)

    # Tiny synthetic dataset: ``num_tasks`` tasks of the "what's 1+1" flavour.
    records: dict[str, list] = {
        "prompt": [], "task_id": [], "eval_MRN": [], "instruction": [],
        "context": [],
    }
    for i in range(args.num_tasks):
        records["prompt"].append([
            {"role": "user",
             "content": "Return the string '[1.5]'. Only output the final token."},
        ])
        records["task_id"].append(f"task4_{i+1}")
        records["eval_MRN"].append(f"P{i}")
        records["instruction"].append("")
        records["context"].append("")
    ds = Dataset.from_dict(records)

    with tempfile.TemporaryDirectory() as tmp:
        cfg = GRPOConfig(
            output_dir=tmp,
            max_steps=args.max_steps,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            num_generations=2,
            max_prompt_length=128,
            max_completion_length=32,
            learning_rate=1e-5,
            beta=0.04,
            temperature=1.0,
            logging_steps=1,
            save_steps=10_000,
            use_vllm=False,
            bf16=False, fp16=False,
            report_to=[],
        )
        trainer = GRPOTrainer(
            model=model,
            args=cfg,
            train_dataset=ds,
            processing_class=tok,
            reward_funcs=[correctness_reward, decision_density_reward],
        )
        logger.info("Starting small GRPO smoke run...")
        t0 = time.time()
        trainer.train()
        elapsed = time.time() - t0

    logger.info("--mode small PASSED in %.1fs", elapsed)
    return 0


# ------------------------------------------------------------------- live mode


def _live_check_vllm(base_url: str, model_id: str, timeout: float = 10.0) -> None:
    """Confirm the vLLM server is up and serving the expected model id.

    Cheap (~one HTTP round-trip) and catches the top failure modes before
    we spend a full rollout: server not yet warm, model id mismatch between
    launcher and eval flags, wrong port, LoRA adapter not loaded.
    """
    import requests

    url = base_url.rstrip("/") + "/models"
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
    except Exception as exc:
        raise SmokeTestFailure(
            f"vLLM /models endpoint unreachable at {url}: {exc}. "
            "Start it with bash rl_training/scripts/launch_vllm_server.sh."
        ) from exc

    payload = resp.json()
    served_ids = [m.get("id") for m in payload.get("data", [])]
    if model_id not in served_ids:
        raise SmokeTestFailure(
            f"vLLM at {url} is not serving model_id={model_id!r}. "
            f"Available ids: {served_ids}. If you're evaluating a LoRA, "
            "make sure SERVE_MODE=lora and LORA_NAME match --model."
        )
    logger.info("vLLM OK: serving %s at %s (ids=%s)", model_id, url, served_ids)


def _live_check_fhir(fhir_api_base: str,
                     snapshot_path: str | None = None,
                     timeout: float = 5.0) -> bool:
    """Confirm either the FHIR server or a snapshot is usable.

    Returns True if live FHIR is reachable; False if we should rely on a
    snapshot. Raises if neither is available.
    """
    import requests

    meta_url = fhir_api_base.rstrip("/") + "/metadata"
    try:
        resp = requests.get(meta_url, timeout=timeout,
                            headers={"Accept": "application/fhir+json"})
        if resp.status_code < 500:
            logger.info("FHIR OK: %s responded with %d", meta_url, resp.status_code)
            return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("FHIR live server unreachable (%s); will rely on snapshot", exc)

    if snapshot_path and Path(snapshot_path).exists():
        size = Path(snapshot_path).stat().st_size
        logger.info("FHIR snapshot present at %s (%.1f MB)", snapshot_path, size / 1e6)
        return False
    raise SmokeTestFailure(
        f"Neither FHIR server at {fhir_api_base} nor snapshot at "
        f"{snapshot_path} is available. Build the snapshot first with "
        "rl_training/scripts/build_fhir_snapshot.py."
    )


def _live_pick_task(tasks: list[dict], task_id: str | None) -> dict:
    if task_id:
        for t in tasks:
            if t.get("id") == task_id:
                return t
        raise SmokeTestFailure(f"--task-id={task_id!r} not found in benchmark tasks")
    # Prefer a task4 (time-windowed magnesium) if available — it exercises
    # the temporal-grounding reward signal end-to-end.
    preferred = [t for t in tasks if t.get("id", "").startswith("task4_")]
    pool = preferred or tasks
    return pool[0]


def run_live_mode(args: argparse.Namespace) -> int:
    """Cloud preflight: one full rollout of the 32B model against one real task.

    Intended to run on the same pod that will host training, right after
    ``launch_vllm_server.sh`` comes up but before ``train_grpo_32b.py``
    starts. Cost: ~$0.50-$1 (one rollout + a few /models pings).

    What it proves, in order:
        1. The vLLM server is up and serving the intended model id.
        2. Either the live FHIR server or the snapshot jsonl is usable.
        3. ``VLLMPolicy`` + ``MedAgentEnv`` + ``Evaluator._rollout`` compose
           without runtime errors on a real 32B chat completion.
        4. The rollout produces a well-formed trajectory with at least one
           assistant turn and a terminal state.
        5. The clinical reward functions can be computed on the resulting
           trajectory without raising.

    This is the last gate before we commit to the full ~$50 fine-tune run.
    If this fails, the next ~$50 of GPU time would be wasted.
    """
    import importlib.util

    missing = [
        mod for mod in ("yaml", "requests", "openai", "rich")
        if importlib.util.find_spec(mod) is None
    ]
    if missing:
        logger.error(
            "--mode live needs: %s. Install with: pip install -r requirements-gpu.txt",
            ", ".join(missing),
        )
        return 2

    import json

    import yaml

    # 1. Sanity-check the config and load it. Training configs
    #    (qwen3_32b_grpo.yaml) intentionally lack env.data_file, so fall
    #    back to the default eval config for the benchmark data path.
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        logger.error("Config not found: %s", cfg_path)
        return 2
    with open(cfg_path) as f:
        config = yaml.safe_load(f)

    if "func_file" not in config.get("env", {}):
        default_cfg_path = Path("rl_training/configs/default.yaml")
        if default_cfg_path.exists():
            with open(default_cfg_path) as f:
                default_cfg = yaml.safe_load(f)
            config.setdefault("env", {})
            for k, v in default_cfg["env"].items():
                config["env"].setdefault(k, v)

    data_file = args.data_file or config["env"].get("data_file")
    if not data_file:
        default_cfg_path = Path("rl_training/configs/default.yaml")
        if default_cfg_path.exists():
            with open(default_cfg_path) as f:
                data_file = yaml.safe_load(f)["env"]["data_file"]
    if not data_file or not Path(data_file).exists():
        logger.error(
            "Benchmark tasks not found (data_file=%s). Pass --data-file "
            "pointing at e.g. data/medagentbench/test_data_v2.json.",
            data_file,
        )
        return 2
    with open(data_file) as f:
        tasks = json.load(f)
    logger.info("Loaded %d benchmark tasks from %s", len(tasks), data_file)

    # 2. Ping vLLM.
    try:
        _live_check_vllm(args.vllm_base_url, args.model)
    except SmokeTestFailure as exc:
        logger.error("vLLM check failed: %s", exc)
        return 1

    # 3. Ping FHIR / confirm snapshot.
    snapshot_path = args.snapshot_path or config["env"].get("snapshot_path")
    try:
        _live_check_fhir(config["env"]["fhir_api_base"], snapshot_path)
    except SmokeTestFailure as exc:
        logger.error("FHIR check failed: %s", exc)
        return 1

    # 4. Rollout one task.
    from rl_training.agent.vllm_policy import VLLMPolicy
    from rl_training.env.medagent_env import MedAgentEnv
    from rl_training.evaluation.clinical_metrics import compute_clinical_metrics
    from rl_training.evaluation.evaluator import Evaluator

    os.environ.setdefault("VLLM_BASE_URL", args.vllm_base_url)
    policy = VLLMPolicy(
        model_id=args.model,
        base_url=args.vllm_base_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_parallel=1,
    )
    env = MedAgentEnv.from_config(config)
    task = _live_pick_task(tasks, args.task_id)
    logger.info("Rollout task: id=%s MRN=%s", task.get("id"), task.get("eval_MRN"))

    evaluator = Evaluator(env=env, benchmark_tasks=[task])

    t0 = time.time()
    try:
        traj = evaluator._rollout(policy, task)  # noqa: SLF001
    except Exception as exc:
        logger.exception("Rollout raised: %s", exc)
        return 1
    elapsed = time.time() - t0

    # 5. Validate trajectory shape.
    try:
        assistant_turns = [
            t for t in traj.turns if getattr(t, "role", None) in ("assistant", "agent")
        ]
        _assert(
            len(assistant_turns) >= 1,
            f"no assistant turns in trajectory (got {len(traj.turns)} total)",
        )
        _assert(
            traj.status in {"completed", "invalid_action", "limit_reached"},
            f"unexpected trajectory status: {traj.status!r}",
        )
        _assert(traj.num_steps >= 1, f"num_steps too low: {traj.num_steps}")
    except SmokeTestFailure as exc:
        logger.error("Trajectory shape check failed: %s", exc)
        return 1

    # 6. Clinical metrics must compute on the single trajectory without error.
    try:
        clinical = compute_clinical_metrics([traj])
    except Exception as exc:
        logger.exception("Clinical metric computation failed: %s", exc)
        return 1

    logger.info(
        "Rollout OK: task=%s correct=%s status=%s steps=%d elapsed=%.1fs",
        traj.task_id, traj.correct, traj.status, traj.num_steps, elapsed,
    )
    logger.info(
        "Clinical preflight: total=%d temporal_inconsistency_rate=%.3f "
        "over_deferral_rate=%.3f under_deferral_rate=%.3f",
        clinical.total,
        clinical.temporal_inconsistency_rate,
        clinical.over_deferral_rate,
        clinical.under_deferral_rate,
    )

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "live_smoke_trajectory.jsonl", "w") as f:
            f.write(traj.to_jsonl_line() + "\n")
        with open(out_dir / "live_smoke_summary.json", "w") as f:
            json.dump({
                "model": args.model,
                "vllm_base_url": args.vllm_base_url,
                "task_id": traj.task_id,
                "status": traj.status,
                "correct": traj.correct,
                "num_steps": traj.num_steps,
                "rollout_seconds": elapsed,
                "clinical": {
                    "total": clinical.total,
                    "temporal_inconsistency_rate": clinical.temporal_inconsistency_rate,
                    "over_deferral_rate": clinical.over_deferral_rate,
                    "under_deferral_rate": clinical.under_deferral_rate,
                    "evidence_omission_rate": clinical.evidence_omission_rate,
                },
            }, f, indent=2)
        logger.info("Wrote live smoke outputs to %s", out_dir)

    logger.info("--mode live PASSED. Safe to launch the full training run.")
    return 0


# ------------------------------------------------------------------------- main


def main() -> int:
    parser = argparse.ArgumentParser(description="Local smoke tests for the RL stack")
    parser.add_argument("--mode", required=True, choices=["unit", "small", "live"])
    parser.add_argument("--vllm-base-url", default="http://127.0.0.1:8000/v1",
                        help="(live mode) vLLM OpenAI-compatible endpoint")
    parser.add_argument("--model", default="Qwen/Qwen3-32B-Instruct",
                        help="(live mode) model id as reported by vLLM /models")
    parser.add_argument("--config", default="rl_training/configs/default.yaml",
                        help="(live mode) base env config (fhir_api_base + data_file)")
    parser.add_argument("--data-file", default=None,
                        help="(live mode) override benchmark tasks JSON path")
    parser.add_argument("--task-id", default=None,
                        help="(live mode) run this specific task id (default: first task4_*)")
    parser.add_argument("--snapshot-path", default=None,
                        help="(live mode) FHIR snapshot jsonl, used if live FHIR is down")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="(live mode) sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=2048,
                        help="(live mode) per-turn max tokens")
    parser.add_argument("--output-dir", default=None,
                        help="(live mode) optional dir to persist the smoke trajectory + summary")
    parser.add_argument("--small-model", default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="(small mode) tiny model to run the full TRL loop")
    parser.add_argument("--max-steps", type=int, default=2,
                        help="(small mode) GRPO steps")
    parser.add_argument("--num-tasks", type=int, default=2,
                        help="(small mode) training tasks to use")
    args = parser.parse_args()

    if args.mode == "unit":
        return run_unit_tests()
    if args.mode == "small":
        return run_small_mode(args)
    if args.mode == "live":
        return run_live_mode(args)
    parser.error(f"unknown mode {args.mode!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
