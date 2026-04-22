"""Convert MedAgentBench tasks into a HuggingFace Dataset for TRL training.

Produces a Dataset with columns:
  - prompt:      list[dict]  (chat messages for the user turn)
  - task_id:     str
  - eval_MRN:    str
  - instruction: str
  - context:     str
  - sol:         Optional list (for task1)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from datasets import Dataset

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


_SYSTEM_PROMPT = (
    "You are an expert in using FHIR functions to assist medical professionals. "
    "You are given a question and a set of possible functions. Based on the "
    "question, you will need to make one or more function/tool calls to achieve "
    "the purpose.\n\n"
    "1. If you decide to invoke a GET function, you MUST put it in the format of\n"
    "GET url?param_name1=param_value1&param_name2=param_value2...\n\n"
    "2. If you decide to invoke a POST function, you MUST put it in the format of\n"
    "POST url\n[your payload data in JSON format]\n\n"
    "3. If you have got answers for all the questions and finished all the "
    "requested tasks, you MUST call to finish the conversation in the format of "
    "(make sure the list is JSON loadable.)\n"
    "FINISH([answer1, answer2, ...])\n\n"
    "Your response must be in the format of one of the three cases, and you can "
    "call only one function each time. You SHOULD NOT include any other text in "
    "the response."
)


def task_to_prompt(task: dict[str, Any], fhir_api_base: str) -> list[dict[str, str]]:
    """Build the chat-message prompt for one task."""
    user_content = f"Context: {task['context']}\n\nQuestion: {task['instruction']}"
    if fhir_api_base:
        user_content += f"\n\nNote: The FHIR server base URL is {fhir_api_base}"
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def tasks_to_dataset(
    tasks: list[dict[str, Any]],
    fhir_api_base: str = "http://localhost:8080/fhir/",
) -> Dataset:
    """Convert a list of task dicts to a HuggingFace Dataset."""
    records: dict[str, list] = {
        "prompt": [],
        "task_id": [],
        "eval_MRN": [],
        "instruction": [],
        "context": [],
        # Full task JSON for refsol-aligned rewards (sol, etc.). TRL passes this
        # into ``MedAgentBenchEnv.reset`` as a kwarg when present.
        "ref_task_json": [],
    }
    for task in tasks:
        records["prompt"].append(task_to_prompt(task, fhir_api_base))
        records["task_id"].append(task["id"])
        records["eval_MRN"].append(task.get("eval_MRN", ""))
        records["instruction"].append(task["instruction"])
        records["context"].append(task["context"])
        records["ref_task_json"].append(json.dumps(task, ensure_ascii=False))
    return Dataset.from_dict(records)


def load_benchmark_dataset(
    data_path: str = "data/medagentbench/test_data_v2.json",
    fhir_api_base: str = "http://localhost:8080/fhir/",
) -> Dataset:
    """Load the 300 benchmark tasks as a HF Dataset."""
    with open(data_path) as f:
        tasks = json.load(f)
    return tasks_to_dataset(tasks, fhir_api_base)


def load_training_dataset(
    training_tasks_path: str,
    fhir_api_base: str = "http://localhost:8080/fhir/",
) -> Dataset:
    """Load pre-generated training tasks as a HF Dataset."""
    with open(training_tasks_path) as f:
        tasks = json.load(f)
    return tasks_to_dataset(tasks, fhir_api_base)


def expert_trajectories_to_sft_dataset(
    trajectories_path: str,
) -> Dataset:
    """Convert expert trajectories (JSONL) to SFT chat format.

    Each trajectory becomes one training example with the full
    multi-turn conversation in OpenAI messages format.
    """
    from rl_training.data.trajectory_store import TrajectoryStore

    store = TrajectoryStore(trajectories_path)
    trajs = store.filter(correct=True)

    records: dict[str, list] = {"messages": []}
    for traj in trajs:
        messages = traj.to_openai_messages()
        if messages:
            records["messages"].append(messages)

    return Dataset.from_dict(records)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare datasets for TRL training")
    parser.add_argument("--data-file", default="data/medagentbench/test_data_v2.json")
    parser.add_argument("--fhir-base", default="http://localhost:8080/fhir/")
    parser.add_argument("--output", default="rl_training/outputs/benchmark_dataset")
    args = parser.parse_args()

    ds = load_benchmark_dataset(args.data_file, args.fhir_base)
    ds.save_to_disk(args.output)
    print(f"Saved {len(ds)} examples to {args.output}")
