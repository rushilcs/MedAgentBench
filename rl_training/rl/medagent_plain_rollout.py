"""Plain-text multi-turn rollout for MedAgentEnv -> TRL GRPOTrainer.rollout_func.

TRL's ``environment_factory`` path only supports JSON tool-calls. The
MedAgentBench paper grades on the plain-text GET/POST/FINISH interface
(``rl_training/env/medagent_env.py``), so to be directly comparable we need to
run multi-turn plain-text rollouts ourselves and hand TRL the resulting
``prompt_ids / completion_ids / logprobs / env_mask`` tensors plus per-rollout
``extra_fields`` that the reward function consumes.

Contract (verified against ``trl/trainer/grpo_trainer.py`` main, _generate):
- ``rollout_func(prompts, trainer) -> dict`` with required keys
  ``prompt_ids``, ``completion_ids``, ``logprobs`` (each a list, one entry per
  prompt; entries are token-id / float lists).
- Optional ``env_mask`` (list of int lists, same shape as completion_ids):
  1 = model-generated token, 0 = env-injected token. TRL uses this as
  ``tool_mask`` to zero env tokens out of the policy-gradient loss.
- Any other keys land in ``extra_fields`` and are forwarded to reward funcs
  as kwargs (one value per generation, indexed by row in the batch).

We also stash a per-rollout ``tool_log`` / ``finish_result`` / ``correct`` /
``task_id`` so :func:`rl_training.rl.medagent_reward.benchmark_aligned_reward`
can score without needing the JSON-tool ``MedAgentBenchEnv`` instance.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import re
from typing import Any, Callable

import torch

from rl_training.env.action_parser import parse_action
from rl_training.env.medagent_env import MedAgentEnv

logger = logging.getLogger(__name__)


# Qwen chat template scaffolding tokens we need to re-emit between turns.
_ASSIST_END = "<|im_end|>"


def _prompt_hash(prompt: list[dict[str, str]]) -> str:
    """Stable hash of a chat prompt, used to look up the originating task.

    rollout_func only receives ``prompts`` (the chat messages), but we need
    the full task dict (``id``, ``sol``, etc.) for ``env.reset`` and
    refsol-aligned grading. We build a prompt-hash -> task lookup once when
    the trainer starts and consult it inside the rollout.
    """
    canon = json.dumps(prompt, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def _render_initial_prompt_ids(tokenizer, prompt: list[dict[str, str]]) -> list[int]:
    """Apply the chat template with the assistant generation prompt appended."""
    text = tokenizer.apply_chat_template(
        prompt,
        add_generation_prompt=True,
        tokenize=False,
    )
    return tokenizer(text, add_special_tokens=False).input_ids


def _render_env_reply_ids(tokenizer, reply_text: str) -> list[int]:
    """Tokenize the user-channel env reply with the assistant prompt re-opened.

    After the model emits ``<|im_end|>`` we append a user turn with the env
    reply, then immediately re-open the assistant turn so the next generate
    call continues in-character. We tokenize the literal scaffolding instead
    of re-applying the chat template each turn so positional alignment with
    our env_mask stays exact.
    """
    text = (
        f"\n<|im_start|>user\n{reply_text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return tokenizer(text, add_special_tokens=False).input_ids


def _ensure_assist_end(text: str) -> str:
    """Force ``<|im_end|>`` at the end of an assistant turn if missing."""
    return text if text.rstrip().endswith(_ASSIST_END) else text.rstrip() + _ASSIST_END


def _looks_like_action_complete(text: str) -> bool:
    """Cheap heuristic: stop generating early if we already have a full action.

    Qwen3 with thinking off usually emits ``<|im_end|>`` itself; this is a
    belt-and-suspenders short-circuit so a runaway completion can't burn the
    full ``max_completion_length`` after the action is already emitted.
    """
    if "FINISH(" in text:
        depth = 0
        for ch in text[text.index("FINISH(") + len("FINISH("):]:
            if ch == "(":
                depth += 1
            elif ch == ")":
                if depth == 0:
                    return True
                depth -= 1
        return False
    if "\n" in text and re.search(r"^(GET\s+\S|POST\s+\S)", text, re.MULTILINE):
        # GET on its own line, or POST followed by a JSON line that closes
        if text.lstrip().startswith("GET"):
            return "\n" in text.lstrip()
        if text.lstrip().startswith("POST"):
            try:
                body = text.split("\n", 1)[1]
                json.loads(body.strip())
                return True
            except Exception:
                return False
    return False


def _generate_turn(
    model,
    tokenizer,
    input_ids: list[int],
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: torch.device,
    eos_ids: list[int],
) -> tuple[list[int], list[float]]:
    """Run one assistant turn through HF generate, returning (ids, per-token logprobs).

    Logprobs are detached floats (the "old" sampling logprobs GRPO needs for
    the importance ratio); the policy gradient is recomputed by TRL on the
    forward pass with grad enabled.
    """
    ids = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
    attn = torch.ones_like(ids)
    with torch.inference_mode():
        out = model.generate(
            input_ids=ids,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=eos_ids,
            output_scores=True,
            return_dict_in_generate=True,
            use_cache=True,
        )
    seq = out.sequences[0]
    new_ids = seq[ids.shape[1]:].tolist()
    # ``out.scores`` is a tuple of (1, vocab) tensors, one per generated token.
    logprobs: list[float] = []
    for step, score in enumerate(out.scores):
        if step >= len(new_ids):
            break
        lp = torch.log_softmax(score[0].float(), dim=-1)[new_ids[step]].item()
        logprobs.append(lp)
    return new_ids, logprobs


def _decode_action_text(tokenizer, ids: list[int]) -> str:
    """Decode an assistant turn back to text for action parsing."""
    return tokenizer.decode(ids, skip_special_tokens=True)


def _build_task_lookup(dataset) -> dict[str, dict[str, Any]]:
    """Map prompt-hash -> task dict (parsed from ``ref_task_json`` column)."""
    lookup: dict[str, dict[str, Any]] = {}
    for row in dataset:
        prompt = row["prompt"]
        try:
            task = json.loads(row["ref_task_json"])
        except (KeyError, TypeError, json.JSONDecodeError):
            task = {
                "id": row.get("task_id", ""),
                "instruction": row.get("instruction", ""),
                "context": row.get("context", ""),
                "eval_MRN": row.get("eval_MRN", ""),
            }
        lookup[_prompt_hash(prompt)] = task
    return lookup


def _grade_with_refsol(env: MedAgentEnv) -> bool:
    """Wrap MedAgentEnv.grade() with broad exception trap.

    Refsol graders raise on shape mismatches we'd rather absorb as ``False``
    than crash the rollout. The trainer's snapshot patch already coerces
    dict->str so the most common TypeError path is gone, but stay defensive.
    """
    try:
        return bool(env.grade())
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("grade() exception: %s", exc)
        return False


def make_medagent_plain_rollout(
    *,
    tokenizer,
    env_factory: Callable[[], MedAgentEnv],
    max_rounds: int,
    max_completion_length: int,
    temperature: float,
    top_p: float,
    task_lookup: dict[str, dict[str, Any]],
    fhir_api_base: str,
):
    """Build a TRL-compatible ``rollout_func`` for plain-text MedAgent rollouts."""

    def rollout_func(prompts, trainer):
        # ``prompts`` arrives with each unique prompt repeated num_generations
        # times consecutively (TRL's RepeatSampler). We treat each row as an
        # independent rollout; sampling temperature provides the diversity
        # GRPO needs across the group.
        device = trainer.accelerator.device
        model = trainer.accelerator.unwrap_model(trainer.model)
        model.eval()

        eos_ids: list[int] = []
        for tok in (tokenizer.eos_token, _ASSIST_END):
            if tok is None:
                continue
            tid = tokenizer.convert_tokens_to_ids(tok)
            if isinstance(tid, int) and tid >= 0 and tid not in eos_ids:
                eos_ids.append(tid)

        prompt_ids_out: list[list[int]] = []
        completion_ids_out: list[list[int]] = []
        logprobs_out: list[list[float]] = []
        env_mask_out: list[list[int]] = []
        tool_logs: list[list[dict[str, Any]]] = []
        finish_results: list[str | None] = []
        corrects: list[bool] = []
        task_ids: list[str] = []
        ref_task_jsons: list[str] = []

        for prompt in prompts:
            # Look up the originating task by prompt-hash; copy so per-rollout
            # mutations (e.g. env._task) don't bleed across the batch.
            task = task_lookup.get(_prompt_hash(prompt))
            if task is None:
                logger.warning(
                    "rollout_func: no task lookup for prompt-hash; skipping with empty rollout",
                )
                # Emit a minimal-but-valid record so TRL shapes line up.
                p_ids = _render_initial_prompt_ids(tokenizer, prompt)
                prompt_ids_out.append(p_ids)
                completion_ids_out.append([tokenizer.eos_token_id or 0])
                logprobs_out.append([0.0])
                env_mask_out.append([1])
                tool_logs.append([])
                finish_results.append(None)
                corrects.append(False)
                task_ids.append("")
                ref_task_jsons.append("{}")
                continue
            task = copy.deepcopy(task)

            env = env_factory()
            env.reset(task)

            prompt_ids = _render_initial_prompt_ids(tokenizer, prompt)
            full_ids = list(prompt_ids)
            completion_ids: list[int] = []
            logprobs: list[float] = []
            env_mask: list[int] = []

            tool_log: list[dict[str, Any]] = []
            finish_result: str | None = None
            step_count = 0

            for _ in range(max_rounds):
                budget = max_completion_length - len(completion_ids)
                if budget <= 0:
                    break

                new_ids, new_lps = _generate_turn(
                    model,
                    tokenizer,
                    full_ids,
                    max_new_tokens=min(budget, max_completion_length),
                    temperature=temperature,
                    top_p=top_p,
                    device=device,
                    eos_ids=eos_ids,
                )
                if not new_ids:
                    break

                full_ids.extend(new_ids)
                completion_ids.extend(new_ids)
                logprobs.extend(new_lps)
                env_mask.extend([1] * len(new_ids))

                assistant_text = _decode_action_text(tokenizer, new_ids)
                action = parse_action(assistant_text)
                step_count += 1

                if action.kind == "get":
                    step_result = env.step(f"GET {action.url}")
                    reply = step_result.state.history[-1]["content"]
                    tool_log.append({
                        "step": step_count,
                        "action": "GET",
                        "url": action.url,
                        "success": not reply.startswith("Error"),
                        "response_len": len(reply),
                        "timestamps": [],
                    })
                elif action.kind == "post":
                    payload_str = (
                        json.dumps(action.payload)
                        if action.payload is not None else ""
                    )
                    step_result = env.step(
                        f"POST {action.url}\n{payload_str}",
                    )
                    reply = step_result.state.history[-1]["content"]
                    success = not reply.startswith("Invalid") and not reply.startswith("Error")
                    tool_log.append({
                        "step": step_count,
                        "action": "POST",
                        "url": action.url,
                        "success": success,
                        "response_len": len(payload_str),
                        "payload": payload_str,
                        "timestamps": [],
                    })
                elif action.kind == "finish":
                    env.step(f"FINISH({action.result})")
                    finish_result = action.result
                    tool_log.append({
                        "step": step_count,
                        "action": "FINISH",
                        "url": "",
                        "success": True,
                        "response_len": len(action.result or ""),
                        "answers": action.result,
                        "timestamps": [],
                    })
                    break
                else:
                    # Invalid: tell the env via a no-op step so its done flag
                    # flips, then break out of the loop. The reward path will
                    # see r_first_invalid (or similar) for the failed parse.
                    env.step(assistant_text)
                    break

                # Append env reply (user turn) plus the re-opened assistant
                # prompt so the next generate continues in-character. These
                # tokens are env-injected: logprob=0, env_mask=0.
                reply_ids = _render_env_reply_ids(tokenizer, reply)
                full_ids.extend(reply_ids)
                completion_ids.extend(reply_ids)
                logprobs.extend([0.0] * len(reply_ids))
                env_mask.extend([0] * len(reply_ids))

            # If the rollout produced literally zero completion tokens (degenerate
            # generate), pad with a single EOS so TRL doesn't choke on empty
            # tensors during the pad/forward pass.
            if not completion_ids:
                eos = tokenizer.eos_token_id or 0
                completion_ids = [eos]
                logprobs = [0.0]
                env_mask = [1]

            correct = _grade_with_refsol(env)

            prompt_ids_out.append(prompt_ids)
            completion_ids_out.append(completion_ids)
            logprobs_out.append(logprobs)
            env_mask_out.append(env_mask)
            tool_logs.append(tool_log)
            finish_results.append(finish_result)
            corrects.append(bool(correct))
            task_ids.append(str(task.get("id", "")))
            ref_task_jsons.append(json.dumps(task, ensure_ascii=False))

        model.train()  # restore training mode for the optimizer step that follows

        # Lift batch-level avg_correct so the trainer's progress callback can
        # surface it. TRL forwards reward/extra_field metrics via reward funcs;
        # this kwarg is also visible in the rollout itself for any caller that
        # wants to log it directly.
        try:
            avg_correct = sum(int(c) for c in corrects) / max(1, len(corrects))
            logger.info(
                "rollout_func: batch=%d avg_correct=%.3f finish_rate=%.3f mean_steps=%.2f",
                len(corrects),
                avg_correct,
                sum(1 for fr in finish_results if fr is not None) / max(1, len(finish_results)),
                sum(len(t) for t in tool_logs) / max(1, len(tool_logs)),
            )
        except Exception:  # pragma: no cover - logging only
            pass

        return {
            "prompt_ids": prompt_ids_out,
            "completion_ids": completion_ids_out,
            "logprobs": logprobs_out,
            "env_mask": env_mask_out,
            # Extra fields for benchmark_aligned_reward (refactored to read
            # these instead of environments[i]._tool_log).
            "rollout_tool_log": tool_logs,
            "rollout_finish_result": finish_results,
            "rollout_correct": corrects,
            "rollout_task_id": task_ids,
            "rollout_ref_task_json": ref_task_jsons,
            "rollout_fhir_api_base": [fhir_api_base] * len(prompts),
        }

    return rollout_func


__all__ = ["make_medagent_plain_rollout", "_build_task_lookup", "_prompt_hash"]
