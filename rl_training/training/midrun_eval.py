"""Mid-train validation eval + best-checkpoint tracking (plan §0.3).

The callback fires on ``on_step_end`` when ``state.global_step`` is a
positive multiple of ``every_steps``. It:

  1. Runs greedy plain-text rollouts of the current trainer model over a
     held-out validation set (``rl_training/data/validation_tasks_v2.json``
     by default — never ``test_data_v2.json``).
  2. Writes ``<output_dir>/midrun_step{N}.json`` with overall + per-task
     validation SR (``CloudSyncCallback.also_sync`` then pushes the file
     to B2).
  3. Tracks the best mid-eval val SR seen so far. When the current step
     beats the best, snapshots the live LoRA adapter to
     ``<output_dir>/best_adapter/`` and writes ``best_step.json``.
  4. Aborts (raises ``RuntimeError``) when val SR drops more than
     ``abort_on_regression_pp`` between consecutive evals — the best
     adapter is preserved on disk so Phase 2 can still merge it.

Eval is in-process (no subprocess + no extra vLLM server) so it shares
the trainer's GPUs with no contention. Greedy + ``max_new_tokens=2048``
+ ``max_rounds=8`` keeps each eval to ~1-3 min per 30-task batch.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    from transformers import TrainerCallback
except ImportError:  # pragma: no cover
    class TrainerCallback:  # type: ignore[no-redef]
        pass


def _load_val_tasks(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        logger.warning("Validation tasks file missing: %s", p)
        return []
    return json.loads(p.read_text())


def _greedy_rollout(
    model: Any,
    tokenizer: Any,
    env: Any,
    task: dict[str, Any],
    *,
    max_rounds: int,
    max_new_tokens: int,
    enable_thinking: bool,
) -> bool:
    """One greedy plain-text rollout. Returns refsol pass."""
    import torch

    state = env.reset(task)
    for _ in range(max_rounds):
        if state.done:
            break
        # Render prompt as chat. The env uses ``user``/``agent`` roles;
        # convert ``agent`` -> ``assistant`` so the chat template renders
        # the assistant turn correctly.
        messages = []
        for h in state.history:
            role = h.get("role", "user")
            if role == "agent":
                role = "assistant"
            messages.append({"role": role, "content": h.get("content", "")})
        try:
            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                enable_thinking=enable_thinking,
            )
        except TypeError:
            # Some tokenizer.apply_chat_template signatures don't accept
            # enable_thinking; the trainer-side patch sets it as a default
            # so this path is the safe fallback.
            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        if input_ids is None:
            return False
        device = next(model.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        new_tokens = out[0, input_ids.shape[-1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        result = env.step(text)
        state = result.state
        if state.done:
            break
    return env.grade()


def run_validation_eval(
    *,
    model: Any,
    tokenizer: Any,
    val_tasks: list[dict[str, Any]],
    fhir_api_base: str,
    funcs: list[dict[str, Any]],
    max_rounds: int = 8,
    max_new_tokens: int = 2048,
    enable_thinking: bool = False,
) -> dict[str, Any]:
    """Run greedy plain-text rollouts over ``val_tasks``. Returns metrics dict."""
    from rl_training.env.medagent_env import MedAgentEnv
    from rl_training.rl.verifiers.task_masks import task_type_from_id

    per_task: dict[str, list[int]] = {}
    correct = 0
    total = 0
    for task in val_tasks:
        env = MedAgentEnv(
            fhir_api_base=fhir_api_base,
            funcs=funcs,
            max_rounds=max_rounds,
        )
        try:
            ok = _greedy_rollout(
                model, tokenizer, env, task,
                max_rounds=max_rounds,
                max_new_tokens=max_new_tokens,
                enable_thinking=enable_thinking,
            )
        except Exception as exc:
            logger.warning("midrun rollout failed for %s: %s", task.get("id"), exc)
            ok = False
        total += 1
        if ok:
            correct += 1
        tt = task_type_from_id(task.get("id", ""))
        if tt is not None:
            fam = f"task{tt}"
            per_task.setdefault(fam, []).append(1 if ok else 0)
    sr = (correct / total) if total else 0.0
    per_task_sr = {
        f: (sum(v) / len(v) if v else 0.0) for f, v in per_task.items()
    }
    return {
        "overall_sr": sr,
        "correct": correct,
        "total": total,
        "per_task_sr": per_task_sr,
    }


class MidrunValidationCallback(TrainerCallback):
    """In-process val-set eval + best-adapter tracking + 5pp-regression abort."""

    def __init__(
        self,
        *,
        output_dir: str,
        validation_tasks_path: str,
        every_steps: int,
        fhir_api_base: str,
        func_file: str,
        max_rounds: int = 8,
        max_new_tokens: int = 2048,
        enable_thinking: bool = False,
        abort_on_regression_pp: float = 5.0,
    ) -> None:
        self.output_dir = output_dir
        self.validation_tasks_path = validation_tasks_path
        self.every_steps = max(1, int(every_steps))
        self.fhir_api_base = fhir_api_base
        self.func_file = func_file
        self.max_rounds = max_rounds
        self.max_new_tokens = max_new_tokens
        self.enable_thinking = enable_thinking
        self.abort_on_regression_pp = float(abort_on_regression_pp)

        self._val_tasks = _load_val_tasks(validation_tasks_path)
        self._funcs: list[dict[str, Any]] | None = None
        try:
            with open(func_file) as fh:
                self._funcs = json.load(fh)
        except OSError as exc:
            logger.warning("midrun callback: failed to load funcs: %s", exc)

        self._best_sr: float | None = None
        self._best_step: int | None = None
        self._prev_sr: float | None = None
        self._disabled = not self._val_tasks or self._funcs is None
        if self._disabled:
            logger.warning(
                "MidrunValidationCallback disabled (val_tasks=%d, funcs_loaded=%s)",
                len(self._val_tasks), self._funcs is not None,
            )

    # ------------------------------------------------------------------
    # TrainerCallback hooks

    def on_step_end(self, args, state, control, **kwargs: Any):  # noqa: D401
        if self._disabled:
            return control
        step = int(getattr(state, "global_step", 0))
        if step <= 0 or step % self.every_steps != 0:
            return control
        model = kwargs.get("model")
        tokenizer = kwargs.get("processing_class") or kwargs.get("tokenizer")
        if model is None or tokenizer is None:
            logger.debug("midrun: model/tokenizer missing in callback kwargs")
            return control
        logger.info("midrun eval @ step %d on %d val tasks", step, len(self._val_tasks))
        was_training = model.training
        try:
            model.eval()
            metrics = run_validation_eval(
                model=model,
                tokenizer=tokenizer,
                val_tasks=self._val_tasks,
                fhir_api_base=self.fhir_api_base,
                funcs=self._funcs or [],
                max_rounds=self.max_rounds,
                max_new_tokens=self.max_new_tokens,
                enable_thinking=self.enable_thinking,
            )
        except Exception as exc:
            logger.warning("midrun eval @ step %d failed: %s", step, exc)
            return control
        finally:
            if was_training:
                model.train()

        sr = float(metrics["overall_sr"])
        metrics["step"] = step
        out_path = Path(self.output_dir) / f"midrun_step{step}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            out_path.write_text(json.dumps(metrics, indent=2))
        except OSError as exc:
            logger.warning("midrun: failed to write %s: %s", out_path, exc)
        logger.info(
            "midrun @ step %d: SR=%.3f (%d/%d) per_task=%s",
            step, sr, metrics["correct"], metrics["total"], metrics["per_task_sr"],
        )

        # Best-checkpoint tracking
        if self._best_sr is None or sr > self._best_sr:
            self._best_sr = sr
            self._best_step = step
            self._snapshot_best_adapter(model, step, sr, metrics["per_task_sr"])

        # Regression abort
        if (
            self._prev_sr is not None
            and (self._prev_sr - sr) * 100.0 > self.abort_on_regression_pp
        ):
            logger.error(
                "midrun ABORT @ step %d: SR=%.3f vs prev=%.3f (drop > %.1f pp). "
                "best_adapter/ at step %s (SR=%.3f) preserved on disk for Phase 2 merge.",
                step, sr, self._prev_sr, self.abort_on_regression_pp,
                self._best_step, self._best_sr or 0.0,
            )
            raise RuntimeError(
                f"midrun regression abort @ step {step}: "
                f"val SR {sr:.3f} dropped > {self.abort_on_regression_pp:.1f}pp "
                f"vs previous {self._prev_sr:.3f}. best_adapter/ preserved."
            )
        self._prev_sr = sr
        return control

    def _snapshot_best_adapter(
        self,
        model: Any,
        step: int,
        sr: float,
        per_task_sr: dict[str, float],
    ) -> None:
        best_dir = Path(self.output_dir) / "best_adapter"
        try:
            if best_dir.exists():
                shutil.rmtree(best_dir)
            best_dir.mkdir(parents=True, exist_ok=True)
            # PeftModel exposes save_pretrained with adapter weights only.
            model.save_pretrained(str(best_dir))
            (Path(self.output_dir) / "best_step.json").write_text(
                json.dumps(
                    {"step": step, "val_sr": sr, "val_per_task_sr": per_task_sr},
                    indent=2,
                )
            )
            logger.info("midrun: snapshotted best adapter @ step %d (SR=%.3f)", step, sr)
        except Exception as exc:
            logger.warning("midrun: best_adapter snapshot failed: %s", exc)
