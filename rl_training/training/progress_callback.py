"""Rich progress callback for TRL training.

Adds a live progress bar to the training loop that shows, at a glance:

  * step / max_steps + percentage + ETA
  * rolling mean reward (last 20 steps)
  * rolling mean correctness (last 20 steps, if the ``correctness_reward``
    is among the logged rewards)
  * tokens/sec (very rough - derived from elapsed + ``train_tokens_per_second``
    when TRL reports it)

It also writes a structured JSONL row to ``<output_dir>/progress.jsonl`` on
every log event. The ``CloudSyncCallback`` rsyncs that file on each save so
you can ``tail -f`` it from your laptop.

Both effects are optional:

  * Progress bar falls back to a logger line if ``rich`` isn't importable
    or the current TTY can't render it (e.g. tmux pane piped to nohup).
  * JSONL writer silently skips if ``output_dir`` doesn't exist.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import deque
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    from transformers import TrainerCallback
except ImportError:  # pragma: no cover
    class TrainerCallback:  # type: ignore[no-redef]
        pass

try:
    from rich.progress import (
        BarColumn, MofNCompleteColumn, Progress, SpinnerColumn,
        TextColumn, TimeElapsedColumn, TimeRemainingColumn,
    )
    _HAS_RICH = True
except ImportError:  # pragma: no cover
    _HAS_RICH = False


class ProgressCallback(TrainerCallback):
    """Rich progress bar + structured JSONL log for GRPO training.

    Expected log fields (best-effort; missing ones are ignored):
      * ``loss``, ``reward``, ``reward_correctness_reward``,
        ``kl``, ``learning_rate``
    TRL's GRPOTrainer emits these on every ``logging_steps`` step.
    """

    def __init__(
        self,
        output_dir: str,
        max_steps: int,
        rolling_window: int = 20,
    ):
        self.output_dir = output_dir
        self.max_steps = max_steps
        self.rolling_window = rolling_window
        self._progress: Any | None = None
        self._task_id: Any | None = None
        self._rewards: deque = deque(maxlen=rolling_window)
        self._correct: deque = deque(maxlen=rolling_window)
        self._start_time: float = 0.0
        self._jsonl_path: str = os.path.join(output_dir, "progress.jsonl")

    # ---- TrainerCallback hooks

    def on_train_begin(self, args, state, control, **kwargs: Any):  # noqa: D401
        self._start_time = time.time()
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        if _HAS_RICH:
            try:
                self._progress = Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    MofNCompleteColumn(),
                    TextColumn("  reward={task.fields[reward]:.3f}  "
                               "SR~{task.fields[sr]:.1%}"),
                    TimeElapsedColumn(),
                    TextColumn("ETA"),
                    TimeRemainingColumn(),
                )
                self._progress.start()
                self._task_id = self._progress.add_task(
                    "GRPO training", total=self.max_steps,
                    reward=0.0, sr=0.0,
                )
            except Exception as exc:
                logger.warning("Rich progress unavailable, falling back: %s", exc)
                self._progress = None
        return control

    def on_train_end(self, args, state, control, **kwargs: Any):  # noqa: D401
        if self._progress is not None:
            try:
                self._progress.stop()
            except Exception:
                pass
            self._progress = None
        return control

    def on_log(self, args, state, control, **kwargs: Any):  # noqa: D401
        logs = kwargs.get("logs") or {}
        reward = self._pick(logs, [
            "reward", "reward/mean", "train_reward", "reward_mean",
        ])
        correctness = self._pick(logs, [
            "reward_correctness_reward", "reward/correctness",
            "reward_correctness",
        ])
        bench = self._pick(logs, ["reward_benchmark_aligned_reward"])
        if correctness is None and bench is not None:
            # Terminal refsol success is +r_succ (~10); use as coarse SR~ for the bar.
            correctness = min(1.0, max(0.0, bench / 10.0))
        if reward is not None:
            self._rewards.append(reward)
        if correctness is not None:
            self._correct.append(correctness)

        avg_reward = (
            sum(self._rewards) / len(self._rewards) if self._rewards else 0.0
        )
        avg_correct = (
            sum(self._correct) / len(self._correct) if self._correct else 0.0
        )

        # Update progress bar
        if self._progress is not None and self._task_id is not None:
            try:
                self._progress.update(
                    self._task_id,
                    completed=state.global_step,
                    reward=avg_reward,
                    sr=avg_correct,
                )
            except Exception:
                pass
        else:
            elapsed = time.time() - self._start_time
            pct = 100.0 * state.global_step / max(1, self.max_steps)
            logger.info(
                "step=%d/%d (%.1f%%) reward=%.3f SR~%.1f%% elapsed=%.0fs",
                state.global_step, self.max_steps, pct,
                avg_reward, 100.0 * avg_correct, elapsed,
            )

        # Write JSONL row
        try:
            row = {
                "t": time.time(),
                "step": state.global_step,
                "elapsed_s": time.time() - self._start_time,
                "avg_reward": avg_reward,
                "avg_correct": avg_correct,
            }
            for key in (
                "loss", "learning_rate", "kl",
                "reward", "reward_correctness_reward",
                "reward_benchmark_aligned_reward",
                "reward_temporal_grounding_reward",
                "reward_risk_calibrated_deferral_reward",
                "reward_decision_density_reward",
                "train_tokens_per_second",
            ):
                if key in logs:
                    row[key] = logs[key]
            with open(self._jsonl_path, "a") as f:
                f.write(json.dumps(row) + "\n")
        except Exception as exc:
            logger.debug("progress.jsonl write failed: %s", exc)
        return control

    @staticmethod
    def _pick(logs: dict[str, Any], candidates: list[str]) -> float | None:
        for k in candidates:
            if k in logs:
                try:
                    return float(logs[k])
                except (TypeError, ValueError):
                    continue
        return None
