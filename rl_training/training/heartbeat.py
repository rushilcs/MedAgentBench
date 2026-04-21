"""Heartbeat callback for GRPO training.

Touches a heartbeat file on every training step so an external watchdog
can tell when the trainer has silently died. Also optionally POSTs a
one-line status to an ntfy.sh topic every N steps for push notifications.

The watchdog is separate (``rl_training/scripts/watchdog.sh``) - this
module only emits the signal.

Heartbeat contract:
  * File path: ``/tmp/trainer_heartbeat`` by default (config-overridable).
  * File content: a single line ``step=<N> t=<unix_ts>``.
  * Updated on every ``on_step_end`` and ``on_log``.

Billing protection: if the trainer dies, the heartbeat stops updating;
the watchdog sees the stale mtime and stops the pod, so you stop paying
for a dead GPU within ~15 minutes.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    from transformers import TrainerCallback
except ImportError:  # pragma: no cover
    class TrainerCallback:  # type: ignore[no-redef]
        pass


class HeartbeatCallback(TrainerCallback):
    """Emit a heartbeat signal on every training step.

    Args:
        heartbeat_path: file to touch each step.
        ntfy_topic: optional ntfy.sh topic to POST to. If ``None``, push
            notifications are disabled.
        ntfy_every: only POST to ntfy every N steps (notifications get
            noisy if sent every single step).
    """

    def __init__(
        self,
        heartbeat_path: str = "/tmp/trainer_heartbeat",
        ntfy_topic: str | None = None,
        ntfy_every: int = 20,
    ):
        self.heartbeat_path = heartbeat_path
        self.ntfy_topic = ntfy_topic or os.environ.get("NTFY_TOPIC")
        self.ntfy_every = max(1, ntfy_every)
        self._disabled = False

    def _touch(self, step: int) -> None:
        try:
            Path(self.heartbeat_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.heartbeat_path, "w") as f:
                f.write(f"step={step} t={time.time():.0f}\n")
        except Exception as exc:
            if not self._disabled:
                logger.warning("Heartbeat write failed (disabling): %s", exc)
                self._disabled = True

    def _ntfy(self, step: int, logs: dict[str, Any]) -> None:
        if not self.ntfy_topic:
            return
        try:
            import requests
            reward = logs.get("reward") or logs.get("reward/mean") or 0.0
            loss = logs.get("loss", 0.0)
            msg = (
                f"step={step} reward={reward:.3f} loss={loss:.3f} "
                f"t={int(time.time())}"
            )
            requests.post(
                f"https://ntfy.sh/{self.ntfy_topic}",
                data=msg.encode("utf-8"), timeout=5,
            )
        except Exception as exc:
            logger.debug("ntfy POST failed: %s", exc)

    # ---- TrainerCallback hooks

    def on_step_end(self, args, state, control, **kwargs: Any):  # noqa: D401
        if not self._disabled:
            self._touch(state.global_step)
        return control

    def on_log(self, args, state, control, **kwargs: Any):  # noqa: D401
        if not self._disabled:
            self._touch(state.global_step)
        logs = kwargs.get("logs") or {}
        if state.global_step % self.ntfy_every == 0:
            self._ntfy(state.global_step, logs)
        return control
