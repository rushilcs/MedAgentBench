from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal


@dataclass
class Turn:
    role: Literal["user", "assistant"]
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class Trajectory:
    task_id: str
    task_data: dict[str, Any]
    turns: list[Turn]
    reward: float = 0.0
    step_rewards: list[float] = field(default_factory=list)
    correct: bool = False
    status: str = "running"
    num_steps: int = 0
    model_id: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    def to_openai_messages(self) -> list[dict[str, str]]:
        """Convert to the OpenAI fine-tuning chat format.

        The MedAgentBench environment uses ``role="agent"`` in its history,
        but OpenAI expects ``role="assistant"``.
        """
        messages: list[dict[str, str]] = []
        for turn in self.turns:
            role = "assistant" if turn.role in ("agent", "assistant") else "user"
            messages.append({"role": role, "content": turn.content})
        return messages

    def to_openai_jsonl_line(self) -> str:
        """Serialize as a single JSONL line suitable for OpenAI fine-tuning."""
        return json.dumps({"messages": self.to_openai_messages()})

    # ------------------------------------------------------------------
    # Full serialization (for our trajectory store)
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_data": self.task_data,
            "turns": [t.to_dict() for t in self.turns],
            "reward": self.reward,
            "step_rewards": self.step_rewards,
            "correct": self.correct,
            "status": self.status,
            "num_steps": self.num_steps,
            "model_id": self.model_id,
            "timestamp": self.timestamp,
        }

    def to_jsonl_line(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Trajectory":
        turns = [Turn(role=t["role"], content=t["content"]) for t in data["turns"]]
        return cls(
            task_id=data["task_id"],
            task_data=data["task_data"],
            turns=turns,
            reward=data.get("reward", 0.0),
            step_rewards=data.get("step_rewards", []),
            correct=data.get("correct", False),
            status=data.get("status", "running"),
            num_steps=data.get("num_steps", 0),
            model_id=data.get("model_id", ""),
            timestamp=data.get("timestamp", ""),
        )

    @classmethod
    def from_env_history(
        cls,
        task: dict[str, Any],
        history: list[dict[str, str]],
        *,
        correct: bool,
        status: str,
        reward: float = 0.0,
        step_rewards: list[float] | None = None,
        model_id: str = "",
    ) -> "Trajectory":
        """Construct a Trajectory from a raw environment history."""
        turns = [Turn(role=h["role"], content=h["content"]) for h in history]
        num_steps = sum(1 for t in turns if t.role in ("agent", "assistant"))
        return cls(
            task_id=task["id"],
            task_data=task,
            turns=turns,
            reward=reward,
            step_rewards=step_rewards or [],
            correct=correct,
            status=status,
            num_steps=num_steps,
            model_id=model_id,
        )
