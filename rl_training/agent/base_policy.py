from __future__ import annotations

from abc import ABC, abstractmethod


class BasePolicy(ABC):
    """Abstract interface for a policy that maps conversation history to actions."""

    @abstractmethod
    def act(self, history: list[dict]) -> str:
        """Given the current conversation history, return the next action string."""

    def act_batch(self, histories: list[list[dict]]) -> list[str]:
        """Batch inference.  Default implementation calls ``act`` sequentially."""
        return [self.act(h) for h in histories]
