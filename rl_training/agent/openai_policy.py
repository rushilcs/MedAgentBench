from __future__ import annotations

import os
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

from .base_policy import BasePolicy

logger = logging.getLogger(__name__)


_REASONING_PREFIXES = ("o1", "o3", "o4")


def _is_reasoning_model(model_id: str) -> bool:
    """Return True for o1/o3/o4 families (base or fine-tuned)."""
    name = model_id
    if name.startswith("ft:"):
        parts = name.split(":")
        if len(parts) >= 2:
            name = parts[1]
    return any(name.startswith(p) for p in _REASONING_PREFIXES)


class OpenAIPolicy(BasePolicy):
    """Policy backed by an OpenAI Chat Completions model (base or fine-tuned).

    Auto-detects reasoning models (o1/o3/o4, incl. ``ft:o4-mini-*``) and swaps
    the request shape accordingly: reasoning models use ``max_completion_tokens``
    and ``reasoning_effort`` and reject ``temperature`` other than 1.
    """

    def __init__(
        self,
        model_id: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 2048,
        api_key: str | None = None,
        max_retries: int = 8,
        retry_delay: float = 2.0,
        max_parallel: int = 5,
        reasoning_effort: str = "medium",
    ):
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.max_parallel = max_parallel
        self.reasoning_effort = reasoning_effort
        self._reasoning = _is_reasoning_model(model_id)

        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            max_retries=3,
            timeout=120.0 if self._reasoning else 60.0,
        )

    def _to_openai_messages(self, history: list[dict]) -> list[dict]:
        """Map MedAgentBench history (role=agent) to OpenAI format (role=assistant)."""
        messages: list[dict] = []
        for msg in history:
            role = "assistant" if msg["role"] in ("agent", "assistant") else "user"
            messages.append({"role": role, "content": msg["content"]})
        return messages

    def _build_params(self, messages: list[dict]) -> dict:
        params: dict = {"model": self.model_id, "messages": messages}
        if self._reasoning:
            params["max_completion_tokens"] = self.max_tokens
            params["reasoning_effort"] = self.reasoning_effort
        else:
            params["temperature"] = self.temperature
            params["max_tokens"] = self.max_tokens
        return params

    def act(self, history: list[dict]) -> str:
        messages = self._to_openai_messages(history)
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(**self._build_params(messages))
                return response.choices[0].message.content or ""
            except Exception as exc:
                wait = min(self.retry_delay * (2 ** attempt), 60.0)
                logger.warning(
                    "OpenAI API error (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1, self.max_retries, wait, exc,
                )
                if attempt < self.max_retries - 1:
                    time.sleep(wait)
                else:
                    raise

    def act_batch(self, histories: list[list[dict]]) -> list[str]:
        """Parallel inference using a thread pool."""
        results: list[str | None] = [None] * len(histories)
        with ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
            future_to_idx = {executor.submit(self.act, h): i for i, h in enumerate(histories)}
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    logger.error("Batch inference failed for index %d: %s", idx, exc)
                    results[idx] = ""
        return [r or "" for r in results]
