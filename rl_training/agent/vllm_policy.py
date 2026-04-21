"""Policy that talks to a vLLM OpenAI-compatible server.

Used to evaluate a Qwen3-32B (or any vLLM-served model) with the existing
``Evaluator`` class, which until now only knew about ``OpenAIPolicy``.

vLLM exposes an OpenAI-compatible ``/v1/chat/completions`` endpoint. We
reuse the ``openai`` Python client with ``base_url`` pointed at the local
server, so the authentication + retry flow is identical to ``OpenAIPolicy``.
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

from .base_policy import BasePolicy

logger = logging.getLogger(__name__)


class VLLMPolicy(BasePolicy):
    """Policy backed by a vLLM server's OpenAI-compatible endpoint.

    Args:
        model_id: the ``--model`` arg passed to vLLM at launch. vLLM routes
            chat-completions requests for this exact id.
        base_url: the vLLM server's OpenAI endpoint
            (``http://127.0.0.1:8000/v1`` by default).
        api_key: vLLM accepts any non-empty string as the key unless you
            enabled ``--api-key``. Defaults to ``EMPTY``.
        temperature / max_tokens / max_retries / retry_delay / max_parallel:
            standard generation + retry knobs.
    """

    def __init__(
        self,
        model_id: str,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        max_retries: int = 8,
        retry_delay: float = 2.0,
        max_parallel: int = 8,
        extra_body: dict | None = None,
    ):
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.max_parallel = max_parallel
        self.extra_body = extra_body or {}

        self.client = OpenAI(
            api_key=api_key or os.environ.get("VLLM_API_KEY") or "EMPTY",
            base_url=base_url or os.environ.get(
                "VLLM_BASE_URL", "http://127.0.0.1:8000/v1"
            ),
            max_retries=3,
            timeout=300.0,
        )

    def _to_messages(self, history: list[dict]) -> list[dict]:
        """Map MedAgentBench history (role=agent) to OpenAI format (role=assistant)."""
        messages: list[dict] = []
        for msg in history:
            role = "assistant" if msg["role"] in ("agent", "assistant") else "user"
            messages.append({"role": role, "content": msg["content"]})
        return messages

    def act(self, history: list[dict]) -> str:
        messages = self._to_messages(history)
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    extra_body=self.extra_body or None,
                )
                return response.choices[0].message.content or ""
            except Exception as exc:
                wait = min(self.retry_delay * (2 ** attempt), 60.0)
                logger.warning(
                    "vLLM error (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1, self.max_retries, wait, exc,
                )
                if attempt < self.max_retries - 1:
                    time.sleep(wait)
                else:
                    raise

    def act_batch(self, histories: list[list[dict]]) -> list[str]:
        """Parallel inference using a thread pool.

        vLLM internally batches concurrent HTTP requests so throughput
        scales well up to ``max_num_seqs`` configured on the server.
        """
        results: list[str | None] = [None] * len(histories)
        with ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
            future_to_idx = {
                executor.submit(self.act, h): i for i, h in enumerate(histories)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    logger.error("Batch inference failed for index %d: %s", idx, exc)
                    results[idx] = ""
        return [r or "" for r in results]
