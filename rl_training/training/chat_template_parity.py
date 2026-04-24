"""Chat-template parity fingerprint and assertion.

A historical class of bugs in this repo: SFT trains with one chat template (or
``apply_chat_template`` kwargs), eval serves with another, and the per-task SR
silently moves -- usually because vLLM defaults ``enable_thinking=True`` while
SFT defaults it to ``False``. The "tokenizer overwrite no-op" note in
``rl_training/outputs/sft_v2_article/run_metadata.json`` only proved that
``chat_template.jinja`` was byte-identical between two paths, not that the
*kwargs* applied at format-time were identical.

This module gives both ends (SFT trainer, vLLM eval) a single small artifact:

  ``chat_template_fingerprint.json`` -- {md5_of_template, kwargs_used,
  rendered_sample_md5}

SFT writes it next to the merged checkpoint; eval reads it, recomputes the
fingerprint against the same sample, and refuses to start if they differ.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

# Canonical sample messages used for the rendered-sample fingerprint. Kept
# tiny and stable so the hash isn't influenced by trajectory drift in the
# corpus. The shape mirrors what every MedAgentBench task looks like at SFT
# and at eval-time.
CANONICAL_SAMPLE: list[dict[str, str]] = [
    {
        "role": "user",
        "content": (
            "You are an expert in using FHIR functions to assist medical "
            "professionals.\n\nContext: It's 2023-11-13T10:15:00+00:00 now.\n"
            "Question: parity-fingerprint probe."
        ),
    },
    {"role": "assistant", "content": "GET http://localhost:8080/fhir/Patient?identifier=S0"},
]


def _md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def compute_fingerprint(tokenizer: Any, *, enable_thinking: bool) -> dict[str, Any]:
    """Compute a deterministic fingerprint of how this tokenizer will format chats.

    Captures:
      * md5 of ``tokenizer.chat_template`` (the raw jinja string)
      * the kwargs we'll feed apply_chat_template at SFT/eval time
      * md5 of the rendered CANONICAL_SAMPLE under those kwargs
    """
    template = getattr(tokenizer, "chat_template", "") or ""
    rendered = tokenizer.apply_chat_template(
        CANONICAL_SAMPLE,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=enable_thinking,
    )
    return {
        "template_md5": _md5(template),
        "template_len": len(template),
        "kwargs": {"enable_thinking": enable_thinking},
        "rendered_sample_md5": _md5(rendered),
        "rendered_sample_len": len(rendered),
    }


def write_fingerprint(tokenizer: Any, out_dir: str | Path, *, enable_thinking: bool) -> Path:
    """Write the fingerprint as ``chat_template_fingerprint.json`` in ``out_dir``."""
    fp = compute_fingerprint(tokenizer, enable_thinking=enable_thinking)
    out_path = Path(out_dir) / "chat_template_fingerprint.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(fp, indent=2) + "\n")
    return out_path


def assert_parity(
    tokenizer: Any,
    fingerprint_path: str | Path,
    *,
    enable_thinking: bool,
) -> dict[str, Any]:
    """Compare current tokenizer's fingerprint against a saved one. Raise on mismatch.

    Returns the dict comparison for logging.
    """
    fp_path = Path(fingerprint_path)
    if not fp_path.exists():
        raise FileNotFoundError(
            f"chat-template fingerprint missing at {fp_path}; "
            "did SFT write one alongside the merged checkpoint?"
        )
    saved = json.loads(fp_path.read_text())
    current = compute_fingerprint(tokenizer, enable_thinking=enable_thinking)
    diffs = {
        k: {"saved": saved.get(k), "current": current.get(k)}
        for k in ("template_md5", "kwargs", "rendered_sample_md5")
        if saved.get(k) != current.get(k)
    }
    if diffs:
        raise RuntimeError(
            "Chat-template parity FAILURE between SFT and eval. "
            "This is the historical 'eval default differs from SFT default' "
            "bug class. Diffs:\n" + json.dumps(diffs, indent=2)
        )
    return {"saved": saved, "current": current, "match": True}
