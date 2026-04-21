"""Optional completion canonicalization before action parsing (parser-only path).

FSM / grammar-constrained decoding belongs behind ``use_fsm_constrained_decode``
for ablations; default is lightweight stripping compatible with
``rl_training.env.action_parser.parse_action``.
"""

from __future__ import annotations

import os

from rl_training.env.action_parser import _strip_reasoning


def canonicalize_completion(text: str) -> str:
    """Strip reasoning blocks / channel prefixes so GET/POST/FINISH anchors parse."""
    return _strip_reasoning(text or "").strip()


def use_fsm_constrained_decode() -> bool:
    """When true, rollout pipeline may apply stricter decoding (ablation hook)."""
    return os.environ.get("MEDAGENT_RL_FSM_DECODE", "").lower() in ("1", "true", "yes")


def apply_decode_postprocess(text: str) -> str:
    """Parser path + optional FSM-style first-line extraction (GET/FINISH only).

    POST actions keep the full multi-line body so JSON payloads stay intact.
    """
    base = canonicalize_completion(text)
    if not use_fsm_constrained_decode():
        return base
    lines = base.splitlines()
    if not lines:
        return base
    head = lines[0].strip()
    if head.startswith("POST "):
        return base
    for line in lines:
        s = line.strip()
        if s.startswith("GET ") or s.startswith("FINISH("):
            return s
    return base
