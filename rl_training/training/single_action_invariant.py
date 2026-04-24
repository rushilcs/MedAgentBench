"""Single-action-per-assistant-turn invariant.

The MedAgentBench env contract: each assistant turn must contain exactly one of
- ``GET <url>``
- ``POST <url>\n<json payload>``
- ``FINISH([...])``

If a gold trajectory or an SFT training row violates this (e.g. two POSTs in
one turn, or prose before the action), the eval grader will fail the trajectory
even though the intent was correct (refsol's ``extract_posts`` only sees POSTs
that are immediately followed by ``"POST request accepted"`` in the next user
turn).

This module gives both the expert collector and the SFT data loader a single
validator they can call to fail loudly at build/load time rather than silently
training the model on illegal turns.
"""

from __future__ import annotations

import re
from typing import Iterable

# Match how rl_training/env/action_parser.py recognises actions: lines starting
# with GET / POST / FINISH (after stripping <think> blocks). We tolerate
# leading whitespace because some v1 trajectories had a single blank line.
_GET_RE = re.compile(r"^\s*GET\s+", re.MULTILINE)
_POST_RE = re.compile(r"^\s*POST\s+", re.MULTILINE)
_FINISH_RE = re.compile(r"^\s*FINISH\s*\(", re.MULTILINE)


def classify_turn(content: str) -> tuple[int, int, int]:
    """Return (n_get, n_post, n_finish) for an assistant turn body."""
    return (
        len(_GET_RE.findall(content)),
        len(_POST_RE.findall(content)),
        len(_FINISH_RE.findall(content)),
    )


def is_single_action_turn(content: str) -> bool:
    """True iff the turn contains exactly one action (GET, POST, or FINISH)."""
    n = sum(classify_turn(content))
    return n == 1


def violations(messages: Iterable[dict]) -> list[dict]:
    """Return a list of violation records for assistant turns in ``messages``.

    Each record: ``{"index": int, "content": str, "counts": (g, p, f)}``.
    Empty list means the trajectory is valid.
    """
    bad: list[dict] = []
    for i, msg in enumerate(messages):
        if msg.get("role") not in ("agent", "assistant"):
            continue
        c = msg.get("content", "")
        counts = classify_turn(c)
        if sum(counts) != 1:
            bad.append({"index": i, "content": c, "counts": counts})
    return bad


def assert_valid(messages: Iterable[dict], *, context: str = "") -> None:
    """Raise ``ValueError`` if any assistant turn violates the invariant."""
    msgs = list(messages)
    bad = violations(msgs)
    if bad:
        head = bad[0]
        raise ValueError(
            f"single-action-per-turn violation in {context or '<unnamed>'}: "
            f"{len(bad)} bad turn(s); first at idx={head['index']} "
            f"counts(g,p,f)={head['counts']} "
            f"content[:120]={head['content'][:120]!r}"
        )
