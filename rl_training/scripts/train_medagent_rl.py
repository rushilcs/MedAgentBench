#!/usr/bin/env python3
"""Benchmark-aligned MedAgentBench GRPO entrypoint (plan file layout).

Delegates to ``train_grpo_32b.main`` with the same CLI flags and YAML schema.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from rl_training.scripts.train_grpo_32b import main  # noqa: E402


if __name__ == "__main__":
    main()
