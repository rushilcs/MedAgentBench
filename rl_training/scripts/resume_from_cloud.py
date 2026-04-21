#!/usr/bin/env python3
"""Pull the latest LoRA checkpoint from cloud and stage it locally.

Typical use in ``launch_runpod.sh``:

    python rl_training/scripts/resume_from_cloud.py \\
        --backend b2 \\
        --bucket medagentbench-checkpoints \\
        --prefix qwen3_32b_grpo/clinical \\
        --output-dir rl_training/outputs/qwen3_32b_grpo

Prints the staged local checkpoint directory (or an empty string if nothing
was found) so callers can feed it to ``--resume-from-checkpoint`` directly.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

logger = logging.getLogger(__name__)


def fetch_latest_checkpoint(
    bucket: str,
    prefix: str,
    output_dir: str,
    backend: str = "b2",
) -> str | None:
    """Download the highest-numbered remote ``checkpoint-*`` into ``output_dir``.

    Returns the local path to the staged checkpoint, or ``None`` if no
    remote checkpoint was found.
    """
    from rl_training.training.checkpoint_sync import make_backend

    backend_impl = make_backend(backend, bucket)
    names = backend_impl.list_remote_checkpoints(prefix.rstrip("/") + "/")
    if not names:
        logger.info("No checkpoints at %s/%s", bucket, prefix)
        return None
    latest = names[-1]
    logger.info("Latest remote checkpoint: %s", latest)

    local_dir = Path(output_dir) / latest
    local_dir.mkdir(parents=True, exist_ok=True)
    remote_prefix = f"{prefix.rstrip('/')}/{latest}"
    backend_impl.download_directory(remote_prefix, str(local_dir))
    logger.info("Staged checkpoint to %s", local_dir)
    return str(local_dir)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="Resume from cloud checkpoint")
    parser.add_argument("--backend", default="b2", choices=["b2", "s3"])
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--prefix", required=True,
                        help="Prefix under which checkpoint-* dirs live")
    parser.add_argument("--output-dir", required=True,
                        help="Local parent dir to stage the checkpoint into")
    args = parser.parse_args()

    try:
        local = fetch_latest_checkpoint(
            backend=args.backend,
            bucket=args.bucket,
            prefix=args.prefix,
            output_dir=args.output_dir,
        )
    except Exception as exc:
        logger.error("Resume failed: %s", exc)
        print("")
        return 0
    print(local or "")
    return 0


if __name__ == "__main__":
    sys.exit(main())
