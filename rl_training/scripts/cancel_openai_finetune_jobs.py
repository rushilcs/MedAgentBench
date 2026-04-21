#!/usr/bin/env python3
"""Cancel all OpenAI fine-tuning jobs that are not in a terminal state.

Terminal states: succeeded, failed, cancelled.
Active states: validating_files, queued, running.

Usage:
  export OPENAI_API_KEY=...
  python rl_training/scripts/cancel_openai_finetune_jobs.py
  python rl_training/scripts/cancel_openai_finetune_jobs.py --dry-run
"""
from __future__ import annotations

import argparse
import os
import sys

from openai import OpenAI


def main() -> None:
    parser = argparse.ArgumentParser(description="Cancel running OpenAI fine-tuning jobs")
    parser.add_argument("--dry-run", action="store_true", help="List jobs only, do not cancel")
    args = parser.parse_args()

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    after: str | None = None
    terminal = {"succeeded", "failed", "cancelled"}
    cancelled = 0
    checked = 0

    while True:
        page = client.fine_tuning.jobs.list(limit=100, after=after)
        for job in page.data:
            checked += 1
            st = job.status
            if st in terminal:
                continue
            print(f"{job.id}  status={st}  model={job.model}")
            if args.dry_run:
                continue
            try:
                client.fine_tuning.jobs.cancel(job.id)
                print(f"  -> cancel requested")
                cancelled += 1
            except Exception as exc:
                print(f"  -> cancel failed: {exc}", file=sys.stderr)

        if not page.has_more or not page.data:
            break
        after = page.data[-1].id

    print(f"Checked {checked} job(s); cancelled {cancelled} non-terminal job(s).")
    if cancelled == 0 and checked > 0:
        print("No active fine-tuning jobs (nothing to cancel).")


if __name__ == "__main__":
    main()
