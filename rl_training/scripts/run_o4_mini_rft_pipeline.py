#!/usr/bin/env python3
"""End-to-end orchestrator for the o4-mini RFT pipeline.

Sequences:
    1. Baseline eval of ``o4-mini-2025-04-16``            → 01_baseline_o4_mini_ootb.json
    2. Generate + pre-fetch RFT dataset                   → train.jsonl / val.jsonl
    3. Launch OpenAI RFT job                              → finetuned_model_id.txt
    4. Post-RFT eval of the fine-tuned model              → 02_finetuned_o4_mini_benchmark.json
    5. Write delta report                                 → report.md

Each stage can be independently skipped with ``--skip-*`` flags; this script
shells out to the stage-specific CLIs so they remain usable in isolation.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _run(cmd: list[str]) -> None:
    logger.info("$ %s", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=_ROOT)
    if proc.returncode != 0:
        raise SystemExit(f"Stage failed (exit {proc.returncode}): {' '.join(cmd)}")


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _fmt_delta(a: float, b: float) -> str:
    d = (b - a) * 100
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.1f}"


def _write_report(
    *,
    run_dir: Path,
    baseline: dict | None,
    post: dict | None,
    ft_meta: dict | None,
    config_path: str,
) -> None:
    lines: list[str] = []
    lines.append("# o4-mini RFT Pipeline Report")
    lines.append("")
    lines.append(f"- Config: `{config_path}`")
    if ft_meta:
        lines.append(f"- Fine-tuned model: `{ft_meta.get('fine_tuned_model')}`")
        lines.append(f"- RFT job id: `{ft_meta.get('job_id')}`")
        lines.append(f"- Base model: `{ft_meta.get('base_model')}`")
        lines.append(f"- Train rows: {ft_meta.get('n_train')}  |  Val rows: {ft_meta.get('n_val')}")
        last_m = ft_meta.get("last_metrics") or {}
        if last_m:
            lines.append(
                f"- Last metrics step={last_m.get('step')} "
                f"train_reward={last_m.get('train_reward_mean') or last_m.get('train_mean_reward')} "
                f"valid_reward={last_m.get('valid_reward_mean') or last_m.get('full_valid_mean_reward')}"
            )
    lines.append("")

    if baseline and post:
        lines.append("## Overall success rate")
        lines.append("")
        lines.append("| Metric | Baseline | Post-RFT | Delta (pp) |")
        lines.append("|---|---|---|---|")
        for key in ("success_rate", "query_sr", "action_sr"):
            a = baseline.get(key, 0.0)
            b = post.get(key, 0.0)
            lines.append(f"| {key} | {_fmt_pct(a)} | {_fmt_pct(b)} | {_fmt_delta(a, b)} |")
        lines.append(
            f"| invalid_action_rate | {_fmt_pct(baseline.get('invalid_action_rate', 0))}"
            f" | {_fmt_pct(post.get('invalid_action_rate', 0))}"
            f" | {_fmt_delta(baseline.get('invalid_action_rate', 0), post.get('invalid_action_rate', 0))} |"
        )
        lines.append(
            f"| avg_steps | {baseline.get('avg_steps', 0):.2f} | {post.get('avg_steps', 0):.2f}"
            f" | {post.get('avg_steps', 0) - baseline.get('avg_steps', 0):+.2f} |"
        )
        lines.append("")

        lines.append("## Per-task success rate")
        lines.append("")
        lines.append("| Task | Baseline | Post-RFT | Delta (pp) |")
        lines.append("|---|---|---|---|")
        per_a = baseline.get("per_task_sr", {}) or {}
        per_b = post.get("per_task_sr", {}) or {}
        keys = sorted(set(per_a.keys()) | set(per_b.keys()), key=lambda k: int(k.replace("task", "")))
        for k in keys:
            a = per_a.get(k, 0.0)
            b = per_b.get(k, 0.0)
            lines.append(f"| {k} | {_fmt_pct(a)} | {_fmt_pct(b)} | {_fmt_delta(a, b)} |")
        lines.append("")
    else:
        missing = []
        if not baseline:
            missing.append("baseline")
        if not post:
            missing.append("post-RFT")
        lines.append(f"_Report incomplete: missing {', '.join(missing)} eval payload._")
        lines.append("")

    lines.append("## Artifacts")
    lines.append("")
    for fname in (
        "01_baseline_o4_mini_ootb.json",
        "02_finetuned_o4_mini_benchmark.json",
        "train.jsonl",
        "val.jsonl",
        "finetuned_model_id.txt",
        "finetuned_model_id.json",
        "rft_job_id.txt",
        "rft_events_tail.json",
    ):
        p = run_dir / fname
        marker = "ok" if p.exists() else "missing"
        lines.append(f"- `{fname}` ({marker})")
    lines.append("")

    report_path = run_dir / "report.md"
    report_path.write_text("\n".join(lines))
    logger.info("Wrote report → %s", report_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end o4-mini RFT pipeline")
    parser.add_argument("--config", default="rl_training/configs/o4_mini_rft.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-data", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-post-eval", action="store_true")
    parser.add_argument("--task-limit", type=int, default=None,
                        help="Pass through to eval stages for smoke testing")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    run_dir = Path(args.output_dir or config["output"]["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    python = sys.executable

    if not args.skip_baseline:
        cmd = [
            python, "rl_training/scripts/eval_o4_mini_baseline.py",
            "--config", args.config,
            "--output-dir", str(run_dir),
        ]
        if args.task_limit is not None:
            cmd += ["--task-limit", str(args.task_limit)]
        _run(cmd)

    if not args.skip_data:
        _run([
            python, "rl_training/scripts/build_rft_dataset.py",
            "--config", args.config,
            "--output-dir", str(run_dir),
        ])

    if not args.skip_train:
        _run([
            python, "rl_training/scripts/finetune_o4_mini_rft.py",
            "--config", args.config,
            "--output-dir", str(run_dir),
        ])

    if not args.skip_post_eval:
        cmd = [
            python, "rl_training/scripts/eval_o4_mini_post_rft.py",
            "--config", args.config,
            "--output-dir", str(run_dir),
        ]
        if args.task_limit is not None:
            cmd += ["--task-limit", str(args.task_limit)]
        _run(cmd)

    baseline = _load_json(run_dir / "01_baseline_o4_mini_ootb.json")
    post = _load_json(run_dir / "02_finetuned_o4_mini_benchmark.json")
    ft_meta = _load_json(run_dir / "finetuned_model_id.json")

    _write_report(
        run_dir=run_dir,
        baseline=baseline,
        post=post,
        ft_meta=ft_meta,
        config_path=args.config,
    )

    if baseline and post:
        a = baseline.get("success_rate", 0.0)
        b = post.get("success_rate", 0.0)
        print(f"\nBaseline SR: {a * 100:.1f}%  |  Post-RFT SR: {b * 100:.1f}%  |  Delta: {(b - a) * 100:+.1f}pp")


if __name__ == "__main__":
    main()
