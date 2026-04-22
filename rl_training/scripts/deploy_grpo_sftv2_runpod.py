#!/usr/bin/env python3
"""Deploy a 2xH100 RunPod and print GRPO (SFT v2) bootstrap steps.

Mirrors prior agent runs: uses ``.agent_run/runpod.py`` GraphQL helpers, loads
secrets from ``MEDAGENT_RUNPOD_ENV_FILE`` (default ``.agent_run/env.json``),
then waits for SSH/Web terminal.

SSH from some networks is blocked; use the RunPod **Web terminal** (Connect)
and paste the printed bootstrap block.

Usage (repo root, ``/usr/bin/python3`` recommended on macOS)::

    # API key optional if present in .agent_run/env.json as RUNPOD_API_KEY
    /usr/bin/python3 rl_training/scripts/deploy_grpo_sftv2_runpod.py

    # Re-print instructions for an existing pod:
    /usr/bin/python3 rl_training/scripts/deploy_grpo_sftv2_runpod.py --pod-id POD_ID
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_DEFAULT_ENV = _REPO / ".agent_run" / "env.json"
_RUNPOD_MOD = _REPO / ".agent_run" / "runpod.py"
_SUMMARY = _REPO / "rl_training" / "outputs" / "last_grpo_runpod_launch.json"
_QUICKSTART = _REPO / "rl_training" / "scripts" / "RUNPOD_GRPO_QUICKSTART.md"


def _load_runpod():
    spec = importlib.util.spec_from_file_location("runpod_local", _RUNPOD_MOD)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _pod_env(secrets: dict) -> dict[str, str]:
    # PUBLIC_KEY MUST be passed: the runpod/pytorch image installs it into
    # /root/.ssh/authorized_keys at boot. Without it, both direct SSH and the
    # ssh.runpod.io proxy reject with "Permission denied (publickey)" and
    # the agent can only drive the pod via the RunPod Web Terminal.
    skip = {"JUPYTER_PASSWORD"}
    return {k: str(v) for k, v in secrets.items() if v and k not in skip}


def _deploy(rp, env: dict[str, str], name: str) -> dict:
    image = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
    tries = [
        ("NVIDIA H100 80GB HBM3", 2, "SECURE"),
        ("NVIDIA H100 PCIe", 2, "SECURE"),
        ("NVIDIA H100 80GB HBM3", 2, "COMMUNITY"),
    ]
    last_err = None
    for gpu_id, gcount, cloud in tries:
        try:
            return rp.find_and_deploy(
                name=f"{name}-{cloud[:3].lower()}",
                gpu_type_id=gpu_id,
                gpu_count=gcount,
                container_disk_gb=200,
                volume_gb=80,
                image_name=image,
                env=env,
                ports="22/tcp,8000/http",
                min_vcpu=16,
                min_memory_gb=125,
                cloud_type=cloud,
            )
        except RuntimeError as e:
            last_err = e
            continue
    raise last_err or RuntimeError("deploy failed")


def _ssh_line(rp, pod_id: str) -> str | None:
    import subprocess

    out = subprocess.run(
        [sys.executable, str(_RUNPOD_MOD), "ssh-info", "--id", pod_id],
        capture_output=True,
        text=True,
        cwd=str(_REPO),
        env={**os.environ},
    )
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


_SFT_V2_GUARD_PREFIXES = (
    "qwen3_32b_run3/qwen3_32b_sft_v2",
    "qwen3_32b_run3/artifacts/sft_v2_merged",
    "qwen3_32b_run3/grpo_v2_ckpts",  # JSON-tool GRPO ckpts also preserved
)


def _assert_preserves_sft_v2(config_path: str, b2_prefix: str) -> None:
    """Refuse to launch if the run could overwrite preserved SFT v2 artifacts.

    The plain-text GRPO run must use a brand-new B2 prefix
    (``qwen3_32b_run3/grpo_v2_plain_ckpts``); both the SFT v2 merged weights
    (read-only source) and the prior JSON-tool GRPO ckpts must stay
    untouched in B2 across this experiment so we can always roll back to
    the validated 75% baseline.
    """
    for guard in _SFT_V2_GUARD_PREFIXES:
        if b2_prefix == guard or b2_prefix.startswith(guard + "/"):
            raise SystemExit(
                f"REFUSING TO DEPLOY: B2_PREFIX={b2_prefix!r} would collide "
                f"with preserved prefix {guard!r}. Pick a fresh prefix "
                "(e.g. qwen3_32b_run3/grpo_v2_plain_ckpts) so the SFT v2 "
                "weights and JSON-tool GRPO ckpts stay intact in B2."
            )
    cfg = Path(config_path)
    if cfg.is_file():
        text = cfg.read_text()
        if "rm -rf" in text or "b2 rm" in text or "delete_file_version" in text:
            raise SystemExit(
                f"REFUSING TO DEPLOY: config {config_path!r} contains a "
                "delete operation. Plain-text GRPO must be download/write-only."
            )


def _bootstrap_block(config: str, b2_prefix: str) -> str:
    return rf"""# Paste into RunPod Web Terminal (Connect → Terminal), repo root will be created.

set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq git tmux curl ca-certificates jq rsync

mkdir -p /workspace
cd /workspace
if [[ ! -d MedAgentBench/.git ]]; then
  git clone --depth 1 https://github.com/rushilcs/MedAgentBench.git MedAgentBench
fi
cd /workspace/MedAgentBench
git pull --ff-only || true

export PYTHONPATH="/workspace/MedAgentBench:${{PYTHONPATH:-}}"
export STAGE=grpo
export SKIP_FLASH_ATTN=1
# Plain-text GRPO: rollout_func runs MedAgentEnv (GET/POST/FINISH text actions)
# so the run is directly comparable to the MedAgentBench paper / SFT v2 75% eval.
export CONFIG="${{CONFIG:-{config}}}"
export TRAINING_TASKS="${{TRAINING_TASKS:-rl_training/data/training_tasks_v2.json}}"
export OUTPUT_DIR="${{OUTPUT_DIR:-rl_training/outputs/qwen3_32b_grpo_v2_plain}}"
# NEW prefix; SFT v2 merged + JSON-tool GRPO ckpts under qwen3_32b_run3/* are NOT touched.
export B2_PREFIX="${{B2_PREFIX:-{b2_prefix}}}"
export B2_PREFIX_MERGED="${{B2_PREFIX_MERGED:-qwen3_32b_run3/artifacts/sft_v2_merged}}"
export MERGED_MODEL_DIR="${{MERGED_MODEL_DIR:-/qwen3_32b_sft_v2_merged}}"
export FHIR_SNAPSHOT_LOCAL="${{FHIR_SNAPSHOT_LOCAL:-rl_training/outputs/fhir_snapshot.jsonl}}"
export GRPO_TORCHRUN_PROCS="${{GRPO_TORCHRUN_PROCS:-2}}"

# Belt-and-suspenders SFT v2 preservation guard on the pod side.
case "$B2_PREFIX" in
  qwen3_32b_run3/qwen3_32b_sft_v2*|qwen3_32b_run3/artifacts/sft_v2_merged*|qwen3_32b_run3/grpo_v2_ckpts*)
    echo "REFUSING TO LAUNCH: B2_PREFIX=$B2_PREFIX would overwrite preserved SFT v2 artifacts." >&2
    exit 2 ;;
esac

# Env vars (B2_*, HF_TOKEN, RUNPOD_*) should already exist on the pod from RunPod UI.
# If not, re-add them in the RunPod console, stop pod, start pod, or export here.

bash .agent_run/remote_bootstrap_grpo.sh
"""


def _write_quickstart(
    *,
    pod_id: str,
    console_url: str,
    ssh_cmd: str | None,
    bootstrap: str,
) -> None:
    ssh_section = (
        f"```bash\n{ssh_cmd}\n```\n"
        if ssh_cmd
        else "_Run `python .agent_run/runpod.py ssh-info --id POD_ID` from repo root for the current port._\n"
    )
    body = f"""# RunPod → MedAgentBench GRPO (SFT v2) quickstart

Auto-generated by `deploy_grpo_sftv2_runpod.py`. **Pod ID:** `{pod_id}`

## A. What you should have already

- RunPod account + billing.
- Secrets in **`.agent_run/env.json`** (or set env vars on the pod in the RunPod UI):  
  `HF_TOKEN`, `B2_*`, `RUNPOD_API_KEY`, `B2_BUCKET`, etc.
- **GitHub:** push your latest `MedAgentBench` commits so `git pull` on the pod matches your laptop.

---

## B. On your laptop (repo root: `MedAgentBench`)

Create/reuse the pod and refresh this file:

```bash
cd /path/to/MedAgentBench

# If RUNPOD_API_KEY is not exported, the deploy script reads it from .agent_run/env.json
/usr/bin/python3 rl_training/scripts/deploy_grpo_sftv2_runpod.py

# Re-use an existing pod (no new billing):
/usr/bin/python3 rl_training/scripts/deploy_grpo_sftv2_runpod.py --pod-id {pod_id}
```

Summary JSON: `rl_training/outputs/last_grpo_runpod_launch.json`

---

## C. Start training (inside RunPod — Web Terminal)

1. Open: **{console_url}**
2. Click **Connect** → **Terminal** (or **Start** if the pod was stopped).
3. **Paste the whole block below** (one shot). It clones the repo, pulls, and runs the same bootstrap as before.

```bash
{bootstrap.strip()}
```

That runs `.agent_run/remote_bootstrap_grpo.sh` → `launch_runpod.sh` → **tmux** sessions (`trainer`, `watchdog`; `vllm` only if config enables it).

---

## D. Watch progress (still on the pod, Web Terminal or SSH)

```bash
tmux ls
tmux attach -t trainer    # Ctrl+B then D to detach
tail -f rl_training/outputs/qwen3_32b_grpo_v2/progress.jsonl
tail -f rl_training/outputs/trainer.log
```

---

## E. Optional: SSH from your Mac terminal (not Cursor agent)

From repo root (paths relative to repo):

{ssh_section}

If you see **connection refused**, use the **Web Terminal** (section C); some networks block direct TCP to RunPod.

---

## F. Stop billing when done

RunPod UI → **Stop** or **Terminate** the pod, or:

```bash
/usr/bin/python3 .agent_run/runpod.py terminate --id {pod_id}
```
"""
    _QUICKSTART.parent.mkdir(parents=True, exist_ok=True)
    _QUICKSTART.write_text(body, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pod-id",
        help="Skip deploy; only wait (if needed) and print bootstrap for this pod",
    )
    ap.add_argument(
        "--env-file",
        default=os.environ.get("MEDAGENT_RUNPOD_ENV_FILE", str(_DEFAULT_ENV)),
        help="JSON file with HF_TOKEN, B2_*, RUNPOD_API_KEY, ...",
    )
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument(
        "--config",
        default="rl_training/configs/qwen3_32b_grpo_post_sft_v2_plain.yaml",
        help="Trainer config to bake into the bootstrap block.",
    )
    ap.add_argument(
        "--b2-prefix",
        default="qwen3_32b_run3/grpo_v2_plain_ckpts",
        help="B2 destination prefix for ckpt sync (must NOT collide with SFT v2 prefixes).",
    )
    args = ap.parse_args()

    _assert_preserves_sft_v2(args.config, args.b2_prefix)

    env_path = Path(args.env_file)
    if not env_path.is_file():
        print(f"ERROR: env file missing: {env_path}", file=sys.stderr)
        sys.exit(2)
    secrets = json.loads(env_path.read_text())
    if not os.environ.get("RUNPOD_API_KEY") and secrets.get("RUNPOD_API_KEY"):
        os.environ["RUNPOD_API_KEY"] = str(secrets["RUNPOD_API_KEY"])

    rp = _load_runpod()
    if not os.environ.get("RUNPOD_API_KEY"):
        print("ERROR: set RUNPOD_API_KEY or add it to env file", file=sys.stderr)
        sys.exit(2)

    pod_env = _pod_env(secrets)

    if args.pod_id:
        pod_id = args.pod_id.strip()
        print(f"Using existing pod: {pod_id}", flush=True)
    else:
        name = "medagent-grpo-" + time.strftime("%m%d-%H%M")
        print(f"Deploying pod ({name})...", flush=True)
        out = _deploy(rp, pod_env, name)
        pod_id = out["id"]
        print("Deployed:", json.dumps(out, indent=2), flush=True)

    print("Waiting for RUNNING + ports...", flush=True)
    rp.wait_until_running(pod_id, timeout_s=args.timeout)
    ssh = _ssh_line(rp, pod_id)
    console = f"https://www.runpod.io/console/pods/{pod_id}"

    summary = {
        "pod_id": pod_id,
        "console_url": console,
        "ssh_command": ssh,
        "config": args.config,
        "b2_prefix": args.b2_prefix,
        "note": (
            "If plain SSH fails from your laptop, open the Web Terminal from the "
            "console URL and run the bootstrap block."
        ),
    }
    _SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    _SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    boot = _bootstrap_block(args.config, args.b2_prefix)
    print("\n--- Web terminal bootstrap (also in repo .agent_run/remote_bootstrap_grpo.sh) ---\n")
    print(boot)
    _write_quickstart(
        pod_id=pod_id,
        console_url=console,
        ssh_cmd=ssh,
        bootstrap=boot,
    )
    print(f"\nWrote: {_SUMMARY.relative_to(_REPO)}")
    print(f"Wrote: {_QUICKSTART.relative_to(_REPO)}")


if __name__ == "__main__":
    main()
