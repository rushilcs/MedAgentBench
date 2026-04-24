#!/usr/bin/env python3
"""Deploy a 1xH100 RunPod for the Phase B SFT v2 re-eval on the v3 pipeline.

Validates end-to-end before the $30/hr 8xH100 burn:
  * SFT v2 merged downloads + serves cleanly.
  * The cloudflared tunnel from pod -> Mac docker FHIR is reachable.
  * The bug-fixed run_post_train_eval (enable_thinking=false, parity gate,
    infra-error tracking, snapshot fall-through cache) works end-to-end.
  * Auto-terminates the pod once /workspace/.sft_v2_v3pipeline_eval_done appears.

Pre-reqs:
  1. .agent_run/env.json populated.
  2. .agent_run/ssh/id_ed25519 SSH key matching env.json PUBLIC_KEY.
  3. Local docker FHIR up + cloudflared tunnel running. Tunnel URL passed via:
       --tunnel-url https://xyz.trycloudflare.com
     (or auto-read from .agent_run/tunnel_url.txt).

Usage::

    bash .agent_run/start_local_fhir_tunnel.sh
    /usr/bin/python3 rl_training/scripts/deploy_sft_v2_v3pipeline_eval.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_DEFAULT_ENV = _REPO / ".agent_run" / "env.json"
_RUNPOD_MOD = _REPO / ".agent_run" / "runpod.py"
_BOOTSTRAP_SH = _REPO / ".agent_run" / "pod_sft_v2_v3pipeline_eval_bootstrap.sh"
_SUMMARY = _REPO / "rl_training" / "outputs" / "last_sft_v2_v3pipeline_eval_launch.json"
_SSH_KEY = _REPO / ".agent_run" / "ssh" / "id_ed25519"
_TUNNEL_URL_FILE = _REPO / ".agent_run" / "tunnel_url.txt"

_DONE_MARKER = "/workspace/.sft_v2_v3pipeline_eval_done"


def _load_runpod():
    spec = importlib.util.spec_from_file_location("runpod_local", _RUNPOD_MOD)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _pod_env(secrets: dict, *, tunnel_url: str | None) -> dict[str, str]:
    skip = {"JUPYTER_PASSWORD"}
    env = {k: str(v) for k, v in secrets.items() if v and k not in skip}
    if tunnel_url:
        env["TUNNEL_URL"] = tunnel_url
        # _default_live_getter appends "/fhir" itself; pass the host root only.
        env["FHIR_LIVE_BASE_OVERRIDE"] = tunnel_url.rstrip("/")
    return env


def _deploy(rp, env: dict[str, str], name: str) -> dict:
    image = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
    tries = [
        ("NVIDIA H100 80GB HBM3", 1, "SECURE"),
        ("NVIDIA H100 PCIe", 1, "SECURE"),
        ("NVIDIA H100 80GB HBM3", 1, "COMMUNITY"),
    ]
    last_err = None
    for gpu_id, gcount, cloud in tries:
        try:
            return rp.find_and_deploy(
                name=f"{name}-{cloud[:3].lower()}",
                gpu_type_id=gpu_id, gpu_count=gcount,
                container_disk_gb=200, volume_gb=250,
                image_name=image, env=env,
                ports="22/tcp,8000/http",
                min_vcpu=8, min_memory_gb=80, cloud_type=cloud,
            )
        except RuntimeError as e:
            last_err = e
            continue
    raise last_err or RuntimeError("deploy failed")


def _ssh_info(rp, pod_id: str):
    return rp._ssh_info(rp.pod(pod_id))  # noqa: SLF001


def _run_ssh(host, port, cmd, *, capture=False):
    full = ["ssh", "-i", str(_SSH_KEY), "-p", str(port),
            "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=30",
            f"root@{host}", cmd]
    return subprocess.run(full, capture_output=capture, text=True, check=False)


def _scp(host, port, src, dst):
    full = ["scp", "-i", str(_SSH_KEY), "-P", str(port),
            "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=30",
            str(src), f"root@{host}:{dst}"]
    return subprocess.run(full, capture_output=True, text=True, check=False)


def _wait_ssh(rp, pod_id, *, timeout_s=900):
    start = time.time()
    while time.time() - start < timeout_s:
        info = _ssh_info(rp, pod_id)
        if info:
            host, port = info
            r = _run_ssh(host, port, "echo READY", capture=True)
            if r.returncode == 0 and "READY" in r.stdout:
                return host, port
        time.sleep(15)
        print(f"   ... still waiting for SSH (elapsed {int(time.time()-start)}s)")
    raise RuntimeError(f"SSH not ready within {timeout_s}s for pod {pod_id}")


def _terminate(rp, pod_id):
    try:
        print(f"\nTerminating pod {pod_id} ...")
        rp.terminate(pod_id)
        print("  termination request sent")
    except Exception as exc:
        print(f"  ERROR terminating pod: {exc}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pod-id", default=None,
                    help="Reuse an existing pod instead of deploying.")
    ap.add_argument("--name", default=f"medagent-sft-v2-reeval-{time.strftime('%m%d-%H%M')}")
    ap.add_argument("--env-file", default=str(_DEFAULT_ENV))
    ap.add_argument("--tunnel-url", default=None,
                    help="Cloudflare tunnel URL pointing at local docker FHIR. "
                         "If unset, reads from .agent_run/tunnel_url.txt.")
    ap.add_argument("--no-kick", action="store_true")
    ap.add_argument("--dont-terminate", action="store_true",
                    help="Leave pod running after eval completes.")
    ap.add_argument("--poll-interval-s", type=int, default=120,
                    help="Poll for done marker this often.")
    ap.add_argument("--max-wait-min", type=int, default=120,
                    help="Give up waiting for done marker after this many minutes.")
    args = ap.parse_args()

    secrets = json.loads(Path(args.env_file).read_text())
    for required in ("RUNPOD_API_KEY", "PUBLIC_KEY", "B2_BUCKET", "B2_APPLICATION_KEY_ID", "B2_APPLICATION_KEY"):
        if not secrets.get(required):
            raise SystemExit(f"{required} missing from {args.env_file}")
    if not _SSH_KEY.exists():
        raise SystemExit(f"SSH private key missing at {_SSH_KEY}")
    if not _BOOTSTRAP_SH.exists():
        raise SystemExit(f"Bootstrap script missing at {_BOOTSTRAP_SH}")

    tunnel_url = args.tunnel_url
    if not tunnel_url and _TUNNEL_URL_FILE.exists():
        tunnel_url = _TUNNEL_URL_FILE.read_text().strip()
        print(f"Using tunnel URL from {_TUNNEL_URL_FILE}: {tunnel_url}")
    if not tunnel_url:
        print("WARNING: no --tunnel-url and no .agent_run/tunnel_url.txt; "
              "eval will run snapshot-only on the pod (no live FHIR).")
    else:
        # Local tunnel pre-flight is best-effort. Some local DNS resolvers
        # (Tailscale MagicDNS, corporate DNS) don't cache the trycloudflare
        # subdomain quickly; the pod's resolver will. So we try locally,
        # warn if it fails, but DON'T abort -- the pod will re-test via
        # curl in the bootstrap and report there.
        try:
            import urllib.request
            with urllib.request.urlopen(tunnel_url.rstrip("/") + "/fhir/metadata", timeout=10) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"tunnel returned HTTP {resp.status}")
            print(f"Tunnel sanity OK locally: {tunnel_url}/fhir/metadata returns 200")
        except Exception as exc:
            print(f"WARNING: local tunnel pre-flight failed ({exc}). "
                  f"This is usually a local-DNS quirk; the pod will retry "
                  f"with public DNS. Continuing.")

    os.environ["RUNPOD_API_KEY"] = secrets["RUNPOD_API_KEY"]
    rp = _load_runpod()

    if args.pod_id:
        pod_id = args.pod_id
        print(f"Reusing existing pod {pod_id}")
    else:
        env = _pod_env(secrets, tunnel_url=tunnel_url)
        print(f"Deploying new pod {args.name} (1xH100 80GB) ...")
        info = _deploy(rp, env, args.name)
        pod_id = info["id"]
        print(f"  pod_id   = {pod_id}")
        print(f"  console  = https://www.runpod.io/console/pods/{pod_id}")

    print(f"\nWaiting for pod {pod_id} RUNNING ...")
    rp.wait_until_running(pod_id, timeout_s=900)
    print("Pod RUNNING. Waiting for SSH ...")
    host, port = _wait_ssh(rp, pod_id)
    print(f"SSH ready: ssh -i {_SSH_KEY} -p {port} root@{host}")

    summary = {
        "pod_id": pod_id,
        "console_url": f"https://www.runpod.io/console/pods/{pod_id}",
        "ssh_command": f"ssh -i {_SSH_KEY} -p {port} root@{host}",
        "host": host, "port": port,
        "tunnel_url": tunnel_url,
        "bootstrap_remote": "/root/pod_sft_v2_v3pipeline_eval_bootstrap.sh",
        "b2_prefix_out": "qwen3_32b_run3/sft_v2_v3pipeline_eval",
    }

    if args.no_kick:
        _SUMMARY.parent.mkdir(parents=True, exist_ok=True)
        _SUMMARY.write_text(json.dumps(summary, indent=2))
        print("\n--no-kick set; SSH in and run the bootstrap manually.")
        return

    print("\nUploading bootstrap script ...")
    r = _scp(host, port, _BOOTSTRAP_SH, "/root/pod_sft_v2_v3pipeline_eval_bootstrap.sh")
    if r.returncode != 0:
        print(r.stdout); print(r.stderr, file=sys.stderr)
        raise SystemExit("scp failed")

    print("Kicking bootstrap inside detached tmux ...")
    kick = (
        "apt-get update -qq && apt-get install -y -qq tmux >/dev/null 2>&1 || true; "
        "chmod +x /root/pod_sft_v2_v3pipeline_eval_bootstrap.sh; "
        "tmux kill-session -t v2reeval 2>/dev/null || true; "
        "tmux new-session -d -s v2reeval "
        "'bash /root/pod_sft_v2_v3pipeline_eval_bootstrap.sh 2>&1 | tee /workspace/v2reeval_outer.log'; "
        "echo BOOTSTRAP_KICKED"
    )
    r = _run_ssh(host, port, kick, capture=True)
    print(r.stdout)
    if "BOOTSTRAP_KICKED" not in r.stdout:
        print(r.stderr, file=sys.stderr)
        raise SystemExit("bootstrap kick failed")

    summary["kicked"] = True
    _SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    _SUMMARY.write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 65)
    print(" Phase B SFT v2 re-eval is running on the pod.")
    print(f" Pod console : {summary['console_url']}")
    print(f" Watch log   : ssh -i {_SSH_KEY} -p {port} root@{host} \\")
    print( "                 'tail -f /workspace/sft_v2_v3pipeline_eval_bootstrap.log'")
    print(f" Done marker : {_DONE_MARKER}")
    print(f" B2 dest     : b2://{secrets['B2_BUCKET']}/{summary['b2_prefix_out']}/")
    if not args.dont_terminate:
        print(f" Auto-term   : ON (will terminate pod when done marker appears)")
    print("=" * 65 + "\n")

    # Poll for done marker, then terminate if requested.
    if args.dont_terminate:
        return

    deadline = time.time() + args.max_wait_min * 60
    print(f"Polling for {_DONE_MARKER} every {args.poll_interval_s}s "
          f"(max wait {args.max_wait_min} min) ...")
    while time.time() < deadline:
        time.sleep(args.poll_interval_s)
        r = _run_ssh(host, port,
                     f"test -f {_DONE_MARKER} && echo DONE || echo PENDING",
                     capture=True)
        status = (r.stdout or "").strip().splitlines()[-1] if r.stdout else "?"
        elapsed_min = int((time.time() - (deadline - args.max_wait_min * 60)) / 60)
        print(f"  [+{elapsed_min} min] status: {status}")
        if status == "DONE":
            print("Done marker present.")
            # Pull eval.json summary back for the operator.
            _run_ssh(host, port,
                     f"cat /workspace/MedAgentBench/rl_training/outputs/sft_v2_v3pipeline_eval/eval.json",
                     capture=False)
            _terminate(rp, pod_id)
            return
    print(f"WARNING: max wait ({args.max_wait_min} min) reached. Pod NOT terminated; "
          f"check manually: ssh -i {_SSH_KEY} -p {port} root@{host}")


if __name__ == "__main__":
    main()
