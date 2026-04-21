#!/usr/bin/env python3
"""Minimal RunPod GraphQL helper for provisioning, polling, and terminating pods.

Used by the orchestration layer that drives the Qwen3-32B baseline -> SFT -> GRPO
benchmark run. Reads the API key from $RUNPOD_API_KEY.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

API = "https://api.runpod.io/graphql"


def _headers() -> dict:
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        raise RuntimeError("RUNPOD_API_KEY env var required")
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def gql(query: str, variables: dict | None = None) -> dict:
    r = requests.post(
        API,
        headers=_headers(),
        json={"query": query, "variables": variables or {}},
        timeout=60,
    )
    r.raise_for_status()
    data = r.json()
    if "errors" in data:
        raise RuntimeError(f"RunPod GraphQL error: {data['errors']}")
    return data["data"]


def myself() -> dict:
    return gql("query { myself { email clientBalance currentSpendPerHr } }")["myself"]


def find_and_deploy(
    *,
    name: str,
    gpu_type_id: str,
    gpu_count: int,
    container_disk_gb: int,
    volume_gb: int,
    image_name: str,
    env: dict[str, str],
    ports: str = "22/tcp,8000/http",
    min_vcpu: int = 8,
    min_memory_gb: int = 64,
    cloud_type: str = "SECURE",
    docker_args: str = "",
) -> dict:
    env_list = [{"key": k, "value": str(v)} for k, v in env.items()]
    mutation = """
    mutation deploy($input: PodFindAndDeployOnDemandInput!) {
      podFindAndDeployOnDemand(input: $input) {
        id
        name
        imageName
        machineId
        desiredStatus
        lastStatusChange
        runtime {
          uptimeInSeconds
          ports { ip isIpPublic privatePort publicPort type }
        }
      }
    }
    """
    input_ = {
        "cloudType": cloud_type,
        "gpuCount": gpu_count,
        "volumeInGb": volume_gb,
        "containerDiskInGb": container_disk_gb,
        "minVcpuCount": min_vcpu,
        "minMemoryInGb": min_memory_gb,
        "gpuTypeId": gpu_type_id,
        "name": name,
        "imageName": image_name,
        "dockerArgs": docker_args,
        "ports": ports,
        "volumeMountPath": "/workspace",
        "env": env_list,
    }
    return gql(mutation, {"input": input_})["podFindAndDeployOnDemand"]


def pod(pod_id: str) -> dict:
    q = """
    query p($id: String!) {
      pod(input: {podId: $id}) {
        id name desiredStatus lastStatusChange
        machineId imageName
        runtime {
          uptimeInSeconds
          ports { ip isIpPublic privatePort publicPort type }
          gpus { id gpuUtilPercent memoryUtilPercent }
        }
        machine { podHostId }
      }
    }
    """
    return gql(q, {"id": pod_id})["pod"]


def terminate(pod_id: str) -> bool:
    m = """
    mutation t($id: String!) { podTerminate(input: {podId: $id}) }
    """
    gql(m, {"id": pod_id})
    return True


def stop(pod_id: str) -> bool:
    m = """
    mutation s($id: String!) { podStop(input: {podId: $id}) { id desiredStatus } }
    """
    gql(m, {"id": pod_id})
    return True


def wait_until_running(pod_id: str, timeout_s: int = 600) -> dict:
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout_s:
        p = pod(pod_id)
        last = p
        status = p.get("desiredStatus")
        rt = p.get("runtime") or {}
        has_ports = bool(rt.get("ports"))
        if status == "RUNNING" and has_ports:
            return p
        time.sleep(5)
    raise TimeoutError(f"Pod {pod_id} not running after {timeout_s}s; last={last}")


def _ssh_info(p: dict) -> tuple[str, int] | None:
    rt = p.get("runtime") or {}
    for port in rt.get("ports") or []:
        if port.get("privatePort") == 22 and port.get("isIpPublic"):
            return port["ip"], port["publicPort"]
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("myself")

    p_ls = sub.add_parser("pod")
    p_ls.add_argument("--id", required=True)

    p_deploy = sub.add_parser("deploy")
    p_deploy.add_argument("--name", required=True)
    p_deploy.add_argument("--gpu-type", required=True)
    p_deploy.add_argument("--gpu-count", type=int, default=1)
    p_deploy.add_argument("--container-disk", type=int, default=200)
    p_deploy.add_argument("--volume", type=int, default=50)
    p_deploy.add_argument("--image", required=True)
    p_deploy.add_argument("--env-file", help="Path to JSON dict of env vars")
    p_deploy.add_argument("--ports", default="22/tcp,8000/http")
    p_deploy.add_argument("--cloud", default="SECURE")

    p_wait = sub.add_parser("wait")
    p_wait.add_argument("--id", required=True)
    p_wait.add_argument("--timeout", type=int, default=600)

    p_term = sub.add_parser("terminate")
    p_term.add_argument("--id", required=True)

    p_stop = sub.add_parser("stop")
    p_stop.add_argument("--id", required=True)

    p_ssh = sub.add_parser("ssh-info")
    p_ssh.add_argument("--id", required=True)

    args = ap.parse_args()

    if args.cmd == "myself":
        print(json.dumps(myself(), indent=2))
    elif args.cmd == "pod":
        print(json.dumps(pod(args.id), indent=2))
    elif args.cmd == "deploy":
        env = json.loads(Path(args.env_file).read_text()) if args.env_file else {}
        result = find_and_deploy(
            name=args.name,
            gpu_type_id=args.gpu_type,
            gpu_count=args.gpu_count,
            container_disk_gb=args.container_disk,
            volume_gb=args.volume,
            image_name=args.image,
            env=env,
            ports=args.ports,
            cloud_type=args.cloud,
        )
        print(json.dumps(result, indent=2))
    elif args.cmd == "wait":
        print(json.dumps(wait_until_running(args.id, args.timeout), indent=2))
    elif args.cmd == "terminate":
        print(json.dumps({"ok": terminate(args.id)}))
    elif args.cmd == "stop":
        print(json.dumps({"ok": stop(args.id)}))
    elif args.cmd == "ssh-info":
        p = pod(args.id)
        info = _ssh_info(p)
        if info:
            ip, port = info
            print(f"ssh -i .agent_run/ssh/id_ed25519 -p {port} root@{ip}")
        else:
            print("SSH port not yet available", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
