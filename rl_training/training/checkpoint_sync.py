"""Cloud-sync of LoRA checkpoints for GRPO training.

``CloudSyncCallback`` is a ``transformers.TrainerCallback`` that uploads the
most recent checkpoint directory (``<output_dir>/checkpoint-<step>``) to a
cloud bucket after every save, and prunes everything except the last N
remote checkpoints.

Supported backends:

  * **b2** (default): Backblaze B2 via the ``b2sdk`` package. Uses env vars
    ``B2_APPLICATION_KEY_ID`` and ``B2_APPLICATION_KEY``.
  * **s3**: AWS S3 via ``boto3``. Uses standard AWS env vars.

If the required SDK isn't installed or credentials are missing, the callback
logs a warning once and turns itself into a no-op for the rest of training -
we never want a bad credential to crash an RL run.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    from transformers import TrainerCallback
except ImportError:  # pragma: no cover - transformers missing in lightweight envs
    class TrainerCallback:  # type: ignore[no-redef]
        """Stub used when transformers isn't importable (local unit testing)."""


class _CloudBackend:
    """Abstract interface for a bucket uploader."""

    def upload_directory(self, local_dir: str, remote_prefix: str) -> None: ...
    def list_remote_checkpoints(self, remote_prefix: str) -> list[str]: ...
    def delete_remote_prefix(self, remote_prefix: str) -> None: ...
    def download_directory(self, remote_prefix: str, local_dir: str) -> None: ...


class _B2Backend(_CloudBackend):
    def __init__(self, bucket_name: str):
        try:
            from b2sdk.v2 import InMemoryAccountInfo, B2Api
        except ImportError as exc:
            raise RuntimeError("b2sdk not installed; run: pip install b2sdk") from exc
        key_id = os.environ.get("B2_APPLICATION_KEY_ID")
        app_key = os.environ.get("B2_APPLICATION_KEY")
        if not (key_id and app_key):
            raise RuntimeError(
                "B2_APPLICATION_KEY_ID and B2_APPLICATION_KEY env vars required"
            )
        info = InMemoryAccountInfo()
        self._api = B2Api(info)
        self._api.authorize_account("production", key_id, app_key)
        self._bucket = self._api.get_bucket_by_name(bucket_name)

    def upload_directory(self, local_dir: str, remote_prefix: str) -> None:
        for path in Path(local_dir).rglob("*"):
            if path.is_file():
                rel = path.relative_to(local_dir).as_posix()
                key = f"{remote_prefix.rstrip('/')}/{rel}"
                self._bucket.upload_local_file(
                    local_file=str(path), file_name=key,
                )

    def list_remote_checkpoints(self, remote_prefix: str) -> list[str]:
        seen: set[str] = set()
        for file_info, _ in self._bucket.ls(remote_prefix, recursive=True, latest_only=True):
            name = file_info.file_name
            rest = name[len(remote_prefix):].lstrip("/")
            top = rest.split("/", 1)[0]
            if top.startswith("checkpoint-"):
                seen.add(top)
        return sorted(seen, key=lambda n: int(n.split("-")[-1]))

    def delete_remote_prefix(self, remote_prefix: str) -> None:
        for file_info, _ in self._bucket.ls(remote_prefix, recursive=True, latest_only=False):
            self._api.delete_file_version(file_info.id_, file_info.file_name)

    def download_directory(self, remote_prefix: str, local_dir: str) -> None:
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        for file_info, _ in self._bucket.ls(remote_prefix, recursive=True, latest_only=True):
            rest = file_info.file_name[len(remote_prefix):].lstrip("/")
            local_file = Path(local_dir) / rest
            local_file.parent.mkdir(parents=True, exist_ok=True)
            downloaded = self._bucket.download_file_by_name(file_info.file_name)
            with open(local_file, "wb") as f:
                downloaded.save(f)


class _S3Backend(_CloudBackend):
    def __init__(self, bucket_name: str):
        try:
            import boto3
        except ImportError as exc:
            raise RuntimeError("boto3 not installed; run: pip install boto3") from exc
        self._s3 = boto3.client("s3")
        self._bucket = bucket_name

    def upload_directory(self, local_dir: str, remote_prefix: str) -> None:
        for path in Path(local_dir).rglob("*"):
            if path.is_file():
                rel = path.relative_to(local_dir).as_posix()
                key = f"{remote_prefix.rstrip('/')}/{rel}"
                self._s3.upload_file(str(path), self._bucket, key)

    def list_remote_checkpoints(self, remote_prefix: str) -> list[str]:
        seen: set[str] = set()
        paginator = self._s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self._bucket, Prefix=remote_prefix):
            for obj in page.get("Contents", []):
                rest = obj["Key"][len(remote_prefix):].lstrip("/")
                top = rest.split("/", 1)[0]
                if top.startswith("checkpoint-"):
                    seen.add(top)
        return sorted(seen, key=lambda n: int(n.split("-")[-1]))

    def delete_remote_prefix(self, remote_prefix: str) -> None:
        paginator = self._s3.get_paginator("list_objects_v2")
        keys = []
        for page in paginator.paginate(Bucket=self._bucket, Prefix=remote_prefix):
            for obj in page.get("Contents", []):
                keys.append({"Key": obj["Key"]})
        if keys:
            self._s3.delete_objects(
                Bucket=self._bucket, Delete={"Objects": keys},
            )

    def download_directory(self, remote_prefix: str, local_dir: str) -> None:
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        paginator = self._s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self._bucket, Prefix=remote_prefix):
            for obj in page.get("Contents", []):
                rest = obj["Key"][len(remote_prefix):].lstrip("/")
                if not rest:
                    continue
                local_file = Path(local_dir) / rest
                local_file.parent.mkdir(parents=True, exist_ok=True)
                self._s3.download_file(self._bucket, obj["Key"], str(local_file))


def make_backend(name: str, bucket: str) -> _CloudBackend:
    if name == "b2":
        return _B2Backend(bucket)
    if name == "s3":
        return _S3Backend(bucket)
    raise ValueError(f"Unknown cloud sync backend: {name!r}")


class CloudSyncCallback(TrainerCallback):
    """Upload checkpoints + progress log to a cloud bucket after each save.

    Args:
        backend: ``"b2"`` or ``"s3"``.
        bucket: bucket name.
        prefix: remote path prefix (e.g. ``"qwen3_32b_grpo/clinical"``).
        keep_last: number of remote checkpoints to keep; older ones pruned.
        progress_jsonl: optional path to a JSONL progress log that is also
            uploaded (to ``<prefix>/progress.jsonl``) on every save.
    """

    def __init__(
        self,
        backend: str,
        bucket: str,
        prefix: str,
        keep_last: int = 3,
        progress_jsonl: str | None = None,
    ):
        self.backend_name = backend
        self.bucket = bucket
        self.prefix = prefix.rstrip("/")
        self.keep_last = keep_last
        self.progress_jsonl = progress_jsonl
        self._backend: _CloudBackend | None = None
        self._disabled = False
        try:
            self._backend = make_backend(backend, bucket)
        except Exception as exc:
            logger.warning(
                "CloudSyncCallback disabled (backend init failed): %s", exc,
            )
            self._disabled = True

    # ---- TrainerCallback hooks

    def on_save(self, args, state, control, **kwargs: Any):  # noqa: D401
        if self._disabled or self._backend is None:
            return control
        ckpt_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if not ckpt_dir.is_dir():
            logger.warning("on_save: checkpoint dir missing: %s", ckpt_dir)
            return control
        remote = f"{self.prefix}/checkpoint-{state.global_step}"
        t0 = time.time()
        try:
            self._backend.upload_directory(str(ckpt_dir), remote)
            dt = time.time() - t0
            logger.info("Uploaded checkpoint-%d to %s/%s (%.1fs)",
                        state.global_step, self.bucket, remote, dt)
        except Exception as exc:
            logger.warning("Checkpoint upload failed: %s", exc)
            return control
        self._prune_remote()
        self._maybe_upload_progress()
        return control

    def on_log(self, args, state, control, **kwargs: Any):  # noqa: D401
        if self._disabled or self._backend is None or not self.progress_jsonl:
            return control
        # Rate-limit: upload progress every N log events (here: every step,
        # but it's tiny JSONL so cost is negligible).
        self._maybe_upload_progress()
        return control

    def _prune_remote(self) -> None:
        if self._backend is None:
            return
        try:
            names = self._backend.list_remote_checkpoints(self.prefix + "/")
            if len(names) <= self.keep_last:
                return
            to_delete = names[: len(names) - self.keep_last]
            for name in to_delete:
                self._backend.delete_remote_prefix(f"{self.prefix}/{name}")
                logger.info("Pruned remote checkpoint %s", name)
        except Exception as exc:
            logger.warning("Checkpoint pruning failed: %s", exc)

    def _maybe_upload_progress(self) -> None:
        if (
            self._backend is None
            or not self.progress_jsonl
            or not os.path.exists(self.progress_jsonl)
        ):
            return
        # Upload just the single file to <prefix>/progress.jsonl
        remote = f"{self.prefix}/progress.jsonl"
        try:
            # Use the "directory upload" path with a temp dir shim: simpler is
            # direct file ops, but we keep the interface minimal.
            parent = str(Path(self.progress_jsonl).parent)
            name = Path(self.progress_jsonl).name
            single_dir = Path(parent) / f"__progress_staging__"
            single_dir.mkdir(parents=True, exist_ok=True)
            staged = single_dir / name
            try:
                staged.write_bytes(Path(self.progress_jsonl).read_bytes())
                self._backend.upload_directory(
                    str(single_dir), f"{self.prefix}/__staging__",
                )
                # Rename-like behaviour: overwrite the canonical key by
                # re-uploading just this file with a single-file directory.
                self._backend.upload_directory(
                    str(single_dir), self.prefix,
                )
            finally:
                if staged.exists():
                    staged.unlink()
                if single_dir.exists():
                    single_dir.rmdir()
        except Exception as exc:
            logger.debug("progress upload skipped: %s", exc)
