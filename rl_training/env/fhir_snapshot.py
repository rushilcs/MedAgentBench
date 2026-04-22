"""FHIR snapshot recorder/replayer.

During GRPO rollouts, the environment makes many GET requests against the FHIR
server. Hitting a live server from a cloud training box is slow (network RTT),
flaky (timeout retries), and tightly couples training throughput to an external
service.

This module lets us:
  1. **Record** all GETs issued during one or more rollouts and serialize the
     (url -> response) mapping to a JSONL file.
  2. **Replay** those cached responses during training, with optional fall-through
     to a live server for cache misses.

The snapshot is a drop-in replacement for the existing ``send_get_request``
helper: anywhere that function is called (``MedAgentBenchEnv.get_fhir_resource``,
``refsol`` graders, ``task_generator``), callers can route through a
``FhirSnapshot`` instance instead.

Canonicalization: URLs are normalized before lookup so equivalent queries
(different parameter ordering, presence/absence of ``&_format=json``) hit the
same cache row.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import requests


def _canonicalize_url(url: str) -> str:
    """Return a canonical form of a FHIR URL.

    - Keeps scheme + netloc + path
    - Sorts query parameters alphabetically
    - Strips ``_format=json`` so presence/absence does not create duplicate keys
    - Drops any trailing empty fragment
    """
    parsed = urlparse(url)
    params = sorted(
        (k, v) for (k, v) in parse_qsl(parsed.query, keep_blank_values=True)
        if k != "_format"
    )
    return urlunparse((
        parsed.scheme, parsed.netloc, parsed.path, parsed.params,
        urlencode(params), "",
    ))


@dataclass
class SnapshotEntry:
    """One recorded GET response."""
    url: str               # canonical URL (the lookup key)
    status_code: int
    data: Any              # parsed JSON or raw text; serializable
    content_type: str = "application/json"


class FhirSnapshot:
    """Record/replay store for FHIR GET responses.

    Three modes:
      - ``record``: every GET hits the live server and is cached (in-memory +
        appended to disk if ``path`` is set).
      - ``replay``: GETs are served from cache. On miss, if ``fallthrough`` is
        true the request hits the live server (and is cached); otherwise an
        error response is returned.
      - ``off``: bypass entirely (equivalent to calling ``requests.get``
        directly), but still records if ``path`` is set.

    Thread-safe: multiple rollout threads can share one instance.
    """

    def __init__(
        self,
        mode: str = "replay",
        path: str | None = None,
        fallthrough: bool = True,
        live_getter: Callable[[str], dict[str, Any]] | None = None,
        miss_log_path: str | None = None,
    ):
        if mode not in {"record", "replay", "off"}:
            raise ValueError(f"Unknown mode: {mode!r}")
        self.mode = mode
        self.path = path
        self.fallthrough = fallthrough
        self._live = live_getter or _default_live_getter
        self._cache: dict[str, SnapshotEntry] = {}
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._live_calls = 0
        self._loaded_from: str | None = None
        # Diagnostic: when set (or via FHIR_SNAPSHOT_MISS_LOG env), every cache
        # miss is appended as one JSON line. If avg_correct stays at 0 we read
        # this file to know exactly which URLs the policy is emitting that we
        # never recorded -- no guesswork required to expand the snapshot.
        self.miss_log_path = miss_log_path or os.environ.get("FHIR_SNAPSHOT_MISS_LOG")
        self._miss_log_lock = threading.Lock()
        self._miss_seen: set[str] = set()

        if path and mode in {"replay", "off"} and os.path.exists(path):
            self.load(path)

    # ------------------------------------------------------------------ IO

    def load(self, path: str) -> int:
        """Load cache entries from a JSONL file. Returns count loaded."""
        count = 0
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                entry = SnapshotEntry(
                    url=row["url"],
                    status_code=row.get("status_code", 200),
                    data=row.get("data"),
                    content_type=row.get("content_type", "application/json"),
                )
                with self._lock:
                    self._cache[entry.url] = entry
                count += 1
        self._loaded_from = path
        return count

    def save(self, path: str | None = None) -> int:
        """Write cache to a JSONL file (full dump). Returns count written."""
        target = path or self.path
        if not target:
            raise ValueError("save() requires a path (explicit or via constructor)")
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            entries = list(self._cache.values())
        with open(target, "w") as f:
            for entry in entries:
                f.write(json.dumps({
                    "url": entry.url,
                    "status_code": entry.status_code,
                    "data": entry.data,
                    "content_type": entry.content_type,
                }) + "\n")
        return len(entries)

    def _append_to_disk(self, entry: SnapshotEntry) -> None:
        if not self.path:
            return
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps({
                "url": entry.url,
                "status_code": entry.status_code,
                "data": entry.data,
                "content_type": entry.content_type,
            }) + "\n")

    # --------------------------------------------------------------- lookup

    def send_get_request(self, url: str) -> dict[str, Any]:
        """Drop-in replacement for ``src.server.tasks.medagentbench.utils.send_get_request``.

        Returns the same shape: ``{"status_code": int, "data": ...}`` on success
        or ``{"error": str}`` on failure.
        """
        key = _canonicalize_url(url)

        with self._lock:
            entry = self._cache.get(key)

        if entry is not None:
            self._hits += 1
            return {"status_code": entry.status_code, "data": entry.data}

        # Miss path: in replay-strict mode, return an error without hitting live.
        if self.mode == "replay" and not self.fallthrough:
            self._misses += 1
            self._log_miss(key, raw_url=url)
            return {"error": f"FHIR snapshot cache miss for: {key}"}

        # Hit the live server.
        self._live_calls += 1
        live = self._live(url)
        if "error" in live:
            self._misses += 1
            return live

        entry = SnapshotEntry(
            url=key,
            status_code=live.get("status_code", 200),
            data=live.get("data"),
            content_type="application/json",
        )
        with self._lock:
            self._cache[key] = entry

        # Persist in record mode (or any mode with a path + miss-fallthrough).
        if self.mode == "record" or self.path:
            self._append_to_disk(entry)

        self._misses += 1
        return {"status_code": entry.status_code, "data": entry.data}

    # ------------------------------------------------------------------ util

    def _log_miss(self, canonical_key: str, raw_url: str) -> None:
        """Append a one-line JSON record for one cache miss (deduped per key).

        No-op when ``miss_log_path`` is not set. Best-effort: never raises so
        that diagnostic plumbing can't fail a rollout.
        """
        if not self.miss_log_path:
            return
        with self._miss_log_lock:
            if canonical_key in self._miss_seen:
                return
            self._miss_seen.add(canonical_key)
            try:
                Path(self.miss_log_path).parent.mkdir(parents=True, exist_ok=True)
                with open(self.miss_log_path, "a") as f:
                    f.write(json.dumps({
                        "canonical": canonical_key,
                        "raw": raw_url,
                    }) + "\n")
            except Exception:
                pass

    def stats(self) -> dict[str, int]:
        return {
            "cache_size": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "live_calls": self._live_calls,
        }

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
        self._hits = 0
        self._misses = 0
        self._live_calls = 0


def _default_live_getter(url: str) -> dict[str, Any]:
    """Hit the FHIR server directly. Mirrors
    ``src.server.tasks.medagentbench.utils.send_get_request``.

    If FHIR_LIVE_BASE_OVERRIDE is set (e.g. a Cloudflare Tunnel pointing at a
    dev box's docker FHIR), rewrite localhost:8080/fhir on the wire so cache
    keys stay canonical (localhost) while the actual hop traverses the
    override. This makes both eval and trainer reach the same live FHIR with
    no per-call plumbing.
    """
    override = os.environ.get("FHIR_LIVE_BASE_OVERRIDE", "").strip()
    target = url
    if override and "localhost:8080/fhir" in url:
        target = url.replace(
            "http://localhost:8080/fhir", override.rstrip("/") + "/fhir"
        )
    try:
        resp = requests.get(target, timeout=30)
        resp.raise_for_status()
        ct = resp.headers.get("Content-Type", "")
        data = resp.json() if "json" in ct else resp.text
        return {"status_code": resp.status_code, "data": data}
    except Exception as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------- global singleton

_GLOBAL_SNAPSHOT: FhirSnapshot | None = None


def get_global_snapshot() -> FhirSnapshot | None:
    """Return the process-wide snapshot if one has been installed."""
    return _GLOBAL_SNAPSHOT


def install_global_snapshot(snapshot: FhirSnapshot | None) -> None:
    """Install a process-wide snapshot (or clear with ``None``).

    When set, modules that want to respect the snapshot can call
    ``get_global_snapshot()`` and route their GETs through it. This avoids
    having to thread the snapshot object through every constructor.
    """
    global _GLOBAL_SNAPSHOT
    _GLOBAL_SNAPSHOT = snapshot
