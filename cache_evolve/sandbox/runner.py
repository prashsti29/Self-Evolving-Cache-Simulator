from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from cache_evolve.evaluator import score

from .loader import load_policy_class
from .validator import PolicyValidationError, validate_policy_source


def _linux_preexec(limits: SandboxLimits):
    if not sys.platform.startswith("linux"):
        return None

    def _apply() -> None:
        import resource

        cpu = max(1, int(limits.cpu_seconds))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
        mem = limits.memory_mb * 1024 * 1024
        if hasattr(resource, "RLIMIT_AS"):
            resource.setrlimit(resource.RLIMIT_AS, (mem, mem))

    return _apply


@dataclass
class SandboxLimits:
    cpu_seconds: float = 2.0
    wall_seconds: float = 5.0
    memory_mb: int = 128


@dataclass
class SandboxResult:
    ok: bool
    hit_rate: float | None = None
    error: str | None = None
    traceback: str | None = None


def _worker_script() -> str:
    return """
import json, sys, traceback
from cache_evolve.sandbox.loader import load_policy_class
from cache_evolve.evaluator import score

payload = json.loads(sys.stdin.read())
try:
    cls = load_policy_class(payload["source"])
    hr = score(cls, payload["trace"], payload["capacity"])
    print(json.dumps({"ok": True, "hit_rate": hr}))
except Exception as exc:
    print(json.dumps({"ok": False, "error": str(exc), "traceback": traceback.format_exc()}))
"""


def run_policy_in_subprocess(
    source: str,
    trace: Sequence[object],
    capacity: int,
    limits: SandboxLimits | None = None,
) -> SandboxResult:
    limits = limits or SandboxLimits()
    try:
        validate_policy_source(source)
    except PolicyValidationError as exc:
        return SandboxResult(ok=False, error=str(exc))

    payload = {"source": source, "trace": list(trace), "capacity": capacity}
    cmd = [sys.executable, "-c", _worker_script()]
    preexec = _linux_preexec(limits)
    try:
        proc = subprocess.run(
            cmd,
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=limits.wall_seconds,
            preexec_fn=preexec,
        )
    except subprocess.TimeoutExpired:
        return SandboxResult(ok=False, error="wall-clock limit exceeded")

    if proc.returncode != 0:
        return SandboxResult(ok=False, error=proc.stderr.strip() or "subprocess failed", traceback=proc.stderr)

    try:
        data = json.loads(proc.stdout.strip() or "{}")
    except json.JSONDecodeError:
        return SandboxResult(ok=False, error="invalid worker output", traceback=proc.stdout)

    if data.get("ok"):
        return SandboxResult(ok=True, hit_rate=float(data["hit_rate"]))
    return SandboxResult(ok=False, error=data.get("error"), traceback=data.get("traceback"))


def run_policy_in_process(source: str, trace: Sequence[object], capacity: int) -> SandboxResult:
    try:
        cls = load_policy_class(source)
        hr = score(cls, trace, capacity)
        return SandboxResult(ok=True, hit_rate=hr)
    except Exception as exc:
        import traceback

        return SandboxResult(ok=False, error=str(exc), traceback=traceback.format_exc())
