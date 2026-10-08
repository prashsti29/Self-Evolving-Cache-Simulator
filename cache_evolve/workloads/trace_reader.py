from __future__ import annotations

from pathlib import Path


def read_trace_file(path: str | Path) -> list[int]:
    """Read a public integer trace (one decimal key per line; # comments allowed)."""
    path = Path(path)
    keys: list[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        keys.append(int(stripped.split()[0]))
    if not keys:
        raise ValueError(f"trace file is empty: {path}")
    return keys
