from __future__ import annotations

from typing import Callable, Iterable

TraceFn = Callable[[], list[int]]


def compose_phases(phases: Iterable[tuple[TraceFn, int]]) -> list[int]:
    """Concatenate workload segments; each fn generates one phase trace."""
    trace: list[int] = []
    for generator, length in phases:
        segment = generator()
        if len(segment) != length:
            raise ValueError(
                f"phase length mismatch: expected {length}, got {len(segment)}"
            )
        trace.extend(segment)
    return trace
