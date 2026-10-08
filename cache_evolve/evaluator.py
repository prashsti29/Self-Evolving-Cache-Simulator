from __future__ import annotations

from collections import Counter
from typing import Callable, Sequence, Type

from .sim.cache import Cache
from .sim.policy import Policy


def score(policy: Type[Policy] | Callable[[int], Policy], trace_window: Sequence[object], capacity: int) -> float:
    if isinstance(policy, type):
        inst = policy(capacity)
    else:
        inst = policy(capacity)
    cache = Cache(capacity, inst)
    for key in trace_window:
        cache.access(key)
    return cache.hit_rate()


def time_ordered_split(trace: Sequence[object], train_ratio: float = 0.7) -> tuple[list[object], list[object]]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1")
    cut = int(len(trace) * train_ratio)
    if cut <= 0 or cut >= len(trace):
        raise ValueError("split produced empty segment")
    return list(trace[:cut]), list(trace[cut:])


def phase_ordered_split(trace: Sequence[object], phase_len: int) -> tuple[list[object], list[object]]:
    if phase_len <= 0 or phase_len >= len(trace):
        raise ValueError("invalid phase_len")
    cut = (len(trace) // phase_len) * phase_len // 2
    cut = max(phase_len, cut)
    if cut >= len(trace):
        cut = len(trace) - phase_len
    return list(trace[:cut]), list(trace[cut:])


def trace_summary_stats(trace: Sequence[object]) -> dict[str, float]:
    if not trace:
        return {
            "skew": 0.0,
            "one_hit_wonder_pct": 0.0,
            "scan_length": 0.0,
            "reuse_p50": 0.0,
            "reuse_p90": 0.0,
            "reuse_p99": 0.0,
        }

    counts = Counter(trace)
    freqs = sorted(counts.values(), reverse=True)
    total = sum(freqs)
    skew = freqs[0] / total if freqs else 0.0
    one_hit = sum(1 for c in counts.values() if c == 1) / len(counts)

    scan_len = 1
    best_scan = 1
    prev = trace[0]
    for key in trace[1:]:
        if key == prev + 1 or key == prev:
            scan_len += 1
            best_scan = max(best_scan, scan_len)
        else:
            scan_len = 1
        prev = key

    last_seen: dict[object, int] = {}
    gaps: list[int] = []
    for i, key in enumerate(trace):
        if key in last_seen:
            gaps.append(i - last_seen[key])
        last_seen[key] = i

    gaps.sort()

    def pct(p: float) -> float:
        if not gaps:
            return 0.0
        idx = min(len(gaps) - 1, int(p * len(gaps)))
        return float(gaps[idx])

    return {
        "skew": float(skew),
        "one_hit_wonder_pct": float(one_hit),
        "scan_length": float(best_scan),
        "reuse_p50": pct(0.50),
        "reuse_p90": pct(0.90),
        "reuse_p99": pct(0.99),
    }
