from __future__ import annotations

import random
from pathlib import Path

from cache_evolve.config import ExperimentConfig
from cache_evolve.sim import LRUPolicy, run_trace
from cache_evolve.workloads import (
    burst_trace,
    compose_phases,
    read_trace_file,
    sequential_scan,
    shifting_hot_set,
    zipf_trace,
)

# Fixed benchmark (commit 3). Gap measured once with LRU; threshold = 80% of (zipf - scan).
_BENCH = ExperimentConfig()
_BENCH.apply_seed()
_rng = random.Random(_BENCH.seed)
_UNIVERSE = _BENCH.workload_universe
_ZIPF = zipf_trace(
    _BENCH.trace_length,
    _UNIVERSE,
    alpha=_BENCH.zipf_alpha,
    rng=_rng,
)
_SCAN = sequential_scan(_BENCH.trace_length, _UNIVERSE, rng=_rng)
_LRU_ZIPF = run_trace(_ZIPF, _BENCH.capacity, LRUPolicy)
_LRU_SCAN = run_trace(_SCAN, _BENCH.capacity, LRUPolicy)
_MIN_SCAN_ZIPF_GAP = 0.8 * (_LRU_ZIPF - _LRU_SCAN)


def test_zipf_vs_scan_lru_gap():
    assert _LRU_ZIPF - _LRU_SCAN >= _MIN_SCAN_ZIPF_GAP, (
        f"zipf={_LRU_ZIPF:.4f} scan={_LRU_SCAN:.4f} "
        f"gap={_LRU_ZIPF - _LRU_SCAN:.4f} need>={_MIN_SCAN_ZIPF_GAP:.4f}"
    )


def test_workload_generators_lengths():
    rng = random.Random(0)
    assert len(zipf_trace(100, 50, rng=rng)) == 100
    assert len(sequential_scan(100, 50, rng=rng)) == 100
    assert len(burst_trace(100, [1, 2, 3], rng=rng)) == 100
    assert len(shifting_hot_set(100, 64, rng=rng)) == 100


def test_compose_phases():
    rng = random.Random(1)
    trace = compose_phases(
        [
            (lambda: zipf_trace(50, 100, rng=rng), 50),
            (lambda: sequential_scan(50, 100, rng=rng), 50),
        ]
    )
    assert len(trace) == 100


def test_read_public_trace(tmp_path: Path):
    path = tmp_path / "trace.txt"
    path.write_text("# comment\n1\n2\n3\n", encoding="utf-8")
    assert read_trace_file(path) == [1, 2, 3]
