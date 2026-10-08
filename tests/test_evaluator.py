from __future__ import annotations

from cache_evolve.evaluator import phase_ordered_split, score, time_ordered_split, trace_summary_stats
from cache_evolve.sim.baselines import LRUPolicy


def test_time_ordered_split_not_shuffled():
    trace = list(range(100))
    train, held = time_ordered_split(trace, 0.7)
    assert train == list(range(70))
    assert held == list(range(70, 100))


def test_phase_ordered_split():
    trace = list(range(1000))
    train, held = phase_ordered_split(trace, phase_len=100)
    assert len(train) + len(held) == len(trace)
    assert train[-1] < held[0]


def test_score_and_summary():
    trace = [1, 1, 2, 3, 2, 2]
    hr = score(LRUPolicy, trace, capacity=2)
    stats = trace_summary_stats(trace)
    assert 0.0 <= hr <= 1.0
    assert "skew" in stats and "reuse_p50" in stats
