from __future__ import annotations

from cache_evolve.sim import (
    Cache,
    LRUPolicy,
    LFUPolicy,
    ARCPolicy,
    TinyLFUPolicy,
    belady_hit_rate,
    run_trace,
)


def test_lru_known_sequence():
    trace = [1, 1, 2, 2, 3, 3, 1, 1]
    rate = run_trace(trace, capacity=2, policy_cls=LRUPolicy)
    assert rate == 4 / 8


def test_lfu_favors_frequent_key():
    trace = [1, 1, 1, 2, 3, 2, 2, 2, 3, 3, 3, 1]
    rate = run_trace(trace, capacity=2, policy_cls=LFUPolicy)
    assert rate >= 0.5


def test_belady_optimal_on_small_trace():
    trace = [1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5]
    optimal = belady_hit_rate(trace, capacity=3)
    lru = run_trace(trace, capacity=3, policy_cls=LRUPolicy)
    assert optimal >= lru


def test_arc_and_tinylfu_smoke():
    trace = list(range(50)) * 3
    arc_rate = run_trace(trace, capacity=10, policy_cls=ARCPolicy)
    tinylfu_rate = run_trace(trace, capacity=10, policy_cls=TinyLFUPolicy)
    assert 0.0 <= arc_rate <= 1.0
    assert 0.0 <= tinylfu_rate <= 1.0


def test_cache_without_policy_uses_insertion_order():
    cache = Cache(2)
    cache.access("a")
    cache.access("b")
    cache.access("a")
    cache.access("c")
    assert cache.hits == 1
    assert cache.misses == 3
