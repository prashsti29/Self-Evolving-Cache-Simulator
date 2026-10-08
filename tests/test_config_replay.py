from __future__ import annotations

import random
from pathlib import Path

from cache_evolve.config import ExperimentConfig
from cache_evolve.llm.replay import PromptReplayCache
from cache_evolve.sim import Cache, run_trace, LRUPolicy


def test_same_seed_same_lru_results():
    cfg = ExperimentConfig(seed=7, capacity=8, trace_length=200)
    cfg.apply_seed()
    a = [random.randrange(100) for _ in range(cfg.trace_length)]
    cfg.apply_seed()
    b = [random.randrange(100) for _ in range(cfg.trace_length)]
    assert a == b
    r1 = run_trace(a, cfg.capacity, LRUPolicy)
    r2 = run_trace(b, cfg.capacity, LRUPolicy)
    assert r1 == r2


def test_prompt_replay_cache_roundtrip(tmp_path: Path):
    path = tmp_path / "llm.json"
    cache = PromptReplayCache(path)
    prompt = "generate policy"
    response = "class P: pass"
    cache.store(prompt, response)
    reloaded = PromptReplayCache(path)
    assert reloaded.get(prompt) == response
