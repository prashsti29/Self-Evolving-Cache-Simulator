from __future__ import annotations

from pathlib import Path

from cache_evolve.evaluator import trace_summary_stats
from cache_evolve.llm.client import ReplayLLMClient
from cache_evolve.llm.generator import generate_policy
from cache_evolve.llm.prompt import build_prompt
from cache_evolve.llm.replay import PromptReplayCache
from cache_evolve.sim.baselines import BASELINE_POLICIES

VALID = """
from cache_evolve.sim.policy import Policy
from collections import OrderedDict

class Evolved(Policy):
    def __init__(self, capacity):
        super().__init__(capacity)
        self.o = OrderedDict()
    def on_hit(self, key):
        if key in self.o:
            self.o.move_to_end(key)
    def on_miss(self, key):
        self.o[key] = None
    def evict(self):
        return self.o.popitem(last=False)[0]
"""


def test_generate_policy_replay_e2e(tmp_path: Path):
    trace = [1, 2, 1, 3, 2, 1]
    from cache_evolve.evaluator import score

    baselines = {n: score(c, trace, 2) for n, c in BASELINE_POLICIES.items()}
    prompt = build_prompt("# empty", baselines, trace_summary_stats(trace))
    cache = PromptReplayCache(tmp_path / "replay.json")
    cache.store(prompt, VALID)
    client = ReplayLLMClient(cache)
    result = generate_policy(client, policy_code="# empty", trace=trace, capacity=2, max_retries=2)
    assert result.ok and result.hit_rate is not None
    assert result.attempts
