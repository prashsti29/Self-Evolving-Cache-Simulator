from __future__ import annotations

from cache_evolve.sandbox import (
    PolicyValidationError,
    SandboxLimits,
    run_policy_in_process,
    run_policy_in_subprocess,
    validate_policy_source,
)

VALID = """
from cache_evolve.sim.policy import Policy
from collections import OrderedDict

class P(Policy):
    def __init__(self, capacity):
        super().__init__(capacity)
        self.o = OrderedDict()
    def on_hit(self, key):
        self.o.move_to_end(key)
    def on_miss(self, key):
        self.o[key] = None
    def evict(self):
        return self.o.popitem(last=False)[0]
"""


def test_valid_policy_runs():
    trace = [1, 2, 1, 3]
    sb = run_policy_in_process(VALID, trace, 2)
    assert sb.ok and sb.hit_rate is not None


def test_rejects_os_import():
    src = "import os\n" + VALID
    try:
        validate_policy_source(src)
        assert False
    except PolicyValidationError:
        pass


def test_infinite_loop_times_out():
    src = """
from cache_evolve.sim.policy import Policy
class P(Policy):
    def __init__(self, c):
        super().__init__(c)
    def on_hit(self, key):
        while True:
            pass
    def on_miss(self, key):
        pass
    def evict(self):
        return 0
"""
    sb = run_policy_in_subprocess(src, [1, 1], 1)
    assert not sb.ok


def test_invalid_eviction_still_runs_with_fallback():
    src = """
from cache_evolve.sim.policy import Policy
class P(Policy):
    def __init__(self, c):
        super().__init__(c)
        self.k = []
    def on_hit(self, key):
        pass
    def on_miss(self, key):
        self.k.append(key)
    def evict(self):
        return 999
"""
    sb = run_policy_in_process(src, [1, 2, 1], 1)
    assert sb.ok


def test_memory_bomb_rejected_or_killed():
    src = """
from cache_evolve.sim.policy import Policy
class P(Policy):
    def __init__(self, c):
        super().__init__(c)
        self.x = [0] * (10**7)
    def on_hit(self, key):
        pass
    def on_miss(self, key):
        pass
    def evict(self):
        return 0
"""
    sb = run_policy_in_subprocess(src, [1], 1, limits=SandboxLimits(memory_mb=8, wall_seconds=3))
    assert not sb.ok
