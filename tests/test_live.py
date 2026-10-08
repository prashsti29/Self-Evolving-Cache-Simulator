from __future__ import annotations

from cache_evolve.live import Deployer, LiveController, RollingMonitor
from cache_evolve.sim.baselines import LRUPolicy, TinyLFUPolicy


def test_phase_shift_deploy_and_rollback():
    deployer = Deployer(capacity=4, cold_start=True)
    monitor = RollingMonitor(window_size=8, rollback_intervals=2, threshold_drop=0.0)
    ctrl = LiveController(deployer, monitor)
    deployer.deploy(TinyLFUPolicy)

    keys = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 9, 8, 7, 6, 5, 4]
    ctrl.run_window(keys[:8])
    monitor.tick_phase_shift()
    assert monitor.evolve_requested
    ctrl.run_window(keys[8:], new_policy=LRUPolicy)

    # Force poor live performance vs shadow to trigger rollback.
    monitor._bad_streak = monitor.rollback_intervals
    monitor.observe_pair(False, True)
    monitor.observe_pair(False, True)
    assert monitor.rollback_requested
    deployer.rollback_to_lru()
    assert isinstance(deployer.live.policy, LRUPolicy)
