from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Type

from cache_evolve.sim.policy import Policy

from .deployer import Deployer
from .monitor import RollingMonitor


@dataclass
class LiveController:
    deployer: Deployer
    monitor: RollingMonitor

    def run_window(self, keys: Iterable[object], new_policy: Type[Policy] | None = None) -> dict[str, bool]:
        if new_policy is not None:
            self.deployer.deploy(new_policy)
            self.monitor.evolve_requested = False

        for key in keys:
            live, shadow = self.deployer.access(key)
            self.monitor.observe_pair(live == "hit", shadow == "hit")

        if self.monitor.rollback_requested:
            self.deployer.rollback_to_lru()
            self.monitor.rollback_requested = False

        return {
            "rollback": False,
            "evolve": self.monitor.evolve_requested,
        }
