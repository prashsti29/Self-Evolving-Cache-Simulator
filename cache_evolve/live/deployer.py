from __future__ import annotations

from dataclasses import dataclass
from typing import Type

from cache_evolve.sim.baselines import LRUPolicy
from cache_evolve.sim.cache import Cache
from cache_evolve.sim.policy import Policy


@dataclass
class Deployer:
    capacity: int
    cold_start: bool = True
    live: Cache | None = None
    shadow: Cache | None = None

    def deploy(self, policy_cls: Type[Policy]) -> None:
        # Default cold start: new cache state on each deploy.
        if self.cold_start or self.live is None:
            self.live = Cache(self.capacity, policy_cls(self.capacity))
        else:
            self.live = Cache(self.capacity, policy_cls(self.capacity))
        if self.shadow is None:
            self.shadow = Cache(self.capacity, LRUPolicy(self.capacity))

    def rollback_to_lru(self) -> None:
        self.deploy(LRUPolicy)

    def access(self, key: object) -> tuple[str, str]:
        if self.live is None or self.shadow is None:
            raise RuntimeError("deploy a policy first")
        shadow = self.shadow.access(key)
        live = self.live.access(key)
        return live, shadow
