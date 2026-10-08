from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, Optional

from .policy import Policy


class Cache:
    """Simple in-memory cache with pluggable eviction policy.

    This is intentionally minimal for commit 1: a deterministic access loop,
    hit/miss accounting, and a policy interface. The actual eviction algorithms
    are introduced in later commits.
    """

    def __init__(self, capacity: int, policy: Optional[Policy] = None):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.policy = policy
        self._store: OrderedDict[Any, None] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.accesses = 0

    @property
    def size(self) -> int:
        return len(self._store)

    def contains(self, key: Any) -> bool:
        return key in self._store

    def access(self, key: Any) -> str:
        self.accesses += 1

        if key in self._store:
            self.hits += 1
            if self.policy is not None:
                self.policy.on_hit(key)
            self._store.move_to_end(key)
            return "hit"

        self.misses += 1
        if self.policy is not None:
            self.policy.on_miss(key)

        if self.capacity <= 0:
            raise ValueError("capacity must be positive")

        if len(self._store) >= self.capacity:
            if self.policy is None:
                self._store.popitem(last=False)
            else:
                victim = self.policy.evict()
                if victim in self._store:
                    del self._store[victim]
                else:
                    # Guard against invalid or stale eviction keys.
                    self._store.popitem(last=False)

        self._store[key] = None
        return "miss"

    def hit_rate(self) -> float:
        if self.accesses == 0:
            return 0.0
        return self.hits / self.accesses

    def snapshot(self) -> Dict[str, Any]:
        return {
            "capacity": self.capacity,
            "size": self.size,
            "hits": self.hits,
            "misses": self.misses,
            "accesses": self.accesses,
            "hit_rate": self.hit_rate(),
            "keys": list(self._store.keys()),
        }

    def reset(self) -> None:
        self._store.clear()
        self.hits = 0
        self.misses = 0
        self.accesses = 0
