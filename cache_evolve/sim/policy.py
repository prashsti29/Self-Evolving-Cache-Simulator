from __future__ import annotations

from abc import ABC, abstractmethod


class Policy(ABC):
    """Interface for cache eviction policies.

    The simulator calls these hooks as it processes cache accesses. LLM-generated
    policies are expected to implement the same contract.
    """

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)

    @abstractmethod
    def on_hit(self, key: object) -> None:
        """Called every time a key is accessed and already in cache."""

    @abstractmethod
    def on_miss(self, key: object) -> None:
        """Called every time a key is accessed and missing from cache."""

    @abstractmethod
    def evict(self) -> object:
        """Return the key to evict, or raise if none is available."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(capacity={self.capacity})"


class NoopPolicy(Policy):
    """Minimal placeholder useful for skeleton tests and smoke validation."""

    def __init__(self, capacity: int):
        super().__init__(capacity)
        self._order: list[object] = []

    def on_hit(self, key: object) -> None:
        if key in self._order:
            self._order.remove(key)
            self._order.append(key)

    def on_miss(self, key: object) -> None:
        self._order.append(key)

    def evict(self) -> object:
        if not self._order:
            raise RuntimeError("No eviction candidate available in NoopPolicy")
        victim = self._order.pop(0)
        return victim
