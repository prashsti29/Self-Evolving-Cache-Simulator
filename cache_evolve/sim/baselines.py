from __future__ import annotations

from collections import OrderedDict, defaultdict
from typing import Hashable, Iterable, Sequence, Type

from .policy import Policy


class LRUPolicy(Policy):
    def __init__(self, capacity: int):
        super().__init__(capacity)
        self._order: OrderedDict[Hashable, None] = OrderedDict()

    def on_hit(self, key: object) -> None:
        if key in self._order:
            self._order.move_to_end(key)

    def on_miss(self, key: object) -> None:
        self._order[key] = None

    def evict(self) -> object:
        if not self._order:
            raise RuntimeError("LRUPolicy: nothing to evict")
        key, _ = self._order.popitem(last=False)
        return key


class LFUPolicy(Policy):
    def __init__(self, capacity: int):
        super().__init__(capacity)
        self._freq: dict[object, int] = defaultdict(int)
        self._order: OrderedDict[Hashable, None] = OrderedDict()

    def on_hit(self, key: object) -> None:
        self._freq[key] += 1
        if key in self._order:
            self._order.move_to_end(key)

    def on_miss(self, key: object) -> None:
        self._freq[key] += 1
        self._order[key] = None

    def evict(self) -> object:
        if not self._order:
            raise RuntimeError("LFUPolicy: nothing to evict")
        min_freq = min(self._freq[k] for k in self._order)
        for key in self._order:
            if self._freq[key] == min_freq:
                del self._order[key]
                return key
        raise RuntimeError("LFUPolicy: eviction failed")


class TinyLFUPolicy(Policy):
    """Windowed frequency with LRU tie-break (TinyLFU-style eviction)."""

    def __init__(self, capacity: int, window: int = 10_000):
        super().__init__(capacity)
        self._window = max(1, int(window))
        self._recent: list[object] = []
        self._freq: dict[object, int] = defaultdict(int)
        self._order: OrderedDict[Hashable, None] = OrderedDict()

    def _touch_window(self, key: object) -> None:
        self._recent.append(key)
        self._freq[key] += 1
        if len(self._recent) > self._window:
            old = self._recent.pop(0)
            self._freq[old] -= 1
            if self._freq[old] <= 0:
                del self._freq[old]

    def on_hit(self, key: object) -> None:
        self._touch_window(key)
        if key in self._order:
            self._order.move_to_end(key)

    def on_miss(self, key: object) -> None:
        self._touch_window(key)
        self._order[key] = None

    def evict(self) -> object:
        if not self._order:
            raise RuntimeError("TinyLFUPolicy: nothing to evict")
        victim = min(self._order, key=lambda k: (self._freq.get(k, 0), list(self._order.keys()).index(k)))
        del self._order[victim]
        return victim


class ARCPolicy(Policy):
    """Adaptive Replacement Cache (T1/T2 with ghost lists B1/B2)."""

    def __init__(self, capacity: int):
        super().__init__(capacity)
        self.p = 0
        self.t1: OrderedDict[Hashable, None] = OrderedDict()
        self.t2: OrderedDict[Hashable, None] = OrderedDict()
        self.b1: OrderedDict[Hashable, None] = OrderedDict()
        self.b2: OrderedDict[Hashable, None] = OrderedDict()
        self._pending: object | None = None
        self._pending_in_t2 = False

    def _ghost(self, cache: OrderedDict[Hashable, None], key: object) -> None:
        cache[key] = None
        if len(cache) > self.capacity:
            cache.popitem(last=False)

    def _replace(self) -> object:
        if self.t1 and (not self.t2 or len(self.t1) > self.p):
            key, _ = self.t1.popitem(last=False)
            self._ghost(self.b1, key)
            return key
        if self.t2:
            key, _ = self.t2.popitem(last=False)
            self._ghost(self.b2, key)
            return key
        if self.t1:
            key, _ = self.t1.popitem(last=False)
            self._ghost(self.b1, key)
            return key
        raise RuntimeError("ARCPolicy: nothing to evict")

    def _insert_pending(self) -> None:
        if self._pending is None:
            return
        if self._pending_in_t2:
            self.t2[self._pending] = None
        else:
            self.t1[self._pending] = None
        self._pending = None

    def on_hit(self, key: object) -> None:
        if key in self.t1:
            del self.t1[key]
            self.t2[key] = None
        elif key in self.t2:
            self.t2.move_to_end(key)

    def on_miss(self, key: object) -> None:
        in_b1 = key in self.b1
        in_b2 = key in self.b2
        if in_b1:
            delta = max(1, len(self.b2) // max(len(self.b1), 1))
            self.p = min(self.capacity, self.p + delta)
            del self.b1[key]
        elif in_b2:
            delta = max(1, len(self.b1) // max(len(self.b2), 1))
            self.p = max(0, self.p - delta)
            del self.b2[key]

        self._pending = key
        self._pending_in_t2 = bool(in_b2)
        if len(self.t1) + len(self.t2) < self.capacity:
            self._insert_pending()

    def evict(self) -> object:
        victim = self._replace()
        self._insert_pending()
        return victim


def belady_hit_rate(trace: Sequence[Hashable], capacity: int) -> float:
    """Offline optimal (Belady MIN) — not deployable online."""
    if capacity <= 0:
        raise ValueError("capacity must be positive")
    if not trace:
        return 0.0

    cache: set[Hashable] = set()
    hits = 0
    n = len(trace)

    def next_use(idx: int, key: Hashable) -> int:
        for j in range(idx + 1, n):
            if trace[j] == key:
                return j
        return n + 1

    for i, key in enumerate(trace):
        if key in cache:
            hits += 1
            continue
        if len(cache) >= capacity:
            victim = max(cache, key=lambda k: next_use(i, k))
            cache.remove(victim)
        cache.add(key)

    return hits / len(trace)


def run_trace(trace: Iterable[Hashable], capacity: int, policy_cls: Type[Policy]) -> float:
    from .cache import Cache

    cache = Cache(capacity, policy_cls(capacity))
    for key in trace:
        cache.access(key)
    return cache.hit_rate()


BASELINE_POLICIES: dict[str, Type[Policy]] = {
    "lru": LRUPolicy,
    "lfu": LFUPolicy,
    "arc": ARCPolicy,
    "tinylfu": TinyLFUPolicy,
}
