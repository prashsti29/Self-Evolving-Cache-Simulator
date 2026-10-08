from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class RollingMonitor:
    window_size: int = 256
    threshold_drop: float = 0.05
    cooldown_intervals: int = 3
    rollback_intervals: int = 2
    _live_hits: deque[int] = field(default_factory=deque)
    _shadow_hits: deque[int] = field(default_factory=deque)
    _cooldown: int = 0
    _bad_streak: int = 0
    rollback_requested: bool = False
    evolve_requested: bool = False

    def observe_pair(self, live_hit: bool, shadow_hit: bool) -> None:
        self._live_hits.append(1 if live_hit else 0)
        self._shadow_hits.append(1 if shadow_hit else 0)
        if len(self._live_hits) > self.window_size:
            self._live_hits.popleft()
            self._shadow_hits.popleft()

        if len(self._live_hits) < self.window_size:
            return

        live_rate = sum(self._live_hits) / len(self._live_hits)
        shadow_rate = sum(self._shadow_hits) / len(self._shadow_hits)
        if live_rate + self.threshold_drop < shadow_rate:
            self._bad_streak += 1
        else:
            self._bad_streak = 0

        if self._bad_streak >= self.rollback_intervals:
            self.rollback_requested = True
            self._bad_streak = 0
            self._cooldown = self.cooldown_intervals

    def tick_phase_shift(self) -> None:
        if self._cooldown > 0:
            self._cooldown -= 1
            return
        self.evolve_requested = True
