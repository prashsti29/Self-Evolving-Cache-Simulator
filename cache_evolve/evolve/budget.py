from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EvaluationBudget:
    limit: int
    used: int = 0
    rejected: int = 0

    def consume(self) -> bool:
        if self.used >= self.limit:
            return False
        self.used += 1
        return True

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)
