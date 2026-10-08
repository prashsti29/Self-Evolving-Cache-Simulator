from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class CandidateRecord:
    source: str
    hit_rate: float
    method: str
    generation: int = 0
    metadata: dict[str, Any] | None = None


@dataclass
class Archive:
    path: Path
    top_n: int = 20
    entries: list[CandidateRecord] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.entries is None:
            self.entries = self._load()

    def _load(self) -> list[CandidateRecord]:
        if not self.path.exists():
            return []
        data = json.loads(self.path.read_text(encoding="utf-8"))
        return [CandidateRecord(**row) for row in data.get("entries", [])]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"entries": [asdict(e) for e in self.entries[: self.top_n]]}
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def add(self, record: CandidateRecord) -> None:
        self.entries.append(record)
        self.entries.sort(key=lambda e: e.hit_rate, reverse=True)
        self.entries = self.entries[: self.top_n]
        self.save()

    def sample_parents(self, k: int, rng) -> list[CandidateRecord]:
        if not self.entries:
            return []
        k = min(k, len(self.entries))
        return rng.sample(self.entries, k=k)
