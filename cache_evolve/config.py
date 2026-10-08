from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None


@dataclass
class ExperimentConfig:
    """Single source of truth for deterministic experiments.

    This intentionally keeps the simulator deterministic, while LLM output is
    replayed from a prompt-hash cache instead of re-generated during replay runs.
    """

    seed: int = 42
    capacity: int = 128
    trace_length: int = 10_000
    max_candidates: int = 50
    llm_replay_cache_path: str = "llm_replay_cache.json"
    evaluation_budget: int = 100
    cost_cap_usd: float = 5.0
    zipf_alpha: float = 1.2
    workload_universe: int = 10_000
    # Commit 7: every candidate evaluation counts toward budget, including sandbox rejects.
    # Commit 8 rollback baseline: shadow LRU on the same live window (implemented in monitor).

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentConfig":
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "ExperimentConfig":
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(payload)

    @classmethod
    def load_file(cls, path: str | Path) -> "ExperimentConfig":
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        suffix = path.suffix.lower()

        if suffix in {".json"}:
            return cls.from_dict(json.loads(text))
        if suffix in {".yaml", ".yml"}:
            if yaml is None:
                raise RuntimeError("PyYAML is required to load YAML config files.")
            return cls.from_dict(yaml.safe_load(text) or {})
        raise ValueError(f"Unsupported config format: {path.suffix!r}")

    def apply_seed(self) -> None:
        """Seed Python's random module for deterministic workload generation."""
        random.seed(self.seed)

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        if self.trace_length <= 0:
            raise ValueError("trace_length must be positive")
        if self.max_candidates <= 0:
            raise ValueError("max_candidates must be positive")
        if self.evaluation_budget <= 0:
            raise ValueError("evaluation_budget must be positive")
        if self.cost_cap_usd < 0:
            raise ValueError("cost_cap_usd must be non-negative")
