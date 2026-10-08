from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class PromptReplayCache:
    """Deterministic replay cache for LLM responses.

    Every prompt is hashed before the response is stored. A replay run can then
    fetch the exact cached output without re-contacting the model.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._cache = self._load()

    def _load(self) -> dict[str, str]:
        if not self.path.exists():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        if isinstance(payload, dict):
            return {str(k): str(v) for k, v in payload.items()}
        return {}

    def _save(self) -> None:
        self.path.write_text(json.dumps(self._cache, indent=2, sort_keys=True), encoding="utf-8")

    @staticmethod
    def hash_prompt(prompt: str) -> str:
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    def get(self, prompt: str) -> str | None:
        return self._cache.get(self.hash_prompt(prompt))

    def set(self, prompt: str, response: str) -> str:
        key = self.hash_prompt(prompt)
        self._cache[key] = response
        self._save()
        return key

    def store(self, prompt: str, response: str) -> str:
        return self.set(prompt, response)

    def __contains__(self, prompt: str) -> bool:
        return self.hash_prompt(prompt) in self._cache

    def entries(self) -> dict[str, str]:
        return dict(self._cache)

    def clear(self) -> None:
        self._cache.clear()
        self._save()
