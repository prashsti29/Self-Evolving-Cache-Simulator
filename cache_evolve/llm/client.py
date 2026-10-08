from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .replay import PromptReplayCache


class LLMClient(Protocol):
    def complete(self, prompt: str) -> str: ...


@dataclass
class ReplayLLMClient:
    cache: PromptReplayCache

    def complete(self, prompt: str) -> str:
        cached = self.cache.get(prompt)
        if cached is None:
            raise RuntimeError("LLM replay miss — tests/CI must not call live models")
        return cached


@dataclass
class RecordingLLMClient:
    """Wraps replay cache; records new responses when explicitly seeded."""

    cache: PromptReplayCache

    def complete(self, prompt: str) -> str:
        return self.cache.get(prompt) or ""
