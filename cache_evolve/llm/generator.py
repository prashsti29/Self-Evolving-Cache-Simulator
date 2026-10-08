from __future__ import annotations

from dataclasses import dataclass, field

from cache_evolve.evaluator import score, trace_summary_stats
from cache_evolve.sandbox.runner import SandboxResult, run_policy_in_subprocess
from cache_evolve.sim.baselines import BASELINE_POLICIES

from .client import LLMClient
from .parser import extract_python_class
from .prompt import build_prompt


@dataclass
class GenerationAttempt:
    prompt: str
    raw_response: str | None = None
    source: str | None = None
    sandbox: SandboxResult | None = None
    hit_rate: float | None = None
    error: str | None = None


@dataclass
class GenerationResult:
    ok: bool
    source: str | None = None
    hit_rate: float | None = None
    attempts: list[GenerationAttempt] = field(default_factory=list)


def _baseline_scores(trace, capacity: int) -> dict[str, float]:
    return {name: score(cls, trace, capacity) for name, cls in BASELINE_POLICIES.items()}


def generate_policy(
    client: LLMClient,
    *,
    policy_code: str,
    trace,
    capacity: int,
    max_retries: int = 3,
) -> GenerationResult:
    summary = trace_summary_stats(trace)
    baselines = _baseline_scores(trace, capacity)
    attempts: list[GenerationAttempt] = []
    feedback = ""

    for _ in range(max_retries):
        prompt = build_prompt(policy_code, baselines, summary)
        if feedback:
            prompt += f"\nPrevious failure:\n{feedback}\n"
        attempt = GenerationAttempt(prompt=prompt)
        try:
            raw = client.complete(prompt)
        except Exception as exc:
            attempt.error = str(exc)
            attempts.append(attempt)
            feedback = str(exc)
            continue

        attempt.raw_response = raw
        source = extract_python_class(raw)
        attempt.source = source
        sb = run_policy_in_subprocess(source, trace, capacity)
        attempt.sandbox = sb
        if not sb.ok:
            attempt.error = sb.error
            feedback = f"{sb.error}\n{sb.traceback or ''}"
            attempts.append(attempt)
            continue

        attempt.hit_rate = sb.hit_rate
        attempts.append(attempt)
        return GenerationResult(ok=True, source=source, hit_rate=sb.hit_rate, attempts=attempts)

    return GenerationResult(ok=False, attempts=attempts)
