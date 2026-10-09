from __future__ import annotations

import random
from dataclasses import dataclass

from cache_evolve.evaluator import score, time_ordered_split
from cache_evolve.llm.client import LLMClient
from cache_evolve.llm.parser import extract_python_class
from cache_evolve.llm.prompt import build_prompt
from cache_evolve.sandbox.runner import run_policy_in_subprocess
from cache_evolve.sandbox.validator import validate_policy_source
from cache_evolve.sim.baselines import BASELINE_POLICIES, LRUPolicy

from .archive import Archive, CandidateRecord
from .budget import EvaluationBudget
from .search import LRU_SOURCE, genetic_search, random_search, scan_heavy_replay_window


@dataclass
class EvolutionOutcome:
    method: str
    train_hit_rate: float | None
    held_out_hit_rate: float | None
    best_source: str | None
    best_hit_rate: float | None
    budget_used: int
    rejected: int


def run_evolution_loop(
    *,
    method: str,
    trace: list[int],
    capacity: int,
    budget: EvaluationBudget,
    rng: random.Random,
    archive: Archive,
    client: LLMClient | None = None,
    held_out_gate: bool = True,
    workload_summary: dict[str, float] | None = None,
) -> EvolutionOutcome:
    train, held_out = time_ordered_split(trace, train_ratio=0.7)
    replay = scan_heavy_replay_window(len(trace), block=min(capacity, 64), rng=rng)

    if not archive.entries:
        from cache_evolve.sandbox.loader import load_policy_class

        seed_cls = load_policy_class(LRU_SOURCE)
        archive.add(
            CandidateRecord(
                source=LRU_SOURCE,
                hit_rate=score(seed_cls, replay, capacity),
                method="seed",
            )
        )

    best: CandidateRecord | None = None
    if method == "random":
        best = random_search(replay, capacity, budget, rng, archive).best
    elif method == "genetic":
        best = genetic_search(replay, capacity, budget, rng, archive).best
    elif method == "llm":
        if client is None:
            raise ValueError("LLM client required")
        parent = archive.entries[0].source
        baselines = {n: score(c, train, capacity) for n, c in BASELINE_POLICIES.items()}
        summary = workload_summary or {}
        feedback = ""
        while budget.remaining > 0:
            prompt = build_prompt(parent, baselines, summary)
            if feedback:
                prompt += f"\nPrevious failure:\n{feedback}\n"
            try:
                raw = client.complete(prompt)
            except Exception as exc:
                budget.consume()
                budget.rejected += 1
                feedback = str(exc)
                continue
            source = extract_python_class(raw)
            try:
                validate_policy_source(source)
            except Exception as exc:
                budget.consume()
                budget.rejected += 1
                feedback = str(exc)
                continue
            sb = run_policy_in_subprocess(source, replay, capacity)
            budget.consume()
            if not sb.ok:
                budget.rejected += 1
                feedback = f"{sb.error}\n{sb.traceback or ''}"
                continue
            best = CandidateRecord(source=source, hit_rate=float(sb.hit_rate or 0.0), method="llm")
            archive.add(best)
            break
    else:
        raise ValueError(f"unknown method: {method}")

    held_hr = None
    if best and held_out_gate:
        from cache_evolve.sandbox.loader import load_policy_class

        cls = load_policy_class(best.source)
        held_hr = score(cls, held_out, capacity)
        if held_hr < score(LRUPolicy, held_out, capacity):
            best = None

    return EvolutionOutcome(
        method=method,
        train_hit_rate=best.hit_rate if best else None,
        held_out_hit_rate=held_hr,
        best_source=best.source if best else None,
        best_hit_rate=best.hit_rate if best else None,
        budget_used=budget.used,
        rejected=budget.rejected,
    )
