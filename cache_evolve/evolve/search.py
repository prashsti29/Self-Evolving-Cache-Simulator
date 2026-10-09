from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Sequence

from cache_evolve.sandbox.runner import SandboxResult, run_policy_in_subprocess

from .archive import Archive, CandidateRecord
from .budget import EvaluationBudget

LRU_SOURCE = """
from cache_evolve.sim.policy import Policy
from collections import OrderedDict

class LRUPolicyClone(Policy):
    def __init__(self, capacity):
        super().__init__(capacity)
        self._order = OrderedDict()
    def on_hit(self, key):
        if key in self._order:
            self._order.move_to_end(key)
    def on_miss(self, key):
        self._order[key] = None
    def evict(self):
        return self._order.popitem(last=False)[0]
""".strip()


@dataclass
class SearchResult:
    best: CandidateRecord | None
    evaluations: int
    rejected: int


def _evaluate_source_fixed_budget(
    source: str, trace: Sequence[object], capacity: int, budget: EvaluationBudget
) -> tuple[SandboxResult, float | None]:
    if not budget.consume():
        return SandboxResult(ok=False, error="budget exhausted"), None
    sb = run_policy_in_subprocess(source, trace, capacity)
    if not sb.ok:
        budget.rejected += 1
    return sb, sb.hit_rate


def random_search(
    trace: Sequence[object],
    capacity: int,
    budget: EvaluationBudget,
    rng: random.Random,
    archive: Archive | None = None,
) -> SearchResult:
    best: CandidateRecord | None = None
    templates = [LRU_SOURCE, LRU_SOURCE.replace("LRUPolicyClone", "RandA"), LRU_SOURCE.replace("popitem(last=False)", "popitem(last=True)")]
    while budget.remaining > 0:
        base = rng.choice(templates)
        source = base + f"\n# seed={rng.randint(0, 10**9)}\n"
        sb, hr = _evaluate_source_fixed_budget(source, trace, capacity, budget)
        if sb.ok and hr is not None:
            rec = CandidateRecord(source=source, hit_rate=hr, method="random")
            if archive:
                archive.add(rec)
            if best is None or hr > best.hit_rate:
                best = rec
    return SearchResult(best=best, evaluations=budget.used, rejected=budget.rejected)


def mutate_source(source: str, rng: random.Random) -> str:
    lines = source.splitlines()
    if not lines:
        return source
    idx = rng.randrange(len(lines))
    lines[idx] = lines[idx] + "  # mut"
    return "\n".join(lines)


def genetic_search(
    trace: Sequence[object],
    capacity: int,
    budget: EvaluationBudget,
    rng: random.Random,
    archive: Archive,
) -> SearchResult:
    best: CandidateRecord | None = archive.entries[0] if archive.entries else None
    parents = archive.sample_parents(3, rng) or [CandidateRecord(source=LRU_SOURCE, hit_rate=0.0, method="seed")]
    while budget.remaining > 0:
        parent = rng.choice(parents).source
        child = mutate_source(parent, rng)
        sb, hr = _evaluate_source_fixed_budget(child, trace, capacity, budget)
        if sb.ok and hr is not None:
            rec = CandidateRecord(source=child, hit_rate=hr, method="genetic", generation=1)
            archive.add(rec)
            if best is None or hr > best.hit_rate:
                best = rec
    if archive.entries:
        best = max(archive.entries, key=lambda e: e.hit_rate)
    return SearchResult(best=best, evaluations=budget.used, rejected=budget.rejected)


def parallel_evaluate(
    sources: Sequence[str],
    trace: Sequence[object],
    capacity: int,
    budget: EvaluationBudget,
    workers: int = 4,
) -> list[tuple[str, SandboxResult, float | None]]:
    results: list[tuple[str, SandboxResult, float | None]] = []

    def job(src: str) -> tuple[str, SandboxResult, float | None]:
        sb, hr = _evaluate_source_fixed_budget(src, trace, capacity, budget)
        return src, sb, hr

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(job, s) for s in sources if budget.remaining > 0]
        for fut in as_completed(futures):
            results.append(fut.result())
    return results


def scan_heavy_replay_window(length: int, block: int, rng: random.Random) -> list[int]:
    """Two-pass scan over block keys — replay window with locality."""
    seq = list(range(block)) + list(range(block))
    if length <= len(seq):
        return seq[:length]
    out = []
    while len(out) < length:
        out.extend(seq)
    return out[:length]
