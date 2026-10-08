from __future__ import annotations

import random
from pathlib import Path

from cache_evolve.evaluator import score
from cache_evolve.evolve.archive import Archive
from cache_evolve.evolve.budget import EvaluationBudget
from cache_evolve.evolve.loop import run_evolution_loop
from cache_evolve.evolve.search import scan_heavy_replay_window
from cache_evolve.sim.baselines import LRUPolicy


def test_equal_budget_counts_rejections(tmp_path: Path):
    trace = list(range(500))
    cap = 16
    for method in ("random", "genetic"):
        budget = EvaluationBudget(limit=5)
        archive = Archive(tmp_path / f"{method}.json")
        run_evolution_loop(
            method=method,
            trace=trace,
            capacity=cap,
            budget=budget,
            rng=random.Random(0),
            archive=archive,
            held_out_gate=False,
        )
        assert budget.used == 5


def test_beats_lru_on_scan_heavy_replay(tmp_path: Path):
    rng = random.Random(1)
    replay = scan_heavy_replay_window(4000, block=64, rng=rng)
    lru = score(LRUPolicy, replay, 64)
    budget = EvaluationBudget(limit=8)
    archive = Archive(tmp_path / "gen.json")
    out = run_evolution_loop(
        method="genetic",
        trace=replay,
        capacity=64,
        budget=budget,
        rng=rng,
        archive=archive,
        held_out_gate=False,
    )
    assert (out.best_hit_rate or 0.0) >= lru
