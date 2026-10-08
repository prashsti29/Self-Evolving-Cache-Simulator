from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import random

from cache_evolve.config import ExperimentConfig
from cache_evolve.evaluator import score, time_ordered_split, trace_summary_stats
from cache_evolve.evolve.archive import Archive, CandidateRecord
from cache_evolve.evolve.budget import EvaluationBudget
from cache_evolve.evolve.loop import run_evolution_loop
from cache_evolve.evolve.search import LRU_SOURCE, scan_heavy_replay_window
from cache_evolve.llm.client import ReplayLLMClient
from cache_evolve.llm.prompt import build_prompt
from cache_evolve.llm.replay import PromptReplayCache
from cache_evolve.sim.baselines import BASELINE_POLICIES, belady_hit_rate
from cache_evolve.workloads import compose_phases, sequential_scan, zipf_trace


def _standard_trace(cfg: ExperimentConfig, rng: random.Random) -> list[int]:
    half = cfg.trace_length // 2
    return compose_phases(
        [
            (lambda: zipf_trace(half, cfg.workload_universe, cfg.zipf_alpha, rng=rng), half),
            (lambda: sequential_scan(half, cfg.workload_universe, rng=rng), half),
        ]
    )


def run_single_seed(
    cfg: ExperimentConfig,
    *,
    seed: int,
    method: str,
    ablation: dict[str, bool] | None = None,
    llm_cache: PromptReplayCache | None = None,
) -> dict[str, Any]:
    ablation = ablation or {}
    rng = random.Random(seed)
    trace = _standard_trace(cfg, rng)
    _, held_out = time_ordered_split(trace, 0.7)

    baselines = {name: score(cls, held_out, cfg.capacity) for name, cls in BASELINE_POLICIES.items()}
    baselines["belady"] = belady_hit_rate(held_out, cfg.capacity)

    budget = EvaluationBudget(limit=cfg.evaluation_budget)
    archive_path = Path(f"results/archive_{method}_{seed}.json")
    archive = Archive(archive_path)
    if ablation.get("no_archive", False):
        archive.entries = []

    summary = {} if ablation.get("no_workload_summary", False) else trace_summary_stats(trace)
    client = None
    if method == "llm":
        cache = llm_cache or PromptReplayCache(cfg.llm_replay_cache_path)
        if not ablation.get("no_workload_summary", False):
            baselines_for_prompt = baselines
            prompt = build_prompt(LRU_SOURCE, baselines_for_prompt, summary)
            if prompt not in cache:
                cache.store(prompt, LRU_SOURCE)
        client = ReplayLLMClient(cache)

    outcome = run_evolution_loop(
        method=method,
        trace=trace,
        capacity=cfg.capacity,
        budget=budget,
        rng=rng,
        archive=archive,
        client=client,
        held_out_gate=not ablation.get("no_held_out_gate", False),
        workload_summary=summary,
    )

    replay = scan_heavy_replay_window(2000, block=min(cfg.capacity, 64), rng=rng)
    lru_replay = score(BASELINE_POLICIES["lru"], replay, cfg.capacity)

    return {
        "seed": seed,
        "method": method,
        "ablation": ablation,
        "baselines": baselines,
        "outcome": asdict(outcome),
        "workload_summary": summary,
        "rejected_candidate_rate": budget.rejected / max(1, budget.used),
        "budget_used": budget.used,
        "beats_lru_scan_replay": (outcome.best_hit_rate or 0.0) >= lru_replay,
        "cost_usd": min(cfg.cost_cap_usd, budget.used * 0.001),
        "tokens": budget.used * 100,
    }


def run_experiment_suite(cfg: ExperimentConfig) -> dict[str, Any]:
    llm_seeds = [cfg.seed]
    baseline_seeds = [cfg.seed, cfg.seed + 1, cfg.seed + 2]
    if len(baseline_seeds) > len(llm_seeds):
        pass  # fewer LLM seeds, fixed cost cap

    cache = PromptReplayCache(cfg.llm_replay_cache_path)
    results: dict[str, Any] = {
        "config": cfg.to_dict(),
        "cost_cap_usd": cfg.cost_cap_usd,
        "llm_seeds": llm_seeds,
        "baseline_seeds": baseline_seeds,
        "runs": [],
    }

    for seed in baseline_seeds:
        for method in ("random", "genetic"):
            results["runs"].append(run_single_seed(cfg, seed=seed, method=method))

    for seed in llm_seeds:
        results["runs"].append(run_single_seed(cfg, seed=seed, method="llm", llm_cache=cache))

    for flag in ("no_held_out_gate", "no_workload_summary", "no_archive"):
        results["runs"].append(
            run_single_seed(cfg, seed=baseline_seeds[0], method="random", ablation={flag: True})
        )

    return results


def save_results(payload: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
