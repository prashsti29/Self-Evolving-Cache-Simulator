from __future__ import annotations

from cache_evolve.config import ExperimentConfig
from cache_evolve.experiments.runner import run_experiment_suite


def test_experiment_suite_smoke(tmp_path, monkeypatch):
    cfg = ExperimentConfig(
        seed=1,
        trace_length=400,
        evaluation_budget=3,
        llm_replay_cache_path=str(tmp_path / "llm.json"),
    )
    monkeypatch.chdir(tmp_path)
    results = run_experiment_suite(cfg)
    assert results["runs"]
    assert all(r["budget_used"] <= cfg.evaluation_budget for r in results["runs"])
