#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from cache_evolve.config import ExperimentConfig
from cache_evolve.experiments.runner import run_experiment_suite, save_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cache-evolve experiment suite")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("results/experiment.json"))
    args = parser.parse_args()

    cfg = ExperimentConfig.load_file(args.config) if args.config else ExperimentConfig()
    cfg.apply_seed()
    payload = run_experiment_suite(cfg)
    save_results(payload, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
