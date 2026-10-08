#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
python3 -m venv .venv
. .venv/bin/pip install -q -r requirements.txt -e .
. .venv/bin/pytest -q
. .venv/bin/python scripts/run_experiment.py --out results/reproduce.json
