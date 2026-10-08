from __future__ import annotations

PROMPT_TEMPLATE = """You are evolving a cache eviction policy in Python.

Current policy code:
{policy_code}

Scores vs baselines (hit rate):
{baseline_scores}

Workload summary:
{workload_summary}

Return ONLY a Python class that subclasses Policy with methods:
__init__(self, capacity), on_hit, on_miss, evict.
Import Policy from cache_evolve.sim.policy.
"""


def build_prompt(
    policy_code: str,
    baseline_scores: dict[str, float],
    workload_summary: dict[str, float],
) -> str:
    scores = "\n".join(f"  {k}: {v:.4f}" for k, v in sorted(baseline_scores.items()))
    summary = "\n".join(f"  {k}: {v:.4f}" for k, v in sorted(workload_summary.items()))
    return PROMPT_TEMPLATE.format(
        policy_code=policy_code.strip() or "# empty",
        baseline_scores=scores,
        workload_summary=summary,
    )
