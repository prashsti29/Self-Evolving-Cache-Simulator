from __future__ import annotations

import random
from typing import Sequence


def zipf_trace(
    length: int,
    universe: int,
    alpha: float = 1.2,
    *,
    rng: random.Random | None = None,
) -> list[int]:
    if length <= 0:
        raise ValueError("length must be positive")
    if universe <= 0:
        raise ValueError("universe must be positive")
    if alpha <= 0:
        raise ValueError("alpha must be positive")

    rng = rng or random.Random()
    weights = [1.0 / ((i + 1) ** alpha) for i in range(universe)]
    return [rng.choices(range(universe), weights=weights, k=1)[0] for _ in range(length)]


def sequential_scan(length: int, universe: int, *, rng: random.Random | None = None) -> list[int]:
    if length <= 0 or universe <= 0:
        raise ValueError("length and universe must be positive")
    rng = rng or random.Random()
    start = rng.randrange(universe)
    out: list[int] = []
    i = start
    for _ in range(length):
        out.append(i)
        i = (i + 1) % universe
    return out


def burst_trace(
    length: int,
    hot_keys: Sequence[int],
    burst_len: int = 32,
    *,
    rng: random.Random | None = None,
) -> list[int]:
    if length <= 0:
        raise ValueError("length must be positive")
    if not hot_keys:
        raise ValueError("hot_keys must be non-empty")
    rng = rng or random.Random()
    hot = list(hot_keys)
    out: list[int] = []
    while len(out) < length:
        key = rng.choice(hot)
        chunk = min(burst_len, length - len(out))
        out.extend([key] * chunk)
    return out


def shifting_hot_set(
    length: int,
    universe: int,
    hot_size: int = 8,
    phase_len: int = 500,
    *,
    rng: random.Random | None = None,
) -> list[int]:
    if length <= 0 or universe <= 0:
        raise ValueError("length and universe must be positive")
    if hot_size <= 0 or hot_size > universe:
        raise ValueError("hot_size must be in 1..universe")
    rng = rng or random.Random()

    keys = list(range(universe))
    rng.shuffle(keys)
    out: list[int] = []
    phase = 0
    while len(out) < length:
        start = (phase * hot_size) % universe
        hot = [(start + i) % universe for i in range(hot_size)]
        chunk = min(phase_len, length - len(out))
        for _ in range(chunk):
            out.append(rng.choices(hot, k=1)[0] if rng.random() < 0.85 else rng.randrange(universe))
        phase += 1
    return out
