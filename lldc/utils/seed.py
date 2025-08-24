# lldc/utils/seed.py

from typing import List, Sequence
import random

DEFAULT_SEEDS: Sequence[int] = (42,) # (13, ....)


def resolve_seeds(seed: str | int | None, num_runs: int | None) -> List[int]:
    if seed == "auto" or seed is None:
        seeds = list(DEFAULT_SEEDS)
    elif isinstance(seed, int):
        rnd = random.Random(seed)
        seeds = [rnd.randrange(0, 2**31 - 1) for _ in range(num_runs or 1)]
    else:
        raise ValueError(f"Unsupported seed={seed}")
    if num_runs:
        seeds = seeds[:num_runs]
    return seeds
