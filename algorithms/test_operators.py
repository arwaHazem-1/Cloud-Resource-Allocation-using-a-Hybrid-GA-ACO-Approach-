import random

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from algorithms.operators import (
    mutation_random_reset,
    mutation_swap,
    one_point_crossover,
    uniform_crossover,
)


def _in_range(genome, n_vms: int) -> bool:
    return all(0 <= int(g) < n_vms for g in genome)


n_vms = 3
rng = random.Random(123)

p1 = [0, 1, 2, 1]
p2 = [2, 0, 1, 0]

print("Parents:", p1, p2)

print("\n--- Crossover ---")
c1, c2 = one_point_crossover(p1, p2, rng=rng)
u1, u2 = uniform_crossover(p1, p2, rng=rng)
print("One-point:", c1, c2)
print("Uniform:", u1, u2)
assert len(c1) == len(p1) and len(c2) == len(p1)
assert len(u1) == len(p1) and len(u2) == len(p1)

print("\n--- Mutation ---")
m1 = mutation_random_reset(p1, n_vms=n_vms, rng=rng, mutation_rate=1.0)
m2 = mutation_swap(p1, rng=rng, mutation_rate=1.0)
print("Random reset:", m1)
print("Swap:", m2)
assert _in_range(m1, n_vms)
assert _in_range(m2, n_vms)

print("\nAll operator smoke-tests passed.")