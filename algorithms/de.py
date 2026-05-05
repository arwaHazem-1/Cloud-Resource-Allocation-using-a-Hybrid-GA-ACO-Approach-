from __future__ import annotations

import random
from typing import List, Sequence, Tuple


def de_mutation_rand1(
    x_a: Sequence[int],
    x_b: Sequence[int],
    x_c: Sequence[int],
    mutation_factor: float,
    n_vms: int,
) -> List[int]:
    """
    DE/rand/1 mutation in integer assignment space:
      v = x_a + F * (x_b - x_c)
    """
    mutant: List[int] = []
    for i in range(len(x_a)):
        value = x_a[i] + mutation_factor * (x_b[i] - x_c[i])
        mutant.append(int(max(0, min(n_vms - 1, round(value)))))
    return mutant


def de_binomial_crossover(
    target: Sequence[int],
    donor: Sequence[int],
    crossover_rate: float,
    rng: random.Random,
) -> List[int]:
    """
    Binomial crossover with at least one donor dimension.
    """
    d = len(target)
    j_rand = rng.randrange(d)
    trial: List[int] = []
    for j in range(d):
        if rng.random() < crossover_rate or j == j_rand:
            trial.append(int(donor[j]))
        else:
            trial.append(int(target[j]))
    return trial


def de_generate_trial(
    population: Sequence[Sequence[int]],
    target_idx: int,
    mutation_factor: float,
    crossover_rate: float,
    n_vms: int,
    rng: random.Random,
) -> Tuple[List[int], List[int]]:
    """
    Returns (target, trial) genome for DE selection step.
    """
    target = list(population[target_idx])
    candidate_indices = [i for i in range(len(population)) if i != target_idx]
    a_idx, b_idx, c_idx = rng.sample(candidate_indices, 3)
    donor = de_mutation_rand1(
        x_a=population[a_idx],
        x_b=population[b_idx],
        x_c=population[c_idx],
        mutation_factor=mutation_factor,
        n_vms=n_vms,
    )
    trial = de_binomial_crossover(target=target, donor=donor, crossover_rate=crossover_rate, rng=rng)
    return target, trial
