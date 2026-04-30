import random
from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple


FitnessFn = Callable[[Sequence[int]], float]


@dataclass(frozen=True)
class ScoredIndividual:
    genome: List[int]
    fitness: float


def tournament_selection(
    population: Sequence[ScoredIndividual],
    tournament_size: int,
    rng: random.Random,
) -> List[int]:
    if tournament_size < 2:
        raise ValueError("tournament_size must be >= 2")
    if tournament_size > len(population):
        raise ValueError("tournament_size must be <= population size")

    competitors = rng.sample(list(population), k=tournament_size)
    winner = min(competitors, key=lambda ind: ind.fitness)  # minimisation
    return list(winner.genome)


def roulette_wheel_selection(
    population: Sequence[ScoredIndividual],
    rng: random.Random,
    epsilon: float = 1e-12,
) -> List[int]:
    """
    Fitness is minimised, so we convert it to a positive 'score' to maximise:
      score_i = 1 / (fitness_i - min_fitness + 1)
    """
    min_fit = min(ind.fitness for ind in population)
    scores = []
    for ind in population:
        denom = (ind.fitness - min_fit) + 1.0
        scores.append(1.0 / max(denom, epsilon))

    total = sum(scores)
    if total <= 0:
        return list(rng.choice(list(population)).genome)

    pick = rng.random() * total
    cdf = 0.0
    for ind, s in zip(population, scores):
        cdf += s
        if cdf >= pick:
            return list(ind.genome)
    return list(population[-1].genome)


def one_point_crossover(p1: Sequence[int], p2: Sequence[int], rng: random.Random) -> Tuple[List[int], List[int]]:
    if len(p1) != len(p2):
        raise ValueError("Parents must have same length")
    if len(p1) < 2:
        return list(p1), list(p2)
    cut = rng.randint(1, len(p1) - 1)
    c1 = list(p1[:cut]) + list(p2[cut:])
    c2 = list(p2[:cut]) + list(p1[cut:])
    return c1, c2


def uniform_crossover(
    p1: Sequence[int],
    p2: Sequence[int],
    rng: random.Random,
    swap_prob: float = 0.5,
) -> Tuple[List[int], List[int]]:
    if len(p1) != len(p2):
        raise ValueError("Parents must have same length")
    c1 = list(p1)
    c2 = list(p2)
    for i in range(len(c1)):
        if rng.random() < swap_prob:
            c1[i], c2[i] = c2[i], c1[i]
    return c1, c2


def mutation_random_reset(
    genome: Sequence[int],
    n_vms: int,
    rng: random.Random,
    mutation_rate: float,
) -> List[int]:
    if n_vms < 1:
        raise ValueError("n_vms must be >= 1")
    child = list(genome)
    for i in range(len(child)):
        if rng.random() < mutation_rate:
            child[i] = rng.randrange(n_vms)
    return child


def mutation_swap(
    genome: Sequence[int],
    rng: random.Random,
    mutation_rate: float,
) -> List[int]:
    child = list(genome)
    if len(child) < 2:
        return child
    if rng.random() < mutation_rate:
        i, j = rng.sample(range(len(child)), k=2)
        child[i], child[j] = child[j], child[i]
    return child

