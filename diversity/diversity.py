"""
diversity.py  –  Member 5: Diversity Preservation
===================================================
Two pluggable approaches, both returning a *new* list of ScoredIndividual
objects with adjusted fitness scores so that the GA selection step
automatically applies diversity pressure.

Approach 1 – Fitness Sharing
------------------------------
Each individual's fitness is divided by its "niche count":

    shared_fitness_i = raw_fitness_i / sum_j( sh(d(i, j)) )

where the sharing function sh(d) = 1 - (d / sigma)^alpha  if d < sigma,
else 0.  Distance is normalised Hamming distance between integer genomes.

A higher niche count means the individual lives in a crowded region of the
search space, so its effective fitness worsens, making crowded individuals
less likely to be selected.

Approach 2 – Island Model
---------------------------
Population is split into ``n_islands`` sub-populations.  Each sub-population
evolves independently; every ``migration_interval`` calls to the function,
``migration_k`` top individuals from each island migrate to the next island
(ring topology).  This maintains separated niches that explore different
regions in parallel.

Usage
-----
Pass either function as ``diversity_fn`` to ``run_hybrid``:

    from diversity import fitness_sharing, IslandModel
    island = IslandModel(n_islands=4, migration_interval=5, migration_k=2)
    result = run_hybrid(env, evaluate, cfg, seed=42, diversity_fn=island)

    # Or with fitness sharing:
    from functools import partial
    fs = partial(fitness_sharing, sigma=0.3, alpha=1.0)
    result = run_hybrid(env, evaluate, cfg, seed=42, diversity_fn=fs)
"""

from __future__ import annotations

import math
import random
from typing import List, Optional

from algorithms.operators import ScoredIndividual


# ── Helpers ───────────────────────────────────────────────────────────────────

def _hamming_distance_normalised(g1: List[int], g2: List[int]) -> float:
    """
    Normalised Hamming distance in [0, 1].
    For integer-encoded genomes, counts positions where values differ.
    """
    if not g1:
        return 0.0
    diffs = sum(1 for a, b in zip(g1, g2) if a != b)
    return diffs / len(g1)


def _sharing_fn(distance: float, sigma: float, alpha: float) -> float:
    """
    Standard sharing function.
    Returns a value in [0, 1]; 1 when distance == 0, 0 when distance >= sigma.
    """
    if distance >= sigma:
        return 0.0
    return 1.0 - (distance / sigma) ** alpha


# ── Approach 1: Fitness Sharing ───────────────────────────────────────────────

def fitness_sharing(
    population: List[ScoredIndividual],
    env,                          # kept for API compatibility with DiversityFn
    sigma: float = 0.3,
    alpha: float = 1.0,
    min_niche_count: float = 1.0,
) -> List[ScoredIndividual]:
    """
    Return a new population with shared (adjusted) fitness values.

    Parameters
    ----------
    population  : current scored population
    env         : CloudEnvironment (unused; required by DiversityFn signature)
    sigma       : niche radius – individuals further than sigma apart don't
                  affect each other (default 0.3 = 30 % of max Hamming dist)
    alpha       : shape of sharing curve (1.0 = linear)
    min_niche_count : floor for the denominator to avoid zero-division

    Notes
    -----
    Since we are *minimising*, a higher shared fitness = worse individual.
    """
    n = len(population)
    if n == 0:
        return population

    niche_counts: List[float] = []

    for i in range(n):
        nc = 0.0
        for j in range(n):
            d  = _hamming_distance_normalised(
                list(population[i].genome), list(population[j].genome)
            )
            nc += _sharing_fn(d, sigma, alpha)
        niche_counts.append(max(nc, min_niche_count))

    shared = [
        ScoredIndividual(
            genome=ind.genome,
            fitness=ind.fitness * niche_counts[i],
        )
        for i, ind in enumerate(population)
    ]
    return shared


# ── Approach 2: Island Model ──────────────────────────────────────────────────

class IslandModel:
    """
    Stateful island-model diversity manager.

    The island model splits the population into ``n_islands`` sub-populations
    (islands).  Each call to the instance applies migration every
    ``migration_interval`` generations.

    The internal ``_call_count`` tracks how many times __call__ has been
    invoked so migration fires automatically at the right interval.

    Ring-topology migration: island i sends its top-k individuals to island
    (i+1) % n_islands.

    Usage
    -----
        island_model = IslandModel(n_islands=4, migration_interval=5, migration_k=2)
        # pass as diversity_fn:
        run_hybrid(..., diversity_fn=island_model)
    """

    def __init__(
        self,
        n_islands: int = 4,
        migration_interval: int = 5,
        migration_k: int = 2,
        rng: Optional[random.Random] = None,
    ) -> None:
        if n_islands < 2:
            raise ValueError("n_islands must be >= 2")
        if migration_k < 1:
            raise ValueError("migration_k must be >= 1")

        self.n_islands          = n_islands
        self.migration_interval = migration_interval
        self.migration_k        = migration_k
        self.rng                = rng or random.Random()
        self._call_count        = 0
        # Internal island storage: list of lists of ScoredIndividual
        self._islands: List[List[ScoredIndividual]] = []

    # ── Public interface ──────────────────────────────────────────────────────

    def __call__(
        self,
        population: List[ScoredIndividual],
        env,
    ) -> List[ScoredIndividual]:
        """
        Called once per GA generation by the hybrid pipeline.
        Partitions population into islands, optionally migrates, returns flat list.
        Returned fitness values are unchanged (island model preserves raw fitness).
        """
        self._call_count += 1

        # (Re-)partition population into islands
        self._islands = self._partition(population)

        # Migrate every migration_interval calls
        if self._call_count % self.migration_interval == 0:
            self._migrate()

        # Return flat population (order changed to group islands together,
        # which can bias mating but is harmless with the GA's random parent selection)
        flat = [ind for island in self._islands for ind in island]

        # Pad / trim in case of rounding
        if len(flat) < len(population):
            flat.extend(population[: len(population) - len(flat)])
        return flat[: len(population)]

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _partition(self, population: List[ScoredIndividual]) -> List[List[ScoredIndividual]]:
        """Round-robin assignment of individuals to islands."""
        islands: List[List[ScoredIndividual]] = [[] for _ in range(self.n_islands)]
        for i, ind in enumerate(population):
            islands[i % self.n_islands].append(ind)
        return islands

    def _migrate(self) -> None:
        """
        Ring-topology migration: top-k from island i → island (i+1) % n_islands.
        Replaces the bottom-k of the receiving island.
        """
        n = self.n_islands
        migrants: List[List[ScoredIndividual]] = []

        for i in range(n):
            island = self._islands[i]
            if len(island) == 0:
                migrants.append([])
                continue
            k = min(self.migration_k, len(island))
            top_k = sorted(island, key=lambda x: x.fitness)[:k]
            migrants.append(top_k)

        for i in range(n):
            target = (i + 1) % n
            incoming = migrants[i]
            if not incoming or len(self._islands[target]) == 0:
                continue
            # Replace worst individuals in target island
            k = len(incoming)
            self._islands[target].sort(key=lambda x: x.fitness, reverse=True)
            self._islands[target][:k] = incoming

    def reset(self) -> None:
        """Reset call counter and island state (call between independent runs)."""
        self._call_count = 0
        self._islands    = []