from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence

from algorithms.de import de_generate_trial
from algorithms.ga import GAConfig, _initialise_random, _make_child, _score_population
from algorithms.operators import ScoredIndividual, tournament_selection


@dataclass
class DEGAConfig:
    population_size: int = 60
    ga_generations: int = 80
    de_generations: int = 80
    mutation_factor: float = 0.6
    crossover_rate: float = 0.8
    ga_tournament_size: int = 3
    elitism_k: int = 2


def run_de_ga(
    env,
    fitness_fn: Callable[[Sequence[int], object], float],
    cfg: DEGAConfig,
    seed: int,
) -> Dict[str, object]:
    """
    Hybrid Evolutionary Algorithm (DE + GA):
    1) GA exploration phase
    2) DE exploitation/refinement phase
    """
    rng = random.Random(seed)
    n_tasks = len(env.tasks)
    n_vms = len(env.vms)

    def fit(genome: Sequence[int]) -> float:
        return float(fitness_fn(genome, env))

    # Initial population
    population = [_initialise_random(n_tasks, n_vms, rng) for _ in range(cfg.population_size)]
    scored = _score_population(population, fit)
    best = min(scored, key=lambda s: s.fitness)
    history_best = [best.fitness]

    # GA phase
    ga_cfg = GAConfig(
        population_size=cfg.population_size,
        generations=cfg.ga_generations,
        selection="tournament",
        tournament_size=cfg.ga_tournament_size,
        survivor_strategy="elitism",
        elitism_k=cfg.elitism_k,
        crossover="one_point",
        mutation="random_reset",
        mutation_rate=0.05,
    )
    for _ in range(cfg.ga_generations):
        next_gen: List[List[int]] = []
        elites = sorted(scored, key=lambda x: x.fitness)[: cfg.elitism_k]
        next_gen.extend([list(e.genome) for e in elites])
        while len(next_gen) < cfg.population_size:
            p1 = tournament_selection(scored, tournament_size=ga_cfg.tournament_size, rng=rng)
            p2 = tournament_selection(scored, tournament_size=ga_cfg.tournament_size, rng=rng)
            c1, c2 = _make_child(p1, p2, cfg=ga_cfg, n_vms=n_vms, rng=rng)
            next_gen.append(c1)
            if len(next_gen) < cfg.population_size:
                next_gen.append(c2)
        scored = _score_population(next_gen, fit)
        gen_best = min(scored, key=lambda s: s.fitness)
        if gen_best.fitness < best.fitness:
            best = gen_best
        history_best.append(best.fitness)

    # DE phase
    population = [list(s.genome) for s in scored]
    fitness_values = [float(s.fitness) for s in scored]
    for _ in range(cfg.de_generations):
        for i in range(cfg.population_size):
            _target, trial = de_generate_trial(
                population=population,
                target_idx=i,
                mutation_factor=cfg.mutation_factor,
                crossover_rate=cfg.crossover_rate,
                n_vms=n_vms,
                rng=rng,
            )
            trial_fit = fit(trial)
            if trial_fit <= fitness_values[i]:
                population[i] = trial
                fitness_values[i] = trial_fit
                if trial_fit < best.fitness:
                    best = ScoredIndividual(genome=list(trial), fitness=trial_fit)
        history_best.append(best.fitness)

    return {
        "best_genome": list(best.genome),
        "best_fitness": float(best.fitness),
        "history_best": history_best,
        "seed": seed,
        "config": cfg,
    }


def run_de_ga_hybrid(
    env,
    fitness_fn: Callable[[Sequence[int], object], float],
    cfg: DEGAConfig,
    seed: int,
) -> Dict[str, object]:
    """
    Backward-compatible alias.
    """
    return run_de_ga(env=env, fitness_fn=fitness_fn, cfg=cfg, seed=seed)
