"""
hybrid.py  –  Member 5: Hybrid GA–ACO Pipeline
================================================
Architecture
------------
The run is divided into ``n_cycles`` cycles. Each cycle:

  1. **GA phase**  – evolve the population for ``ga_gens_per_cycle`` generations.
  2. **Seeding**   – extract the top-K genomes and boost pheromone on their
                     (task → VM) edges so ACO starts with a warm graph.
  3. **ACO phase** – run ``aco_iters_per_cycle`` iterations. ACO refines the
                     best GA solutions using pheromone-guided local search.
  4. **Injection** – ACO's best genome replaces the worst individual in the GA
                     population for the next cycle (steady-state injection).

Diversity is managed inside the GA phase through the pluggable
``diversity_fn`` hook (see diversity.py).  Pass ``diversity_fn=None``
to skip it.

Return dict mirrors run_ga / run_aco so the existing experiment runner
(experiments.py) works unchanged.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from algorithms.aco import ACOConfig, build_ant_solution, init_tau_matrix, local_search_assignment, _evaporate, _deposit_global_best
from algorithms.ga import GAConfig, _initialise_random, _initialise_heuristic_seeded, _score_population, _make_child
from algorithms.operators import ScoredIndividual, tournament_selection, roulette_wheel_selection
from fitness.evaluator import evaluate


# ── Types ─────────────────────────────────────────────────────────────────────

FitnessFnEnv = Callable[[Sequence[int], object], float]
DiversityFn  = Callable[[List[ScoredIndividual], object], List[ScoredIndividual]]


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class HybridConfig:
    # ---- GA settings (reuse GAConfig) ----
    ga: GAConfig = field(default_factory=lambda: GAConfig(
        population_size=60,
        generations=200,       # total GA budget; split across cycles
        crossover_rate=0.9,
        mutation_rate=0.05,
        patience=None,         # patience handled at hybrid level
    ))

    # ---- ACO settings (reuse ACOConfig) ----
    aco: ACOConfig = field(default_factory=lambda: ACOConfig(
        n_ants=20,
        n_iterations=50,
        alpha=1.0,
        beta=2.0,
        rho=0.15,
        Q=100.0,
        tau_init=0.1,
        variant="AS",
        patience=20,
        local_search_max_rounds=256,
    ))

    # ---- Hybrid-level knobs ----
    n_cycles: int = 4
    """Number of alternating GA→ACO cycles."""

    ga_gens_per_cycle: int = 50
    """GA generations per cycle (total GA gens ≈ n_cycles * ga_gens_per_cycle)."""

    aco_iters_per_cycle: int = 50
    """ACO iterations per cycle."""

    top_k_to_seed: int = 5
    """How many top GA solutions to use when warming the pheromone matrix."""

    pheromone_boost: float = 3.0
    """Multiplicative boost applied to pheromone edges found in top-K GA solutions."""

    inject_best_back: bool = True
    """Replace the worst GA individual with ACO's best after each cycle."""

    patience: Optional[int] = 30
    """Hybrid-level patience: stop if global best does not improve for this many cycles."""


# ── Pheromone seeding from GA solutions ───────────────────────────────────────

def _seed_tau_from_solutions(
    tau: List[List[float]],
    solutions: List[List[int]],
    boost: float,
) -> None:
    """
    Boost pheromone on (task, vm) pairs that appear in the provided solutions.
    Called once per cycle after the GA phase.
    """
    for sol in solutions:
        for t_idx, vm_id in enumerate(sol):
            tau[t_idx][vm_id] *= boost


def _run_aco_phase(
    env,
    fitness_fn: FitnessFnEnv,
    tau: List[List[float]],
    cfg: ACOConfig,
    n_iters: int,
    rng: random.Random,
    incumbent_genome: List[int],
    incumbent_fitness: float,
) -> tuple[List[int], float, List[float]]:
    """
    Run ``n_iters`` ACO iterations on an *existing* pheromone matrix.
    Returns (best_genome, best_fitness, history_best_this_phase).
    """
    tasks = env.tasks
    vms   = env.vms
    n_vms = len(vms)

    best_genome  = list(incumbent_genome)
    best_fitness = incumbent_fitness
    history: List[float] = []

    stagnation = 0

    for _ in range(n_iters):
        iter_fits: List[float] = []

        for _ in range(cfg.n_ants):
            sol = build_ant_solution(env, tau, cfg, rng)
            sol = local_search_assignment(
                sol, env, fitness_fn, rng=rng,
                max_rounds=cfg.local_search_max_rounds,
            )
            f = float(fitness_fn(sol, env))
            iter_fits.append(f)

            if f + 1e-15 < best_fitness:
                best_fitness = f
                best_genome  = list(sol)
                stagnation   = 0

        _evaporate(tau, cfg.rho)
        deposit = cfg.Q / max(best_fitness, 1e-12)
        _deposit_global_best(tau, best_genome, deposit)

        history.append(best_fitness)

        stagnation += 1
        if cfg.patience is not None and stagnation >= cfg.patience:
            break

    return best_genome, best_fitness, history


def _run_ga_phase(
    env,
    fitness_fn: FitnessFnEnv,
    scored_pop: List[ScoredIndividual],
    cfg: GAConfig,
    n_gens: int,
    rng: random.Random,
    diversity_fn: Optional[DiversityFn],
) -> tuple[List[ScoredIndividual], List[float], List[float]]:
    """
    Evolve ``scored_pop`` for ``n_gens`` generations.
    Returns (new_scored_pop, history_best, history_mean).
    """
    n_vms  = len(env.vms)
    history_best: List[float] = []
    history_mean: List[float] = []

    def fit(g: Sequence[int]) -> float:
        return float(fitness_fn(g, env))

    for _ in range(n_gens):
        # Optional: apply diversity pressure before selection
        effective_pop = diversity_fn(scored_pop, env) if diversity_fn else scored_pop

        next_gen: List[List[int]] = []

        # Elitism
        if cfg.survivor_strategy == "elitism":
            elites = sorted(scored_pop, key=lambda x: x.fitness)[: max(0, cfg.elitism_k)]
            next_gen.extend(list(e.genome) for e in elites)

        # Breed
        while len(next_gen) < cfg.population_size:
            if cfg.selection == "tournament":
                p1 = tournament_selection(effective_pop, cfg.tournament_size, rng)
                p2 = tournament_selection(effective_pop, cfg.tournament_size, rng)
            else:
                p1 = roulette_wheel_selection(effective_pop, rng)
                p2 = roulette_wheel_selection(effective_pop, rng)

            c1, c2 = _make_child(p1, p2, cfg=cfg, n_vms=n_vms, rng=rng)
            next_gen.append(c1)
            if len(next_gen) < cfg.population_size:
                next_gen.append(c2)

        scored_pop = _score_population(next_gen, fit)

        gen_best = min(scored_pop, key=lambda x: x.fitness).fitness
        gen_mean = sum(x.fitness for x in scored_pop) / len(scored_pop)
        history_best.append(gen_best)
        history_mean.append(gen_mean)

    return scored_pop, history_best, history_mean


# ── Main hybrid runner ────────────────────────────────────────────────────────

def run_hybrid(
    env,
    fitness_fn: FitnessFnEnv,
    cfg: HybridConfig,
    seed: int,
    diversity_fn: Optional[DiversityFn] = None,
) -> Dict[str, object]:
    """
    Run the Hybrid GA–ACO algorithm.

    Parameters
    ----------
    env          : CloudEnvironment
    fitness_fn   : callable(genome, env) → float  (minimisation)
    cfg          : HybridConfig
    seed         : int  (for full reproducibility)
    diversity_fn : optional callable applied to the GA population each generation
                   (see diversity.py for fitness_sharing and island_model)

    Returns
    -------
    dict with keys:
        best_genome, best_fitness,
        history_best (all GA gens concatenated + ACO phases),
        history_mean (GA only),
        seed, config, cycles_ran
    """
    rng    = random.Random(seed)
    tasks  = env.tasks
    vms    = env.vms
    n_tasks = len(tasks)
    n_vms   = len(vms)

    if n_tasks == 0 or n_vms == 0:
        raise ValueError("Environment must have at least one task and one VM.")

    ga_cfg  = cfg.ga
    aco_cfg = copy.copy(cfg.aco)
    aco_cfg = ACOConfig(
        n_ants=aco_cfg.n_ants,
        n_iterations=cfg.aco_iters_per_cycle,
        alpha=aco_cfg.alpha,
        beta=aco_cfg.beta,
        rho=aco_cfg.rho,
        Q=aco_cfg.Q,
        tau_init=aco_cfg.tau_init,
        variant=aco_cfg.variant,
        q0=aco_cfg.q0,
        xi=aco_cfg.xi,
        local_search_max_rounds=aco_cfg.local_search_max_rounds,
        patience=aco_cfg.patience,
    )

    def fit(g: Sequence[int]) -> float:
        return float(fitness_fn(g, env))

    # ── Initialise GA population ──────────────────────────────────
    population: List[List[int]]
    if ga_cfg.init == "heuristic_seeded":
        seeded = max(1, int(round(ga_cfg.population_size * ga_cfg.heuristic_seed_ratio)))
        population  = [_initialise_heuristic_seeded(tasks, vms, rng) for _ in range(seeded)]
        population += [_initialise_random(n_tasks, n_vms, rng)
                       for _ in range(ga_cfg.population_size - seeded)]
    else:
        population = [_initialise_random(n_tasks, n_vms, rng)
                      for _ in range(ga_cfg.population_size)]

    scored_pop = _score_population(population, fit)

    # ── Initialise pheromone matrix ───────────────────────────────
    tau = init_tau_matrix(n_tasks, n_vms, aco_cfg.tau_init)

    # ── Global tracking ───────────────────────────────────────────
    global_best = min(scored_pop, key=lambda x: x.fitness)
    global_best_genome:  List[int] = list(global_best.genome)
    global_best_fitness: float     = global_best.fitness

    history_best: List[float] = [global_best_fitness]
    history_mean: List[float] = [sum(x.fitness for x in scored_pop) / len(scored_pop)]

    stagnation   = 0
    cycles_ran   = 0

    # ── Cycle loop ────────────────────────────────────────────────
    for cycle in range(cfg.n_cycles):
        cycles_ran += 1

        # 1. GA phase
        scored_pop, h_best, h_mean = _run_ga_phase(
            env, fitness_fn, scored_pop,
            ga_cfg, cfg.ga_gens_per_cycle, rng, diversity_fn,
        )
        history_best.extend(h_best)
        history_mean.extend(h_mean)

        # Update global best from GA
        cycle_ga_best = min(scored_pop, key=lambda x: x.fitness)
        if cycle_ga_best.fitness + 1e-12 < global_best_fitness:
            global_best_fitness = cycle_ga_best.fitness
            global_best_genome  = list(cycle_ga_best.genome)
            stagnation = 0
        else:
            stagnation += 1

        # 2. Seed ACO pheromone from top-K GA solutions
        top_k = sorted(scored_pop, key=lambda x: x.fitness)[: cfg.top_k_to_seed]
        top_k_genomes = [list(ind.genome) for ind in top_k]
        _seed_tau_from_solutions(tau, top_k_genomes, cfg.pheromone_boost)

        # 3. ACO phase (runs on existing warmed pheromone)
        aco_best_genome, aco_best_fitness, aco_history = _run_aco_phase(
            env, fitness_fn, tau, aco_cfg,
            n_iters=cfg.aco_iters_per_cycle,
            rng=rng,
            incumbent_genome=global_best_genome,
            incumbent_fitness=global_best_fitness,
        )
        # Append ACO history as continued best-so-far
        history_best.extend(aco_history)

        # Update global best from ACO
        if aco_best_fitness + 1e-12 < global_best_fitness:
            global_best_fitness = aco_best_fitness
            global_best_genome  = list(aco_best_genome)
            stagnation = 0
        else:
            stagnation += 1

        # 4. Inject ACO best back into GA population
        if cfg.inject_best_back:
            worst_idx = max(range(len(scored_pop)), key=lambda i: scored_pop[i].fitness)
            scored_pop[worst_idx] = ScoredIndividual(
                genome=list(aco_best_genome),
                fitness=float(aco_best_fitness),
            )

        # Hybrid-level early stopping
        if cfg.patience is not None and stagnation >= cfg.patience:
            break

    return {
        "best_genome":   list(global_best_genome),
        "best_fitness":  float(global_best_fitness),
        "history_best":  history_best,
        "history_mean":  history_mean,
        "seed":          seed,
        "config":        cfg,
        "cycles_ran":    cycles_ran,
    }
