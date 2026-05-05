import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Literal, Optional, Sequence, Tuple

from algorithms.operators import (
    ScoredIndividual,
    mutation_random_reset,
    mutation_swap,
    one_point_crossover,
    roulette_wheel_selection,
    tournament_selection,
    uniform_crossover,
)


FitnessFn = Callable[[Sequence[int]], float]
SelectionMethod = Literal["tournament", "roulette"]
SurvivorStrategy = Literal["generational", "elitism"]
CrossoverMethod = Literal["one_point", "uniform"]
MutationMethod = Literal["random_reset", "swap"]
InitMethod = Literal["random", "heuristic_seeded"]


@dataclass
class GAConfig:
    population_size: int = 60
    generations: int = 200
    selection: SelectionMethod = "tournament"
    tournament_size: int = 3
    survivor_strategy: SurvivorStrategy = "elitism"
    elitism_k: int = 2
    crossover: CrossoverMethod = "one_point"
    crossover_rate: float = 0.9
    mutation: MutationMethod = "random_reset"
    mutation_rate: float = 0.05
    init: InitMethod = "random"
    heuristic_seed_ratio: float = 0.3
    patience: Optional[int] = 30  # stop if no improvement


def _make_child(
    p1: Sequence[int],
    p2: Sequence[int],
    cfg: GAConfig,
    n_vms: int,
    rng: random.Random,
) -> Tuple[List[int], List[int]]:
    if rng.random() < cfg.crossover_rate:
        if cfg.crossover == "one_point":
            c1, c2 = one_point_crossover(p1, p2, rng)
        else:
            c1, c2 = uniform_crossover(p1, p2, rng)
    else:
        c1, c2 = list(p1), list(p2)

    if cfg.mutation == "random_reset":
        c1 = mutation_random_reset(c1, n_vms=n_vms, rng=rng, mutation_rate=cfg.mutation_rate)
        c2 = mutation_random_reset(c2, n_vms=n_vms, rng=rng, mutation_rate=cfg.mutation_rate)
    else:
        c1 = mutation_swap(c1, rng=rng, mutation_rate=cfg.mutation_rate)
        c2 = mutation_swap(c2, rng=rng, mutation_rate=cfg.mutation_rate)

    return c1, c2


def _initialise_random(n_tasks: int, n_vms: int, rng: random.Random) -> List[int]:
    return [rng.randrange(n_vms) for _ in range(n_tasks)]


def _initialise_heuristic_seeded(tasks, vms, rng: random.Random) -> List[int]:
    """
    Greedy heuristic:
    - place larger tasks first (cpu*ram*length)
    - for each task choose VM minimising estimated incremental cost while
      discouraging infeasibility by a large soft penalty.
    """
    n_vms = len(vms)
    assignment = [0] * len(tasks)
    vm_cpu = [0.0] * n_vms
    vm_ram = [0.0] * n_vms
    vm_time = [0.0] * n_vms

    order = list(range(len(tasks)))
    order.sort(key=lambda i: (tasks[i].cpu * tasks[i].ram * tasks[i].length), reverse=True)

    for t_idx in order:
        task = tasks[t_idx]
        best_vm = None
        best_score = float("inf")
        candidates = list(range(n_vms))
        rng.shuffle(candidates)
        for vm_id in candidates:
            vm = vms[vm_id]
            exec_time = task.length / vm.speed
            new_time = vm_time[vm_id] + exec_time
            inc_cost = exec_time * vm.cost_per_time

            cpu_over = max(0.0, (vm_cpu[vm_id] + task.cpu) - vm.cpu_capacity)
            ram_over = max(0.0, (vm_ram[vm_id] + task.ram) - vm.ram_capacity)
            infeas_pen = 1e6 * (cpu_over + ram_over)

            score = inc_cost + 0.05 * new_time + infeas_pen
            if score < best_score:
                best_score = score
                best_vm = vm_id

        assignment[t_idx] = int(best_vm) if best_vm is not None else rng.randrange(n_vms)
        vm = vms[assignment[t_idx]]
        vm_cpu[assignment[t_idx]] += task.cpu
        vm_ram[assignment[t_idx]] += task.ram
        vm_time[assignment[t_idx]] += task.length / vm.speed

    return assignment


def _score_population(population: Sequence[List[int]], fitness_fn: FitnessFn) -> List[ScoredIndividual]:
    return [ScoredIndividual(genome=list(g), fitness=float(fitness_fn(g))) for g in population]


def run_ga(
    env,
    fitness_fn: Callable[[Sequence[int], object], float],
    cfg: GAConfig,
    seed: int,
) -> Dict[str, object]:
    rng = random.Random(seed)
    tasks = env.tasks
    vms = env.vms
    n_tasks = len(tasks)
    n_vms = len(vms)

    def fit(g: Sequence[int]) -> float:
        return fitness_fn(g, env)

    population: List[List[int]] = []
    if cfg.init == "random":
        population = [_initialise_random(n_tasks, n_vms, rng) for _ in range(cfg.population_size)]
    else:
        seeded = int(round(cfg.population_size * cfg.heuristic_seed_ratio))
        seeded = max(1, min(cfg.population_size, seeded))
        population.extend(_initialise_heuristic_seeded(tasks, vms, rng) for _ in range(seeded))
        population.extend(_initialise_random(n_tasks, n_vms, rng) for _ in range(cfg.population_size - seeded))

    scored = _score_population(population, fit)
    best = min(scored, key=lambda ind: ind.fitness)
    history_best: List[float] = [best.fitness]
    history_mean: List[float] = [sum(ind.fitness for ind in scored) / len(scored)]

    no_improve = 0
    for _gen in range(cfg.generations):
        next_gen: List[List[int]] = []

        if cfg.survivor_strategy == "elitism":
            elites = sorted(scored, key=lambda ind: ind.fitness)[: max(0, cfg.elitism_k)]
            next_gen.extend([list(e.genome) for e in elites])

        while len(next_gen) < cfg.population_size:
            if cfg.selection == "tournament":
                p1 = tournament_selection(scored, tournament_size=cfg.tournament_size, rng=rng)
                p2 = tournament_selection(scored, tournament_size=cfg.tournament_size, rng=rng)
            else:
                p1 = roulette_wheel_selection(scored, rng=rng)
                p2 = roulette_wheel_selection(scored, rng=rng)

            c1, c2 = _make_child(p1, p2, cfg=cfg, n_vms=n_vms, rng=rng)
            next_gen.append(c1)
            if len(next_gen) < cfg.population_size:
                next_gen.append(c2)

        scored_next = _score_population(next_gen, fit)

if cfg.survivor_strategy == "generational":
    scored = scored_next
else:
    # Apply elitism correctly
    combined = scored_next + scored  # combine old + new
    combined_sorted = sorted(combined, key=lambda ind: ind.fitness)
    scored = combined_sorted[:cfg.population_size]

        gen_best = min(scored, key=lambda ind: ind.fitness)
        gen_mean = sum(ind.fitness for ind in scored) / len(scored)
        history_best.append(gen_best.fitness)
        history_mean.append(gen_mean)

        if gen_best.fitness + 1e-12 < best.fitness:
            best = gen_best
            no_improve = 0
        else:
            no_improve += 1

        if cfg.patience is not None and no_improve >= cfg.patience:
            break

    return {
        "best_genome": list(best.genome),
        "best_fitness": float(best.fitness),
        "history_best": history_best,
        "history_mean": history_mean,
        "seed": seed,
        "config": cfg,
    }

