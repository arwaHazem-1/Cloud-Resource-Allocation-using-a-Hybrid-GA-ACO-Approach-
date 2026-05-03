"""
Ant Colony Optimization for cloud task-to-VM assignment.

Pheromone is stored per (task_index, vm_index): tau[t][v] encodes accumulated
preference for placing task t on VM v. Ants build solutions by visiting tasks
in a random order each tour and probabilistically picking a VM using
tau^alpha * eta^beta. Heuristic eta is computed dynamically from incremental
cost, estimated load imbalance, and soft capacity overflow (consistent with the
seeded heuristic in ga.py).

Supports Ant System (AS) with global-best deposit and optional Ant Colony
System (ACS) with pseudo-random proportional choice and local pheromone
updates.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Literal, Optional, Sequence

ACOVariant = Literal["AS", "ACS"]

FitnessFnEnv = Callable[[Sequence[int], object], float]


@dataclass
class ACOConfig:
    n_ants: int = 30
    n_iterations: int = 200
    alpha: float = 1.0
    beta: float = 2.0
    rho: float = 0.1
    """Evaporation factor in [0, 1]; all tau *= (1 - rho) each iteration."""
    Q: float = 100.0
    """Scaling for pheromone deposit (deposit amount ~ Q / best_fitness)."""
    tau_init: float = 0.1
    variant: ACOVariant = "AS"
    q0: float = 0.9
    """ACS: probability of exploiting argmax tau^alpha * eta^beta."""
    xi: float = 0.1
    """ACS: local update mixing toward tau0."""
    local_search_max_rounds: int = 512
    """Safety cap on first-improvement local search outer rounds."""
    patience: Optional[int] = 40
    """Stop outer iterations early if global best stagnates this many iterations."""


def _marginal_score(
    task,
    vm,
    vm_cpu: Sequence[float],
    vm_ram: Sequence[float],
    vm_time: Sequence[float],
    vm_idx: int,
) -> float:
    """Lower is better — matches intuition of heuristic in ga._initialise_heuristic_seeded."""
    exec_time = float(task.length) / float(vm.speed)
    new_time = float(vm_time[vm_idx]) + exec_time
    inc_cost = exec_time * float(vm.cost_per_time)

    cpu_over = max(0.0, (float(vm_cpu[vm_idx]) + float(task.cpu)) - float(vm.cpu_capacity))
    ram_over = max(0.0, (float(vm_ram[vm_idx]) + float(task.ram)) - float(vm.ram_capacity))
    infeas_pen = 1e6 * (cpu_over + ram_over)

    return inc_cost + 0.05 * new_time + infeas_pen


def _heuristic_from_score(score: float) -> float:
    return 1.0 / (score + 1e-9)


def _assign_task(vm_idx: int, task, vms, vm_cpu: List[float], vm_ram: List[float], vm_time: List[float]) -> None:
    vm = vms[vm_idx]
    vm_cpu[vm_idx] += float(task.cpu)
    vm_ram[vm_idx] += float(task.ram)
    vm_time[vm_idx] += float(task.length) / float(vm.speed)


def _unassign_task(vm_idx: int, task, vms, vm_cpu: List[float], vm_ram: List[float], vm_time: List[float]) -> None:
    vm = vms[vm_idx]
    vm_cpu[vm_idx] -= float(task.cpu)
    vm_ram[vm_idx] -= float(task.ram)
    vm_time[vm_idx] -= float(task.length) / float(vm.speed)


def build_ant_solution(
    env,
    tau: List[List[float]],
    cfg: ACOConfig,
    rng: random.Random,
    *,
    seed_hint: Optional[Sequence[int]] = None,
) -> List[int]:
    """
    Construct one ant's assignment using pheromone on (task, vm) pairs.

    Task visit order is shuffled each ant.

    ``seed_hint`` is reserved for hybrid pipelines (e.g. bias first ant from GA);
    not used in the standard run_aco loop.
    """
    del seed_hint  # API hook for GA–ACO integration
    tasks = env.tasks
    vms = env.vms
    n_tasks = len(tasks)
    n_vms = len(vms)
    order = list(range(n_tasks))
    rng.shuffle(order)

    assignment = [0] * n_tasks
    vm_cpu = [0.0] * n_vms
    vm_ram = [0.0] * n_vms
    vm_time = [0.0] * n_vms

    for t_idx in order:
        weights: List[float] = []
        for v in range(n_vms):
            score = _marginal_score(tasks[t_idx], vms[v], vm_cpu, vm_ram, vm_time, v)
            eta = _heuristic_from_score(score)
            tau_v = max(tau[t_idx][v], 1e-12)
            weights.append((tau_v**cfg.alpha) * (eta**cfg.beta))

        if cfg.variant == "ACS" and rng.random() < cfg.q0:
            chosen = max(range(n_vms), key=lambda vv: weights[vv])
        else:
            total_w = sum(weights)
            if total_w <= 0 or not math.isfinite(total_w):
                chosen = rng.randrange(n_vms)
            else:
                r = rng.random() * total_w
                cdf = 0.0
                chosen = n_vms - 1
                for v in range(n_vms):
                    cdf += weights[v]
                    if cdf >= r:
                        chosen = v
                        break

        if cfg.variant == "ACS":
            tau[t_idx][chosen] = (1.0 - cfg.xi) * tau[t_idx][chosen] + cfg.xi * cfg.tau_init

        assignment[t_idx] = chosen
        _assign_task(chosen, tasks[t_idx], vms, vm_cpu, vm_ram, vm_time)

    return assignment


def local_search_assignment(
    solution: Sequence[int],
    env,
    fitness_fn: FitnessFnEnv,
    rng: Optional[random.Random] = None,
    max_rounds: int = 512,
) -> List[int]:
    """
    First-improvement neighbourhood: reassign one task to another VM if fitness improves.
    """
    rng = rng or random.Random()
    tasks = env.tasks
    vms = env.vms
    n_tasks = len(tasks)
    n_vms = len(vms)
    current = list(solution)
    best_fit = float(fitness_fn(current, env))

    improved = True
    rounds = 0
    while improved and rounds < max_rounds:
        improved = False
        rounds += 1

        indices = list(range(n_tasks))
        rng.shuffle(indices)
        order_vms = list(range(n_vms))

        for t_idx in indices:
            curr_vm = current[t_idx]
            rng.shuffle(order_vms)
            for v in order_vms:
                if v == curr_vm:
                    continue
                trial = list(current)
                trial[t_idx] = v
                fit = float(fitness_fn(trial, env))
                if fit + 1e-15 < best_fit:
                    current = trial
                    best_fit = fit
                    improved = True
                    break
            if improved:
                break

    return current


def _evaporate(tau: List[List[float]], rho: float) -> None:
    factor = max(0.0, min(1.0, 1.0 - rho))
    for row in tau:
        for j in range(len(row)):
            row[j] *= factor


def _deposit_global_best(
    tau: List[List[float]],
    genome: Sequence[int],
    deposit: float,
) -> None:
    for i, vm_id in enumerate(genome):
        tau[i][vm_id] += deposit


def init_tau_matrix(n_tasks: int, n_vms: int, tau0: float) -> List[List[float]]:
    return [[tau0 for _ in range(n_vms)] for _ in range(n_tasks)]


def run_aco(
    env,
    fitness_fn: FitnessFnEnv,
    cfg: ACOConfig,
    seed: int,
) -> Dict[str, object]:
    """
    Run ACO minimising ``fitness_fn(genome, env)``.

    Mirrors ``run_ga`` return keys used by experiments: ``best_genome``,
    ``best_fitness``, iteration-wise ``history_best`` / ``history_mean``,
    ``seed``, ``config``.
    """
    rng = random.Random(seed)
    tasks = env.tasks
    vms = env.vms
    n_tasks = len(tasks)
    n_vms = len(vms)
    if n_tasks == 0:
        raise ValueError("ACO requires at least one task")
    if n_vms < 1:
        raise ValueError("ACO requires at least one VM")

    tau = init_tau_matrix(n_tasks, n_vms, cfg.tau_init)
    iterations_ran = 0

    def fit(g: Sequence[int]) -> float:
        return float(fitness_fn(g, env))

    global_best_genome = [rng.randrange(n_vms) for _ in range(n_tasks)]
    global_best_genome = local_search_assignment(
        global_best_genome,
        env,
        fitness_fn,
        rng=rng,
        max_rounds=cfg.local_search_max_rounds,
    )
    global_best_fitness = fit(global_best_genome)

    history_best: List[float] = [global_best_fitness]
    history_mean: List[float] = [global_best_fitness]
    stagnation = 0

    for _it in range(cfg.n_iterations):
        iteration_genomes: List[List[int]] = []
        iteration_fits: List[float] = []

        # Work on a copy of tau for ant construction if ACS local update mutates graph
        for _a in range(cfg.n_ants):
            sol = build_ant_solution(env, tau, cfg, rng)
            sol = local_search_assignment(
                sol,
                env,
                fitness_fn,
                rng=rng,
                max_rounds=cfg.local_search_max_rounds,
            )
            f = fit(sol)
            iteration_genomes.append(sol)
            iteration_fits.append(f)

            if f + 1e-15 < global_best_fitness:
                global_best_fitness = f
                global_best_genome = list(sol)

        # Evaporation + global-best deposit (AS / ACS)
        _evaporate(tau, cfg.rho)
        deposit = cfg.Q / max(global_best_fitness, 1e-12)
        _deposit_global_best(tau, global_best_genome, deposit)

        mean_it = sum(iteration_fits) / len(iteration_fits)
        history_best.append(global_best_fitness)
        history_mean.append(mean_it)
        iterations_ran += 1

        if history_best[-1] + 1e-15 < history_best[-2]:
            stagnation = 0
        else:
            stagnation += 1
        if cfg.patience is not None and stagnation >= cfg.patience:
            break

    return {
        "best_genome": list(global_best_genome),
        "best_fitness": float(global_best_fitness),
        "history_best": history_best,
        "history_mean": history_mean,
        "seed": seed,
        "config": cfg,
        "iterations_ran": iterations_ran,
    }
