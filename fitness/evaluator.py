from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence, Tuple


ResponseTimeMetric = Literal["makespan", "mean_vm_time"]


@dataclass(frozen=True)
class FitnessConfig:
    """
    Weighted single-objective fitness (minimisation):
      fitness = w_cost * total_cost + w_time * response_time + w_penalty * penalty

    penalty = sum of CPU/RAM overflow across VMs (units of resource).
    """

    w_cost: float = 1.0
    w_time: float = 1.0
    w_penalty: float = 10.0
    response_time_metric: ResponseTimeMetric = "makespan"


def evaluate_components(
    individual: Sequence[int],
    env,
    cfg: FitnessConfig | None = None,
) -> Tuple[float, float, float, float]:
    cfg = cfg or FitnessConfig()

    tasks = env.tasks
    vms = env.vms
    if len(individual) != len(tasks):
        raise ValueError("individual length must equal number of tasks")
    if not vms:
        raise ValueError("environment must contain at least one VM")

    vm_cpu_usage = [0.0] * len(vms)
    vm_ram_usage = [0.0] * len(vms)
    vm_time = [0.0] * len(vms)

    for task_idx, vm_id in enumerate(individual):
        if vm_id < 0 or vm_id >= len(vms):
            raise ValueError(f"vm_id out of range at index {task_idx}: {vm_id}")
        task = tasks[task_idx]
        vm = vms[vm_id]

        vm_cpu_usage[vm_id] += float(task.cpu)
        vm_ram_usage[vm_id] += float(task.ram)

        exec_time = float(task.length) / float(vm.speed)
        vm_time[vm_id] += exec_time

    total_cost = 0.0
    penalty = 0.0
    for i, vm in enumerate(vms):
        if vm_cpu_usage[i] > vm.cpu_capacity:
            penalty += vm_cpu_usage[i] - float(vm.cpu_capacity)
        if vm_ram_usage[i] > vm.ram_capacity:
            penalty += vm_ram_usage[i] - float(vm.ram_capacity)
        total_cost += vm_time[i] * float(vm.cost_per_time)

    makespan = max(vm_time) if vm_time else 0.0
    mean_vm_time = (sum(vm_time) / len(vm_time)) if vm_time else 0.0

    if cfg.response_time_metric == "makespan":
        response_time = makespan
    else:
        response_time = mean_vm_time

    fitness = (cfg.w_cost * total_cost) + (cfg.w_time * response_time) + (cfg.w_penalty * penalty)
    return fitness, total_cost, response_time, penalty


def evaluate(individual: Sequence[int], env, cfg: FitnessConfig | None = None) -> float:
    fitness, _cost, _rt, _pen = evaluate_components(individual=individual, env=env, cfg=cfg)
    return fitness