from __future__ import annotations

import random
from typing import List, Optional

from environment.cloud_model import Task, VM


def generate_tasks(n_tasks: int, rng: Optional[random.Random] = None) -> List[Task]:
    rng = rng or random
    tasks: List[Task] = []
    for i in range(n_tasks):
        tasks.append(
            Task(
                task_id=i,
                cpu=rng.randint(1, 4),
                ram=rng.randint(1, 8),
                length=rng.randint(100, 1000),
            )
        )
    return tasks


def generate_tasks_by_load(n_tasks: int, load_level: str, rng: Optional[random.Random] = None) -> List[Task]:
    """Generate tasks with explicit workload intensity profiles.

    This is used by experiments to create clearly-defined scenarios.
    """
    rng = rng or random
    profiles = {
        "low": {"cpu": (1, 2), "ram": (1, 4), "length": (80, 300)},
        "medium": {"cpu": (2, 4), "ram": (2, 8), "length": (200, 700)},
        "high": {"cpu": (3, 6), "ram": (4, 12), "length": (500, 1200)},
    }
    if load_level not in profiles:
        raise ValueError(f"Unknown load_level: {load_level}. Expected one of {sorted(profiles)}")
    p = profiles[load_level]
    tasks: List[Task] = []
    for i in range(n_tasks):
        tasks.append(
            Task(
                task_id=i,
                cpu=rng.randint(*p["cpu"]),
                ram=rng.randint(*p["ram"]),
                length=rng.randint(*p["length"]),
            )
        )
    return tasks


def generate_vms(n_vms: int, rng: Optional[random.Random] = None) -> List[VM]:
    rng = rng or random
    vms: List[VM] = []
    for i in range(n_vms):
        vms.append(
            VM(
                vm_id=i,
                cpu_capacity=rng.randint(4, 16),
                ram_capacity=rng.randint(8, 32),
                cost_per_time=rng.uniform(0.5, 2.0),
                speed=rng.uniform(1.0, 3.0),
            )
        )
    return vms