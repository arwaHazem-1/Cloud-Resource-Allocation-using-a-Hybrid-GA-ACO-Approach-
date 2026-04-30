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