from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence


@dataclass(frozen=True)
class Task:
    """A cloud workload unit to be scheduled on a VM."""

    task_id: int
    cpu: float
    ram: float
    length: float

    @property
    def id(self) -> int:
        """Back-compat alias used elsewhere in the project."""
        return self.task_id


@dataclass(frozen=True)
class VM:
    """A cloud compute resource with capacity, speed and cost."""

    vm_id: int
    cpu_capacity: float
    ram_capacity: float
    cost_per_time: float
    speed: float

    @property
    def id(self) -> int:
        """Back-compat alias used elsewhere in the project."""
        return self.vm_id


@dataclass(frozen=True)
class CloudEnvironment:
    """Container for tasks and VMs (the optimization environment)."""

    tasks: Sequence[Task]
    vms: Sequence[VM]

    def __post_init__(self) -> None:
        # Materialize to lists for stable iteration/order across algorithms.
        object.__setattr__(self, "tasks", list(self.tasks))
        object.__setattr__(self, "vms", list(self.vms))
