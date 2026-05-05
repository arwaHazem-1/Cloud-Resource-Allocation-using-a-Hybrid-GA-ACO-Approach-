"""Scenario definitions for reproducible experiments.

This file exists to make the experimental design explicit and consistent across:
- CLI experiment runners
- the Streamlit UI

Scenarios represent workload intensity plus (tasks, vms) sizing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal

LoadLevel = Literal["low", "medium", "high"]


@dataclass(frozen=True)
class Scenario:
    name: str
    tasks_n: int
    vms_n: int
    load_level: LoadLevel


DEFAULT_SCENARIOS: List[Scenario] = [
    Scenario(name="low_load", tasks_n=20, vms_n=4, load_level="low"),
    Scenario(name="medium_load", tasks_n=50, vms_n=10, load_level="medium"),
    Scenario(name="high_load", tasks_n=100, vms_n=20, load_level="high"),
]

