from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import List, Optional, Tuple

from environment.cloud_model import Task, VM
from environment.dataset_loader import generate_tasks, generate_tasks_by_load, generate_vms


def load_trace_tasks(
    trace_csv: str | Path,
    n_tasks: int,
    rng: Optional[random.Random] = None,
) -> List[Task]:
    """
    Load tasks from a trace-like CSV with columns:
    task_id,cpu,ram,length
    If ``n_tasks`` exceeds file rows, rows are sampled with replacement.
    """
    rng = rng or random.Random()
    trace_path = Path(trace_csv)
    rows: List[Tuple[int, int, int]] = []

    with open(trace_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((int(row["cpu"]), int(row["ram"]), int(row["length"])))

    if not rows:
        raise ValueError(f"No workload rows found in {trace_path}")

    tasks: List[Task] = []
    for i in range(n_tasks):
        cpu, ram, length = rng.choice(rows)
        tasks.append(Task(task_id=i, cpu=cpu, ram=ram, length=length))
    return tasks


def build_workload(
    workload_mode: str,
    n_tasks: int,
    n_vms: int,
    rng: Optional[random.Random] = None,
    trace_csv: str | Path | None = None,
    task_load_level: str | None = None,
) -> tuple[List[Task], List[VM]]:
    rng = rng or random.Random()
    if workload_mode == "trace":
        if trace_csv is None:
            raise ValueError("trace_csv must be provided when workload_mode='trace'")
        tasks = load_trace_tasks(trace_csv=trace_csv, n_tasks=n_tasks, rng=rng)
    else:
        if task_load_level is None:
            tasks = generate_tasks(n_tasks=n_tasks, rng=rng)
        else:
            tasks = generate_tasks_by_load(n_tasks=n_tasks, load_level=task_load_level, rng=rng)

    vms = generate_vms(n_vms=n_vms, rng=rng)
    return tasks, vms
