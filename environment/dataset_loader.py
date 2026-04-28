import random
from environment.cloud_model import Task, VM


def generate_tasks(n_tasks):
    tasks = []
    for i in range(n_tasks):
        tasks.append(Task(
            task_id=i,
            cpu=random.randint(1, 4),
            ram=random.randint(1, 8),
            length=random.randint(100, 1000)
        ))
    return tasks


def generate_vms(n_vms):
    vms = []
    for i in range(n_vms):
        vms.append(VM(
            vm_id=i,
            cpu_capacity=random.randint(4, 16),
            ram_capacity=random.randint(8, 32),
            cost_per_time=random.uniform(0.5, 2.0),
            speed=random.uniform(1.0, 3.0)
        ))
    return vms