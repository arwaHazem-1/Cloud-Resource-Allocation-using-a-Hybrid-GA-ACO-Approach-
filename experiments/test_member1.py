import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environment.dataset_loader import generate_tasks, generate_vms
from environment.cloud_model import CloudEnvironment
from fitness.evaluator import evaluate
import random


def main():
    tasks = generate_tasks(10)
    vms = generate_vms(3)

    env = CloudEnvironment(tasks, vms)

    solution = [random.randint(0, len(vms) - 1) for _ in range(len(tasks))]

    print("Solution:", solution)
    print("Fitness:", evaluate(solution, env))


if __name__ == "__main__":
    main()