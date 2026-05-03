import os
import random
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from algorithms.aco import ACOConfig, run_aco
from environment.cloud_model import CloudEnvironment
from environment.dataset_loader import generate_tasks, generate_vms
from fitness.evaluator import FitnessConfig, evaluate


def test_environment_and_evaluator():
    """Member 1 wiring: tasks, VMs, random assignment, fitness."""
    tasks = generate_tasks(10)
    vms = generate_vms(3)
    env = CloudEnvironment(tasks, vms)
    solution = [random.randint(0, len(vms) - 1) for _ in range(len(tasks))]

    print("Solution:", solution)
    cfg = FitnessConfig(w_cost=1.0, w_time=1.0, w_penalty=10.0, response_time_metric="makespan")
    print("Fitness:", evaluate(solution, env, cfg=cfg))


def test_aco_integration():
    """Member 4 wiring: ACO uses same env + evaluate as GA."""
    rng = random.Random(42)
    tasks = generate_tasks(12, rng=rng)
    vms = generate_vms(4, rng=rng)
    env = CloudEnvironment(tasks, vms)

    naive = [rng.randrange(len(vms)) for _ in range(len(tasks))]
    naive_fit = evaluate(naive, env)

    cfg = ACOConfig(
        n_ants=14,
        n_iterations=35,
        alpha=1.0,
        beta=2.0,
        rho=0.15,
        Q=120.0,
        variant="AS",
        patience=80,
        local_search_max_rounds=200,
    )
    out = run_aco(env=env, fitness_fn=evaluate, cfg=cfg, seed=12345)
    assert "best_genome" in out and len(out["best_genome"]) == len(tasks)
    assert out["best_fitness"] <= naive_fit + 1e-6
    assert len(out["history_best"]) >= 2


def main():
    print("--- Environment + evaluator ---")
    test_environment_and_evaluator()
    print()
    print("--- ACO integration ---")
    test_aco_integration()
    print("aco integration OK")


if __name__ == "__main__":
    main()
