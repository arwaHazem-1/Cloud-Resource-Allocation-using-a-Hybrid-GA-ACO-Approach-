import argparse
import json
import os
import sys
from dataclasses import asdict
from typing import Dict, List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from algorithms.aco import ACOConfig, run_aco  # noqa: E402
from algorithms.ga import GAConfig, run_ga  # noqa: E402
from environment.cloud_model import CloudEnvironment  # noqa: E402
from environment.dataset_loader import generate_tasks, generate_vms  # noqa: E402
from fitness.evaluator import evaluate  # noqa: E402


def load_seeds(path: str) -> List[int]:
    with open(path, "r", encoding="utf-8") as f:
        seeds = []
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            seeds.append(int(line))
    if len(seeds) < 30:
        raise ValueError(f"Need at least 30 seeds, found {len(seeds)} in {path}")
    return seeds


def run_30_ga(
    cfg: GAConfig,
    tasks_n: int,
    vms_n: int,
    seeds_path: str,
) -> Dict[str, object]:
    seeds = load_seeds(seeds_path)[:30]
    results = []
    for seed in seeds:
        import random

        random.seed(seed)
        tasks = generate_tasks(tasks_n)
        vms = generate_vms(vms_n)
        env = CloudEnvironment(tasks, vms)

        out = run_ga(env=env, fitness_fn=evaluate, cfg=cfg, seed=seed)
        results.append(
            {
                "seed": seed,
                "best_fitness": out["best_fitness"],
                "best_genome": out["best_genome"],
                "generations_ran": len(out["history_best"]) - 1,
            }
        )

    best = min(results, key=lambda r: r["best_fitness"])
    mean = sum(r["best_fitness"] for r in results) / len(results)
    return {"config": asdict(cfg), "algorithm": "ga", "tasks_n": tasks_n, "vms_n": vms_n, "best": best, "mean_best_fitness": mean, "runs": results}


def run_30_aco(
    cfg: ACOConfig,
    tasks_n: int,
    vms_n: int,
    seeds_path: str,
) -> Dict[str, object]:
    seeds = load_seeds(seeds_path)[:30]
    results = []
    for seed in seeds:
        import random

        random.seed(seed)
        tasks = generate_tasks(tasks_n)
        vms = generate_vms(vms_n)
        env = CloudEnvironment(tasks, vms)

        out = run_aco(env=env, fitness_fn=evaluate, cfg=cfg, seed=seed)
        results.append(
            {
                "seed": seed,
                "best_fitness": out["best_fitness"],
                "best_genome": out["best_genome"],
                "iterations_ran": out.get("iterations_ran", len(out["history_best"]) - 1),
            }
        )

    best = min(results, key=lambda r: r["best_fitness"])
    mean = sum(r["best_fitness"] for r in results) / len(results)
    return {"config": asdict(cfg), "algorithm": "aco", "tasks_n": tasks_n, "vms_n": vms_n, "best": best, "mean_best_fitness": mean, "runs": results}


def run_30(
    algorithm: str,
    ga_cfg: GAConfig,
    aco_cfg: ACOConfig,
    tasks_n: int,
    vms_n: int,
    seeds_path: str,
) -> Dict[str, object]:
    if algorithm == "ga":
        return run_30_ga(ga_cfg, tasks_n, vms_n, seeds_path)
    if algorithm == "aco":
        return run_30_aco(aco_cfg, tasks_n, vms_n, seeds_path)
    raise ValueError(f"unknown algorithm {algorithm}")


def main() -> None:
    parser = argparse.ArgumentParser(description="30-run seeded experiment runner (GA or ACO).")
    parser.add_argument("--algorithm", type=str, choices=["ga", "aco"], default="ga")
    parser.add_argument("--tasks", type=int, default=30)
    parser.add_argument("--vms", type=int, default=6)
    parser.add_argument("--seeds", type=str, default=os.path.join(os.path.dirname(__file__), "seeds.txt"))
    parser.add_argument("--selection", type=str, choices=["tournament", "roulette"], default="tournament")
    parser.add_argument("--survivor", type=str, choices=["generational", "elitism"], default="elitism")
    parser.add_argument("--init", type=str, choices=["random", "heuristic_seeded"], default="heuristic_seeded")
    parser.add_argument("--aco-variant", type=str, choices=["AS", "ACS"], default="AS")
    parser.add_argument("--aco-ants", type=int, default=30)
    parser.add_argument("--aco-iters", type=int, default=200)
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    ga_cfg = GAConfig(selection=args.selection, survivor_strategy=args.survivor, init=args.init)
    aco_cfg = ACOConfig(n_ants=args.aco_ants, n_iterations=args.aco_iters, variant=args.aco_variant)  # type: ignore[arg-type]

    default_out = os.path.join(os.path.dirname(__file__), "..", "results", f"{args.algorithm}_runs.json")
    out_path = args.out if args.out else default_out

    summary = run_30(
        algorithm=args.algorithm,
        ga_cfg=ga_cfg,
        aco_cfg=aco_cfg,
        tasks_n=args.tasks,
        vms_n=args.vms,
        seeds_path=args.seeds,
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Wrote:", out_path)
    print("Mean best fitness:", summary["mean_best_fitness"])
    print("Best run:", summary["best"])


if __name__ == "__main__":
    main()

