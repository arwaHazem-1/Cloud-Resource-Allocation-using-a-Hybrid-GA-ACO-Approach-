import argparse
import json
import os
import sys
from dataclasses import asdict
from typing import Dict, List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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


def run_30(
    cfg: GAConfig,
    tasks_n: int,
    vms_n: int,
    seeds_path: str,
) -> Dict[str, object]:
    seeds = load_seeds(seeds_path)[:30]
    results = []
    for seed in seeds:
        # ensure environment is reproducible per run as well
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
    return {"config": asdict(cfg), "tasks_n": tasks_n, "vms_n": vms_n, "best": best, "mean_best_fitness": mean, "runs": results}


def main() -> None:
    parser = argparse.ArgumentParser(description="30-run GA experiment runner (seeded).")
    parser.add_argument("--tasks", type=int, default=30)
    parser.add_argument("--vms", type=int, default=6)
    parser.add_argument("--seeds", type=str, default=os.path.join(os.path.dirname(__file__), "seeds.txt"))
    parser.add_argument("--selection", type=str, choices=["tournament", "roulette"], default="tournament")
    parser.add_argument("--survivor", type=str, choices=["generational", "elitism"], default="elitism")
    parser.add_argument("--init", type=str, choices=["random", "heuristic_seeded"], default="heuristic_seeded")
    parser.add_argument("--out", type=str, default=os.path.join(os.path.dirname(__file__), "..", "results", "ga_runs.json"))
    args = parser.parse_args()

    cfg = GAConfig(selection=args.selection, survivor_strategy=args.survivor, init=args.init)
    summary = run_30(cfg=cfg, tasks_n=args.tasks, vms_n=args.vms, seeds_path=args.seeds)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Wrote:", args.out)
    print("Mean best fitness:", summary["mean_best_fitness"])
    print("Best run:", summary["best"])


if __name__ == "__main__":
    main()

