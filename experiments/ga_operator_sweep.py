"""GA operator sweep experiment (reproducible).

Purpose (rubric):
- Experimental Design: systematically vary selection/crossover/mutation
- Comparative Analysis: produce a clear table of impacts

Output:
- results/ga_operator_sweep.csv          (scenario × operator combo summary)
- results/ga_operator_sweep.json         (full per-seed run data)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Literal, Sequence, Tuple

from algorithms.ga import GAConfig, run_ga
from environment.cloud_model import CloudEnvironment
from environment.standard_workload import build_workload
from experiments.comparative_runner import load_seeds
from experiments.scenarios import DEFAULT_SCENARIOS, Scenario
from fitness.evaluator import evaluate, evaluate_metrics

Selection = Literal["tournament", "roulette"]
Crossover = Literal["one_point", "uniform"]
Mutation = Literal["random_reset", "swap"]

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def _std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = sum(values) / len(values)
    return float((sum((v - m) ** 2 for v in values) / (len(values) - 1)) ** 0.5)


def _build_env(sc: Scenario, *, seed: int, workload_mode: str, trace_csv: Path) -> CloudEnvironment:
    rng_seed = seed  # keep 1 seed → 1 workload instance
    if workload_mode == "trace":
        tasks, vms = build_workload(
            workload_mode="trace",
            n_tasks=sc.tasks_n,
            n_vms=sc.vms_n,
            rng=None,
            trace_csv=trace_csv,
        )
    else:
        import random

        rng = random.Random(rng_seed)
        tasks, vms = build_workload(
            workload_mode="synthetic",
            n_tasks=sc.tasks_n,
            n_vms=sc.vms_n,
            rng=rng,
            task_load_level=sc.load_level,
        )
    return CloudEnvironment(tasks=tasks, vms=vms)


def run_sweep(
    *,
    scenarios: Sequence[Scenario],
    seeds: Sequence[int],
    workload_mode: str,
    trace_csv: Path,
    population_size: int,
    generations: int,
    crossover_rate: float,
    mutation_rate: float,
) -> Dict[str, object]:
    selections: List[Selection] = ["tournament", "roulette"]
    crossovers: List[Crossover] = ["one_point", "uniform"]
    mutations: List[Mutation] = ["random_reset", "swap"]

    rows: List[Dict[str, object]] = []
    all_runs: List[Dict[str, object]] = []

    for sc in scenarios:
        for sel in selections:
            for cx in crossovers:
                for mut in mutations:
                    cfg = GAConfig(
                        population_size=population_size,
                        generations=generations,
                        init="heuristic_seeded",
                        selection=sel,
                        survivor_strategy="elitism",
                        elitism_k=2,
                        crossover=cx,
                        crossover_rate=crossover_rate,
                        mutation=mut,
                        mutation_rate=mutation_rate,
                        patience=None,
                    )

                    fitnesses: List[float] = []
                    costs: List[float] = []
                    rts: List[float] = []
                    utils: List[float] = []
                    fairs: List[float] = []

                    per_seed: List[Dict[str, object]] = []
                    for seed in seeds:
                        env = _build_env(sc, seed=seed, workload_mode=workload_mode, trace_csv=trace_csv)
                        out = run_ga(env=env, fitness_fn=evaluate, cfg=cfg, seed=seed)
                        m = evaluate_metrics(out["best_genome"], env)

                        fitnesses.append(float(m["fitness"]))
                        costs.append(float(m["total_cost"]))
                        rts.append(float(m["response_time"]))
                        utils.append(float(m["resource_utilization"]))
                        fairs.append(float(m["jains_fairness_index"]))

                        per_seed.append(
                            {
                                "seed": seed,
                                "best_genome": out["best_genome"],
                                "metrics": m,
                                "history_best_len": len(out["history_best"]),
                            }
                        )

                    combo = {
                        "scenario": sc.name,
                        "tasks_n": sc.tasks_n,
                        "vms_n": sc.vms_n,
                        "load_level": sc.load_level,
                        "selection": sel,
                        "crossover": cx,
                        "mutation": mut,
                        "ga_config": asdict(cfg),
                        "n_runs": len(seeds),
                        "per_seed_runs": per_seed,
                    }
                    all_runs.append(combo)

                    rows.append(
                        {
                            "scenario": sc.name,
                            "tasks_n": sc.tasks_n,
                            "vms_n": sc.vms_n,
                            "load_level": sc.load_level,
                            "selection": sel,
                            "crossover": cx,
                            "mutation": mut,
                            "mean_fitness": sum(fitnesses) / len(fitnesses),
                            "std_fitness": _std(fitnesses),
                            "mean_cost": sum(costs) / len(costs),
                            "mean_response_time": sum(rts) / len(rts),
                            "mean_resource_utilization": sum(utils) / len(utils),
                            "mean_jains_fairness_index": sum(fairs) / len(fairs),
                        }
                    )

    return {"summary_rows": rows, "all_runs": all_runs}


def _write_csv(rows: Sequence[Dict[str, object]], path: Path) -> None:
    import csv

    if not rows:
        raise ValueError("No rows to write")
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="GA operator sweep (selection × crossover × mutation).")
    parser.add_argument("--runs", type=int, default=30, help="Number of seeded runs per setting.")
    parser.add_argument("--workload-mode", choices=["synthetic", "trace"], default="synthetic")
    parser.add_argument("--trace-csv", type=str, default=str(ROOT / "data" / "google_cluster_sample.csv"))
    parser.add_argument("--population", type=int, default=60)
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument("--crossover-rate", type=float, default=0.9)
    parser.add_argument("--mutation-rate", type=float, default=0.05)
    args = parser.parse_args()

    seeds = load_seeds(args.runs)
    scenarios = DEFAULT_SCENARIOS
    trace_csv = Path(args.trace_csv)

    RESULTS.mkdir(parents=True, exist_ok=True)

    out = run_sweep(
        scenarios=scenarios,
        seeds=seeds,
        workload_mode=args.workload_mode,
        trace_csv=trace_csv,
        population_size=args.population,
        generations=args.generations,
        crossover_rate=args.crossover_rate,
        mutation_rate=args.mutation_rate,
    )

    _write_csv(out["summary_rows"], RESULTS / "ga_operator_sweep.csv")
    with open(RESULTS / "ga_operator_sweep.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("Wrote:")
    print(" - results/ga_operator_sweep.csv")
    print(" - results/ga_operator_sweep.json")


if __name__ == "__main__":
    main()

