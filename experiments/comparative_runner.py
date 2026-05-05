from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
SEEDS_FILE = ROOT / "experiments" / "seeds.txt"
RESULTS = ROOT / "results"

sys.path.insert(0, str(ROOT))

from algorithms.aco import ACOConfig, run_aco  # noqa: E402
from algorithms.de_ga_hybrid import DEGAConfig, run_de_ga  # noqa: E402
from algorithms.ga import GAConfig, run_ga  # noqa: E402
from algorithms.hybrid import HybridConfig, run_hybrid  # noqa: E402
from environment.cloud_model import CloudEnvironment, Task  # noqa: E402
from environment.standard_workload import build_workload  # noqa: E402
from fitness.evaluator import evaluate, evaluate_metrics  # noqa: E402


def load_seeds(n: int = 30) -> List[int]:
    if not SEEDS_FILE.exists():
        raise FileNotFoundError(f"Missing shared seeds file: {SEEDS_FILE}")
    seeds: List[int] = []
    with open(SEEDS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                seeds.append(int(line))
    if len(seeds) < n:
        raise ValueError(f"Need at least {n} seeds in {SEEDS_FILE}, found {len(seeds)}")
    return seeds[:n]


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def mann_whitney_u_test(sample_a: Sequence[float], sample_b: Sequence[float]) -> Tuple[float, float]:
    """
    Two-sided Mann-Whitney U test with normal approximation.
    Returns (u_stat, p_value).
    """
    n1 = len(sample_a)
    n2 = len(sample_b)
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0

    combined = [(float(v), 0) for v in sample_a] + [(float(v), 1) for v in sample_b]
    combined.sort(key=lambda x: x[0])

    ranks = [0.0] * len(combined)
    tie_counts: List[int] = []
    i = 0
    while i < len(combined):
        j = i + 1
        while j < len(combined) and combined[j][0] == combined[i][0]:
            j += 1
        avg_rank = 0.5 * (i + 1 + j)
        for k in range(i, j):
            ranks[k] = avg_rank
        tie_counts.append(j - i)
        i = j

    rank_sum_a = sum(ranks[idx] for idx in range(len(combined)) if combined[idx][1] == 0)
    u1 = rank_sum_a - (n1 * (n1 + 1)) / 2.0
    u2 = n1 * n2 - u1
    u = min(u1, u2)

    n = n1 + n2
    tie_term = 0.0
    for t in tie_counts:
        tie_term += t**3 - t
    variance = (n1 * n2 / 12.0) * ((n + 1) - (tie_term / (n * (n - 1))) if n > 1 else 0.0)
    if variance <= 0.0:
        return u, 1.0

    mean_u = n1 * n2 / 2.0
    z = (u - mean_u) / math.sqrt(variance)
    p = 2.0 * (1.0 - _normal_cdf(abs(z)))
    return u, max(0.0, min(1.0, p))


def wilcoxon_signed_rank_test(sample_a: Sequence[float], sample_b: Sequence[float]) -> Tuple[float, float]:
    """
    Two-sided Wilcoxon signed-rank test with normal approximation.
    Paired samples only.
    """
    if len(sample_a) != len(sample_b):
        raise ValueError("Wilcoxon test requires paired samples of equal length")

    diffs = [float(a) - float(b) for a, b in zip(sample_a, sample_b)]
    pairs = [(abs(d), d) for d in diffs if abs(d) > 1e-12]
    n = len(pairs)
    if n == 0:
        return 0.0, 1.0

    pairs.sort(key=lambda x: x[0])
    ranks = [0.0] * n
    tie_counts: List[int] = []
    i = 0
    while i < n:
        j = i + 1
        while j < n and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = 0.5 * (i + 1 + j)
        for k in range(i, j):
            ranks[k] = avg_rank
        tie_counts.append(j - i)
        i = j

    w_plus = sum(ranks[i] for i in range(n) if pairs[i][1] > 0)
    w_minus = sum(ranks[i] for i in range(n) if pairs[i][1] < 0)
    w = min(w_plus, w_minus)

    mean_w = n * (n + 1) / 4.0
    tie_term = sum(t * (t + 1) * (2 * t + 1) for t in tie_counts)
    variance = (n * (n + 1) * (2 * n + 1) - tie_term) / 24.0
    if variance <= 0.0:
        return w, 1.0

    z = (w - mean_w) / math.sqrt(variance)
    p = 2.0 * (1.0 - _normal_cdf(abs(z)))
    return w, max(0.0, min(1.0, p))


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return float((sum((v - m) ** 2 for v in values) / (len(values) - 1)) ** 0.5)


def _generate_tasks_by_load(tasks_n: int, load_level: str, rng: random.Random) -> List[Task]:
    profiles = {
        "low": {"cpu": (1, 2), "ram": (1, 4), "length": (80, 300)},
        "medium": {"cpu": (2, 4), "ram": (2, 8), "length": (200, 700)},
        "high": {"cpu": (3, 6), "ram": (4, 12), "length": (500, 1200)},
    }
    p = profiles[load_level]
    tasks: List[Task] = []
    for i in range(tasks_n):
        tasks.append(
            Task(
                task_id=i,
                cpu=rng.randint(*p["cpu"]),
                ram=rng.randint(*p["ram"]),
                length=rng.randint(*p["length"]),
            )
        )
    return tasks


def _build_env(
    tasks_n: int,
    vms_n: int,
    seed: int,
    workload_mode: str,
    trace_csv: Path,
    load_level: str,
) -> CloudEnvironment:
    rng = random.Random(seed)
    if workload_mode == "synthetic":
        tasks, vms = build_workload(workload_mode="synthetic", n_tasks=tasks_n, n_vms=vms_n, rng=rng)
        tasks = _generate_tasks_by_load(tasks_n=tasks_n, load_level=load_level, rng=rng)
    else:
        tasks, vms = build_workload(
            workload_mode="trace",
            n_tasks=tasks_n,
            n_vms=vms_n,
            rng=rng,
            trace_csv=trace_csv,
        )
    return CloudEnvironment(tasks, vms)


def run_scenario(
    scenario_name: str,
    tasks_n: int,
    vms_n: int,
    load_level: str,
    seeds: Sequence[int],
    workload_mode: str,
    trace_csv: Path,
) -> Dict[str, object]:
    ga_cfg = GAConfig(init="heuristic_seeded", selection="tournament", survivor_strategy="elitism")
    aco_cfg = ACOConfig(n_ants=30, n_iterations=200, variant="AS")
    hybrid_cfg = HybridConfig(
        ga=GAConfig(
            population_size=60,
            crossover_rate=0.9,
            mutation_rate=0.05,
            selection="tournament",
            survivor_strategy="elitism",
            elitism_k=2,
            init="heuristic_seeded",
            heuristic_seed_ratio=0.3,
            patience=None,
        ),
        aco=ACOConfig(
            n_ants=20,
            alpha=1.0,
            beta=2.0,
            rho=0.15,
            Q=100.0,
            tau_init=0.1,
            variant="AS",
            patience=20,
            local_search_max_rounds=256,
        ),
        n_cycles=4,
        ga_gens_per_cycle=50,
        aco_iters_per_cycle=50,
        top_k_to_seed=5,
        pheromone_boost=3.0,
        inject_best_back=True,
        patience=10,
    )
    de_ga_cfg = DEGAConfig(
        population_size=60,
        ga_generations=80,
        de_generations=80,
        mutation_factor=0.6,
        crossover_rate=0.8,
        ga_tournament_size=3,
        elitism_k=2,
    )

    per_algo: Dict[str, List[Dict[str, float]]] = {"GA": [], "ACO": [], "Hybrid": [], "DE+GA": []}
    for seed in seeds:
        env = _build_env(
            tasks_n=tasks_n,
            vms_n=vms_n,
            seed=seed,
            workload_mode=workload_mode,
            trace_csv=trace_csv,
            load_level=load_level,
        )

        ga_out = run_ga(env=env, fitness_fn=evaluate, cfg=ga_cfg, seed=seed)
        aco_out = run_aco(env=env, fitness_fn=evaluate, cfg=aco_cfg, seed=seed)
        hybrid_out = run_hybrid(env=env, fitness_fn=evaluate, cfg=hybrid_cfg, seed=seed)
        de_ga_out = run_de_ga(env=env, fitness_fn=evaluate, cfg=de_ga_cfg, seed=seed)

        for name, out in [("GA", ga_out), ("ACO", aco_out), ("Hybrid", hybrid_out), ("DE+GA", de_ga_out)]:
            metrics = evaluate_metrics(out["best_genome"], env)
            per_algo[name].append(
                {
                    "seed": float(seed),
                    "fitness": metrics["fitness"],
                    "total_cost": metrics["total_cost"],
                    "response_time": metrics["response_time"],
                    "resource_utilization": metrics["resource_utilization"],
                    "jains_fairness_index": metrics["jains_fairness_index"],
                }
            )

    summaries: Dict[str, Dict[str, float]] = {}
    for algo_name, rows in per_algo.items():
        summaries[algo_name] = {
            "mean_fitness": _mean([r["fitness"] for r in rows]),
            "std_fitness": _std([r["fitness"] for r in rows]),
            "mean_cost": _mean([r["total_cost"] for r in rows]),
            "mean_response_time": _mean([r["response_time"] for r in rows]),
            "mean_resource_utilization": _mean([r["resource_utilization"] for r in rows]),
            "mean_jains_fairness_index": _mean([r["jains_fairness_index"] for r in rows]),
        }

    ga_fit = [r["fitness"] for r in per_algo["GA"]]
    aco_fit = [r["fitness"] for r in per_algo["ACO"]]
    hybrid_fit = [r["fitness"] for r in per_algo["Hybrid"]]
    de_ga_fit = [r["fitness"] for r in per_algo["DE+GA"]]

    tests = {
        "Hybrid_vs_GA": {
            "mann_whitney_u": mann_whitney_u_test(hybrid_fit, ga_fit),
            "wilcoxon_signed_rank": wilcoxon_signed_rank_test(hybrid_fit, ga_fit),
        },
        "Hybrid_vs_ACO": {
            "mann_whitney_u": mann_whitney_u_test(hybrid_fit, aco_fit),
            "wilcoxon_signed_rank": wilcoxon_signed_rank_test(hybrid_fit, aco_fit),
        },
        "GA_vs_ACO": {
            "mann_whitney_u": mann_whitney_u_test(ga_fit, aco_fit),
            "wilcoxon_signed_rank": wilcoxon_signed_rank_test(ga_fit, aco_fit),
        },
        "DEGA_vs_GA": {
            "mann_whitney_u": mann_whitney_u_test(de_ga_fit, ga_fit),
            "wilcoxon_signed_rank": wilcoxon_signed_rank_test(de_ga_fit, ga_fit),
        },
        "DEGA_vs_ACO": {
            "mann_whitney_u": mann_whitney_u_test(de_ga_fit, aco_fit),
            "wilcoxon_signed_rank": wilcoxon_signed_rank_test(de_ga_fit, aco_fit),
        },
        "DEGA_vs_Hybrid": {
            "mann_whitney_u": mann_whitney_u_test(de_ga_fit, hybrid_fit),
            "wilcoxon_signed_rank": wilcoxon_signed_rank_test(de_ga_fit, hybrid_fit),
        },
    }

    return {
        "scenario": scenario_name,
        "tasks_n": tasks_n,
        "vms_n": vms_n,
        "load_level": load_level,
        "workload_mode": workload_mode,
        "n_runs": len(seeds),
        "per_algorithm_runs": per_algo,
        "summary": summaries,
        "statistical_tests": tests,
    }


def write_summary_csv(rows: Sequence[Dict[str, object]], path: Path) -> None:
    fieldnames = [
        "scenario",
        "algorithm",
        "mean_fitness",
        "std_fitness",
        "mean_cost",
        "mean_response_time",
        "mean_resource_utilization",
        "mean_jains_fairness_index",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_stats_csv(rows: Sequence[Dict[str, object]], path: Path) -> None:
    fieldnames = [
        "scenario",
        "comparison",
        "mann_whitney_u_stat",
        "mann_whitney_p_value",
        "wilcoxon_w_stat",
        "wilcoxon_p_value",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-scenario comparative runner: GA vs ACO vs Hybrid")
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--workload-mode", choices=["synthetic", "trace"], default="synthetic")
    parser.add_argument(
        "--trace-csv",
        type=str,
        default=str(ROOT / "data" / "google_cluster_sample.csv"),
        help="Task trace CSV used when --workload-mode trace",
    )
    args = parser.parse_args()

    seeds = load_seeds(args.runs)
    scenarios = [
        {"name": "low_load", "tasks_n": 20, "vms_n": 4, "load_level": "low"},
        {"name": "medium_load", "tasks_n": 50, "vms_n": 10, "load_level": "medium"},
        {"name": "high_load", "tasks_n": 100, "vms_n": 20, "load_level": "high"},
    ]
    trace_csv = Path(args.trace_csv)

    RESULTS.mkdir(parents=True, exist_ok=True)
    all_json: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    stats_rows: List[Dict[str, object]] = []

    for sc in scenarios:
        scenario_name = sc["name"]
        out = run_scenario(
            scenario_name=scenario_name,
            tasks_n=sc["tasks_n"],
            vms_n=sc["vms_n"],
            load_level=sc["load_level"],
            seeds=seeds,
            workload_mode=args.workload_mode,
            trace_csv=trace_csv,
        )
        all_json.append(out)

        for algo in ["GA", "ACO", "Hybrid", "DE+GA"]:
            s = out["summary"][algo]
            summary_rows.append(
                {
                    "scenario": scenario_name,
                    "algorithm": algo,
                    "mean_fitness": f"{s['mean_fitness']:.6f}",
                    "std_fitness": f"{s['std_fitness']:.6f}",
                    "mean_cost": f"{s['mean_cost']:.6f}",
                    "mean_response_time": f"{s['mean_response_time']:.6f}",
                    "mean_resource_utilization": f"{s['mean_resource_utilization']:.6f}",
                    "mean_jains_fairness_index": f"{s['mean_jains_fairness_index']:.6f}",
                }
            )

        for comp, tests in out["statistical_tests"].items():
            u_stat, u_p = tests["mann_whitney_u"]
            w_stat, w_p = tests["wilcoxon_signed_rank"]
            stats_rows.append(
                {
                    "scenario": scenario_name,
                    "comparison": comp,
                    "mann_whitney_u_stat": f"{u_stat:.6f}",
                    "mann_whitney_p_value": f"{u_p:.6f}",
                    "wilcoxon_w_stat": f"{w_stat:.6f}",
                    "wilcoxon_p_value": f"{w_p:.6f}",
                }
            )

    with open(RESULTS / "comparative_multi_scenario.json", "w", encoding="utf-8") as f:
        json.dump(all_json, f, indent=2)

    write_summary_csv(summary_rows, RESULTS / "comparative_head_to_head.csv")
    write_stats_csv(stats_rows, RESULTS / "comparative_significance_tests.csv")

    print("Wrote:")
    print(" - results/comparative_multi_scenario.json")
    print(" - results/comparative_head_to_head.csv")
    print(" - results/comparative_significance_tests.csv")


if __name__ == "__main__":
    main()
