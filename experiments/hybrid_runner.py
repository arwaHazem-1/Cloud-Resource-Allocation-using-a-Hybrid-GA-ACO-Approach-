"""
hybrid_runner.py  –  Member 5
==============================
Runs all five configurations (GA, ACO, Hybrid, Hybrid+Island, Hybrid+Sharing)
across 30 seeds and produces:
  • results/ga_{suffix}.json
  • results/aco_{suffix}.json
  • results/hybrid_t{N}_v{M}.json
  • results/hybrid_island_t{N}_v{M}.json
  • results/hybrid_sharing_t{N}_v{M}.json
  • results/hybrid_comparison_t{N}_v{M}.csv   (all 5 algorithms)
  • results/plots/convergence_t{N}_v{M}.png
  • results/plots/boxplot_t{N}_v{M}.png

Seeds are loaded from the shared experiments/seeds.txt so all algorithms
use identical seeds for a fair comparison.

Run
---
    python experiments/hybrid_runner.py
    python experiments/hybrid_runner.py --tasks 30 --vms 6
    python experiments/hybrid_runner.py --configs ga aco hybrid hybrid_island hybrid_sharing
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional

ROOT       = Path(__file__).resolve().parent.parent
SEEDS_FILE = ROOT / "experiments" / "seeds.txt"
RESULTS    = ROOT / "results"
PLOTS      = ROOT / "results" / "plots"

sys.path.insert(0, str(ROOT))

from algorithms.hybrid import HybridConfig, run_hybrid              # noqa: E402
from algorithms.aco import ACOConfig, run_aco                       # noqa: E402
from algorithms.ga import GAConfig, run_ga                          # noqa: E402
from diversity.diversity import IslandModel, fitness_sharing        # noqa: E402
from environment.cloud_model import CloudEnvironment                # noqa: E402
from environment.dataset_loader import generate_tasks, generate_vms # noqa: E402
from fitness.evaluator import evaluate, evaluate_components         # noqa: E402

try:
    import matplotlib
    import matplotlib.pyplot as plt
    plt.switch_backend("Agg")
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARNING] matplotlib not found – plots will be skipped.")


# ── Seeds ─────────────────────────────────────────────────────────────────────

def load_seeds(n: int = 30) -> List[int]:
    if SEEDS_FILE.exists():
        seeds = []
        with open(SEEDS_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    seeds.append(int(line))
        if len(seeds) < n:
            raise ValueError(
                f"seeds.txt has only {len(seeds)} seeds — need at least {n}."
            )
        print(f"[INFO] Loaded {n} seeds from {SEEDS_FILE}")
        return seeds[:n]

    rng   = random.Random(2026)
    seeds = [rng.randint(0, 2**31 - 1) for _ in range(n)]
    SEEDS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SEEDS_FILE, "w") as f:
        f.write("# Shared seeds – used by runner.py AND hybrid_runner.py\n")
        f.write("\n".join(str(s) for s in seeds))
    print(f"[INFO] Generated {n} seeds → {SEEDS_FILE}")
    return seeds


# ── Extra metrics helper ───────────────────────────────────────────────────────

def _compute_extra_metrics(genome, env):
    """
    Returns (fitness, cost, response_time, penalty, resource_util, load_balance).
    """
    fitness, cost, response_time, penalty = evaluate_components(genome, env)

    vm_time = [0.0] * len(env.vms)
    vm_cpu  = [0.0] * len(env.vms)
    for ti, vi in enumerate(genome):
        t = env.tasks[ti]
        vm_time[vi] += t.length / env.vms[vi].speed
        vm_cpu[vi]  += t.cpu

    resource_util = (
        sum(vm_cpu[i] / env.vms[i].cpu_capacity for i in range(len(env.vms)))
        / len(env.vms)
    )
    load_balance = (max(vm_time) - min(vm_time)) / (max(vm_time) + 1e-9)

    return fitness, cost, response_time, penalty, resource_util, load_balance


# ── 30-run core ───────────────────────────────────────────────────────────────

def run_30(
    label: str,
    runner: Callable,
    seeds: List[int],
    tasks_n: int,
    vms_n: int,
) -> Dict:
    runs      = []
    histories = []

    for seed in seeds:
        random.seed(seed)
        tasks = generate_tasks(tasks_n)
        vms   = generate_vms(vms_n)
        env   = CloudEnvironment(tasks, vms)

        result = runner(env=env, seed=seed)

        _, cost, response_time, penalty, resource_util, load_balance = (
            _compute_extra_metrics(result["best_genome"], env)
        )

        runs.append({
            "seed":           seed,
            "best_fitness":   float(result["best_fitness"]),
            "cost":           cost,
            "response_time":  response_time,
            "penalty":        penalty,
            "resource_util":  resource_util,
            "load_balance":   load_balance,
            "cycles_ran":     result.get("cycles_ran", "-"),
        })
        histories.append([float(v) for v in result["history_best"]])

    fits     = [r["best_fitness"] for r in runs]
    best_run = min(runs, key=lambda r: r["best_fitness"])

    return {
        "label":              label,
        "tasks_n":            tasks_n,
        "vms_n":              vms_n,
        "n_runs":             len(runs),
        "mean_fitness":       sum(fits) / len(fits),
        "best_fitness":       best_run["best_fitness"],
        "worst_fitness":      max(fits),
        "std_fitness":        _std(fits),
        "best_seed":          best_run["seed"],
        "mean_cost":          sum(r["cost"]          for r in runs) / len(runs),
        "mean_response_time": sum(r["response_time"] for r in runs) / len(runs),
        "mean_resource_util": sum(r["resource_util"] for r in runs) / len(runs),
        "mean_load_balance":  sum(r["load_balance"]  for r in runs) / len(runs),
        "runs":               runs,
        "histories":          histories,
    }


def _std(values: List[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return (sum((v - mean) ** 2 for v in values) / (n - 1)) ** 0.5


# ── Runner factories ──────────────────────────────────────────────────────────

def _base_hybrid_cfg() -> HybridConfig:
    return HybridConfig(
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


# GA standalone
def ga_runner(env, seed: int) -> Dict:
    cfg = GAConfig(
        population_size=60,
        crossover_rate=0.9,
        mutation_rate=0.05,
        selection="tournament",
        survivor_strategy="elitism",
        elitism_k=2,
        init="heuristic_seeded",
        heuristic_seed_ratio=0.3,
        patience=30,
    )
    return run_ga(env=env, fitness_fn=evaluate, cfg=cfg, seed=seed)


# ACO standalone
def aco_runner(env, seed: int) -> Dict:
    cfg = ACOConfig(
        n_ants=30,
        n_iterations=200,
        alpha=1.0,
        beta=2.0,
        rho=0.1,
        Q=100.0,
        variant="AS",
        patience=40,
        local_search_max_rounds=512,
    )
    return run_aco(env=env, fitness_fn=evaluate, cfg=cfg, seed=seed)


# Hybrid variants
def hybrid_runner(env, seed: int) -> Dict:
    return run_hybrid(
        env=env, fitness_fn=evaluate,
        cfg=_base_hybrid_cfg(), seed=seed,
        diversity_fn=None,
    )


def hybrid_island_runner(env, seed: int) -> Dict:
    island = IslandModel(n_islands=4, migration_interval=5, migration_k=2)
    return run_hybrid(
        env=env, fitness_fn=evaluate,
        cfg=_base_hybrid_cfg(), seed=seed,
        diversity_fn=island,
    )


def hybrid_sharing_runner(env, seed: int) -> Dict:
    fs = partial(fitness_sharing, sigma=0.3, alpha=1.0)
    return run_hybrid(
        env=env, fitness_fn=evaluate,
        cfg=_base_hybrid_cfg(), seed=seed,
        diversity_fn=fs,
    )


REGISTRY = {
    "ga":             ("GA",                       ga_runner),
    "aco":            ("ACO",                      aco_runner),
    "hybrid":         ("Hybrid (no diversity)",    hybrid_runner),
    "hybrid_island":  ("Hybrid + Island Model",    hybrid_island_runner),
    "hybrid_sharing": ("Hybrid + Fitness Sharing", hybrid_sharing_runner),
}


# ── Plots ─────────────────────────────────────────────────────────────────────

COLOURS = {
    "GA":                       "#888780",
    "ACO":                      "#185FA5",
    "Hybrid (no diversity)":    "#534AB7",
    "Hybrid + Island Model":    "#0F6E56",
    "Hybrid + Fitness Sharing": "#BA7517",
}


def _mean_curve(histories: List[List[float]], length: int) -> List[float]:
    padded = []
    for h in histories:
        padded.append(
            h[:length] if len(h) >= length else h + [h[-1]] * (length - len(h))
        )
    return [sum(col) / len(col) for col in zip(*padded)]


def plot_convergence(summaries: List[Dict], path: str) -> None:
    if not HAS_MPL:
        return
    max_len = max(max(len(h) for h in s["histories"]) for s in summaries)
    fig, ax = plt.subplots(figsize=(10, 6))
    for s in summaries:
        colour = COLOURS.get(s["label"], "#888")
        curve  = _mean_curve(s["histories"], max_len)
        ax.plot(curve, label=s["label"], color=colour, linewidth=2)
        if len(s["histories"]) >= 2:
            padded = []
            for h in s["histories"]:
                padded.append(
                    h[:max_len] if len(h) >= max_len
                    else h + [h[-1]] * (max_len - len(h))
                )
            upper = [
                sum(col) / len(col) + 0.5 * _std(list(col))
                for col in zip(*padded)
            ]
            lower = [
                sum(col) / len(col) - 0.5 * _std(list(col))
                for col in zip(*padded)
            ]
            ax.fill_between(range(max_len), lower, upper, color=colour, alpha=0.12)
    ax.set_xlabel("Evaluation step")
    ax.set_ylabel("Best fitness (lower is better)")
    ax.set_title(
        "All Algorithms – Convergence Curves (mean ± 0.5 SD, 30 runs)"
    )
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[PLOT] Convergence → {path}")


def plot_boxplot(summaries: List[Dict], path: str) -> None:
    if not HAS_MPL:
        return
    labels  = [s["label"] for s in summaries]
    data    = [[r["best_fitness"] for r in s["runs"]] for s in summaries]
    colours = [COLOURS.get(lbl, "#888") for lbl in labels]
    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(
        data, patch_artist=True, notch=False,
        medianprops=dict(color="white", linewidth=2),
    )
    for patch, colour in zip(bp["boxes"], colours):
        patch.set_facecolor(colour)
        patch.set_alpha(0.75)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=10, ha="right")
    ax.set_ylabel("Final best fitness (lower is better)")
    ax.set_title("All Algorithms – Distribution over 30 Runs")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[PLOT] Box plot → {path}")


# ── CSV table ─────────────────────────────────────────────────────────────────

def save_csv(summaries: List[Dict], path: str) -> None:
    rows = [{
        "Algorithm":          s["label"],
        "Tasks":              s["tasks_n"],
        "VMs":                s["vms_n"],
        "Runs":               s["n_runs"],
        "Mean fitness":       f"{s['mean_fitness']:.4f}",
        "Best fitness":       f"{s['best_fitness']:.4f}",
        "Worst fitness":      f"{s['worst_fitness']:.4f}",
        "Std dev":            f"{s['std_fitness']:.4f}",
        "Mean cost":          f"{s.get('mean_cost', 0):.4f}",
        "Mean response time": f"{s.get('mean_response_time', 0):.4f}",
        "Mean resource util": f"{s.get('mean_resource_util', 0):.4f}",
        "Mean load balance":  f"{s.get('mean_load_balance', 0):.4f}",
    } for s in summaries]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[CSV]  Comparison table → {path}")


# ── JSON ──────────────────────────────────────────────────────────────────────

def save_json(summary: Dict, path: str) -> None:
    exportable = {k: v for k, v in summary.items() if k != "histories"}
    with open(path, "w") as f:
        json.dump(exportable, f, indent=2)
    print(f"[JSON] {summary['label']} → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="30-run experiment runner – all algorithms (Member 5)"
    )
    parser.add_argument("--tasks", type=int, default=30)
    parser.add_argument("--vms",   type=int, default=6)
    parser.add_argument(
        "--configs", nargs="+",
        choices=["ga", "aco", "hybrid", "hybrid_island", "hybrid_sharing"],
        default=["ga", "aco", "hybrid", "hybrid_island", "hybrid_sharing"],
    )
    args = parser.parse_args()

    seeds   = load_seeds(30)
    tasks_n = args.tasks
    vms_n   = args.vms
    suffix  = f"t{tasks_n}_v{vms_n}"

    RESULTS.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)

    summaries = []

    for key in args.configs:
        label, runner = REGISTRY[key]
        print(f"\n{'='*60}")
        print(f" Running 30 × {label}  (tasks={tasks_n}, vms={vms_n})")
        print(f"{'='*60}")

        summary = run_30(label, runner, seeds, tasks_n, vms_n)
        summaries.append(summary)

        print(f"  Mean fitness    : {summary['mean_fitness']:.4f}")
        print(f"  Best fitness    : {summary['best_fitness']:.4f}")
        print(f"  Std dev         : {summary['std_fitness']:.4f}")
        print(f"  Mean cost       : {summary['mean_cost']:.4f}")
        print(f"  Mean resp. time : {summary['mean_response_time']:.4f}")
        print(f"  Mean res. util  : {summary['mean_resource_util']:.4f}")
        print(f"  Mean load bal.  : {summary['mean_load_balance']:.4f}")

        # Save individual JSON (use key as filename prefix)
        save_json(summary, str(RESULTS / f"{key}_{suffix}.json"))

    plot_convergence(summaries, str(PLOTS / f"convergence_{suffix}.png"))
    plot_boxplot(summaries,     str(PLOTS / f"boxplot_{suffix}.png"))
    save_csv(summaries,         str(RESULTS / f"hybrid_comparison_{suffix}.csv"))

    print("\nDone.")


if __name__ == "__main__":
    main()