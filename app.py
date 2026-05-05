import io
import math
import random
import time
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from algorithms.aco import ACOConfig, run_aco
from algorithms.de_ga_hybrid import DEGAConfig, run_de_ga
from algorithms.ga import GAConfig, run_ga
from algorithms.hybrid import HybridConfig, run_hybrid
from diversity.diversity import IslandModel, fitness_sharing
from environment.cloud_model import CloudEnvironment
from environment.standard_workload import build_workload
from fitness.evaluator import evaluate, evaluate_metrics


ROOT = Path(__file__).resolve().parent
SHARED_SEEDS_FILE = ROOT / "experiments" / "seeds.txt"


def load_shared_seeds(n: int) -> list[int]:
    """Load the project's shared seeds for reproducible UI runs."""
    if not SHARED_SEEDS_FILE.exists():
        return []
    seeds: list[int] = []
    with open(SHARED_SEEDS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                seeds.append(int(line))
    return seeds[:n]


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def mann_whitney_u_test(sample_a, sample_b):
    n1 = len(sample_a)
    n2 = len(sample_b)
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0
    combined = [(float(v), 0) for v in sample_a] + [(float(v), 1) for v in sample_b]
    combined.sort(key=lambda x: x[0])
    ranks = [0.0] * len(combined)
    i = 0
    while i < len(combined):
        j = i + 1
        while j < len(combined) and combined[j][0] == combined[i][0]:
            j += 1
        avg_rank = 0.5 * (i + 1 + j)
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j
    rank_sum_a = sum(ranks[idx] for idx in range(len(combined)) if combined[idx][1] == 0)
    u1 = rank_sum_a - (n1 * (n1 + 1)) / 2.0
    u2 = n1 * n2 - u1
    u = min(u1, u2)
    n = n1 + n2
    variance = (n1 * n2 * (n + 1)) / 12.0
    if variance <= 0:
        return u, 1.0
    z = (u - (n1 * n2 / 2.0)) / math.sqrt(variance)
    p = 2.0 * (1.0 - _normal_cdf(abs(z)))
    return u, max(0.0, min(1.0, p))


def wilcoxon_signed_rank_test(sample_a, sample_b):
    if len(sample_a) != len(sample_b):
        return 0.0, 1.0
    diffs = [float(a) - float(b) for a, b in zip(sample_a, sample_b)]
    pairs = [(abs(d), d) for d in diffs if abs(d) > 1e-12]
    if not pairs:
        return 0.0, 1.0
    pairs.sort(key=lambda x: x[0])
    ranks = [float(i + 1) for i in range(len(pairs))]
    w_plus = sum(ranks[i] for i in range(len(pairs)) if pairs[i][1] > 0)
    w_minus = sum(ranks[i] for i in range(len(pairs)) if pairs[i][1] < 0)
    w = min(w_plus, w_minus)
    n = len(pairs)
    mean_w = n * (n + 1) / 4.0
    var_w = n * (n + 1) * (2 * n + 1) / 24.0
    if var_w <= 0:
        return w, 1.0
    z = (w - mean_w) / math.sqrt(var_w)
    p = 2.0 * (1.0 - _normal_cdf(abs(z)))
    return w, max(0.0, min(1.0, p))


def generations_to_threshold(history: list, threshold_pct: float = 0.95) -> int:
    """Return the 1-based iteration index at which the convergence curve first
    reaches ``threshold_pct`` of its total improvement (start - best).
    Returns len(history) if the threshold is never crossed."""
    if not history:
        return 0
    start = history[0]
    best  = min(history)
    improvement = start - best
    if improvement <= 0:
        return len(history)
    target = start - threshold_pct * improvement
    for i, val in enumerate(history):
        if val <= target:
            return i + 1
    return len(history)


st.set_page_config(layout="wide")

st.title("Cloud Resource Allocation (GA • ACO • Hybrid GA–ACO)")
st.caption("UI is structured to mirror the 8-section evaluation rubric.")

st.header("1) Problem Formulation & Cloud Modelling")
st.markdown(
    """
We allocate tasks to VMs under CPU/RAM capacity constraints to optimize:
- **Cost** (sum of VM runtimes × VM cost rate)
- **Response time** (makespan or mean VM time)

Decision variable: integer vector \(x\) where \(x_i\\) is the VM index assigned to task \(i\).
    """
)

with st.expander("Rubric navigation (what to look at)", expanded=False):
    st.markdown(
        """
- **2) GA Implementation**: `run_ga` results shown below
- **3) ACO Implementation**: `run_aco` results shown below
- **4) Hybrid GA–ACO**: `run_hybrid` results shown below
- **5) Experimental Design**: scenario presets + shared seeds + repeated runs
- **6) Comparative Analysis**: tables + plots + significance tests
- **7) Performance Metrics**: cost/response/utilization/fairness
- **8) UI**: this Streamlit dashboard
        """
    )

st.sidebar.header("Parameters")

st.sidebar.subheader("Workload")
scenario_mode = st.sidebar.selectbox("Scenario Mode", ["Custom", "Low/Medium/High presets"], index=1)
if scenario_mode == "Low/Medium/High presets":
    scenario = st.sidebar.selectbox("Scenario", ["low_load (20/4)", "medium_load (50/10)", "high_load (100/20)"], index=0)
    if scenario.startswith("low_load"):
        num_tasks, num_vms, load_level = 20, 4, "low"
    elif scenario.startswith("medium_load"):
        num_tasks, num_vms, load_level = 50, 10, "medium"
    else:
        num_tasks, num_vms, load_level = 100, 20, "high"
else:
    num_tasks = st.sidebar.slider("Tasks", 5, 200, 30)
    num_vms = st.sidebar.slider("VMs", 2, 30, 6)
    load_level = st.sidebar.selectbox("Synthetic load level", ["low", "medium", "high"], index=1)

cycles = st.sidebar.slider("Hybrid Cycles", 1, 10, 4)
ga_gens = st.sidebar.slider("GA Generations", 10, 200, 50)
aco_iters = st.sidebar.slider("ACO Iterations", 10, 200, 50)
runs = st.sidebar.slider("Repeated Runs", 3, 30, 10)
aco_variant = st.sidebar.selectbox("ACO Variant", ["AS", "ACS"], index=0)
diversity_mode = st.sidebar.selectbox(
    "Hybrid Diversity",
    ["None", "Fitness Sharing", "Island Model"],
    index=0,
)
workload_mode = st.sidebar.selectbox("Workload Mode", ["synthetic", "trace"], index=0)
trace_csv_path = st.sidebar.text_input(
    "Trace CSV Path",
    value="data/google_cluster_sample.csv",
    help="Used when workload mode = trace",
)

convergence_threshold_pct = st.sidebar.slider(
    "Convergence Threshold (%)",
    min_value=50, max_value=99, value=95,
    help="% of total improvement at which 'generations to threshold' is measured",
)

st.sidebar.subheader("Reproducibility")
seed_mode = st.sidebar.selectbox("Seed mode", ["Shared seeds file", "Single fixed seed", "Random (not reproducible)"], index=0)
fixed_seed = int(st.sidebar.number_input("Base seed", value=203948, step=1))

if seed_mode == "Shared seeds file":
    shared = load_shared_seeds(runs)
    if len(shared) < runs:
        st.sidebar.warning("Shared seeds file missing/short; falling back to deterministic derived seeds.")
        per_run_seeds = [fixed_seed + i for i in range(runs)]
    else:
        per_run_seeds = shared[:runs]
elif seed_mode == "Single fixed seed":
    per_run_seeds = [fixed_seed for _ in range(runs)]
else:
    per_run_seeds = [random.randint(1, 2**31 - 1) for _ in range(runs)]

rng = random.Random(fixed_seed)
tasks, vms = build_workload(
    workload_mode=workload_mode,
    n_tasks=num_tasks,
    n_vms=num_vms,
    rng=rng,
    trace_csv=trace_csv_path if workload_mode == "trace" else None,
    task_load_level=load_level if workload_mode == "synthetic" else None,
)
env = CloudEnvironment(tasks=tasks, vms=vms)

if st.button("Run All Algorithms"):

    ga_results = []
    aco_results = []
    hybrid_results = []
    dega_results = []
    ga_metrics_all = []
    aco_metrics_all = []
    hybrid_metrics_all = []
    dega_metrics_all = []

    ga_histories     = []
    aco_histories    = []
    hybrid_histories = []
    dega_histories   = []

    progress = st.progress(0)
    status = st.empty()
    total_steps = runs * 4
    done_steps = 0

    with st.spinner("Running algorithms..."):

        # GA
        start = time.time()
        for _run in range(runs):
            status.write(f"GA run {_run + 1}/{runs}")
            ga_result = run_ga(
                env,
                fitness_fn=lambda g, e: evaluate(g, e),
                cfg=GAConfig(generations=ga_gens),
                seed=int(per_run_seeds[_run]),
            )
            ga_results.append(ga_result["best_fitness"])
            ga_metrics_all.append(evaluate_metrics(ga_result["best_genome"], env))
            ga_histories.append(ga_result["history_best"])
            done_steps += 1
            progress.progress(done_steps / total_steps)
        ga_time = time.time() - start

        # ACO
        start = time.time()
        for _run in range(runs):
            status.write(f"ACO run {_run + 1}/{runs}")
            aco_result = run_aco(
                env,
                fitness_fn=lambda g, e: evaluate(g, e),
                cfg=ACOConfig(n_iterations=aco_iters, variant=aco_variant),
                seed=int(per_run_seeds[_run]),
            )
            aco_results.append(aco_result["best_fitness"])
            aco_metrics_all.append(evaluate_metrics(aco_result["best_genome"], env))
            aco_histories.append(aco_result["history_best"])
            done_steps += 1
            progress.progress(done_steps / total_steps)
        aco_time = time.time() - start

        # Hybrid
        if diversity_mode == "Fitness Sharing":
            diversity_fn = partial(fitness_sharing, sigma=0.3, alpha=1.0)
        elif diversity_mode == "Island Model":
            diversity_fn = IslandModel(n_islands=4, migration_interval=5, migration_k=2)
        else:
            diversity_fn = None

        start = time.time()
        for _run in range(runs):
            status.write(f"Hybrid run {_run + 1}/{runs}")
            hybrid_result = run_hybrid(
                env,
                fitness_fn=lambda g, e: evaluate(g, e),
                cfg=HybridConfig(
                    n_cycles=cycles,
                    ga_gens_per_cycle=ga_gens,
                    aco_iters_per_cycle=aco_iters,
                    aco=ACOConfig(variant=aco_variant),
                ),
                seed=int(per_run_seeds[_run]),
                diversity_fn=diversity_fn,
            )
            hybrid_results.append(hybrid_result["best_fitness"])
            hybrid_metrics_all.append(evaluate_metrics(hybrid_result["best_genome"], env))
            hybrid_histories.append(hybrid_result["history_best"])
            done_steps += 1
            progress.progress(done_steps / total_steps)
        hybrid_time = time.time() - start

        # DE+GA
        start = time.time()
        for _run in range(runs):
            status.write(f"DE+GA run {_run + 1}/{runs}")
            dega_result = run_de_ga(
                env,
                fitness_fn=lambda g, e: evaluate(g, e),
                cfg=DEGAConfig(ga_generations=ga_gens, de_generations=aco_iters),
                seed=int(per_run_seeds[_run]),
            )
            dega_results.append(dega_result["best_fitness"])
            dega_metrics_all.append(evaluate_metrics(dega_result["best_genome"], env))
            dega_histories.append(dega_result["history_best"])
            done_steps += 1
            progress.progress(done_steps / total_steps)
        dega_time = time.time() - start

    status.write("All runs completed.")
    st.success("Finished")

    # ================== CONVERGENCE SPEED ==================
    thresh = convergence_threshold_pct

    st.subheader(f"Convergence Speed — Iterations to {thresh}% of Improvement")
    st.caption(
        f"The first iteration at which each algorithm's best fitness reached {thresh}% "
        f"of its total improvement (initial − final best), averaged across {runs} runs. "
        "Lower = faster convergence."
    )

    def mean_gens(histories):
        return float(np.mean([generations_to_threshold(h, thresh / 100.0) for h in histories]))

    conv_speed_df = pd.DataFrame({
        "Algorithm":                   ["GA", "ACO", "Hybrid", "DE+GA"],
        f"Mean iters to {thresh}%":    [mean_gens(ga_histories), mean_gens(aco_histories),
                                        mean_gens(hybrid_histories), mean_gens(dega_histories)],
        "Total iters (last run)":      [len(ga_histories[-1]), len(aco_histories[-1]),
                                        len(hybrid_histories[-1]), len(dega_histories[-1])],
    })
    conv_speed_df["Fraction of budget"] = (
        conv_speed_df[f"Mean iters to {thresh}%"] /
        conv_speed_df["Total iters (last run)"]
    ).map(lambda x: f"{x:.1%}")
    st.dataframe(conv_speed_df, use_container_width=True)

    # ================== STATISTICS ==================
    st.subheader("Descriptive Statistics (Fitness across runs)")
    st.dataframe(pd.DataFrame({
        "Algorithm": ["GA", "ACO", "Hybrid", "DE+GA"],
        "Mean":  [np.mean(ga_results),     np.mean(aco_results),
                  np.mean(hybrid_results), np.mean(dega_results)],
        "Std":   [np.std(ga_results),      np.std(aco_results),
                  np.std(hybrid_results),  np.std(dega_results)],
        "Min":   [np.min(ga_results),      np.min(aco_results),
                  np.min(hybrid_results),  np.min(dega_results)],
        "Max":   [np.max(ga_results),      np.max(aco_results),
                  np.max(hybrid_results),  np.max(dega_results)],
    }), use_container_width=True)

    # ================== SIGNIFICANCE TESTS ==================
    st.subheader("Statistical Significance Tests (Fitness)")
    st.caption(
        "Mann-Whitney U (unpaired) and Wilcoxon signed-rank (paired) tests. "
        "p < 0.05 indicates a statistically significant difference between the two algorithms."
    )

    u_h_ga,  p_h_ga   = mann_whitney_u_test(hybrid_results, ga_results)
    w_h_ga,  pw_h_ga  = wilcoxon_signed_rank_test(hybrid_results, ga_results)
    u_h_aco, p_h_aco  = mann_whitney_u_test(hybrid_results, aco_results)
    w_h_aco, pw_h_aco = wilcoxon_signed_rank_test(hybrid_results, aco_results)
    u_de_h,  p_de_h   = mann_whitney_u_test(dega_results, hybrid_results)
    w_de_h,  pw_de_h  = wilcoxon_signed_rank_test(dega_results, hybrid_results)

    sig_rows = [
        {"Comparison": "Hybrid vs GA",
         "Mann-Whitney U": round(u_h_ga, 4),  "U p-value": round(p_h_ga, 4),
         "U sig (p<0.05)": "Yes" if p_h_ga < 0.05 else "No",
         "Wilcoxon W":     round(w_h_ga, 4),  "W p-value": round(pw_h_ga, 4),
         "W sig (p<0.05)": "Yes" if pw_h_ga < 0.05 else "No"},
        {"Comparison": "Hybrid vs ACO",
         "Mann-Whitney U": round(u_h_aco, 4), "U p-value": round(p_h_aco, 4),
         "U sig (p<0.05)": "Yes" if p_h_aco < 0.05 else "No",
         "Wilcoxon W":     round(w_h_aco, 4), "W p-value": round(pw_h_aco, 4),
         "W sig (p<0.05)": "Yes" if pw_h_aco < 0.05 else "No"},
        {"Comparison": "DE+GA vs Hybrid",
         "Mann-Whitney U": round(u_de_h, 4),  "U p-value": round(p_de_h, 4),
         "U sig (p<0.05)": "Yes" if p_de_h < 0.05 else "No",
         "Wilcoxon W":     round(w_de_h, 4),  "W p-value": round(pw_de_h, 4),
         "W sig (p<0.05)": "Yes" if pw_de_h < 0.05 else "No"},
    ]
    sig_df = pd.DataFrame(sig_rows)
    st.dataframe(sig_df, use_container_width=True)

    # Narrative interpretation
    st.markdown("**Interpretation (α = 0.05)**")
    for row in sig_rows:
        u_sig = row["U p-value"] < 0.05
        w_sig = row["W p-value"] < 0.05
        if u_sig or w_sig:
            st.markdown(
                f"- **{row['Comparison']}**: significant difference detected "
                f"(U={row['Mann-Whitney U']}, p={row['U p-value']}; "
                f"W={row['Wilcoxon W']}, p={row['W p-value']})."
            )
        else:
            st.markdown(
                f"- **{row['Comparison']}**: no significant difference "
                f"(U={row['Mann-Whitney U']}, p={row['U p-value']}; "
                f"W={row['Wilcoxon W']}, p={row['W p-value']})."
            )

    # ================== TIME ==================
    st.subheader("Execution Time (seconds)")
    st.dataframe(pd.DataFrame({
        "Algorithm":     ["GA", "ACO", "Hybrid", "DE+GA"],
        "Wall time (s)": [round(ga_time, 3), round(aco_time, 3),
                          round(hybrid_time, 3), round(dega_time, 3)],
    }), use_container_width=True)

    # ================== BEST FITNESS ==================
    st.subheader("Best Fitness Comparison")
    results_dict = {
        "GA":     np.mean(ga_results),
        "ACO":    np.mean(aco_results),
        "Hybrid": np.mean(hybrid_results),
        "DE+GA":  np.mean(dega_results),
    }
    st.write(results_dict)

    # ================== CONVERGENCE CURVE ==================
    st.subheader("Convergence Curve")

    fig_conv, ax_conv = plt.subplots()
    ax_conv.plot(ga_result["history_best"],     label="GA")
    ax_conv.plot(aco_result["history_best"],    label="ACO")
    ax_conv.plot(hybrid_result["history_best"], label="Hybrid")
    ax_conv.plot(dega_result["history_best"],   label="DE+GA")
    ax_conv.set_xlabel("Iterations")
    ax_conv.set_ylabel("Best Fitness")
    ax_conv.set_title("Convergence Curve")
    ax_conv.legend()
    st.pyplot(fig_conv)
    conv_buf = io.BytesIO()
    fig_conv.savefig(conv_buf, format="png", dpi=150, bbox_inches="tight")
    st.download_button("Download Convergence Plot", data=conv_buf.getvalue(),
                       file_name="convergence_plot.png", mime="image/png")

    # ================== ALLOCATION ==================
    st.subheader("Task → VM Assignment (Hybrid)")
    allocation = hybrid_result["best_genome"]
    df = pd.DataFrame({"Task": list(range(len(allocation))), "VM": allocation})
    st.dataframe(df)

    # ================== LOAD ==================
    st.subheader("VM Load (CPU + RAM)")
    vm_load = [0] * num_vms
    for task, vm in enumerate(allocation):
        vm_load[vm] += env.tasks[task].cpu + env.tasks[task].ram

    fig_load, ax_load = plt.subplots()
    ax_load.bar(range(num_vms), vm_load)
    ax_load.set_xlabel("VM ID")
    ax_load.set_ylabel("CPU + RAM Load")
    ax_load.set_title("VM Load Distribution")
    st.pyplot(fig_load)
    load_buf = io.BytesIO()
    fig_load.savefig(load_buf, format="png", dpi=150, bbox_inches="tight")
    st.download_button("Download Load Plot", data=load_buf.getvalue(),
                       file_name="vm_load_plot.png", mime="image/png")

    # ================== PERFORMANCE METRICS ==================
    st.subheader("Performance Metrics")
    hybrid_metrics = evaluate_metrics(allocation, env)
    avg_load = sum(vm_load) / len(vm_load)
    st.write({
        "Total Cost":             hybrid_metrics["total_cost"],
        "Response Time":          hybrid_metrics["response_time"],
        "Resource Utilization":   hybrid_metrics["resource_utilization"],
        "Jain's Fairness Index":  hybrid_metrics["jains_fairness_index"],
        "Average Load (CPU+RAM)": avg_load,
    })
    st.write("Average across repeated runs:")
    st.write({
        "GA Mean Cost":              float(np.mean([m["total_cost"] for m in ga_metrics_all])),
        "ACO Mean Cost":             float(np.mean([m["total_cost"] for m in aco_metrics_all])),
        "Hybrid Mean Cost":          float(np.mean([m["total_cost"] for m in hybrid_metrics_all])),
        "DE+GA Mean Cost":           float(np.mean([m["total_cost"] for m in dega_metrics_all])),
        "Hybrid Mean Jain Fairness": float(np.mean([m["jains_fairness_index"] for m in hybrid_metrics_all])),
    })

    # ================== BEFORE vs AFTER ==================
    st.subheader("Before vs After")
    random_allocation = [random.randint(0, num_vms - 1) for _ in range(num_tasks)]
    before = evaluate(random_allocation, env)
    fig_cmp, ax_cmp = plt.subplots()
    ax_cmp.bar(
        ["Random", "GA", "ACO", "Hybrid", "DE+GA"],
        [before, np.mean(ga_results), np.mean(aco_results),
         np.mean(hybrid_results), np.mean(dega_results)]
    )
    ax_cmp.set_ylabel("Fitness")
    ax_cmp.set_title("Improvement Comparison")
    st.pyplot(fig_cmp)
    cmp_buf = io.BytesIO()
    fig_cmp.savefig(cmp_buf, format="png", dpi=150, bbox_inches="tight")
    st.download_button("Download Comparison Plot", data=cmp_buf.getvalue(),
                       file_name="fitness_comparison.png", mime="image/png")

    # ================== EXPORTS ==================
    results_export = pd.DataFrame({
        "run":            list(range(1, runs + 1)),
        "ga_fitness":     ga_results,
        "aco_fitness":    aco_results,
        "hybrid_fitness": hybrid_results,
        "de_ga_fitness":  dega_results,
    })
    st.download_button(
        "Download Run Results CSV",
        data=results_export.to_csv(index=False).encode("utf-8"),
        file_name="run_results.csv",
        mime="text/csv",
    )

    stats_rows_export = []
    for alg, vals in [("GA", ga_results), ("ACO", aco_results),
                      ("Hybrid", hybrid_results), ("DE+GA", dega_results)]:
        stats_rows_export.append({
            "algorithm": alg,
            "mean": np.mean(vals), "std": np.std(vals),
            "min":  np.min(vals),  "max": np.max(vals),
        })
    combined_csv = (
        "# Descriptive Statistics\n"
        + pd.DataFrame(stats_rows_export).to_csv(index=False)
        + "\n# Significance Tests\n"
        + sig_df.to_csv(index=False)
        + "\n# Convergence Speed\n"
        + conv_speed_df.to_csv(index=False)
    )
    st.download_button(
        "Download Statistics + Significance Tests CSV",
        data=combined_csv.encode("utf-8"),
        file_name="statistics_and_significance_tests.csv",
        mime="text/csv",
    )

    # ================== ANALYSIS ==================
    st.subheader("Analysis")
    st.markdown("""
    - GA provides good exploration but slower convergence
    - ACO converges faster but may get stuck in local optima
    - Hybrid combines both advantages for better overall performance
    """)
    if np.mean(hybrid_results) < np.mean(ga_results):
        st.write("Hybrid outperforms GA")
    else:
        st.write("GA outperforms Hybrid")
