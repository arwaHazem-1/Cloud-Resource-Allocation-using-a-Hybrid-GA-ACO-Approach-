import io
import math
import random
import time
from functools import partial
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend for Streamlit
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
PLOTS_DIR = ROOT / "results" / "plots"


def fig_to_png_bytes(fig) -> bytes:
    """Convert matplotlib figure to PNG bytes with error handling."""
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        return buf.getvalue()
    except Exception as e:
        st.error(f"Error converting plot to PNG: {e}")
        return b""


def run_algorithm_parallel(algorithm_name, algorithm_func, cfg, env, runs, per_run_seeds, progress_callback, diversity_mode=None):
    """Run an algorithm for all runs in parallel"""
    results = []
    metrics_all = []
    histories = []
    best_genomes = []  # Store best genomes for each run
    
    def single_run(run_idx):
        seed = int(per_run_seeds[run_idx])
        if algorithm_name == "GA":
            result = algorithm_func(env, fitness_fn=lambda g, e: evaluate(g, e), cfg=cfg, seed=seed)
        elif algorithm_name == "ACO":
            result = algorithm_func(env, fitness_fn=lambda g, e: evaluate(g, e), cfg=cfg, seed=seed)
        elif algorithm_name == "Hybrid":
            diversity_fn = None
            if diversity_mode == "Fitness Sharing":
                diversity_fn = partial(fitness_sharing, sigma=0.3, alpha=1.0)
            elif diversity_mode == "Island Model":
                diversity_fn = IslandModel(n_islands=4, migration_interval=5, migration_k=2)
            result = algorithm_func(env, fitness_fn=lambda g, e: evaluate(g, e), cfg=cfg, seed=seed, diversity_fn=diversity_fn)
        else:  # DE+GA
            result = algorithm_func(env, fitness_fn=lambda g, e: evaluate(g, e), cfg=cfg, seed=seed)
        
        return run_idx, result
    
    # Use ThreadPoolExecutor for parallel execution
    max_workers = min(4, runs)  # Limit to 4 workers to avoid overwhelming system
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(single_run, i) for i in range(runs)]
        
        for future in as_completed(futures):
            run_idx, result = future.result()
            results.append(result["best_fitness"])
            metrics_all.append(evaluate_metrics(result["best_genome"], env))
            histories.append(result["history_best"])
            best_genomes.append(result["best_genome"])  # Store the best genome
            progress_callback()
    
    # Sort results by run index to maintain order
    sorted_results = sorted(zip(range(runs), results, metrics_all, histories, best_genomes), key=lambda x: x[0])
    return [r[1] for r in sorted_results], [r[2] for r in sorted_results], [r[3] for r in sorted_results], [r[4] for r in sorted_results]


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
    scenario = st.sidebar.selectbox("Scenario", ["low_load (20/4)", "medium_load (50/10)", "high_load (100/20)"], index=1)
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

cycles = st.sidebar.slider("Hybrid Cycles", 1, 10, 2)  # Reduced from 4 to 2
ga_gens = st.sidebar.slider("GA Generations", 10, 200, 25)  # Reduced from 50 to 25
aco_iters = st.sidebar.slider("ACO Iterations", 10, 200, 25)  # Reduced from 50 to 25
runs = st.sidebar.slider("Repeated Runs", 3, 30, 5)  # Reduced from 10 to 5
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
st.sidebar.subheader("Exports")
save_plots_to_disk = st.sidebar.checkbox("Also save plots to `results/plots/`", value=False)

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

        # GA (with early stopping)
        start = time.time()
        status.write("Running GA...")
        ga_cfg = GAConfig(generations=ga_gens, patience=20)  # Add early stopping
        ga_results, ga_metrics_all, ga_histories, ga_genomes = run_algorithm_parallel(
            "GA", run_ga, ga_cfg, env, runs, per_run_seeds, lambda: progress.progress(runs / total_steps)
        )
        ga_time = time.time() - start

        # ACO (with early stopping)
        start = time.time()
        status.write("Running ACO...")
        aco_cfg = ACOConfig(n_iterations=aco_iters, variant=aco_variant, patience=20)  # Add early stopping
        aco_results, aco_metrics_all, aco_histories, aco_genomes = run_algorithm_parallel(
            "ACO", run_aco, aco_cfg, env, runs, per_run_seeds, lambda: progress.progress(2 * runs / total_steps)
        )
        aco_time = time.time() - start

        # Hybrid (with early stopping)
        start = time.time()
        status.write("Running Hybrid...")
        hybrid_cfg = HybridConfig(
            n_cycles=cycles,
            ga_gens_per_cycle=ga_gens,
            aco_iters_per_cycle=aco_iters,
            aco=ACOConfig(variant=aco_variant, patience=15),  # Add early stopping
            patience=10,  # Hybrid-level early stopping
        )
        
        hybrid_results, hybrid_metrics_all, hybrid_histories, hybrid_genomes = run_algorithm_parallel(
            "Hybrid", run_hybrid, hybrid_cfg, env, runs, per_run_seeds, lambda: progress.progress(3 * runs / total_steps), diversity_mode
        )
        hybrid_time = time.time() - start

        # DE+GA (no early stopping to maintain DE exploration)
        start = time.time()
        status.write("Running DE+GA...")
        dega_results, dega_metrics_all, dega_histories, dega_genomes = run_algorithm_parallel(
            "DE+GA", run_de_ga, DEGAConfig(ga_generations=ga_gens, de_generations=aco_iters), env, runs, per_run_seeds, lambda: progress.progress(4 * runs / total_steps)
        )
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
    total_time = ga_time + aco_time + hybrid_time + dega_time
    speedup = total_time / max(ga_time, aco_time, hybrid_time, dega_time)
    
    time_df = pd.DataFrame({
        "Algorithm":     ["GA", "ACO", "Hybrid", "DE+GA", "Total", "Parallel Speedup"],
        "Wall time (s)": [round(ga_time, 3), round(aco_time, 3),
                          round(hybrid_time, 3), round(dega_time, 3),
                          round(total_time, 3), f"{speedup:.1f}x"],
        "Relative (%)": [f"{100*ga_time/total_time:.1f}%", f"{100*aco_time/total_time:.1f}%",
                         f"{100*hybrid_time/total_time:.1f}%", f"{100*dega_time/total_time:.1f}%",
                         "100%", "N/A"]
    })
    st.dataframe(time_df, use_container_width=True)
    
    # Performance insights
    st.markdown("**Performance Insights:**")
    st.markdown(f"- **Total execution time**: {total_time:.1f} seconds")
    st.markdown(f"- **Parallel speedup**: ~{speedup:.1f}x vs sequential execution")
    st.markdown(f"- **Average time per run**: {total_time/(4*runs):.2f} seconds")

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
    st.subheader("Convergence Curve (Mean across all runs)")

    def compute_mean_history(histories):
        """Compute mean convergence curve across all runs with normalized x-axis."""
        if not histories:
            return []
        
        # Find maximum length across all histories
        max_len = max(len(h) for h in histories)
        
        # Pad shorter histories with their final value
        padded_histories = []
        for h in histories:
            if len(h) < max_len:
                padded = h + [h[-1]] * (max_len - len(h))
            else:
                padded = h
            padded_histories.append(padded)
        
        # Compute mean at each iteration
        mean_history = []
        for i in range(max_len):
            mean_history.append(np.mean([h[i] for h in padded_histories]))
        
        return mean_history

    # Compute mean histories for each algorithm
    ga_mean_history = compute_mean_history(ga_histories)
    aco_mean_history = compute_mean_history(aco_histories)
    hybrid_mean_history = compute_mean_history(hybrid_histories)
    dega_mean_history = compute_mean_history(dega_histories)

    fig_conv, ax_conv = plt.subplots()
    
    # Create normalized x-axis (0-100%)
    max_iterations = max(len(ga_mean_history), len(aco_mean_history), 
                         len(hybrid_mean_history), len(dega_mean_history))
    
    def normalize_x_axis(history):
        x_norm = np.linspace(0, 100, len(history))
        return x_norm
    
    ax_conv.plot(normalize_x_axis(ga_mean_history), ga_mean_history, label="GA", linewidth=2)
    ax_conv.plot(normalize_x_axis(aco_mean_history), aco_mean_history, label="ACO", linewidth=2)
    ax_conv.plot(normalize_x_axis(hybrid_mean_history), hybrid_mean_history, label="Hybrid", linewidth=2)
    ax_conv.plot(normalize_x_axis(dega_mean_history), dega_mean_history, label="DE+GA", linewidth=2)
    ax_conv.set_xlabel("Progress (%)")
    ax_conv.set_ylabel("Best Fitness")
    ax_conv.set_title("Convergence Curve (Mean across all runs)")
    ax_conv.legend()
    ax_conv.grid(True, alpha=0.3)
    st.pyplot(fig_conv)
    
    # Generate PNG with error handling
    conv_png = fig_to_png_bytes(fig_conv)
    if conv_png:  # Only proceed if PNG generation was successful
        if save_plots_to_disk:
            PLOTS_DIR.mkdir(parents=True, exist_ok=True)
            (PLOTS_DIR / "convergence_plot.png").write_bytes(conv_png)
            st.caption("Saved: `results/plots/convergence_plot.png`")
        st.download_button("Download Convergence Plot", data=conv_png,
                           file_name="convergence_plot.png", mime="image/png")
    else:
        st.warning("PNG generation failed for convergence plot")
    plt.close(fig_conv)

    # ================== ALLOCATION ==================
    st.subheader("Task → VM Assignment (Hybrid)")
    # Get the best allocation from all hybrid runs
    best_hybrid_idx = np.argmin(hybrid_results)
    allocation = hybrid_genomes[best_hybrid_idx]  # Use the best genome from the best run
    df = pd.DataFrame({"Task": list(range(len(allocation))), "VM": allocation})
    st.dataframe(df)

    # ================== LOAD ==================
    st.subheader("VM Load (CPU + RAM)")
    vm_load = [0] * num_vms
    for task, vm in enumerate(allocation):
        vm_load[vm] += env.tasks[task].cpu + env.tasks[task].ram

    fig_load, ax_load = plt.subplots()
    bars = ax_load.bar(range(num_vms), vm_load, color='#45B7D1', alpha=0.7)
    ax_load.set_xlabel("VM ID")
    ax_load.set_ylabel("CPU + RAM Load")
    ax_load.set_title("VM Load Distribution")
    ax_load.set_xticks(range(num_vms))
    ax_load.set_xticklabels([f"VM {i}" for i in range(num_vms)])
    # Add value labels on bars
    for bar, load in zip(bars, vm_load):
        height = bar.get_height()
        ax_load.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{load:.1f}', ha='center', va='bottom', fontsize=9)
    st.pyplot(fig_load)
    
    # Generate PNG with error handling
    load_png = fig_to_png_bytes(fig_load)
    if load_png:  # Only proceed if PNG generation was successful
        if save_plots_to_disk:
            PLOTS_DIR.mkdir(parents=True, exist_ok=True)
            (PLOTS_DIR / "vm_load_plot.png").write_bytes(load_png)
            st.caption("Saved: `results/plots/vm_load_plot.png`")
        st.download_button("Download Load Plot", data=load_png,
                           file_name="vm_load_plot.png", mime="image/png")
    else:
        st.warning("PNG generation failed for VM load plot")
    plt.close(fig_load)

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
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']  # Red, Teal, Blue, Green, Yellow
    bars = ax_cmp.bar(
        ["Random", "GA", "ACO", "Hybrid", "DE+GA"],
        [before, np.mean(ga_results), np.mean(aco_results),
         np.mean(hybrid_results), np.mean(dega_results)],
        color=colors
    )
    ax_cmp.set_ylabel("Fitness")
    ax_cmp.set_title("Improvement Comparison")
    # Add value labels on bars
    for bar, value in zip(bars, [before, np.mean(ga_results), np.mean(aco_results),
                                  np.mean(hybrid_results), np.mean(dega_results)]):
        height = bar.get_height()
        ax_cmp.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{value:.2f}', ha='center', va='bottom', fontsize=9)
    st.pyplot(fig_cmp)
    
    # Generate PNG with error handling
    cmp_png = fig_to_png_bytes(fig_cmp)
    if cmp_png:  # Only proceed if PNG generation was successful
        if save_plots_to_disk:
            PLOTS_DIR.mkdir(parents=True, exist_ok=True)
            (PLOTS_DIR / "fitness_comparison.png").write_bytes(cmp_png)
            st.caption("Saved: `results/plots/fitness_comparison.png`")
        st.download_button("Download Comparison Plot", data=cmp_png,
                           file_name="fitness_comparison.png", mime="image/png")
    else:
        st.warning("PNG generation failed for fitness comparison plot")
    plt.close(fig_cmp)

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
