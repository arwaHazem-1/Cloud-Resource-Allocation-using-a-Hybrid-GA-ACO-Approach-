import streamlit as st
import random
import pandas as pd
import matplotlib.pyplot as plt
import time
import numpy as np

from algorithms.ga import run_ga, GAConfig
from algorithms.aco import run_aco, ACOConfig
from algorithms.hybrid import run_hybrid, HybridConfig
from environment.cloud_model import CloudEnvironment, Task, VM
from fitness.evaluator import evaluate

st.set_page_config(layout="wide")

st.title("Hybrid GA-ACO Cloud Resource Allocation")
st.markdown("""
    ### Problem Description
    We allocate tasks to VMs under CPU and RAM constraints to minimize cost and improve load balancing.
    """)

st.sidebar.header("Parameters")

num_tasks = st.sidebar.slider("Tasks", 5, 100, 20)
num_vms = st.sidebar.slider("VMs", 2, 20, 5)
cycles = st.sidebar.slider("Hybrid Cycles", 1, 10, 4)
ga_gens = st.sidebar.slider("GA Generations", 10, 200, 50)
aco_iters = st.sidebar.slider("ACO Iterations", 10, 200, 50)

seed = st.sidebar.number_input("Seed (0 = random)", value=0, key="seed_input")

if seed == 0:
    seed = random.randint(1, 100000)

random.seed(seed)

tasks = [
    Task(
        i,
        cpu=random.randint(1, 10),
        ram=random.randint(1, 16),
        length=random.randint(10, 100)
    )
    for i in range(num_tasks)
]

vms = [
    VM(
        i,
        cpu_capacity=random.randint(10, 50),
        ram_capacity=random.randint(16, 64),
        cost_per_time=random.uniform(0.1, 1.0),
        speed=random.uniform(1.0, 3.0)
    )
    for i in range(num_vms)
]

env = CloudEnvironment(tasks, vms)

if st.button("Run All Algorithms"):

    runs = 5

    ga_results = []
    aco_results = []
    hybrid_results = []

    with st.spinner("Running algorithms..."):

        # GA
        start = time.time()
        for _ in range(runs):
            ga_result = run_ga(
                env,
                fitness_fn=lambda g, e: evaluate(g, e),
                cfg=GAConfig(generations=ga_gens),
                seed=random.randint(1, 100000)
            )
            ga_results.append(ga_result["best_fitness"])
        ga_time = time.time() - start

        # ACO
        start = time.time()
        for _ in range(runs):
            aco_result = run_aco(
                env,
                fitness_fn=lambda g, e: evaluate(g, e),
                cfg=ACOConfig(n_iterations=aco_iters),
                seed=random.randint(1, 100000)
            )
            aco_results.append(aco_result["best_fitness"])
        aco_time = time.time() - start

        # Hybrid
        start = time.time()
        for _ in range(runs):
            hybrid_result = run_hybrid(
                env,
                fitness_fn=lambda g, e: evaluate(g, e),
                cfg=HybridConfig(
                    n_cycles=cycles,
                    ga_gens_per_cycle=ga_gens,
                    aco_iters_per_cycle=aco_iters
                ),
                seed=random.randint(1, 100000)
            )
            hybrid_results.append(hybrid_result["best_fitness"])
        hybrid_time = time.time() - start

    st.success("Finished")

    # ================== STATISTICS ==================
    st.subheader("Statistical Analysis")
    st.write({
        "GA Mean": np.mean(ga_results),
        "GA Std": np.std(ga_results),
        "ACO Mean": np.mean(aco_results),
        "ACO Std": np.std(aco_results),
        "Hybrid Mean": np.mean(hybrid_results),
        "Hybrid Std": np.std(hybrid_results),
    })

    # ================== TIME ==================
    st.subheader("Execution Time (seconds)")
    st.write({
        "GA": ga_time,
        "ACO": aco_time,
        "Hybrid": hybrid_time
    })

    # RESPONSE TIME
    response_time = hybrid_time
    st.write({"Response Time": response_time})

    # ================== BEST FITNESS ==================
    st.subheader("Best Fitness Comparison")

    results_dict = {
        "GA": np.mean(ga_results),
        "ACO": np.mean(aco_results),
        "Hybrid": np.mean(hybrid_results)
    }

    st.write(results_dict)

    # ================== CONVERGENCE ==================
    st.subheader("Convergence Curve")

    plt.figure()

    plt.plot(ga_result["history_best"], label="GA")
    plt.plot(aco_result["history_best"], label="ACO")
    plt.plot(hybrid_result["history_best"], label="Hybrid")

    plt.xlabel("Iterations")
    plt.ylabel("Best Fitness")
    plt.title("Convergence Curve")
    plt.legend()

    st.pyplot(plt)

    # ================== ALLOCATION ==================
    st.subheader("Task → VM Assignment (Hybrid)")

    allocation = hybrid_result["best_genome"]

    df = pd.DataFrame({
        "Task": list(range(len(allocation))),
        "VM": allocation
    })

    st.dataframe(df)
    df.to_csv("allocation_results.csv", index=False)

    # ================== LOAD ==================
    st.subheader("VM Load (CPU Usage)")

    vm_load = [0] * num_vms
    for task, vm in enumerate(allocation):
        vm_load[vm] += env.tasks[task].cpu + env.tasks[task].ram

    plt.figure()
    plt.bar(range(num_vms), vm_load)

    plt.xlabel("VM ID")
    plt.ylabel("CPU + RAM Load")
    plt.title("VM Load Distribution")

    st.pyplot(plt)

    # ================== PERFORMANCE ==================
    st.subheader("Performance Metrics")

    total_cost = sum(vm.cost_per_time for vm in env.vms)
    avg_load = sum(vm_load) / len(vm_load)

    st.write({
        "Total Cost": total_cost,
        "Average Load": avg_load
    })

    # ================== COMPARISON ==================
    st.subheader("Before vs After")

    random_allocation = [random.randint(0, num_vms - 1) for _ in range(num_tasks)]

    before = evaluate(random_allocation, env)

    plt.figure()
    plt.bar(
        ["Random", "GA", "ACO", "Hybrid"],
        [before, np.mean(ga_results), np.mean(aco_results), np.mean(hybrid_results)]
    )

    plt.ylabel("Fitness")
    plt.title("Improvement Comparison")

    st.pyplot(plt)

    # ================== ANALYSIS ==================
    st.subheader("Analysis")

    st.markdown("""
    - GA provides good exploration but slower convergence  
    - ACO converges faster but may get stuck in local optimum  
    - Hybrid combines both advantages for better performance  
    """)

    # DYNAMIC ANALYSIS
    if np.mean(hybrid_results) < np.mean(ga_results):
        st.write("Hybrid outperforms GA")
    else:
        st.write("GA outperforms Hybrid")