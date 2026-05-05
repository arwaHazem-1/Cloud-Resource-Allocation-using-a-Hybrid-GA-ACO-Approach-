# Final Evaluation and Project Details

## Final Rubric Evaluation

### 1) Problem Formulation and Cloud Modelling (4/4)

Score: **4 / 4**

Why:
- Cloud entities are clearly modeled:
  - `Task(task_id, cpu, ram, length)` in `environment/cloud_model.py`
  - `VM(vm_id, cpu_capacity, ram_capacity, cost_per_time, speed)` in `environment/cloud_model.py`
  - `CloudEnvironment(tasks, vms)` as the shared simulation container.
- Objective is formally implemented in `fitness/evaluator.py` with weighted optimization:
  - `fitness = w_cost * total_cost + w_time * response_time + w_penalty * penalty`
- Resource constraints are explicitly enforced through overflow penalties on CPU and RAM.
- The mathematical formulation and assumptions are documented in `README.md`.

---

### 2) Implementation of GA Algorithm (3/3)

Score: **3 / 3**

Why:
- GA core is complete in `algorithms/ga.py`:
  - selection: tournament + roulette
  - crossover: one-point + uniform
  - mutation: random-reset + swap
  - survivor strategy: elitism + generational behavior
  - initialization: random + heuristic-seeded
  - early stopping: patience-based stagnation.
- Population history (`history_best`, `history_mean`) is captured for convergence analysis.

---

### 3) Implementation of ACO Algorithm (3/3)

Score: **3 / 3**

Why:
- ACO implementation in `algorithms/aco.py` includes:
  - pheromone matrix per `(task, vm)` pair
  - evaporation and global-best deposit
  - AS and ACS variants (`variant="AS" | "ACS"`)
  - probabilistic construction using `tau^alpha * eta^beta`
  - post-construction local search refinement
  - patience-based early stop.
- Heuristic (`eta`) combines incremental cost, time impact, and feasibility pressure.

---

### 4) Hybrid Evolutionary Algorithm (DE + GA) (3/3)

Score: **3 / 3**

Why:
- Explicit DE operators implemented in `algorithms/de.py`:
  - DE mutation: `x_a + F * (x_b - x_c)`
  - binomial crossover with `CR`
- Explicit DE+GA wrapper implemented in `algorithms/de_ga_hybrid.py`:
  - GA phase for exploration
  - DE phase for refinement
  - exported as `run_de_ga(...)`.
- GA+ACO hybrid is also implemented in `algorithms/hybrid.py` and retained for the main project objective.

---

### 5) Experimental Design and Workload Simulation (3/3)

Score: **3 / 3**

Why:
- Repeated seeded runs are supported with deterministic seed lists (`experiments/seeds.txt`).
- Multiple scenarios are explicitly defined and labeled in `experiments/comparative_runner.py`:
  - `low_load` (20 tasks / 4 VMs)
  - `medium_load` (50 tasks / 10 VMs)
  - `high_load` (100 tasks / 20 VMs)
- Supports both:
  - synthetic workloads
  - trace-based workloads via `environment/standard_workload.py` and `data/google_cluster_sample.csv`.
- Batch comparison runner outputs consistent multi-scenario results.

---

### 6) Comparative Analysis Between Algorithms (3/3)

Score: **3 / 3**

Why:
- Comparative runner includes 4-way evaluation:
  - `GA`, `ACO`, `Hybrid (GA+ACO)`, `DE+GA`.
- Statistical significance tests included:
  - Mann-Whitney U
  - Wilcoxon signed-rank.
- Comparative output files:
  - `results/comparative_head_to_head.csv`
  - `results/comparative_significance_tests.csv`
  - `results/comparative_multi_scenario.json`
- UI also exposes algorithm comparison and significance summaries.

---

### 7) Performance Metrics and Evaluation (3/3)

Score: **3 / 3**

Why:
- Metrics implemented in `fitness/evaluator.py`:
  - fitness
  - total cost
  - response time
  - penalty
  - CPU/RAM utilization
  - resource utilization
  - Jain's fairness index.
- Convergence behavior tracked and visualized.
- Statistical analysis present in both CLI experiments and UI.

---

### 8) UI (2/2)

Score: **2 / 2**

Why:
- Streamlit app (`app.py`) provides:
  - parameter controls
  - ACO variant control (AS/ACS)
  - hybrid diversity controls (none / fitness sharing / island model)
  - workload mode selection (synthetic / trace)
  - repeated runs with progress bar and status updates
  - convergence, load, and comparison plots
  - significance table
  - CSV and PNG download/export buttons.

---

## Final Total

**24 / 24**

---

## Detailed Project Explanation

## A) Problem Context

Cloud systems must assign many tasks to limited VM resources while balancing:
- cost,
- response time,
- and load distribution.

This project treats allocation as a combinatorial optimization problem over integer assignments (`task -> vm`).

## B) Core Model

- Each task has CPU, RAM, and execution length demand.
- Each VM has CPU/RAM capacities, processing speed, and time-based cost.
- A candidate solution is an integer genome where each index is a task and each value is the selected VM.

## C) Objective and Constraints

Given allocation `x`:
- Compute VM-level CPU/RAM usage and execution times.
- Penalize capacity overflows.
- Minimize weighted sum:
  - total monetary cost
  - response-time metric (makespan or mean VM time)
  - penalty term for infeasibility.

This produces one scalar fitness value while still preserving component metrics for reporting.

## D) Algorithms Used

- **GA**: strong global exploration via recombination and mutation.
- **ACO**: edge-learning over `(task, vm)` assignments with pheromone memory.
- **Hybrid GA+ACO**: alternates GA cycles with ACO refinement, then injects improvements back.
- **Hybrid DE+GA**: explicit DE phase added for rubric compliance and additional search behavior.

## E) Diversity Preservation

To avoid premature convergence in hybrid runs:
- fitness sharing discourages crowded genome niches,
- island model maintains partially isolated sub-populations with migration.

## F) Experiment Pipeline

`experiments/comparative_runner.py` runs repeated seeded experiments over labeled load scenarios and generates:
- summary comparisons,
- significance tests,
- scenario-wise detailed JSON.

It supports synthetic and trace-based tasks for broader validity.

## G) Output and Reporting

The project outputs:
- algorithm-level performance summaries,
- statistical significance comparisons,
- convergence and distribution visualizations,
- UI exports for plots and run-level CSVs.

This gives both quantitative and visual evidence of algorithm behavior.

## H) Practical Strengths

- Modular architecture (model, fitness, algorithms, diversity, experiments, UI).
- Reproducibility through fixed seeds.
- Multiple hybrid strategies and metrics.
- Usable interface for demonstration and result export.

## I) Potential Future Work

- Integrate larger real traces (e.g., filtered Google/Borg style traces).
- Add deadline/SLA hard constraints and energy-aware objectives.
- Add automated report generation from experiment outputs.
