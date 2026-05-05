# Cloud Resource Allocation using Hybrid GA–ACO

## 1) Project Idea
This project models **cloud task-to-VM allocation** as a constrained optimization problem and solves it using:
- a **Genetic Algorithm (GA)** baseline,
- an **Ant Colony Optimization (ACO)** baseline,
- a **hybrid GA→ACO** pipeline where GA explores and ACO refines.

The optimization goals are:
- **minimize monetary cost**,
- **minimize response time** (makespan or mean VM time),
while discouraging infeasible allocations via constraint penalties.

## 2) System Functionalities
- Generate workloads:
  - **synthetic** tasks/VMs,
  - **trace-based** tasks loaded from a CSV sample.
- Run optimizers:
  - GA, ACO (AS/ACS), Hybrid GA–ACO, and DE+GA (extra baseline).
- Produce artifacts:
  - run-level metrics (fitness, cost, response time, utilization, fairness),
  - summary CSV/JSON,
  - convergence plots (where enabled).
- Provide a **Streamlit UI** for interactive demonstrations and exporting results.

## 3) Similar Applications (Real-world relevance)
This formulation maps to common cloud scheduling scenarios:
- **Batch job schedulers** that must trade off cost vs completion time (e.g., spot vs on-demand VM choices).
- **Serverless / microservice placement** where requests are assigned to heterogeneous compute pools.
- **Multi-tenant datacenter allocation** where over-commitment is allowed but penalized (soft constraints).

## 4) Literature Review (4–6 papers)
The following works motivate (i) metaheuristic scheduling in cloud environments, (ii) ACO and GA for task scheduling, and (iii) hybrid GA–ACO designs.

1. **Kim, J., et al.** “A slave ants based ant colony optimization algorithm for task scheduling in cloud computing environments.” *Human-centric Computing and Information Sciences*, 2017. **DOI:** 10.1186/s13673-017-0109-2.
2. **Ilankumaran, A., Narayanan, S. J.** “An Energy-Aware QoS Load Balance Scheduling Using Hybrid GAACO Algorithm for Cloud.” *Cybernetics and Information Technologies*, 2023. **DOI:** 10.2478/cait-2023-0009.
3. **Tang, L., Zhang, X., Li, Z., Zhang, Y.** “A New Hybrid Task Scheduling Algorithm Designed Based on ACO and GA.” *Journal of Information Hiding and Multimedia Signal Processing*, 2018. (Hybrid GA–ACO scheduling; venue paper, DOI not listed in the PDF.)
4. **Ilkhechi, A. S., et al.** “HWACOA Scheduler: Hybrid Weighted Ant Colony Optimization Algorithm for Task Scheduling in Cloud Computing.” *Applied Sciences*, 2023. **DOI:** 10.3390/app13063433.
5. **Ge, Y., Yuan, Q.** “Cloud task scheduling algorithm based on improved genetic algorithm.” *IC CSEE 2013 proceedings* (baseline GA scheduling). **DOI:** 10.2991/iccsee.2013.537.
6. **(Optional classic foundation)** Dorigo & colleagues’ ACO foundations are assumed known and are indirectly cited by most ACO scheduling papers above.

## 5) Dataset / Workload Description
Two workload modes are supported:

### A) Synthetic workloads
- Tasks are generated with integer CPU/RAM demands and workload length.
- VMs are generated with CPU/RAM capacity, speed, and cost per time.
- This mode enables controlled benchmarking, repeatability, and operator studies.

### B) Trace-based workloads
- Tasks are sampled from a CSV file with columns:
  - `task_id,cpu,ram,length`
- A sample trace file is included at `data/google_cluster_sample.csv`.

## 6) Problem Formulation (Objective + Constraints)
Decision variable: for each task \(t_i\), choose a VM index \(x_i\).

Fitness (minimization):
\[
F(x) = w_{cost}\cdot Cost(x) + w_{time}\cdot ResponseTime(x) + w_{penalty}\cdot Penalty(x)
\]

Constraints:
- Each VM has CPU/RAM capacities; allocations exceeding capacity incur overflow penalties (soft constraint handling).

## 7) Algorithms Implemented
### A) Genetic Algorithm (GA)
- Representation: integer vector of length \(n_{tasks}\).
- Selection: tournament / roulette.
- Crossover: one-point / uniform.
- Mutation: random-reset / swap.
- Survivor: elitism (configurable).
- Termination: max generations + stagnation patience.

### B) Ant Colony Optimization (ACO)
- Pheromone matrix \(\tau[task][vm]\).
- AS and ACS variants.
- Dynamic heuristic combining incremental cost, time impact, and infeasibility pressure.
- Local search refinement per ant.
- Termination: max iterations + stagnation patience.

### C) Hybrid GA–ACO
Cycle:
1) GA evolves population
2) Top-K GA solutions seed the pheromone matrix
3) ACO refines on warmed pheromone
4) ACO best is injected back into GA population

### D) Additional baseline: DE+GA
GA exploration followed by integer-adapted DE refinement for additional comparison.

## 8) Experimental Design and Results
Reproducibility:
- Seeds are stored in `experiments/seeds.txt` and used by CLI runners.

Core comparisons:
- GA vs ACO vs Hybrid vs DE+GA across multiple workload scenarios.
- Statistical tests: Mann–Whitney U and Wilcoxon signed-rank.

Operator study (GA):
- `experiments/ga_operator_sweep.py` performs a seeded sweep over:
  - selection × crossover × mutation.

Outputs:
- JSON/CSV summaries in `results/` (and plots where enabled).

## 9) Platform / Tools
- Python implementation.
- Streamlit UI (`app.py`).
- Matplotlib/NumPy/Pandas for plots and analysis in the UI/experiment scripts.

