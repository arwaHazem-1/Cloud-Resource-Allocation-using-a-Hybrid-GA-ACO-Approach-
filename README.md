# Cloud Resource Allocation using Hybrid GA-ACO

## Problem Formulation

We model cloud resource allocation as a constrained optimization problem.

- Tasks: `T = {t_1, ..., t_n}`
- VMs: `V = {v_1, ..., v_m}`
- Decision variable: `x_i in {0, ..., m-1}` maps task `t_i` to VM `v_{x_i}`

Each task has:
- CPU demand `cpu_i`
- RAM demand `ram_i`
- workload length `len_i`

Each VM has:
- CPU capacity `CPU_j`
- RAM capacity `RAM_j`
- speed `s_j`
- cost per unit time `c_j`

Execution time of task `i` on VM `j`:

`time_{i,j} = len_i / s_j`

Objective function (minimization):

`F(x) = w_cost * Cost(x) + w_time * ResponseTime(x) + w_penalty * Penalty(x)`

where:
- `Cost(x)` is total execution cost across all VMs
- `ResponseTime(x)` is makespan (or mean VM time)
- `Penalty(x)` is summed overflow of CPU and RAM constraints:
  - CPU overflow: `max(0, usedCPU_j - CPU_j)`
  - RAM overflow: `max(0, usedRAM_j - RAM_j)`

## Implemented Algorithms

- GA (`algorithms/ga.py`): tournament/roulette selection, one-point/uniform crossover, random-reset/swap mutation, elitism, heuristic seeding, early stopping.
- ACO (`algorithms/aco.py`): AS and ACS variants, pheromone evaporation/deposit, dynamic heuristic, local search refinement.
- Hybrid GA-ACO (`algorithms/hybrid.py`): alternating GA -> ACO cycles, top-K GA pheromone seeding, ACO best injection into GA population.
- Hybrid DE+GA (`algorithms/de_ga_hybrid.py`): GA exploration followed by integer-adapted DE refinement (`DE/rand/1/bin`) for explicit DE+GA rubric compatibility.

## Diversity Mechanisms

In the hybrid pipeline, diversity pressure is pluggable through `diversity_fn`:

- Fitness sharing (`diversity/fitness_sharing`)
- Island model migration (`diversity/IslandModel`)

## Workloads and Experimental Design

The project supports:

- Synthetic workload generation (`environment/dataset_loader.py`)
- Trace-based workload mode (`environment/standard_workload.py`)
  - Sample trace file: `data/google_cluster_sample.csv`

Comparative runner:

- `experiments/comparative_runner.py`
- Multi-scenario batch (`20/4`, `30/6`, `50/10`, `100/20` tasks/VMs)
- Repeated seeded runs (`experiments/seeds.txt`)
- Head-to-head GA vs ACO vs Hybrid
- Statistical tests:
  - Mann-Whitney U
  - Wilcoxon signed-rank

## Metrics

Implemented metrics include:

- Fitness
- Total cost
- Response time
- Resource utilization
- Jain's fairness index
- Convergence histories

## UI

Streamlit app (`app.py`) provides:

- Parameter controls for GA/ACO/Hybrid
- ACO variant selection (AS/ACS)
- Diversity method selection
- Workload mode (synthetic/trace)
- Progress bar during repeated runs
- Statistical test table
- Plot and CSV export buttons

## Setup

### Requirements

- Python 3.10 or later

### Install dependencies

```bash
pip install -r requirements.txt
```

All required packages are listed in `requirements.txt`:

| Package | Purpose |
|---|---|
| `streamlit` | Interactive web UI |
| `numpy` | Numerical operations |
| `pandas` | Data manipulation and CSV export |
| `matplotlib` | Convergence and comparison plots |

### Project structure

```
.
├── app.py                        # Streamlit UI entry point
├── requirements.txt              # Python dependencies
├── algorithms/
│   ├── ga.py                     # Genetic Algorithm
│   ├── aco.py                    # Ant Colony Optimisation (AS + ACS)
│   ├── hybrid.py                 # Hybrid GA-ACO pipeline
│   ├── de_ga_hybrid.py           # DE+GA hybrid
│   └── operators.py              # Shared GA operators
├── diversity/
│   └── diversity.py              # Fitness sharing & Island Model
├── environment/
│   ├── cloud_model.py            # Task/VM data classes
│   ├── dataset_loader.py         # Synthetic workload generator
│   └── standard_workload.py      # Workload builder (synthetic + trace)
├── fitness/
│   └── evaluator.py              # Fitness & metrics (cost, response time, Jain's index)
├── experiments/
│   ├── comparative_runner.py     # Batch experiment runner
│   ├── hybrid_runner.py          # Hybrid-specific runner
│   └── seeds.txt                 # Fixed seeds for reproducibility
└── data/
    └── google_cluster_sample.csv # Sample Google cluster trace
```

## Run

### Streamlit UI

```bash
streamlit run app.py
```

### Comparative experiments

Synthetic:

`python experiments/comparative_runner.py --runs 30 --workload-mode synthetic`

Trace-based:

`python experiments/comparative_runner.py --runs 30 --workload-mode trace --trace-csv data/google_cluster_sample.csv`
