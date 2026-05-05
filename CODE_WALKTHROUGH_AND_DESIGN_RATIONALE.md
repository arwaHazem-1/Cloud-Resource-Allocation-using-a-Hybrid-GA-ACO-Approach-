# Code Walkthrough and Design Rationale

This file explains each major code part, the key functions, and why specific algorithmic choices (especially mutation/crossover) were used.

## 1) Project Structure

- `environment/`
  - `cloud_model.py`: data models (`Task`, `VM`, `CloudEnvironment`)
  - `dataset_loader.py`: synthetic task/VM generation
  - `standard_workload.py`: synthetic + trace workload construction
- `fitness/`
  - `evaluator.py`: objective, penalties, and evaluation metrics
- `algorithms/`
  - `operators.py`: GA operators
  - `ga.py`: GA engine
  - `aco.py`: ACO engine (AS/ACS)
  - `hybrid.py`: GA+ACO hybrid cycle
  - `de.py`: DE operators
  - `de_ga_hybrid.py`: DE+GA hybrid runner
- `diversity/`
  - `diversity.py`: fitness sharing + island model
- `experiments/`
  - `runner.py`: GA/ACO seeded batch runs
  - `hybrid_runner.py`: hybrid variants batch runs
  - `comparative_runner.py`: multi-scenario 4-way comparison + significance tests
- `app.py`: Streamlit UI.

---

## 2) Environment and Data Models

## `environment/cloud_model.py`

- `Task`
  - attributes: `id`, `cpu`, `ram`, `length`
- `VM`
  - attributes: `id`, `cpu_capacity`, `ram_capacity`, `cost_per_time`, `speed`
- `CloudEnvironment`
  - bundles tasks and vms into one object for all evaluators and algorithms.

Why this design:
- Keeps algorithms independent from data source and UI.
- Makes algorithms reusable across synthetic and trace workloads.

## `environment/dataset_loader.py`

- `generate_tasks(n_tasks, rng)`
- `generate_vms(n_vms, rng)`

Why:
- controlled random generation for reproducibility and fast benchmarking.

## `environment/standard_workload.py`

- `load_trace_tasks(trace_csv, n_tasks, rng)`
- `build_workload(workload_mode, n_tasks, n_vms, rng, trace_csv)`

Why:
- Adds non-synthetic workload path for stronger experiment realism.

---

## 3) Fitness and Metrics

## `fitness/evaluator.py`

Main functions:
- `evaluate_components(individual, env, cfg)`
  - returns `(fitness, total_cost, response_time, penalty)`
- `evaluate(individual, env, cfg)`
  - scalar objective for optimizers
- `evaluate_metrics(individual, env, cfg)`
  - expanded KPI dictionary (cost/time/utilization/fairness)
- `_jains_fairness(values)`
  - load balancing score in `[0, 1]`.

Why weighted single-objective:
- compatible with GA/ACO/DE directly
- still preserves component metrics for reporting and analysis.

Why soft penalties instead of hard rejection:
- infeasible candidates still guide search gradients
- avoids collapsing exploration early.

---

## 4) GA Implementation Details

## `algorithms/operators.py` + `algorithms/ga.py`

Key operators:
- Selection:
  - tournament
  - roulette
- Crossover:
  - one-point
  - uniform
- Mutation:
  - random-reset
  - swap

### Why these mutation types?

- **Random-reset mutation**
  - changes selected gene(s) to random VM IDs.
  - strong exploration, useful when population is trapped.
  - good for discrete integer encoding.

- **Swap mutation**
  - swaps VM assignments between two task positions.
  - smaller structural perturbation than random-reset.
  - preserves many existing assignment patterns.

Using both gives a practical exploration/exploitation balance in discrete allocation search.

### Why not Gaussian or polynomial mutation?

- Those are designed for continuous vectors.
- Here genomes are integer VM indices; discrete mutations are more natural and stable.

### Why one-point and uniform crossover?

- **One-point** preserves contiguous genome segments (building blocks).
- **Uniform** mixes parent genes more aggressively and can escape local patterns.
- Together they provide complementary recombination behavior.

### Why tournament/roulette selection?

- Tournament offers robust pressure and is easy to tune.
- Roulette introduces probabilistic sampling proportional to quality.
- Both are standard, interpretable, and lightweight for educational and comparative settings.

### Additional GA choices

- Elitism preserves top solutions across generations.
- Heuristic-seeded init improves initial feasibility and convergence speed.
- Patience-based early stop reduces wasted iterations.

---

## 5) ACO Implementation Details

## `algorithms/aco.py`

Core functions:
- `build_ant_solution(...)`
- `local_search_assignment(...)`
- `_evaporate(...)`
- `_deposit_global_best(...)`
- `run_aco(...)`

Key logic:
- Pheromone matrix `tau[task][vm]`
- Probabilistic assignment via `tau^alpha * eta^beta`
- `eta` derived from marginal score (cost/time/feasibility)
- optional ACS behavior with local pheromone updates.

Why local search is included:
- Construction can be noisy; local search improves each ant's final assignment quality.

Why AS + ACS both supported:
- AS is simpler and stable baseline.
- ACS offers stronger exploitation and often faster convergence.

---

## 6) Hybrid GA+ACO Design

## `algorithms/hybrid.py`

Cycle architecture:
1. Run GA phase.
2. Seed pheromone from top-K GA solutions.
3. Run ACO phase on warmed pheromone.
4. Inject ACO best genome back into GA population.

Why this hybrid is effective:
- GA explores globally across population space.
- ACO intensifies around promising assignment edges.
- Injection closes the loop so both methods reinforce each other.

Diversity integration:
- `diversity_fn` hook allows plug-in mechanisms without changing core hybrid logic.

---

## 7) Differential Evolution Components

## `algorithms/de.py`

Functions:
- `de_mutation_rand1(x_a, x_b, x_c, F, n_vms)`
- `de_binomial_crossover(target, donor, CR, rng)`
- `de_generate_trial(population, target_idx, F, CR, n_vms, rng)`

Discrete adaptation:
- mutated values are rounded and clipped to valid VM index range.

## `algorithms/de_ga_hybrid.py`

Function:
- `run_de_ga(env, fitness_fn, cfg, seed)`

Flow:
1. GA phase for broad exploration.
2. DE phase for refinement through vector-difference perturbation and greedy replacement.

Why GA then DE:
- GA quickly builds diverse useful structures.
- DE can then exploit directional differences between good candidates.

---

## 8) Diversity Mechanisms

## `diversity/diversity.py`

- `fitness_sharing(population, env, sigma, alpha, ...)`
  - increases effective fitness in crowded niches (for minimization, makes crowded solutions less preferred).

- `IslandModel(...)`
  - partitions population into islands and migrates top individuals periodically.

Why these two:
- Fitness sharing combats local overcrowding continuously.
- Island model preserves separated sub-searches and reduces premature homogenization.

---

## 9) Experiment and Statistical Analysis

## `experiments/comparative_runner.py`

Provides:
- multi-scenario benchmarking:
  - `low_load`, `medium_load`, `high_load`
- 4-way algorithm comparison:
  - `GA`, `ACO`, `Hybrid`, `DE+GA`
- repeated seeded runs
- metrics aggregation
- significance tests:
  - Mann-Whitney U (non-parametric independent)
  - Wilcoxon signed-rank (paired non-parametric).

Why non-parametric tests:
- fitness distributions are often non-Gaussian and noisy in evolutionary algorithms.

---

## 10) Streamlit UI

## `app.py`

What it does:
- parameter setup
- workload mode selection
- algorithm runs with progress
- plot rendering
- significance table
- downloadable outputs.

Why this UI design:
- enables demonstration + reproducibility + artifact export from one place.

---

## 11) Why these methods and not others?

- **Why not NSGA-II / true multi-objective Pareto?**
  - current scope emphasizes one scalar fitness plus reported components; simpler for consistent GA/ACO/DE integration.

- **Why not PSO instead of DE?**
  - rubric explicitly references DE+GA in one interpretation, so DE was prioritized.

- **Why not strict hard constraints only?**
  - pure rejection can reduce search diversity and stall early generations.

- **Why not only one algorithm?**
  - the project objective is comparative and hybrid; individual + hybrid baselines are required for meaningful analysis.

---

## 12) Suggested Next Improvements (optional)

- Add SLA/deadline constraints as either penalty or hard feasibility filter.
- Add energy-aware objective term.
- Integrate larger external traces.
- Auto-generate a PDF report from final experiment CSV/JSON.
