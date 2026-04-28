# Cloud Resource Allocation using Evolutionary Algorithms

Part 1 (Problem Modeling & Fitness Function)**

## 📌 Project Overview

This project models a cloud computing environment and solves the **resource allocation problem** using evolutionary algorithms (GA + ACO in later stages).

The goal is to assign tasks to virtual machines (VMs) efficiently while minimizing:

* Execution cost
* Total completion time (makespan)
* Constraint violations

## 🧱 System Components

### 1. Tasks

Each task has:

* CPU requirement
* RAM requirement
* Length (execution size)

### 2. Virtual Machines (VMs)

Each VM has:

* CPU capacity
* RAM capacity
* Cost per unit time
* Processing speed


## 🧠 Problem Formulation

The problem is modeled as a **constrained optimization problem**:

* Each task must be assigned to exactly one VM
* VM capacity constraints must not be exceeded
* Objective is to minimize overall system cost and execution time


## 🔢 Solution Representation

A solution is represented as a list:

```
[vm_id_for_task_0, vm_id_for_task_1, ..., vm_id_for_task_n]
```

Example:

```
[0, 1, 2, 0, 1]
```

Meaning:

* Task 0 → VM 0
* Task 1 → VM 1
* Task 2 → VM 2
* ...


## 📊 Fitness Function

The fitness function evaluates a solution based on:

```
Fitness = Total Cost + Makespan + Penalty
```

Where:

* **Total Cost** = Σ (execution time × VM cost)
* **Makespan** = max execution time among VMs
* **Penalty** = constraint violations (CPU / RAM overflow)

Penalty is scaled to heavily punish invalid solutions.


## ⚙️ How It Works

1. Generate random tasks and VMs
2. Create a random allocation (solution)
3. Evaluate the solution using the fitness function
4. Lower fitness = better solution


## 🧪 Example Output

```
Solution: [0, 1, 2, 0, 1, 1, 0]
Fitness: 4523.76
```

---

## 📁 Project Structure

```
project/
│
├── environment/
│   ├── cloud_model.py
│   └── dataset_loader.py
│
├── fitness/
│   └── evaluator.py
│
├── experiments/
│   └── test_member1.py
```

---

## 🚀 Future Work

* Implement Genetic Algorithm (GA)
* Integrate Ant Colony Optimization (ACO)
* Hybrid GA–ACO optimization
* Performance comparison & visualization


## 💡 Notes

* Lower fitness values indicate better solutions
* Penalty ensures invalid allocations are discouraged
* This module serves as the foundation for all optimization algorithms


## 🏁 Conclusion

This module establishes a complete simulation environment and evaluation framework for cloud resource allocation, forming the foundation for evolutionary optimization techniques.
