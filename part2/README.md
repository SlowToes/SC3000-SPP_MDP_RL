# Part 2 – Solving MDP and Reinforcement Learning Problems Using a Grid World

A 5×5 stochastic grid world environment where an agent navigates from (0, 0) to (4, 4), avoiding roadblocks at (2, 1) and (2, 3). Each action incurs a step cost of −1, reaching the goal yields +10, and the discount factor is γ = 0.9.

## Project Structure

```
part2/
├── grid_world.py              # GridWorld environment (MDP)
├── mdp.py                     # Abstract MDP base class
├── policy.py                  # Abstract policy classes (Deterministic / Stochastic)
├── tabular_policy.py          # Tabular policy implementations
├── qfunction.py               # Abstract Q-function base class
├── qtable.py                  # Tabular Q-function (dictionary-based)
├── task1/                     # Task 1: Planning (model-based)
│   ├── task1_main.py          # Entry point
│   ├── value_iteration.py     # Value Iteration algorithm
│   ├── policy_iteration.py    # Policy Iteration algorithm
│   ├── value_function.py      # Abstract value function
│   └── tabular_value_function.py
├── task2/                     # Task 2: Monte Carlo Control (model-free)
│   ├── task2_main.py          # Entry point
│   ├── monte_carlo_control.py # First-visit MC control
│   ├── epsilon_greedy.py      # ε-greedy action selection
│   ├── multi_armed_bandit.py  # Abstract bandit base class
│   └── model_free_learner.py  # Abstract model-free learner
└── task3/                     # Task 3: Q-Learning (model-free)
    ├── task3_main.py          # Entry point
    ├── qlearning.py           # Q-Learning algorithm
    └── temporal_difference_learner.py  # TD learner base class
```

## Tasks

### Task 1 – Value Iteration & Policy Iteration

Solves the grid world as an MDP with **known, deterministic** transitions (noise = 0). Computes optimal value functions and policies using both Value Iteration and Policy Iteration, then compares the resulting policies.

### Task 2 – Monte Carlo Control

The agent has **no access to the transition model**. Uses first-visit Monte Carlo control with ε-greedy exploration (ε = 0.1) to estimate state–action values from complete episodes. The stochastic environment (noise = 0.1) is used.

### Task 3 – Q-Learning

Tabular Q-learning with ε-greedy exploration (ε = 0.1) and a fixed learning rate (α = 0.1). The agent learns by interacting with the stochastic environment (noise = 0.1) without knowledge of the transition model.

## How to Run

All commands should be run from the **project root** directory (the parent of `part2/`).

```bash
# Task 1: Value Iteration & Policy Iteration
python -m part2.task1.task1_main

# Task 2: Monte Carlo Control
python -m part2.task2.task2_main

# Task 3: Q-Learning
python -m part2.task3.task3_main
```

Each task will display matplotlib visualisations of the learned value/Q-function and the extracted policy.

## Dependencies

- Python 3.12+
- NumPy
- Matplotlib
