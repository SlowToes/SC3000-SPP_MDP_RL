from part2.grid_world import GridWorld
from part2.tabular_policy import TabularDeterministicPolicy
from part2.task1.policy_iteration import PolicyIteration
from part2.task1.tabular_value_function import TabularValueFunction
from part2.task1.value_iteration import ValueIteration


def part1():
    gridworld = GridWorld()

    # Task 1 specifies deterministic transitions
    gridworld.noise = 0.0

    values = TabularValueFunction()
    iterations = ValueIteration(gridworld, values).value_iteration(max_iterations=100)
    policy = values.extract_policy(gridworld)

    gridworld.visualise_value_function(values, f"Value function after iteration {iterations}")
    gridworld.visualise_policy(policy, f"Policy after iteration {iterations}")


def part2():
    gridworld = GridWorld()

    # Task 1 specifies deterministic transitions
    gridworld.noise = 0.0

    policy = TabularDeterministicPolicy()
    iterations, values = PolicyIteration(gridworld, policy).policy_iteration(max_iterations=100)

    gridworld.visualise_value_function(values, f"Value function after iteration {iterations}")
    gridworld.visualise_policy(policy, f"Policy after iteration {iterations}")


part1()
part2()