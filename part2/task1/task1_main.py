from part2.grid_world import GridWorld
from part2.tabular_policy import TabularDeterministicPolicy
from part2.task1.policy_iteration import PolicyIteration
from part2.task1.tabular_value_function import TabularValueFunction
from part2.task1.value_iteration import ValueIteration


def part1():
    # Initialise the gridworld.
    gridworld = GridWorld()

    # Initialise the value function.
    values = TabularValueFunction()

    # Run the value iteration algorithm.
    iterations = ValueIteration(gridworld, values).value_iteration(max_iterations=1000)

    # Extract the policy from the value function.
    policy = values.extract_policy(gridworld)

    # Visualise the value function and policy.
    gridworld.visualise_value_function(values, f"Value function after iteration {iterations}")
    gridworld.visualise_policy(policy, f"Policy after iteration {iterations}")


def part2():
    # Initialise the gridworld.
    gridworld = GridWorld()

    # Initialise the policy.
    policy = TabularDeterministicPolicy()

    # Run the policy iteration algorithm.
    iterations, values = PolicyIteration(gridworld, policy).policy_iteration(max_iterations=1000)

    # Visualise the value function and policy.
    gridworld.visualise_value_function(values, f"Value function after iteration {iterations}")
    gridworld.visualise_policy(policy, f"Policy after iteration {iterations}")


part1()
part2()