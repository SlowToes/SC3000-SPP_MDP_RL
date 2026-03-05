from grid_world import GridWorld
from tabular_policy import TabularDeterministicPolicy
from policy_iteration import PolicyIteration
from tabular_value_function import TabularValueFunction
from value_iteration import ValueIteration


def part1():
    gridworld = GridWorld()

    # Task 1 specifies deterministic transitions
    gridworld.noise = 0.0

    values = TabularValueFunction()
    iterations = ValueIteration(gridworld, values).value_iteration(max_iterations=100)
    policy = values.extract_policy(gridworld)

    print(f"Value Iteration completed in {iterations} iterations.")
    gridworld.visualise_value_function(values, f"Value function after iteration {iterations}")
    gridworld.visualise_policy(policy, f"Policy after iteration {iterations}")


def part2():
    gridworld = GridWorld()

    # Task 1 specifies deterministic transitions
    gridworld.noise = 0.0

    policy = TabularDeterministicPolicy()
    iterations, values = PolicyIteration(gridworld, policy).policy_iteration(max_iterations=100)

    print(f"Policy Iteration completed in {iterations} iterations.")
    gridworld.visualise_value_function(values, f"Value function after iteration {iterations}")
    gridworld.visualise_policy(policy, f"Policy after iteration {iterations}")


def main():
    part1()
    part2()


if __name__ == "__main__":
    main()