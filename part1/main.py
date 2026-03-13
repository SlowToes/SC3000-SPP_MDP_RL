import argparse
from tokenize import ContStr

import Task1
import Task2
import Task3
import Task2_Optimised
import Task3_Optimised


def format_submission_path(path):
    if len(path) >= 2:
        return ["S"] + path[1:-1] + ["T"]
    return path


def print_submission_output(path, distance, energy):
    display_path = format_submission_path(path)
    print(f"Shortest path: {'->'.join(display_path)}.")
    print(f"Shortest distance: {distance}.")
    print(f"Total energy cost: {energy}.")


def run_task1():
    start, goal = "1", "50"
    distance_dict, parent = Task1.uniform_cost_search(Task1.G, Task1.Dist, start, goal)
    if goal not in distance_dict:
        print("No path found.")
        return
    path = Task1.reconstruct_path(parent, start, goal)
    total_distance = distance_dict[goal]
    total_energy = Task1.compute_energy(path, Task1.Cost)
    print_submission_output(path, total_distance, total_energy)


def run_task2():
    start, goal = "1", "50"
    budget = 287932
    G, Dist, Cost = Task2.load_instance()
    path, best_distance, best_energy = Task2.ucs_energy_constrained(
        G, Dist, Cost, start, goal, budget
    )
    if path is None:
        print("No feasible path found within energy budget.")
        return
    print_submission_output(path, best_distance, best_energy)


def run_task3():
    start, goal = "1", "50"
    budget = 287932
    G, Dist, Cost, Coord = Task3.load_instance()
    path, best_distance, best_energy = Task3.astar_energy_constrained(
        G, Dist, Cost, Coord, start, goal, budget
    )
    if path is None:
        print("No feasible path found within energy budget.")
        return
    print_submission_output(path, best_distance, best_energy)


def run_task2_optimised():
    start, goal = "1", "50"
    budget = 287932
    G, Dist, Cost = Task2_Optimised.load_instance()
    path, best_distance, best_energy = Task2_Optimised.ucs_energy_constrained_optimised(
        G, Dist, Cost, start, goal, budget
    )
    if path is None:
        print("No feasible path found within energy budget.")
        return
    print_submission_output(path, best_distance, best_energy)


def run_task3_optimised():
    start, goal = "1", "50"
    budget = 287932
    G, Dist, Cost = Task3_Optimised.load_instance()
    path, best_distance, best_energy = Task3_Optimised.astar_energy_constrained_optimised(
        G, Dist, Cost, start, goal, budget
    )
    if path is None:
        print("No feasible path found within energy budget.")
        return
    print_submission_output(path, best_distance, best_energy)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        choices=["1", "2", "3", "2_optimised", "3_optimised", "all"],
        default="all",
        help="Task to run (default: all)",
    )
    args = parser.parse_args()

    if args.task in ("1", "all"):
        print("Task 1")
        run_task1()
        print()
    # if args.task in ("2", "all"):
    #     print("Task 2")
    #     run_task2()
    #     print()
    # if args.task in ("3", "all"):
    #     print("Task 3")
    #     run_task3()
    #     print()
    if args.task in ("2_optimised", "all"):
        print("Task 2 Optimised")
        run_task2_optimised()
        print()
    if args.task in ("3_optimised", "all"):
        print("Task 3 Optimised")
        run_task3_optimised()
        print()

if __name__ == "__main__":
    main()
