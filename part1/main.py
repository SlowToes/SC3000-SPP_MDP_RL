import argparse

import task1
import task2
import task3


def run_task1():
    """Run Task 1 and print its result."""
    task1.run_task1()


def run_task2():
    """Run Task 2 and print its result."""
    task2.run_task2()


def run_task3():
    """Run Task 3 and print its result."""
    task3.run_task3()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        choices=["1", "2", "3", "all"],
        default="all",
        help="Task to run (default: all)",
    )
    args = parser.parse_args()

    if args.task in ("1", "all"):
        print("Task 1")
        run_task1()
        print()
    if args.task in ("2", "all"):
        print("Task 2")
        run_task2()
        print()
    if args.task in ("3", "all"):
        print("Task 3")
        run_task3()
        print()

if __name__ == "__main__":
    main()