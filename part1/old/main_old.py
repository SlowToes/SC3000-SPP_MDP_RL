import argparse

import task2_old
import task3_old


def run_task2_old():
    """Run old Task 2 baseline."""
    task2_old.run_task2_old()


def run_task3_old():
    """Run old Task 3 baseline."""
    task3_old.run_task3_old()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        choices=["2", "3", "all"],
        default="all",
        help="Old task to run (default: all)",
    )
    args = parser.parse_args()

    if args.task in ("2", "all"):
        print("Task 2 Old")
        run_task2_old()
        print()
    if args.task in ("3", "all"):
        print("Task 3 Old")
        run_task3_old()
        print()


if __name__ == "__main__":
    main()
