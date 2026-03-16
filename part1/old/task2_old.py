import heapq
import json


def load_instance():
    """Load graph, distance, and energy dictionaries from JSON files."""
    with open("G.json") as f:
        graph = json.load(f)
    with open("Dist.json") as f:
        distance = json.load(f)
    with open("Cost.json") as f:
        energy = json.load(f)
    return graph, distance, energy


def reconstruct_path(parent, end_state):
    """Reconstruct node path from parent pointers on expanded states."""
    path = []
    state = end_state
    while state is not None:
        node, _energy_used = state
        path.append(node)
        state = parent.get(state)
    path.reverse()
    return path


def dijkstra_energy_constrained(graph, distance, energy, start, goal, budget):
    """Constrained Dijkstra on (node, energy_used) states."""
    start_state = (start, 0)
    pq = [(0.0, start_state)]
    best_distance_by_state = {start_state: 0.0}
    parent = {start_state: None}

    best_goal_state = None
    best_goal_distance = float("inf")

    while pq:
        dist_so_far, state = heapq.heappop(pq)
        if dist_so_far != best_distance_by_state.get(state):
            continue

        node, energy_used = state
        if node == goal and dist_so_far < best_goal_distance:
            best_goal_distance = dist_so_far
            best_goal_state = state

        for neighbour in graph.get(node, []):
            edge_key = f"{node},{neighbour}"
            step_dist = distance[edge_key]
            step_energy = energy[edge_key]

            new_energy = energy_used + step_energy
            if new_energy > budget:
                continue

            new_dist = dist_so_far + step_dist
            new_state = (neighbour, new_energy)
            if new_dist < best_distance_by_state.get(new_state, float("inf")):
                best_distance_by_state[new_state] = new_dist
                parent[new_state] = state
                heapq.heappush(pq, (new_dist, new_state))

    if best_goal_state is None:  # No path found
        return None, None, None

    path = reconstruct_path(parent, best_goal_state)
    total_distance = best_goal_distance
    total_energy = best_goal_state[1]
    return path, total_distance, total_energy


def format_submission_path(path):
    """Render output path as S -> ... -> T."""
    if len(path) >= 2:
        return ["S"] + path[1:-1] + ["T"]
    return path


def run_task2_old(start="1", goal="50", budget=287932, print_output=True):
    """Run Task 2 old."""
    graph, distance, energy = load_instance()
    path, best_distance, best_energy = dijkstra_energy_constrained(
        graph, distance, energy, start, goal, budget
    )

    if path is None:  # No path found
        if print_output:
            print("No feasible path found within energy budget.")
        return None

    display_path = format_submission_path(path)
    result = {
        "path": path,
        "display_path": display_path,
        "distance": best_distance,
        "energy": best_energy,
        "budget": budget,
    }

    if print_output:
        print(f"Shortest path: {'->'.join(display_path)}.")
        print(f"Shortest distance: {best_distance}.")
        print(f"Total energy cost: {best_energy}.")

    return result


if __name__ == "__main__":
    run_task2_old()
