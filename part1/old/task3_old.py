import heapq
import json
import math


def load_instance():
    """Load graph, distance, energy, and coordinates from JSON files."""
    with open("G.json") as f:
        graph = json.load(f)
    with open("Dist.json") as f:
        distance = json.load(f)
    with open("Cost.json") as f:
        energy = json.load(f)
    with open("Coord.json") as f:
        coord = json.load(f)
    return graph, distance, energy, coord


def euclidean_heuristic(node, goal, coord):
    """Compute straight-line distance between a node and the goal."""
    x1, y1 = coord[node]
    x2, y2 = coord[goal]
    return math.hypot(x2 - x1, y2 - y1)


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


def astar_energy_constrained_euclidean(graph, distance, energy, coord, start, goal, budget):
    """A* on (node, energy_used) states with Euclidean heuristic."""
    start_state = (start, 0)
    g_best_by_state = {start_state: 0.0}
    parent = {start_state: None}
    h0 = euclidean_heuristic(start, goal, coord)
    pq = [(h0, 0.0, start_state)]  # (f, g, state)

    while pq:
        f_score, g_score, state = heapq.heappop(pq)
        if g_score != g_best_by_state.get(state):
            continue

        node, energy_used = state
        if node == goal:
            path = reconstruct_path(parent, state)
            return path, g_score, energy_used

        for nbr in graph.get(node, []):
            edge_key = f"{node},{nbr}"
            step_dist = distance[edge_key]
            step_energy = energy[edge_key]

            new_energy = energy_used + step_energy
            if new_energy > budget:
                continue

            new_g = g_score + step_dist
            new_state = (nbr, new_energy)
            if new_g < g_best_by_state.get(new_state, float("inf")):
                g_best_by_state[new_state] = new_g
                parent[new_state] = state
                h = euclidean_heuristic(nbr, goal, coord)
                heapq.heappush(pq, (new_g + h, new_g, new_state))

    return None, None, None


def format_submission_path(path):
    """Render output path as S -> ... -> T."""
    if len(path) >= 2:
        return ["S"] + path[1:-1] + ["T"]
    return path


def run_task3_old(start="1", goal="50", budget=287932, print_output=True):
    """Run Task 3 old baseline (A* with Euclidean heuristic)."""
    graph, distance, energy, coord = load_instance()
    path, best_distance, best_energy = astar_energy_constrained_euclidean(
        graph, distance, energy, coord, start, goal, budget
    )

    if path is None:
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
    run_task3_old()
