import heapq
import json


def load_instance():
    """Load graph, distance, and energy dictionaries from JSON files."""
    with open("G.json") as f:
        G = json.load(f)
    with open("Dist.json") as f:
        Dist = json.load(f)
    with open("Cost.json") as f:
        Cost = json.load(f)
    return G, Dist, Cost


def uniform_cost_search(graph, distance, start, goal):
    """Run Uniform Cost Search to find the minimum-distance path.

    UCS is Dijkstra's algorithm for non-negative edge weights. Since Task 1
    optimises only distance, expanding the smallest known distance first
    guarantees the first optimal distance to each node.
    """
    # Min-heap ordered by total distance from start to a frontier node
    frontier = [(0.0, start)]
    # Best known distance to each node, used for pruning stale queue entries
    best_distance = {start: 0.0}
    # Parent pointers allow full path reconstruction once goal is reached
    parent = {}

    while frontier:
        current_distance, current_node = heapq.heappop(frontier)

        # Ignore outdated heap entries that are worse than the recorded best
        if current_distance != best_distance.get(current_node):
            continue

        # Safe early exit if the current node is the goal
        if current_node == goal:
            break

        # Check whether going from the current node to neighbour gives a shorter path than the currently known distance to neighbour
        for neighbour in graph.get(current_node, []):
            edge_key = f"{current_node},{neighbour}"
            step_distance = distance[edge_key]
            new_distance = current_distance + step_distance

            # Keep only strictly better distance
            if new_distance < best_distance.get(neighbour, float("inf")):
                best_distance[neighbour] = new_distance
                parent[neighbour] = current_node
                heapq.heappush(frontier, (new_distance, neighbour))

    return best_distance, parent


def reconstruct_path(parent, start, goal):
    """Rebuild path from start to goal using parent pointers.

    Search stores predecessors, then this function reconstructs the
    explicit route once the destination is confirmed reachable.
    """
    path = [goal]
    node = goal

    while node != start:
        node = parent[node]
        path.append(node)

    path.reverse()
    return path


def compute_energy(path, energy):
    """Compute total energy along an already-chosen path."""
    total_energy = 0
    for i in range(len(path) - 1):
        edge_key = f"{path[i]},{path[i + 1]}"
        total_energy += energy[edge_key]
    return total_energy


def format_submission_path(path):
    """Render output path as S -> ... -> T."""
    if len(path) >= 2:
        return ["S"] + path[1:-1] + ["T"]
    return path


def run_task1(start="1", goal="50", print_output=True):
    """Execute Task 1."""
    G, Dist, Cost = load_instance()
    distance_dict, parent = uniform_cost_search(G, Dist, start, goal)

    if goal not in distance_dict:  # No path found
        if print_output:
            print("No path found.")
        return None

    path = reconstruct_path(parent, start, goal)
    total_distance = distance_dict[goal]
    total_energy = compute_energy(path, Cost)
    display_path = format_submission_path(path)

    result = {
        "path": path,
        "display_path": display_path,
        "distance": total_distance,
        "energy": total_energy,
    }

    if print_output:
        print(f"Shortest path: {'->'.join(display_path)}.")
        print(f"Shortest distance: {total_distance}.")
        print(f"Total energy cost: {total_energy}.")

    return result


if __name__ == "__main__":
    run_task1()