import json
import heapq


# ==============================
# Load JSON Files
# ==============================
def load_instance():
    with open("G.json") as f:
        G = json.load(f)

    with open("Dist.json") as f:
        Dist = json.load(f)

    with open("Cost.json") as f:
        Cost = json.load(f)

    # Coord not needed for Task 2 (used for A* in Task 3)
    return G, Dist, Cost


# ==============================
# Task 2: UCS with Energy Budget
# State = (node, energy_used)
# Priority = smallest distance so far
# ==============================
def ucs_energy_constrained(G, Dist, Cost, start, goal, budget):
    """
    Returns:
      path_nodes (list[str]) or None
      best_distance (float) or None
      best_energy (int/float) or None
    """

    # PQ entries: (distance_so_far, node, energy_used)
    pq = []
    heapq.heappush(pq, (0.0, start, 0))

    # Best distance found for each expanded state (node, energy_used)
    best_dist = {(start, 0): 0.0}

    # Parent map to reconstruct path: parent[(node, energy)] = (prev_node, prev_energy)
    parent = {}

    while pq:
        dist_so_far, node, energy_so_far = heapq.heappop(pq)
        state = (node, energy_so_far)

        # Skip stale entries
        if dist_so_far != best_dist.get(state, float("inf")):
            continue

        # Goal reached: UCS guarantees this is the shortest feasible distance
        if node == goal:
            path = reconstruct_path(parent, (start, 0), state)
            return path, dist_so_far, energy_so_far

        # Expand neighbors
        for nbr in G[node]:
            edge_key = f"{node},{nbr}"

            step_dist = Dist[edge_key]
            step_energy = Cost[edge_key]

            new_energy = energy_so_far + step_energy
            if new_energy > budget:
                continue  # violates energy budget

            new_dist = dist_so_far + step_dist
            new_state = (nbr, new_energy)

            # Relaxation in expanded state-space
            if new_dist < best_dist.get(new_state, float("inf")):
                best_dist[new_state] = new_dist
                parent[new_state] = state
                heapq.heappush(pq, (new_dist, nbr, new_energy))

    return None, None, None


def reconstruct_path(parent, start_state, goal_state):
    nodes = []
    cur = goal_state
    while cur != start_state:
        nodes.append(cur[0])
        cur = parent[cur]
    nodes.append(start_state[0])
    nodes.reverse()
    return nodes


def format_submission_path(path):
    if len(path) >= 2:
        return ["S"] + path[1:-1] + ["T"]
    return path


# ==============================
# Main
# ==============================
if __name__ == "__main__":
    G, Dist, Cost = load_instance()

    start = "1"
    goal = "50"
    BUDGET = 287932  # Tasks 2 & 3 energy budget :contentReference[oaicite:1]{index=1}

    path, best_distance, best_energy = ucs_energy_constrained(G, Dist, Cost, start, goal, BUDGET)

    if path is None:
        print("No feasible path found within energy budget.")
    else:
        display_path = format_submission_path(path)
        print(f"Shortest path: {'->'.join(display_path)}.")
        print(f"Shortest distance: {best_distance}.")
        print(f"Total energy cost: {best_energy}.")