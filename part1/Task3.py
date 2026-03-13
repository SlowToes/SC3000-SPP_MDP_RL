import json
import heapq
import math


def load_instance():
    with open("G.json") as f:
        G = json.load(f)

    with open("Dist.json") as f:
        Dist = json.load(f)

    with open("Cost.json") as f:
        Cost = json.load(f)

    with open("Coord.json") as f:
        Coord = json.load(f)

    return G, Dist, Cost, Coord


def euclidean_heuristic(Coord, node, goal):
    x1, y1 = Coord[node]
    x2, y2 = Coord[goal]
    # Make robust in case JSON stores numbers as strings
    x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


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


def astar_energy_constrained(G, Dist, Cost, Coord, start, goal, budget):
    # PQ entries: (f, g, node, energy_used)
    h0 = euclidean_heuristic(Coord, start, goal)
    pq = []
    heapq.heappush(pq, (h0, 0.0, start, 0))

    # Best g for each expanded state (node, energy_used)
    best_g = {(start, 0): 0.0}

    # Parent pointers: parent[(node, energy)] = (prev_node, prev_energy)
    parent = {}

    while pq:
        f, g, node, energy_used = heapq.heappop(pq)
        state = (node, energy_used)

        # Skip stale PQ entries
        if g != best_g.get(state, float("inf")):
            continue

        # Goal reached
        if node == goal:
            path = reconstruct_path(parent, (start, 0), state)
            return path, g, energy_used

        for nbr in G[node]:
            edge_key = f"{node},{nbr}"
            step_dist = Dist[edge_key]
            step_energy = Cost[edge_key]

            new_energy = energy_used + step_energy
            if new_energy > budget:
                continue

            new_g = g + step_dist
            new_state = (nbr, new_energy)

            if new_g < best_g.get(new_state, float("inf")):
                best_g[new_state] = new_g
                parent[new_state] = state

                h = euclidean_heuristic(Coord, nbr, goal)
                heapq.heappush(pq, (new_g + h, new_g, nbr, new_energy))

    return None, None, None


if __name__ == "__main__":
    G, Dist, Cost, Coord = load_instance()

    start = "1"
    goal = "50"
    BUDGET = 287932  # Tasks 2 & 3 energy budget

    path, best_distance, best_energy = astar_energy_constrained(
        G, Dist, Cost, Coord, start, goal, BUDGET
    )

    if path is None:
        print("No feasible path found within energy budget.")
    else:
        display_path = format_submission_path(path)
        print(f"Shortest path: {'->'.join(display_path)}.")
        print(f"Shortest distance: {best_distance}.")
        print(f"Total energy cost: {best_energy}.")