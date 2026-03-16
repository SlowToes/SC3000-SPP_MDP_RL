import heapq
import json
from collections import defaultdict


def load_instance():
    """Load graph, distance, and energy dictionaries from JSON files."""
    with open("G.json") as f:
        G = json.load(f)
    with open("Dist.json") as f:
        Dist = json.load(f)
    with open("Cost.json") as f:
        Cost = json.load(f)
    return G, Dist, Cost


def build_graphs(G, Dist, Cost):
    """Build forward and reverse adjacency structures used by A* pruning.

    - Forward adjacency drives state expansion.
    - Reverse energy adjacency gives feasibility bounds.
    - Reverse distance adjacency gives an admissible distance heuristic.
    """
    adj = defaultdict(list)
    rev_energy = defaultdict(list)
    rev_dist = defaultdict(list)
    for u, neighbors in G.items():
        for v in neighbors:
            key = f"{u},{v}"
            d = Dist[key]
            c = Cost[key]
            adj[u].append((v, d, c))
            rev_energy[v].append((u, c))
            rev_dist[v].append((u, d))
    return adj, rev_energy, rev_dist


def reverse_dijkstra_to_goal(goal, rev_adj):
    """Compute shortest reversed-path costs from all nodes to goal.

    Running Dijkstra from the goal on reversed edges efficiently computes
    exact lower bounds for whichever edge weight (energy or distance)
    is supplied in rev_adj.
    """
    best = {goal: 0}
    pq = [(0, goal)]
    while pq:
        cur, node = heapq.heappop(pq)
        if cur != best.get(node):
            continue
        for prev_node, w in rev_adj.get(node, []):
            nxt = cur + w
            if nxt < best.get(prev_node, float("inf")):
                best[prev_node] = nxt
                heapq.heappush(pq, (nxt, prev_node))
    return best


def is_dominated(path_states, state_ids, new_dist, new_energy):
    """Return True if an existing alive state dominates the candidate state."""
    for state_id in state_ids:
        if not path_states[state_id]["alive"]:
            continue
        if (
            path_states[state_id]["dist"] <= new_dist
            and path_states[state_id]["energy"] <= new_energy
        ):
            return True
    return False


def remove_dominated(path_states, state_ids, new_dist, new_energy):
    """Deactivate states dominated by the candidate state and keep survivors."""
    kept = []
    for state_id in state_ids:
        if not path_states[state_id]["alive"]:
            continue
        if (
            new_dist <= path_states[state_id]["dist"]
            and new_energy <= path_states[state_id]["energy"]
        ):
            path_states[state_id]["alive"] = False
        else:
            kept.append(state_id)
    return kept


def reconstruct_path(path_states, goal_state_id):
    """Reconstruct full path by walking parent state pointers backward."""
    path = []
    cur = goal_state_id
    while cur is not None:
        path.append(path_states[cur]["node"])
        cur = path_states[cur]["parent"]
    path.reverse()
    return path


def astar_energy_constrained(G, Dist, Cost, start, goal, budget):
    """A* search with an energy budget using nondominated states per node."""
    adj, rev_energy, rev_dist = build_graphs(G, Dist, Cost)

    min_energy_to_goal = reverse_dijkstra_to_goal(goal, rev_energy)
    # Fast infeasibility check: cannot satisfy budget even with best-case energy
    if start not in min_energy_to_goal or min_energy_to_goal[start] > budget:
        return None, None, None

    # Strong admissible heuristic: unconstrained shortest distance to goal
    min_dist_to_goal = reverse_dijkstra_to_goal(goal, rev_dist)
    if start not in min_dist_to_goal:
        return None, None, None

    path_states = []
    nondominated_state_ids_by_node = defaultdict(list)
    pq = []

    start_id = 0
    path_states.append(
        {"node": start, "dist": 0.0, "energy": 0, "parent": None, "alive": True}
    )
    nondominated_state_ids_by_node[start].append(start_id)
    h0 = min_dist_to_goal[start]
    # Priority tuple: (f = g + h, g, state_id) for deterministic tie handling
    heapq.heappush(pq, (h0, 0.0, start_id))

    while pq:
        f, g, state_id = heapq.heappop(pq)
        if not path_states[state_id]["alive"]:
            continue
        # Drop stale queue entries for states already replaced by a better one
        if g != path_states[state_id]["dist"]:
            continue

        node = path_states[state_id]["node"]
        energy_so_far = path_states[state_id]["energy"]

        if node == goal:
            path = reconstruct_path(path_states, state_id)
            return path, g, energy_so_far

        for neighbour, step_dist, step_energy in adj.get(node, []):
            new_energy = energy_so_far + step_energy
            if new_energy > budget:
                continue

            rem_energy_lb = min_energy_to_goal.get(neighbour, float("inf"))
            # If even optimistic remaining energy breaks budget, prune
            if new_energy + rem_energy_lb > budget:
                continue

            new_dist = g + step_dist
            node_state_ids = nondominated_state_ids_by_node[neighbour]

            if is_dominated(path_states, node_state_ids, new_dist, new_energy):
                continue

            nondominated_state_ids_by_node[neighbour] = remove_dominated(
                path_states, node_state_ids, new_dist, new_energy
            )

            h = min_dist_to_goal.get(neighbour, float("inf"))
            # Unreachable-to-goal nodes provide no valid completion
            if h == float("inf"):
                continue

            new_id = len(path_states)
            path_states.append(
                {
                    "node": neighbour,
                    "dist": new_dist,
                    "energy": new_energy,
                    "parent": state_id,
                    "alive": True,
                }
            )
            nondominated_state_ids_by_node[neighbour].append(new_id)
            # Standard A* push with f = g + h
            heapq.heappush(pq, (new_dist + h, new_dist, new_id))

    return None, None, None


def format_submission_path(path):
    """Render output path as S -> ... -> T."""
    if len(path) >= 2:
        return ["S"] + path[1:-1] + ["T"]
    return path


def run_task3(start="1", goal="50", budget=287932, print_output=True):
    """Execute Task 3."""
    G, Dist, Cost = load_instance()
    path, best_distance, best_energy = astar_energy_constrained(
        G, Dist, Cost, start, goal, budget
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
    run_task3()
