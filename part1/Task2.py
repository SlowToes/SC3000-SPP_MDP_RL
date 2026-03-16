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
    """Build forward adjacency and reverse-energy adjacency lists.

    - adj supports forward expansion during search.
    - rev_energy supports reverse Dijkstra to compute a lower bound on the
      minimum additional energy needed to reach the goal.
    """
    adj = defaultdict(list)
    rev_energy = defaultdict(list)
    for u, neighbors in G.items():
        for v in neighbors:
            key = f"{u},{v}"
            d = Dist[key]
            c = Cost[key]
            adj[u].append((v, d, c))
            rev_energy[v].append((u, c))
    return adj, rev_energy


def reverse_dijkstra_to_goal(goal, rev_adj):
    """Compute minimum cost to go from every node to goal on reversed edges.

    This gives an admissible lower bound for pruning: if current energy plus
    this bound exceeds budget, that partial path can never become feasible.
    """
    best = {goal: 0}
    pq = [(0, goal)]
    while pq:
        curr, node = heapq.heappop(pq)
        if curr != best.get(node):
            continue
        for prev, w in rev_adj.get(node, []):
            next = curr + w
            if next < best.get(prev, float("inf")):
                best[prev] = next
                heapq.heappush(pq, (next, prev))
    return best


def is_dominated(path_states, state_ids, new_dist, new_energy):
    """Return True if an existing alive state dominates the candidate state.

    For this multi-criteria shortest path, dominated states can
    never lead to a better feasible solution and should be skipped early.
    """
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
    """Deactivate states dominated by the candidate state and keep survivors.

    This maintains a frontier for each node and prevents unnecessary
    expansions while preserving all potentially optimal trade-offs.
    """
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
    """Rebuild final node sequence by following parent state links."""
    path = []
    cur = goal_state_id
    while cur is not None:
        path.append(path_states[cur]["node"])
        cur = path_states[cur]["parent"]
    path.reverse()
    return path


def ucs_energy_constrained(G, Dist, Cost, start, goal, budget):
    """Find minimum-distance path subject to an energy budget
    using UCS with dominated states pruning.

    This algorithm keeps multiple nondominated states per node
    and prunes states that are infeasible or dominated.
    """
    adj, rev_energy = build_graphs(G, Dist, Cost)
    min_energy_to_goal = reverse_dijkstra_to_goal(goal, rev_energy)
    # Immediate infeasibility check before running the full search
    if start not in min_energy_to_goal or min_energy_to_goal[start] > budget:
        return None, None, None

    path_states = []
    nondominated_state_ids_by_node = defaultdict(list)
    pq = []

    start_id = 0
    # A state is one partial-path summary: (node, distance, energy, parent)
    path_states.append(
        {"node": start, "dist": 0.0, "energy": 0, "parent": None, "alive": True}
    )
    nondominated_state_ids_by_node[start].append(start_id)
    # Priority by distance keeps UCS behavior among feasible states
    heapq.heappush(pq, (0.0, start_id))

    while pq:
        dist_so_far, state_id = heapq.heappop(pq)
        if not path_states[state_id]["alive"]:
            continue
        # Skip stale heap entries if this state was improved/replaced earlier
        if dist_so_far != path_states[state_id]["dist"]:
            continue

        node = path_states[state_id]["node"]
        energy_so_far = path_states[state_id]["energy"]

        # First time goal is popped gives the optimal feasible distance
        if node == goal:
            path = reconstruct_path(path_states, state_id)
            return path, dist_so_far, energy_so_far

        for neighbour, step_dist, step_energy in adj.get(node, []):
            new_energy = energy_so_far + step_energy
            # Hard budget check
            if new_energy > budget:
                continue

            rem_energy_lb = min_energy_to_goal.get(neighbour, float("inf"))
            # Lower-bound pruning: cannot possibly satisfy budget downstream
            if new_energy + rem_energy_lb > budget:
                continue

            new_dist = dist_so_far + step_dist
            node_state_ids = nondominated_state_ids_by_node[neighbour]

            # Existing state already better (or equal) in both dimensions
            if is_dominated(path_states, node_state_ids, new_dist, new_energy):
                continue

            # Remove states made obsolete by this better trade-off
            nondominated_state_ids_by_node[neighbour] = remove_dominated(
                path_states, node_state_ids, new_dist, new_energy
            )

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
            heapq.heappush(pq, (new_dist, new_id))

    return None, None, None


def format_submission_path(path):
    """Render output path as S -> ... -> T."""
    if len(path) >= 2:
        return ["S"] + path[1:-1] + ["T"]
    return path


def run_task2(start="1", goal="50", budget=287932, print_output=True):
    """Execute Task 2."""
    G, Dist, Cost = load_instance()
    path, best_distance, best_energy = ucs_energy_constrained(
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
    run_task2()