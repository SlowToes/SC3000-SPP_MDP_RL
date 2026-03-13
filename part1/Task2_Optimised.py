import heapq
import json
from collections import defaultdict


def load_instance():
    with open("G.json") as f:
        G = json.load(f)
    with open("Dist.json") as f:
        Dist = json.load(f)
    with open("Cost.json") as f:
        Cost = json.load(f)
    return G, Dist, Cost


def build_graphs(G, Dist, Cost):
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


def is_dominated(labels, label_ids, new_dist, new_energy):
    for lid in label_ids:
        if not labels[lid]["alive"]:
            continue
        if labels[lid]["dist"] <= new_dist and labels[lid]["energy"] <= new_energy:
            return True
    return False


def remove_dominated(labels, label_ids, new_dist, new_energy):
    kept = []
    for lid in label_ids:
        if not labels[lid]["alive"]:
            continue
        if new_dist <= labels[lid]["dist"] and new_energy <= labels[lid]["energy"]:
            labels[lid]["alive"] = False
        else:
            kept.append(lid)
    return kept


def reconstruct_path(labels, goal_label_id):
    path = []
    cur = goal_label_id
    while cur is not None:
        path.append(labels[cur]["node"])
        cur = labels[cur]["parent"]
    path.reverse()
    return path


def ucs_energy_constrained_optimised(G, Dist, Cost, start, goal, budget):
    adj, rev_energy = build_graphs(G, Dist, Cost)
    min_energy_to_goal = reverse_dijkstra_to_goal(goal, rev_energy)
    if start not in min_energy_to_goal or min_energy_to_goal[start] > budget:
        return None, None, None

    labels = []
    pareto = defaultdict(list)
    pq = []

    start_id = 0
    labels.append(
        {"node": start, "dist": 0.0, "energy": 0, "parent": None, "alive": True}
    )
    pareto[start].append(start_id)
    heapq.heappush(pq, (0.0, start_id))

    while pq:
        dist_so_far, lid = heapq.heappop(pq)
        if not labels[lid]["alive"]:
            continue
        if dist_so_far != labels[lid]["dist"]:
            continue

        node = labels[lid]["node"]
        energy_so_far = labels[lid]["energy"]

        if node == goal:
            path = reconstruct_path(labels, lid)
            return path, dist_so_far, energy_so_far

        for nbr, step_dist, step_energy in adj.get(node, []):
            new_energy = energy_so_far + step_energy
            if new_energy > budget:
                continue

            rem_energy_lb = min_energy_to_goal.get(nbr, float("inf"))
            if new_energy + rem_energy_lb > budget:
                continue

            new_dist = dist_so_far + step_dist
            nbr_labels = pareto[nbr]

            if is_dominated(labels, nbr_labels, new_dist, new_energy):
                continue

            pareto[nbr] = remove_dominated(labels, nbr_labels, new_dist, new_energy)

            new_id = len(labels)
            labels.append(
                {
                    "node": nbr,
                    "dist": new_dist,
                    "energy": new_energy,
                    "parent": lid,
                    "alive": True,
                }
            )
            pareto[nbr].append(new_id)
            heapq.heappush(pq, (new_dist, new_id))

    return None, None, None


def format_submission_path(path):
    if len(path) >= 2:
        return ["S"] + path[1:-1] + ["T"]
    return path


if __name__ == "__main__":
    G, Dist, Cost = load_instance()
    start = "1"
    goal = "50"
    BUDGET = 287932

    path, best_distance, best_energy = ucs_energy_constrained_optimised(
        G, Dist, Cost, start, goal, BUDGET
    )

    if path is None:
        print("No feasible path found within energy budget.")
    else:
        display_path = format_submission_path(path)
        print(f"Shortest path: {'->'.join(display_path)}.")
        print(f"Shortest distance: {best_distance}.")
        print(f"Total energy cost: {best_energy}.")
