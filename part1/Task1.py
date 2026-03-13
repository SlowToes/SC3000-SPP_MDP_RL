import json
import heapq


# ==============================
# Load JSON Files
# ==============================

with open("G.json") as f:
    G = json.load(f)

with open("Dist.json") as f:
    Dist = json.load(f)

with open("Cost.json") as f:
    Cost = json.load(f)

with open("Coord.json") as f:
    Coord = json.load(f)


# ==============================
# Uniform Cost Search (Task 1)
# ==============================

def uniform_cost_search(G, Dist, start, goal):

    # Priority queue stores (distance_so_far, node)
    frontier = []
    heapq.heappush(frontier, (0, start))

    # Best known distance to each node
    visited_cost = {start: 0}

    # Parent dictionary to reconstruct path
    parent = {}

    while frontier:

        current_cost, current_node = heapq.heappop(frontier)

        # If we reached goal, stop early
        if current_node == goal:
            break

        # Skip if this is not the best path anymore
        if current_cost > visited_cost[current_node]:
            continue

        # Expand neighbors
        for neighbor in G[current_node]:

            edge_key = f"{current_node},{neighbor}"
            edge_distance = Dist[edge_key]

            new_cost = current_cost + edge_distance

            # If better path found
            if neighbor not in visited_cost or new_cost < visited_cost[neighbor]:
                visited_cost[neighbor] = new_cost
                parent[neighbor] = current_node
                heapq.heappush(frontier, (new_cost, neighbor))

    return visited_cost, parent


# ==============================
# Reconstruct Path
# ==============================

def reconstruct_path(parent, start, goal):

    path = []
    node = goal

    while node != start:
        path.append(node)
        node = parent[node]

    path.append(start)
    path.reverse()

    return path


# ==============================
# Compute Energy Cost (for output)
# ==============================

def compute_energy(path, Cost):

    total_energy = 0

    for i in range(len(path) - 1):
        edge_key = f"{path[i]},{path[i+1]}"
        total_energy += Cost[edge_key]

    return total_energy


def format_submission_path(path):
    if len(path) >= 2:
        return ["S"] + path[1:-1] + ["T"]
    return path


# ==============================
# Main
# ==============================

if __name__ == "__main__":

    start = '1'
    goal = '50'

    distance_dict, parent = uniform_cost_search(G, Dist, start, goal)

    if goal not in distance_dict:
        print("No path found.")
    else:
        path = reconstruct_path(parent, start, goal)
        total_distance = distance_dict[goal]
        total_energy = compute_energy(path, Cost)
        display_path = format_submission_path(path)

        print(f"Shortest path: {'->'.join(display_path)}.")
        print(f"Shortest distance: {total_distance}.")
        print(f"Total energy cost: {total_energy}.")