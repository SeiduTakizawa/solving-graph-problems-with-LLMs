def hasCycle(graph):
    visited = set()

    print("Starting cycle detection...\n")

    for node in graph:
        if node not in visited:
            print(f"Starting DFS at node {node}")
            if dfs(node, -1, visited, graph):
                print("Cycle found!\n")
                return "There is a cycle"
            else:
                print(f"No cycle from component starting at {node}\n")

    print("Finished checking all nodes.\n")
    return "There is no cycle"


def dfs(current, parent, visited, graph):
    print(f"Visiting node {current}, came from parent {parent}")
    visited.add(current)

    for neighbor in graph[current]:
        print(f"  At node {current}: checking neighbor {neighbor}")

        if neighbor not in visited:
            print(f"    Neighbor {neighbor} not visited yet. Going deeper...")
            if dfs(neighbor, current, visited, graph):
                print(f"    Cycle detected deeper from node {neighbor}!")
                return True
        elif neighbor != parent:
            print(f"    Neighbor {neighbor} already visited and is NOT the parent → Cycle!")
            return True

    print(f"Returning from node {current} — no cycle found here.\n")
    return False


# 🧪 Example graph (undirected) — contains a cycle: 1-2-3-1
graph = {
    0: [1],
    1: [0, 2, 3],
    2: [1, 3],
    3: [1, 2]
}

# You can also try a graph with no cycles:
# graph = {
#     0: [1],
#     1: [0, 2],
#     2: [1, 3],
#     3: [2]
# }

# Run the test
result = hasCycle(graph)
print("Final result:", result)
