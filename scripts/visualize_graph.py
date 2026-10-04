"""Draw one graph from the dataset.

Usage: python3 scripts/visualize_graph.py data/graphs/er/large/5.txt
"""
import sys

import matplotlib.pyplot as plt
import networkx as nx

graph_path = sys.argv[1] if len(sys.argv) > 1 else "data/graphs/er/large/5.txt"

# Load graph from adjacency list (ignores lines starting with '#')
G = nx.read_adjlist(graph_path, comments='#', nodetype=int)

plt.figure(figsize=(6, 6))
nx.draw(G, with_labels=True, node_color='skyblue', edge_color='gray', node_size=800, font_size=14)
plt.title("Graph Visualization")
plt.show()
