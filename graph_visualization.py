import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # Or 'QtAgg', depending on what you have installed

matplotlib.use('TkAgg')
# Path to your graph file
graph_path = r"C:\Users\thrig\PycharmProjects\graph-reasoning-llms\graphs\er\large\5.txt"

# Load graph from edge list (ignores lines starting with '#')
G = nx.read_adjlist(graph_path, comments='#', nodetype=int)

# Draw the graph
plt.figure(figsize=(6, 6))
nx.draw(G, with_labels=True, node_color='skyblue', edge_color='gray', node_size=800, font_size=14)
plt.title("Graph Visualization")
plt.show()
