import torch

# Load the .pt file
data = torch.load('graph_data.pt')


# Number of nodes
num_nodes = data.num_nodes

# Number of edges
num_edges = data.num_edges

# Check if the graph is directed
is_directed = data.is_directed()

# List of available attributes in the data
attributes = data.keys

print(f"Number of nodes: {num_nodes}")
print(f"Number of edges: {num_edges}")
print(f"Is the graph directed? {is_directed}")
print(f"Attributes in the data: {attributes}")

# Node feature matrix
print("Node features (x):", data.x)

# Shape of node feature matrix
print("Shape of node feature matrix:", data.x.shape)

# Node labels, if available
if data.y is not None:
    print("Node labels (y):", data.y)
else:
    print("No node labels found.")


# Edge index tensor
print("Edge indices (edge_index):", data.edge_index)

# Edge attributes, if available
if data.edge_attr is not None:
    print("Edge attributes (edge_attr):", data.edge_attr)
else:
    print("No edge attributes found.")

# Edge labels, if available
if hasattr(data, 'edge_y') and data.edge_y is not None:
    print("Edge labels (edge_y):", data.edge_y)
else:
    print("No edge labels found.")


import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx

# Convert to NetworkX graph for visualization
G = to_networkx(data, to_undirected=True)  # Use `to_undirected=True` if the graph is undirected

# Draw the graph
plt.figure(figsize=(8, 6))
nx.draw(G, with_labels=True, node_size=500, node_color="lightblue", font_size=10)
plt.show()
