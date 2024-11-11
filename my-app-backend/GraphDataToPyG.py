# my-app-backend/GraphDataToPyG.py

import torch
from torch_geometric.data import Data
from typing import Dict, Any
import networkx as nx
from networkx.readwrite import json_graph

class GraphDataToPyG:
    def __init__(self, graph_data: Dict[str, Any], node_label_column: str = None, edge_label_column: str = None):
        """
        Converts graph data in node-link format to PyTorch Geometric Data object.   
        
        Parameters:
        - graph_data (Dict): The graph data in node-link format.
        - node_label_column (str): The node attribute to be used as node labels.
        - edge_label_column (str): The edge attribute to be used as edge labels.
        """
        self.graph_data = graph_data
        self.node_label_column = node_label_column
        self.edge_label_column = edge_label_column
        self.pyg_data = None  # Will hold the PyG Data object after conversion

    def convert(self):
        # Convert node-link data back to NetworkX graph
        G = json_graph.node_link_graph(self.graph_data)

        # Map node IDs to indices
        node_id_map = {node_id: idx for idx, node_id in enumerate(G.nodes())}
        num_nodes = len(node_id_map)

        # Initialize lists for node features and labels
        node_features = []
        node_labels = []

        # Collect node features and labels
        for node_id, attr in G.nodes(data=True):
            feature_values = []

            # Extract numerical and embedding features
            for key, value in attr.items():
                if key == self.node_label_column:
                    continue  # Skip label column from features
                if isinstance(value, list):
                    feature_values.extend([float(v) for v in value])
                elif isinstance(value, (int, float)):
                    feature_values.append(float(value))
                # Handle categorical/string features if needed

            node_features.append(feature_values)

            # Node labels
            if self.node_label_column and self.node_label_column in attr:
                node_labels.append(attr[self.node_label_column])
            else:
                node_labels.append(0)  # Default label if none provided

        # Ensure all node features have the same length
        if node_features:
            max_feature_length = max(len(f) for f in node_features)
            for feature in node_features:
                if len(feature) < max_feature_length:
                    feature.extend([0.0] * (max_feature_length - len(feature)))
        else:
            node_features = [[0.0]] * num_nodes  # Assign default features if none are present

        # Convert to torch tensors
        x = torch.tensor(node_features, dtype=torch.float)
        y = torch.tensor(node_labels, dtype=torch.long) if node_labels else None

        # Initialize lists for edge indices, features, and labels
        edge_indices = []
        edge_features = []
        edge_labels = []

        # Collect edge indices, features, and labels
        for u, v, attr in G.edges(data=True):
            source_idx = node_id_map[u]
            target_idx = node_id_map[v]
            edge_indices.append([source_idx, target_idx])

            feature_values = []

            # Extract numerical and embedding features
            for key, value in attr.items():
                if key == self.edge_label_column:
                    continue  # Skip label column from features
                if isinstance(value, list):
                    feature_values.extend([float(v) for v in value])
                elif isinstance(value, (int, float)):
                    feature_values.append(float(value))
                # Handle categorical/string features if needed

            edge_features.append(feature_values)

            # Edge labels
            if self.edge_label_column and self.edge_label_column in attr:
                edge_labels.append(attr[self.edge_label_column])
            else:
                edge_labels.append(0)  # Default label if none provided

        # Ensure all edge features have the same length
        if edge_features:
            max_edge_feature_length = max(len(f) for f in edge_features)
            for feature in edge_features:
                if len(feature) < max_edge_feature_length:
                    feature.extend([0.0] * (max_edge_feature_length - len(feature)))
            edge_attr = torch.tensor(edge_features, dtype=torch.float)
        else:
            edge_attr = None

        # Convert edge indices to tensor
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

        # Edge labels tensor
        edge_y = torch.tensor(edge_labels, dtype=torch.long) if edge_labels else None

        # Create PyG Data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

        # Add edge labels if available
        if edge_y is not None:
            data.edge_y = edge_y

        self.pyg_data = data
        return data
