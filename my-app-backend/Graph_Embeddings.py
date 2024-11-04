import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Dict, List
from Tabular_to_Graph import DataFrameToGraph 

class GraphEmbeddingManager(nn.Module):
    def __init__(self, 
                 data_graph: DataFrameToGraph, 
                 embedding_strategy: str = 'separate', 
                 embedding_dim: int = 64):
        """
        Manages embeddings and message passing for a heterogeneous graph.

        Parameters:
        - data_graph (DataFrameToGraph): Instance of the DataFrameToGraph class.
        - embedding_strategy (str): The embedding strategy to use. Options:
            - 'separate': Separate embeddings per node type.
            - 'cross_attention': Attention mechanism for cross-type interactions.
            - 'heterogeneous': Type-specific message passing.
            - 'fusion': Final fusion layer for unifying embeddings.
        - embedding_dim (int): Dimension for embeddings after projection.
        """
        super(GraphEmbeddingManager, self).__init__()
        self.graph = data_graph.get_graph()
        self.embedding_strategy = embedding_strategy
        self.embedding_dim = embedding_dim
        self.embeddings = nn.ModuleDict()
        
        # Initialize embeddings based on node types in the graph
        self.node_types = set(nx.get_node_attributes(self.graph, 'type').values())
        for node_type in self.node_types:
            input_dim = 500 if node_type == "Post" else 20  # Example: sentence vs metadata
            self.embeddings[node_type] = nn.Linear(input_dim, embedding_dim)
        
        # Initialize strategy-specific components
        if embedding_strategy == 'cross_attention':
            self.attention_layer = nn.MultiheadAttention(embedding_dim, num_heads=4)
        elif embedding_strategy == 'heterogeneous':
            self.relational_layers = nn.ModuleDict({
                f"{src_type}-{tgt_type}": nn.Linear(embedding_dim, embedding_dim)
                for src_type in self.node_types for tgt_type in self.node_types
            })
        elif embedding_strategy == 'fusion':
            self.fusion_layer = nn.Linear(len(self.node_types) * embedding_dim, embedding_dim)
    
    def forward(self, node_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass to compute embeddings based on the selected strategy.

        Parameters:
        - node_features (Dict[str, torch.Tensor]): A dictionary with node types as keys and
          tensors of features as values.

        Returns:
        - torch.Tensor: The unified embedding for each node.
        """
        if self.embedding_strategy == 'separate':
            return self._separate_embeddings(node_features)
        elif self.embedding_strategy == 'cross_attention':
            return self._cross_attention_embeddings(node_features)
        elif self.embedding_strategy == 'heterogeneous':
            return self._heterogeneous_message_passing(node_features)
        elif self.embedding_strategy == 'fusion':
            return self._fusion_embedding(node_features)
    
    def _separate_embeddings(self, node_features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply separate embedding layers for each node type."""
        embedded_features = {}
        for node_type, features in node_features.items():
            embedded_features[node_type] = self.embeddings[node_type](features)
        return embedded_features
    
    def _cross_attention_embeddings(self, node_features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply cross-type attention to compute embeddings."""
        embedded_features = self._separate_embeddings(node_features)
        combined_features = []
        for _, embedding in embedded_features.items():
            combined_features.append(embedding.unsqueeze(0))  # Prepare for attention layer
        
        combined_features = torch.cat(combined_features, dim=0)
        attn_output, _ = self.attention_layer(combined_features, combined_features, combined_features)
        return attn_output
    
    def _heterogeneous_message_passing(self, node_features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply heterogeneous message passing for type-specific transformations."""
        embedded_features = self._separate_embeddings(node_features)
        updated_features = {}
        for src_type, src_embedding in embedded_features.items():
            for tgt_type, tgt_embedding in embedded_features.items():
                if src_type != tgt_type:
                    transformed = self.relational_layers[f"{src_type}-{tgt_type}"](tgt_embedding)
                    updated_features[src_type] = updated_features.get(src_type, 0) + transformed
                else:
                    updated_features[src_type] = updated_features.get(src_type, 0) + src_embedding
        return updated_features
    
    def _fusion_embedding(self, node_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Apply fusion layer to unify embeddings."""
        embedded_features = self._separate_embeddings(node_features)
        concatenated = torch.cat(list(embedded_features.values()), dim=1)
        return self.fusion_layer(concatenated)
    
# Example usage
# Assuming DataFrameToGraph instance `df_to_graph` is already created
embedding_strategy = 'cross_attention'  # or 'separate', 'heterogeneous', 'fusion'
embedding_manager = GraphEmbeddingManager(data_graph=df_to_graph, 
                                          embedding_strategy=embedding_strategy, 
                                          embedding_dim=128)

# Example dummy features for each node type
node_features = {
    "Post": torch.randn(10, 500),  # Example batch of "Post" node features
    "User": torch.randn(10, 20)    # Example batch of "User" node features
}

# Compute embeddings
output_embeddings = embedding_manager(node_features)
