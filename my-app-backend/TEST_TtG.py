# TEST_TtG

import torch
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple, Optional, Callable
import logging
import matplotlib.patches as mpatches
from torch_geometric.data import Data as PyGData
from transformers import BertModel, BertTokenizer
from sklearn.preprocessing import StandardScaler

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataFrameToGraph:
    def __init__(self, 
                 df: pd.DataFrame, 
                 config: Dict[str, Any],
                 graph_type: str = 'directed'):
        """
        Initializes the DataFrameToGraph instance.

        Parameters:
        - df (pd.DataFrame): The input DataFrame containing tabular data.
        - config (Dict[str, Any]): Configuration dictionary defining column roles and feature types.
        - graph_type (str): Type of the graph ('directed' or 'undirected').
        """
        self.df = df
        self.config = config
        self.graph_type = graph_type.lower()
        self.graph = self._initialize_graph()
        self.node_registry = {}
        self.edge_registry = {}
        
        self._validate_config()
        self._parse_dataframe()

    def _initialize_graph(self) -> nx.Graph:
        """Initializes the NetworkX graph based on the specified type."""
        if self.graph_type == 'directed':
            return nx.MultiDiGraph()
        elif self.graph_type == 'undirected':
            return nx.MultiGraph()
        else:
            raise ValueError("graph_type must be 'directed' or 'undirected'.")

    def _validate_config(self):
        """Validates the configuration dictionary."""
        required_keys = ['nodes', 'relationships']
        for key in required_keys:
            if key not in self.config:
                raise KeyError(f"Configuration missing required key: '{key}'")
        
        # Validate nodes configuration
        for node_conf in self.config['nodes']:
            if 'id' not in node_conf:
                raise KeyError("Each node configuration must have an 'id' key.")
            if 'type' not in node_conf:
                logger.warning(f"Node configuration {node_conf} missing 'type'. Defaulting to 'default'.")
            if 'features' not in node_conf:
                logger.warning(f"Node configuration {node_conf} missing 'features'. No features will be extracted.")
            else:
                for feature in node_conf['features']:
                    if 'name' not in feature or 'type' not in feature:
                        raise KeyError("Each feature must have 'name' and 'type' keys.")
        
        # Validate relationships configuration
        for rel_conf in self.config['relationships']:
            if 'source' not in rel_conf or 'target' not in rel_conf:
                raise KeyError("Each relationship configuration must have 'source' and 'target' keys.")
            if 'type' not in rel_conf:
                logger.warning(f"Relationship configuration {rel_conf} missing 'type'. Defaulting to 'default'.")
            if 'features' not in rel_conf:
                logger.warning(f"Relationship configuration {rel_conf} missing 'features'. No features will be extracted.")
            else:
                for feature in rel_conf['features']:
                    if 'name' not in feature or 'type' not in feature:
                        raise KeyError("Each feature must have 'name' and 'type' keys.")

    def _parse_dataframe(self):
        """Parses the DataFrame and constructs the graph."""
        for index, row in self.df.iterrows():
            # Add nodes
            for node_conf in self.config['nodes']:
                node_id = row.get(node_conf['id'], None)
                if pd.isnull(node_id):
                    logger.warning(f"Row {index}: Missing node ID for '{node_conf['id']}'. Skipping node addition.")
                    continue
                # Convert node_id to string for consistency
                node_id_str = str(int(node_id)) if isinstance(node_id, float) and node_id.is_integer() else str(node_id)
                node_type = node_conf.get('type', 'default')
                features = self._extract_features(row, node_conf.get('features', []))
                self._add_node(node_id_str, node_type, features)
            
            # Add edges
            for rel_conf in self.config['relationships']:
                source_col = rel_conf['source']
                target_col = rel_conf['target']
                relationship_type = rel_conf.get('type', 'default')
                features = self._extract_features(row, rel_conf.get('features', []))
                
                source_id = row.get(source_col, None)
                target_id = row.get(target_col, None)
                
                if pd.isnull(source_id) or pd.isnull(target_id):
                    logger.warning(f"Row {index}: Missing source or target ID for relationship '{relationship_type}'. Skipping edge addition.")
                    continue
                
                # Handle inconsistent data types
                try:
                    source_id = int(source_id)
                    source_id_str = str(source_id)
                except (ValueError, TypeError):
                    logger.warning(f"Row {index}: Invalid data type for source_id '{source_id}'. Skipping edge addition.")
                    continue

                try:
                    if isinstance(target_id, float) and target_id.is_integer():
                        target_id = int(target_id)
                    target_id_str = str(target_id)
                except (ValueError, TypeError):
                    logger.warning(f"Row {index}: Invalid data type for target_id '{target_id}'. Skipping edge addition.")
                    continue

                self._add_edge(source_id_str, target_id_str, relationship_type, features)

    def _extract_features(self, row: pd.Series, feature_cols: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extracts and processes features from the DataFrame row based on feature type.

        Parameters:
        - row (pd.Series): The DataFrame row.
        - feature_cols (List[Dict[str, Any]]): List of feature configurations.

        Returns:
        - Dict[str, Any]: Dictionary of processed features.
        """
        features = {}
        for feature in feature_cols:
            feat_name = feature['name']
            feat_type = feature['type']
            if feat_name in row:
                value = row[feat_name]
                if feat_type == 'numeric':
                    if isinstance(value, list):
                        # If numeric feature is a list, take its length or another aggregation
                        features[feat_name] = len(value)
                    elif pd.isnull(value):
                        logger.info(f"Feature '{feat_name}' is missing in row. Assigning default value 0.0.")
                        features[feat_name] = 0.0
                    else:
                        try:
                            features[feat_name] = float(value)
                        except ValueError:
                            logger.warning(f"Non-numeric feature value encountered: {value}. Replacing with 0.0.")
                            features[feat_name] = 0.0
                elif feat_type == 'text':
                    if isinstance(value, list):
                        # Join list items into a single string
                        features[feat_name] = ','.join(map(str, value))
                    elif pd.isnull(value):
                        logger.info(f"Feature '{feat_name}' is missing in row. Assigning empty string.")
                        features[feat_name] = ""
                    else:
                        features[feat_name] = str(value)
                else:
                    logger.warning(f"Unknown feature type '{feat_type}' for feature '{feat_name}'. Assigning default value 0.0.")
                    features[feat_name] = 0.0
            else:
                if feature['type'] == 'numeric':
                    features[feat_name] = 0.0
                elif feature['type'] == 'text':
                    features[feat_name] = ""
                else:
                    features[feat_name] = 0.0
        return features


    def _add_node(self, node_id: str, node_type: str, features: Dict[str, Any]):
        """
        Adds a node to the graph or updates its features if it already exists.

        Parameters:
        - node_id (str): Unique identifier for the node.
        - node_type (str): Type/category of the node.
        - features (Dict[str, Any]): Features to assign to the node.
        """
        if node_id not in self.node_registry:
            self.node_registry[node_id] = {'type': node_type, 'features': features}
            self.graph.add_node(node_id, type=node_type, **features)
            logger.info(f"Added node {node_id} of type '{node_type}'.")
        else:
            # Update existing node features
            existing_features = self.node_registry[node_id]['features']
            updated_features = {k: v for k, v in features.items() if v != ""}
            existing_features.update(updated_features)
            self.graph.nodes[node_id].update(updated_features)
            logger.info(f"Updated node {node_id} with features {updated_features}.")

    def _add_edge(self, source_id: str, target_id: str, rel_type: str, features: Dict[str, Any]):
        """
        Adds an edge to the graph or updates its features if it already exists.

        Parameters:
        - source_id (str): Source node identifier.
        - target_id (str): Target node identifier.
        - rel_type (str): Type/category of the relationship.
        - features (Dict[str, Any]): Features to assign to the edge.
        """
        edge_key = (source_id, target_id, rel_type)
        if self.graph_type == 'undirected':
            edge_key = tuple(sorted([source_id, target_id])) + (rel_type,)
        
        if not self.graph.has_edge(source_id, target_id, key=rel_type):
            self.edge_registry[edge_key] = features
            self.graph.add_edge(source_id, target_id, key=rel_type, type=rel_type, **features)
            logger.info(f"Added edge from {source_id} to {target_id} of type '{rel_type}'.")
        else:
            # Update existing edge features
            existing_features = self.edge_registry.get(edge_key, {})
            updated_features = {k: v for k, v in features.items() if v != ""}
            existing_features.update(updated_features)
            self.edge_registry[edge_key] = existing_features
            self.graph[source_id][target_id][rel_type].update(updated_features)
            logger.info(f"Updated edge from {source_id} to {target_id} of type '{rel_type}' with features {updated_features}.")

    def get_graph(self) -> nx.Graph:
        """Returns the constructed NetworkX graph."""
        return self.graph

    def export_graph(self, format: str = 'adjacency', path: Optional[str] = None):
        """
        Exports the graph in the specified format.

        Parameters:
        - format (str): The format to export the graph ('adjacency', 'edge_list', etc.).
        - path (str): File path to save the exported graph. If None, returns the data.

        Returns:
        - The exported graph data if path is None.
        """
        if format == 'adjacency':
            data = nx.to_dict_of_dicts(self.graph)
        elif format == 'edge_list':
            # For MultiGraph/MultiDiGraph, each edge is represented with (source, target, key, data)
            # We'll flatten this for the edge list
            edges = []
            for source, target, key, attrs in self.graph.edges(keys=True, data=True):
                edge_data = {
                    'source': source,
                    'target': target,
                    'key': key
                }
                # Exclude 'type' from attributes to prevent duplication
                edge_attributes = {k: (v if v is not None else "") for k, v in attrs.items() if k != 'type'}
                edge_data.update(edge_attributes)
                edges.append(edge_data)
            data = pd.DataFrame(edges)
        else:
            raise ValueError("Unsupported format. Use 'adjacency' or 'edge_list'.")
        
        if path:
            if format == 'adjacency':
                pd.DataFrame.from_dict(data, orient='index').to_csv(path)
                logger.info(f"Graph exported in adjacency format to '{path}'.")
            elif format == 'edge_list':
                data.to_csv(path, index=False)
                logger.info(f"Graph exported in edge list format to '{path}'.")
        else:
            return data

    def graph_visual(self, graph: nx.Graph):
        """
        Visualizes the NetworkX graph with dynamic node and edge types.

        Parameters:
        - graph (nx.Graph): The NetworkX graph to visualize.
        """
        # Extract unique node types
        node_types = set(data.get('type', 'default') for _, data in graph.nodes(data=True))
        node_type_list = sorted(node_types)  # Sort for consistency

        # Assign colors to node types using a colormap
        cmap_nodes = plt.get_cmap('tab10', len(node_type_list))  # Updated line
        node_type_color_map = {ntype: cmap_nodes(i) for i, ntype in enumerate(node_type_list)}

        # Assign colors to nodes based on their type
        node_colors = [node_type_color_map[data.get('type', 'default')] for _, data in graph.nodes(data=True)]

        # Define node sizes based on node degree
        degrees = dict(graph.degree())
        max_degree = max(degrees.values()) if degrees else 1
        node_sizes = [30 + (degrees[node] / max_degree) * 70 for node in graph.nodes()]  # Scale sizes between 30 and 100

        # Extract unique edge relationship types
        edge_types = set(data.get('type', 'default') for _, _, data in graph.edges(data=True))
        edge_type_list = sorted(edge_types)  # Sort for consistency

        # Assign colors to edge types using a colormap
        cmap_edges = plt.get_cmap('Set2', len(edge_type_list))  # Updated line
        edge_type_color_map = {etype: cmap_edges(i) for i, etype in enumerate(edge_type_list)}

        # Assign colors to edges based on their type
        edge_colors = [edge_type_color_map[data.get('type', 'default')] for _, _, data in graph.edges(data=True)]

        # Generate positions for all nodes using spring layout for better visualization
        pos = nx.spring_layout(graph, seed=42, k=0.15, iterations=50)

        plt.figure(figsize=(12, 8))
        
        # Draw nodes with dynamic colors and sizes
        nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=node_sizes, alpha=0.9)

        # Draw edges with dynamic colors
        if self.graph_type == 'directed':
            nx.draw_networkx_edges(graph, pos, edge_color=edge_colors, arrows=True, alpha=0.7, width=1.0)
        else:
            nx.draw_networkx_edges(graph, pos, edge_color=edge_colors, alpha=0.7, width=1.0)

        # Optionally, draw labels (commented out for clarity)
        # nx.draw_networkx_labels(graph, pos, font_size=8, font_family='sans-serif')

        # Create legend for node types
        node_patches = [mpatches.Patch(color=node_type_color_map[ntype], label=ntype) for ntype in node_type_list]

        # Create legend for edge types
        edge_patches = [mpatches.Patch(color=edge_type_color_map[etype], label=etype) for etype in edge_type_list]

        # Combine legends
        plt.legend(handles=node_patches + edge_patches, loc='upper right', bbox_to_anchor=(1.3, 1))

        # Remove axis
        plt.axis('off')

        # Set title
        plt.title("Graph Visualization", fontsize=15)

        # Adjust layout to make room for legends
        plt.tight_layout()

        # Display the graph
        plt.show()



class GraphConverter:
    def __init__(self,
                 df_to_graph: 'DataFrameToGraph',
                 config: Dict[str, Any],
                 methods: Optional[List[str]] = None):
        """
        Initializes the GraphConverter instance.

        Parameters:
        - df_to_graph (DataFrameToGraph): An instance of the DataFrameToGraph class.
        - config (Dict[str, Any]): Configuration dictionary for GraphConverter.
        - methods (List[str], optional): List of method names to execute upon instantiation.
                                         If None, all available methods will be executed.
        """
        self.df_to_graph = df_to_graph
        self.config = config
        self.graph = self.df_to_graph.get_graph()
        self.methods = methods or [
            'convert_to_torch_geometric',
            'embed_text_features',
            'embed_numeric_features'
        ]
        self.pyg_data = None
        self.embeddings = {}
        self.scalers = {}
        self.encoders = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize BERT model and tokenizer for text embeddings
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased').to(self.device)
        self.bert_model.eval()  # Set to evaluation mode

        # Execute selected methods
        for method in self.methods:
            if hasattr(self, method):
                logger.info(f"Executing method: {method}")
                getattr(self, method)()
            else:
                logger.warning(f"Method '{method}' not found in GraphConverter.")

    def convert_to_torch_geometric(self):
        """
        Converts the NetworkX graph into a PyTorch Geometric Data object.
        """
        logger.info("Converting NetworkX graph to PyTorch Geometric Data object.")

        # Ensure the graph is a simple graph (no multiedges) for PyG
        if isinstance(self.graph, (nx.MultiGraph, nx.MultiDiGraph)):
            logger.warning("MultiGraph detected. Converting to simple Graph by merging multiple edges.")
            simple_graph = nx.Graph(self.graph)
        else:
            simple_graph = self.graph

        # Mapping node IDs to integer indices
        node_id_mapping = {node_id: idx for idx, node_id in enumerate(simple_graph.nodes())}
        self.node_id_mapping = node_id_mapping  # Store for reference

        # Extract edge indices and attributes
        edge_index = []
        edge_attr = []
        numeric_edge_features = self.config.get('numeric_features', {}).get('edges', [])

        for source, target, data in simple_graph.edges(data=True):
            edge_index.append([node_id_mapping[source], node_id_mapping[target]])

            if numeric_edge_features:
                # Extract only the numeric edge features
                current_edge_attr = []
                for feat in numeric_edge_features:
                    value = data.get(feat, 0.0)
                    try:
                        current_edge_attr.append(float(value))
                    except (ValueError, TypeError):
                        logger.warning(f"Non-numeric feature value encountered: {value}. Replacing with 0.0.")
                        current_edge_attr.append(0.0)
                edge_attr.append(current_edge_attr)
            else:
                # No numeric edge features
                pass  # Do not append anything to edge_attr

        if edge_index:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        if numeric_edge_features:
            # Ensure all edge_attr entries have the same length
            feature_length = len(numeric_edge_features)
            for idx, attr in enumerate(edge_attr):
                if len(attr) != feature_length:
                    logger.error(f"Edge attribute at index {idx} has length {len(attr)}, expected {feature_length}.")
                    raise ValueError(f"Inconsistent edge attribute length at index {idx}.")

            edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        else:
            edge_attr = torch.empty((0, 0), dtype=torch.float)

        # Extract node features
        node_features = []
        numeric_node_features = self.config.get('numeric_features', {}).get('nodes', [])

        for node_id, data in simple_graph.nodes(data=True):
            if numeric_node_features:
                current_node_features = []
                for feat in numeric_node_features:
                    value = data.get(feat, 0.0)
                    try:
                        current_node_features.append(float(value))
                    except (ValueError, TypeError):
                        logger.warning(f"Non-numeric feature value encountered: {value}. Replacing with 0.0.")
                        current_node_features.append(0.0)
                node_features.append(current_node_features)
            else:
                # No numeric node features
                pass  # Do not append anything to node_features

        if numeric_node_features:
            # Ensure all node_features entries have the same length
            feature_length = len(numeric_node_features)
            for idx, feat in enumerate(node_features):
                if len(feat) != feature_length:
                    logger.error(f"Node feature at index {idx} has length {len(feat)}, expected {feature_length}.")
                    raise ValueError(f"Inconsistent node feature length at index {idx}.")

            node_features = torch.tensor(node_features, dtype=torch.float)
        else:
            node_features = torch.empty((simple_graph.number_of_nodes(), 0), dtype=torch.float)

        # Create PyTorch Geometric Data object
        self.pyg_data = PyGData(x=node_features,
                                edge_index=edge_index,
                                edge_attr=edge_attr)

        logger.info("Conversion to PyTorch Geometric Data object completed.")


    def embed_text_features(self):
        """
        Converts text features into sentence embeddings using BERT.
        """
        logger.info("Embedding text features using BERT.")

        # Identify node and edge features that are textual
        text_features_config = self.config.get('text_features', {})
        node_text_features = text_features_config.get('nodes', [])
        edge_text_features = text_features_config.get('edges', [])

        # Embed node text features
        for node_conf in self.df_to_graph.config.get('nodes', []):
            node_type = node_conf.get('type', 'default')
            for feat in node_conf.get('features', []):
                if feat in node_text_features:
                    self._embed_node_feature(feat)

        # Embed edge text features
        for rel_conf in self.df_to_graph.config.get('relationships', []):
            rel_type = rel_conf.get('type', 'default')
            for feat in rel_conf.get('features', []):
                if feat in edge_text_features:
                    self._embed_edge_feature(feat)

        logger.info("Text feature embedding completed.")

    def _embed_node_feature(self, feature_name: str):
        """
        Embeds a specific node feature.

        Parameters:
        - feature_name (str): The name of the node feature to embed.
        """
        logger.info(f"Embedding node feature: {feature_name}")
        nodes = self.graph.nodes(data=True)
        texts = [data.get(feature_name, "") for _, data in nodes]

        # Generate embeddings
        embeddings = self._generate_text_embeddings(texts)

        # Store embeddings
        self.embeddings[feature_name] = embeddings

        # Append embeddings to node features in pyg_data if available
        if self.pyg_data:
            if self.pyg_data.x.shape[1] == 0:
                self.pyg_data.x = torch.tensor(embeddings, dtype=torch.float)
            else:
                self.pyg_data.x = torch.cat([self.pyg_data.x, torch.tensor(embeddings, dtype=torch.float)], dim=1)

    def _embed_edge_feature(self, feature_name: str):
        """
        Embeds a specific edge feature.

        Parameters:
        - feature_name (str): The name of the edge feature to embed.
        """
        logger.info(f"Embedding edge feature: {feature_name}")
        edges = self.graph.edges(data=True)
        texts = [data.get(feature_name, "") for _, _, data in edges]

        # Generate embeddings
        embeddings = self._generate_text_embeddings(texts)

        # Store embeddings
        self.embeddings[feature_name] = embeddings

        # Append embeddings to edge features in pyg_data if available
        if self.pyg_data and self.pyg_data.edge_attr is not None and self.pyg_data.edge_attr.shape[1] > 0:
            self.pyg_data.edge_attr = torch.cat([self.pyg_data.edge_attr, torch.tensor(embeddings, dtype=torch.float)], dim=1)
        elif self.pyg_data:
            self.pyg_data.edge_attr = torch.tensor(embeddings, dtype=torch.float)

    def _generate_text_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generates BERT embeddings for a list of texts.

        Parameters:
        - texts (List[str]): List of text strings to embed.

        Returns:
        - np.ndarray: Array of embeddings.
        """
        embeddings = []
        batch_size = 16  # Adjust based on your GPU/CPU capacity

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                encoded = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt')
                input_ids = encoded['input_ids'].to(self.device)
                attention_mask = encoded['attention_mask'].to(self.device)

                outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
                last_hidden_states = outputs.last_hidden_state
                # Use the [CLS] token representation
                cls_embeddings = last_hidden_states[:, 0, :].cpu().numpy()
                embeddings.append(cls_embeddings)

        embeddings = np.vstack(embeddings)
        return embeddings

    def embed_numeric_features(self):
        """
        Transforms numerical and categorical features into embeddings.
        """
        logger.info("Embedding numerical and categorical features.")

        numeric_features_config = self.config.get('numeric_features', {})
        node_numeric_features = numeric_features_config.get('nodes', [])
        edge_numeric_features = numeric_features_config.get('edges', [])

        # Embed node numeric features
        for node_conf in self.df_to_graph.config.get('nodes', []):
            node_type = node_conf.get('type', 'default')
            for feat in node_conf.get('features', []):
                if feat in node_numeric_features:
                    self._embed_node_numeric_feature(feat)

        # Embed edge numeric features
        for rel_conf in self.df_to_graph.config.get('relationships', []):
            rel_type = rel_conf.get('type', 'default')
            for feat in rel_conf.get('features', []):
                if feat in edge_numeric_features:
                    self._embed_edge_numeric_feature(feat)

        logger.info("Numerical and categorical feature embedding completed.")

    def _embed_node_numeric_feature(self, feature_name: str):
        """
        Embeds a specific node numerical feature.

        Parameters:
        - feature_name (str): The name of the node feature to embed.
        """
        logger.info(f"Embedding node numerical feature: {feature_name}")
        nodes = self.graph.nodes(data=True)
        values = [data.get(feature_name, 0.0) for _, data in nodes]

        # Scale numerical features
        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(np.array(values).reshape(-1, 1)).flatten()
        self.scalers[feature_name] = scaler

        # Store embeddings
        self.embeddings[feature_name] = scaled_values

        # Append to node features in pyg_data if available
        if self.pyg_data:
            if self.pyg_data.x.shape[1] == 0:
                self.pyg_data.x = torch.tensor(scaled_values, dtype=torch.float).unsqueeze(1)
            else:
                self.pyg_data.x = torch.cat([self.pyg_data.x, torch.tensor(scaled_values, dtype=torch.float).unsqueeze(1)], dim=1)

    def _embed_edge_numeric_feature(self, feature_name: str):
        """
        Embeds a specific edge numerical feature.

        Parameters:
        - feature_name (str): The name of the edge feature to embed.
        """
        logger.info(f"Embedding edge numerical feature: {feature_name}")
        edges = self.graph.edges(data=True)
        values = [data.get(feature_name, 0.0) for _, _, data in edges]

        # Scale numerical features
        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(np.array(values).reshape(-1, 1)).flatten()
        self.scalers[feature_name] = scaler

        # Store embeddings
        self.embeddings[feature_name] = scaled_values

        # Append to edge features in pyg_data if available
        if self.pyg_data and self.pyg_data.edge_attr is not None:
            self.pyg_data.edge_attr = torch.cat([
                self.pyg_data.edge_attr,
                torch.tensor(scaled_values, dtype=torch.float).unsqueeze(1)
            ], dim=1)
        elif self.pyg_data:
            self.pyg_data.edge_attr = torch.tensor(scaled_values, dtype=torch.float).unsqueeze(1)

    def _validate_numerical_features(self, features: List[Any]) -> List[float]:
        """
        Validates and converts feature values to float.

        Parameters:
        - features (List[Any]): List of feature values.

        Returns:
        - List[float]: Cleaned list of float values.
        """
        cleaned_features = []
        for feature in features:
            try:
                # Convert to float if possible
                cleaned_features.append(float(feature))
            except (ValueError, TypeError):
                logger.warning(f"Non-numeric feature value encountered: {feature}. Replacing with 0.0.")
                cleaned_features.append(0.0)  # Replace invalid values with a default, e.g., 0.0
        return cleaned_features

    def add_custom_method(self, method_name: str, method: Callable):
        """
        Adds a custom method to the GraphConverter class.

        Parameters:
        - method_name (str): The name of the method.
        - method (Callable): The method function.
        """
        if hasattr(self, method_name):
            logger.warning(f"Method '{method_name}' already exists and will be overwritten.")
        setattr(self, method_name, method)
        logger.info(f"Custom method '{method_name}' added to GraphConverter.")

    def get_pyg_data(self) -> Optional[PyGData]:
        """
        Retrieves the PyTorch Geometric Data object.

        Returns:
        - PyGData or None: The PyG Data object if conversion has been done.
        """
        return self.pyg_data

    def get_embeddings(self) -> Dict[str, Any]:
        """
        Retrieves the embeddings generated.

        Returns:
        - Dict[str, Any]: Dictionary of embeddings.
        """
        return self.embeddings

    def save_pyg_data(self, path: str):
        """
        Saves the PyTorch Geometric Data object to disk.

        Parameters:
        - path (str): File path to save the data.
        """
        if self.pyg_data:
            torch.save(self.pyg_data, path)
            logger.info(f"PyTorch Geometric Data object saved to '{path}'.")
        else:
            logger.warning("PyG Data object is not available. Conversion might not have been performed.")

    def load_pyg_data(self, path: str):
        """
        Loads a PyTorch Geometric Data object from disk.

        Parameters:
        - path (str): File path from which to load the data.
        """
        self.pyg_data = torch.load(path)
        logger.info(f"PyTorch Geometric Data object loaded from '{path}'.")


def main():

    # Enhanced Sample DataFrame for a Social Network
    df = pd.read_csv('prince-toronto.csv')
    
    # Configuration Dictionary (Including 'features' with their types)
    config = {
        "nodes": [
            {
                "id": "tweet_id",
                "type": "Post",
                "features": [
                    {"name": "retweet_count", "type": "numeric"},
                    {"name": "lang", "type": "text"}
                ]
            },
            {
                "id": "reply_to_tweet_id",  # Added node definition for reply_to_tweet_id
                "type": "Post",
                "features": [
                    {"name": "retweet_count", "type": "numeric"},
                    {"name": "lang", "type": "text"}
                ]
            },
            {
                "id": "user_id",
                "type": "User",
                "features": [
                    {"name": "favorite_count", "type": "numeric"},
                    {"name": "user_friends_count", "type": "numeric"}
                ]
            }
        ],
        "relationships": [
            {
                "source": "tweet_id",
                "target": "reply_to_tweet_id",
                "type": "replied",
                "features": [
                    {"name": "mentions", "type": "text"},
                    {"name": "hashtags", "type": "text"}
                ]
            },
            {
                "source": "user_id",
                "target": "tweet_id",
                "type": "posted",
                "features": [
                    {"name": "geo", "type": "text"}
                ]
            }
        ]
    }

    # GraphConverter Configuration
    graph_converter_config = {
        "text_features": {
            "nodes": ["lang"],
            "edges": ["mentions", "hashtags", "geo"]  # Include all textual edge features
        },
        "numeric_features": {
            "nodes": ["retweet_count", "favorite_count", "user_friends_count"],
            "edges": []  # No numeric edge features in this configuration
        }
    }

    # Initialize the DataFrameToGraph instance with enhanced configuration
    df_to_graph = DataFrameToGraph(df, config, graph_type='directed')
    
    # Retrieve the constructed graph
    graph = df_to_graph.get_graph()

    # Display nodes with attributes
    print("Nodes:")
    for node, attrs in graph.nodes(data=True):
        print(node, attrs)

    # Display edges with attributes
    print("\nEdges:")
    for source, target, key, attrs in graph.edges(keys=True, data=True):
        print(f"{source} -> {target} [type={attrs.get('type')}, key={key}] {attrs}")

    # Export the graph as an edge list CSV
    df_to_graph.export_graph(format='edge_list', path='graph_edge_list.csv')

    # Alternatively, get the adjacency dictionary
    adjacency_dict = df_to_graph.export_graph(format='adjacency')
    print("\nAdjacency Dictionary:")
    print(adjacency_dict)

    # Visualization Code Starts Here
    # --------------------------------

    # Visualize the graph
    df_to_graph.graph_visual(graph)

    # Visualization Code Ends Here
    # --------------------------------

    # Initialize the GraphConverter instance with updated method order
    graph_converter = GraphConverter(
        df_to_graph=df_to_graph,
        config=graph_converter_config,
        methods=[
            'embed_text_features',
            'embed_numeric_features',
            'convert_to_torch_geometric'
        ]
    )

    # Retrieve the PyTorch Geometric Data object
    pyg_data = graph_converter.get_pyg_data()
    print("\nPyTorch Geometric Data Object:")
    print(pyg_data)

    #Display information about the PyG Data object
    print("\nPyG Data Object Information:")
    print(f"Number of nodes: {pyg_data.num_nodes}")
    print(f"Number of edges: {pyg_data.num_edges}")
    print(f"Node features: {pyg_data.x.shape}")
    print(f"Edge features: {pyg_data.edge_attr.shape}")
    print(f"Edge indices: {pyg_data.edge_index.shape}")
    
    # Save the PyG Data object to disk
    graph_converter.save_pyg_data('graph_pyg_data.pt')

    # Optionally, load the PyG Data object from disk
    # graph_converter.load_pyg_data('graph_pyg_data.pt')

    # Access embeddings
    embeddings = graph_converter.get_embeddings()
    print("\nEmbeddings:")
    for feature, embed in embeddings.items():
        print(f"{feature}: {embed.shape}")


if __name__ == '__main__':
    main()