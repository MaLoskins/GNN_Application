# DataFrameToGraph.py
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple, Optional
import logging
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import matplotlib.colors as colors

# Set up logging
# Set up logging to file instead of console
logging.basicConfig(
    level=logging.INFO,
    filename='logs/data_to_graph.log',         # Specify the file name for the log file
    filemode='w',               # 'w' for write mode, 'a' for append mode
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataFrameToGraph:
    def __init__(
        self,
        df: pd.DataFrame,
        config: Dict[str, Any],
        graph_type: str = 'directed',
        feature_space: Optional[Dict[str, Any]] = None
    ):
        """
        Initializes the DataFrameToGraph instance.

        Parameters:
        - df (pd.DataFrame): The input DataFrame containing tabular data.
        - config (Dict[str, Any]): Configuration dictionary defining column roles.
        - graph_type (str): Type of the graph ('directed' or 'undirected').
        - feature_space (Dict[str, Any]): Feature space data for nodes and edges.
        """
        self.df = df
        self.config = config
        self.graph_type = graph_type.lower()
        self.feature_space = feature_space  # Store feature space
        self.graph = self._initialize_graph()
        self.node_registry = {}
        self.edge_registry = {}
        
        self._validate_config()

        # Convert feature_space to DataFrame if necessary
        if self.feature_space is not None:
            # Convert feature_space to DataFrame
            self.feature_space = pd.DataFrame(self.feature_space)
            # Ensure that the index of feature_space is node IDs
            if 'index' in self.feature_space.columns:
                self.feature_space.set_index('index', inplace=True)
            else:
                self.feature_space.index = self.feature_space.index.astype(str)

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

        # Validate relationships configuration
        for rel_conf in self.config['relationships']:
            if 'source' not in rel_conf or 'target' not in rel_conf:
                raise KeyError("Each relationship configuration must have 'source' and 'target' keys.")
            if 'type' not in rel_conf:
                logger.warning(f"Relationship configuration {rel_conf} missing 'type'. Defaulting to 'default'.")

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

                # Ensure both source and target nodes are present in the graph
                if source_id_str not in self.node_registry:
                    logger.warning(f"Row {index}: Source node '{source_id_str}' not in selected nodes. Skipping edge addition.")
                    continue
                if target_id_str not in self.node_registry:
                    logger.warning(f"Row {index}: Target node '{target_id_str}' not in selected nodes. Skipping edge addition.")
                    continue

                self._add_edge(source_id_str, target_id_str, relationship_type, features)

    def _extract_features(self, row: pd.Series, feature_cols: List[str]) -> Dict[str, Any]:
        """Extracts feature data from the row based on specified columns."""
        features = {}
        for feat in feature_cols:
            if feat in row:
                value = row[feat]
                if isinstance(value, list):
                    # Assume the list is present and valid
                    features[feat] = value
                elif pd.isnull(value):
                    logger.info(f"Feature '{feat}' is missing in row. Assigning default value.")
                    features[feat] = ""
                else:
                    features[feat] = value
            else:
                logger.info(f"Feature '{feat}' not found in row. Assigning default value.")
                features[feat] = ""
        return features

    def _add_node(self, node_id: str, node_type: Optional[str], features: Dict[str, Any]):
        """
        Adds a node to the graph or updates its features if it already exists.

        Parameters:
        - node_id (str): Unique identifier for the node.
        - node_type (str): Type/category of the node.
        - features (Dict[str, Any]): Features to assign to the node.
        """
        node_type = node_type or 'default'  # Set default node type if None

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

    def graph_visual(self):
        """
        Visualizes the NetworkX graph with dynamic node and edge types.

        Parameters:
        - graph (nx.Graph): The NetworkX graph to visualize.
        """
        graph = self.graph
        
        # Extract unique node types
        node_types = set(data.get('type', 'default') for _, data in graph.nodes(data=True))
        node_type_list = sorted(node_types)  # Sort for consistency

        # Assign colors to node types using a colormap
        cmap_nodes = cm.get_cmap('tab10', len(node_type_list))
        node_type_color_map = {ntype: cmap_nodes(i) for i, ntype in enumerate(node_type_list)}

        # Assign colors to nodes based on their type
        node_colors = [node_type_color_map[data.get('type', 'default')] for _, data in graph.nodes(data=True)]

        # Define node sizes based on node degree
        degrees = dict(graph.degree())
        max_degree = max(degrees.values()) if degrees else 1
        node_sizes = [30 + (degrees[node] / max_degree) * 70 for node in graph.nodes()]  # Scale sizes between 300 and 1000

        # Extract unique edge relationship types
        edge_types = set(data.get('type', 'default') for _, _, data in graph.edges(data=True))
        edge_type_list = sorted(edge_types)  # Sort for consistency

        # Assign colors to edge types using a colormap
        cmap_edges = cm.get_cmap('Set2', len(edge_type_list))
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

def main():

    # Enhanced Sample DataFrame for a Social Network
    df = pd.read_csv('prince-toronto.csv')
    # Configuration Dictionary (Including 'features' Key)
    config = {
        "nodes": [
            {
                "id": "tweet_id",
                "type": "Post",
                "features": ["retweet_count", "lang"]
            },
            {
                "id": "reply_to_tweet_id",  # Added node definition for reply_to_tweet_id
                "type": "Post",
                "features": ["retweet_count", "lang"]
            },
            {
                "id": "user_id",
                "type": "User",
                "features": ["favorite_count", "user_friends_count"]
            }
        ],
        "relationships": [
            {
                "source": "tweet_id",
                "target": "reply_to_tweet_id",
                "type": "replied",
                "features": ["mentions", "hashtags"]
            },
            {
                "source": "user_id",
                "target": "tweet_id",
                "type": "posted",
                "features": ["geo"]
            }
        ]
    }

    # Initialize the DataFrameToGraph instance
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
    df_to_graph.graph_visual()



if __name__ == "__main__":
    main()