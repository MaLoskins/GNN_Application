// src/App.js
import React, { useState, useEffect } from 'react';
import { processData } from './api';
import { 
  ReactFlowProvider, 
  addEdge, 
  useNodesState, 
  useEdgesState 
} from 'react-flow-renderer'; // Named Imports
import FileUploader from './components/FileUploader/FileUploader';
import ConfigurationPanel from './components/ConfigurationPanel/ConfigurationPanel';
import GraphVisualizer from './components/GraphVisualizer/GraphVisualizer';
import RelationshipModal from './components/RelationshipModal/RelationshipModal';
import ReactFlowWrapper from './components/ReactFlowWrapper/ReactFlowWrapper'; // Import the new component
import './App.css';
import 'react-flow-renderer/dist/style.css'; // Import React Flow's default styles

function App() {
  const [csvData, setCsvData] = useState([]);
  const [columns, setColumns] = useState([]);
  const [config, setConfig] = useState({ nodes: [], relationships: [], graph_type: 'directed' });
  const [graphData, setGraphData] = useState(null);
  const [loading, setLoading] = useState(false);

  // Initialize nodes and edges using hooks
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);

  // Modal States
  const [modalIsOpen, setModalIsOpen] = useState(false);
  const [currentEdge, setCurrentEdge] = useState(null);

  // Handle file drop
  const handleFileDrop = (data, fields) => {
    setCsvData(data);
    setColumns(fields);
    initializeConfig(fields);
    initializeReactFlow(fields);
  };

  // Initialize config with default selections
  const initializeConfig = (fields) => {
    setConfig({
      nodes: fields.map((field) => ({
        id: field,
        type: 'default',
        features: [],
      })),
      relationships: [],
      graph_type: 'directed', // You can change this to 'undirected' if needed
    });
  };

  // Initialize React Flow nodes with grid layout
  const initializeReactFlow = (fields) => {
    const columnsPerRow = 6; // Increased columns per row for wider screens
    const nodeSpacingX = 150; // Reduced horizontal spacing
    const nodeSpacingY = 150; // Reduced vertical spacing
    const flowNodes = fields.map((field, index) => {
      const row = Math.floor(index / columnsPerRow);
      const col = index % columnsPerRow;
      const x = col * nodeSpacingX;
      const y = row * nodeSpacingY;

      return {
        id: field,
        type: 'default',
        data: { label: field },
        position: { x, y },
      };
    });

    setNodes(flowNodes);
  };

  // Handle relationship submission
  const handleSaveRelationship = ({ relationshipType, relationshipFeatures }) => {
    if (!relationshipType) {
      alert('Please enter a relationship type.');
      return;
    }

    // Update config.relationships using functional updater to access previous state
    setConfig((prevConfig) => {
      const newRelationship = {
        source: currentEdge.source,
        target: currentEdge.target,
        type: relationshipType,
        features: relationshipFeatures,
      };

      // Create the new edge with data
      const edgeWithData = {
        ...currentEdge,
        type: 'smoothstep', // You can choose any edge type you prefer   
        
        animated: prevConfig.graph_type === 'directed',
        arrowHeadType: prevConfig.graph_type === 'directed' ? 'arrowclosed' : 'none',
        label: relationshipType,
        data: { type: relationshipType, features: relationshipFeatures },
      };

      // Add the new relationship to the config
      const updatedConfig = {
        ...prevConfig,
        relationships: [...prevConfig.relationships, newRelationship],
      };

      // Update edges state
      setEdges((eds) => addEdge(edgeWithData, eds));

      return updatedConfig;
    });

    // Reset modal state
    setModalIsOpen(false);
    setCurrentEdge(null);
  };

  // Handle edge creation
  const onConnectHandler = (params) => {
    setCurrentEdge(params);
    setModalIsOpen(true);
  };

  // Handle form submission
  const handleSubmit = async () => {
    // Basic validation
    for (let node of config.nodes) {
      if (!node.id) {
        alert('Please select an ID column for all nodes.');
        return;
      }
      if (!node.type) {
        alert('Please specify a type for all nodes.');
        return;
      }
    }
    for (let rel of config.relationships) {
      if (!rel.source || !rel.target) {
        alert('Please select source and target columns for all relationships.');
        return;
      }
      if (!rel.type) {
        alert('Please specify a type for all relationships.');
        return;
      }
    }

    setLoading(true);
    try {
      const response = await processData(csvData, config);
      const receivedGraph = response.graph;

      // Compute node degrees
      const degrees = {};
      receivedGraph.links.forEach((link) => {
        if (receivedGraph.directed) {
          degrees[link.source] = (degrees[link.source] || 0) + 1; // Out-degree
          degrees[link.target] = (degrees[link.target] || 0) + 1; // In-degree
        } else {
          degrees[link.source] = (degrees[link.source] || 0) + 1;
          degrees[link.target] = (degrees[link.target] || 0) + 1;
        }
      });

      // Assign degree to nodes
      const nodesWithSize = receivedGraph.nodes.map((node) => ({
        ...node,
        val: degrees[node.id] || 1, // Default size if degree is 0
      }));

      const updatedGraphData = {
        ...receivedGraph,
        nodes: nodesWithSize,
      };

      setGraphData(updatedGraphData);
    } catch (error) {
      console.error('Error processing data:', error);
      alert('Error processing data. Check console for details.');
    }
    setLoading(false);
  };

  // Handle window resize for responsive graph
  const [dimensions, setDimensions] = useState({
    width: window.innerWidth * 0.9, // Increased width to 90%
    height: (window.innerWidth * 0.9) * (2 / 3), // Maintained 3:2 ratio
  });

  useEffect(() => {
    const handleResize = () => {
      setDimensions({
        width: window.innerWidth * 0.9, // 90% width
        height: (window.innerWidth * 0.9) * (2 / 3), // 3:2 ratio
      });
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  return (
    <ReactFlowProvider>
      <div className="App">
        <h1>CSV to Graph Application</h1>

        {/* CSV Upload Section */}
        <FileUploader onFileDrop={handleFileDrop} />

        {/* Configuration Section */}
        {columns.length > 0 && (
          <ConfigurationPanel
            columns={columns}
            onSubmit={handleSubmit}
            loading={loading}
          />
        )}

        {/* React Flow Configurator */}
        {columns.length > 0 && (
          <ReactFlowWrapper
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnectHandler}
          />
        )}

        {/* Graph Visualization Section */}
        {graphData && (
          <GraphVisualizer graphData={graphData} dimensions={dimensions} />
        )}

        {/* Relationship Modal */}
        <RelationshipModal
          isOpen={modalIsOpen}
          onRequestClose={() => setModalIsOpen(false)}
          columns={columns}
          onSaveRelationship={handleSaveRelationship}
        />
      </div>
    </ReactFlowProvider>
  );
}

export default App;

/*
Detailed Explanation:

The `App` component serves as the root component of the React application, orchestrating the overall workflow from CSV file upload to graph visualization. Here's a comprehensive breakdown of its structure and functionality:

1. **Imports**:
   - **React**, **useState**, **useEffect**: Core React library and hooks for managing state and side effects.
   - **processData** from `./api`: A function responsible for sending the uploaded CSV data and configuration to the backend for processing.
   - **ReactFlowProvider**, **addEdge**, **useNodesState**, **useEdgesState** from `react-flow-renderer`: Components and hooks from the `react-flow-renderer` library for managing and rendering interactive flow diagrams.
   - **Custom Components**: `FileUploader`, `ConfigurationPanel`, `GraphVisualizer`, `RelationshipModal`, and `ReactFlowWrapper` are custom components that handle specific parts of the application.
   - **CSS Files**: `App.css` and `react-flow-renderer/dist/style.css` for styling the application and the React Flow diagrams.

2. **State Variables**:
   - **`csvData`**: Holds the data parsed from the uploaded CSV file.
   - **`columns`**: Stores the headers (column names) extracted from the CSV file.
   - **`config`**: An object containing the configuration for nodes, relationships, and graph type (`directed` by default).
   - **`graphData`**: Stores the processed graph data received from the backend, ready for visualization.
   - **`loading`**: A boolean indicating whether the data processing is currently underway.
   - **`nodes`**, **`setNodes`**, **`onNodesChange`**: Managed by `useNodesState`, these handle the state and updates of nodes in the React Flow diagram.
   - **`edges`**, **`setEdges`**, **`onEdgesChange`**: Managed by `useEdgesState`, these handle the state and updates of edges in the React Flow diagram.
   - **`modalIsOpen`**: Controls the visibility of the `RelationshipModal`.
   - **`currentEdge`**: Holds the information about the edge currently being configured in the modal.

3. **Handlers and Initialization Functions**:
   - **`handleFileDrop`**: Invoked when a user uploads a CSV file. It updates `csvData` and `columns`, and initializes the configuration and React Flow nodes based on the uploaded data.
   - **`initializeConfig`**: Sets up the default configuration for nodes based on the CSV columns, initializing each node with an `id`, `type`, and empty `features`.
   - **`initializeReactFlow`**: Arranges the nodes in a grid layout within the React Flow diagram, determining their initial positions based on predefined spacing and the number of columns per row.
   - **`handleSaveRelationship`**: Called when a user saves a relationship configuration from the modal. It updates the `config` with the new relationship, adds the corresponding edge to the React Flow diagram, and closes the modal.
   - **`onConnectHandler`**: Triggered when a new edge is created between nodes in the React Flow diagram. It sets the current edge being connected and opens the `RelationshipModal`.
   - **`handleSubmit`**: Handles the submission of the entire configuration. It performs basic validation to ensure all required fields are filled, sends the data to the backend for processing via `processData`, and updates `graphData` based on the response. It also calculates node degrees to determine their sizes in the visualization.

4. **Responsive Design Handling**:
   - **`dimensions`**: Maintains the width and height of the graph visualization based on the window size, ensuring responsiveness.
   - **`useEffect`**: Adds an event listener for window resize events to dynamically update `dimensions`, maintaining a 3:2 aspect ratio for the graph.

5. **Rendering**:
   - **`ReactFlowProvider`**: Wraps the entire application to provide context for React Flow components.
   - **`div.App`**: The main container for the application, styled via `App.css`.
   - **Components Rendered**:
     - **`FileUploader`**: Allows users to upload CSV files.
     - **`ConfigurationPanel`**: Displays configuration options once columns are available.
     - **`ReactFlowWrapper`**: Renders the interactive flow diagram based on nodes and edges.
     - **`GraphVisualizer`**: Visualizes the processed graph data in a force-directed layout.
     - **`RelationshipModal`**: Provides a modal interface for configuring relationships between nodes.

6. **Export**:
   - The `App` component is exported as the default export, serving as the entry point of the React application.

**Purpose in the Application**:
The `App` component orchestrates the entire workflow of the application:

1. **Data Ingestion**: Users upload a CSV file using the `FileUploader`, which parses the data and extracts column information.
2. **Configuration**: The `ConfigurationPanel` allows users to define how different columns relate to each other, specifying nodes and relationships.
3. **Visualization Setup**: The `ReactFlowWrapper` presents an interactive diagram where users can visually arrange and connect nodes.
4. **Relationship Management**: When users create or modify relationships between nodes, the `RelationshipModal` facilitates detailed configuration.
5. **Data Processing**: Upon submission, the application sends the configured data to the backend (`processData`) for processing into a graph structure.
6. **Graph Visualization**: The `GraphVisualizer` component displays the resulting graph, allowing users to explore and analyze the relationships in a dynamic, interactive manner.

Overall, the `App` component integrates all sub-components to provide a seamless experience for transforming CSV data into interactive graph visualizations, enabling users to gain insights from their data through configuration and visualization tools.

*/

