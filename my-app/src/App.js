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
import NodeEditModal from './components/NodeEditModal/NodeEditModal'; // Import the new component
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

  // Modal States for Relationships
  const [relationshipModalIsOpen, setRelationshipModalIsOpen] = useState(false);
  const [currentEdge, setCurrentEdge] = useState(null);

  // Modal States for Node Editing
  const [nodeEditModalIsOpen, setNodeEditModalIsOpen] = useState(false);
  const [currentNode, setCurrentNode] = useState(null);

  // Handle file drop
  const handleFileDrop = (data, fields) => {
    setCsvData(data);
    setColumns(fields);
    initializeConfig(fields);
    initializeReactFlow(fields);
  };

  // Initialize config with selectable nodes
  const initializeConfig = (fields) => {
    setConfig({
      nodes: [], // Initially no nodes are selected
      relationships: [],
      graph_type: 'directed', // You can change this to 'undirected' if needed
    });
  };

  // Initialize React Flow nodes with grid layout (only selected nodes will appear)
  const initializeReactFlow = (fields) => {
    // Initially, no nodes are selected, so ReactFlow will render empty
    setNodes([]);
    setEdges([]);
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
    setRelationshipModalIsOpen(false);
    setCurrentEdge(null);
  };

  // Handle edge creation
  const onConnectHandler = (params) => {
    setCurrentEdge(params);
    setRelationshipModalIsOpen(true);
  };

  // Handle node click for editing
  const onNodeClickHandler = (node) => {
    setCurrentNode(node);
    setNodeEditModalIsOpen(true);
  };

  // Handle node type and feature updates
  const handleSaveNodeEdit = ({ nodeType, nodeFeatures }) => {
    if (!nodeType) {
      alert('Please enter a node type.');
      return;
    }

    // Update config.nodes using functional updater
    setConfig((prevConfig) => {
      const updatedNodes = prevConfig.nodes.map((n) => {
        if (n.id === currentNode.id) {
          return {
            ...n,
            type: nodeType,
            features: nodeFeatures,
          };
        }
        return n;
      });

      return {
        ...prevConfig,
        nodes: updatedNodes,
      };
    });

    // Update React Flow node data
    setNodes((nds) =>
      nds.map((node) => {
        if (node.id === currentNode.id) {
          return {
            ...node,
            type: nodeType,
            data: { ...node.data, label: `${node.data.label} (${nodeType})` },
          };
        }
        return node;
      })
    );

    // Reset modal state
    setNodeEditModalIsOpen(false);
    setCurrentNode(null);
  };

  // Handle node selection for inclusion as a node in the graph
  const handleSelectNode = (column) => {
    // Check if the column is already selected as a node
    const isSelected = config.nodes.find((n) => n.id === column);

    if (isSelected) {
      // If already selected, remove it
      setConfig((prevConfig) => ({
        ...prevConfig,
        nodes: prevConfig.nodes.filter((n) => n.id !== column),
      }));

      // Remove from React Flow
      setNodes((nds) => nds.filter((n) => n.id !== column));

      // Optionally, remove related edges
      setEdges((eds) => eds.filter((e) => e.source !== column && e.target !== column));
    } else {
      // If not selected, add it with default type
      setConfig((prevConfig) => ({
        ...prevConfig,
        nodes: [...prevConfig.nodes, { id: column, type: 'default', features: [] }],
      }));

      // Add to React Flow with default type and position
      const newNode = {
        id: column,
        type: 'default',
        data: { label: column },
        position: {
          x: Math.random() * 600 - 300, // Random position for simplicity
          y: Math.random() * 600 - 300,
        },
      };

      setNodes((nds) => nds.concat(newNode));
    }
  };

  // Handle form submission
  const handleSubmit = async () => {
    // Basic validation
    for (let node of config.nodes) {
      if (!node.id) {
        alert('Please select an ID column for all nodes.');
        return;
      }
      if (!node.type || node.type === 'default') { // Ensure type is specified
        alert(`Please specify a valid type for node '${node.id}'.`);
        return;
      }
    }
    for (let rel of config.relationships) {
      if (!rel.source || !rel.target) {
        alert('Please select source and target columns for all relationships.');
        return;
      }
      if (!rel.type || rel.type === 'default') { // Ensure relationship type is specified
        alert('Please specify a valid type for all relationships.');
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

  // Update config.nodes whenever nodes state changes (e.g., after node edits)
  useEffect(() => {
    if (columns.length === 0) return;
    setConfig((prevConfig) => ({
      ...prevConfig,
      nodes: nodes.map((node) => ({
        id: node.id,
        type: node.type || 'default',
        features: prevConfig.nodes.find((n) => n.id === node.id)?.features || [],
      })),
    }));
  }, [nodes, columns]);

  return (
    <ReactFlowProvider>
      <div className="App">
        <h1>CSV to Graph Application</h1>

        {/* CSV Upload Section */}
        <FileUploader onFileDrop={handleFileDrop} />

        {/* Configuration Section for Selecting Nodes and Relationships */}
        {columns.length > 0 && (
          <ConfigurationPanel
            columns={columns}
            onSelectNode={handleSelectNode}
            onSubmit={handleSubmit}
            loading={loading}
            selectedNodes={config.nodes.map((n) => n.id)}
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
            onNodeClick={onNodeClickHandler} // Pass node click handler
          />
        )}

        {/* Graph Visualization Section */}
        {graphData && (
          <GraphVisualizer graphData={graphData} dimensions={dimensions} />
        )}

        {/* Relationship Modal */}
        <RelationshipModal
          isOpen={relationshipModalIsOpen}
          onRequestClose={() => setRelationshipModalIsOpen(false)}
          columns={columns}
          onSaveRelationship={handleSaveRelationship}
        />

        {/* Node Edit Modal */}
        {currentNode && (
          <NodeEditModal
            isOpen={nodeEditModalIsOpen}
            onRequestClose={() => setNodeEditModalIsOpen(false)}
            node={currentNode}
            onSaveNodeEdit={handleSaveNodeEdit}
          />
        )}
      </div>
    </ReactFlowProvider>
  );
}

export default App;

/*
Detailed Explanation:

1. **Selective Node Inclusion:**
   - **`handleSelectNode`**: This function toggles the selection of a column as a node. If the column is already selected, it removes it; otherwise, it adds it with a default type.
   - **`ConfigurationPanel`**: Updated to include node selection UI (see the next section).

2. **Dynamic Validation:**
   - The `handleSubmit` function now only validates node types for columns selected as nodes.
   - Previously, all columns were treated as nodes with default types, leading to validation errors for unselected nodes.

3. **Integration with `ReactFlowWrapper`:**
   - **`onNodeClickHandler`**: Handles the event when a node is clicked, opening the `NodeEditModal` for dynamic editing.

4. **State Management:**
   - **`config.nodes`**: Now only includes columns selected as nodes.
   - **`config.relationships`**: Manages relationships defined via `RelationshipModal`.
   - **`nodes` and `edges`**: Managed by React Flow hooks.

5. **UI Components:**
   - **`NodeEditModal`**: Allows dynamic editing of node types and features directly within the graph.

6. **Initial ReactFlow Nodes:**
   - Initially, no nodes are selected; nodes are added by the user via the `ConfigurationPanel`.

7. **Edge Handling:**
   - Only edges between selected nodes are allowed. You might need to enforce this in the `RelationshipModal` or during processing.

8. **Responsive Design:**
   - The `dimensions` state ensures the graph remains responsive to window resizing.

9. **Assumptions:**
   - The `ConfigurationPanel` now includes node selection capabilities (see the next section).
   - The `NodeEditModal` component allows dynamic editing of node types and features.

10. **Backend Synchronization:**
    - The `processData` function sends the updated `config` to the backend, which now only includes selected nodes with their respective types and features.

*/
