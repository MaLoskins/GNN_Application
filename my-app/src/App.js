// my-app/src/App.js
import React, { useState, useEffect } from 'react';
import { processData } from './api';
import ReactFlow, { 
  ReactFlowProvider, 
  addEdge, 
  useNodesState, 
  useEdgesState 
} from 'react-flow-renderer'; // Correctly import ReactFlow as default
import FileUploader from './components/FileUploader';
import ConfigurationPanel from './components/ConfigurationPanel';
import GraphVisualizer from './components/GraphVisualizer';
import RelationshipModal from './components/RelationshipModal';
import './App.css';

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
  const onConnect = (params) => {
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

        {/* React Flow Graph */}
        {columns.length > 0 && (
          <div className="react-flow-wrapper">
            <ReactFlow
              nodes={nodes}
              edges={edges}
              onNodesChange={onNodesChange}
              onEdgesChange={onEdgesChange}
              onConnect={onConnect}
              deleteKeyCode={46} /* 'delete'-key */
              fitView
              style={{ width: '100%', height: '100%' }}
            />
          </div>
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
