// src/App.js

import React, { useState, useEffect } from 'react';
import { processData, createFeatureSpace } from './api'; // Ensure createFeatureSpace is imported
import { 
  ReactFlowProvider, 
  addEdge, 
  useNodesState, 
  useEdgesState 
} from 'react-flow-renderer';
import FileUploader from './components/FileUploader/FileUploader';
import ConfigurationPanel from './components/ConfigurationPanel/ConfigurationPanel';
import GraphVisualizer from './components/GraphVisualizer/GraphVisualizer';
import RelationshipModal from './components/RelationshipModal/RelationshipModal';
import ReactFlowWrapper from './components/ReactFlowWrapper/ReactFlowWrapper';
import NodeEditModal from './components/NodeEditModal/NodeEditModal';
import FeatureSpaceCreatorTab from './components/FeatureSpaceCreatorTab/FeatureSpaceCreatorTab'; // New Tab Component
import './App.css';
import 'react-flow-renderer/dist/style.css';

function App() {
  const [csvData, setCsvData] = useState([]);
  const [columns, setColumns] = useState([]);
  const [config, setConfig] = useState({ nodes: [], relationships: [], graph_type: 'directed' });
  const [graphData, setGraphData] = useState(null);
  const [featureSpaceData, setFeatureSpaceData] = useState(null); // State for Feature Space
  const [loading, setLoading] = useState(false);
  
  // Tab State
  const [activeTab, setActiveTab] = useState('graph'); // 'graph' or 'featureSpace'

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
          x: Math.random() * 400 - 200, // Random position for simplicity
          y: Math.random() * 400 - 200,
        },
      };

      setNodes((nds) => nds.concat(newNode));
    }
  };

  // Handle form submission for graph processing
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

  // Handle Feature Space submission
  const handleFeatureSpaceSubmit = async (featureConfig) => {
    if (csvData.length === 0) {
      alert('Please upload and process CSV data first.');
      return;
    }

    setLoading(true);
    try {
      const response = await createFeatureSpace(csvData, featureConfig);

      // Destructure the response to get features, multi_graph_settings, and feature_space
      const { features, multi_graph_settings, feature_space } = response;

      // Convert feature_space from JSON string to JavaScript object
      const featureSpaceObj = JSON.parse(feature_space);

      // Attach features and multi_graph_settings to featureSpaceObj
      featureSpaceObj.features = features;
      featureSpaceObj.multi_graph_settings = multi_graph_settings;

      // Set the feature space data state
      setFeatureSpaceData(featureSpaceObj);
      alert('Feature space created successfully.');
    } catch (error) {
      console.error('Error creating feature space:', error);
      alert('Error creating feature space. Check console for details.');
    }
    setLoading(false);
  };


  // Handle window resize for responsive graph
  const [dimensions, setDimensions] = useState({
    width: window.innerWidth * 0.9, // 90% width
    height: (window.innerWidth * 0.9) * (2 / 3), // 3:2 ratio
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

        {/* Tab Navigation */}
        <div className="tab-navigation">
          <button
            className={`tab-button ${activeTab === 'graph' ? 'active' : ''}`}
            onClick={() => setActiveTab('graph')}
          >
            Graph
          </button>
          <button
            className={`tab-button ${activeTab === 'featureSpace' ? 'active' : ''}`}
            onClick={() => setActiveTab('featureSpace')}
          >
            Feature Space Creator
          </button>
        </div>

        {/* Tab Content */}
        {activeTab === 'graph' && (
          <>
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

          </>
        )}

        {activeTab === 'featureSpace' && (
          <FeatureSpaceCreatorTab
            csvData={csvData}
            columns={columns}
            onSubmit={handleFeatureSpaceSubmit}
            loading={loading}
            featureSpaceData={featureSpaceData}
          />
        )}
      </div>
    </ReactFlowProvider>
  );
}

export default App;
