// src/hooks/useGraph.js

import { useState, useEffect } from 'react';
import { processData, createFeatureSpace } from '../api';
import { addEdge, useNodesState, useEdgesState } from 'react-flow-renderer';

const useGraph = (csvData, columns) => {
  const [config, setConfig] = useState({ nodes: [], relationships: [], graph_type: 'directed' });
  const [graphData, setGraphData] = useState(null);
  const [featureSpaceData, setFeatureSpaceData] = useState(null);
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

  // Handle node click for editing
  const onNodeClickHandler = (node) => {
    setCurrentNode(node);
    setNodeEditModalIsOpen(true);
  };

  // Handle file drop
  const handleFileDrop = (data, fields) => {
    initializeConfig(fields);
    initializeReactFlow(fields);
  };

  // Initialize config with selectable nodes
  const initializeConfig = (fields) => {
    setConfig({
      nodes: [],
      relationships: [],
      graph_type: 'directed',
    });
  };

  // Initialize React Flow nodes
  const initializeReactFlow = (fields) => {
    setNodes([]);
    setEdges([]);
  };

  // Handle relationship submission
  const handleSaveRelationship = ({ relationshipType, relationshipFeatures }) => {
    if (!relationshipType) {
      alert('Please enter a relationship type.');
      return;
    }

    setConfig((prevConfig) => {
      const newRelationship = {
        source: currentEdge.source,
        target: currentEdge.target,
        type: relationshipType,
        features: relationshipFeatures,
      };

      const edgeWithData = {
        ...currentEdge,
        type: 'smoothstep',
        animated: prevConfig.graph_type === 'directed',
        arrowHeadType: prevConfig.graph_type === 'directed' ? 'arrowclosed' : 'none',
        label: relationshipType,
        data: { type: relationshipType, features: relationshipFeatures },
      };

      const updatedConfig = {
        ...prevConfig,
        relationships: [...prevConfig.relationships, newRelationship],
      };

      setEdges((eds) => addEdge(edgeWithData, eds));

      return updatedConfig;
    });

    setRelationshipModalIsOpen(false);
    setCurrentEdge(null);
  };

  // Handle edge creation
  const onConnectHandler = (params) => {
    setCurrentEdge(params);
    setRelationshipModalIsOpen(true);
  };

  // Handle node type and feature updates
  const handleSaveNodeEdit = ({ nodeType, nodeFeatures }) => {
    if (!nodeType) {
      alert('Please enter a node type.');
      return;
    }

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

    setNodeEditModalIsOpen(false);
    setCurrentNode(null);
  };

  // Handle node selection for inclusion as a node in the graph
  const handleSelectNode = (column) => {
    const isSelected = config.nodes.find((n) => n.id === column);

    if (isSelected) {
      setConfig((prevConfig) => ({
        ...prevConfig,
        nodes: prevConfig.nodes.filter((n) => n.id !== column),
      }));

      setNodes((nds) => nds.filter((n) => n.id !== column));

      setEdges((eds) => eds.filter((e) => e.source !== column && e.target !== column));
    } else {
      setConfig((prevConfig) => ({
        ...prevConfig,
        nodes: [...prevConfig.nodes, { id: column, type: 'default', features: [] }],
      }));

      const newNode = {
        id: column,
        type: 'default',
        data: { label: column },
        position: {
          x: Math.random() * 400 - 200,
          y: Math.random() * 400 - 200,
        },
        className: 'custom-node-style',
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
      if (!node.type || node.type === 'default') {
        alert(`Please specify a valid type for node '${node.id}'.`);
        return;
      }
    }
    for (let rel of config.relationships) {
      if (!rel.source || !rel.target) {
        alert('Please select source and target columns for all relationships.');
        return;
      }
      if (!rel.type || rel.type === 'default') {
        alert('Please specify a valid type for all relationships.');
        return;
      }
    }

    setLoading(true);
    try {
      const response = await processData(csvData, config);
      const receivedGraph = response.graph;

      const degrees = {};
      receivedGraph.links.forEach((link) => {
        if (receivedGraph.directed) {
          degrees[link.source] = (degrees[link.source] || 0) + 1;
          degrees[link.target] = (degrees[link.target] || 0) + 1;
        } else {
          degrees[link.source] = (degrees[link.source] || 0) + 1;
          degrees[link.target] = (degrees[link.target] || 0) + 1;
        }
      });

      const nodesWithSize = receivedGraph.nodes.map((node) => ({
        ...node,
        val: degrees[node.id] || 1,
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

      const { features, multi_graph_settings, feature_space } = response;

      const featureSpaceObj = JSON.parse(feature_space);

      featureSpaceObj.features = features;
      featureSpaceObj.multi_graph_settings = multi_graph_settings;

      setFeatureSpaceData(featureSpaceObj);
      alert('Feature space created successfully.');
    } catch (error) {
      console.error('Error creating feature space:', error);
      alert('Error creating feature space. Check console for details.');
    }
    setLoading(false);
  };

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

  // Return all necessary state and functions
  return {
    config,
    graphData,
    loading,
    nodes,
    edges,
    relationshipModalIsOpen,
    nodeEditModalIsOpen,
    currentEdge,
    currentNode,
    featureSpaceData,
    handleFileDrop,
    handleSelectNode,
    handleSubmit,
    handleFeatureSpaceSubmit,
    onNodesChange,
    onEdgesChange,
    onConnectHandler,
    onNodeClickHandler,
    handleSaveRelationship,
    handleSaveNodeEdit,
    setRelationshipModalIsOpen,
    setNodeEditModalIsOpen,
  };
};

export default useGraph;
