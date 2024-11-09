// src/hooks/useGraph.js

import { useState } from 'react';
import { processData, createFeatureSpace } from '../api';
import { addEdge, useNodesState, useEdgesState } from 'react-flow-renderer';

const useGraph = (csvData, columns) => {
  const [config, setConfig] = useState({ nodes: [], relationships: [], graph_type: 'directed' });
  const [graphData, setGraphData] = useState(null);
  const [featureSpaceData, setFeatureSpaceData] = useState(null);
  const [loading, setLoading] = useState(false);

  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);

  const [relationshipModalIsOpen, setRelationshipModalIsOpen] = useState(false);
  const [currentEdge, setCurrentEdge] = useState(null);

  const [nodeEditModalIsOpen, setNodeEditModalIsOpen] = useState(false);
  const [currentNode, setCurrentNode] = useState(null);

  const onNodeClickHandler = (event, node) => {
    setCurrentNode(node);
    setNodeEditModalIsOpen(true);
  };

  const handleFileDrop = (data, fields) => {
    initializeConfig(fields);
    initializeReactFlow(fields);
    setFeatureSpaceData(null);
    setGraphData(null);
  };

  const initializeConfig = (fields) => {
    setConfig({
      nodes: [],
      relationships: [],
      graph_type: 'directed',
    });
  };

  const initializeReactFlow = (fields) => {
    setNodes([]);
    setEdges([]);
  };

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

  const onConnectHandler = (params) => {
    setCurrentEdge(params);
    setRelationshipModalIsOpen(true);
  };

  const handleSaveNodeEdit = ({ nodeType, nodeFeatures }) => {
    nodeType = nodeType || 'default';

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
            data: { ...node.data, label: `${node.data.label.split(' ')[0]} (${nodeType})` },
            features: nodeFeatures,
          };
        }
        return node;
      })
    );

    setNodeEditModalIsOpen(false);
    setCurrentNode(null);
  };

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
      const newNodeConfig = { id: column, type: 'default', features: [] };

      setConfig((prevConfig) => ({
        ...prevConfig,
        nodes: [...prevConfig.nodes, newNodeConfig],
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
        features: [],
      };

      setNodes((nds) => nds.concat(newNode));
    }
  };

  const handleSubmit = async () => {
    for (let node of config.nodes) {
      if (!node.id) {
        alert('Please select an ID column for all nodes.');
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
      const response = await processData(
        csvData,
        config,
        featureSpaceData ? JSON.stringify(featureSpaceData) : null
      );
      const receivedGraph = response.graph;

      const degrees = {};
      receivedGraph.links.forEach((link) => {
        degrees[link.source] = (degrees[link.source] || 0) + 1;
        degrees[link.target] = (degrees[link.target] || 0) + 1;
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
