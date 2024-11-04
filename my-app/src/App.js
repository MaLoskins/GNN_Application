// my-app/src/App.js
import React, { useState, useEffect } from 'react';
import { processData } from './api';
import Papa from 'papaparse';
import { useDropzone } from 'react-dropzone';
import ForceGraph2D from 'react-force-graph-2d';
import ReactFlow, {
  ReactFlowProvider,
  addEdge,
  useNodesState,
  useEdgesState,
} from 'react-flow-renderer';
import Modal from 'react-modal';
import './App.css';

Modal.setAppElement('#root'); // For accessibility

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
  const [relationshipType, setRelationshipType] = useState('');
  const [relationshipFeatures, setRelationshipFeatures] = useState([]);

  // Handle file drop
  const onDrop = (acceptedFiles) => {
    if (acceptedFiles.length === 0) return;
    const file = acceptedFiles[0];
    Papa.parse(file, {
      header: true,
      dynamicTyping: true,
      complete: (results) => {
        setCsvData(results.data);
        setColumns(results.meta.fields);
        initializeConfig(results.meta.fields);
        initializeReactFlow(results.meta.fields);
      },
      error: (error) => {
        console.error('Error parsing CSV:', error);
      },
    });
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: '.csv',
  });

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
  const handleRelationshipSubmit = (e) => {
    e.preventDefault();
    if (!relationshipType) {
      alert('Please enter a relationship type.');
      return;
    }

    // Update config.relationships
    const newRelationship = {
      source: currentEdge.source,
      target: currentEdge.target,
      type: relationshipType,
      features: relationshipFeatures,
    };

    setConfig((prevConfig) => ({
      ...prevConfig,
      relationships: [...prevConfig.relationships, newRelationship],
    }));

    // Add the edge with data
    const edgeWithData = {
      ...currentEdge,
      type: 'smoothstep', // You can choose any edge type you prefer
      animated: config.graph_type === 'directed',
      arrowHeadType: config.graph_type === 'directed' ? 'arrowclosed' : 'none',
      label: relationshipType,
      data: { type: relationshipType, features: relationshipFeatures },
    };

    setEdges((eds) => addEdge(edgeWithData, eds));

    // Reset modal state
    setModalIsOpen(false);
    setCurrentEdge(null);
    setRelationshipType('');
    setRelationshipFeatures([]);
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
        <div {...getRootProps()} className="dropzone">
          <input {...getInputProps()} />
          {isDragActive ? (
            <p>Drop the CSV file here...</p>
          ) : (
            <p>Drag and drop a CSV file here, or click to select file</p>
          )}
        </div>

        {/* Configuration Section */}
        {columns.length > 0 && (
          <div className="config-section">
            <h2>Configuration</h2>

            {/* Instructions */}
            <p>
              Drag connections between columns to define relationships. After connecting, specify the relationship type and features.
            </p>

            {/* Submit Button */}
            <button onClick={handleSubmit} disabled={loading}>
              {loading ? 'Processing...' : 'Submit Configuration'}
            </button>
          </div>
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
          <div className="graph-section">
            <h2>Graph Visualization</h2>
            <div className="graph-container">
              <ForceGraph2D
                graphData={graphData}
                nodeAutoColorBy="type"
                linkAutoColorBy="type"
                nodeLabel="id"
                linkLabel="type"
                nodeVal={(node) => node.val}
                linkDirectionalArrowLength={graphData.directed ? 6 : 0}
                linkDirectionalArrowRelPos={0.5}
                width={dimensions.width}
                height={dimensions.height}
              />
            </div>
          </div>
        )}

        {/* Relationship Modal */}
        <Modal
          isOpen={modalIsOpen}
          onRequestClose={() => setModalIsOpen(false)}
          contentLabel="Relationship Configuration"
          className="modal"
          overlayClassName="overlay"
        >
          <h2>Configure Relationship</h2>
          <form onSubmit={handleRelationshipSubmit}>
            <label>
              Relationship Type:
              <input
                type="text"
                value={relationshipType}
                onChange={(e) => setRelationshipType(e.target.value)}
                required
              />
            </label>
            <label>
              Features:
              <select
                multiple
                value={relationshipFeatures}
                onChange={(e) =>
                  setRelationshipFeatures(
                    Array.from(e.target.selectedOptions).map((option) => option.value)
                  )
                }
              >
                {columns.map((col) => (
                  <option key={col} value={col}>
                    {col}
                  </option>
                ))}
              </select>
            </label>
            <div className="modal-buttons">
              <button type="submit">Save</button>
              <button type="button" onClick={() => setModalIsOpen(false)}>
                Cancel
              </button>
            </div>
          </form>
        </Modal>
      </div>
    </ReactFlowProvider>
  );
}

export default App;