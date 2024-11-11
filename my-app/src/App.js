// src/App.js

import React, { useState, useEffect } from 'react';
import { ReactFlowProvider } from 'react-flow-renderer';
import FileUploader from './components/FileUploader/FileUploader';
import ConfigurationPanel from './components/ConfigurationPanel/ConfigurationPanel';
import GraphVisualizer from './components/GraphVisualizer/GraphVisualizer';
import RelationshipModal from './components/RelationshipModal/RelationshipModal';
import NodeEditModal from './components/NodeEditModal/NodeEditModal';
import FeatureSpaceCreatorTab from './components/FeatureSpaceCreatorTab/FeatureSpaceCreatorTab';
import ReactFlowWrapper from './components/ReactFlowWrapper/ReactFlowWrapper';
import Sidebar from './components/Sidebar/Sidebar'; // Import the Sidebar component
import useGraph from './hooks/useGraph';
import './App.css';

function App() {
  const [csvData, setCsvData] = useState([]);
  const [columns, setColumns] = useState([]);
  const [activeTab, setActiveTab] = useState('graph');

  const {
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
  } = useGraph(csvData, columns);

  const [dimensions, setDimensions] = useState({
    width: window.innerWidth * 0.9,
    height: window.innerWidth * 0.9 * (2 / 3),
  });

  useEffect(() => {
    const handleResize = () => {
      setDimensions({
        width: window.innerWidth * 0.9,
        height: window.innerWidth * 0.9 * (2 / 3),
      });
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  return (
    <ReactFlowProvider>
      <div className="App">
        {/* Sidebar */}
        <Sidebar
          csvData={csvData}
          columns={columns}
          config={config}
          graphData={graphData}
          featureSpaceData={featureSpaceData}
        />

        <div className="main-content">
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
              <FileUploader
                onFileDrop={(data, fields) => {
                  setCsvData(data);
                  setColumns(fields);
                  handleFileDrop(data, fields);
                }}
              />

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
                  onNodeClick={onNodeClickHandler}
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
                featureSpaceData={featureSpaceData}
              />

              {/* Node Edit Modal */}
              {currentNode && (
                <NodeEditModal
                  isOpen={nodeEditModalIsOpen}
                  onRequestClose={() => setNodeEditModalIsOpen(false)}
                  node={currentNode}
                  onSaveNodeEdit={handleSaveNodeEdit}
                  featureSpaceData={featureSpaceData}
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
      </div>
    </ReactFlowProvider>
  );
}

export default App;