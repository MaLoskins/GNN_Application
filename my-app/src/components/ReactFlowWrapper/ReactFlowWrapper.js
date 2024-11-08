// src/components/ReactFlowWrapper/ReactFlowWrapper.js

import React from 'react';
import ReactFlow, { Background, Controls } from 'react-flow-renderer';
import './ReactFlowWrapper.css';

const ReactFlowWrapper = ({
  nodes,
  edges,
  onNodesChange,
  onEdgesChange,
  onConnect,
  onNodeClick,
}) => {
  const handleNodeClick = (event, node) => {
    if (onNodeClick) {
      onNodeClick(node);
    }
  };

  return (
    <div className="wrapper">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onNodeClick={handleNodeClick}
        deleteKeyCode={46}
        fitView
        style={{ width: '100%', height: '100%' }}
      >
        <Background color="#aaa" gap={16} />
        <Controls />
      </ReactFlow>
    </div>
  );
};

export default ReactFlowWrapper;
