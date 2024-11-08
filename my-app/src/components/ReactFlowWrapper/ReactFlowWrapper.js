// src/components/ReactFlowWrapper/ReactFlowWrapper.js

import React from 'react';
import ReactFlow, { Background, Controls } from 'react-flow-renderer';
import './ReactFlowWrapper.css'; 
import 'react-flow-renderer/dist/style.css'; // Import React Flow's default styles

const ReactFlowWrapper = ({ nodes, edges, onNodesChange, onEdgesChange, onConnect, onNodeClick }) => {
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
        deleteKeyCode={46} /* 'delete'-key */
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
