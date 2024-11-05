// src/components/ReactFlowWrapper/ReactFlowWrapper.js
import React from 'react';
import ReactFlow from 'react-flow-renderer';
import './ReactFlowWrapper.css'; // Correct Import
import 'react-flow-renderer/dist/style.css'; // Import React Flow's default styles

const ReactFlowWrapper = ({ nodes, edges, onNodesChange, onEdgesChange, onConnect }) => {
  return (
    <div className={'wrapper'}>
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
  );
};

export default ReactFlowWrapper;
