// src/components/ReactFlowWrapper/ReactFlowWrapper.js
import React from 'react';
import ReactFlow, { Background, Controls } from 'react-flow-renderer';
import './ReactFlowWrapper.css'; 
import 'react-flow-renderer/dist/style.css'; // Import React Flow's default styles

const ReactFlowWrapper = ({ nodes, edges, onNodesChange, onEdgesChange, onConnect, onNodeClick }) => {
  const handleNodeClick = (event, node) => {
    onNodeClick(node);
  };

  return (
    <div className="wrapper">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onNodeClick={handleNodeClick} // Handle node clicks
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

/*
Detailed Explanation:

1. **Props Received:**
   - `nodes`: Array of node objects.
   - `edges`: Array of edge objects.
   - `onNodesChange`: Handler for node state changes.
   - `onEdgesChange`: Handler for edge state changes.
   - `onConnect`: Handler for creating new connections (edges).
   - `onNodeClick`: Handler for node click events to open the edit modal.

2. **Handling Node Clicks:**
   - **`handleNodeClick`**: Invoked when a node is clicked. It calls the `onNodeClick` prop with the clicked node as an argument.

3. **ReactFlow Components:**
   - **`Background`**: Adds a grid background to the flow diagram.
   - **`Controls`**: Provides zoom and pan controls.

4. **Styling:**
   - Ensure that `ReactFlowWrapper.css` styles the wrapper appropriately. Here's an example:


*/