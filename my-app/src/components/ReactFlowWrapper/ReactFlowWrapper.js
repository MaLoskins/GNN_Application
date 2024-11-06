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

/*
Detailed Explanation:

The `ReactFlowWrapper` component encapsulates the `ReactFlow` component from the `react-flow-renderer` library, providing a streamlined interface for rendering and managing interactive flow diagrams or node-based graphs within the application. Here's an in-depth look at its structure and functionality:

1. **Imports**:
   - **React**: The primary library for building user interfaces.
   - **ReactFlow** from **react-flow-renderer**: A component that facilitates the creation of interactive, customizable flow diagrams, allowing for features like node dragging, zooming, and dynamic edge creation.
   - **ReactFlowWrapper.css**: The CSS stylesheet that styles the wrapper and ensures the flow diagram fits within its container.
   - **'react-flow-renderer/dist/style.css'**: Imports the default styles provided by `react-flow-renderer`, ensuring that the flow diagram has baseline styling and functionality out of the box.

2. **Component Definition**:
   - **Functional Component**: `ReactFlowWrapper` is a functional component that accepts the following props:
     - `nodes`: An array of node objects defining the individual elements within the flow diagram. Each node typically includes properties like `id`, `type`, `position`, and `data`.
     - `edges`: An array of edge objects defining the connections between nodes. Each edge includes properties like `id`, `source`, `target`, and potentially custom data.
     - `onNodesChange`: A callback function that handles updates or changes to the nodes, such as repositioning or editing node data.
     - `onEdgesChange`: A callback function that manages updates or changes to the edges, like adding or removing connections.
     - `onConnect`: A callback function that is invoked when a new connection (edge) is created between nodes, typically by dragging from one node to another.

3. **Rendering**:
   - The component returns a JSX structure comprising:
     - A `div` with the class `wrapper` that serves as the container for the flow diagram. This container is styled via `ReactFlowWrapper.css` to ensure proper sizing, layout, and responsiveness.
     - Inside the `div`, the `ReactFlow` component is rendered with the following configurations:
       - **Data Props**:
         - `nodes`: Passes the array of node objects to define the nodes within the flow.
         - `edges`: Passes the array of edge objects to define the connections between nodes.
       - **Event Handlers**:
         - `onNodesChange`: Assigns the callback for handling changes to nodes.
         - `onEdgesChange`: Assigns the callback for handling changes to edges.
         - `onConnect`: Assigns the callback for handling the creation of new connections between nodes.
       - **Additional Configurations**:
         - `deleteKeyCode={46}`: Specifies that pressing the 'Delete' key (key code 46) will delete selected nodes or edges.
         - `fitView`: Automatically adjusts the view to fit all nodes and edges within the visible area, ensuring that the entire flow is visible upon rendering or resizing.
         - `style={{ width: '100%', height: '100%' }}`: Ensures that the `ReactFlow` component occupies the full width and height of its parent container (`div.wrapper`), allowing for responsive resizing and proper layout.

4. **Export**:
   - The component is exported as the default export, enabling it to be easily imported and used in other parts of the application.

**Purpose in the Application**:
The `ReactFlowWrapper` component is pivotal for visualizing and managing the relationships between different data columns or entities within the application. By leveraging `react-flow-renderer`, it provides users with an intuitive interface to create, modify, and visualize connections in a node-based graph format. Features like node dragging, dynamic edge creation, and responsive resizing enhance the user experience, making it easier to understand and interact with complex data relationships. This component likely interacts closely with the `ConfigurationPanel` and `RelationshipModal` components to reflect user-defined configurations in real-time visualizations.

*/

