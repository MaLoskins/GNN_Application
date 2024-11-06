// src/components/GraphVisualizer/GraphVisualizer.js
import React from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import './GraphVisualizer.css'; // Updated path

const GraphVisualizer = ({ graphData, dimensions }) => {
  return (
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
          cooldownTicks={300}
          enableNodeDrag={true}
          enableZoomPanInteraction={true}
          backgroundColor={null}
        />
      </div>
    </div>
  );
};

export default GraphVisualizer;

/*
Detailed Explanation:

The `GraphVisualizer` component is responsible for rendering a 2D force-directed graph based on the provided graph data. It leverages the `react-force-graph-2d` library to create interactive and dynamic graph visualizations. Here's a comprehensive breakdown of its structure and functionality:

1. **Imports**:
   - **React**: The core library for building user interfaces.
   - **ForceGraph2D** from **react-force-graph-2d**: A component for rendering 2D force-directed graphs, allowing for rich interactivity and customization.
   - **GraphVisualizer.css**: The CSS stylesheet that styles the graph visualization section, ensuring consistency with the application's design.

2. **Component Definition**:
   - **Functional Component**: `GraphVisualizer` is a functional component that accepts two props:
     - `graphData`: An object representing the graph structure, typically containing `nodes` and `links` arrays. Each node and link can have various attributes that define their properties and relationships.
     - `dimensions`: An object specifying the `width` and `height` for the graph canvas, allowing the visualization to be responsive and fit within its container.

3. **Rendering**:
   - The component returns a JSX structure comprising:
     - A `div` with the class `graph-section` that wraps the entire visualization area.
     - An `h2` header titled "Graph Visualization" to denote the purpose of the section.
     - A nested `div` with the class `graph-container` that houses the `ForceGraph2D` component, applying specific styles for layout and responsiveness.
     - **ForceGraph2D Configuration**:
       - `graphData`: Passes the graph data to the visualization component.
       - `nodeAutoColorBy="type"`: Automatically assigns colors to nodes based on their `type` attribute, enhancing visual differentiation.
       - `linkAutoColorBy="type"`: Similarly, assigns colors to links based on their `type` attribute.
       - `nodeLabel="id"`: Sets the label for each node to display its `id` attribute when hovered over.
       - `linkLabel="type"`: Sets the label for each link to display its `type` attribute when hovered over.
       - `nodeVal={(node) => node.val}`: Determines the size of each node based on its `val` attribute, allowing for size differentiation based on node importance or degree.
       - `linkDirectionalArrowLength={graphData.directed ? 6 : 0}`: Conditionally renders directional arrows on links if the graph is directed.
       - `linkDirectionalArrowRelPos={0.5}`: Positions the directional arrow at the midpoint of the link.
       - `width` and `height`: Sets the dimensions of the graph canvas based on the `dimensions` prop, ensuring the graph fits its container.
       - `cooldownTicks={300}`: Controls the simulation's cooldown ticks, affecting how long the simulation runs to stabilize node positions.
       - `enableNodeDrag={true}`: Allows users to manually drag nodes, providing interactive control over the graph layout.
       - `enableZoomPanInteraction={true}`: Enables zooming and panning within the graph, facilitating exploration of large or complex graphs.
       - `backgroundColor={null}`: Sets the background color of the graph canvas. `null` typically defaults to transparent or the parent container's background.

4. **Export**:
   - The component is exported as the default export, making it accessible for import and use in other parts of the application.

**Purpose in the Application**:
The `GraphVisualizer` is a key component for data visualization within the application. After users upload and configure their CSV data, this component takes the processed graph data and renders it interactively. Features like automatic coloring, dynamic sizing, and interactivity (dragging, zooming, panning) enhance the user's ability to analyze and understand complex relationships within the data. By leveraging `react-force-graph-2d`, the component ensures efficient rendering and smooth interactions, even with large datasets.

*/

