// src/components/GraphVisualizer/GraphVisualizer.js

import React from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import './GraphVisualizer.css';

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
