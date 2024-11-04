// src/utils/layout.js

import dagre from 'dagre';

/**
 * Arranges nodes using dagre for a circular layout.
 *
 * @param {Array} nodes - Array of node objects.
 * @param {Array} edges - Array of edge objects.
 * @returns {Object} - Object containing arranged nodes and edges.
 */
export const arrangeNodesWithDagre = (nodes, edges) => {
  const dagreGraph = new dagre.graphlib.Graph();
  dagreGraph.setDefaultEdgeLabel(() => ({}));
  dagreGraph.setGraph({ rankdir: 'LR', nodesep: 50, ranksep: 50 });

  nodes.forEach((node) => {
    dagreGraph.setNode(node.id, { width: 100, height: 50 });
  });

  edges.forEach((edge) => {
    dagreGraph.setEdge(edge.source, edge.target);
  });

  dagre.layout(dagreGraph);

  const arrangedNodes = nodes.map((node) => {
    const nodeWithPosition = dagreGraph.node(node.id);
    return {
      ...node,
      position: {
        x: nodeWithPosition.x - 50, // Adjust for node width
        y: nodeWithPosition.y - 25, // Adjust for node height
      },
    };
  });

  return { arrangedNodes, arrangedEdges: edges };
};
