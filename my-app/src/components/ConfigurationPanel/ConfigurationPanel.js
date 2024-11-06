// src/components/ConfigurationPanel/ConfigurationPanel.js

import React from 'react';
import './ConfigurationPanel.css'; // Ensure correct path

const ConfigurationPanel = ({ columns, onSelectNode, onSubmit, loading, selectedNodes }) => {
  return (
    <div className="config-section">
      <h2>Select Nodes and Relationships</h2>
      <div className="node-selection">
        {columns.map((col) => (
          <div key={col} className="node-selector">
            <input
              type="checkbox"
              id={`node-${col}`}
              value={col}
              checked={selectedNodes.includes(col)}
              onChange={() => onSelectNode(col)}
            />
            <label htmlFor={`node-${col}`}>{col}</label>
          </div>
        ))}
      </div>
      <button onClick={onSubmit} disabled={loading}>
        {loading ? 'Processing...' : 'Process Graph'}
      </button>
    </div>
  );
};

export default ConfigurationPanel;
