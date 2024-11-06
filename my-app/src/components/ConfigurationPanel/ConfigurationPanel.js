// src/components/ConfigurationPanel/ConfigurationPanel.js
import React from 'react';
import './ConfigurationPanel.css'; // Updated path

const ConfigurationPanel = ({ columns, onSelectNode, onSubmit, loading, selectedNodes }) => {
  return (
    <div className="config-section">
      <h2>Configuration</h2>
      <p>
        Select the columns you want to include as nodes in the graph. Only selected columns will require a node type and features.
      </p>

      <div className="node-selection">
        {columns.map((column) => (
          <div key={column} className="node-selector">
            <input
              type="checkbox"
              id={`node-select-${column}`}
              checked={selectedNodes.includes(column)}
              onChange={() => onSelectNode(column)}
            />
            <label htmlFor={`node-select-${column}`}>{column}</label>
          </div>
        ))}
      </div>

      <button onClick={onSubmit} disabled={loading}>
        {loading ? 'Processing...' : 'Submit Configuration'}
      </button>
    </div>
  );
};

export default ConfigurationPanel;

/*
Detailed Explanation:

1. **Props Received:**
   - `columns`: An array of column names from the uploaded CSV.
   - `onSelectNode`: Function to handle the selection/deselection of a column as a node.
   - `onSubmit`: Function to handle the submission of the configuration.
   - `loading`: Boolean indicating whether a submission is in progress.
   - `selectedNodes`: Array of currently selected node columns.

2. **UI Elements:**
   - **Checkboxes for Node Selection:**
     - Each column is presented with a checkbox. Checking it includes the column as a node; unchecking it excludes it.
   
   - **Submit Button:**
     - Allows users to submit the selected configuration. Disabled when `loading` is `true`.

3. **Styling:**
   - Ensure that `ConfigurationPanel.css` styles the node selection UI appropriately.

*/

