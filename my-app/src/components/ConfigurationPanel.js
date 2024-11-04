// my-app/src/components/ConfigurationPanel.js
import React from 'react';

const ConfigurationPanel = ({ columns, onSubmit, loading }) => {
  return (
    <div className="config-section">
      <h2>Configuration</h2>
      <p>
        Drag connections between columns to define relationships. After connecting, specify the relationship type and features.
      </p>
      <button onClick={onSubmit} disabled={loading}>
        {loading ? 'Processing...' : 'Submit Configuration'}
      </button>
    </div>
  );
};

export default ConfigurationPanel;
