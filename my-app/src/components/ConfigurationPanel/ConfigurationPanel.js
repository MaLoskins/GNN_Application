// src/components/ConfigurationPanel/ConfigurationPanel.js
import React from 'react';
import './ConfigurationPanel.css'; // Updated path

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
