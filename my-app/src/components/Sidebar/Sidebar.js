// src/components/Sidebar/Sidebar.js

import React, { useState } from 'react';
import './Sidebar.css';
import { downloadGraph } from '../../api';
import {
  FiChevronDown,
  FiChevronUp,
  FiDownload,
  FiX,
  FiMenu,
  FiFileText,
  FiSettings,
  FiDatabase,
  FiLayers,
} from 'react-icons/fi';

const Sidebar = ({ csvData, columns, config, graphData, featureSpaceData }) => {
  // State for sidebar visibility
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);

  // State for collapsible sections
  const [csvSectionOpen, setCsvSectionOpen] = useState(true);
  const [configSectionOpen, setConfigSectionOpen] = useState(true);
  const [graphDataSectionOpen, setGraphDataSectionOpen] = useState(true);
  const [featureSpaceSectionOpen, setFeatureSpaceSectionOpen] = useState(true);

  // State for selected download format
  const [downloadFormat, setDownloadFormat] = useState('graphml');

  // Helper functions
  const calculateTotalEmbeddingDimensions = () => {
    if (
      !featureSpaceData ||
      !featureSpaceData.multi_graph_settings ||
      !featureSpaceData.multi_graph_settings.embedding_shapes
    )
      return 0;
    const shapes = featureSpaceData.multi_graph_settings.embedding_shapes;
    return Object.values(shapes).reduce((acc, dim) => acc + dim, 0);
  };

  const countFeatureTypes = () => {
    if (!featureSpaceData || !featureSpaceData.features) return { text: 0, numeric: 0 };
    const features = featureSpaceData.features;
    let text = 0;
    let numeric = 0;
    features.forEach((feature) => {
      if (feature.type === 'text') text += 1;
      if (feature.type === 'numeric') numeric += 1;
    });
    return { text, numeric };
  };

  // Handle download graph
  const handleDownloadGraph = async () => {
    try {
      const response = await downloadGraph(csvData, config, featureSpaceData, downloadFormat);
      const blob = new Blob([response.data], { type: 'application/xml' });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `graph.${downloadFormat}`);
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    } catch (error) {
      console.error('Error downloading graph:', error);
      alert('Failed to download graph.');
    }
  };

  return (
    <>
      <div className={`sidebar ${isSidebarOpen ? 'open' : 'closed'}`}>
        <div className="sidebar-header">
          <h2>Data Overview</h2>
          <button
            className="sidebar-toggle"
            onClick={() => setIsSidebarOpen(!isSidebarOpen)}
            aria-label="Close Sidebar"
          >
            <FiX />
          </button>
        </div>

        {/* CSV Data Information */}
        <div className="sidebar-section">
          <h3 onClick={() => setCsvSectionOpen(!csvSectionOpen)}>
            <FiFileText className="section-icon" />
            CSV Data
            {csvSectionOpen ? <FiChevronUp /> : <FiChevronDown />}
          </h3>
          {csvSectionOpen && (
            <div className="section-content">
              {csvData && csvData.length > 0 ? (
                <>
                  <p>
                    <strong>Rows:</strong> {csvData.length}
                  </p>
                  <p>
                    <strong>Columns:</strong> {columns.length}
                  </p>
                  {/* Sample Data */}
                  <div className="sample-data">
                    <p>
                      <strong>Sample Data:</strong>
                    </p>
                    <table className="sample-table">
                      <thead>
                        <tr>
                          {columns.slice(0, 5).map((col) => (
                            <th key={col}>{col}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {csvData.slice(0, 3).map((row, idx) => (
                          <tr key={idx}>
                            {columns.slice(0, 5).map((col) => (
                              <td key={col}>{row[col]}</td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </>
              ) : (
                <p>No data uploaded.</p>
              )}
            </div>
          )}
        </div>

        {/* Graph Configuration Information */}
        <div className="sidebar-section">
          <h3 onClick={() => setConfigSectionOpen(!configSectionOpen)}>
            <FiSettings className="section-icon" />
            Graph Configuration
            {configSectionOpen ? <FiChevronUp /> : <FiChevronDown />}
          </h3>
          {configSectionOpen && (
            <div className="section-content">
              {config && config.nodes && config.nodes.length > 0 ? (
                <>
                  <p>
                    <strong>Nodes Configured:</strong> {config.nodes.length}
                  </p>
                  <ul className="config-list">
                    {config.nodes.map((node) => (
                      <li key={node.id}>
                        <button
                          className="config-item"
                          onClick={() =>
                            alert(
                              `Node ID: ${node.id}\nType: ${node.type || 'default'}\nFeatures: ${
                                node.features.length > 0 ? node.features.join(', ') : 'None'
                              }`
                            )
                          }
                        >
                          {node.id} ({node.type || 'default'})
                        </button>
                      </li>
                    ))}
                  </ul>
                </>
              ) : (
                <p>No nodes configured.</p>
              )}
              {config && config.relationships && config.relationships.length > 0 ? (
                <>
                  <p>
                    <strong>Relationships Configured:</strong> {config.relationships.length}
                  </p>
                  <ul className="config-list">
                    {config.relationships.map((rel, index) => (
                      <li key={index}>
                        <button
                          className="config-item"
                          onClick={() =>
                            alert(
                              `Source: ${rel.source}\nTarget: ${rel.target}\nType: ${rel.type}\nFeatures: ${
                                rel.features.length > 0 ? rel.features.join(', ') : 'None'
                              }`
                            )
                          }
                        >
                          {rel.source} - [{rel.type}] -&gt; {rel.target}
                        </button>
                      </li>
                    ))}
                  </ul>
                </>
              ) : (
                <p>No relationships configured.</p>
              )}
            </div>
          )}
        </div>

        {/* Graph Data Information */}
        <div className="sidebar-section">
          <h3 onClick={() => setGraphDataSectionOpen(!graphDataSectionOpen)}>
            <FiDatabase className="section-icon" />
            Graph Data
            {graphDataSectionOpen ? <FiChevronUp /> : <FiChevronDown />}
          </h3>
          {graphDataSectionOpen && (
            <div className="section-content">
              {graphData ? (
                <>
                  <p>
                    <strong>Total Nodes:</strong> {graphData.nodes.length}
                  </p>
                  <p>
                    <strong>Total Links:</strong> {graphData.links.length}
                  </p>
                  {/* Download Graph */}
                  <div className="download-section">
                    <label htmlFor="download-format">Download Format:</label>
                    <select
                      id="download-format"
                      value={downloadFormat}
                      onChange={(e) => setDownloadFormat(e.target.value)}
                    >
                      <option value="graphml">GraphML</option>
                      <option value="gexf">GEXF</option>
                      <option value="gml">GML</option>
                    </select>
                    <button className="download-button" onClick={handleDownloadGraph}>
                      <FiDownload className="button-icon" />
                      Download Graph
                    </button>
                  </div>
                </>
              ) : (
                <p>No graph data available.</p>
              )}
            </div>
          )}
        </div>

        {/* Feature Space Information */}
        <div className="sidebar-section">
          <h3 onClick={() => setFeatureSpaceSectionOpen(!featureSpaceSectionOpen)}>
            <FiLayers className="section-icon" />
            Feature Space
            {featureSpaceSectionOpen ? <FiChevronUp /> : <FiChevronDown />}
          </h3>
          {featureSpaceSectionOpen && (
            <div className="section-content">
              {featureSpaceData ? (
                <>
                  <p>
                    <strong>Total Features:</strong> {featureSpaceData.features.length}
                  </p>
                  <p>
                    <strong>Total Embedding Dimensions:</strong>{' '}
                    {calculateTotalEmbeddingDimensions()}
                  </p>
                  <p>
                    <strong>Text Features:</strong> {countFeatureTypes().text}
                  </p>
                  <p>
                    <strong>Numeric Features:</strong> {countFeatureTypes().numeric}
                  </p>
                  {/* Feature Details */}
                  <ul className="feature-list">
                    {featureSpaceData.features.map((feature, idx) => (
                      <li key={idx}>
                        <button
                          className="config-item"
                          onClick={() =>
                            alert(
                              `Feature: ${feature.column_name}\nType: ${feature.type}\nEmbedding Dimension: ${
                                feature.embedding_dim || 'N/A'
                              }`
                            )
                          }
                        >
                          {feature.column_name} ({feature.type})
                        </button>
                      </li>
                    ))}
                  </ul>
                </>
              ) : (
                <p>No feature space created.</p>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Sidebar Toggle Button */}
      {!isSidebarOpen && (
        <button
          className="sidebar-open-button"
          onClick={() => setIsSidebarOpen(true)}
          aria-label="Open Sidebar"
        >
          <FiMenu />
        </button>
      )}
    </>
  );
};

export default Sidebar;
