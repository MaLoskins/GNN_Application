// src/components/FeatureSpaceCreatorTab/FeatureSpaceCreatorTab.js

import React, { useState } from 'react';
import './FeatureSpaceCreatorTab.css';

function FeatureSpaceCreatorTab({ csvData, columns, onSubmit, loading, featureSpaceData }) {
  const [features, setFeatures] = useState([
    {
      column_name: '',
      type: 'text', // 'text' or 'numeric'
      // Text feature options
      embedding_method: 'bert', // 'bert', 'glove', 'word2vec'
      embedding_dim: 768,
      dim_reduction_method: 'none', // 'none', 'pca', 'umap'
      dim_reduction_target_dim: 100,
      // Numeric feature options
      data_type: 'float', // 'int' or 'float'
      processing: 'none', // 'none', 'standardize', 'normalize'
      projection_method: 'none', // 'none', 'linear'
      projection_target_dim: 1,
    },
  ]);

  const handleFeatureChange = (index, field, value) => {
    const updatedFeatures = [...features];
    updatedFeatures[index][field] = value;
    setFeatures(updatedFeatures);
  };

  const handleAddFeature = () => {
    setFeatures([
      ...features,
      {
        column_name: '',
        type: 'text',
        embedding_method: 'bert',
        embedding_dim: 768,
        dim_reduction_method: 'none',
        dim_reduction_target_dim: 100,
        data_type: 'float',
        processing: 'none',
        projection_method: 'none',
        projection_target_dim: 1,
      },
    ]);
  };

  const handleRemoveFeature = (index) => {
    const updatedFeatures = [...features];
    updatedFeatures.splice(index, 1);
    setFeatures(updatedFeatures);
  };

  const handleSubmit = (e) => {
    e.preventDefault();

    // Validation: Ensure all required fields are filled
    for (let i = 0; i < features.length; i++) {
      const feature = features[i];
      if (!feature.column_name) {
        alert(`Feature ${i + 1}: Please select a column.`);
        return;
      }

      if (feature.type === 'text') {
        if (!feature.embedding_method) {
          alert(`Feature ${i + 1}: Please select an embedding method.`);
          return;
        }
        if (feature.dim_reduction_method !== 'none' && !feature.dim_reduction_target_dim) {
          alert(`Feature ${i + 1}: Please specify a target dimension for dimensionality reduction.`);
          return;
        }
      }

      if (feature.type === 'numeric') {
        if (!feature.data_type) {
          alert(`Feature ${i + 1}: Please select a data type.`);
          return;
        }
        if (feature.processing !== 'none' && !feature.processing) {
          alert(`Feature ${i + 1}: Please select a processing method.`);
          return;
        }
        if (feature.projection_method !== 'none' && !feature.projection_target_dim) {
          alert(`Feature ${i + 1}: Please specify a target dimension for projection.`);
          return;
        }
      }
    }

    // Prepare configuration object
    const config = {
      features: features.map((feature) => {
        const { column_name, type } = feature;
        const featureConfig = {
          column_name,
          type,
        };
        if (type === 'text') {
          featureConfig.embedding_method = feature.embedding_method;
          featureConfig.embedding_dim = parseInt(feature.embedding_dim, 10);
          if (feature.dim_reduction_method !== 'none') {
            featureConfig.dim_reduction = {
              method: feature.dim_reduction_method,
              target_dim: parseInt(feature.dim_reduction_target_dim, 10),
            };
          }
        }
        if (type === 'numeric') {
          featureConfig.data_type = feature.data_type;
          featureConfig.processing = feature.processing;
          if (feature.projection_method !== 'none') {
            featureConfig.projection = {
              method: feature.projection_method,
              target_dim: parseInt(feature.projection_target_dim, 10),
            };
          }
        }
        return featureConfig;
      }),
      multi_graph_settings: {
        embedding_shapes: features.reduce((acc, feature) => {
          if (feature.type === 'text') {
            acc[feature.column_name] = feature.dim_reduction_method !== 'none'
              ? parseInt(feature.dim_reduction_target_dim, 10)
              : parseInt(feature.embedding_dim, 10);
          }
          if (feature.type === 'numeric') {
            acc[feature.column_name] = feature.projection_method !== 'none'
              ? parseInt(feature.projection_target_dim, 10)
              : 1;
          }
          return acc;
        }, {})
      }
    };

    // Submit the configuration and data to the backend
    onSubmit(config);
  };

  // Helper function to calculate total embedding dimensions
  const calculateTotalDimensions = () => {
    if (!featureSpaceData || !featureSpaceData.multi_graph_settings || !featureSpaceData.multi_graph_settings.embedding_shapes) return 0;
    const shapes = featureSpaceData.multi_graph_settings.embedding_shapes;
    return Object.values(shapes).reduce((acc, dim) => acc + dim, 0);
  };

  // Helper function to count feature types
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

  // Define color schemes for different types and methods
  const typeColors = {
    text: '#1890ff',
    numeric: '#52c41a',
  };

  const methodColors = {
    bert: '#faad14',
    glove: '#eb2f96',
    word2vec: '#13c2c2',
  };

  return (
    <div className="feature-space-creator-tab">
      <h2>Feature Space Creator</h2>
      <form onSubmit={handleSubmit}>
        {features.map((feature, index) => (
          <div key={index} className="feature-config">
            <h3>Feature {index + 1}</h3>
            {features.length > 1 && (
              <button 
                type="button" 
                onClick={() => handleRemoveFeature(index)} 
                className="remove-feature-button"
              >
                Remove Feature
              </button>
            )}
            <div className="form-group">
              <label>Column Name:</label>
              <select
                value={feature.column_name}
                onChange={(e) => handleFeatureChange(index, 'column_name', e.target.value)}
                required
              >
                <option value="">Select Column</option>
                {columns.map((col, idx) => (
                  <option key={idx} value={col}>{col}</option>
                ))}
              </select>
            </div>

            <div className="form-group">
              <label>Type:</label>
              <select
                value={feature.type}
                onChange={(e) => handleFeatureChange(index, 'type', e.target.value)}
                required
              >
                <option value="text">Text</option>
                <option value="numeric">Numeric</option>
              </select>
            </div>

            {feature.type === 'text' && (
              <>
                <div className="form-group">
                  <label>Embedding Method:</label>
                  <select
                    value={feature.embedding_method}
                    onChange={(e) => handleFeatureChange(index, 'embedding_method', e.target.value)}
                    required
                  >
                    <option value="bert">BERT</option>
                    <option value="glove">GloVe</option>
                    <option value="word2vec">Word2Vec</option>
                  </select>
                </div>

                <div className="form-group">
                  <label>Embedding Dimension:</label>
                  <input
                    type="number"
                    value={feature.embedding_dim}
                    onChange={(e) => handleFeatureChange(index, 'embedding_dim', e.target.value)}
                    min="1"
                    required
                  />
                </div>

                <div className="form-group">
                  <label>Dimensionality Reduction:</label>
                  <select
                    value={feature.dim_reduction_method}
                    onChange={(e) => handleFeatureChange(index, 'dim_reduction_method', e.target.value)}
                  >
                    <option value="none">None</option>
                    <option value="pca">PCA</option>
                    <option value="umap">UMAP</option>
                  </select>
                </div>

                {feature.dim_reduction_method !== 'none' && (
                  <div className="form-group">
                    <label>Target Dimension:</label>
                    <input
                      type="number"
                      value={feature.dim_reduction_target_dim}
                      onChange={(e) => handleFeatureChange(index, 'dim_reduction_target_dim', e.target.value)}
                      min="1"
                      required
                    />
                  </div>
                )}
              </>
            )}

            {feature.type === 'numeric' && (
              <>
                <div className="form-group">
                  <label>Data Type:</label>
                  <select
                    value={feature.data_type}
                    onChange={(e) => handleFeatureChange(index, 'data_type', e.target.value)}
                    required
                  >
                    <option value="float">Float</option>
                    <option value="int">Integer</option>
                  </select>
                </div>

                <div className="form-group">
                  <label>Processing:</label>
                  <select
                    value={feature.processing}
                    onChange={(e) => handleFeatureChange(index, 'processing', e.target.value)}
                  >
                    <option value="none">None</option>
                    <option value="standardize">Standardize</option>
                    <option value="normalize">Normalize</option>
                  </select>
                </div>

                <div className="form-group">
                  <label>Projection:</label>
                  <select
                    value={feature.projection_method}
                    onChange={(e) => handleFeatureChange(index, 'projection_method', e.target.value)}
                  >
                    <option value="none">None</option>
                    <option value="linear">Linear</option>
                  </select>
                </div>

                {feature.projection_method !== 'none' && (
                  <div className="form-group">
                    <label>Target Dimension:</label>
                    <input
                      type="number"
                      value={feature.projection_target_dim}
                      onChange={(e) => handleFeatureChange(index, 'projection_target_dim', e.target.value)}
                      min="1"
                      required
                    />
                  </div>
                )}
              </>
            )}
          </div>
        ))}

        <button type="button" onClick={handleAddFeature} className="add-feature-button">
          Add Feature
        </button>

        <button type="submit" className="submit-button" disabled={loading}>
          {loading ? 'Processing...' : 'Create Feature Space'}
        </button>
      </form>

      {/* Display Feature Space Statistics and Summary */}
      {featureSpaceData && (
        <div className="feature-space-summary">
          <h3>Feature Space Summary</h3>

          {/* Metrics Section */}
          <div className="metrics">
            <div className="metric">
              <span className="metric-label">Total Features:</span>
              <span className="metric-value">{featureSpaceData.features.length}</span>
            </div>
            <div className="metric">
              <span className="metric-label">Total Embedding Dimensions:</span>
              <span className="metric-value">{calculateTotalDimensions()}</span>
            </div>
            <div className="metric">
              <span className="metric-label">Text Features:</span>
              <span className="metric-value">{countFeatureTypes().text}</span>
            </div>
            <div className="metric">
              <span className="metric-label">Numeric Features:</span>
              <span className="metric-value">{countFeatureTypes().numeric}</span>
            </div>
          </div>

          {/* Feature Summary Table */}
          <table className="feature-table">
            <thead>
              <tr>
                <th>Column Name</th>
                <th>Type</th>
                <th>Embedding Method</th>
                <th>Embedding Dimension</th>
                <th>Embedding Shape</th> {/* New Column */}
                <th>Dimensionality Reduction</th>
                <th>Processing</th>
              </tr>
            </thead>
            <tbody>
              {featureSpaceData.features && featureSpaceData.features.length > 0 ? (
                featureSpaceData.features.map((feature, idx) => {
                  // Retrieve the embedding shape from multi_graph_settings
                  const embeddingShape = featureSpaceData.multi_graph_settings.embedding_shapes
                    ? featureSpaceData.multi_graph_settings.embedding_shapes[feature.column_name]
                    : 'N/A';

                  return (
                    <tr key={idx}>
                      <td>{feature.column_name}</td>
                      <td>
                        <span 
                          className="badge" 
                          style={{ backgroundColor: typeColors[feature.type] || '#ccc' }}
                        >
                          {feature.type.charAt(0).toUpperCase() + feature.type.slice(1)}
                        </span>
                      </td>
                      <td>
                        {feature.type === 'text' ? (
                          <span 
                            className="badge" 
                            style={{ backgroundColor: methodColors[feature.embedding_method] || '#ccc' }}
                          >
                            {feature.embedding_method.charAt(0).toUpperCase() + feature.embedding_method.slice(1)}
                          </span>
                        ) : (
                          'N/A'
                        )}
                      </td>
                      <td>
                        {feature.type === 'text' 
                          ? feature.embedding_dim 
                          : feature.data_type === 'float' 
                            ? 'Float' 
                            : 'Int'}
                      </td>
                      <td>
                        {embeddingShape !== 'N/A' 
                          ? `(${embeddingShape})` 
                          : 'N/A'}
                      </td>
                      <td>
                        {feature.type === 'text' && feature.dim_reduction ? (
                          `${feature.dim_reduction.method.toUpperCase()} (${feature.dim_reduction.target_dim})`
                        ) : (
                          'N/A'
                        )}
                      </td>
                      <td>
                        {feature.type === 'numeric' ? (
                          feature.processing.charAt(0).toUpperCase() + feature.processing.slice(1)
                        ) : (
                          'N/A'
                        )}
                      </td>
                    </tr>
                  );
                })
              ) : (
                <tr>
                  <td colSpan="7" style={{ textAlign: 'center', padding: '20px' }}>
                    No features available.
                  </td>
                </tr>
              )}
            </tbody>
          </table>

          {/* Download Option */}
          <div className="download-section">
            <a 
              href={`data:text/json;charset=utf-8,${encodeURIComponent(JSON.stringify(featureSpaceData))}`} 
              download="feature_space.json"
              className="download-link"
            >
              Download Feature Space JSON
            </a>
          </div>
        </div>
      )}
    </div>
  );
}

export default FeatureSpaceCreatorTab;
