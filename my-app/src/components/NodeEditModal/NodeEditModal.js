// src/components/NodeEditModal/NodeEditModal.js

import React, { useState, useEffect } from 'react';
import Modal from 'react-modal';
import './NodeEditModal.css';

Modal.setAppElement('#root');

const NodeEditModal = ({ isOpen, onRequestClose, node, onSaveNodeEdit, featureSpaceData }) => {
  const [nodeType, setNodeType] = useState(node.type || '');
  const [nodeFeatures, setNodeFeatures] = useState(node.features || []);
  const [availableFeatures, setAvailableFeatures] = useState([]);

  useEffect(() => {
    if (isOpen) {
      const features =
        featureSpaceData && featureSpaceData.features
          ? featureSpaceData.features.map((f) => f.column_name)
          : [];
      setAvailableFeatures(features);
      setNodeType(node.type || '');
      setNodeFeatures(node.features || []);
    }
  }, [isOpen, node, featureSpaceData]);

  const handleFeatureChange = (e) => {
    const { value, checked } = e.target;
    if (checked) {
      setNodeFeatures([...nodeFeatures, value]);
    } else {
      setNodeFeatures(nodeFeatures.filter((feature) => feature !== value));
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    // Allow saving even if nodeType is empty (will default to 'default' later)
    onSaveNodeEdit({ nodeType, nodeFeatures });
  };

  const handleAddFeature = () => {
    const newFeature = prompt('Enter new feature name:');
    if (newFeature && !availableFeatures.includes(newFeature)) {
      setAvailableFeatures([...availableFeatures, newFeature]);
      setNodeFeatures([...nodeFeatures, newFeature]);
    }
  };

  return (
    <Modal
      isOpen={isOpen}
      onRequestClose={onRequestClose}
      contentLabel="Edit Node"
      className="node-edit-modal"
      overlayClassName="overlay"
    >
      <h2>Edit Node: {node.id}</h2>
      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label htmlFor="node-type">Type:</label>
          <input
            type="text"
            id="node-type"
            value={nodeType}
            onChange={(e) => setNodeType(e.target.value)}
            placeholder="e.g., User, Post"
          />
        </div>
        <div className="form-group">
          <label>Features:</label>
          <div className="features-list">
            {availableFeatures.map((feature) => (
              <div key={feature} className="feature-item">
                <input
                  type="checkbox"
                  id={`feature-${feature}`}
                  value={feature}
                  checked={nodeFeatures.includes(feature)}
                  onChange={handleFeatureChange}
                />
                <label htmlFor={`feature-${feature}`}>{feature}</label>
              </div>
            ))}
            <button
              type="button"
              onClick={handleAddFeature}
              className="add-feature-button"
            >
              + Add Feature
            </button>
          </div>
        </div>
        <div className="modal-buttons">
          <button type="submit">Save</button>
          <button type="button" onClick={onRequestClose}>
            Cancel
          </button>
        </div>
      </form>
    </Modal>
  );
};

export default NodeEditModal;
