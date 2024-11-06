// src/components/NodeEditModal/NodeEditModal.js
import React, { useState, useEffect } from 'react';
import Modal from 'react-modal';
import './NodeEditModal.css'; // Create this CSS file for styling

Modal.setAppElement('#root'); // For accessibility

const NodeEditModal = ({ isOpen, onRequestClose, node, onSaveNodeEdit }) => {
  const [nodeType, setNodeType] = useState(node.type || '');
  const [nodeFeatures, setNodeFeatures] = useState(node.features || []);
  const [availableFeatures, setAvailableFeatures] = useState([]);

  useEffect(() => {
    // Dynamically generate available features based on CSV data or other logic
    // For simplicity, using a static list here
    setAvailableFeatures([
      'feature1',
      'feature2',
      'feature3',
      'feature4',
      'feature5',
    ]);
  }, [node]);

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
    if (!nodeType.trim()) {
      alert('Please enter a node type.');
      return;
    }
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
            required
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
            <button type="button" onClick={handleAddFeature} className="add-feature-button">
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

/*
Detailed Explanation:

1. **Props Received:**
   - `isOpen`: Boolean indicating whether the modal is open.
   - `onRequestClose`: Function to close the modal.
   - `node`: The node object that is being edited.
   - `onSaveNodeEdit`: Function to handle saving the edited node details.

2. **State Variables:**
   - `nodeType`: Holds the current type of the node.
   - `nodeFeatures`: Holds the current list of features associated with the node.
   - `availableFeatures`: List of possible features users can assign to the node.

3. **Dynamic Features:**
   - **`handleAddFeature`**: Allows users to add new features dynamically if the existing list doesn't suffice.

4. **Form Handling:**
   - **`handleSubmit`**: Validates input and invokes `onSaveNodeEdit` with the updated type and features.
   - **`handleFeatureChange`**: Manages the selection and deselection of features.

5. **Styling:**
   - The modal uses CSS classes like `node-edit-modal`, `overlay`, `form-group`, `features-list`, `feature-item`, `add-feature-button`, and `modal-buttons` for styling. You need to create `NodeEditModal.css` to style these appropriately.

*/