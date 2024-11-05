// src/components/RelationshipModal/RelationshipModal.js
import React, { useState, useEffect } from 'react';
import Modal from 'react-modal';
import './RelationshipModal.css'; // Updated path

Modal.setAppElement('#root'); // For accessibility

const RelationshipModal = ({
  isOpen,
  onRequestClose,
  columns,
  onSaveRelationship,
}) => {
  const [relationshipType, setRelationshipType] = useState('');
  const [relationshipFeatures, setRelationshipFeatures] = useState([]);

  useEffect(() => {
    if (isOpen) {
      setRelationshipType('');
      setRelationshipFeatures([]);
    }
  }, [isOpen]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!relationshipType.trim()) {
      alert('Please enter a relationship type.');
      return;
    }
    onSaveRelationship({ relationshipType, relationshipFeatures });
    // Reset modal state
    setRelationshipType('');
    setRelationshipFeatures([]);
  };

  const handleFeatureChange = (e) => {
    const selectedOptions = Array.from(e.target.selectedOptions).map(option => option.value);
    setRelationshipFeatures(selectedOptions);
  };

  return (
    <Modal
      isOpen={isOpen}
      onRequestClose={onRequestClose}
      contentLabel="Relationship Configuration"
      className="relationship-modal"
      overlayClassName="overlay"
    >
      <h2>Configure Relationship</h2>
      <form onSubmit={handleSubmit}>
        <label>
          Relationship Type:
          <input
            type="text"
            value={relationshipType}
            onChange={(e) => setRelationshipType(e.target.value)}
            required
            className="relationship-type-input"
            placeholder="e.g., connects, influences"
          />
        </label>
        <label>
          Features:
          <select
            multiple
            value={relationshipFeatures}
            onChange={handleFeatureChange}
            className="relationship-features-select"
          >
            {columns.map((col) => (
              <option key={col} value={col}>
                {col}
              </option>
            ))}
          </select>
        </label>
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

export default RelationshipModal;
