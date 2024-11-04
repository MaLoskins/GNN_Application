// my-app/src/components/RelationshipModal.js
import React, { useState } from 'react';
import Modal from 'react-modal';

Modal.setAppElement('#root'); // For accessibility

const RelationshipModal = ({
  isOpen,
  onRequestClose,
  columns,
  onSaveRelationship,
}) => {
  const [relationshipType, setRelationshipType] = useState('');
  const [relationshipFeatures, setRelationshipFeatures] = useState([]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!relationshipType) {
      alert('Please enter a relationship type.');
      return;
    }
    onSaveRelationship({ relationshipType, relationshipFeatures });
    // Reset modal state
    setRelationshipType('');
    setRelationshipFeatures([]);
  };

  return (
    <Modal
      isOpen={isOpen}
      onRequestClose={onRequestClose}
      contentLabel="Relationship Configuration"
      className="modal"
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
          />
        </label>
        <label>
          Features:
          <select
            multiple
            value={relationshipFeatures}
            onChange={(e) =>
              setRelationshipFeatures(
                Array.from(e.target.selectedOptions).map((option) => option.value)
              )
            }
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
