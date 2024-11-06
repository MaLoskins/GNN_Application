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

/*
Detailed Explanation:

The `RelationshipModal` component provides a modal dialog that allows users to configure relationships between different data columns. This modal is essential for defining how various entities within the uploaded CSV data relate to each other, facilitating the creation of meaningful graph structures. Here's a comprehensive breakdown of its structure and functionality:

1. **Imports**:
   - **React**, **useState**, **useEffect**: Core React library and hooks for managing state and side effects within the component.
   - **Modal** from **react-modal**: A library for creating accessible modal dialogs in React applications.
   - **RelationshipModal.css**: The CSS stylesheet that styles the modal, ensuring it aligns with the application's design and accessibility standards.

2. **Accessibility Setup**:
   - `Modal.setAppElement('#root')`: Sets the root element of the app for accessibility purposes, ensuring that screen readers and other assistive technologies can appropriately handle the modal's presence.

3. **Component Definition**:
   - **Functional Component**: `RelationshipModal` is a functional component that accepts the following props:
     - `isOpen`: A boolean that determines whether the modal is open or closed.
     - `onRequestClose`: A callback function that is invoked when the modal requests to be closed (e.g., when the user clicks outside the modal or presses the escape key).
     - `columns`: An array of column names or objects representing the data columns available for defining relationships.
     - `onSaveRelationship`: A callback function that is invoked when the user saves the relationship configuration. It receives an object containing the `relationshipType` and `relationshipFeatures`.

4. **State Management**:
   - **`relationshipType`**: A state variable that holds the type of relationship being defined (e.g., "connects", "influences"). It's initialized to an empty string.
   - **`relationshipFeatures`**: A state variable that holds an array of selected features associated with the relationship. It's initialized to an empty array.

5. **Side Effects (`useEffect`)**:
   - Monitors the `isOpen` prop. When the modal is opened (`isOpen` becomes `true`), it resets the `relationshipType` and `relationshipFeatures` to their initial states, ensuring that each time the modal is opened, it starts fresh without residual data from previous interactions.

6. **Event Handlers**:
   - **`handleSubmit`**:
     - Prevents the default form submission behavior.
     - Validates that the `relationshipType` is not empty. If it is, it alerts the user to enter a relationship type.
     - Invokes the `onSaveRelationship` prop with an object containing the `relationshipType` and `relationshipFeatures`.
     - Resets the modal's state by clearing `relationshipType` and `relationshipFeatures`.
   - **`handleFeatureChange`**:
     - Handles changes to the multi-select input for relationship features.
     - Extracts the selected options and updates the `relationshipFeatures` state accordingly.

7. **Rendering**:
   - Utilizes the `Modal` component to create an accessible modal dialog with the following configurations:
     - **`isOpen`**: Controls the visibility of the modal based on the `isOpen` prop.
     - **`onRequestClose`**: Assigns the callback for closing the modal.
     - **`contentLabel`**: Provides an accessible label for the modal content.
     - **`className` and `overlayClassName`**: Assigns CSS classes for styling the modal and its overlay, respectively.
   - Inside the modal:
     - An `h2` header titled "Configure Relationship" to indicate the modal's purpose.
     - A `form` element that contains:
       - **Relationship Type Input**:
         - A labeled text input where users can specify the type of relationship.
         - Includes a placeholder example (e.g., "connects", "influences") to guide users.
       - **Features Selection**:
         - A labeled multi-select dropdown that allows users to select multiple features related to the relationship.
         - Dynamically generates `option` elements based on the `columns` prop, ensuring that users can associate relevant data columns with the relationship.
       - **Buttons**:
         - **Save**: A submit button that triggers the `handleSubmit` function to save the relationship configuration.
         - **Cancel**: A button that invokes the `onRequestClose` callback to close the modal without saving changes.

8. **Export**:
   - The component is exported as the default export, making it accessible for import and use in other parts of the application.

**Purpose in the Application**:
The `RelationshipModal` is integral for defining how different data entities interact within the graph. By allowing users to specify the type and features of relationships, it enables the application to construct more nuanced and informative graph structures. This customization ensures that the resulting visualizations accurately reflect the underlying data relationships, enhancing data analysis and insights.

*/

