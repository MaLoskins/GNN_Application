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

/*
Detailed Explanation:

The `ConfigurationPanel` component serves as a user interface element that allows users to configure relationships between different data columns within the application. Here's a breakdown of its structure and functionality:

1. **Imports**:
   - **React**: The core library for building user interfaces in React.
   - **ConfigurationPanel.css**: The CSS stylesheet that styles the component, ensuring it adheres to the application's design system.

2. **Component Definition**:
   - **Functional Component**: `ConfigurationPanel` is defined as a functional component that accepts three props:
     - `columns`: An array of column names or objects representing the data columns available for configuration.
     - `onSubmit`: A callback function that is invoked when the user clicks the "Submit Configuration" button. This function likely handles the submission of the configured relationships to the backend or updates the application's state.
     - `loading`: A boolean value indicating whether a submission process is currently ongoing. When `loading` is `true`, the submit button is disabled and displays "Processing..." to inform the user that their submission is being processed.

3. **Rendering**:
   - The component returns a JSX structure comprising:
     - A `div` with the class `config-section` that encapsulates the entire configuration panel, applying styles defined in the CSS file.
     - An `h2` header titled "Configuration" to denote the purpose of the panel.
     - A `p` tag containing instructional text guiding the user on how to define relationships by dragging connections between columns and specifying relationship types and features.
     - A `button` element that triggers the `onSubmit` function when clicked. The button's label dynamically changes based on the `loading` state:
       - Displays "Processing..." when `loading` is `true`.
       - Displays "Submit Configuration" when `loading` is `false`.
     - The button is disabled during the loading state to prevent multiple submissions.

4. **Export**:
   - The component is exported as the default export, allowing it to be easily imported and used in other parts of the application.

**Purpose in the Application**:
The `ConfigurationPanel` is a crucial component that facilitates user interaction for setting up data relationships. By allowing users to define how different data columns relate to each other, it enables the dynamic generation of graphs or other data visualizations based on user-defined configurations. This component likely interacts closely with other components like `ReactFlowWrapper` for visualizing relationships and `App.js` for managing the overall application state.

*/

