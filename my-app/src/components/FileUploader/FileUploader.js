// src/components/FileUploader/FileUploader.js
import React from 'react';
import { useDropzone } from 'react-dropzone';
import Papa from 'papaparse';
import './FileUploader.css'; // Updated path

const FileUploader = ({ onFileDrop }) => {
  const onDrop = (acceptedFiles) => {
    if (acceptedFiles.length === 0) return;
    const file = acceptedFiles[0];
    Papa.parse(file, {
      header: true,
      dynamicTyping: true,
      complete: (results) => {
        onFileDrop(results.data, results.meta.fields);
      },
      error: (error) => {
        console.error('Error parsing CSV:', error);
      },
    });
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: '.csv',
  });

  return (
    <div {...getRootProps()} className="dropzone">
      <input {...getInputProps()} />
      {isDragActive ? (
        <p>Drop the CSV file here...</p>
      ) : (
        <p>Drag and drop a CSV file here, or click to select file</p>
      )}
    </div>
  );
};

export default FileUploader;

/*
Detailed Explanation:

The `FileUploader` component provides a user interface for uploading CSV files by allowing users to drag and drop files or select them via a file dialog. Here's an in-depth look at its functionality:

1. **Imports**:
   - **React**: The primary library for building user interfaces.
   - **useDropzone** from **react-dropzone**: A custom hook that facilitates drag-and-drop file uploads.
   - **Papa** from **papaparse**: A powerful CSV parsing library used to convert CSV files into usable JavaScript objects.
   - **FileUploader.css**: The CSS stylesheet that styles the dropzone area and its elements.

2. **Component Definition**:
   - **Functional Component**: `FileUploader` is a functional component that accepts one prop:
     - `onFileDrop`: A callback function that is invoked after successfully parsing the CSV file. It receives two arguments:
       - `results.data`: An array of objects representing the rows in the CSV file.
       - `results.meta.fields`: An array of strings representing the header fields (column names) of the CSV.

3. **File Handling (`onDrop` Function)**:
   - **File Validation**: The function first checks if any files have been dropped. If no files are present (`acceptedFiles.length === 0`), it exits early.
   - **CSV Parsing**:
     - **Selection**: It selects the first file from the dropped files (`acceptedFiles[0]`), assuming only one file is to be processed.
     - **Papa.parse Configuration**:
       - `header: true`: Instructs PapaParse to treat the first row of the CSV as headers.
       - `dynamicTyping: true`: Enables automatic type conversion (e.g., strings to numbers).
       - `complete`: A callback function that is executed upon successful parsing, which then calls the `onFileDrop` prop with the parsed data and headers.
       - `error`: A callback function that logs any errors encountered during parsing to the console.

4. **Dropzone Setup**:
   - **useDropzone Hook**: Initializes the dropzone with specific configurations:
     - `onDrop`: The function defined above to handle file drops.
     - `accept: '.csv'`: Restricts the dropzone to accept only CSV files.

5. **Rendering**:
   - **Dropzone Area**:
     - A `div` is rendered with properties and event handlers spread from `getRootProps()`, and it is assigned the class `dropzone` for styling.
     - An `input` element is included with properties spread from `getInputProps()`, which manages file selection via the dialog.
     - **Dynamic Text**: The content inside the `div` changes based on the `isDragActive` state:
       - When a file is being dragged over the dropzone, it displays "Drop the CSV file here...".
       - Otherwise, it displays "Drag and drop a CSV file here, or click to select file".

6. **Export**:
   - The component is exported as the default export, making it available for use in other parts of the application.

**Purpose in the Application**:
The `FileUploader` component is essential for importing data into the application. By allowing users to upload CSV files, it enables the application to process and visualize data based on user-provided datasets. The integration of `react-dropzone` and `papaparse` ensures a smooth and efficient file upload and parsing experience, handling common edge cases like invalid file types and parsing errors gracefully.

*/

