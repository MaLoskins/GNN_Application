// src/components/FileUploader/FileUploader.js
import React from 'react';
import { useDropzone } from 'react-dropzone';
import Papa from 'papaparse';
import './FileUploader.css'; // Create this CSS file for styling

const FileUploader = ({ onFileDrop }) => {
  const onDrop = (acceptedFiles) => {
    if (acceptedFiles.length === 0) return;

    const file = acceptedFiles[0];
    Papa.parse(file, {
      header: true,
      skipEmptyLines: true,
      complete: function(results) {
        const data = results.data;
        const fields = results.meta.fields;
        onFileDrop(data, fields);
      },
      error: function(error) {
        console.error('Error parsing CSV:', error);
        alert('Error parsing CSV file.');
      }
    });
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: '.csv',
    multiple: false
  });

  return (
    <div className="file-uploader">
      <div {...getRootProps()} className={`dropzone ${isDragActive ? 'active' : ''}`}>
        <input {...getInputProps()} />
        {
          isDragActive ?
            <p>Drop the CSV file here ...</p> :
            <p>Drag 'n' drop a CSV file here, or click to select file</p>
        }
      </div>
    </div>
  );
};

export default FileUploader;

/*
Detailed Explanation:

1. **Props Received:**
   - `onFileDrop`: Function to handle the parsed CSV data and column headers.

2. **File Parsing:**
   - Uses `Papa.parse` to parse the uploaded CSV file.
   - On successful parsing, it calls `onFileDrop` with the data and headers.
   - Handles parsing errors gracefully.

3. **Styling:**
   - The `FileUploader.css` should style the dropzone area appropriately.
*/

