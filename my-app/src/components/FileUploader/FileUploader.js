// src/components/FileUploader/FileUploader.js

import React from 'react';
import { useDropzone } from 'react-dropzone';
import Papa from 'papaparse';
import './FileUploader.css'; // Ensure correct path

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
            <p>Drop the CSV file here...</p> :
            <p>Drag & drop a CSV file here, or click to select file</p>
        }
      </div>
    </div>
  );
};

export default FileUploader;
