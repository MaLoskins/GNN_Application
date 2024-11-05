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
