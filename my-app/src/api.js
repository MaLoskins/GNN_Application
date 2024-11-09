// src/api.js

import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000'; // Adjust as needed

export const getRoot = async () => {
  try {
    const response = await axios.get(`${API_BASE_URL}/`);
    return response.data;
  } catch (error) {
    throw error;
  }
};

export const processData = async (data, config, featureSpaceData) => {
  try {
    const response = await axios.post(`${API_BASE_URL}/process-data`, {
      data,
      config,
      feature_space_data: featureSpaceData, // Use snake_case
    });
    return response.data;
  } catch (error) {
    throw error;
  }
};

export const createFeatureSpace = async (data, config) => {
  try {
    const response = await axios.post(`${API_BASE_URL}/create-feature-space`, { data, config });
    return response.data;
  } catch (error) {
    throw error;
  }
};

export const downloadGraph = async (data, config, featureSpaceData, format) => {
  try {
    const response = await axios.post(
      `${API_BASE_URL}/download-graph`,
      { data, config, featureSpaceData, format },
      {
        responseType: 'blob',
      }
    );
    return response;
  } catch (error) {
    throw error;
  }
};

export const downloadPyGData = async (data, config, featureSpaceData, nodeLabelColumn, edgeLabelColumn) => {
  try {
    const extendedConfig = {
      ...config,
      node_label_column: nodeLabelColumn,
      edge_label_column: edgeLabelColumn,
    };
    const response = await axios.post(
      `${API_BASE_URL}/download-pyg`,
      { data, config: extendedConfig, featureSpaceData },
      {
        responseType: 'blob',
      }
    );
    return response;
  } catch (error) {
    throw error;
  }
};