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

export const processData = async (data, config) => {
  try {
    const response = await axios.post(`${API_BASE_URL}/process-data`, { data, config });
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
