// src/api.js
import axios from 'axios';

const API_BASE_URL = 'http://127.0.0.1:8000';

export const getRoot = async () => {
  try {
    const response = await axios.get(`${API_BASE_URL}/`);
    return response.data;
  } catch (error) {
    throw error;
  }
};

export const processData = async (data) => {
  try {
    const response = await axios.post(`${API_BASE_URL}/process-data`, { data });
    return response.data;
  } catch (error) {
    throw error;
  }
};
