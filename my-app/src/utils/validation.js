// src/utils/validation.js

export const validateGraphConfig = (config) => {
  for (let node of config.nodes) {
    if (!node.id) {
      return 'Please select an ID column for all nodes.';
    }
    if (!node.type || node.type === 'default') {
      return `Please specify a valid type for node '${node.id}'.`;
    }
  }
  for (let rel of config.relationships) {
    if (!rel.source || !rel.target) {
      return 'Please select source and target columns for all relationships.';
    }
    if (!rel.type || rel.type === 'default') {
      return 'Please specify a valid type for all relationships.';
    }
  }
  return null;
};
