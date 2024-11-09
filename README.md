# CSV to Graph and Feature Space Creator

An interactive web application that transforms CSV data into graph visualizations and creates feature spaces for machine learning models. This application provides a user-friendly interface to upload CSV files, configure nodes and relationships, visualize graphs, and generate feature spaces with customizable options.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
    - [Backend Setup](#backend-setup)
    - [Frontend Setup](#frontend-setup)
- [Usage](#usage)
  - [Uploading CSV Files](#uploading-csv-files)
  - [Configuring Nodes and Relationships](#configuring-nodes-and-relationships)
  - [Visualizing the Graph](#visualizing-the-graph)
  - [Creating Feature Spaces](#creating-feature-spaces)
- [Project Structure](#project-structure)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This application allows users to:

- **Transform CSV Data into Graphs**: Upload CSV files and convert them into interactive graph representations based on selected columns.
- **Customize Nodes and Relationships**: Define how columns map to nodes and relationships, including types and features.
- **Visualize Graphs Interactively**: View and interact with the generated graphs directly within the browser.
- **Create Feature Spaces**: Generate feature spaces from the data, applying embedding methods and dimensionality reduction techniques.

---

## Features

- **CSV Upload with Drag-and-Drop Support**
- **Dynamic Node and Relationship Configuration**
- **Interactive Graph Visualization using React Flow and ForceGraph2D**
- **Feature Space Creator with Support for Text and Numeric Features**
- **Custom Embedding Methods (BERT, GloVe, Word2Vec)**
- **Dimensionality Reduction Options (PCA, UMAP)**
- **Responsive Design for Various Screen Sizes**

---

## Getting Started

### Prerequisites

- **Node.js** (v12 or higher)
- **npm** (v6 or higher)
- **Python** (3.7 or higher)
- **pip** (Python package manager)
- **Virtual Environment** (Recommended for Python dependencies)

### Installation

#### Backend Setup

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name/my-app-backend
   ```

2. **Create and Activate a Virtual Environment (Recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Backend Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Backend Server**

   ```bash
   uvicorn main:app --reload
   ```

   The backend server will start at `http://localhost:8000`.

#### Frontend Setup

1. **Navigate to the Frontend Directory**

   ```bash
   cd ../my-app
   ```

2. **Install Frontend Dependencies**

   ```bash
   npm install
   ```

3. **Run the Frontend Application**

   ```bash
   npm start
   ```

   The frontend application will start at `http://localhost:3000`.

---

## Usage

### Uploading CSV Files

1. **Access the Application**

   Open your web browser and navigate to `http://localhost:3000`.

2. **Upload a CSV File**

   - Use the drag-and-drop area to upload a CSV file.
   - Alternatively, click the area to select a file from your computer.

### Configuring Nodes and Relationships

1. **Select Nodes**

   - After uploading, you'll see a list of columns.
   - Check the boxes next to columns you want to represent as nodes.

2. **Configure Relationships**

   - Use the React Flow interface to draw connections between nodes.
   - A modal will appear to set the relationship type and associated features.

3. **Edit Nodes (Optional)**

   - Click on a node to open the Node Edit Modal.
   - Set the node type and select features.

4. **Process the Graph**

   - Click the **Process Graph** button.
   - The application will send the data to the backend for processing.

### Visualizing the Graph

- Once processed, the graph visualization will appear.
- You can zoom, pan, and interact with the graph elements.

### Creating Feature Spaces

1. **Switch to Feature Space Creator**

   - Click on the **Feature Space Creator** tab.

2. **Add Features**

   - Click **Add Feature** to define new features.
   - For each feature, select:
     - Column Name
     - Type (Text or Numeric)
     - Embedding and Processing Options

3. **Create the Feature Space**

   - Click **Create Feature Space**.
   - The application will process the data and generate the feature space.

4. **View and Download**

   - Review the feature space summary.
   - Download the feature space as a JSON file if desired.

---

## Project Structure

```
project-root/
├── my-app/                    # Frontend Application
│   ├── src/
│   │   ├── components/
│   │   │   ├── ConfigurationPanel/
│   │   │   ├── FeatureSpaceCreatorTab/
│   │   │   ├── FileUploader/
│   │   │   ├── GraphVisualizer/
│   │   │   ├── NodeEditModal/
│   │   │   ├── ReactFlowWrapper/
│   │   │   └── RelationshipModal/
│   │   ├── hooks/
│   │   │   └── useGraph.js
│   │   ├── App.js
│   │   ├── App.css
│   │   ├── api.js
│   │   └── index.js
│   ├── package.json
│   └── README.md
├── my-app-backend/            # Backend Application
│   ├── main.py
│   ├── DataFrameToGraph.py
│   ├── FeatureSpaceCreator.py
│   ├── requirements.txt
│   └── README.md
└── README.md                  # Project README
```

---

## Future Enhancements

- **User Authentication and Profiles**
- **Database Integration for Persistent Storage**
- **Enhanced Graph Analytics and Metrics**
- **Support for Additional File Formats (e.g., JSON, Excel)**
- **Integration with Machine Learning Models**
- **Advanced Error Handling and Validation**
- **Internationalization and Localization**
- **Accessibility Improvements**
- **Deployment Scripts for Cloud Platforms**

---

## Contributing

Contributions are welcome! If you'd like to contribute, please follow these steps:

1. **Fork the Repository**

   Click the **Fork** button at the top-right corner of this page.

2. **Clone Your Fork**

   ```bash
   git clone https://github.com/yourusername/your-forked-repo.git
   ```

3. **Create a Feature Branch**

   ```bash
   git checkout -b feature/YourFeatureName
   ```

4. **Commit Your Changes**

   ```bash
   git commit -am 'Add some feature'
   ```

5. **Push to the Branch**

   ```bash
   git push origin feature/YourFeatureName
   ```

6. **Open a Pull Request**

   Submit a pull request to the original repository.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Note**: Please ensure that all dependencies are correctly installed and that the backend and frontend applications are properly connected. Adjust the origins in the CORS settings of the backend (`main.py`) if accessing from a different host or port.

---

Feel free to expand this README with additional details as you develop new features or if you need to provide more specific instructions.