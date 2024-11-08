# my-app-backend/main.py
# To boot server use: uvicorn main:app --reload

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
import networkx as nx
from networkx.readwrite import json_graph
from DataFrameToGraph import DataFrameToGraph  # Ensure this class is accessible
from FeatureSpaceCreator import FeatureSpaceCreator  # Import the FeatureSpaceCreator class

app = FastAPI()

# CORS Configuration
origins = [
    "http://localhost:3000",  # React app origin
    # Add other origins if needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data Model
class DataModel(BaseModel):
    data: List[Dict[str, Any]]  # List of dictionaries representing CSV rows
    config: Dict[str, Any]      # Configuration dictionary

@app.get("/")
def read_root():
    return {"message": "Hello from the Python Backend!"}

@app.post("/process-data")
def process_data(model: DataModel):
    try:
        df = pd.DataFrame(model.data)
        config = model.config
        graph_type = config.get('graph_type', 'directed')  # Optional graph type

        # Initialize DataFrameToGraph
        df_to_graph = DataFrameToGraph(df, config, graph_type=graph_type)
        graph = df_to_graph.get_graph()

        # Serialize graph to node-link format
        graph_data = json_graph.node_link_data(graph)
        
        return {"graph": graph_data}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# New Endpoint for Feature Space Creation
class FeatureSpaceRequest(BaseModel):
    data: List[Dict[str, Any]]  # List of dictionaries representing CSV rows
    config: Dict[str, Any]      # Configuration dictionary

@app.post("/create-feature-space")
def create_feature_space(request: FeatureSpaceRequest):
    try:
        df = pd.DataFrame(request.data)
        config = request.config

        # Initialize FeatureSpaceCreator
        feature_space_creator = FeatureSpaceCreator(config=config, device="cpu")  # Adjust device as needed

        # Process the DataFrame to create feature space
        feature_space = feature_space_creator.process(df)

        # Convert feature_space DataFrame to JSON
        feature_space_json = feature_space.to_json(orient="split")  # 'split' format for better structure

        # Prepare the response with features and multi_graph_settings
        response = {
            "features": config.get("features", []),
            "multi_graph_settings": config.get("multi_graph_settings", {}),
            "feature_space": feature_space_json
        }

        return response
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

