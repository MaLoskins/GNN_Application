# my-app-backend/main.py
# To boot server use: uvicorn main:app --reload

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field, ValidationError
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, Optional
import networkx as nx
from networkx.readwrite import json_graph
from DataFrameToGraph import DataFrameToGraph
from FeatureSpaceCreator import FeatureSpaceCreator
import io
import torch
from fastapi.responses import Response
from GraphDataToPyG import GraphDataToPyG

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

# Data Models
class DataModel(BaseModel):
    data: List[Dict[str, Any]]
    config: Dict[str, Any]
    feature_space_data: Optional[Any] = None  # Use snake_case

    class Config:
        arbitrary_types_allowed = True


class FeatureSpaceRequest(BaseModel):
    data: List[Dict[str, Any]]  # List of dictionaries representing CSV rows
    config: Dict[str, Any]      # Configuration dictionary

class DownloadGraphRequest(BaseModel):
    data: List[Dict[str, Any]]
    config: Dict[str, Any]
    feature_space_data: Optional[Dict[str, Any]] = None  # Use snake_case
    format: str = 'graphml'

@app.get("/")
def read_root():
    return {"message": "Hello from the Python Backend!"}


def parse_feature_space_data(feature_space_data):
    if feature_space_data is None:
        return None

    if isinstance(feature_space_data, dict):
        if 'feature_space' in feature_space_data:
            # 'feature_space' should be a dict representing the DataFrame in 'split' format
            feature_space_json = feature_space_data['feature_space']
            # Convert the 'split' format to a DataFrame
            feature_space_df = pd.DataFrame(**feature_space_json)
            return feature_space_df
        else:
            # If it's already in a suitable format, return as is
            return feature_space_data
    else:
        # Handle other types if necessary
        return None
    
@app.post("/process-data")
def process_data(model: DataModel):
    try:
        df = pd.DataFrame(model.data)
        config = model.config
        graph_type = config.get('graph_type', 'directed')
        feature_space_data = parse_feature_space_data(model.feature_space_data)

        # Initialize DataFrameToGraph
        df_to_graph = DataFrameToGraph(df, config, graph_type=graph_type, feature_space=feature_space_data)
        graph = df_to_graph.get_graph()

        # Serialize graph to node-link format
        graph_data = json_graph.node_link_data(graph)
        
        return {"graph": graph_data}
    except ValidationError as ve:
        # Log the detailed validation errors
        print(ve.json())
        raise HTTPException(status_code=422, detail=ve.errors())
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


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

@app.post("/download-graph")
def download_graph(request: DownloadGraphRequest):
    try:
        df = pd.DataFrame(request.data)
        config = request.config
        graph_type = config.get('graph_type', 'directed')
        feature_space_data = request.feature_space_data  # Corrected to 'feature_space_data'
        format = request.format.lower()

        # Initialize DataFrameToGraph
        df_to_graph = DataFrameToGraph(df, config, graph_type=graph_type, feature_space=feature_space_data)
        graph = df_to_graph.get_graph()

        if format == 'graphml':
            graphml_str = '\n'.join(nx.generate_graphml(graph))
            return Response(content=graphml_str, media_type='application/xml')
        elif format == 'gexf':
            gexf_str = '\n'.join(nx.generate_gexf(graph))
            return Response(content=gexf_str, media_type='application/xml')
        elif format == 'gml':
            gml_str = '\n'.join(nx.generate_gml(graph))
            return Response(content=gml_str, media_type='text/plain')
        else:
            raise HTTPException(status_code=400, detail='Unsupported format requested.')

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/download-pyg")
def download_pyg(model: DataModel):
    try:
        df = pd.DataFrame(model.data)
        config = model.config
        graph_type = config.get('graph_type', 'directed')
        feature_space_data = parse_feature_space_data(model.feature_space_data)

        # Initialize DataFrameToGraph
        df_to_graph = DataFrameToGraph(df, config, graph_type=graph_type, feature_space=feature_space_data)
        graph = df_to_graph.get_graph()

        # Serialize graph to node-link format
        graph_data = json_graph.node_link_data(graph)

        # Get node and edge label columns from config
        node_label_column = config.get('node_label_column', None)
        edge_label_column = config.get('edge_label_column', None)

        # Convert to PyG Data object
        converter = GraphDataToPyG(graph_data, node_label_column=node_label_column, edge_label_column=edge_label_column)
        pyg_data = converter.convert()

        # Serialize the PyG Data object to bytes
        buffer = io.BytesIO()
        torch.save(pyg_data, buffer)
        buffer.seek(0)

        # Return as a downloadable file
        return Response(
            content=buffer.getvalue(),
            media_type='application/octet-stream',
            headers={"Content-Disposition": f"attachment; filename=graph_data.pt"}
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))