# FeatureSpaceCreator.py

import os
import re
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Union
from gensim.models import Word2Vec
import torchtext   
from torchtext.vocab import GloVe
from transformers import BertModel, BertTokenizerFast
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from umap import UMAP
import warnings
import logging
import spacy  # Import for spaCy



# Disable the torchtext deprecation warning
torchtext.disable_torchtext_deprecation_warning()

# Suppress all other warnings for cleaner output
warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Import DataProcessor
from Process_Data import DataProcessor

class TextPreprocessor:
    def __init__(
        self,
        target_column: str = 'text',
        include_stopwords: bool = True,
        remove_ats: bool = True,
        word_limit: int = 100,
        tokenizer: Optional[Any] = None
    ):
        """
        Initializes the TextPreprocessor.

        Parameters:
        - target_column (str): The column in the DataFrame containing text data.
        - include_stopwords (bool): Whether to remove stopwords.
        - remove_ats (bool): Whether to remove tokens starting with '@'.
        - word_limit (int): Maximum number of words to retain in each text.
        - tokenizer (callable): Optional tokenizer function.
        """
        self.target_column = target_column
        self.include_stopwords = include_stopwords
        self.remove_ats = remove_ats
        self.word_limit = word_limit
        self.tokenizer = tokenizer if tokenizer else self.spacy_tokenizer

        # Initialize spaCy model for stopwords and tokenization
        if include_stopwords:
            try:
                self.nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
                self.stop_words = self.nlp.Defaults.stop_words
            except OSError:
                try:
                    import subprocess
                    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
                    self.nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
                    self.stop_words = self.nlp.Defaults.stop_words
                except Exception as e:
                    raise OSError(f"Failed to install the spaCy model 'en_core_web_sm': {e}")
        else:
            self.stop_words = set()

        # Compile regex patterns
        self.re_pattern = re.compile(r'[^\w\s]')
        self.at_pattern = re.compile(r'@\S+')

    def spacy_tokenizer(self, text: str) -> List[str]:
        """
        Tokenizes text using spaCy's tokenizer.

        Parameters:
        - text (str): Input text.

        Returns:
        - List[str]: List of tokens.
        """
        doc = self.nlp(text)
        return [token.text for token in doc]

    def clean_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans and tokenizes text data in the DataFrame.

        Parameters:
        - df (pd.DataFrame): Input DataFrame.

        Returns:
        - pd.DataFrame: DataFrame with an additional 'tokenized_text' column.
        """
        if self.target_column not in df.columns:
            raise ValueError(f"The target column '{self.target_column}' does not exist.")

        df = df.copy()
        df[self.target_column] = df[self.target_column].astype(str).str.lower()

        if self.remove_ats:
            df[self.target_column] = df[self.target_column].str.replace(self.at_pattern, '', regex=True)

        df[self.target_column] = df[self.target_column].str.replace(self.re_pattern, '', regex=True)
        df['tokenized_text'] = df[self.target_column].apply(self.tokenizer)

        if self.include_stopwords:
            df['tokenized_text'] = df['tokenized_text'].apply(
                lambda tokens: [word for word in tokens if word not in self.stop_words and len(word) <= self.word_limit]
            )
        else:
            df['tokenized_text'] = df['tokenized_text'].apply(
                lambda tokens: [word for word in tokens if len(word) <= self.word_limit]
            )

        return df


class EmbeddingCreator:
    def __init__(
        self,
        embedding_method: str = "bert",
        embedding_dim: int = 768,
        glove_cache_path: Optional[str] = None,
        word2vec_model_path: Optional[str] = None,
        bert_model_name: str = "bert-base-uncased",
        bert_cache_dir: Optional[str] = None,
        device: str = "cuda"
    ):
        """
        Initializes the EmbeddingCreator.

        Parameters:
        - embedding_method (str): 'bert', 'glove', or 'word2vec'.
        - embedding_dim (int): Dimension of embeddings.
        - glove_cache_path (str): Path to GloVe cache.
        - word2vec_model_path (str): Path to Word2Vec model. If None and 'word2vec' is selected, model will be trained on data.
        - bert_model_name (str): Name of the BERT model.
        - bert_cache_dir (str): Directory to cache BERT models.
        - device (str): 'cuda' or 'cpu'.
        """
        self.embedding_method = embedding_method.lower()
        self.embedding_dim = embedding_dim
        self.glove = None
        self.word2vec_model = None
        self.bert_model = None
        self.tokenizer = None

        # Determine device for BERT
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load embeddings based on the specified method
        if self.embedding_method == "glove":
            self._load_glove(glove_cache_path)
        elif self.embedding_method == "word2vec":
            if word2vec_model_path:
                self._load_word2vec(word2vec_model_path)
            else:
                self.word2vec_model = None  # To be trained later
        elif self.embedding_method == "bert":
            self._load_bert(bert_model_name, bert_cache_dir)
        else:
            raise ValueError("Unsupported embedding method. Choose from 'glove', 'word2vec', or 'bert'.")

    def _load_glove(self, glove_cache_path: str):
        if not glove_cache_path:
            raise ValueError("glove_cache_path must be provided for GloVe embeddings.")
        if not os.path.exists(glove_cache_path):
            raise FileNotFoundError(f"GloVe cache path '{glove_cache_path}' does not exist.")
        self.glove = GloVe(name="6B", dim=self.embedding_dim, cache=glove_cache_path)

    def _load_word2vec(self, word2vec_model_path: str):
        if not word2vec_model_path or not os.path.exists(word2vec_model_path):
            raise ValueError("A valid word2vec_model_path must be provided for Word2Vec embeddings.")
        self.word2vec_model = Word2Vec.load(word2vec_model_path)
        if self.word2vec_model.vector_size != self.embedding_dim:
            raise ValueError(f"Word2Vec model dimension ({self.word2vec_model.vector_size}) does not match embedding_dim ({self.embedding_dim}).")

    def _load_bert(self, bert_model_name: str, bert_cache_dir: Optional[str]):
        # Initialize the fast tokenizer
        self.tokenizer = BertTokenizerFast.from_pretrained(bert_model_name, cache_dir=bert_cache_dir)
        self.bert_model = BertModel.from_pretrained(bert_model_name, cache_dir=bert_cache_dir)
        self.bert_model.to(self.device)
        self.bert_model.eval()
        self.embedding_dim = self.bert_model.config.hidden_size  # Update embedding_dim to BERT's hidden size

    def train_word2vec(self, sentences: List[List[str]], vector_size: int = 300, window: int = 5, min_count: int = 1, workers: int = 4):
        """
        Trains a Word2Vec model on the provided sentences.

        Parameters:
        - sentences (List[List[str]]): List of tokenized sentences.
        - vector_size (int): Dimensionality of the word vectors.
        - window (int): Maximum distance between the current and predicted word within a sentence.
        - min_count (int): Ignores all words with total frequency lower than this.
        - workers (int): Use these many worker threads to train the model.
        """
        if self.embedding_method != "word2vec":
            raise ValueError("train_word2vec can only be called for 'word2vec' embedding method.")
        self.word2vec_model = Word2Vec(
            sentences=sentences,
            vector_size=self.embedding_dim,
            window=window,
            min_count=min_count,
            workers=workers,
            seed=42
        )
        # Optionally, save the trained model
        # self.word2vec_model.save("trained_word2vec.model")

    def get_embedding(self, tokens: List[str]) -> np.ndarray:
        """
        Generates an embedding for a list of tokens.

        Parameters:
        - tokens (List[str]): List of tokens.

        Returns:
        - np.ndarray: Embedding vector.
        """
        if self.embedding_method in ["glove", "word2vec"]:
            if self.embedding_method == "word2vec" and self.word2vec_model is None:
                raise ValueError("Word2Vec model is not trained. Please train the model before getting embeddings.")
            return self._get_average_embedding(tokens)
        elif self.embedding_method == "bert":
            return self._get_bert_embedding(tokens)
        else:
            raise ValueError("Unsupported embedding method.")

    def get_word_embeddings(self, tokens: List[str]) -> np.ndarray:
        """
        Generates word-level embeddings for a list of tokens.

        Parameters:
        - tokens (List[str]): List of tokens.

        Returns:
        - np.ndarray: Array of word embeddings.
        """
        if self.embedding_method in ["glove", "word2vec"]:
            if self.embedding_method == "word2vec" and self.word2vec_model is None:
                raise ValueError("Word2Vec model is not trained. Please train the model before getting embeddings.")
            return self._get_individual_embeddings(tokens)
        elif self.embedding_method == "bert":
            return self._get_bert_word_embeddings(tokens)
        else:
            raise ValueError("Unsupported embedding method.")

    def _get_average_embedding(self, tokens: List[str]) -> np.ndarray:
        embeddings = []
        for token in tokens:
            if self.embedding_method == "glove" and token in self.glove.stoi:
                embeddings.append(self.glove[token].numpy())
            elif self.embedding_method == "word2vec" and token in self.word2vec_model.wv:
                embeddings.append(self.word2vec_model.wv[token])
            else:
                continue
        if embeddings:
            return np.mean(embeddings, axis=0)
        else:
            return np.zeros(self.embedding_dim)

    def _get_individual_embeddings(self, tokens: List[str]) -> np.ndarray:
        embeddings = []
        for token in tokens:
            if self.embedding_method == "glove" and token in self.glove.stoi:
                embeddings.append(self.glove[token].numpy())
            elif self.embedding_method == "word2vec" and token in self.word2vec_model.wv:
                embeddings.append(self.word2vec_model.wv[token])
            else:
                # Handle unknown tokens by assigning a zero vector
                embeddings.append(np.zeros(self.embedding_dim))
        return np.array(embeddings)

    def _get_bert_embedding(self, tokens: List[str]) -> np.ndarray:
        text = ' '.join(tokens)
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        return cls_embedding

    def _get_bert_word_embeddings(self, tokens: List[str]) -> np.ndarray:
        text = tokens  # Pass tokens as a list for `is_split_into_words=True`
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            is_split_into_words=True,  # Indicate that the input is already split into words
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            last_hidden_state = outputs.last_hidden_state.squeeze(0)  # [seq_length x hidden_size]
            # Align tokens with words (handle subword tokens)
            word_ids = inputs.word_ids(batch_index=0)  # Get word IDs for each token
            if word_ids is None:
                raise ValueError("word_ids() returned None. Ensure you are using a fast tokenizer.")
            word_embeddings = {}
            for idx, word_id in enumerate(word_ids):
                if word_id is not None:
                    if word_id not in word_embeddings:
                        word_embeddings[word_id] = []
                    word_embeddings[word_id].append(last_hidden_state[idx].cpu().numpy())
            # Average subword embeddings for each word
            averaged_embeddings = []
            for word_id in sorted(word_embeddings.keys()):
                embeddings = np.array(word_embeddings[word_id])
                averaged = embeddings.mean(axis=0)
                averaged_embeddings.append(averaged)
        return np.array(averaged_embeddings)


class FeatureAggregatorSimple(nn.Module):
    def __init__(
        self,
        sentence_dim: int,
        categorical_columns: List[str],
        categorical_dims: Dict[str, int],
        categorical_embed_dim: int
    ):
        """
        Initializes the FeatureAggregatorSimple.

        Parameters:
        - sentence_dim (int): Dimension of sentence embeddings.
        - categorical_columns (List[str]): List of categorical column names.
        - categorical_dims (Dict[str, int]): Number of categories for each categorical column.
        - categorical_embed_dim (int): Embedding dimension for categorical features.
        """
        super(FeatureAggregatorSimple, self).__init__()
        self.categorical_columns = categorical_columns
        self.categorical_dims = categorical_dims
        self.categorical_embed_dim = categorical_embed_dim

        # Define embedding layers for each categorical feature
        self.embeddings = nn.ModuleDict({
            col: nn.Embedding(num_embeddings=dim, embedding_dim=categorical_embed_dim)
            for col, dim in categorical_dims.items()
        })

        # Define a linear layer to project sentence embeddings if needed
        self.sentence_projection = nn.Linear(sentence_dim, sentence_dim)

        # Define a linear layer to project categorical embeddings if needed
        self.categorical_projection = nn.Linear(categorical_embed_dim, sentence_dim)

        # Initialize weights dictionary
        self.weights = {col: 1.0 for col in categorical_columns}

    def set_categorical_weights(self, weights: Dict[str, float]):
        """
        Sets the weights for categorical features.

        Parameters:
        - weights (Dict[str, float]): Weights for each categorical feature.
        """
        self.weights = weights

    def forward(self, sentence_embeddings: torch.Tensor, categorical_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass to aggregate sentence and categorical embeddings.

        Parameters:
        - sentence_embeddings (torch.Tensor): Tensor of sentence embeddings.
        - categorical_data (Dict[str, torch.Tensor]): Dictionary of categorical data tensors.

        Returns:
        - torch.Tensor: Aggregated feature tensor.
        """
        aggregated_features = self.sentence_projection(sentence_embeddings)

        for col in self.categorical_columns:
            if col in categorical_data:
                embedded = self.embeddings[col](categorical_data[col])
                embedded = self.categorical_projection(embedded)
                weight = self.weights.get(col, 1.0)
                aggregated_features += weight * embedded

        # Optionally, apply non-linearity
        aggregated_features = torch.relu(aggregated_features)

        return aggregated_features


class FeatureSpaceCreator:
    def __init__(self, config: Dict[str, Any], device: str = "cuda", log_file: str = "logs/feature_space_creator.log"):
        """
        Initializes the FeatureSpaceCreator instance.

        Parameters:
        - config (Dict[str, Any]): Configuration dictionary defining feature processing.
        - device (str): Device to use for computations ('cuda' or 'cpu').
        """
        self.config = config
        self.device = device if torch.cuda.is_available() else "cpu"
        self.features = config.get("features", [])
        self.multi_graph_settings = config.get("multi_graph_settings", {})

        # Initialize containers
        self.text_features = []
        self.numeric_features = []

        # Setup logging
        self.logger = self._setup_logger(log_file=log_file)

        # Parse configuration
        self._parse_config()

        # Initialize EmbeddingCreators for text features
        self.embedding_creators = {}
        for feature in self.text_features:
            method = feature.get("embedding_method", "bert").lower()
            embedding_dim = feature.get("embedding_dim", None)
            additional_params = feature.get("additional_params", {})

            try:
                self.embedding_creators[feature["column_name"]] = EmbeddingCreator(
                    embedding_method=method,
                    embedding_dim=embedding_dim,
                    glove_cache_path=additional_params.get("glove_cache_path"),
                    word2vec_model_path=additional_params.get("word2vec_model_path"),
                    bert_model_name=additional_params.get("bert_model_name", "bert-base-uncased"),
                    bert_cache_dir=additional_params.get("bert_cache_dir"),
                    device=self.device
                )
            except Exception as e:
                self.logger.error(f"Failed to initialize EmbeddingCreator for '{feature['column_name']}': {e}")
                # Optionally, skip this feature or re-raise the exception
                raise e

        # Initialize scalers for numeric features
        self.scalers = {}
        for feature in self.numeric_features:
            processing = feature.get("processing", "none").lower()
            if processing == "standardize":
                self.scalers[feature["column_name"]] = StandardScaler()
            elif processing == "normalize":
                self.scalers[feature["column_name"]] = MinMaxScaler()
            # 'none' implies no scaling

        # Initialize projection layers for numeric features
        self.projection_layers = {}
        for feature in self.numeric_features:
            projection_config = feature.get("projection", {})
            method = projection_config.get("method", "none").lower()
            target_dim = projection_config.get("target_dim", 1)

            if method == "linear" and target_dim > 1:
                # Initialize a linear projection layer from 1 to target_dim
                projection = nn.Linear(1, target_dim).to(self.device)
                projection.eval()  # Set to evaluation mode
                self.projection_layers[feature["column_name"]] = projection

        # Initialize TextPreprocessor
        # Assuming all text features use the same preprocessor settings
        self.text_preprocessor = TextPreprocessor(
            target_column=None,  # Will set dynamically per column
            include_stopwords=True,
            remove_ats=True,
            word_limit=100,
            tokenizer=None  # Use default tokenizer
        )

    def _setup_logger(self, log_file: str) -> logging.Logger:
        """
        Sets up the logger for the class.

        Parameters:
        - log_file (str): Path to the log file.

        Returns:
        - logging.Logger: Configured logger.
        """
        logger = logging.getLogger("FeatureSpaceCreator")
        logger.setLevel(logging.INFO)
        
        # Remove all existing handlers to prevent duplicate logs
        if logger.hasHandlers():
            logger.handlers.clear()
        
        # Create a FileHandler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Define formatter and add it to the handler
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add the handler to the logger
        logger.addHandler(file_handler)
        
        return logger


        
    def _parse_config(self):
        """
        Parses the configuration to separate text and numeric features.
        """
        for feature in self.features:
            f_type = feature.get("type", "").lower()
            if f_type == "text":
                self.text_features.append(feature)
            elif f_type == "numeric":
                self.numeric_features.append(feature)
            else:
                raise ValueError(f"Unsupported feature type: '{f_type}' in feature '{feature.get('column_name')}'.")

    def process(self, dataframe: Union[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Processes the input CSV data and transforms specified columns into feature spaces.

        Parameters:
        - dataframe (str or pd.DataFrame): Path to the CSV file or a pandas DataFrame.

        Returns:
        - pd.DataFrame: DataFrame containing the consolidated feature space.
        """
        # Load data
        if isinstance(dataframe, str):
            if not os.path.exists(dataframe):
                raise FileNotFoundError(f"CSV file not found at path: {dataframe}")
            df = pd.read_csv(dataframe)
            self.logger.info(f"Loaded data from '{dataframe}'.")
        elif isinstance(dataframe, pd.DataFrame):
            df = dataframe.copy()
            self.logger.info("Loaded data from pandas DataFrame.")
        else:
            raise TypeError("dataframe must be a file path (str) or a pandas DataFrame.")

        # Initialize feature space DataFrame
        feature_space = pd.DataFrame(index=df.index)
        self.logger.info("Initialized feature space DataFrame.")

        # Process text features
        for feature in self.text_features:
            col = feature["column_name"]
            if col not in df.columns:
                raise ValueError(f"Text column '{col}' not found in the DataFrame.")

            if df[col].isnull().any():
                self.logger.warning(f"Missing values found in text column '{col}'. Filling with empty strings.")
                df[col] = df[col].fillna("")

            # Preprocess text
            if self.text_preprocessor.target_column != col:
                self.text_preprocessor.target_column = col
            processed_df = self.text_preprocessor.clean_text(df)
            tokens = processed_df["tokenized_text"].tolist()

            # If embedding method is 'word2vec' and model is not loaded, train it
            if feature["embedding_method"].lower() == "word2vec":
                word2vec_model_path = feature.get("additional_params", {}).get("word2vec_model_path", None)
                if not word2vec_model_path:
                    self.logger.info(f"Training Word2Vec model for '{col}' as no model path was provided.")
                    self.embedding_creators[col].train_word2vec(sentences=tokens)
                    self.logger.info(f"Word2Vec model trained for '{col}'.")
                else:
                    # If a path is provided, ensure the model is loaded (already handled in __init__)
                    pass

            # Generate embeddings
            embeddings = []
            for token_list in tokens:
                embedding = self.embedding_creators[col].get_embedding(token_list)
                embeddings.append(embedding)

            # Convert embeddings to numpy array
            embeddings_array = np.vstack(embeddings)  # Shape: (N, embedding_dim)
            self.logger.info(f"Generated embeddings for text column '{col}' with shape {embeddings_array.shape}.")

            # Check for dimensionality reduction
            dim_reduction_config = feature.get("dim_reduction", {})
            method = dim_reduction_config.get("method", "none").lower()
            target_dim = dim_reduction_config.get("target_dim", embeddings_array.shape[1])

            if method in ["pca", "umap"] and target_dim < embeddings_array.shape[1]:
                self.logger.info(f"Applying '{method}' to text feature '{col}' to reduce dimensions to {target_dim}.")
                if method == "pca":
                    reducer = PCA(n_components=target_dim, random_state=42)
                    reduced_embeddings = reducer.fit_transform(embeddings_array)
                elif method == "umap":
                    n_neighbors = dim_reduction_config.get("n_neighbors", 15)
                    min_dist = dim_reduction_config.get("min_dist", 0.1)
                    reducer = UMAP(n_components=target_dim, n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
                    reduced_embeddings = reducer.fit_transform(embeddings_array)

                embeddings_array = reduced_embeddings  # Update embeddings_array
                self.logger.info(f"Dimensionality reduction '{method}' applied to '{col}'. New shape: {embeddings_array.shape}.")

            # Else, no dimensionality reduction applied

            # Add to feature_space DataFrame
            # Convert numpy arrays to lists for storage in DataFrame
            feature_space[f"{col}_embedding"] = list(embeddings_array)

        # Process numeric features
        for feature in self.numeric_features:
            col = feature["column_name"]
            if col not in df.columns:
                raise ValueError(f"Numeric column '{col}' not found in the DataFrame.")

            if df[col].isnull().any():
                self.logger.warning(f"Missing values found in numeric column '{col}'. Filling with column mean.")
                df[col] = df[col].fillna(df[col].mean())

            data_type = feature.get("data_type", "float").lower()
            if data_type not in ["int", "float"]:
                raise ValueError(f"Unsupported data_type '{data_type}' for numeric column '{col}'.")

            # Ensure correct data type
            df[col] = df[col].astype(float) if data_type == "float" else df[col].astype(int)

            # Apply processing
            processing = feature.get("processing", "none").lower()
            if processing in ["standardize", "normalize"]:
                scaler = self.scalers[col]
                df_scaled = scaler.fit_transform(df[[col]])
                feature_vector = df_scaled.flatten()
                self.logger.info(f"Applied '{processing}' to numeric column '{col}'.")
            else:
                feature_vector = df[col].values.astype(float)
                self.logger.info(f"No scaling applied to numeric column '{col}'.")

            # Check for projection
            projection_config = feature.get("projection", {})
            method = projection_config.get("method", "none").lower()
            target_dim = projection_config.get("target_dim", 1)

            if method == "linear" and target_dim > 1:
                self.logger.info(f"Applying '{method}' projection to numeric feature '{col}' to increase dimensions to {target_dim}.")
                projection_layer = self.projection_layers[col]
                with torch.no_grad():
                    # Reshape feature_vector to (N, 1)
                    feature_tensor = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(1).to(self.device)
                    projected_tensor = projection_layer(feature_tensor)
                    projected_features = projected_tensor.cpu().numpy()
                feature_space[f"{col}_feature"] = list(projected_features)
                self.logger.info(f"Projection '{method}' applied to '{col}'. New shape: {projected_features.shape}.")
            else:
                # No projection applied
                feature_space[f"{col}_feature"] = feature_vector
                self.logger.info(f"Added numeric feature '{col}' with shape {feature_vector.shape}.")

        self.logger.info("Feature space creation completed.")
        return feature_space

    def aggregate_features(
        self,
        feature_space: pd.DataFrame,
        categorical_columns: List[str],
        categorical_dims: Dict[str, int],
        sentence_dim: int = 768
    ) -> torch.Tensor:
        """
        Aggregates sentence embeddings with categorical features via concatenation.

        Args:
            feature_space (pd.DataFrame): DataFrame containing embeddings and categorical data.
            categorical_columns (List[str]): List of categorical column names to include.
            categorical_dims (Dict[str, int]): Dictionary mapping categorical columns to number of categories.
            sentence_dim (int): Dimension of sentence embeddings.

        Returns:
            torch.Tensor: Tensor of shape (N, sentence_dim + sum(categorical_embed_dim)) containing final features.
        """
        # Extract sentence embeddings as a torch.Tensor
        # Modify this section if you have multiple sentence embedding columns
        sentence_embedding_cols = [col for col in feature_space.columns if col.endswith("_embedding")]
        if not sentence_embedding_cols:
            raise ValueError("No sentence embedding columns found in feature_space.")
        elif len(sentence_embedding_cols) > 1:
            self.logger.warning(f"Multiple sentence embedding columns found: {sentence_embedding_cols}. Using the first one.")

        sentence_col = sentence_embedding_cols[0]
        sentence_embeddings = torch.tensor(feature_space[sentence_col].tolist(), dtype=torch.float32).to(self.device)

        # Prepare categorical data as a dictionary of tensors
        categorical_data = {}
        for col in categorical_columns:
            if col not in feature_space.columns:
                raise ValueError(f"Categorical column '{col}' not found in feature_space.")
            cat_values = feature_space[col].values
            max_index = cat_values.max()
            if max_index >= categorical_dims[col]:
                raise ValueError(f"Categorical column '{col}' has index {max_index} which exceeds its dimension {categorical_dims[col]}.")
            categorical_data[col] = torch.tensor(cat_values, dtype=torch.long).to(sentence_embeddings.device)

        # Initialize FeatureAggregatorSimple
        aggregator = FeatureAggregatorSimple(
            sentence_dim=sentence_dim,
            categorical_columns=categorical_columns,
            categorical_dims=categorical_dims,
            categorical_embed_dim=sentence_dim  # To ensure projected_cats has sentence_dim
        ).to(sentence_embeddings.device)

        # Optionally set weights for categorical features
        # Example: all weights set to 1.0
        weights_dict = {col: 1.0 for col in categorical_columns}
        aggregator.set_categorical_weights(weights_dict)

        aggregator.eval()  # Set to evaluation mode

        with torch.no_grad():
            final_features = aggregator(sentence_embeddings, categorical_data)  # Shape: (N, sentence_dim + sum(categorical_embed_dim))

        self.logger.info("Aggregated features successfully.")
        return final_features


# --------------------- Test Configurations ---------------------

# Configuration Test 1
config_test_1 = {
    "features": [
        {
            "column_name": "text",
            "type": "text",
            "embedding_method": "bert",
            "embedding_dim": 768,
            "dim_reduction": {  # Optional dimensionality reduction
                "method": "pca",  # Options: "pca", "umap"
                "target_dim": 100  # Desired dimension after reduction
            },
            "additional_params": {
                "bert_model_name": "bert-base-uncased",
                "glove_cache_path": "Glove_Cache",  # Update this path
                "word2vec_model_path": None,  # Set to None to train on data
                "bert_cache_dir": "Bert_Cache"  # Update this path
            }
        },
        {
            "column_name": "hashtags",
            "type": "text",
            "embedding_method": "glove",
            "embedding_dim": 100,  # GloVe 100-dimensional embeddings
            "dim_reduction": {  # Optional dimensionality reduction
                "method": "umap",  # Options: "pca", "umap"
                "target_dim": 50  # Desired dimension after reduction
            },
            "additional_params": {
                "glove_cache_path": "Glove_Cache",  # Update this path
                "word2vec_model_path": None,  # Not needed for GloVe
                "bert_model_name": "bert-base-uncased",
                "bert_cache_dir": "Bert_Cache"  # Update this path
            }
        },
        {
            "column_name": "user_followers_count",
            "type": "numeric",
            "data_type": "int",
            "processing": "normalize",
            "projection": {  # Optional projection
                "method": "none",  # No projection
                "target_dim": 1  # Default dimension
            },
            "target_dim": 1  # To match default
        },
        {
            "column_name": "favorite_count",
            "type": "numeric",
            "data_type": "float",
            "processing": "standardize",
            "projection": {  # Optional projection
                "method": "linear",  # Currently, only linear projection is supported
                "target_dim": 100  # Desired dimension after projection
            }
        }
    ],
    "multi_graph_settings": {
        "embedding_shapes": {
            "text": 768,
            "hashtags": 100,
            "user_followers_count": 1,
            "favorite_count": 100
        }
    }
}

# Configuration Test 2
config_test_2 = {
    "features": [
        {
            "column_name": "text",
            "type": "text",
            "embedding_method": "word2vec",
            "embedding_dim": 300,  # Word2Vec typically uses 300 dimensions
            # No dim_reduction key
            "additional_params": {
                "word2vec_model_path": None,  # Set to None to train on data
                "glove_cache_path": "Glove_Cache",  # Update this path
                "bert_model_name": "bert-base-uncased",
                "bert_cache_dir": "Bert_Cache"  # Update this path
            }
        },
        {
            "column_name": "mentions",
            "type": "text",
            "embedding_method": "glove",
            "embedding_dim": 100,  # GloVe 100-dimensional embeddings
            # No dim_reduction key
            "additional_params": {
                "glove_cache_path": "Glove_Cache",  # Update this path
                "word2vec_model_path": None,  # Not needed for GloVe
                "bert_model_name": "bert-base-uncased",
                "bert_cache_dir": "Bert_Cache"  # Update this path
            }
        },
        {
            "column_name": "user_friends_count",
            "type": "numeric",
            "data_type": "int",
            "processing": "none",
            # No projection key
            "target_dim": 1  # Default dimension
        },
        {
            "column_name": "retweet_count",
            "type": "numeric",
            "data_type": "float",
            "processing": "normalize",
            # No projection key
            "target_dim": 1  # Default dimension
        }
    ],
    "multi_graph_settings": {
        "embedding_shapes": {
            "text": 300,
            "mentions": 100,
            "user_friends_count": 1,
            "retweet_count": 1
        }
    }
}

# Configuration Test 3
config_test_3 = {
    "features": [
        {
            "column_name": "text",
            "type": "text",
            "embedding_method": "bert",
            "embedding_dim": 768,
            "dim_reduction": {  # Optional dimensionality reduction
                "method": "umap",  # Options: "pca", "umap"
                "target_dim": 200  # Desired dimension after reduction
            },
            "additional_params": {
                "bert_model_name": "bert-base-uncased",
                "glove_cache_path": "Glove_Cache",  # Update this path
                "word2vec_model_path": None,  # Not needed for BERT
                "bert_cache_dir": "Bert_Cache"  # Update this path
            }
        },
        {
            "column_name": "user_screen_name",
            "type": "text",  # Treating as text for this test
            "embedding_method": "bert",
            "embedding_dim": 768,
            # No dim_reduction key
            "additional_params": {
                "bert_model_name": "bert-base-uncased",
                "glove_cache_path": "Glove_Cache",  # Update this path
                "word2vec_model_path": None,  # Not needed for BERT
                "bert_cache_dir": "Bert_Cache"  # Update this path
            }
        },
        {
            "column_name": "user_followers_count",
            "type": "numeric",
            "data_type": "int",
            "processing": "standardize",
            "projection": {  # Optional projection
                "method": "linear",  # Currently, only linear projection is supported
                "target_dim": 300  # Desired dimension after projection
            }
        },
        {
            "column_name": "favorite_count",
            "type": "numeric",
            "data_type": "float",
            "processing": "normalize",
            # No projection key
            "target_dim": 1  # Default dimension
        }
    ],
    "multi_graph_settings": {
        "embedding_shapes": {
            "text": 768,
            "user_screen_name": 768,
            "user_followers_count": 300,
            "favorite_count": 1
        }
    }
}

# Configuration Test 4
config_test_4 = {
    "features": [
        {
            "column_name": "text",
            "type": "text",
            "embedding_method": "bert",
            "embedding_dim": 768,
            "dim_reduction": {  # Optional dimensionality reduction
                "method": "umap",  # Options: "pca", "umap"
                "target_dim": 150  # Desired dimension after reduction
            },
            "additional_params": {
                "bert_model_name": "bert-base-uncased",
                "glove_cache_path": "Glove_Cache",  # Update this path
                "word2vec_model_path": None,  # Not needed for BERT
                "bert_cache_dir": "Bert_Cache"  # Update this path
            }
        },
        {
            "column_name": "user_friends_count",
            "type": "numeric",
            "data_type": "int",
            "processing": "normalize",
            "projection": {  # Optional projection
                "method": "linear",  # Currently, only linear projection is supported
                "target_dim": 50  # Desired dimension after projection
            }
        },
        {
            "column_name": "retweet_count",
            "type": "numeric",
            "data_type": "float",
            "processing": "standardize",
            # No projection key
            "target_dim": 1  # Default dimension
        }
    ],
    "multi_graph_settings": {
        "embedding_shapes": {
            "text": 150,
            "user_friends_count": 50,
            "retweet_count": 1
        }
    }
}

# Configuration Test 5
config_test_5 = {
    "features": [
        {
            "column_name": "text",
            "type": "text",
            "embedding_method": "bert",
            "embedding_dim": 768,
            # No dim_reduction key
            "additional_params": {
                "bert_model_name": "bert-base-uncased",
                "glove_cache_path": "Glove_Cache",  # Update this path
                "word2vec_model_path": None,  # Not needed for BERT
                "bert_cache_dir": "Bert_Cache"  # Update this path
            }
        },
        {
            "column_name": "hashtags",
            "type": "text",
            "embedding_method": "glove",
            "embedding_dim": 100,  # GloVe 100-dimensional embeddings
            # No dim_reduction key
            "additional_params": {
                "glove_cache_path": "Glove_Cache",  # Update this path
                "word2vec_model_path": None,  # Not needed for GloVe
                "bert_model_name": "bert-base-uncased",
                "bert_cache_dir": "Bert_Cache"  # Update this path
            }
        },
        {
            "column_name": "user_followers_count",
            "type": "numeric",
            "data_type": "int",
            "processing": "none",
            # No projection key
            "target_dim": 1  # Default dimension
        },
        {
            "column_name": "user_friends_count",
            "type": "numeric",
            "data_type": "int",
            "processing": "none",
            # No projection key
            "target_dim": 1  # Default dimension
        },
        {
            "column_name": "retweet_count",
            "type": "numeric",
            "data_type": "float",
            "processing": "none",
            # No projection key
            "target_dim": 1  # Default dimension
        }
    ],
    "multi_graph_settings": {
        "embedding_shapes": {
            "text": 768,
            "hashtags": 100,
            "user_followers_count": 1,
            "user_friends_count": 1,
            "retweet_count": 1
        }
    }
}

# Configuration Test 6
config_test_6 = {
    "features": [
        {
            "column_name": "text",
            "type": "text",
            "embedding_method": "bert",
            "embedding_dim": 768,
            "dim_reduction": {  # Optional dimensionality reduction
                "method": "pca",  # Options: "pca", "umap"
                "target_dim": 120  # Desired dimension after reduction
            },
            "additional_params": {
                "bert_model_name": "bert-base-uncased",
                "glove_cache_path": "Glove_Cache",  # Update this path
                "word2vec_model_path": None,  # Not needed for BERT
                "bert_cache_dir": "Bert_Cache"  # Update this path
            }
        },
        {
            "column_name": "mentions",
            "type": "text",
            "embedding_method": "word2vec",
            "embedding_dim": 300,  # Word2Vec 300-dimensional embeddings
            # No dim_reduction key
            "additional_params": {
                "word2vec_model_path": None,  # Set to None to train on data
                "glove_cache_path": "Glove_Cache",  # Update this path
                "bert_model_name": "bert-base-uncased",
                "bert_cache_dir": "Bert_Cache"  # Update this path
            }
        }
    ],
    "multi_graph_settings": {
        "embedding_shapes": {
            "text": 120,
            "mentions": 300,
            "lang": 50
        }
    }
}

# Configuration Test 7
config_test_7 = {
    "features": [
        {
            "column_name": "text",
            "type": "text",
            "embedding_method": "bert",
            "embedding_dim": 768,
            # No dim_reduction key
            "additional_params": {
                "bert_model_name": "bert-base-uncased",
                "glove_cache_path": "Glove_Cache",  # Update this path
                "word2vec_model_path": None,  # Not needed for BERT
                "bert_cache_dir": "Bert_Cache"  # Update this path
            }
        },
        {
            "column_name": "urls",
            "type": "text",
            "embedding_method": "glove",
            "embedding_dim": 100,  # GloVe 100-dimensional embeddings
            "dim_reduction": {  # Optional dimensionality reduction
                "method": "umap",  # Options: "pca", "umap"
                "target_dim": 60  # Desired dimension after reduction
            },
            "additional_params": {
                "glove_cache_path": "Glove_Cache",  # Update this path
                "word2vec_model_path": None,  # Not needed for GloVe
                "bert_model_name": "bert-base-uncased",
                "bert_cache_dir": "Bert_Cache"  # Update this path
            }
        },
        {
            "column_name": "user_friends_count",
            "type": "numeric",
            "data_type": "int",
            "processing": "standardize",
            "projection": {  # Optional projection
                "method": "linear",
                "target_dim": 80  # Desired dimension after projection
            },
            "target_dim": 80  # To match projection
        }
    ],
    "multi_graph_settings": {
        "embedding_shapes": {
            "text": 768,
            "urls": 60,
            "user_friends_count": 80
        }
    }
}

# Configuration Test 8
config_test_8 = {
    "features": [
        {
            "column_name": "text",
            "type": "text",
            "embedding_method": "bert",
            "embedding_dim": 768,
            "dim_reduction": {  # Optional dimensionality reduction
                "method": "umap",  # Options: "pca", "umap"
                "target_dim": 180  # Desired dimension after reduction
            },
            "additional_params": {
                "bert_model_name": "bert-base-uncased",
                "glove_cache_path": "Glove_Cache",  # Update this path
                "word2vec_model_path": None,  # Not needed for BERT
                "bert_cache_dir": "Bert_Cache"  # Update this path
            }
        },
        {
            "column_name": "user_screen_name",
            "type": "text",
            "embedding_method": "word2vec",
            "embedding_dim": 300,  # Word2Vec 300-dimensional embeddings
            # No dim_reduction key
            "additional_params": {
                "word2vec_model_path": None,  # Set to None to train on data
                "glove_cache_path": "Glove_Cache",  # Update this path
                "bert_model_name": "bert-base-uncased",
                "bert_cache_dir": "Bert_Cache"  # Update this path
            }
        },
        {
            "column_name": "favorite_count",
            "type": "numeric",
            "data_type": "float",
            "processing": "normalize",
            "projection": {  # Optional projection
                "method": "linear",
                "target_dim": 100  # Desired dimension after projection
            },
            "target_dim": 100  # To match projection
        },
        {
            "column_name": "reply_to_tweet_id",
            "type": "numeric",
            "data_type": "int",
            "processing": "standardize",
            "projection": {  # Optional projection
                "method": "none",
                "target_dim": 1
            }
        }
    ],
    "multi_graph_settings": {
        "embedding_shapes": {
            "text": 180,
            "user_screen_name": 300,
            "favorite_count": 100,
            "reply_to_tweet_id": 1
        }
    }
}

# Configuration Test 9
config_test_9 = {
    "features": [
        {
            "column_name": "text",
            "type": "text",
            "embedding_method": "glove",
            "embedding_dim": 100,  # GloVe 100-dimensional embeddings
            "dim_reduction": {  # Optional dimensionality reduction
                "method": "pca",  # Options: "pca", "umap"
                "target_dim": 90  # Desired dimension after reduction
            },
            "additional_params": {
                "glove_cache_path": "Glove_Cache",  # Update this path
                "word2vec_model_path": None,  # Not needed for GloVe
                "bert_model_name": "bert-base-uncased",
                "bert_cache_dir": "Bert_Cache"  # Update this path
            }
        },
        {
            "column_name": "hashtags",
            "type": "text",
            "embedding_method": "glove",
            "embedding_dim": 100,  # GloVe 100-dimensional embeddings
            "dim_reduction": {  # Optional dimensionality reduction
                "method": "umap",  # Options: "pca", "umap"
                "target_dim": 40  # Desired dimension after reduction
            },
            "additional_params": {
                "glove_cache_path": "Glove_Cache",  # Update this path
                "word2vec_model_path": None,  # Not needed for GloVe
                "bert_model_name": "bert-base-uncased",
                "bert_cache_dir": "Bert_Cache"  # Update this path
            }
        },
        {
            "column_name": "mentions",
            "type": "text",
            "embedding_method": "bert",
            "embedding_dim": 768,
            # No dim_reduction key
            "additional_params": {
                "bert_model_name": "bert-base-uncased",
                "glove_cache_path": "Glove_Cache",  # Update this path
                "word2vec_model_path": None,  # Not needed for BERT
                "bert_cache_dir": "Bert_Cache"  # Update this path
            }
        },
        {
            "column_name": "user_friends_count",
            "type": "numeric",
            "data_type": "int",
            "processing": "normalize",
            "projection": {  # Optional projection
                "method": "linear",
                "target_dim": 60  # Desired dimension after projection
            },
            "target_dim": 60  # To match projection
        },
        {
            "column_name": "user_followers_count",
            "type": "numeric",
            "data_type": "int",
            "processing": "standardize",
            "projection": {  # Optional projection
                "method": "none",
                "target_dim": 1
            }
        }
    ],
    "multi_graph_settings": {
        "embedding_shapes": {
            "text": 90,
            "hashtags": 40,
            "mentions": 768,
            "user_friends_count": 60,
            "user_followers_count": 1
        }
    }
}

# Configuration Test 10
config_test_10 = {
    "features": [
        {
            "column_name": "text",
            "type": "text",
            "embedding_method": "bert",
            "embedding_dim": 768,
            "dim_reduction": {  # Optional dimensionality reduction
                "method": "umap",  # Options: "pca", "umap"
                "target_dim": 160  # Desired dimension after reduction
            },
            "additional_params": {
                "bert_model_name": "bert-base-uncased",
                "glove_cache_path": "Glove_Cache",  # Update this path
                "word2vec_model_path": None,  # Not needed for BERT
                "bert_cache_dir": "Bert_Cache"  # Update this path
            }
        },
        {
            "column_name": "urls",
            "type": "text",
            "embedding_method": "word2vec",
            "embedding_dim": 300,  # Word2Vec 300-dimensional embeddings
            "dim_reduction": {  # Optional dimensionality reduction
                "method": "pca",  # Options: "pca", "umap"
                "target_dim": 100  # Desired dimension after reduction
            },
            "additional_params": {
                "word2vec_model_path": None,  # Set to None to train on data
                "glove_cache_path": "Glove_Cache",  # Update this path
                "bert_model_name": "bert-base-uncased",
                "bert_cache_dir": "Bert_Cache"  # Update this path
            }
        },
        {
            "column_name": "retweet_count",
            "type": "numeric",
            "data_type": "float",
            "processing": "normalize",
            "projection": {  # Optional projection
                "method": "linear",
                "target_dim": 90  # Desired dimension after projection
            },
            "target_dim": 90  # To match projection
        },
        {
            "column_name": "reply_to_tweet_id",
            "type": "numeric",
            "data_type": "int",
            "processing": "standardize",
            "projection": {  # Optional projection
                "method": "linear",
                "target_dim": 70  # Desired dimension after projection
            },
            "target_dim": 70  # To match projection
        }
    ],
    "multi_graph_settings": {
        "embedding_shapes": {
            "text": 160,
            "urls": 100,
            "retweet_count": 90,
            "reply_to_tweet_id": 70,
            "lang": 40
        }
    }
}

# --------------------- Main Function ---------------------

# FeatureSpaceCreator.py


def main():
    """
    Main function to run all test configurations iteratively and save their outputs for comparison.
    """
    import pandas as pd
    import json

    # Define all test configurations with associated names
    test_configs = {
        "test_config_1": config_test_1,
        "test_config_2": config_test_2,
        "test_config_3": config_test_3,
        "test_config_4": config_test_4,
        "test_config_5": config_test_5,
        "test_config_6": config_test_6,
        "test_config_7": config_test_7,
        "test_config_8": config_test_8,
        "test_config_9": config_test_9,
        "test_config_10": config_test_10
    }

    # Initialize a list to hold results for comparison (optional)
    results_summary = []



    # Initialize DataProcessor with custom transformations
    data_processor = DataProcessor(
        input_filepath="prince-toronto.csv",      # Update if necessary
        output_filepath="Processed_Data.csv",      # Temporary file; can be adjusted
        report_path="Type_Conversion_Report.csv",
        return_category_mappings=True,
        mapping_directory="Category_Mappings",
        parallel_processing=True,                   # Enable parallel processing for speed
        dayfirst=True,                             # Adjust based on your date format
        log_level="INFO",
        log_file="logs/data_processor.log",             # Optional: specify a log file
        convert_factors_to_int=True,
        date_format=None,                          # Keep datetime dtype
        save_type="csv"
    )

    # Process the data
    try:
        processed_data = data_processor.process()
    except Exception as e:
        print(f"An error occurred during data processing: {e}")
        return

    # Iterate through each test configuration
    for test_name, config in test_configs.items():
        print(f"\n{'='*60}\nProcessing {test_name}...\n{'='*60}")
        try:
            # Initialize FeatureSpaceCreator
            feature_creator = FeatureSpaceCreator(config=config, device="cuda")

            # Process the DataFrame to create feature space
            feature_space = feature_creator.process(processed_data)

            # Display the first few rows of the feature space
            print(f"Feature Space for {test_name}:")
            print(feature_space.head())

            # Save the feature space to a new CSV
            output_csv = f"processed_features_{test_name}.csv"
            feature_space.to_csv(output_csv, index=False)
            print(f"Processed features saved to '{output_csv}'.")

            # Optionally, add to results_summary
            results_summary.append({
                "Test Name": test_name,
                "Output CSV": output_csv,
                "Shape": feature_space.shape
            })

        except Exception as e:
            print(f"An error occurred while processing {test_name}: {e}")

    # Optionally, display a summary of all test results
    print(f"\n{'='*60}\nSummary of All Tests\n{'='*60}")
    if results_summary:
        summary_df = pd.DataFrame(results_summary)
        print(summary_df)
    else:
        print("No successful test runs.")

if __name__ == "__main__":
    main()