from abc import ABC
import numpy as np
import pickle
from scipy.sparse import csr_matrix

import os
import torch
import torch.nn as nn

class EmbeddingModel(ABC):
    """
    Abstract base class for embedding models.
    
    This class defines the interface that all embedding models should implement,
    providing methods for encoding data to embeddings and decoding embeddings back to data.
    """
    
    model_name: str
    device: torch.device = torch.device('cpu')
    encoder: nn.Module | None = None
    decoder: nn.Module | None = None
    
    def _post_init(self):
        self.encoder.to(self.device)
        self.decoder.to(self.device)
            
    def encode(self, data: np.ndarray | torch.Tensor | csr_matrix, **kwargs) -> np.ndarray:
        """
        Encode input data into cell embedding vectors.
        
        Args:
            data (np.ndarray): Input data array to encode into embeddings
            **kwargs: Additional encoding parameters (e.g., batch_size, normalize)
            
        Returns:
            np.ndarray: Array of cell embedding vectors with shape (n_cells, embedding_dim)
        """
        # Convert input to torch tensor and move to same device as model
        if isinstance(data, torch.Tensor):
            data_tensor = data.float().to(self.device)
        elif isinstance(data, np.ndarray):
            data_tensor = torch.from_numpy(data).float().to(self.device)
        elif isinstance(data, csr_matrix):
            data_tensor = torch.from_numpy(data.toarray()).float().to(self.device)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
        
        # Encode using the encoder module
        with torch.no_grad():
            embeddings = self.encoder(data_tensor)
        
        return embeddings
    
    def decode(self, embeddings: torch.Tensor | np.ndarray | csr_matrix, **kwargs) -> np.ndarray:
        """
        Decode embedding vectors back to original data format.
        
        Note: Not all embedding models support decoding. This method should raise
        NotImplementedError for models that don't support this operation.
        
        Args:
            embeddings (np.ndarray): Array of cell embedding vectors to decode
            **kwargs: Additional decoding parameters
            
        Returns:
            np.ndarray: Decoded data array corresponding to the input embeddings
        """
        # Convert input to torch tensor and move to same device as model
        if isinstance(embeddings, torch.Tensor):
            embeddings_tensor = embeddings.float().to(self.device)
        elif isinstance(embeddings, np.ndarray):
            embeddings_tensor = torch.from_numpy(embeddings).float().to(self.device)
        elif isinstance(embeddings, csr_matrix):
            embeddings_tensor = torch.from_numpy(embeddings.toarray()).float().to(self.device)
        else:
            raise ValueError(f"Unsupported data type: {type(embeddings)}")
        
        # Decode using the decoder module
        with torch.no_grad():
            reconstructed = self.decoder(embeddings_tensor)
        
        return reconstructed

    @classmethod
    def load(cls, filepath: str) -> 'EmbeddingModel':
        """
        Load an embedding model from a file.
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        return model_data['model_class'].load(
            filepath=filepath
        )



class PCAEncoder(nn.Module):
    """PyTorch module for PCA encoding."""
    
    def __init__(self, feature_loadings: torch.Tensor, mean_centering_vector: torch.Tensor):
        super().__init__()
        self.register_buffer('feature_loadings', feature_loadings)
        self.register_buffer('mean_centering_vector', mean_centering_vector)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Center the data
        centered = x - self.mean_centering_vector
        # Project onto principal components
        return centered @ self.feature_loadings.T

class PCADecoder(nn.Module):
    """PyTorch module for PCA decoding."""
    
    def __init__(self, feature_loadings: torch.Tensor, mean_centering_vector: torch.Tensor):
        super().__init__()
        self.register_buffer('feature_loadings', feature_loadings)
        self.register_buffer('mean_centering_vector', mean_centering_vector)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Project back to original space
        reconstructed_centered = x @ self.feature_loadings
        # Add back the mean
        return reconstructed_centered + self.mean_centering_vector

class PCAEmbeddingModel(EmbeddingModel):
    """
    PCA-based embedding model that uses principal component analysis for 
    dimensionality reduction and reconstruction.
    """
    
    def __init__(
        self,
        feature_loadings: np.ndarray,
        mean_centering_vector: np.ndarray,
        model_name: str = 'pca',
        device: torch.device | None = None
    ):
        """
        Initialize the PCA embedding model.
        
        Args:
            model_name (str): Name or identifier of the embedding model
            feature_loadings (np.ndarray): Principal components matrix with shape 
                                         (n_components, n_features)
            mean_centering_vector (np.ndarray): Mean values for centering with shape 
                                              (n_features,)
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model_name = model_name
        self.device = device
        
        # Convert numpy arrays to torch tensors
        self.encoder = PCAEncoder(
            torch.from_numpy(feature_loadings.copy()).float().to(self.device),
            torch.from_numpy(mean_centering_vector.copy()).float().to(self.device)
        )
        self.decoder = PCADecoder(
            torch.from_numpy(feature_loadings.copy()).float().to(self.device),
            torch.from_numpy(mean_centering_vector.copy()).float().to(self.device)
        )
        
        # Validate dimensions
        if feature_loadings.shape[1] != len(mean_centering_vector):
            raise ValueError("feature_loadings and mean_centering_vector dimensions don't match")
            
        super().__init__()
    
    def save(self, filepath: str) -> None:
        """
        Save the PCA embedding model to a file using pickle.
        
        Args:
            filepath (str): Path to save the model file
        """
        # Create directory if it doesn't exist
        filedir = '/'.join(filepath.split('/')[:-1])
        if not os.path.exists(filedir):
            os.makedirs(filedir)
            
        # Save model metadata
        model_data = {
            'model_name': self.model_name,
            'model_class': self.__class__,
            'feature_loadings': self.encoder.feature_loadings.cpu().numpy(),
            'mean_centering_vector': self.encoder.mean_centering_vector.cpu().numpy()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'PCAEmbeddingModel':
        """
        Load a PCA embedding model from a file.
        
        Args:
            filepath (str): Path to the saved model file
            
        Returns:
            PCAEmbeddingModel: Loaded PCA embedding model instance
        """
        # Load the full model state
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create new model instance
        model = cls(
            model_name=model_data['model_name'],
            feature_loadings=model_data['feature_loadings'],
            mean_centering_vector=model_data['mean_centering_vector']
        )
        
        return model
