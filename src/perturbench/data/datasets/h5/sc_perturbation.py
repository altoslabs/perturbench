from __future__ import annotations
from typing import Any, Callable, Sequence
from dataclasses import dataclass, InitVar
import os
import threading

import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset
import logging
import h5py

from ...transforms.base import Compose
from ...types import Batch
from ...utils import (
    map_perturbation_to_emb,
    LRUCache,
    MultiFileIndexMap,
    load_h5_metadata,
)


__all__ = ["SingleCellPerturbation", "MultiFileIndexMap"]

log = logging.getLogger(__name__)


@dataclass
class SingleCellPerturbation(Dataset):
    """Single Cell Perturbation Dataset.

    This dataset contains single cell gene expression data with perturbations
    and covariates. The gene expression data is represented as a sparse matrix
    and the perturbations and covariates are represented as numpy arrays. The
    perturbations and covariates are associated with each cell in the gene
    expression matrix.

    Supports loading from multiple h5ad files with automatic feature alignment.

    Attributes:
        h5_file_paths: list of paths to h5ad files (supports single file for backward compat)
        perturbations: a numpy array of size (n_cells, ) where each element is a
          list of perturbations applied to the cell
        covariates: a dictionary of covariates, where each covariate is a numpy
          array of size (n_cells, )
        cell_ids: a list of cell ids of length n_cells
        gene_names: a list of gene names of length n_genes (unified across files)
        transform: a function that transforms a SingleCellExpressionExample
        embeddings: a numpy array of size (n_cells, embedding_width) representing cell level embeddings
        perturbation_embeddings: a pandas DataFrame of size (n_perturbations, n_features) 
          containing pre-computed perturbation embeddings
        file_feature_masks: list of boolean tensors, one per file, indicating which 
          unified features exist in each file
        file_local_feature_masks: list of boolean tensors, one per file, indicating which
          local features map to the unified feature space
        file_local_to_unified_indices: list of tuples (local_indices, unified_indices) per file,
          for correctly mapping features from local to unified order
        index_map: MultiFileIndexMap for mapping global indices to (file_idx, local_idx)
        info: a dictionary containing metadata about the dataset initialization
    """

    h5_file_paths: list[str]  # List of paths to h5ad files
    perturbations: Sequence[list[str]]
    gene_names: Sequence[str]  # Unified gene names across all files
    split_idx_map: np.ndarray  # Maps dataset index to global h5 index
    index_map: MultiFileIndexMap  # Maps global h5 index to (file_idx, local_idx)
    file_feature_masks: list[torch.Tensor]  # Per-file: which unified features exist
    file_local_feature_masks: list[torch.Tensor]  # Per-file: which local features to use
    file_local_to_unified_indices: list[tuple[torch.Tensor, torch.Tensor]]  # Per-file: (local_idx, unified_idx) pairs
    file_n_genes_original: list[int]  # Per-file: original number of genes
    covariates: dict[str, Sequence[str]] | None = None
    cell_ids: Sequence[str] | None = None
    transform: InitVar[Callable | Sequence[Callable] | None] = None
    embedding_key: str | None = None
    embedding_width: int | None = None
    perturbation_embeddings: pd.DataFrame | None = None
    perturbation_control_value: str = "control"
    cache_size: int = 1000  # Size of LRU caches, 0 to disable caching
    _h5_file_cache: dict = None  # Thread-local file handle cache
    
    # Backward compatibility property
    @property
    def h5_file_path(self) -> str:
        """Return first file path for backward compatibility."""
        return self.h5_file_paths[0] if self.h5_file_paths else None
    
    @property
    def feature_bool_tensor(self) -> torch.Tensor:
        """Return first file's local feature mask for backward compatibility.
        
        Note: For multi-file datasets, this returns the first file's local mask.
        Use file_local_feature_masks[file_idx] for specific files.
        """
        return self.file_local_feature_masks[0] if self.file_local_feature_masks else None
    
    @property
    def unified_feature_mask(self) -> torch.Tensor:
        """Return a boolean mask for the unified feature space (all True).
        
        For multi-file datasets, gene_names represents the unified feature set,
        so this mask is all True with length n_genes.
        """
        return torch.ones(len(self.gene_names), dtype=torch.bool)
    
    def __post_init__(self, transform: Callable | Sequence[Callable] | None):
        try:
            # Input is a list of callables
            if len(transform) > 1:
                transform = Compose(transform)
            # Input is a list of a single callable
            else:
                transform = transform[0]
        # Input is a callable or None
        except TypeError:
            transform = transform

        self._transform = transform
        self._supports_example_mode = True
        self._supports_batch_mode = True
        # Initialize thread-local storage for H5 file handles
        self._h5_file_cache = {}
        self._lock = threading.Lock()
        # Initialize LRU caches (or None if caching disabled)
        if self.cache_size > 0:
            self._gene_expression_cache = LRUCache(maxsize=self.cache_size)
            self._obsm_cache = LRUCache(maxsize=self.cache_size)
        else:
            self._gene_expression_cache = None
            self._obsm_cache = None

    @property
    # pylint: disable-next=missing-function-docstring
    def transform(self) -> Callable | None:  # noqa: F811
        return self._transform

    def set_transform(self, transform: Callable | None):
        self._supports_example_mode = True
        self._supports_batch_mode = True
        
        # If transform is not None, make sure it is a valid transform
        if transform is not None:
            self._transform = None  # Avoid infinite recursion
            
            try:
                batch = self._get_batch(range(0, 3))
                transform(batch)
            except Exception as e:
                raise ValueError(
                    f"Transform {transform} is incompatible with this dataset. Failed batch mode validation with error: {e}"
                ) from e

        self._transform = transform
    
    @transform.setter
    def transform(self, transform: Callable | None):
        self.set_transform(transform)
    
    def _get_h5_file(self, file_idx: int = 0) -> h5py.File:
        """Get thread-safe H5 file handle for a specific file.
        
        Each worker process gets its own file handle to avoid 
        multiprocessing issues with shared file handles.
        
        Args:
            file_idx: Index of the file to open (default 0 for single-file compat)
            
        Returns:
            Open h5py.File handle for the specified file
        """
        thread_id = threading.get_ident()
        process_id = os.getpid()
        cache_key = f"{process_id}_{thread_id}_{file_idx}"
        
        with self._lock:
            if cache_key not in self._h5_file_cache:
                self._h5_file_cache[cache_key] = h5py.File(self.h5_file_paths[file_idx], 'r')
        
        return self._h5_file_cache[cache_key]
    
    def __getstate__(self):
        """Prepare state for pickling (e.g., when forking DataLoader workers).
        
        File handles cannot be pickled, so we exclude them from the state.
        Each worker process will open its own handles on demand.
        """
        state = self.__dict__.copy()
        state['_h5_file_cache'] = {}  # Don't carry file handles across processes
        # threading.Lock is not picklable
        state.pop('_lock', None)
        return state

    def __setstate__(self, state):
        """Restore state after unpickling (e.g., in a DataLoader worker)."""
        self.__dict__.update(state)
        self._lock = threading.Lock()

    def __del__(self):
        """Clean up H5 file handles and caches when dataset is destroyed."""
        # Clean up file handles
        if hasattr(self, '_h5_file_cache') and self._h5_file_cache:
            for h5_file in self._h5_file_cache.values():
                try:
                    h5_file.close()
                except Exception:
                    pass  # Ignore errors during cleanup
        
        # Clear LRU caches to free memory (if they exist)
        if hasattr(self, '_gene_expression_cache') and self._gene_expression_cache is not None:
            self._gene_expression_cache.clear()
        if hasattr(self, '_obsm_cache') and self._obsm_cache is not None:
            self._obsm_cache.clear()

    def __len__(self):
        return len(self.perturbations)


    @property
    def n_genes(self):
        """Number of genes in the unified gene expression matrix."""
        return len(self.gene_names)

    @property
    def n_genes_original(self):
        """Original number of genes in the first file (for backward compatibility)."""
        return self.file_n_genes_original[0] if self.file_n_genes_original else 0
    
    def _group_indices_by_file(self, indices) -> dict[int, list[tuple[int, int]]]:
        """Group batch indices by their source file.
        
        This optimization reduces file I/O in multifile mode by ensuring all reads
        from a single file happen together, improving OS page cache locality and
        HDF5 internal caching.
        
        Args:
            indices: List of dataset indices
            
        Returns:
            Dict mapping file_idx to list of (batch_position, dataset_idx) tuples
        """
        grouped = {}
        for batch_pos, idx in enumerate(indices):
            global_h5_idx = self.split_idx_map[idx]
            file_idx, _ = self.index_map.global_to_local(global_h5_idx)
            if file_idx not in grouped:
                grouped[file_idx] = []
            grouped[file_idx].append((batch_pos, idx))
        return grouped
    
    def _get_batch(self, indices):
        perturbations = [self.perturbations[i] for i in indices]
        if self.perturbation_embeddings is not None:
            perturbations = map_perturbation_to_emb(
                self.perturbation_embeddings, 
                perturbations
            )
        
        # Pre-allocate tensors more efficiently
        batch_size = len(indices)
        gene_expression = torch.empty(batch_size, len(self.gene_names), dtype=torch.float32)
        
        # Group indices by file to reduce file I/O in multifile mode
        grouped = self._group_indices_by_file(indices)
        
        # Fetch gene expressions grouped by file for better cache locality
        for file_idx in sorted(grouped.keys()):
            for batch_pos, idx in grouped[file_idx]:
                gene_expression[batch_pos] = self.fetch_gene_expression(idx)

        if self.embedding_key is not None:
            embeddings = torch.empty(batch_size, self.embedding_width, dtype=torch.float32)
            # Reuse the same grouping for embeddings
            for file_idx in sorted(grouped.keys()):
                for batch_pos, idx in grouped[file_idx]:
                    embeddings[batch_pos] = self.fetch_obsm(idx, self.embedding_key)
        else:
            embeddings = None

        return Batch(
            gene_expression=gene_expression,
            perturbations=perturbations,
            covariates=(
                {cov: value[indices] for cov, value in self.covariates.items()}
                if self.covariates is not None
                else None
            ),
            id=(
                [self.cell_ids[i] for i in indices]
                if self.cell_ids is not None
                else None
            ),
            gene_names=self.gene_names,
            embeddings=embeddings if embeddings is not None else None,
        )

    def __getitems__(self, indices):
        if not self._supports_batch_mode:
            raise ValueError("Dataset does not support batch mode.")
        batch = self._get_batch(indices)
        if self.transform is not None:
            batch = self.transform(batch)  # pylint: disable=not-callable

        return batch
    
    def fetch_gene_expression(self, idx: int) -> torch.Tensor:
        """
        Fetch raw gene counts for a given cell index.

        Supports both CSR‐encoded storage (via `encoding-type = "csr_matrix"`)
        and dense storage in the 'X' dataset. For multi-file datasets, handles
        zero-padding for features that don't exist in the source file.

        Args:
            idx: dataset index (will be mapped to global h5 index, then to file)
        Returns:
            1D FloatTensor of length self.n_genes (unified feature space)
        """
        global_h5_idx = self.split_idx_map[idx]
        
        # Check LRU cache first (if caching enabled)
        if self._gene_expression_cache is not None:
            cached_result = self._gene_expression_cache.get(global_h5_idx)
            if cached_result is not None:
                return cached_result
        
        # Map global index to file and local index
        file_idx, local_h5_idx = self.index_map.global_to_local(global_h5_idx)
        
        h5_file = self._get_h5_file(file_idx)
        n_genes_original = self.file_n_genes_original[file_idx]
        
        attrs = dict(h5_file["X"].attrs)
        if attrs["encoding-type"] == "csr_matrix":
            indptr = h5_file["/X/indptr"]
            start_ptr = indptr[local_h5_idx]
            end_ptr = indptr[local_h5_idx + 1]
            
            # Use numpy arrays first, then convert to torch to reduce memory overhead
            data_slice = h5_file["/X/data"][start_ptr:end_ptr]
            indices_slice = h5_file["/X/indices"][start_ptr:end_ptr]
            
            sub_data = torch.from_numpy(data_slice.astype(np.float32))
            sub_indices = torch.from_numpy(indices_slice.astype(np.int64))
            
            file_data = torch.zeros(n_genes_original, dtype=torch.float32)
            file_data[sub_indices] = sub_data
        else:
            row_data = h5_file["/X"][local_h5_idx]
            file_data = torch.from_numpy(row_data.astype(np.float32))
        
        # Map to unified feature space using index mapping
        # This correctly handles different feature orderings between files
        local_indices, unified_indices = self.file_local_to_unified_indices[file_idx]
        output = torch.zeros(self.n_genes, dtype=torch.float32)
        output[unified_indices] = file_data[local_indices]
        
        # Cache the result using LRU policy (if caching enabled)
        if self._gene_expression_cache is not None:
            self._gene_expression_cache.put(global_h5_idx, output)
        
        return output

    def fetch_obsm(self, idx: int, key: str) -> torch.Tensor:
        """
        Fetch a single row from the /obsm/{key} embedding matrix.

        Args:
            idx: dataset index (will be mapped to global h5 index, then to file)
            key: name of the obsm dataset (e.g. "X_uce", "X_hvg")
        Returns:
            1D FloatTensor of that embedding
        """
        # Create cache key combining global idx and key
        global_h5_idx = self.split_idx_map[idx]
        cache_key = (global_h5_idx, key)
        
        # Check LRU cache first (if caching enabled)
        if self._obsm_cache is not None:
            cached_result = self._obsm_cache.get(cache_key)
            if cached_result is not None:
                return cached_result
        
        # Map global index to file and local index
        file_idx, local_h5_idx = self.index_map.global_to_local(global_h5_idx)
        
        h5_file = self._get_h5_file(file_idx)
        row_data = h5_file[f"/obsm/{key}"][local_h5_idx]
        data = torch.from_numpy(row_data.astype(np.float32))
        
        # Cache the result using LRU policy (if caching enabled)
        if self._obsm_cache is not None:
            self._obsm_cache.put(cache_key, data)
        
        return data


    @staticmethod
    def from_h5(
        adata_path: str | list[str],
        perturbation_key: str,
        split: pd.Series | np.ndarray | None = None,
        perturbation_combination_delimiter: str | None = "+",
        covariate_keys: list[str] | None = None,
        perturbation_control_value: str | None = None,
        embedding_key: str | None = None,
        perturbation_embeddings_path: str | None = None,
        feature_filter_path: str | None = None,
        cache_size: int = 1000,
    ) -> tuple[SingleCellPerturbation, dict[str, Any]]:
        """Create a SingleCellPerturbation dataset from one or more h5ad files.

        Supports loading from multiple h5ad files with automatic feature alignment.
        When multiple files are provided:
        - Features are computed as the inner join (intersection) of var_names across
          all files, unless feature_filter_path is provided
        - Missing features in individual files are zero-padded
        - Required columns (perturbation_key, covariate_keys) are validated across
          all files before loading

        Args:
            adata_path: path to h5ad file(s). Can be a single path (str) or a list
              of paths for multi-file loading
            perturbation_key: the key in obs that contains the perturbations
            split: indices of cells to include. For multi-file datasets, these are
              global indices into the concatenated dataset
            perturbation_combination_delimiter: the delimiter used to separate
              perturbations in the perturbation_key
            covariate_keys: a list of keys in obs that contain the covariates
            perturbation_control_value: the value in obs[perturbation_key] that
              corresponds to control cells
            embedding_key: the key in obsm that contains the embeddings
            perturbation_embeddings_path: the path to the file containing the
              pre-computed perturbation embeddings
            feature_filter_path: the path to the file containing a subset of
              gene names to keep. If not provided and multiple files are used,
              features are computed as the inner join of var_names
            cache_size: size of LRU caches for gene expression and embeddings.
              Set to 0 to disable caching entirely.
            
        Returns:
            A tuple (dataset, info), where dataset is a SingleCellPerturbation
              dataset and info is a dictionary containing supplementary
              information about the dataset not contained in the dataset itself
              (e.g. perturbation and covariate unique values). This information
              can be used to setup data pipelines.
              
        Raises:
            ValueError: if perturbation_control_value is None, if required columns
              are missing from any file, or if no common features exist
        """
        # Load common metadata using shared helper
        meta = load_h5_metadata(
            adata_path=adata_path,
            perturbation_key=perturbation_key,
            split=split,
            perturbation_combination_delimiter=perturbation_combination_delimiter,
            covariate_keys=covariate_keys,
            perturbation_control_value=perturbation_control_value,
            perturbation_embeddings_path=perturbation_embeddings_path,
            feature_filter_path=feature_filter_path,
        )

        # Create supplementary info
        info = dict(
            perturbation_uniques=meta.perturbation_uniques,
            covariate_uniques=meta.covariate_uniques,
            perturbation_key=perturbation_key,
            covariate_keys=covariate_keys,
            perturbation_combination_delimiter=perturbation_combination_delimiter,
            perturbation_control_value=perturbation_control_value,
            perturbation_counts=meta.perturbation_counts,
            features_keep=np.arange(len(meta.unified_features)),
            num_files=len(meta.adata_paths),
        )

        # Create dataset
        dataset = SingleCellPerturbation(
            h5_file_paths=meta.adata_paths,
            perturbations=meta.perturbations,
            gene_names=meta.unified_features,
            split_idx_map=meta.split_idx_map,
            index_map=meta.index_map,
            file_feature_masks=meta.file_feature_masks,
            file_local_feature_masks=meta.file_local_feature_masks,
            file_local_to_unified_indices=meta.file_local_to_unified_indices,
            file_n_genes_original=meta.file_n_genes_original,
            covariates=meta.covariates,
            cell_ids=meta.combined_obs.index[meta.split_bool].to_list(),
            embedding_key=embedding_key,
            perturbation_embeddings=meta.perturbation_embeddings,
            perturbation_control_value=perturbation_control_value,
            cache_size=cache_size,
            transform=None,
        )
        
        if embedding_key is not None:
            sample_embedding = dataset.fetch_obsm(0, embedding_key)
            dataset.embedding_width = sample_embedding.shape[0]
        
        return (
            dataset,
            info,
        )
