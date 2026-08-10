from __future__ import annotations
from copy import deepcopy
from typing import Any, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
import pickle

import pandas as pd
import numpy as np
import torch

from .sc_perturbation import SingleCellPerturbation, MultiFileIndexMap
from ...transforms.base import Transform
from ...transforms.samplers import RowSampler
from ...utils import (
    build_covariate_to_control_dict,
    load_dataframe_from_h5,
)
from ...types import FrozenDictKeyMap


__all__ = ["SingleCellPerturbationWithControls"]


@dataclass
class SingleCellPerturbationWithControls(SingleCellPerturbation):
    """Single Cell Perturbation Dataset with Controls (H5 version).

    This dataset inherits from SingleCellPerturbation and contains both perturbed
    and control cells. Control cells are identified and mapped to each cell (both
    perturbed and control) based on matching covariate conditions. Unlike the inmemory
    version, this class uses lazy loading - control cells are not loaded into
    memory but fetched on demand using the same fetch_gene_expression method
    used for perturbed cells.

    Attributes:
        control_indices_dict: a dictionary that maps from each dataset index (both
          perturbed and control cells) to a list of control cell indices within the
          same dataset. This provides direct O(1) access to controls without covariate
          lookups. Control cells map to themselves or other controls with matching
          covariates.
        perturbed_indices: a list of indices identifying which cells in the dataset
          are perturbed (vs control). Control cells are those not in this list.
        controls_sampler: sampler for selecting control cells
        controls_transform: transform to apply to control cells
        num_control_samples: number of control samples to use per perturbed cell
    """

    control_indices_dict: dict[int, Sequence[int]] | None = None
    perturbed_indices: Sequence[int] | None = None
    controls_sampler: Transform | Callable | None = None
    controls_transform: Transform | Callable | None = None
    num_control_samples: int = 1

    def __post_init__(self, *args, **kwargs):
        super().__post_init__(*args, **kwargs)
        self.controls_sampler = RowSampler(num_samples=self.num_control_samples)
        # Initialize additional LRU caches for control methods (if caching enabled)
        if self.cache_size > 0:
            from .sc_perturbation import LRUCache
            self._control_gene_expression_cache = LRUCache(maxsize=self.cache_size)
            self._control_embeddings_cache = LRUCache(maxsize=self.cache_size)
        else:
            self._control_gene_expression_cache = None
            self._control_embeddings_cache = None
        
    @property
    def batch_controls_sampler(self) -> Callable | None:
        """Get batch control sampler."""
        try:
            batch_sampler = self.controls_sampler.batchify(
                collate_fn=lambda x: np.array(x),
            )
        except AttributeError:
            batch_sampler = self.controls_sampler
        return batch_sampler

    def set_transform(self, transform: Callable | None):
        # Separate out controls transforms to support control sampling
        if transform is not None and isinstance(transform, dict):
            transform = deepcopy(transform)
            self.controls_transform = transform.pop("controls", None)
        super().set_transform(transform)

    def fetch_control_gene_expression(self, control_indices: tuple[int, ...]) -> torch.Tensor:
        """
        Fetch and average gene expression for multiple control cell indices.
        
        Args:
            control_indices: tuple of control cell indices to fetch and average
            
        Returns:
            1D FloatTensor representing the averaged control expression
        """
        # Check LRU cache first (if caching enabled)
        if self._control_gene_expression_cache is not None:
            cached_result = self._control_gene_expression_cache.get(control_indices)
            if cached_result is not None:
                return cached_result
        
        if len(control_indices) == 1:
            result = self.fetch_gene_expression(control_indices[0])
        else:
            # Fetch all control expressions and average them
            control_expressions = []
            for idx in control_indices:
                control_expressions.append(self.fetch_gene_expression(idx))
            
            # Stack and average
            stacked_controls = torch.stack(control_expressions, dim=0)
            result = stacked_controls.mean(dim=0)
        
        # Cache the result using LRU policy (if caching enabled)
        if self._control_gene_expression_cache is not None:
            self._control_gene_expression_cache.put(control_indices, result)
        
        return result

    def fetch_control_embeddings(self, control_indices: tuple[int, ...], key: str) -> torch.Tensor:
        """
        Fetch and average embeddings for multiple control cell indices.
        
        Args:
            control_indices: tuple of control cell indices to fetch and average
            key: embedding key (e.g., "X_uce")
            
        Returns:
            1D FloatTensor representing the averaged control embeddings
        """
        # Create cache key combining control_indices and key
        cache_key = (control_indices, key)
        
        # Check LRU cache first (if caching enabled)
        if self._control_embeddings_cache is not None:
            cached_result = self._control_embeddings_cache.get(cache_key)
            if cached_result is not None:
                return cached_result
        
        if len(control_indices) == 1:
            result = self.fetch_obsm(control_indices[0], key)
        else:
            # Fetch all control embeddings and average them
            control_embeddings = []
            for idx in control_indices:
                control_embeddings.append(self.fetch_obsm(idx, key))
            
            # Stack and average
            stacked_embeddings = torch.stack(control_embeddings, dim=0)
            result = stacked_embeddings.mean(dim=0)
        
        # Cache the result using LRU policy (if caching enabled)
        if self._control_embeddings_cache is not None:
            self._control_embeddings_cache.put(cache_key, result)
        
        return result

    def _get_batch(self, indices):
        batch = super()._get_batch(indices)
        
        if self.control_indices_dict is None:
            # No controls available, return batch as-is
            return batch._replace(controls=None)

        # Get controls for each example using direct index lookup
        batch_control_indices = [np.array(self.control_indices_dict[idx]) for idx in indices]
        return batch._replace(controls=batch_control_indices)

    def __getitems__(self, indices):
        # Get batch with control indices
        batch_with_control_indices = self._get_batch(indices)

        if batch_with_control_indices.controls is None:
            # No controls, proceed with standard flow
            if self.transform is not None:
                batch_with_control_indices = self.transform(batch_with_control_indices)
            return batch_with_control_indices

        # Convert indices to batch with controls
        control_indices_list: list = batch_with_control_indices.controls
        if self.batch_controls_sampler is not None:
            # Sample n matching controls for each index
            control_indices_list = self.batch_controls_sampler(control_indices_list)
        
        # Fetch controls using lazy loading
        controls = []
        control_embeddings = []
        
        for control_indices in control_indices_list:
            # Convert to tuple for caching
            control_indices_tuple = tuple(control_indices)
            
            # Fetch averaged control gene expression
            control_expr = self.fetch_control_gene_expression(control_indices_tuple)
            controls.append(control_expr)
            
            # Fetch averaged control embeddings if available
            if self.embedding_key is not None:
                control_emb = self.fetch_control_embeddings(control_indices_tuple, self.embedding_key)
                control_embeddings.append(control_emb)
        
        # Stack controls and embeddings
        controls_tensor = torch.stack(controls, dim=0)
        control_embeddings_tensor = (
            torch.stack(control_embeddings, dim=0) 
            if control_embeddings else None
        )
        
        # Apply controls transform if available
        if self.controls_transform is not None:
            controls_tensor = self.controls_transform(controls_tensor)
            if control_embeddings_tensor is not None:
                control_embeddings_tensor = self.controls_transform(control_embeddings_tensor)

        # Update batch with controls
        batch = batch_with_control_indices._replace(controls=controls_tensor)
        if control_embeddings_tensor is not None:
            batch = batch._replace(control_embeddings=control_embeddings_tensor)

        # Apply main transform if available
        if self.transform is not None:
            batch = self.transform(batch)

        return batch

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
        num_control_samples: int = 1,
        cache_size: int = 1000,
        control_indices_dict_path: str | list[str] | None = None,
        append_dataset_to_index: bool = False,
    ) -> tuple[SingleCellPerturbationWithControls, dict[str, Any]]:
        """Create a SingleCellPerturbationWithControls dataset from one or more H5 files.

        Supports loading from multiple h5ad files with automatic feature alignment.
        When multiple files are provided, features are computed as the inner join
        of var_names across all files (unless feature_filter_path is provided).

        Args:
            adata_path: path to h5ad file(s). Can be a single path (str) or a list
              of paths for multi-file loading
            perturbation_key: the key in obs that contains the perturbations
            split: if specified, only use the subset specified by split indices.
              For multi-file datasets, these are global indices into the concatenated
              dataset
            perturbation_combination_delimiter: the delimiter used to separate
              perturbations in the perturbation_key
            covariate_keys: a list of keys in obs that contain the covariates
            perturbation_control_value: the value in obs[perturbation_key] that
              corresponds to control cells
            embedding_key: the key in obsm that contains the embeddings
            perturbation_embeddings_path: path to file containing pre-computed
              perturbation embeddings
            feature_filter_path: path to file containing subset of gene names to keep.
              If not provided and multiple files are used, features are computed as
              the inner join of var_names
            num_control_samples: number of control samples to use per perturbed cell
            cache_size: size of LRU caches for gene expression and embeddings.
              Set to 0 to disable caching entirely.
            control_indices_dict_path: optional path(s) to pickle file(s) containing
              dictionaries that map from cell IDs to lists of control cell IDs. Can be:
              - A single path (str): loaded directly (backward compatible)
              - A list of paths: one per h5ad file, loaded and merged. When
                append_dataset_to_index=True, cell IDs in each mapping are transformed
                by appending the dataset name suffix (e.g., "cell_id" -> "cell_id-dataset")
                to match the transformed cell IDs in the combined obs dataframe.
              If not provided, will be built from covariate mappings during dataset
              initialization.
            append_dataset_to_index: if True and control_indices_dict_path is a list,
              transform cell IDs in each mapping by appending '-{dataset_name}' to match
              the cell ID transformation applied by H5LitModule.accessor. Must match the
              append_dataset_to_index setting used in the data config.

        Returns:
            A tuple (dataset, info), where dataset is a SingleCellPerturbationWithControls
            dataset and info is a dictionary containing supplementary information
            about the dataset.

        Raises:
            ValueError: if perturbation_control_value is None, if required columns
              are missing from any file, if no common features exist, or if
              control_indices_dict_path list length doesn't match adata_path list length
        """
        if perturbation_control_value is None:
            raise ValueError("Must specify perturbation_control_value")
        
        # Normalize to list of paths
        if isinstance(adata_path, (str, Path)):
            adata_paths = [str(adata_path)]
        else:
            adata_paths = [str(p) for p in adata_path]
        
        # Load and combine observation dataframes from all files
        obs_dfs = []
        cell_counts = []
        for path in adata_paths:
            obs_df = load_dataframe_from_h5(path, 'obs')
            obs_dfs.append(obs_df)
            cell_counts.append(len(obs_df))
        
        # Create index map for multi-file access
        temp_index_map = MultiFileIndexMap.from_cell_counts(cell_counts)
        
        # Build combined obs dataframe with global indices
        combined_obs_dfs = []
        for i, obs_df in enumerate(obs_dfs):
            obs_df = obs_df.copy()
            obs_df['_file_idx'] = i
            obs_df['_global_idx'] = range(
                temp_index_map.file_boundaries[i],
                temp_index_map.file_boundaries[i + 1]
            )
            
            # Transform cell IDs by appending file index and dataset suffix if requested
            # This mirrors the transformation in H5LitModule.accessor
            # File index is needed to handle cases where multiple files share the same
            # dataset name (e.g., nadig24_hepg2 and nadig24_jurkat both have dataset='nadig24')
            if append_dataset_to_index:
                if 'dataset' not in obs_df.columns:
                    raise ValueError(
                        f"append_dataset_to_index=True requires a 'dataset' column in obs, "
                        f"but file '{adata_paths[i]}' does not have one. "
                        f"Available columns: {list(obs_df.columns)}"
                    )
                dataset_name = obs_df['dataset'].iloc[0]
                obs_df.index = obs_df.index.astype(str) + f'-f{i}-' + dataset_name
            
            combined_obs_dfs.append(obs_df)
        
        combined_obs = pd.concat(combined_obs_dfs, ignore_index=False)
        total_cells = len(combined_obs)
        
        if split is None:
            split = np.array(range(total_cells))
        elif isinstance(split, pd.Series):
            split = split.values
        else:
            split = np.array(split)
        
        # Split into perturbation and control cells (using combined obs)
        perturbed_mask = combined_obs[perturbation_key].values != perturbation_control_value
        
        # Create perturbation dataset using parent class
        dataset, info = SingleCellPerturbation.from_h5(
            adata_path=adata_paths,
            perturbation_key=perturbation_key,
            split=split,
            perturbation_combination_delimiter=perturbation_combination_delimiter,
            covariate_keys=covariate_keys,
            perturbation_control_value=perturbation_control_value,
            embedding_key=embedding_key,
            perturbation_embeddings_path=perturbation_embeddings_path,
            feature_filter_path=feature_filter_path,
            cache_size=cache_size,
        )
        
        # Identify control and perturbed cells in the split (using global indices)
        control_cell_split = set([i for i in split if not perturbed_mask[i]])
        control_cell_split_bool = np.array([i in control_cell_split for i in range(total_cells)])
        control_indices_global = np.where(control_cell_split_bool)[0]
        control_indices_global_set = set(control_indices_global)
        
        # Create mapping from global h5 index to dataset index
        inv_split_idx_map = {v: i for i, v in enumerate(dataset.split_idx_map)}
        global_idx_to_dataset_idx = {global_idx: inv_split_idx_map[global_idx] for global_idx in dataset.split_idx_map}
        
        # Identify perturbed indices in dataset index space
        perturbed_indices_list = [
            global_idx_to_dataset_idx[global_idx] 
            for global_idx in split 
            if perturbed_mask[global_idx] and global_idx in global_idx_to_dataset_idx
        ]
        
        # Build control_indices_dict
        if control_indices_dict_path is not None:
            # Load control indices from pickle file(s)
            if isinstance(control_indices_dict_path, (str, Path)):
                # Single file - load directly (backward compatible)
                with open(control_indices_dict_path, 'rb') as f:
                    cell_id_to_control_ids = pickle.load(f)
            else:
                # List of files - validate length and merge with optional dataset suffix transformation
                if len(control_indices_dict_path) != len(adata_paths):
                    raise ValueError(
                        f"control_indices_dict_path has {len(control_indices_dict_path)} entries "
                        f"but {len(adata_paths)} h5ad files provided. Must match 1:1."
                    )
                
                cell_id_to_control_ids = {}
                for i, (mapping_path, h5_path) in enumerate(zip(control_indices_dict_path, adata_paths, strict=True)):
                    # Load mapping for this file
                    with open(mapping_path, 'rb') as f:
                        file_mapping = pickle.load(f)
                    
                    if append_dataset_to_index:
                        # Get dataset name from obs (same logic as H5LitModule.accessor)
                        # We already loaded obs_dfs earlier, reuse them
                        if 'dataset' not in obs_dfs[i].columns:
                            raise ValueError(
                                f"append_dataset_to_index=True requires a 'dataset' column in obs, "
                                f"but file '{h5_path}' does not have one. "
                                f"Available columns: {list(obs_dfs[i].columns)}"
                            )
                        dataset_name = obs_dfs[i]['dataset'].iloc[0]
                        
                        # Transform cell IDs: append "-f{i}-{dataset_name}" to both keys and values
                        # File index is needed to match the index transformation above
                        for cell_id, control_ids in file_mapping.items():
                            new_cell_id = f"{cell_id}-f{i}-{dataset_name}"
                            new_control_ids = [f"{ctrl_id}-f{i}-{dataset_name}" for ctrl_id in control_ids]
                            cell_id_to_control_ids[new_cell_id] = new_control_ids
                    else:
                        # No transformation needed, just merge
                        cell_id_to_control_ids.update(file_mapping)
            
            # Create mapping from cell ID to global index
            cell_id_to_global_idx = {cell_id: idx for idx, cell_id in enumerate(combined_obs.index)}
            
            # Pre-filter: only process cells that are in the current split
            split_set = set(split)
            split_cell_ids = {combined_obs.index[global_idx] for global_idx in split_set}
            valid_control_cell_ids = set(combined_obs.index[list(control_indices_global_set)])
            
            # Filter cell_id_to_control_ids to only include cells in the split
            filtered_cell_id_to_control_ids = {
                cell_id: control_ids 
                for cell_id, control_ids in cell_id_to_control_ids.items() 
                if cell_id in split_cell_ids
            }
            
            # Build per-cell control mapping and validate
            control_indices_dict = {}
            for cell_id, control_cell_ids in filtered_cell_id_to_control_ids.items():
                global_idx = cell_id_to_global_idx[cell_id]
                
                if global_idx not in global_idx_to_dataset_idx:
                    continue
                
                # Convert control cell IDs to global indices, then to dataset indices
                control_global_indices = [
                    cell_id_to_global_idx[control_cell] 
                    for control_cell in control_cell_ids 
                    if control_cell in valid_control_cell_ids
                ]
                
                # Convert global indices to dataset indices
                control_dataset_indices = [
                    global_idx_to_dataset_idx[idx] 
                    for idx in control_global_indices 
                    if idx in global_idx_to_dataset_idx
                ]
                dataset_idx = global_idx_to_dataset_idx[global_idx]
                control_indices_dict[dataset_idx] = control_dataset_indices
        else:
            # Build from covariates
            control_obs_df = combined_obs.loc[control_cell_split_bool]
            
            # Filter to categorical covariates only using covariate_uniques from info
            # Continuous covariates have None values in covariate_uniques
            categorical_keys = [k for k in (covariate_keys or []) if info["covariate_uniques"].get(k) is not None]
            
            # Build mapping from covariate conditions to control global indices
            covariate_to_control_global_indices = build_covariate_to_control_dict(
                control_obs_df, covariate_keys
            )
            
            # Convert to covariate -> dataset indices mapping
            covariate_to_control_dataset_indices = FrozenDictKeyMap()
            for covariates, relative_indices in covariate_to_control_global_indices.items():
                control_global_idxs = control_indices_global[relative_indices]
                control_dataset_idxs = [
                    global_idx_to_dataset_idx[idx] 
                    for idx in control_global_idxs 
                    if idx in global_idx_to_dataset_idx
                ]
                covariate_to_control_dataset_indices[covariates] = control_dataset_idxs
            
            # Build per-cell control mapping using only categorical covariates
            control_indices_dict = {}
            for i in range(len(dataset)):
                covariates_dict = {k: dataset.covariates[k][i] for k in categorical_keys}
                
                if covariates_dict in covariate_to_control_dataset_indices:
                    control_indices_dict[i] = covariate_to_control_dataset_indices[covariates_dict]
                else:
                    raise ValueError(f"No controls found for covariate condition {covariates_dict}")
        
        # Filter out cells with no matched controls in the current split
        cells_without_controls = [
            idx for idx, controls in control_indices_dict.items() 
            if len(controls) == 0
        ]
        
        if cells_without_controls:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                f"Filtering out {len(cells_without_controls)} cells with no matched controls "
                f"in the current split. This may indicate a split/control mapping mismatch."
            )
            
            # Remove cells without controls from control_indices_dict
            for idx in cells_without_controls:
                del control_indices_dict[idx]
            
            # Create set of dataset indices to keep (those with controls)
            indices_to_keep = set(control_indices_dict.keys())
            
            # Update split_idx_map: filter to only keep indices with controls
            # split_idx_map maps dataset_idx -> global_h5_idx
            new_split_idx_map = [
                global_idx for i, global_idx in enumerate(dataset.split_idx_map) 
                if i in indices_to_keep
            ]
            
            # Create mapping from old dataset indices to new dataset indices
            old_to_new_idx = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted(indices_to_keep))}
            
            # Update control_indices_dict with new indices
            new_control_indices_dict = {}
            for old_idx in sorted(indices_to_keep):
                new_idx = old_to_new_idx[old_idx]
                # Also update control indices to use new indexing
                new_control_indices_dict[new_idx] = [
                    old_to_new_idx[ctrl_idx] for ctrl_idx in control_indices_dict[old_idx]
                    if ctrl_idx in old_to_new_idx
                ]
            control_indices_dict = new_control_indices_dict
            
            # Update perturbations
            new_perturbations = [
                dataset.perturbations[i] for i in range(len(dataset.perturbations))
                if i in indices_to_keep
            ]
            
            # Update covariates (keep as numpy arrays for range-based indexing support)
            new_covariates = {}
            for key, values in dataset.covariates.items():
                new_covariates[key] = np.array([values[i] for i in range(len(values)) if i in indices_to_keep])
            
            # Update perturbed_indices_list with new indices
            new_perturbed_indices_list = [
                old_to_new_idx[old_idx] for old_idx in perturbed_indices_list
                if old_idx in old_to_new_idx
            ]
            
            # Update dataset reference variables
            dataset_split_idx_map = new_split_idx_map
            dataset_perturbations = new_perturbations
            dataset_covariates = new_covariates
            perturbed_indices_list = new_perturbed_indices_list
        else:
            dataset_split_idx_map = dataset.split_idx_map
            dataset_perturbations = dataset.perturbations
            dataset_covariates = dataset.covariates
        
        # Build cell_ids from combined_obs using split_idx_map to guarantee alignment
        # cell_ids[i] corresponds to the cell at global index dataset_split_idx_map[i]
        cell_ids = combined_obs.index[dataset_split_idx_map].tolist()
        
        # Create dataset with controls, passing all multi-file attributes
        # Use filtered versions of split_idx_map, perturbations, covariates if cells were removed
        dataset_with_controls = SingleCellPerturbationWithControls(
            h5_file_paths=dataset.h5_file_paths,
            perturbations=dataset_perturbations,
            gene_names=dataset.gene_names,
            split_idx_map=dataset_split_idx_map,
            index_map=dataset.index_map,
            file_feature_masks=dataset.file_feature_masks,
            file_local_feature_masks=dataset.file_local_feature_masks,
            file_local_to_unified_indices=dataset.file_local_to_unified_indices,
            file_n_genes_original=dataset.file_n_genes_original,
            covariates=dataset_covariates,
            cell_ids=cell_ids,
            embedding_key=dataset.embedding_key,
            embedding_width=dataset.embedding_width,
            perturbation_embeddings=dataset.perturbation_embeddings,
            control_indices_dict=control_indices_dict,
            perturbed_indices=perturbed_indices_list,
            num_control_samples=num_control_samples,
            cache_size=cache_size,
            transform=None,
        )
        
        return dataset_with_controls, info
