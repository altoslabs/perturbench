from __future__ import annotations
import bisect
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence

import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
from pandas.api.types import is_numeric_dtype

from .types import FrozenDictKeyMap
import torch
import h5py
import warnings
import threading
from collections import OrderedDict
from torch.utils.data import get_worker_info


class LRUCache:
    """Thread-safe LRU cache implementation."""

    def __init__(self, maxsize: int = 10000):
        self.maxsize = maxsize
        self.cache = OrderedDict()
        self.lock = threading.Lock()

    def get(self, key):
        """Get item from cache, return None if not found."""
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                value = self.cache.pop(key)
                self.cache[key] = value
                return value
            return None

    def put(self, key, value):
        """Put item in cache, evicting LRU item if necessary."""
        with self.lock:
            if key in self.cache:
                # Update existing item and move to end
                self.cache.pop(key)
            elif len(self.cache) >= self.maxsize:
                # Remove least recently used item (first item)
                self.cache.popitem(last=False)

            # Add new item at end (most recently used)
            self.cache[key] = value

    def clear(self):
        """Clear the cache."""
        with self.lock:
            self.cache.clear()

    def __len__(self):
        return len(self.cache)


def parse_perturbation_combinations(
        combinations: pd.Series,
        delimiter: str | None = "+",
        control_value: str | None = "control",
) -> tuple[ArrayLike, set[str], pd.DataFrame]:
    """Get all perturbations applied to each cell.

    Args:
        combinations: combined perturbations string representation of size (n_cells, )
        delimiter: a string that separates individual perturbations
        control_value: a string that represents the control perturbation

    Returns:
        A tuple (combinations, unique_perturbations), where combinations is an
          (n_cell, ) array of lists of individual perturbations applied to each
          cell, and unique_perturbations is a set of all unique perturbations.
    """
    combinations = combinations.astype('category')
    combinations = combinations.cat.remove_unused_categories()

    # If there is a control value, then remove it from the combined perturbations
    if control_value is not None:
        combinations = combinations.replace(control_value, pd.NA)
    # If there is no delimiter, then the combined perturbations are the unique
    # perturbations
    if delimiter is None:
        parsed = combinations.values.add_categories("").fillna("").to_list()
        uniques = combinations.cat.categories.values
    # Otherwise, split the perturbations by the delimiter
    else:
        parsed = combinations.str.split(delimiter)
        # Replace control locations (nan valued) with empty lists
        parsed = [x if isinstance(x, list) else [] for x in parsed]
        unique_combinations = combinations.cat.categories.str.split(delimiter)

        perturbation_counts = {}
        uniques = []
        # first loop is to make sure the order of perturbations in uniques is the same (so compatible) to older runs
        for combination in unique_combinations:
            for perturbation in combination:
                if perturbation not in perturbation_counts:
                    uniques.append(perturbation)
                    perturbation_counts[perturbation] = 0

        # second loop is for actually counting perturbations
        for combination in parsed:
            for perturbation in combination:
                perturbation_counts[perturbation] += 1

    perturbation_counts = pd.Series(perturbation_counts, index=perturbation_counts.keys())
    return parsed, uniques, perturbation_counts.loc[uniques]


def combine_perturbations(
        parsed_perturbations: list[list[str]],
        delimiter: str | None = "+",
        perturbation_control_value: str | None = "control",
) -> pd.Series:
    """Restore the combined perturbations from a list of perturbations using a specified delimiter and control value

    Args:
        parsed_perturbations: a list of lists of perturbations
        delimiter: a string that separates individual perturbations
        perturbation_control_value: a string that represents the control perturbation

    Returns:
        A pandas Series of combined perturbations
    """
    combined_perturbations = []
    for combined_perts in parsed_perturbations:
        assert isinstance(combined_perts, list)
        pert_joined = delimiter.join(combined_perts)
        if pert_joined == "":
            pert = perturbation_control_value
        else:
            pert = pert_joined
        combined_perturbations.append(pert)

    combined_perturbations = pd.Series(combined_perturbations, dtype="category")
    return combined_perturbations


def get_covariates(df: pd.DataFrame, covariate_keys: list[str]) -> tuple[dict, dict]:
    """Get covariates from a dataframe.

    Args:
        df: a dataframe containing covariates for each cell with n_cells rows
        covariate_keys: a list of covariate keys in the dataframe

    Returns:
        A tuple (covariates, covariate_unique_values), where covariates is a
          dictionary of covariate keys to covariate values with n_cells rows,
          and covariate_unique_values is a dictionary of covariate keys to
          unique covariate values. For numeric (continuous) covariates, the
          unique values will be None.

    Raises:
        KeyError: if a covariate key is not found in the dataframe.
        ValueError: if a covariate column cannot be processed.
    """
    try:
        # Build covariate_unique_values dict, detecting numeric vs categorical
        covariate_unique_values = {}
        for cov in covariate_keys:
            if cov not in df.columns:
                raise KeyError(f"Covariate key '{cov}' not found in dataframe columns: {list(df.columns)}")

            col = df[cov]
            if is_numeric_dtype(col):
                covariate_unique_values[cov] = None
            else:
                covariate_unique_values[cov] = list(col.unique())

        covariates: np.ndarray = df[covariate_keys].values

    except Exception as e:
        raise ValueError(
            f"Error processing covariates {covariate_keys}. "
            f"This may occur if covariate columns have mixed or incompatible types. "
            f"Original error: {e}"
        ) from e

    return dict(zip(covariate_keys, covariates.T, strict=True)), covariate_unique_values


def build_covariate_to_control_dict(
        controls_df: pd.DataFrame,
        covariate_keys: list[str] | None = None,
) -> FrozenDictKeyMap:
    """Build a map from covariate conditions to control cells.

    Args:
        controls_df: a dataframe containing the controls cells and their covariates
        covariate_keys: a list of column keys that contain the covariates. If
          None, use all columns (default: None)

    Returns:
        A dictionary mapping covariate conditions to control cell indices.
        Only categorical (non-numeric) covariates are used for matching;
        continuous covariates are ignored.
    """
    if covariate_keys is None:
        covariate_keys = controls_df.columns.tolist()

    # Filter out continuous covariates - only use categorical for matching
    categorical_keys = [k for k in covariate_keys if not is_numeric_dtype(controls_df[k])]

    covariate_to_controls_map = FrozenDictKeyMap()

    if not categorical_keys:
        # No categorical covariates - all cells share the same control pool
        covariate_to_controls_map[{}] = np.arange(len(controls_df))
        return covariate_to_controls_map

    grouped = controls_df.groupby(list(categorical_keys), observed=True)  # groupby requires a list
    for group_key, group_indices in grouped.indices.items():
        if len(categorical_keys) == 1:
            assert isinstance(group_key, (str, int))
            group_key = (group_key,)
        key = dict(zip(categorical_keys, group_key, strict=True))
        covariate_to_controls_map[key] = group_indices

    return covariate_to_controls_map


def map_perturbation_to_emb(
        perturbation_embeddings: pd.DataFrame,
        perturbations: List[List[str]],
) -> torch.Tensor:
    emb_width = len(perturbation_embeddings.columns)
    loaded_perturbation_embeddings = []

    for pert_names in perturbations:

        perturbations_ = []
        for pert_name in pert_names:
            pert_values = perturbation_embeddings.loc[pert_name].values
            if pert_values.ndim > 1:
                pert_values = pert_values[0]
            perturbations_.append(pert_values)

        if len(perturbations_) > 0:
            # in latent_additive, since datamodule is always SCPWC,
            # the cells loaded here will always be perturbed cells
            # todo, in case of multiple perturbation, find out better way to aggregate them
            # let model handle the aggregation — using transformer, Sebastian had this idea
            loaded_perturbation_embeddings.append(np.array(perturbations_).sum(axis=0))  # additive
        else:
            # control cells
            # todo, perturbations[i] can be empty, in the case of vae models
            loaded_perturbation_embeddings.append(np.zeros(emb_width))

    loaded_perturbation_embeddings = torch.as_tensor(np.array(loaded_perturbation_embeddings).astype(np.float32))
    return loaded_perturbation_embeddings


def build_covariate_to_weight_dict(
        covariates_df: pd.DataFrame,
        oversample_root: float = 2.0,
):
    weights_dictionary = {}
    covariates_df['concatenated'] = covariates_df.apply(
        lambda row: '_'.join(row.values.astype(str)), axis=1
    )
    covariate_fractions = covariates_df['concatenated'].value_counts() / covariates_df.shape[0]
    for covariate, frac in covariate_fractions.items():
        weight = (1 / frac) ** (1.0 / oversample_root)
        weights_dictionary[covariate] = weight

    return weights_dictionary  # returns empty dictionary if covariates_df is empty (i.e., dataset.covariates is None)


def safe_decode_array(arr) -> np.ndarray:
    """
    Decode any byte-strings in `arr` to UTF-8 and cast all entries to Python str.

    Args:
        arr: array-like of bytes or other objects
    Returns:
        np.ndarray[str]: decoded strings
    """
    decoded = []
    for x in arr:
        if isinstance(x, (bytes, bytearray)):
            # decode bytes, ignoring errors
            decoded.append(x.decode("utf-8", errors="ignore"))
        else:
            decoded.append(str(x))
    return np.array(decoded, dtype=str)


def load_dataframe_from_h5(h5_file_path: str, df_key: str, cols_load: list[str] | None = None) -> pd.DataFrame:
    df = pd.DataFrame()
    with h5py.File(h5_file_path, 'r') as h5_file:
        if cols_load is None:
            cols_load = list(h5_file[df_key].attrs['column-order'])

        # Filter cols_load to only include columns that actually exist in the file
        actual_keys = set(h5_file[df_key].keys())
        missing_cols = [c for c in cols_load if c not in actual_keys]
        if missing_cols:
            warnings.warn(
                f"Columns listed in 'column-order' but missing from {h5_file_path}: {missing_cols}. "
                "These will be skipped.", stacklevel=1
            )
            cols_load = [c for c in cols_load if c in actual_keys]

        index_col = h5_file[df_key].attrs['_index']
        if index_col not in cols_load:
            cols_load.append(index_col)

        for k in cols_load:
            if isinstance(h5_file[df_key][k], h5py.Group):
                try:
                    codes = h5_file[df_key][k]['codes'][:].astype(np.int32)
                    cats = safe_decode_array(h5_file[df_key][k]['categories'][:])
                    df[k] = pd.Categorical.from_codes(codes, cats)
                except ValueError:
                    warnings.warn(f"Column {k} could not be loaded from {h5_file_path}", stacklevel=1)
            else:
                df[k] = h5_file[df_key][k][:]

    # Decode any byte string columns to UTF-8 strings
    for col in df.columns:
        if df[col].dtype == 'object':
            # Check if the column contains byte strings
            sample_val = df[col].iloc[0] if len(df) > 0 else None
            if isinstance(sample_val, (bytes, bytearray)):
                df[col] = df[col].apply(
                    lambda x: x.decode('utf-8', errors='ignore') if isinstance(x, (bytes, bytearray)) else str(x))

    if index_col in df.columns:
        df.set_index(index_col, inplace=True, drop=True)

    return df


def validate_columns_across_files(
        adata_paths: list[str],
        perturbation_key: str,
        covariate_keys: list[str] | None = None,
) -> None:
    """Validate that all required columns exist in all h5ad files.

    Args:
        adata_paths: list of paths to h5ad files
        perturbation_key: the column name containing perturbation information
        covariate_keys: list of column names containing covariates

    Raises:
        ValueError: if adata_paths is empty or any required column is missing from any file
        FileNotFoundError: if any path does not exist
    """
    if not adata_paths:
        raise ValueError("adata_paths must not be empty")

    required_cols = [perturbation_key] + (covariate_keys or [])

    for file_idx, path in enumerate(adata_paths):
        if not Path(path).exists():
            raise FileNotFoundError(
                f"h5ad file not found: {path}"
                + (f" (file index {file_idx})" if len(adata_paths) > 1 else "")
            )
        # Check available columns without loading data by reading attrs
        with h5py.File(path, 'r') as h5_file:
            available_cols = set(h5_file['obs'].attrs['column-order'])
            index_col = h5_file['obs'].attrs['_index']
            available_cols.add(index_col)

        missing = [c for c in required_cols if c not in available_cols]
        if missing:
            raise ValueError(
                f"File {path} is missing required columns: {missing}. "
                f"Available columns: {sorted(available_cols)}"
            )


def compute_common_features(adata_paths: list[str], var_dfs: list[pd.DataFrame] | None = None) -> list[str]:
    """Compute inner join of var_names (features) across all h5ad files.

    Args:
        adata_paths: list of paths to h5ad files
        var_dfs: optional list of already-loaded var dataframes (for efficiency)

    Returns:
        List of feature names that exist in all files, preserving first file's order

    Raises:
        ValueError: if no common features are found across all files
    """
    # Load var dataframes if not provided
    if var_dfs is None:
        var_dfs = [load_dataframe_from_h5(path, 'var') for path in adata_paths]

    # Compute intersection of all feature sets
    common_features_set = None
    for var_df in var_dfs:
        features = set(var_df.index)
        if common_features_set is None:
            common_features_set = features
        else:
            common_features_set = common_features_set & features

    if not common_features_set:
        raise ValueError(
            "No common features found across all h5ad files. "
            "Please ensure all files share at least some feature names, "
            "or provide a feature_filter_path with the desired features."
        )

    # Return in first file's order (for backward compatibility)
    return [f for f in var_dfs[0].index if f in common_features_set]


def compute_unified_features(
        adata_paths: list[str],
        feature_filter_path: str | None = None,
        var_dfs: list | None = None,
) -> list[str]:
    """Compute unified features for train/test alignment.

    This function ensures consistent feature selection across training and
    evaluation datasets.

    When feature_filter_path is provided:
      - Computes the union of var_names across ALL h5ad files
      - Returns intersection of CSV features with that union
      - Preserves CSV file order

    When feature_filter_path is None:
      - Returns inner join of var_names across all files (delegates to compute_common_features)

    Args:
        adata_paths: list of paths to h5ad files
        feature_filter_path: optional path to CSV file containing features to keep
        var_dfs: optional list of already-loaded var dataframes (for efficiency)

    Returns:
        List of feature names. With filter: CSV order. Without filter: first file's order.

    Raises:
        ValueError: if no common features are found
    """
    # Load var dataframes if not provided
    if var_dfs is None:
        var_dfs = [load_dataframe_from_h5(path, 'var') for path in adata_paths]

    if feature_filter_path is not None:
        # Load features from CSV filter, preserving CSV order
        features_keep_df = pd.read_csv(feature_filter_path, index_col=False, header=None)
        features_keep_list = list(features_keep_df.values.flatten())

        # Compute union of all files' var_names
        all_features_union = set()
        for var_df in var_dfs:
            all_features_union.update(var_df.index)

        # Intersect CSV features with union, preserving CSV order
        unified_features = [f for f in features_keep_list if f in all_features_union]

        if not unified_features:
            raise ValueError(
                f"No features from feature_filter_path found in any h5ad file. "
                f"Filter has {len(features_keep_list)} features, "
                f"union of all files has {len(all_features_union)} features."
            )

        return unified_features
    else:
        # No filter provided - compute inner join across all files
        return compute_common_features(adata_paths, var_dfs)


@dataclass
class MultiFileIndexMap:
    """Maps global indices to (file_idx, local_idx) pairs for multi-file datasets.

    Attributes:
        file_boundaries: Cumulative cell counts [0, n1, n1+n2, ...] where n_i is
            the number of cells in file i. The length is num_files + 1.
    """
    file_boundaries: list[int] = field(default_factory=list)

    def __post_init__(self):
        # Ensure boundaries start with 0
        if not self.file_boundaries or self.file_boundaries[0] != 0:
            if self.file_boundaries:
                self.file_boundaries = [0] + list(self.file_boundaries)

    @classmethod
    def from_cell_counts(cls, cell_counts: list[int]) -> "MultiFileIndexMap":
        """Create index map from a list of cell counts per file."""
        boundaries = [0]
        for count in cell_counts:
            boundaries.append(boundaries[-1] + count)
        return cls(file_boundaries=boundaries)

    def global_to_local(self, global_idx: int) -> tuple[int, int]:
        """Convert a global index to (file_idx, local_idx) pair.

        Args:
            global_idx: The global index across all files (must satisfy
                0 <= global_idx < total_cells).

        Returns:
            Tuple of (file_idx, local_idx) where file_idx is the file index
            and local_idx is the index within that file.

        Raises:
            IndexError: if global_idx is out of range [0, total_cells).
        """
        if global_idx < 0 or global_idx >= self.total_cells:
            raise IndexError(
                f"global_idx {global_idx} is out of range [0, {self.total_cells})"
            )
        file_idx = bisect.bisect_right(self.file_boundaries, global_idx) - 1
        local_idx = global_idx - self.file_boundaries[file_idx]
        return file_idx, local_idx

    @property
    def total_cells(self) -> int:
        """Total number of cells across all files."""
        return self.file_boundaries[-1] if self.file_boundaries else 0

    @property
    def num_files(self) -> int:
        """Number of files."""
        return len(self.file_boundaries) - 1 if self.file_boundaries else 0


@dataclass
class H5LoadedMetadata:
    """Container for metadata loaded from H5 files.

    This dataclass holds all the common metadata extracted during H5 file loading

    Attributes:
        adata_paths: List of paths to h5ad files
        obs_dfs: List of obs DataFrames, one per file
        var_dfs: List of var DataFrames, one per file
        combined_obs: Combined obs DataFrame with global indices
        index_map: MultiFileIndexMap for global-to-local index mapping
        split_idx_map: Array mapping dataset indices to global h5 indices
        split_bool: Boolean array indicating which cells are in the split
        unified_features: List of unified gene names across all files
        file_feature_masks: Per-file boolean tensors for which unified features exist
        file_local_feature_masks: Per-file boolean tensors for which local features map to unified
        file_local_to_unified_indices: Per-file tuple of (local_indices, unified_indices) for correct feature mapping
        file_n_genes_original: Per-file original gene counts
        perturbations: Parsed perturbation lists per cell
        perturbation_uniques: Set of unique perturbations
        perturbation_counts: DataFrame of perturbation counts
        covariates: Dictionary of covariate name to values per cell
        covariate_uniques: Dictionary of covariate name to unique values
        perturbation_embeddings: Optional DataFrame of perturbation embeddings
    """
    adata_paths: list[str]
    obs_dfs: list[pd.DataFrame]
    var_dfs: list[pd.DataFrame]
    combined_obs: pd.DataFrame
    index_map: MultiFileIndexMap
    split_idx_map: np.ndarray
    split_bool: np.ndarray
    unified_features: list[str]
    file_feature_masks: list[torch.Tensor]
    file_local_feature_masks: list[torch.Tensor]
    file_local_to_unified_indices: list[tuple[torch.Tensor, torch.Tensor]]
    file_n_genes_original: list[int]
    perturbations: Sequence[list[str]]
    perturbation_uniques: set[str]
    perturbation_counts: pd.DataFrame
    covariates: dict[str, Sequence]
    covariate_uniques: dict[str, list]
    perturbation_embeddings: pd.DataFrame | None


def load_h5_metadata(
        adata_path: str | list[str],
        perturbation_key: str,
        split: pd.Series | np.ndarray | None = None,
        perturbation_combination_delimiter: str | None = "+",
        covariate_keys: list[str] | None = None,
        perturbation_control_value: str | None = None,
        perturbation_embeddings_path: str | None = None,
        feature_filter_path: str | None = None,
) -> H5LoadedMetadata:
    """Load and process metadata from one or more h5ad files.

    This function extracts common loading logic used by SingleCellPerturbation
    factory methods.

    Args:
        adata_path: Path to h5ad file(s). Can be a single path or list of paths.
        perturbation_key: The key in obs that contains the perturbations.
        split: Indices of cells to include. For multi-file datasets, these are
            global indices into the concatenated dataset.
        perturbation_combination_delimiter: Delimiter for combined perturbations.
        covariate_keys: List of keys in obs that contain the covariates.
        perturbation_control_value: Value in obs[perturbation_key] that corresponds
            to control cells.
        perturbation_embeddings_path: Path to file containing pre-computed
            perturbation embeddings.
        feature_filter_path: Path to file containing a subset of gene names to keep.

    Returns:
        H5LoadedMetadata containing all extracted metadata.

    Raises:
        ValueError: If perturbation_control_value is None, if required columns
            are missing from any file, or if no common features exist.
    """
    # Normalize to list of paths
    if isinstance(adata_path, (str, Path)):
        adata_paths = [str(adata_path)]
    else:
        adata_paths = [str(p) for p in adata_path]

    if not adata_paths:
        raise ValueError("At least one h5ad file path must be provided")

    # Add random delay to reduce contention between dataloader workers
    worker_info = get_worker_info()
    if worker_info is not None:
        worker_seed = worker_info.id
    else:
        worker_seed = os.getpid()

    _rng = random.Random(worker_seed)
    delay = _rng.uniform(0, 2)
    time.sleep(delay)

    if covariate_keys is None:
        covariate_keys = []

    if perturbation_control_value is None:
        raise ValueError("Must specify perturbation_control_value")

    # Validate required columns exist in all files
    validate_columns_across_files(adata_paths, perturbation_key, covariate_keys)

    # Load obs and var dataframes from all files
    obs_dfs = []
    var_dfs = []
    cell_counts = []

    for path in adata_paths:
        obs_df = load_dataframe_from_h5(path, 'obs')
        var_df = load_dataframe_from_h5(path, 'var')
        obs_dfs.append(obs_df)
        var_dfs.append(var_df)
        cell_counts.append(len(obs_df))

    # Create index map for multi-file access
    index_map = MultiFileIndexMap.from_cell_counts(cell_counts)

    # Determine unified feature set
    unified_features = compute_unified_features(adata_paths, feature_filter_path, var_dfs)
    unified_features_set = set(unified_features)

    # Build per-file feature masks and index mappings
    file_feature_masks = []  # Which unified features exist in each file (kept for backward compat)
    file_local_feature_masks = []  # Which local features map to unified space (kept for backward compat)
    file_local_to_unified_indices = []  # Per-file: (local_indices, unified_indices) for correct mapping
    file_n_genes_original = []

    # Build unified feature name -> index lookup once
    unified_name_to_idx = {name: idx for idx, name in enumerate(unified_features)}

    for var_df in var_dfs:
        file_features = var_df.index.values
        file_n_genes_original.append(len(file_features))

        # Which local features are in the unified set
        local_mask = np.array([f in unified_features_set for f in file_features], dtype=bool)
        file_local_feature_masks.append(torch.from_numpy(local_mask))

        # Which unified features exist in this file
        file_features_in_unified = set(file_features) & unified_features_set
        if not file_features_in_unified:
            raise ValueError(
                "File has no features matching the unified feature set. "
                "Please check that all files share common feature names."
            )

        unified_mask = np.array([f in file_features_in_unified for f in unified_features], dtype=bool)
        file_feature_masks.append(torch.from_numpy(unified_mask))

        # Build index mapping: local_idx -> unified_idx
        # This correctly handles different feature orderings between files
        local_to_unified = []
        for local_idx, feature_name in enumerate(file_features):
            if feature_name in unified_name_to_idx:
                local_to_unified.append((local_idx, unified_name_to_idx[feature_name]))

        # Store as tensor pair (local_indices, unified_indices) for efficient indexing
        local_indices = torch.tensor([x[0] for x in local_to_unified], dtype=torch.long)
        unified_indices = torch.tensor([x[1] for x in local_to_unified], dtype=torch.long)
        file_local_to_unified_indices.append((local_indices, unified_indices))

    # Load embeddings for perturbations
    if perturbation_embeddings_path is not None:
        perturbation_embeddings = pd.read_parquet(perturbation_embeddings_path)
    else:
        perturbation_embeddings = None

    # Create combined obs dataframe with global indices
    combined_obs_dfs = []
    for i, obs_df in enumerate(obs_dfs):
        obs_df = obs_df.copy()
        obs_df['_file_idx'] = i
        obs_df['_global_idx'] = range(
            index_map.file_boundaries[i],
            index_map.file_boundaries[i + 1]
        )
        combined_obs_dfs.append(obs_df)

    combined_obs = pd.concat(combined_obs_dfs, ignore_index=False)
    total_cells = len(combined_obs)

    # Handle split
    if split is None:
        split_set = set(range(total_cells))
        split_bool = np.ones(total_cells, dtype=bool)
    else:
        if isinstance(split, pd.Series):
            split = split.values
        split_set = set(split)
        split_bool = np.array([i in split_set for i in range(total_cells)], dtype=bool)

    # Map subsetted indices to global indices
    split_idx_map = np.where(split_bool)[0]

    # Parse perturbations
    perturbations, perturbation_uniques, perturbation_counts = parse_perturbation_combinations(
        combined_obs.loc[split_bool, perturbation_key],
        perturbation_combination_delimiter,
        perturbation_control_value,
    )

    # Filter by perturbation embeddings if provided
    if perturbation_embeddings is not None:
        index_set = set(perturbation_embeddings.index)
        valid_indices = []
        for i, pert_names in enumerate(perturbations):
            is_valid_example = all([pert_name in index_set for pert_name in pert_names])
            if is_valid_example:
                valid_indices.append(i)

        if len(valid_indices) < len(perturbations):
            original_indices = np.where(split_bool)[0]
            valid_original_indices = original_indices[np.array(valid_indices)]
            split_bool = np.zeros(total_cells, dtype=bool)
            split_bool[valid_original_indices] = True
            split_idx_map = np.where(split_bool)[0]

            # Re-parse perturbations for valid subset
            perturbations, perturbation_uniques, perturbation_counts = parse_perturbation_combinations(
                combined_obs.loc[split_bool, perturbation_key],
                perturbation_combination_delimiter,
                perturbation_control_value,
            )

    # Get covariates
    covariates, covariate_uniques = get_covariates(combined_obs.loc[split_bool], covariate_keys)

    return H5LoadedMetadata(
        adata_paths=adata_paths,
        obs_dfs=obs_dfs,
        var_dfs=var_dfs,
        combined_obs=combined_obs,
        index_map=index_map,
        split_idx_map=split_idx_map,
        split_bool=split_bool,
        unified_features=unified_features,
        file_feature_masks=file_feature_masks,
        file_local_feature_masks=file_local_feature_masks,
        file_local_to_unified_indices=file_local_to_unified_indices,
        file_n_genes_original=file_n_genes_original,
        perturbations=perturbations,
        perturbation_uniques=perturbation_uniques,
        perturbation_counts=perturbation_counts,
        covariates=covariates,
        covariate_uniques=covariate_uniques,
        perturbation_embeddings=perturbation_embeddings,
    )
