from __future__ import annotations
from typing import Any, Callable, Sequence
from dataclasses import dataclass, InitVar, field
from pathlib import Path
import random
import anndata as ad
import pandas as pd
import torch
from torch.utils.data import Dataset
from scipy.sparse import vstack, issparse, csr_matrix
import numpy as np
from copy import deepcopy

from ..transforms.base import Compose
from ..types import SparseMatrix, FrozenDictKeyMap, Batch
from ..utils import (
    get_covariates,
    parse_perturbation_combinations,
    build_covariate_to_control_dict,
    combine_perturbations,
    map_perturbation_to_emb,
    load_dataframe_from_h5,
    validate_columns_across_files,
    MultiFileIndexMap,
    compute_unified_features,
)

import logging

log = logging.getLogger(__name__)


def unique_perturbation_dataframe(
        dataset: ad.AnnData,
        perturbation_key: str,
        covariate_keys: list[str] = [],
):
    """Get a dataframe of unique perturbation/covariate combinations"""
    unique_df = dataset.obs.loc[:, [perturbation_key] + covariate_keys].copy()
    unique_df = unique_df.drop_duplicates()
    unique_df = unique_df.reset_index(drop=True)
    return unique_df


def _concat_adatas_preserve_obs(
        adatas: list[ad.AnnData],
        adata_paths: list[str] | None = None,
        perturbation_key: str | None = None,
        covariate_keys: list[str] | None = None,
        file_indices: list[int] | None = None,
) -> ad.AnnData:
    """Concatenate AnnData objects while preserving all obs columns.

    Makes cell indices unique BEFORE concatenation by appending file index and
    dataset name (if available). This prevents index misalignment issues when
    multiple files have cells with the same original indices.

    Args:
        adatas: List of AnnData objects to concatenate
        adata_paths: Optional paths for column validation (if provided, validates upfront)
        perturbation_key: Required column for validation
        covariate_keys: Required columns for validation
        file_indices: List of file indices corresponding to each adata. If None,
            uses sequential indices [0, 1, 2, ...]. Used to create unique cell
            indices across files.

    Returns:
        Combined AnnData with all obs columns preserved and unique cell indices
    """
    if len(adatas) == 1:
        return adatas[0]

    # Optional: validate columns if paths provided
    if adata_paths is not None and perturbation_key is not None:
        validate_columns_across_files(adata_paths, perturbation_key, covariate_keys)

    # Default file indices if not provided
    if file_indices is None:
        file_indices = list(range(len(adatas)))

    # Make cell indices unique BEFORE concatenation to prevent misalignment
    # This follows the pattern from H5LitModule.accessor and SingleCellPerturbation.from_h5
    for i, (adata, file_idx) in enumerate(zip(adatas, file_indices)):
        obs_df = adata.obs

        # Build unique index: original_index-f{file_idx}-{dataset_name}
        # Dataset name is optional - only append if 'dataset' column exists
        if 'dataset' in obs_df.columns:
            dataset_name = obs_df['dataset'].iloc[0]
            new_index = obs_df.index.astype(str) + f'-f{file_idx}-' + str(dataset_name)
        else:
            new_index = obs_df.index.astype(str) + f'-f{file_idx}'

        # Add file index tracking column
        adata.obs = obs_df.copy()
        adata.obs['_file_idx'] = file_idx
        adata.obs.index = new_index

    # Concatenate obs DataFrames with pandas (outer join preserves all columns)
    # Indices are now unique, so no alignment issues
    obs_dfs = [adata.obs.copy() for adata in adatas]
    combined_obs = pd.concat(obs_dfs, ignore_index=False)

    # Concatenate expression data with anndata
    # No need for index_unique since indices are already unique
    combined = ad.concat(adatas, axis=0)

    # Replace obs with pandas-concatenated version (preserves all columns)
    # Now safe because indices match exactly
    combined_obs.index = combined.obs.index
    combined.obs = combined_obs

    return combined


def _align_adata_to_features(
        adata: ad.AnnData,
        target_features: list[str],
) -> ad.AnnData:
    """Align AnnData to target features with zero-padding for missing features.

    This ensures test datasets have exactly the same features as training datasets,
    even when some files are skipped (no cells in split) and would otherwise be
    missing features that only exist in those skipped files.

    Uses sparse matrix operations throughout to avoid memory-intensive dense
    intermediate arrays.

    Args:
        adata: AnnData to align (must be in memory, not backed)
        target_features: List of feature names to align to

    Returns:
        New AnnData with var_names exactly matching target_features.
        Missing features are zero-padded.
    """
    n_cells = adata.n_obs
    n_features = len(target_features)

    # Build mapping from source column index to target column index
    source_var_names = list(adata.var_names)
    source_to_idx = {name: i for i, name in enumerate(source_var_names)}

    # Find overlapping features and build column index arrays
    source_indices = []
    target_indices = []
    for tgt_idx, feature in enumerate(target_features):
        if feature in source_to_idx:
            source_indices.append(source_to_idx[feature])
            target_indices.append(tgt_idx)

    if not source_indices:
        # No overlapping features, return empty sparse matrix
        aligned_X = csr_matrix((n_cells, n_features), dtype=np.float32)
    else:
        # Get source data as CSR
        source_X = adata.X
        if not issparse(source_X):
            source_X = csr_matrix(source_X, dtype=np.float32)
        else:
            source_X = source_X.tocsr()

        # Build column remapping array: source_col -> target_col (or -1 if not mapped)
        col_remap = np.full(source_X.shape[1], -1, dtype=np.int32)
        col_remap[source_indices] = target_indices

        # Vectorized remapping of column indices
        # Map all source column indices to target indices
        new_col_indices = col_remap[source_X.indices]

        # Create mask for entries that map to valid target columns
        valid_mask = new_col_indices >= 0

        # Extract valid data and indices
        valid_data = source_X.data[valid_mask].astype(np.float32)
        valid_indices = new_col_indices[valid_mask]

        # Build new indptr by counting valid entries per row
        # Use cumsum on the count of valid entries in each row
        row_indices = np.repeat(np.arange(n_cells), np.diff(source_X.indptr))
        valid_row_indices = row_indices[valid_mask]

        # Count valid entries per row and build indptr
        row_counts = np.bincount(valid_row_indices, minlength=n_cells)
        new_indptr = np.zeros(n_cells + 1, dtype=np.int64)
        new_indptr[1:] = np.cumsum(row_counts)

        aligned_X = csr_matrix(
            (valid_data, valid_indices.astype(np.int32), new_indptr),
            shape=(n_cells, n_features)
        )

    # Create new AnnData with aligned features
    aligned_adata = ad.AnnData(X=aligned_X, obs=adata.obs.copy())
    aligned_adata.var_names = target_features

    return aligned_adata


@dataclass
class Counterfactual(Dataset):
    """Counterfactual Dataset.

    This dataset is for generating counterfactual perturbation predictions.
    Each item in this dataset corresponds to a single perturbation applied to a
    unique set of covariates. Iterating over this dataset returns a tuple
    consisting of 1) a Batch with the gene expression set to the control expression
    of the corresponding covariates and the perturbations set to the desired
    counterfactual perturbations and 2) a DataFrame with the perturbation and
    covariate metadata.

    Attributes:
        perturbations: a numpy array of size (n_cells, ) where each element is a
          list of perturbations applied to the cell
        covariates: a dictionary of covariates, where each covariate is a numpy
          array of size (n_cells, )
        control_expression: a gene expression matrix of control cells
        control_indexes: a dictionary that maps from each covariate condition to
          a list of control cell indexes
        gene_names: a list of gene names of length n_genes
        transform: a function that transforms a SingleCellExpressionExample
        info: a dictionary of metadata about the dataset
    """
    perturbations: Sequence[list[str]]
    covariates: dict[str, Sequence[str]]
    control_expression: SparseMatrix
    control_indexes: FrozenDictKeyMap
    gene_names: Sequence[str] | None = None
    transforms: InitVar[Callable | Sequence[Callable] | None] = field(default=None)
    info: dict[str, Any] | None = None
    control_embeddings: np.ndarray | None = None
    perturbation_embeddings: pd.DataFrame | None = None

    def __post_init__(self, transforms: Callable | Sequence[Callable] | None):

        if self.perturbation_embeddings is not None:
            valid_indices = []
            for i, pert_names in enumerate(self.perturbations):
                if all([pert_name in self.perturbation_embeddings.index for pert_name in pert_names]):
                    valid_indices.append(i)
            valid_indices = np.array(valid_indices)

            if valid_indices.shape[0] < len(self.perturbations):
                nonvalid_indices = np.setdiff1d(np.arange(len(self.perturbations)), valid_indices)
                nonvalid_perturbations = np.array(self.perturbations)[nonvalid_indices]
                log.warning(f'{nonvalid_perturbations} discarded due to missing embeddings')

            self.perturbations = np.array(self.perturbations)[valid_indices].tolist()
            for k, v in self.covariates.items():
                self.covariates[k] = np.array(v)[valid_indices].tolist()

        try:
            # Input is a list of callables
            if len(transforms) > 1:
                transform = Compose(transforms)
            # Input is a list of a single callable
            else:
                transform = transforms[0]
        # Input is a callable or None
        except TypeError:
            transform = transforms

        self._transform = transform

    @property
    # pylint: disable-next=missing-function-docstring
    def transform(self) -> Callable | None:
        return self._transform

    @transform.setter
    def transform(self, transform: Callable | None):
        transform = deepcopy(transform)
        transform.pop('controls', None)

        # If transform is not None, make sure it is a valid example transform
        if transform is not None:
            self._transform = (
                None  # Set to None to avoid infinite recursion in __getitem__
            )
            try:
                batch, _ = self._get_counterfactual(range(0, min(3, len(self))))
                transform(batch)

            except Exception as e:
                raise ValueError(
                    f"transform ({transform}) must be a function that "
                    f"transforms a Batch"
                ) from e

        self._transform = transform

    def __len__(self):
        # unique combinations of "condition, treatment, cell_type"
        return len(self.perturbations)

    def _get_counterfactual(self, indices):
        expression_list = []
        covariates = {
            cov: [] for cov in self.covariates.keys()
        }
        perturbations = []
        embedding_list = []
        condition_indices = []

        unique_covariates = set()  # order may change
        for idx in indices:
            covariates_idx = {
                cov: value[idx] for cov, value in self.covariates.items()
            }
            unique_covariates.add(frozenset(covariates_idx.items()))

            # Filter to only categorical covariates for control matching
            # (continuous covariates have None in covariate_uniques)
            categorical_covariates_idx = {
                cov: val for cov, val in covariates_idx.items()
                if self.info['covariate_uniques'].get(cov) is not None
            }
            cells_idx = self.control_indexes[categorical_covariates_idx]
            expression_list.append(self.control_expression[
                                       cells_idx
                                   ])
            if self.control_embeddings is not None:
                embedding_list.append(self.control_embeddings[
                                          cells_idx
                                      ])

            for cov in covariates_idx:
                covariates[cov].extend([covariates_idx[cov]] * len(cells_idx))
            perturbations.extend([self.perturbations[idx]] * len(cells_idx))
            condition_indices.extend([idx] * len(cells_idx))

        if self.perturbation_embeddings is not None:
            perturbations_batch = map_perturbation_to_emb(self.perturbation_embeddings, perturbations)
        else:
            perturbations_batch = perturbations

        # Handle both sparse and dense expression data
        # Convert dense arrays to sparse for consistent vstack behavior
        sparse_expression_list = [
            csr_matrix(expr) if not issparse(expr) else expr
            for expr in expression_list
        ]

        counterfactual_batch = Batch(
            gene_expression=vstack(sparse_expression_list),
            perturbations=perturbations_batch,
            covariates=covariates,
            id=None,
            gene_names=self.gene_names,
            embeddings=torch.Tensor(np.concatenate(embedding_list)) if len(embedding_list) > 0 else None,
        )
        counterfactual_obs = pd.DataFrame(covariates).astype('category')
        counterfactual_obs[self.info['perturbation_key']] = combine_perturbations(
            perturbations,  # desired perturbation labels [plus possibly control cells to be generated synthetically]
            self.info['perturbation_combination_delimiter'],
            self.info['perturbation_control_value'],
        ).astype('category')
        counterfactual_obs['_condition_idx'] = condition_indices

        return (counterfactual_batch, counterfactual_obs)

    def __getitem__(self, idx):
        raise ValueError(
            "Counterfactual dataset does not support single item retrieval"
        )

    def __getitems__(self, indices):
        counterfactual_batch, counterfactual_obs = self._get_counterfactual(indices)
        if self.transform is not None:
            counterfactual_batch = self.transform(counterfactual_batch)
        return counterfactual_batch, counterfactual_obs

    def fetch_control_anndata(self):
        sampled_control_indices = []
        obs_df_list = []
        for key_frozenset, cell_indices in self.control_indexes.items():
            key_dict = {k: [v] * len(cell_indices) for k, v in key_frozenset.copy()}
            obs_df_list.append(pd.DataFrame(key_dict))
            sampled_control_indices.extend(cell_indices)

        obs_df = pd.concat(obs_df_list)
        obs_df[self.info['perturbation_key']] = self.info['perturbation_control_value']
        control_adata = ad.AnnData(
            self.control_expression[sampled_control_indices, :],
            obs=obs_df
        )
        control_adata.var_names = self.gene_names
        # Only subset by categorical covariate keys that are present in obs_df
        # (continuous covariates are excluded from control matching and won't be in obs_df)
        categorical_covariate_keys = [k for k in self.info['covariate_keys'] if k in obs_df.columns]
        control_adata.obs = control_adata.obs[
            [self.info['perturbation_key']] + categorical_covariate_keys
            ]
        return control_adata

    @staticmethod
    def from_anndata(
            adata: ad.AnnData,
            prediction_dataframe: pd.DataFrame,
            perturbation_key: str,
            perturbation_combination_delimiter: str | None = "+",
            covariate_keys: list[str] | None = None,
            perturbation_control_value: str | None = None,
            seed: int = 0,
            max_control_cells_per_covariate: int = 1000,
            embedding_key: str | None = None,
            perturbation_embeddings_path: str | None = None,
            feature_filter_path: str | None = None,
    ) -> tuple[Counterfactual, dict[str, Any]]:
        """
        Counterfactual dataset means cells — control cells, loaded from this dataset
        will be used to do counterfactual predictions.

        Create a Counterfactual dataset from a control AnnData object and a
            prediction_dataframe containing desired counterfactual predictions.

        Args:
            adata: an AnnData object with control cells
            prediction_dataframe: a dataframe containing the desired counterfactual
                predictions. Must contain columns for perturbations and covariates.
            perturbation_key: the key in adata.obs that contains the perturbations
            perturbation_combination_delimiter: the delimiter used to separate
                perturbations in the perturbation_key
            covariate_keys: a list of keys in adata.obs that contain the covariates
            perturbation_control_value: the value in adata.obs[perturbation_key]
                that corresponds to unperturbed control cells
            seed: a random seed for sampling control cells
            max_control_cells_per_covariate: the maximum number of control cells
                to sample for each unique covariate combination for generating
                counterfactual predictions
            embedding_key: the key in adata.obsm that contains the embeddings
            perturbation_embeddings_path: the path to the file containing the
              pre-computed perturbation embeddings
            feature_filter_path: the path to the file containing a subset of
              gene names to keep

        Returns:
            A tuple (dataset, info), where dataset is a Counterfactual
              dataset and info is a dictionary containing supplementary
              information about the dataset not contained in the dataset itself
              (e.g. perturbation and covariate unique values). This information
              can be used to setup data pipelines.
        """
        if covariate_keys is None:
            covariate_keys = ['dummy_covariate']
            prediction_dataframe['dummy_covariate'] = '1'
            adata.obs['dummy_covariate'] = '1'

        if perturbation_control_value is None:
            raise ValueError("Must specify perturbation_control_value")

        # Validate that all covariate_keys exist in the adata.obs
        missing_in_adata = [k for k in covariate_keys if k not in adata.obs.columns]
        if missing_in_adata:
            raise KeyError(
                f"Covariate key(s) {missing_in_adata} not found in adata.obs columns. "
                f"Available columns: {list(adata.obs.columns)}. "
                f"This can cause silent failures where covariates are skipped. "
                f"Please check that your covariate_keys match the column names in your h5 file exactly."
            )

        # prediction_dataframe contains covariate information of the perturbed cells
        prediction_dataframe[perturbation_key] = prediction_dataframe[perturbation_key].astype('category')
        perturbations, perturbation_uniques, perturbation_counts = parse_perturbation_combinations(
            prediction_dataframe[perturbation_key],
            perturbation_combination_delimiter,
            perturbation_control_value,
        )
        # in v1.1, perturbations has 178 items but perturbation_uniques only have 157
        covariates, covariate_uniques = get_covariates(
            prediction_dataframe, covariate_keys
        )
        covariates = {k: list(v) for k, v in covariates.items()}

        if feature_filter_path is not None:
            features_keep = pd.read_csv(feature_filter_path, header=None, index_col=False)
            features_keep = list(set(features_keep.values.flatten()).intersection(adata.var_names))
            feature_indices_keep = [
                i for i, feature in enumerate(adata.var_names)
                if feature in features_keep
            ]
        else:
            feature_indices_keep = list(range(len(adata.var_names)))

        info = dict(
            perturbation_uniques=perturbation_uniques,
            covariate_uniques=covariate_uniques,
            perturbation_key=perturbation_key,
            covariate_keys=covariate_keys,
            perturbation_combination_delimiter=perturbation_combination_delimiter,
            perturbation_control_value=perturbation_control_value,
            perturbation_counts=perturbation_counts,
            features_keep=feature_indices_keep,
        )

        control_adata = adata[
            adata.obs[perturbation_key] == perturbation_control_value,
            feature_indices_keep
        ]
        if control_adata.isbacked:
            control_adata = control_adata.to_memory()

        control_indexes = build_covariate_to_control_dict(
            control_adata.obs,
            covariate_keys
        )

        ## map covariates to fixed set of control cells
        # in 4.1 population aggregation of the paper (not discussed in full)
        random.seed(seed)
        sampled_control_indexes = FrozenDictKeyMap()
        for covs in control_indexes:
            sampled_control_indexes[covs] = random.sample(
                list(control_indexes[covs]),
                k=min(max_control_cells_per_covariate, len(control_indexes[covs])),
            )

        ## Optionally extract embeddings
        control_embeddings = control_adata.obsm[embedding_key] if embedding_key is not None else None

        if perturbation_embeddings_path is not None:
            perturbation_embeddings = pd.read_parquet(perturbation_embeddings_path)

            counter = []
            for perturbation in perturbation_uniques:
                if perturbation not in perturbation_embeddings.index:
                    num_affected = sum([perturbation in pert_list for pert_list in perturbations])
                    counter.append(num_affected)
                    log.warning(f'{perturbation} missing embedding, {num_affected} affected examples')

            # may be double counting some examples with more than one perturbation, but in v1.1 dataset all
            # perturbations we have are single perturbations
            log.warning(f'{len(counter)} perturbations missing, {sum(counter)} affected examples in total.')
        else:
            perturbation_embeddings = None

        # Create perturbation dataset
        dataset = Counterfactual(
            perturbations=perturbations,
            covariates=covariates,
            control_expression=control_adata.X,
            control_indexes=sampled_control_indexes,
            gene_names=control_adata.var_names.to_list(),
            info=info,
            control_embeddings=control_embeddings,
            perturbation_embeddings=perturbation_embeddings,
        )

        return (
            dataset,
            info,
        )

    @staticmethod
    def from_h5(
            adata_path: str | list[str],
            *args,
            **kwargs,
    ):
        """Create a Counterfactual dataset from one or more h5ad files.

        Supports multi-file loading. Only control cells are loaded into memory
        to optimize memory usage.

        Args:
            adata_path: Path to h5ad file(s). Can be a single path or list of paths.
            *args: Positional arguments passed to from_anndata.
            **kwargs: Keyword arguments passed to from_anndata. Must include
                perturbation_key and perturbation_control_value for control filtering.

        Returns:
            Tuple of (dataset, info) from from_anndata.
        """
        # Normalize to list of paths
        if isinstance(adata_path, (str, Path)):
            adata_paths = [str(adata_path)]
        else:
            adata_paths = [str(p) for p in adata_path]

        # Extract required parameters for control filtering
        perturbation_key = kwargs.get('perturbation_key')
        perturbation_control_value = kwargs.get('perturbation_control_value')
        feature_filter_path = kwargs.get('feature_filter_path', None)

        if perturbation_key is None:
            raise ValueError("perturbation_key is required for from_h5")
        if perturbation_control_value is None:
            raise ValueError("perturbation_control_value is required for from_h5")

        # Pre-compute needed covariate values from the prediction_dataframe so
        # we can skip h5ad files whose control cells don't match. The
        # prediction_dataframe is the first positional arg forwarded to
        # from_anndata.
        prediction_dataframe = args[0] if args else None
        covariate_keys = kwargs.get('covariate_keys') or []

        needed_covariate_values: dict[str, set] | None = None
        if prediction_dataframe is not None and covariate_keys:
            categorical_keys = [
                k for k in covariate_keys
                if k in prediction_dataframe.columns
                   and not pd.api.types.is_numeric_dtype(prediction_dataframe[k])
            ]
            if categorical_keys:
                needed_covariate_values = {
                    k: set(prediction_dataframe[k].unique())
                    for k in categorical_keys
                }

        def load_control_cells(
                path: str,
                covariate_filter: dict[str, set] | None = None,
        ) -> ad.AnnData | None:
            """Load only control cells from a single h5ad file.

            When *covariate_filter* is provided, additionally restrict to
            control cells whose categorical covariates match the needed
            values. This avoids the expensive ``read_h5ad`` call for files
            that contain no relevant controls.
            """
            cols_to_load = [perturbation_key]
            if covariate_filter:
                cols_to_load.extend(
                    k for k in covariate_filter if k not in cols_to_load
                )

            obs_df = load_dataframe_from_h5(path, 'obs', cols_to_load)
            control_mask = obs_df[perturbation_key] == perturbation_control_value

            if covariate_filter:
                for key, values in covariate_filter.items():
                    if key in obs_df.columns:
                        control_mask &= obs_df[key].isin(values)

            control_indices = list(np.where(control_mask)[0])
            if len(control_indices) == 0:
                return None

            adata = ad.read_h5ad(path, backed='r')
            adata = adata[control_indices].to_memory()
            return adata

        # Single file case
        if len(adata_paths) == 1:
            adata = load_control_cells(adata_paths[0], needed_covariate_values)
            if adata is None:
                raise ValueError(
                    f"No control cells found in {adata_paths[0]} with "
                    f"{perturbation_key}=={perturbation_control_value}"
                )

            # Align features if filter provided (ensures same order as training)
            if feature_filter_path is not None:
                target_features = compute_unified_features(adata_paths, feature_filter_path)
                adata = _align_adata_to_features(adata, target_features)

            return Counterfactual.from_anndata(
                adata,
                *args,
                **kwargs,
            )

        # Multi-file case
        # Step 1: Load control cells from each file, tracking file indices
        adatas = []
        file_indices = []  # Track which file each adata came from
        for i, path in enumerate(adata_paths):
            adata = load_control_cells(path, needed_covariate_values)
            if adata is not None:
                adatas.append(adata)
                file_indices.append(i)
            else:
                log.debug(
                    "Skipping %s: no control cells matching prediction covariates",
                    Path(path).name,
                )

        if len(adatas) == 0:
            raise ValueError(
                f"No control cells found in any file with "
                f"{perturbation_key}=={perturbation_control_value}"
            )

        # Step 2: Feature alignment using the same logic as training dataset
        # This ensures test features exactly match train features (with zero-padding)
        target_features = compute_unified_features(adata_paths, feature_filter_path)

        # Align each adata to target features with zero-padding for missing features
        aligned_adatas = [_align_adata_to_features(adata, target_features) for adata in adatas]
        adatas = aligned_adatas

        # Step 3: Concatenate and call from_anndata
        combined = _concat_adatas_preserve_obs(
            adatas,
            adata_paths=adata_paths,
            perturbation_key=perturbation_key,
            covariate_keys=kwargs.get('covariate_keys'),
            file_indices=file_indices,
        )

        return Counterfactual.from_anndata(
            combined,
            *args,
            **kwargs,
        )


@dataclass
class CounterfactualWithReference(Counterfactual):
    """Counterfactual Dataset with matched Reference Data.

    This dataset inherits from Counterfactual Dataset and also contains the
    observed reference data as an AnnData object. Iterating over this dataset
    will return a tuple of 1) the counterfactual Batch for generating
    predictions, 2) the counterfactual metadata, and 3) the reference data for
    the corresponding perturbations and covariates.

    Attributes:
        reference_indexes: a dictionary that maps from each covariate/perturbation
            condition to a list of cell indexes in the reference_adata
        reference_adata: a anndata object with the observed reference data
    """
    reference_indexes: dict[str, FrozenDictKeyMap] | None = None
    reference_adata: ad.AnnData | None = None

    @staticmethod
    def from_anndata(
            adata: ad.AnnData,
            perturbation_key: str,
            split: pd.Series | None = None,
            perturbation_combination_delimiter: str | None = "+",
            covariate_keys: list[str] | None = None,
            perturbation_control_value: str | None = None,
            seed: int = 0,
            max_control_cells_per_covariate: int = 1000,
            use_synthetic_controls: bool = False,
            embedding_key: str | None = None,
            perturbation_embeddings_path: str | None = None,
            feature_filter_path: str | None = None,
    ) -> tuple[CounterfactualWithReference, dict[str, Any]]:
        """Create a CounterfactualWithReference dataset from an AnnData object.

        Args:
            adata: the AnnData object to evaluate (i.e. the AnnData that
                corresponds to the test split)
            perturbation_key: the key in adata.obs that contains the perturbations
            perturbation_combination_delimiter: the delimiter used to separate
                perturbations in the perturbation_key
            covariate_keys: a list of keys in adata.obs that contain the covariates
            perturbation_control_value: the value in adata.obs[perturbation_key]
                that corresponds to unperturbed control cells
            seed: a random seed for sampling control cells
            max_control_cells_per_covariate: the maximum number of control cells
                to sample for each unique covariate combination for generating
                counterfactual predictions

        Returns:
            A tuple (dataset, info), where dataset is a CounterfactualWithReference
                dataset and info is a dictionary containing supplementary
                information about the dataset not contained in the dataset itself
                (e.g. perturbation and covariate unique values). This information
                can be used to setup data pipelines.
        """
        if adata.isbacked:
            adata = adata.to_memory()

        if split is not None:
            adata = adata[split]

        if covariate_keys is None:
            covariate_keys = ['dummy_covariate']
            adata.obs['dummy_covariate'] = '1'

        if perturbation_control_value is None:
            raise ValueError("Must specify perturbation_control_value")

        # Validate that all covariate_keys exist in the adata.obs
        missing_in_adata = [k for k in covariate_keys if k not in adata.obs.columns]
        if missing_in_adata:
            raise KeyError(
                f"Covariate key(s) {missing_in_adata} not found in adata.obs columns. "
                f"Available columns: {list(adata.obs.columns)}. "
                f"This can cause silent failures where covariates are skipped. "
                f"Please check that your covariate_keys match the column names in your h5 file exactly."
            )

        if perturbation_control_value not in adata.obs[perturbation_key].unique():
            raise ValueError(
                f"perturbation_control_value {perturbation_control_value} not in "
                f"{perturbation_key} column of adata"
            )

        adata.obs = adata.obs[[perturbation_key] + covariate_keys]

        # unique triplets: perturbation + covariates ...
        prediction_dataframe = unique_perturbation_dataframe(
            adata,
            perturbation_key,
            covariate_keys,
        )

        # If use_synthetic_controls is True, validate that control rows exist for each unique covariate combination
        if use_synthetic_controls:
            # Get unique covariate combinations from perturbations (non-control rows)
            perturbed_rows = prediction_dataframe[prediction_dataframe[perturbation_key] != perturbation_control_value]
            unique_covariates = perturbed_rows[covariate_keys].drop_duplicates()

            # Get control rows from prediction_dataframe
            control_rows = prediction_dataframe[prediction_dataframe[perturbation_key] == perturbation_control_value]
            control_covariates = control_rows[covariate_keys].drop_duplicates()

            # Check if every unique covariate combination has a corresponding control row
            for _, row in unique_covariates.iterrows():
                match = control_covariates
                for cov_key in covariate_keys:
                    match = match[match[cov_key] == row[cov_key]]
                if len(match) == 0:
                    raise ValueError(
                        f"use_synthetic_controls=True but missing control row for covariate combination: "
                        f"{dict(row[covariate_keys])}"
                    )
        else:
            # If not using synthetic controls, filter out control cells from prediction_dataframe
            prediction_dataframe = prediction_dataframe.loc[
                prediction_dataframe[perturbation_key] != perturbation_control_value
                ]

        control_adata = adata[adata.obs[perturbation_key] == perturbation_control_value]
        counterfactual_dataset, info = Counterfactual.from_anndata(
            adata=control_adata,  # only control cells here
            prediction_dataframe=prediction_dataframe,  # contains control rows if use_synthetic_controls=True
            perturbation_key=perturbation_key,
            perturbation_combination_delimiter=perturbation_combination_delimiter,
            covariate_keys=covariate_keys,
            perturbation_control_value=perturbation_control_value,
            seed=seed,
            max_control_cells_per_covariate=max_control_cells_per_covariate,
            embedding_key=embedding_key,
            perturbation_embeddings_path=perturbation_embeddings_path,
            feature_filter_path=feature_filter_path,
        )

        parsed_adata_perturbations = parse_perturbation_combinations(
            adata.obs[perturbation_key],
            perturbation_combination_delimiter,
            perturbation_control_value,
        )[0]
        adata.obs['_parsed_perturbations'] = [frozenset(x) for x in parsed_adata_perturbations]
        # 3 things mapping to idx: _parsed_perturbations, cell_type, treatment
        # 220 unique combinations including control cells
        # _parsed_perturbations contains control cells (and it is necessary)
        reference_indexes = build_covariate_to_control_dict(
            adata.obs,
            ['_parsed_perturbations'] + covariate_keys
        )

        # Create perturbation dataset
        dataset = CounterfactualWithReference(
            perturbations=counterfactual_dataset.perturbations,
            covariates=counterfactual_dataset.covariates,  # same length as counterfactual_dataset.perturbations
            control_expression=counterfactual_dataset.control_expression,
            control_indexes=counterfactual_dataset.control_indexes,
            gene_names=counterfactual_dataset.gene_names,
            info=info,
            reference_adata=adata[:, info['features_keep']],
            reference_indexes=reference_indexes,
            control_embeddings=counterfactual_dataset.control_embeddings,
            perturbation_embeddings=counterfactual_dataset.perturbation_embeddings,
        )

        return (
            dataset,
            info,
        )

    @staticmethod
    def from_h5(
            adata_path: str | list[str],
            *args,
            **kwargs,
    ):
        """Create a CounterfactualWithReference dataset from one or more h5ad files.

        Supports multi-file loading with global split indices. When multiple files
        are provided, splits are interpreted as global indices into the concatenated
        dataset (same semantics as H5LitModule training).

        Args:
            adata_path: Path to h5ad file(s). Can be a single path or list of paths.
            *args: Positional arguments passed to from_anndata.
            **kwargs: Keyword arguments passed to from_anndata. The 'split' argument
                is handled here and not passed to from_anndata.

        Returns:
            Tuple of (dataset, info) from from_anndata.
        """
        # Normalize to list of paths
        if isinstance(adata_path, (str, Path)):
            adata_paths = [str(adata_path)]
        else:
            adata_paths = [str(p) for p in adata_path]

        # Pop split and feature_filter_path from kwargs
        split = kwargs.pop('split', None)
        feature_filter_path = kwargs.get('feature_filter_path', None)

        # Single file case - use original efficient logic
        if len(adata_paths) == 1:
            adata = ad.read_h5ad(adata_paths[0], backed='r')
            if split is not None:
                adata = adata[split].to_memory()
            else:
                adata = adata.to_memory()

            # Align features if filter provided (ensures same order as training)
            if feature_filter_path is not None:
                target_features = compute_unified_features(adata_paths, feature_filter_path)
                adata = _align_adata_to_features(adata, target_features)

            return CounterfactualWithReference.from_anndata(
                adata,
                *args,
                **kwargs,
            )

        # Multi-file case
        # Extract keys needed for validation from kwargs
        perturbation_key = kwargs.get('perturbation_key')
        covariate_keys = kwargs.get('covariate_keys')

        # Step 1: Get cell counts from each file (without loading expression)
        cell_counts = []
        for path in adata_paths:
            obs_df = load_dataframe_from_h5(path, 'obs')
            cell_counts.append(len(obs_df))

        index_map = MultiFileIndexMap.from_cell_counts(cell_counts)

        # Step 2: Convert global split to per-file local indices
        if split is not None:
            per_file_local_indices = [[] for _ in adata_paths]
            for global_idx in split:
                file_idx, local_idx = index_map.global_to_local(global_idx)
                per_file_local_indices[file_idx].append(local_idx)
        else:
            per_file_local_indices = None

        # Step 3: Load each file in backed mode, subset, convert to memory
        adatas = []
        file_indices = []  # Track which file each adata came from
        for i, path in enumerate(adata_paths):
            if per_file_local_indices is not None:
                local_indices = per_file_local_indices[i]
                if len(local_indices) == 0:
                    continue  # Skip files with no split indices
                adata = ad.read_h5ad(path, backed='r')
                adata = adata[local_indices].to_memory()
            else:
                adata = ad.read_h5ad(path, backed='r')
                adata = adata.to_memory()

            adatas.append(adata)
            file_indices.append(i)

        if len(adatas) == 0:
            raise ValueError("No data loaded - split indices may be invalid.")

        # Step 4: Feature alignment - align to target features with zero-padding
        # This ensures test features exactly match train features
        target_features = compute_unified_features(adata_paths, feature_filter_path)

        # Align each adata to target features with zero-padding for missing features
        aligned_adatas = [_align_adata_to_features(adata, target_features) for adata in adatas]
        adatas = aligned_adatas

        # Step 5: Concatenate and call from_anndata
        combined = _concat_adatas_preserve_obs(
            adatas,
            adata_paths=adata_paths,
            perturbation_key=perturbation_key,
            covariate_keys=covariate_keys,
            file_indices=file_indices,
        )

        return CounterfactualWithReference.from_anndata(
            combined,
            *args,
            **kwargs,
        )
