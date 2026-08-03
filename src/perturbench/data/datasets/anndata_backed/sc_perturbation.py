from __future__ import annotations
from typing import Any, Callable, Sequence
from dataclasses import dataclass, InitVar

import anndata
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset
from anndata.io import read_elem, sparse_dataset
import h5py

from ...transforms.base import Compose
from ...types import Batch
from ...utils import (
    get_covariates,
    parse_perturbation_combinations,
    map_perturbation_to_emb,
)


__all__ = ["SingleCellPerturbation"]


# helper function for efficiently loading the h5ad datasets
def read_backed(filepath):

    # with h5py.File(filepath) as hdf5_handle:
    hdf5_handle = h5py.File(filepath)
    return anndata.AnnData(
        sparse_dataset(hdf5_handle["X"]),
        # indexing operation will convert anndata._core.sparse_dataset.CSRDataset to scipy.sparse._csr.csr_matrix
        # do not load obsm which might contain cell embeddings that lead to OOM
        # obsm={key: sparse_dataset(group["obsm"][key])},
        **{
            k: read_elem(hdf5_handle[k]) if k in hdf5_handle else {}
            for k in ["obs", "var", "varm", "uns", "obsp", "varp"]
        }
    )


@dataclass
class SingleCellPerturbation(Dataset):
    """Single Cell Perturbation Dataset — Anndata backed version.

    This dataset contains single cell gene expression data with perturbations
    and covariates. The gene expression data is represented as a sparse matrix
    and the perturbations and covariates are represented as numpy arrays. The
    perturbations and covariates are associated with each cell in the gene
    expression matrix.

    Attributes:
        gene_expression: a sparse matrix of size (n_cells, n_genes)
        perturbations: a numpy array of size (n_cells, ) where each element is a
          list of perturbations applied to the cell
        covariates: a dictionary of covariates, where each covariate is a numpy
          array of size (n_cells, )
        cell_ids: a list of cell ids of length n_cells
        gene_names: a list of gene names of length n_genes
        transform: a function that transforms a SingleCellExpressionExample
        embeddings: a numpy array of size (n_cells, emb_width) representing cell level embeddings
        perturbation_embeddings: a pandas DataFrame of size (n_perturbations, n_features) 
          containing pre-computed perturbation embeddings
        info: a dictionary containing metadata about the dataset initialization
    """

    adata: anndata.AnnData
    filename: str
    obs_indices_keep: np.ndarray[np.int64]
    feature_indices_keep: np.ndarray[np.int64]
    perturbations: Sequence[list[str]]
    covariates: dict[str, Sequence[str]] | None = None
    embedding_key: str | None = None
    cell_ids: Sequence[str] | None = None
    gene_names: Sequence[str] | None = None
    transform: InitVar[Callable | Sequence[Callable] | None]
    perturbation_embeddings: pd.DataFrame | None = None
    
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

        if self.embedding_key is not None:
            # may be accessed by the datamodule to obtain cell embedding width
            self.embeddings = self.adata.obsm[self.embedding_key].copy()
        else:
            self.embeddings = None

        self.adata.file.close()
        self.adata = None

    @property
    # pylint: disable-next=missing-function-docstring
    def transform(self) -> Callable | None:
        return self._transform

    def set_transform(self, transform: Callable | None):
        self._supports_example_mode = True
        self._supports_batch_mode = True
        
        # If transform is not None, make sure it is a valid transform
        if transform is not None:
            self._transform = None  # Avoid infinite recursion

            # Try batch mode first
            try:
                if self.adata is None:
                    self.adata = read_backed(self.filename)
                batch = self._get_batch(range(0, 3))
                transform(batch)
                self.adata.file.close()
                self.adata = None
            except Exception as e:  # pylint: disable=broad-except
                raise ValueError(
                    f"Transform {transform} is incompatible with this dataset. Failed batch mode validation."
                ) from e

        self._transform = transform
    
    @transform.setter
    def transform(self, transform: Callable | None):
        self.set_transform(transform)

    def __len__(self):
        return self.obs_indices_keep.shape[0]

    def _get_batch(self, indices):
        perturbations = [self.perturbations[i] for i in indices]
        if self.perturbation_embeddings is not None:
            perturbations = map_perturbation_to_emb(
                self.perturbation_embeddings, 
                perturbations
            )

        selected_obs_indices = self.obs_indices_keep[indices]
        gene_expression = None
        while gene_expression is None:
            try:
                # reading from hdf5 concurrently and directly from lustre fs can create a lot of issues
                gene_expression = self.adata.X[selected_obs_indices, :][:, self.feature_indices_keep].toarray()
            except Exception as e:
                print(e)
        if self.embedding_key:  # cell embeddings
            embeddings = torch.as_tensor(self.embeddings[selected_obs_indices], dtype=torch.float32)
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
            embeddings=embeddings,
        )

    def __getitems__(self, indices):
        if self.adata is None:
            self.adata = read_backed(self.filename)
        if not self._supports_batch_mode:
            raise ValueError("Dataset does not support batch mode.")
        batch = self._get_batch(indices)
        if self.transform is not None:
            batch = self.transform(batch)  # pylint: disable=not-callable

        return batch

    @staticmethod
    def from_anndata(
        adata: anndata.AnnData,
        perturbation_key: str,
        split: pd.Series | np.ndarray | None = None,
        perturbation_combination_delimiter: str | None = "+",
        covariate_keys: list[str] | None = None,
        perturbation_control_value: str | None = None,
        embedding_key: str | None = None,
        perturbation_embeddings_path: str | None = None,
        feature_filter_path: str | None = None,
    ) -> tuple[SingleCellPerturbation, dict[str, Any]]:
        """Create a SingleCellPerturbation dataset from an AnnData object.

        Args:
            adata: an AnnData object
            perturbation_key: the key in adata.obs that contains the perturbations
            perturbation_combination_delimiter: the delimiter used to separate
              perturbations in the perturbation_key
            covariate_keys: a list of keys in adata.obs that contain the covariates
            perturbation_control_value: the value in adata.obs[perturbation_key] that
              corresponds to control cells
            embedding_key: the key in adata.obsm that contains the embeddings
            perturbation_embeddings_path: the path to the file containing the
              pre-computed perturbation embeddings
            feature_filter_path: the path to the file containing a subset of
              gene names to keep
            
        Returns:
            A tuple (dataset, info), where dataset is a SingleCellPerturbation
              dataset and info is a dictionary containing supplementary
              information about the dataset not contained in the dataset itself
              (e.g. perturbation and covariate unique values). This information
              can be used to setup data pipelines.
        """
        # load embeddings for perturbations
        if perturbation_embeddings_path is not None:
            perturbation_embeddings = pd.read_parquet(perturbation_embeddings_path)
        else:
            perturbation_embeddings = None

        if split is None:
            split_bool = np.array([True] * adata.shape[0])
        else:
            if isinstance(split, pd.Series):
                split = split.values
            split = set(split)
            split_bool = np.array([i in split for i in range(0, adata.shape[0])])

        if perturbation_embeddings is not None:
            # remove perturbations that are missing embeddings in `perturbation_embeddings`
            perturbations, _, _ = parse_perturbation_combinations(
                adata.obs.loc[split_bool,perturbation_key],
                perturbation_combination_delimiter,
                perturbation_control_value,
            )
            # use `set` for O(1) time look up
            index_set = set(perturbation_embeddings.index)
            valid_indices = []
            for i, pert_names in enumerate(perturbations):
                # all([]) = True
                is_valid_example = all([pert_name in index_set for pert_name in pert_names])
                if is_valid_example:
                    valid_indices.append(i)
            
            ## Ensure indices are in the correct order
            original_indices = np.where(split_bool)[0]
            valid_original_indices = original_indices[np.array(valid_indices)]
            split_bool = np.zeros(adata.shape[0], dtype=bool)
            split_bool[valid_original_indices] = True
        
        if feature_filter_path is not None:
            features_keep = pd.read_csv(feature_filter_path, index_col=False, header=None)
            features_keep = list(set(features_keep.values.flatten()).intersection(adata.var_names))
            feature_indices_keep = [
                i for i, feature in enumerate(adata.var_names)
                if feature in features_keep
            ]
        else:
            feature_indices_keep = list(range(len(adata.var_names)))
        
        feature_indices_keep = np.array(feature_indices_keep)

        if covariate_keys is None:
            covariate_keys = []

        if perturbation_control_value is None:
            raise ValueError("Must specify perturbation_control_value")

        obs_indices_keep = np.where(split_bool)[0]

        # Parse (if necessary) perturbations
        perturbations, perturbation_uniques, perturbation_counts = parse_perturbation_combinations(
            adata.obs[perturbation_key].loc[split_bool],
            perturbation_combination_delimiter,
            perturbation_control_value,
        )
        # Get covariates
        covariates, covariate_uniques = get_covariates(adata.obs.loc[split_bool], covariate_keys)

        # Create supplementary information about the dataset not contained in the
        # dataset itself (e.g. perturbation and covariate unique values). This
        # information can be used to setup data pipelines.
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

        # Create perturbation dataset
        filename = adata.uns['filename']
        dataset = SingleCellPerturbation(
            adata,
            filename,
            obs_indices_keep,
            feature_indices_keep,
            perturbations,
            covariates,
            embedding_key=embedding_key,
            cell_ids=adata.obs_names[split_bool].to_list(),
            gene_names=adata.var_names[feature_indices_keep].to_list(),
            perturbation_embeddings=perturbation_embeddings,
        )
        
        return (
            dataset,
            info,
        )
