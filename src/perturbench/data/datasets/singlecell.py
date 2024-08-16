from __future__ import annotations
from copy import deepcopy
from typing import Any, Callable, Sequence
from dataclasses import dataclass, InitVar, field
import warnings

import anndata as ad
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

from ..transforms.base import Compose
from ..types import SparseMatrix, Example, FrozenDictKeyMap, Batch
from ..utils import (
    get_covariates,
    parse_perturbation_combinations,
    build_control_dict,
    restore_perturbation_combinations,
)


__all__ = ["SingleCellPerturbation", "SingleCellPerturbationWithControls"]


@dataclass
class SingleCellPerturbation(Dataset):
    """Single Cell Perturbation Dataset.

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
        embeddings: optional foundation model embeddings to use as input instead
          of gene expression
    """

    gene_expression: SparseMatrix
    perturbations: Sequence[list[str]]
    covariates: dict[str, Sequence[str]] | None = None
    cell_ids: Sequence[str] | None = None
    gene_names: Sequence[str] | None = None
    transforms: InitVar[Callable | Sequence[Callable] | None] = field(default=None)
    embeddings: np.ndarray | None = None

    def __post_init__(self, transforms: Callable | Sequence[Callable] | None):
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
        # If transform is not None, make sure it is a valid example transform
        if transform is not None:
            self._transform = (
                None  # Set to None to avoid infinite recursion in __getitem__
            )
            try:
                batch = self._get_batch(range(0, 3))
                transform(batch)

                example = self[0]
                transform(example)
            except Exception as e:
                raise ValueError(
                    f"transform ({transform}) must be a function that "
                    f"transforms an Example or Batch"
                ) from e
        self._transform = transform

    def __len__(self):
        return self.gene_expression.shape[0]

    def _get_example(self, idx):
        return Example(
            gene_expression=self.gene_expression[idx],
            perturbations=self.perturbations[idx],
            covariates=(
                {cov: value[idx] for cov, value in self.covariates.items()}
                if self.covariates is not None
                else None
            ),
            id=self.cell_ids[idx] if self.cell_ids is not None else None,
            gene_names=self.gene_names,
            embeddings=self.embeddings[idx] if self.embeddings is not None else None,
        )

    def _get_batch(self, idx_list):
        return Batch(
            gene_expression=self.gene_expression[idx_list, :],
            perturbations=[self.perturbations[i] for i in idx_list],
            covariates=(
                {cov: value[idx_list] for cov, value in self.covariates.items()}
                if self.covariates is not None
                else None
            ),
            id=(
                [self.cell_ids[i] for i in idx_list]
                if self.cell_ids is not None
                else None
            ),
            gene_names=self.gene_names,
            embeddings=(
                torch.Tensor(self.embeddings[idx_list, :])
                if self.embeddings is not None
                else None
            ),
        )

    def __getitem__(self, idx):
        if isinstance(idx, int):
            example = self._get_example(idx)
            if self.transform is not None:
                example = self.transform(example)  # pylint: disable=not-callable
            return example
        else:
            batch = self._get_batch(idx)
            if self.transform is not None:
                batch = self.transform(batch)
            return batch

    def add_controls(
        self,
        control_indexes: dict[str, Sequence[int]],
        control_expression: SparseMatrix,
        copy: bool = False,
    ) -> SingleCellPerturbationWithControls:
        """Add controls to the dataset.

        Args:
            controls: a dictionary of controls, where each control is a sparse
              matrix of size (n_controls, n_genes)
            copy: whether to copy the dataset before adding controls

        Returns:
            A SingleCellPerturbationWithControls dataset

        Raises:
            ValueError: if a control is not associated with a covariate condition
        """
        return SingleCellPerturbationWithControls(
            gene_expression=(
                self.gene_expression.copy() if copy else self.gene_expression
            ),
            perturbations=self.perturbations.copy() if copy else self.perturbations,
            covariates=deepcopy(self.covariates) if copy else self.covariates,
            cell_ids=deepcopy(self.cell_ids) if copy else self.cell_ids,
            gene_names=deepcopy(self.gene_names) if copy else self.gene_names,
            transforms=self.transforms,
            control_indexes=control_indexes,
            control_expression=control_expression,
            embeddings=deepcopy(self.embeddings) if copy else self.embeddings,
        )

    @staticmethod
    def from_anndata(
        adata: ad.AnnData,
        perturbation_key: str,
        perturbation_combination_delimiter: str | None = "+",
        covariate_keys: list[str] | None = None,
        perturbation_control_value: str | None = None,
        embedding_key: str | None = None,
    ) -> tuple[SingleCellPerturbation, dict[str, Any]]:
        """Create a SingleCellPerturbation dataset from an AnnData object.

        Args:
            adata: an AnnData object
            perturbation_key: the key in adata.obs that contains the perturbations
            perturbation_combination_delimiter: the delimiter used to separate
                perturbations in the perturbation_key
            covariate_keys: a list of keys in adata.obs that contain the covariates
            perturbation_control_value: the value in adata.obs[perturbation_key]
                that corresponds to unperturbed control cells

        Returns:
            A tuple (dataset, info), where dataset is a SingleCellPerturbation
              dataset and info is a dictionary containing supplementary
              information about the dataset not contained in the dataset itself
              (e.g. perturbation and covariate unique values). This information
              can be used to setup data pipelines.
        """
        if covariate_keys is None:
            covariate_keys = []

        if perturbation_control_value is None:
            raise ValueError("Must specify perturbation_control_value")

        # Get gene expression matrix
        gene_expression = adata.X

        # Optionally get foundation model embeddings
        embeddings = adata.obsm[embedding_key] if embedding_key is not None else None

        # Parse (if necessary) perturbations
        perturbations, perturbation_uniques = parse_perturbation_combinations(
            adata.obs[perturbation_key],
            perturbation_combination_delimiter,
            perturbation_control_value,
        )
        # Get covariates
        covariates, covariate_uniques = get_covariates(adata.obs, covariate_keys)

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
        )

        # Create perturbation dataset
        dataset = SingleCellPerturbation(
            gene_expression,
            perturbations,
            covariates,
            cell_ids=adata.obs_names.to_list(),
            gene_names=adata.var_names.to_list(),
            embeddings=embeddings,
        )

        return (
            dataset,
            info,
        )

    ## TODO: Pass INFO as parameter
    def to_anndata(self, info: dict[str, Any]) -> ad.AnnData:
        """Convert the dataset to an AnnData object.

        Returns:
            An AnnData object
        """
        obs_df = pd.DataFrame(self.covariates)
        obs_df[info["perturbation_key"]] = restore_perturbation_combinations(
            self.perturbations,
            delimiter=info["perturbation_combination_delimiter"],
            perturbation_control_value=info["perturbation_control_value"],
        )

        adata = ad.AnnData(
            X=self.gene_expression,
            obs=obs_df,
        )

        if isinstance(self, SingleCellPerturbationWithControls):
            control_adata_list = []
            for covs, idxs in self.control_indexes.items():
                covs = dict(covs)
                control_adata = ad.AnnData(
                    X=self.control_expression[idxs, :],
                    obs={cov: value for cov, value in covs.items()},
                )
                control_adata.obs[info["perturbation_key"]] = info[
                    "perturbation_control_value"
                ]
                control_adata_list.append(control_adata)
            adata = ad.concat([adata] + control_adata_list)

        if self.cell_ids is not None:
            cell_ids = self.cell_ids
            if isinstance(self, SingleCellPerturbationWithControls):
                cell_ids += self.control_ids
            adata.obs_names = cell_ids

        if self.gene_names is not None:
            adata.var.index = self.gene_names

        adata.obs[info["perturbation_key"]] = adata.obs[
            info["perturbation_key"]
        ].astype("category")
        return adata


@dataclass
class SingleCellPerturbationWithControls(SingleCellPerturbation):
    """Single Cell Perturbation Dataset with Controls.

    This dataset inherits from SingleCellPerturbation and adds control cells
    associated with each covariate conditions. This is captured by adding an
    extra attribute that maps from each covariate condition to a gene expression
    matrix of control cells associated with that covariate condition.

    Attributes:
        control_ids: A list of control cell identifiers
        control_indexes: A dictionary that maps from each covariate condition to
          a list of control cell indexes
        control_expression: A gene expression matrix of control cells
    """

    control_ids: Sequence[str] | None = None
    control_indexes: (
        FrozenDictKeyMap[dict[str, Sequence[str]], Sequence[int]] | None
    ) = None
    control_expression: SparseMatrix | None = None

    def __post_init__(self, *args, **kwargs):
        super().__post_init__(*args, **kwargs)
        # Validate controls
        self._check_controls()

    def _check_controls(self):
        """Validate that controls correspond to perturbed covariates.

        Each combination of covariates found in the perturbed cells needs a
        corresponding entry in the control dictionary.
        """
        for i in range(0, len(self)):
            covariates_i = {k: v[i] for k, v in self.covariates.items()}
            try:
                self.control_indexes[covariates_i]
            except KeyError as exc:
                raise ValueError(
                    f"No controls found for covariate condition {covariates_i}."
                ) from exc

    def _get_example(self, idx):
        example = super()._get_example(idx)
        assert example.controls is None
        # Add controls to example
        try:
            example_control_idxs = self.control_indexes[example.covariates]
            if example_control_idxs is None:
                warnings.warn(
                    f"No controls found for covariate condition {example.covariates}."
                )
            sampled_control_idx = np.random.choice(
                example_control_idxs,
                size=1,
            )[0]
            example = example._replace(
                controls=torch.Tensor(
                    self.control_expression[sampled_control_idx, :].toarray()
                ),
            )

        except AttributeError as exc:
            raise RuntimeError(
                f"Dataset is missing controls for the following covariate "
                f"key: {example.covariates}."
            ) from exc

        return example

    def _get_batch(self, idx_list):
        batch = super()._get_batch(idx_list)
        assert batch.controls is None

        # Add controls to batch
        batch_cov_list = [
            (
                {cov: value[i] for cov, value in batch.covariates.items()}
                if batch.covariates is not None
                else None
            )
            for i in range(0, len(idx_list))
        ]
        sampled_control_idxs = [
            np.random.choice(self.control_indexes[covs], size=1)[0]
            for covs in batch_cov_list
        ]

        batch = batch._replace(
            controls=torch.Tensor(
                self.control_expression[sampled_control_idxs, :].toarray()
            )
        )

        if self.embeddings is not None:
            batch = batch._replace(
                embeddings=torch.Tensor(self.embeddings[sampled_control_idxs, :])
            )

        return batch

    @staticmethod
    def from_anndata(
        adata: ad.AnnData,
        perturbation_key: str,
        perturbation_combination_delimiter: str | None = "+",
        covariate_keys: list[str] | None = None,
        perturbation_control_value: str | None = None,
        embedding_key: str | None = None,
    ):
        """Create a SingleCellPerturbationWithControls dataset from an AnnData object.

        Args:
            adata: an AnnData object
            perturbation_key: the key in adata.obs that contains the perturbations
            perturbation_combination_delimiter: the delimiter used to separate
              perturbations in the perturbation_key
            covariate_keys: a list of keys in adata.obs that contain the covariates
            perturbation_control_value: the value in adata.obs[perturbation_key] that
              corresponds to control cells

        Returns:
            A tuple (dataset, info), where dataset is a SingleCellPerturbationWithControls
            dataset and info is a dictionary containing supplementary information
            about the dataset not contained in the dataset itself (e.g. perturbation
            and covariate unique values). This information can be used to setup data
            pipelines.

        Raises:
            ValueError: if perturbation_control_value is None
        """
        if perturbation_control_value is None:
            raise ValueError("Must specify perturbation_control_value")
        # Split adata into perturbation and control cells
        perturbed_adata = adata[
            adata.obs[perturbation_key] != perturbation_control_value
        ]
        control_adata = adata[adata.obs[perturbation_key] == perturbation_control_value]

        # Create perturbation dataset
        dataset, info = SingleCellPerturbation.from_anndata(
            perturbed_adata,
            perturbation_key,
            perturbation_combination_delimiter,
            covariate_keys,
            perturbation_control_value,
            embedding_key=embedding_key,
        )
        # Get controls
        control_indexes = build_control_dict(control_adata.obs, covariate_keys)

        # Add controls to dataset
        dataset_with_controls = dataset.add_controls(control_indexes, control_adata.X)
        dataset_with_controls.control_ids = control_adata.obs_names.to_list()
        return dataset_with_controls, info
