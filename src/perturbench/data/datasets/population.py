from __future__ import annotations
from typing import Any, Callable, Sequence
from dataclasses import dataclass, InitVar, field
import random
import anndata as ad
import pandas as pd
import torch
from torch.utils.data import Dataset
from scipy.sparse import vstack
import numpy as np

from ..transforms.base import Compose
from ..types import SparseMatrix, FrozenDictKeyMap, Batch
from ..utils import (
    get_covariates,
    parse_perturbation_combinations,
    build_control_dict,
    restore_perturbation_combinations,
)


def unique_perturbation_dataframe(
    dataset: ad.AnnData,
    perturbation_key: str,
    covariate_keys: list[str] = [],
):
    """Get a dataframe of unique perturbation/covariate values"""
    unique_df = dataset.obs.loc[:, [perturbation_key] + covariate_keys].copy()
    unique_df = unique_df.drop_duplicates()
    unique_df = unique_df.reset_index(drop=True)
    return unique_df


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
                batch, _ = self._get_counterfactual(range(0, 3))
                transform(batch)

            except Exception as e:
                raise ValueError(
                    f"transform ({transform}) must be a function that "
                    f"transforms a Batch"
                ) from e
        self._transform = transform

    def __len__(self):
        return len(self.perturbations)

    def _get_counterfactual(self, idx_list):
        expression_list = []
        covariates = {cov: [] for cov in self.covariates.keys()}
        perturbations = []
        embedding_list = []

        for idx in idx_list:
            covariates_idx = {cov: value[idx] for cov, value in self.covariates.items()}

            cells_idx = self.control_indexes[covariates_idx]
            expression_list.append(self.control_expression[cells_idx])
            if self.control_embeddings is not None:
                embedding_list.append(self.control_embeddings[cells_idx])

            for cov in covariates_idx:
                covariates[cov].extend([covariates_idx[cov]] * len(cells_idx))
            perturbations.extend([self.perturbations[idx]] * len(cells_idx))

        counterfactual_batch = Batch(
            gene_expression=torch.Tensor(vstack(expression_list).todense()),
            perturbations=perturbations,
            covariates=covariates,
            id=None,
            gene_names=self.gene_names,
            embeddings=(
                torch.Tensor(np.concatenate(embedding_list))
                if len(embedding_list) > 0
                else None
            ),
        )
        counterfactual_obs = pd.DataFrame(covariates)
        counterfactual_obs[self.info["perturbation_key"]] = (
            restore_perturbation_combinations(
                perturbations,
                self.info["perturbation_combination_delimiter"],
                self.info["perturbation_control_value"],
            )
        )

        return (counterfactual_batch, counterfactual_obs)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            idx = [idx]
        counterfactual_batch, counterfactual_obs = self._get_counterfactual(idx)
        if self.transform is not None:
            counterfactual_batch = self.transform(counterfactual_batch)
        return counterfactual_batch, counterfactual_obs

    @staticmethod
    def from_anndata(
        control_adata: ad.AnnData,
        prediction_dataframe: pd.DataFrame,
        perturbation_key: str,
        perturbation_combination_delimiter: str | None = "+",
        covariate_keys: list[str] | None = None,
        perturbation_control_value: str | None = None,
        seed: int = 0,
        max_control_cells_per_covariate: int = 1000,
        embedding_key: str | None = None,
    ) -> tuple[Counterfactual, dict[str, Any]]:
        """Create a Counterfactual dataset from a control AnnData object and a
            prediction_dataframe containing desired counterfactual predictions.

        Args:
            control_adata: an AnnData object with control cells
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

        Returns:
            A tuple (dataset, info), where dataset is a Counterfactual
              dataset and info is a dictionary containing supplementary
              information about the dataset not contained in the dataset itself
              (e.g. perturbation and covariate unique values). This information
              can be used to setup data pipelines.
        """
        if covariate_keys is None:
            covariate_keys = ["dummy_covariate"]
            prediction_dataframe["dummy_covariate"] = "1"
            control_adata.obs["dummy_covariate"] = "1"

        if perturbation_control_value is None:
            raise ValueError("Must specify perturbation_control_value")

        prediction_dataframe[perturbation_key] = prediction_dataframe[
            perturbation_key
        ].astype("category")
        perturbations, perturbation_uniques = parse_perturbation_combinations(
            prediction_dataframe[perturbation_key],
            perturbation_combination_delimiter,
            perturbation_control_value,
        )
        covariates, covariate_uniques = get_covariates(
            prediction_dataframe, covariate_keys
        )

        info = dict(
            perturbation_uniques=perturbation_uniques,
            covariate_uniques=covariate_uniques,
            perturbation_key=perturbation_key,
            covariate_keys=covariate_keys,
            perturbation_combination_delimiter=perturbation_combination_delimiter,
            perturbation_control_value=perturbation_control_value,
        )

        control_indexes = build_control_dict(control_adata.obs, covariate_keys)
        ## Sample control cells if above max_control_cells_per_covariate
        random.seed(seed)
        sampled_control_indexes = FrozenDictKeyMap()
        for covs in control_indexes:
            sampled_control_indexes[covs] = random.sample(
                list(control_indexes[covs]),
                k=min(max_control_cells_per_covariate, len(control_indexes[covs])),
            )

        ## Optionally extract embeddings
        control_embeddings = (
            control_adata.obsm[embedding_key] if embedding_key is not None else None
        )

        # Create perturbation dataset
        dataset = Counterfactual(
            perturbations=perturbations,
            covariates=covariates,
            control_expression=control_adata.X,
            control_indexes=sampled_control_indexes,
            gene_names=control_adata.var_names.to_list(),
            info=info,
            control_embeddings=control_embeddings,
        )

        return (
            dataset,
            info,
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

    def _get_reference(self, idx_list):
        adata_list = []
        controls_idx = set()
        for idx in idx_list:
            key_dict = {cov: value[idx] for cov, value in self.covariates.items()}
            key_dict["_parsed_perturbations"] = frozenset(self.perturbations[idx])
            adata_list.append(self.reference_adata[self.reference_indexes[key_dict]])

            key_dict["_parsed_perturbations"] = frozenset()
            controls_idx.update(self.reference_indexes[key_dict])

        reference_adata_idx = ad.concat(adata_list)
        reference_adata_idx = reference_adata_idx[
            reference_adata_idx.obs[self.info["perturbation_key"]]
            != self.info["perturbation_control_value"]
        ]
        controls_idx = list(controls_idx)
        reference_adata_idx = ad.concat(
            [reference_adata_idx, self.reference_adata[controls_idx, :]]
        )
        return reference_adata_idx

    def __getitem__(self, idx):
        if isinstance(idx, int):
            idx = [idx]
        counterfactual_batch, counterfactual_obs = super(
            CounterfactualWithReference, self
        ).__getitem__(idx)
        reference_adata = self._get_reference(idx)
        return counterfactual_batch, counterfactual_obs, reference_adata

    @staticmethod
    def from_anndata(
        adata: ad.AnnData,
        perturbation_key: str,
        perturbation_combination_delimiter: str | None = "+",
        covariate_keys: list[str] | None = None,
        perturbation_control_value: str | None = None,
        seed: int = 0,
        max_control_cells_per_covariate: int = 1000,
        embedding_key: str | None = None,
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
        if covariate_keys is None:
            covariate_keys = ["dummy_covariate"]
            adata.obs["dummy_covariate"] = "1"

        if perturbation_control_value is None:
            raise ValueError("Must specify perturbation_control_value")

        if perturbation_control_value not in adata.obs[perturbation_key].unique():
            raise ValueError(
                f"perturbation_control_value {perturbation_control_value} not in "
                f"{perturbation_key} column of adata"
            )

        prediction_dataframe = unique_perturbation_dataframe(
            adata,
            perturbation_key,
            covariate_keys,
        )

        control_adata = adata[adata.obs[perturbation_key] == perturbation_control_value]
        counterfactual_dataset, info = Counterfactual.from_anndata(
            control_adata=control_adata,
            prediction_dataframe=prediction_dataframe,
            perturbation_key=perturbation_key,
            perturbation_combination_delimiter=perturbation_combination_delimiter,
            covariate_keys=covariate_keys,
            perturbation_control_value=perturbation_control_value,
            seed=seed,
            max_control_cells_per_covariate=max_control_cells_per_covariate,
            embedding_key=embedding_key,
        )

        parsed_adata_perturbations = parse_perturbation_combinations(
            adata.obs[perturbation_key],
            perturbation_combination_delimiter,
            perturbation_control_value,
        )[0]
        adata.obs["_parsed_perturbations"] = [
            frozenset(x) for x in parsed_adata_perturbations
        ]
        reference_indexes = build_control_dict(
            adata.obs, ["_parsed_perturbations"] + covariate_keys
        )

        # Create perturbation dataset
        dataset = CounterfactualWithReference(
            perturbations=counterfactual_dataset.perturbations,
            covariates=counterfactual_dataset.covariates,
            control_expression=counterfactual_dataset.control_expression,
            control_indexes=counterfactual_dataset.control_indexes,
            gene_names=counterfactual_dataset.gene_names,
            info=info,
            reference_adata=adata,
            reference_indexes=reference_indexes,
            control_embeddings=counterfactual_dataset.control_embeddings,
        )

        return (
            dataset,
            info,
        )
