from __future__ import annotations
from copy import deepcopy
from typing import Callable, Sequence
from dataclasses import dataclass
import pickle

import pandas as pd
import anndata as ad
import numpy as np

from .sc_perturbation import SingleCellPerturbation
from ...transforms.base import Transform
from ...transforms.samplers import RowSampler

__all__ = ["SingleCellPerturbationWithControls"]


@dataclass
class SingleCellPerturbationWithControls(SingleCellPerturbation):
    """Single Cell Perturbation Dataset with Controls.

    This dataset inherits from SingleCellPerturbation and contains both perturbed
    and control cells within the main dataset. Control cells are identified and
    mapped to each cell (both perturbed and control) based on matching covariate
    conditions.

    Attributes:
        control_indices_dict: a dictionary that maps from each dataset index (both
          perturbed and control cells) to a list of control cell indices within the
          same dataset. This provides direct O(1) access to controls without covariate
          lookups. Control cells map to themselves or other controls with matching
          covariates.
        perturbed_indices: a list of indices identifying which cells in the dataset
          are perturbed (vs control). Control cells are those not in this list.
    """

    control_indices_dict: dict[int, Sequence[int]] | None = None
    perturbed_indices: Sequence[int] | None = None
    controls_sampler: Transform | Callable | None = None
    controls_transform: Transform | Callable | None = None
    num_control_samples: int = 1

    def __post_init__(self, *args, **kwargs):
        super().__post_init__(*args, **kwargs)
        self.controls_sampler = RowSampler(num_samples=self.num_control_samples)

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

    def _get_batch(self, indices):
        batch = super()._get_batch(indices)
        if batch.controls is not None:
            raise ValueError("Expected controls to be None from parent _get_batch")

        # Get controls for each example using direct index lookup
        batch_control_indices = [np.array(self.control_indices_dict[idx]) for idx in indices]
        batch = batch._replace(controls=batch_control_indices)
        return batch

    def __getitems__(self, indices):
        # batch_with_control_indices = super().__getitems__(indices)
        # we don't want to encode perturbation/covariates yet, so use _get_batch
        batch_with_control_indices = self._get_batch(indices)

        # Convert indices to batch with controls
        control_indices: list = batch_with_control_indices.controls
        if self.batch_controls_sampler is not None:
            # Sample n matching controls for each index
            control_indices = self.batch_controls_sampler(control_indices)

        # Go from indices to controls
        controls = np.concatenate(
            [self.gene_expression[control_indices[i], :].copy().mean(axis=0) for i in range(0, len(control_indices))],
            axis=0
        )
        control_embeddings = np.stack(
            [self.embeddings[control_indices[i], :].copy().mean(axis=0) for i in range(0, len(control_indices))],
        ) if self.embeddings is not None else None

        # Apply transform that operates on controls rather than indices
        if self.controls_transform is not None:  # usually just a todense -> tofloat
            controls = self.controls_transform(controls)
            control_embeddings = self.controls_transform(control_embeddings) if control_embeddings is not None else None

        batch = batch_with_control_indices._replace(controls=controls)
        batch = batch._replace(control_embeddings=control_embeddings)

        if self.transform is not None:
            batch = self.transform(batch)

        return batch

    @staticmethod
    def from_anndata(
            adata: ad.AnnData,
            perturbation_key: str,
            split: pd.Series | np.ndarray | None = None,
            perturbation_combination_delimiter: str | None = "+",
            covariate_keys: list[str] | None = None,
            perturbation_control_value: str | None = None,
            embedding_key: str | None = None,
            perturbation_embeddings_path: str | None = None,
            feature_filter_path: str | None = None,
            num_control_samples: int = 1,
            control_indices_dict_path: str | None = None,
    ):
        """Create a SingleCellPerturbationWithControls dataset from an AnnData object.

        Args:
            adata: an AnnData object
            perturbation_key: the key in adata.obs that contains the perturbations
            split: if specified, only use the subset of adata specified by split
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
            num_control_samples: number of control samples to draw per perturbation
            control_indices_dict_path: optional path to a pickle file containing a dictionary
              that maps from cell IDs to lists of control cell IDs. The dictionary will be
              loaded and converted to dataset indices. If not provided, will be built from
              covariate mappings during dataset initialization.

        Returns:
            A tuple (dataset, info), where dataset is a SingleCellPerturbationWithControls
            dataset and info is a dictionary containing supplementary information
            about the dataset not contained in the dataset itself (e.g. perturbation
            and covariate unique values). This information can be used to setup data
            pipelines.

        Raises:
            ValueError: if perturbation_control_value is None, or if control_indices_dict_path
              is provided but contains invalid data (wrong size, control indices that are 
              actually perturbed cells, etc.)
        """
        if split is None:
            split = list(range(0, adata.shape[0]))
        elif isinstance(split, pd.Series):
            split = split.values

        if perturbation_control_value is None:
            raise ValueError("Must specify perturbation_control_value")

        # Don't filter - use all cells in split (both perturbed and control)
        dataset, info = SingleCellPerturbation.from_anndata(
            adata,
            perturbation_key=perturbation_key,
            split=split,  # All cells, not just perturbed
            perturbation_combination_delimiter=perturbation_combination_delimiter,
            covariate_keys=covariate_keys,
            perturbation_control_value=perturbation_control_value,
            embedding_key=embedding_key,
            perturbation_embeddings_path=perturbation_embeddings_path,
            feature_filter_path=feature_filter_path,
        )

        # Identify which indices are perturbed vs control in the dataset
        # Control cells will have empty perturbation lists
        perturbed_indices_list = [i for i in range(len(dataset)) if len(dataset.perturbations[i]) > 0]
        control_indices_set = set(range(len(dataset))) - set(perturbed_indices_list)

        # Build control_indices_dict for ALL cells
        if control_indices_dict_path is not None:
            # Load control indices from pickle file
            with open(control_indices_dict_path, 'rb') as f:
                cell_id_to_control_ids = pickle.load(f)

            # Create mapping from cell ID to dataset index
            cell_id_to_idx = {cell_id: idx for idx, cell_id in enumerate(dataset.cell_ids)}

            # Pre-compute valid cell IDs (those in the dataset)
            valid_cell_ids = set(cell_id_to_idx.keys())

            # Pre-filter: only process cells that are in the dataset
            filtered_cell_id_to_control_ids = {
                cell_id: control_ids
                for cell_id, control_ids in cell_id_to_control_ids.items()
                if cell_id in valid_cell_ids
            }

            # Convert cell IDs to dataset indices
            control_indices_dict = {}
            for cell_id, control_cell_ids in filtered_cell_id_to_control_ids.items():
                cell_idx = cell_id_to_idx[cell_id]
                control_indices = np.array([cell_id_to_idx[control_cell] for control_cell in control_cell_ids if
                                            control_cell in valid_cell_ids])
                control_indices_dict[cell_idx] = control_indices

            # Validate the loaded dictionary
            dataset_size = len(dataset)
            if len(control_indices_dict) != dataset_size:
                raise ValueError(
                    f"Control indices dictionary has {len(control_indices_dict)} entries "
                    f"but dataset has {dataset_size} cells. All cells must have control mappings."
                )

        else:
            # Build covariate-to-control mapping
            # Filter to categorical covariates only using covariate_uniques from info
            # Continuous covariates have None values in covariate_uniques
            categorical_keys = sorted([
                k for k in dataset.covariates.keys()
                if info["covariate_uniques"].get(k) is not None
            ])

            # Get control cells' covariates
            control_covariates_dict = {}
            for ctrl_idx in control_indices_set:
                cov_key = tuple((k, dataset.covariates[k][ctrl_idx]) for k in categorical_keys)
                if cov_key not in control_covariates_dict:
                    control_covariates_dict[cov_key] = []
                control_covariates_dict[cov_key].append(ctrl_idx)

            # Map all cells to their controls
            control_indices_dict = {}
            for i in range(len(dataset)):
                cov_key = tuple((k, dataset.covariates[k][i]) for k in categorical_keys)

                if cov_key in control_covariates_dict:
                    control_indices_dict[i] = control_covariates_dict[cov_key]
                else:
                    covariates = {k: dataset.covariates[k][i] for k in categorical_keys}
                    raise ValueError(f"No controls found for covariate condition {covariates}")

        return SingleCellPerturbationWithControls(
            gene_expression=dataset.gene_expression,
            perturbations=dataset.perturbations,
            covariates=dataset.covariates,
            cell_ids=dataset.cell_ids,
            gene_names=dataset.gene_names,
            transform=dataset.transform,
            control_indices_dict=control_indices_dict,
            perturbed_indices=perturbed_indices_list,
            num_control_samples=num_control_samples,
            embeddings=dataset.embeddings,
            perturbation_embeddings=dataset.perturbation_embeddings,
        ), info
