from __future__ import annotations
import logging

import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
from pandas.api.types import CategoricalDtype

import torch
from torch.utils.data import (
    DataLoader,
    BatchSampler,
    RandomSampler,
    SequentialSampler,
    WeightedRandomSampler,
)

from .types import FrozenDictKeyMap

log = logging.getLogger(__name__)


def unique_perturbations(p: pd.Series, combo_delim: str = "+"):
    """Return unique perturbations from a pandas Series"""
    unique_perts = set()
    for pert in p.unique():
        unique_perts.update(pert.split(combo_delim))
    return list(unique_perts)


def parse_perturbation_combinations(
    combined_perturbations: pd.Series,
    delimiter: str | None = "+",
    perturbation_control_value: str | None = "control",
) -> tuple[ArrayLike, list[str]]:
    """Get all perturbations applied to each cell.

    Args:
        combined_perturbations: combined perturbations string representation of
          size (n_cells, )
        delimiter: a string that separates individual perturbations
        perturbation_control_value: a string that represents the control perturbation

    Returns:
        A tuple (combinations, unique_perturbations), where combinations is an
          (n_cell, ) array of lists of individual perturbations applied to each
          cell, and unique_perturbations is a set of all unique perturbations.
    """
    assert isinstance(combined_perturbations.dtype, CategoricalDtype)

    # Split the perturbations by the delimiter
    parsed = []
    uniques = (
        {}
    )  ## Store unique perturbations as dictionary keys to ensure ordering is the same
    for combination in combined_perturbations.astype(str):
        perturbation_list = []
        for perturbation in combination.split(delimiter):
            if perturbation != perturbation_control_value:
                perturbation_list.append(perturbation)
                uniques[perturbation] = None
        parsed.append(perturbation_list)

    uniques = list(uniques.keys())
    return parsed, uniques


def restore_perturbation_combinations(
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


def get_covariates(df: pd.DataFrame, covariate_keys: list[str]):
    """Get covariates from a dataframe.

    Args:
        df: a dataframe containing covariates for each cell with n_cells rows
        covariate_keys: a list of covariate keys in the dataframe

    Returns:
        A tuple (covariates, covariate_unique_values), where covariates is a
          dictionary of covariate keys to covariate values with n_cells rows,
          and covariate_unique_values is a dictionary of covariate keys to
          unique covariate values.

    Raises:
        KeyError: if a covariate key is not found in the dataframe.
    """
    try:
        covariates: np.ndarray = df[covariate_keys].values
        covariate_unique_values = {
            cov: list(df[cov].unique()) for cov in covariate_keys
        }
    except KeyError as e:
        raise KeyError("Covariate key not found in dataframe: " + str(e)) from e
    return dict(zip(covariate_keys, covariates.T)), covariate_unique_values


def build_control_dict(
    control_covariate_df: pd.DataFrame,
    covariate_keys: list[str] | None = None,
):
    """Build a dictionary of controls for each covariate condition.

    Args:
        control_covariate_df: pandas dataframe containing the covariates for each sample/cell
        covariate_keys: a list of keys in adata.obs that contain the covariates

    Returns:
        A dictionary of controls, where each control is a sparse matrix of size
          (n_controls_per_condition, n_genes) and the keys are
          dict[covariate_key, covariate_value].
    """
    if covariate_keys is None:
        covariate_keys = control_covariate_df.columns.tolist()

    grouped = control_covariate_df.groupby(
        list(covariate_keys)
    )  # groupby requires a list
    control_indexes = FrozenDictKeyMap()
    for group_key, group_indices in grouped.indices.items():
        if len(covariate_keys) == 1:
            assert isinstance(group_key, (str, int))
            group_key = (group_key,)

        key = dict(zip(covariate_keys, group_key))
        control_indexes[key] = group_indices

    return control_indexes


def batch_dataloader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    shuffle: bool = True,
    oversample: bool = False,
    oversample_root: float = 2.0,
    **kwargs,
):
    """
    Build a PyTorch DataLoader from a PyTorch Dataset using a BatchSampler.

    Args:
        dataset: a PyTorch Dataset
        batch_size: the batch size
        shuffle: whether to shuffle the data
        oversample: whether to oversample the data
        oversample_root: oversampling weight will be
          `(1/class_frac)^(1/oversample_root)`
        kwargs: additional arguments to pass to DataLoaders
    """
    if oversample:
        assert oversample_root > 0, "Oversample root must be greater than 0"

        weights_dictionary = {}
        covariates_df = pd.DataFrame(dataset.covariates)
        covariates_df["concatenated"] = covariates_df.apply(
            lambda row: "_".join(row.values.astype(str)), axis=1
        )
        covariate_fractions = (
            covariates_df["concatenated"].value_counts() / covariates_df.shape[0]
        )
        for covariate, frac in covariate_fractions.items():
            weight = (1 / frac) ** (1.0 / oversample_root)
            weights_dictionary[covariate] = weight

        weights = []
        for i in range(0, len(dataset)):
            cov_values = [values[i] for values in dataset.covariates.values()]
            cov_key = "_".join(cov_values)
            weight = weights_dictionary[cov_key]
            weights.append(weight)

        sampler = WeightedRandomSampler(weights, len(dataset), replacement=True)

    elif shuffle:
        sampler = RandomSampler(dataset)

    else:
        sampler = SequentialSampler(dataset)

    batch_sampler = BatchSampler(sampler, batch_size=batch_size, drop_last=False)
    dataloader = DataLoader(dataset, sampler=batch_sampler, **kwargs)
    return dataloader