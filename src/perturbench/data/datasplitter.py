from sklearn.model_selection import train_test_split
from collections import defaultdict
import pandas as pd
import numpy as np
import random
from omegaconf import DictConfig
import os


class PerturbationDataSplitter:
    """Class to split data into train/test/test.

    Attributes:
        obs_dataframe: Dataframe with cell/sample level metadata.
          Must contain a column with the perturbation key
        perturbation_key: Column name of containing the perturbation identity
          of each cell/sample
        covariate_keys:  Column name(s) of the covariates to split on
          (i.e. `cell_type`). If None, a dummy covariate will be created
        perturbation_control_value: Identity of control perturbation
        perturbation_combination_delimiter: Delimiter separating multiple
          perturbations.
    """

    obs_dataframe: pd.DataFrame
    perturbation_key: str
    perturbation_control_value: str
    covariate_keys: list[str] | str | None = None
    perturbation_combination_delimiter: str = "+"

    @staticmethod
    def split_dataset(
        splitter_config: DictConfig,
        obs_dataframe: pd.DataFrame,
        perturbation_key: str,
        perturbation_combination_delimiter: str,
        perturbation_control_value: str,
    ):
        """Split dataset into train/val/test depending on task specified in
           splitter config.

        Args
            splitter_config: Configuration for the splitter
            obs_dataframe: Dataframe with cell/sample level metadata.
            perturbation_key: Dataframe column containing the perturbation
            perturbation_combination_delimiter: Delimiter separating multiple
                perturbations.
            perturbation_control_value: Identity of control perturbation

        Returns:
            split_dict: Dictionary with keys 'train', 'val', 'test' and values
                as numpy arrays of indices for each split
        """
        # AnnData Split
        if "split_path" in splitter_config:
            split = pd.read_csv(
                splitter_config.split_path, index_col=0, header=None
            ).iloc[:, 0]
        else:
            perturbation_datasplitter = PerturbationDataSplitter(
                obs_dataframe=obs_dataframe,
                perturbation_key=perturbation_key,
                covariate_keys=list(splitter_config.covariate_keys),
                perturbation_control_value=perturbation_control_value,
                perturbation_combination_delimiter=perturbation_combination_delimiter,
            )
            if splitter_config.task == "transfer":
                split = perturbation_datasplitter.split_covariates(
                    seed=splitter_config.splitter_seed,
                    min_train_covariates=splitter_config.min_train_covariates,
                    max_heldout_covariates=splitter_config.max_heldout_covariates,
                    max_heldout_fraction_per_covariate=splitter_config.max_heldout_fraction_per_covariate,
                    train_control_fraction=splitter_config.train_control_fraction,
                    downsample_fraction=splitter_config.downsample_fraction,
                )
            elif splitter_config.task == "combine":
                split = perturbation_datasplitter.split_combinations(
                    seed=splitter_config.splitter_seed,
                    max_heldout_fraction_per_covariate=splitter_config.max_heldout_fraction_per_covariate,
                    train_control_fraction=splitter_config.train_control_fraction,
                    downsample_fraction=splitter_config.downsample_fraction,
                )
            elif splitter_config.task == "combine_inverse":
                split = perturbation_datasplitter.split_combinations_inverse(
                    seed=splitter_config.splitter_seed,
                    max_heldout_fraction_per_covariate=splitter_config.max_heldout_fraction_per_covariate,
                    train_control_fraction=splitter_config.train_control_fraction,
                    downsample_fraction=splitter_config.downsample_fraction,
                )
            else:
                raise ValueError(
                    'splitter_config.task must be "transfer", "combine", or "combine_inverse"'
                )

        assert len(split) == obs_dataframe.shape[0]
        assert split.index.equals(obs_dataframe.index)
        for split_value in ["train", "val", "test"]:
            assert split_value in split.unique()

        if splitter_config.get("save"):
            if not os.path.exists(splitter_config.output_path):
                os.makedirs(splitter_config.output_path)
            try:
                split.to_csv(
                    splitter_config.output_path + "train_test_split.csv", index=True
                )
            except PermissionError:
                print(f"Warning: Unable to save split to {splitter_config.output_path}")

        split_dict = {
            split_val: np.where(split == split_val)[0]
            for split_val in ["train", "val", "test"]
        }
        return split_dict

    def __init__(
        self,
        obs_dataframe: pd.DataFrame,
        perturbation_key: str,
        perturbation_control_value: str,
        covariate_keys: list[str] | str | None = None,
        perturbation_combination_delimiter: str = "+",
    ):
        """Initialize PerturbationDataSplitter object."""
        self.obs_dataframe = obs_dataframe
        self.perturbation_key = perturbation_key

        assert (
            perturbation_control_value in self.obs_dataframe[perturbation_key].unique()
        )

        if covariate_keys is None:
            obs_dataframe["dummy_cov"] = "1"
            self.covariate_keys = ["dummy_cov"]
        elif isinstance(covariate_keys, str):
            self.covariate_keys = [covariate_keys]
        else:
            self.covariate_keys = covariate_keys

        self.perturbation_control_value = perturbation_control_value
        self.perturbation_combination_delimiter = perturbation_combination_delimiter

        self.split_params = {}
        self.summary_dataframes = {}

        covariates_list = [
            list(obs_dataframe[covariate_key].values)
            for covariate_key in self.covariate_keys
        ]
        self.covariates_merged = [
            frozenset(covariates) for covariates in zip(*covariates_list)
        ]
        self.perturbation_covariates = [
            (perturbation, covariates)
            for perturbation, covariates in zip(
                obs_dataframe[perturbation_key], self.covariates_merged
            )
        ]

    def _assign_split(
        self,
        seed: int,
        train_perturbation_covariates: list[tuple[str, frozenset[str]]],
        heldout_perturbation_covariates: list[tuple[str, frozenset[str]]],
        split_key: str,
        test_fraction: float = 0.5,
    ):
        covariate_counts = defaultdict(int)
        for _, covariates in heldout_perturbation_covariates:
            covariate_counts[covariates] += 1
        heldout_perturbation_covariates = [
            (perturbation, covariates)
            for perturbation, covariates in heldout_perturbation_covariates
            if covariate_counts[covariates] > 1
        ]
        validation_perturbation_covariates, test_perturbation_covariates = (
            train_test_split(
                heldout_perturbation_covariates,
                stratify=[
                    str(covariates) for _, covariates in heldout_perturbation_covariates
                ],
                test_size=test_fraction,  ## Split test and test perturbations evenly
                random_state=seed,
            )
        )

        train_perturbation_covariates = set(train_perturbation_covariates)
        validation_perturbation_covariates = set(validation_perturbation_covariates)
        test_perturbation_covariates = set(test_perturbation_covariates)

        self.obs_dataframe[split_key] = [None] * self.obs_dataframe.shape[0]
        self.obs_dataframe.loc[
            [x in train_perturbation_covariates for x in self.perturbation_covariates],
            split_key,
        ] = "train"
        self.obs_dataframe.loc[
            [
                x in validation_perturbation_covariates
                for x in self.perturbation_covariates
            ],
            split_key,
        ] = "val"
        self.obs_dataframe.loc[
            [x in test_perturbation_covariates for x in self.perturbation_covariates],
            split_key,
        ] = "test"

    def _split_controls(
        self,
        seed,
        split_key,
        train_control_fraction,
    ):
        random.seed(seed)
        ctrl_ix = list(
            self.obs_dataframe.loc[
                self.obs_dataframe[self.perturbation_key]
                == self.perturbation_control_value
            ].index
        )
        ctrl_ix = random.sample(ctrl_ix, k=len(ctrl_ix))

        val_control_frac = (1 - train_control_fraction) / 2

        train_ctrl_ix, val_ctrl_ix, test_ctrl_ix = np.split(
            ctrl_ix,
            [
                int(train_control_fraction * len(ctrl_ix)),
                int((val_control_frac + train_control_fraction) * len(ctrl_ix)),
            ],
        )

        self.obs_dataframe.loc[train_ctrl_ix, split_key] = "train"
        self.obs_dataframe.loc[val_ctrl_ix, split_key] = "val"
        self.obs_dataframe.loc[test_ctrl_ix, split_key] = "test"

    def _summarize_split(self, split_key):
        unique_covariates_merged = [x for x in set(self.covariates_merged)]
        split_summary_df = pd.DataFrame(
            0,
            index=[str(tuple(x)) for x in unique_covariates_merged],
            columns=["train", "val", "test"],
        )
        for covariates in unique_covariates_merged:
            obs_df_sub = self.obs_dataframe.loc[
                [x == covariates for x in self.covariates_merged]
            ]
            train_perts = obs_df_sub.loc[
                obs_df_sub[split_key] == "train", self.perturbation_key
            ].unique()
            val_perts = obs_df_sub.loc[
                obs_df_sub[split_key] == "val", self.perturbation_key
            ].unique()
            test_perts = obs_df_sub.loc[
                obs_df_sub[split_key] == "test", self.perturbation_key
            ].unique()

            split_summary_df.loc[str(tuple(covariates)), "train"] = len(train_perts)
            split_summary_df.loc[str(tuple(covariates)), "val"] = len(val_perts)
            split_summary_df.loc[str(tuple(covariates)), "test"] = len(test_perts)

        return split_summary_df

    def _downsample_combinatorial_perturbations(
        self, seed: int, downsample_fraction: float
    ):
        unique_perturbations = list(self.obs_dataframe[self.perturbation_key].unique())
        num_sample = np.round(
            len(unique_perturbations) * downsample_fraction, 0
        ).astype(int)

        random.seed(seed)
        sampled_perturbations = random.sample(unique_perturbations, k=num_sample)
        sampled_single_perturbations = set(
            [
                x
                for x in sampled_perturbations
                if self.perturbation_combination_delimiter not in x
            ]
        )
        sampled_single_perturbations.add(self.perturbation_control_value)

        sampled_combo_perturbations = []
        for combo_pert in [
            x
            for x in sampled_perturbations
            if self.perturbation_combination_delimiter in x
        ]:
            single_pert_list = combo_pert.split(self.perturbation_combination_delimiter)
            if all([x in sampled_single_perturbations for x in single_pert_list]):
                sampled_combo_perturbations.append(combo_pert)

        sampled_perturbations = sampled_single_perturbations.union(
            sampled_combo_perturbations
        )
        return sampled_perturbations

    def split_covariates(
        self,
        print_split: bool = True,
        seed: int = 54,
        min_train_covariates: int = 1,
        max_heldout_covariates: int = 2,
        max_heldout_fraction_per_covariate: float = 0.3,
        max_heldout_perturbations_per_covariate: int = 200,
        train_control_fraction: float = 0.5,
        test_fraction: float = 0.5,
        downsample_fraction: float = 1.0,
    ):
        """Holds out perturbations in specific covariates to test the ability
             of a model to transfer perturbation effects to new covariates.

        Args
            print_split: Whether to print the split summary
            seed: Random seed for reproducibility
            min_train_covariates: Minimum number of covariates to include in the
              training set. Must be at least one.
            max_heldout_covariates: Maximum number of covariates to hold out for each
              perturbation. Must be at least one.
            max_heldout_fraction_per_cov: Maximum fraction of perturbations to
              hold out for each unique set of covariates
            test_fraction: Fraction of held out perturbations to include in the test
              vs val set
            train_control_fraction: Fraction of control cells to include in the
              training set

        Returns
            split: Split of the data into train/val/test as a pd.Series
        """

        split_key = "transfer_split_seed" + str(seed)  ## Unique key for this split
        self.split_params[split_key] = {
            "min_train_covariates": min_train_covariates,
            "max_heldout_covariates": max_heldout_covariates,
            "max_heldout_fraction_per_cov": max_heldout_fraction_per_covariate,
            "train_control_fraction": train_control_fraction,
        }

        max_heldout_dict = (
            {}
        )  ## Maximum number of perturbations that can be held out for each unique set of covariates
        for cov_keys, df in self.obs_dataframe.groupby(self.covariate_keys):
            num_cov_perts = df[self.perturbation_key].nunique()
            max_heldout_dict[frozenset(cov_keys)] = min(
                max_heldout_fraction_per_covariate * num_cov_perts,
                max_heldout_perturbations_per_covariate,
            )

        perturbation_covariates_dict = (
            {}
        )  ## Dictionary to store unique covariates for each perturbation
        for pert_key, df in self.obs_dataframe.groupby([self.perturbation_key]):
            pert_key = pert_key[0]
            if pert_key != self.perturbation_control_value:
                unique_pert_covs = []
                cov_key_df = df.loc[:, self.covariate_keys].drop_duplicates()
                unique_pert_covs = [frozenset(x) for x in cov_key_df.values]
                perturbation_covariates_dict[pert_key] = unique_pert_covs

        ## Sort by number of covariates
        perturbation_covariates_dict = dict(
            sorted(
                perturbation_covariates_dict.items(),
                key=lambda item: len(item[1]),
                reverse=True,
            )
        )

        ## Downsample
        if downsample_fraction < 1.0:
            random.seed(seed)
            perturbations_keep = set(
                random.sample(
                    list(perturbation_covariates_dict.keys()),
                    int(downsample_fraction * len(perturbation_covariates_dict)),
                )
            )
            perturbation_covariates_dict = {
                k: v
                for k, v in perturbation_covariates_dict.items()
                if k in perturbations_keep
            }

        rng = np.random.RandomState(seed)
        seed_list = [
            rng.randint(100000) for i in range(0, len(perturbation_covariates_dict))
        ]

        num_total_covs = len(max_heldout_dict)
        num_heldout_dict = defaultdict(
            int
        )  ## Counter for number of perturbations held out for each unique set of covariates

        ## Iterate through each perturbation and choose a random subset of covariates to hold out for that perturbation
        train_perturbation_covariates = []
        heldout_perturbation_covariates = []
        for i, items in enumerate(perturbation_covariates_dict.items()):
            perturbation, covariate_list = items
            num_covariates = len(covariate_list)
            sampled_covariates = []
            if num_covariates > min_train_covariates:
                covariate_pool = [
                    covariates
                    for covariates in covariate_list
                    if num_heldout_dict[covariates] < max_heldout_dict[covariates]
                ]  ## Check if the maximum number of perturbations have been held out for this set of covariates

                if len(covariate_pool) > 0:
                    random.seed(seed_list[i])
                    num_sample_range = (
                        1,
                        np.max(
                            [
                                len(covariate_pool)
                                - num_total_covs
                                + max_heldout_covariates,
                                1,
                            ]
                        ),
                    )
                    num_sample = random.randint(
                        num_sample_range[0], num_sample_range[1]
                    )
                    sampled_covariates = random.sample(
                        covariate_pool, num_sample
                    )  ## Sample a random subset of covariates to hold out

                    for covariates in sampled_covariates:
                        num_heldout_dict[covariates] += 1

                    heldout_perturbation_covariates.extend(
                        [
                            (perturbation, covariates)
                            for covariates in sampled_covariates
                        ]
                    )

            train_perturbation_covariates.extend(
                [
                    (perturbation, covariates)
                    for covariates in covariate_list
                    if covariates not in sampled_covariates
                ]
            )

        ## Split held out perturbation/covariate pairs into val and test sets
        self._assign_split(
            seed,
            train_perturbation_covariates,
            heldout_perturbation_covariates,
            split_key,
            test_fraction=test_fraction,
        )

        ## Split control cells
        self._split_controls(seed, split_key, train_control_fraction)

        ## Print split
        split_summary_df = self._summarize_split(split_key)
        self.summary_dataframes[split_key] = split_summary_df
        if print_split:
            print("Split summary: ")
            print(split_summary_df)

        split = self.obs_dataframe[split_key]

        return split

    def split_combinations(
        self,
        print_split: bool = True,
        seed: int = 54,
        max_heldout_fraction_per_covariate: float = 0.3,
        max_heldout_perturbations_per_covariate: int = 200,
        train_control_fraction: float = 0.5,
        test_fraction: float = 0.5,
        downsample_fraction: float = 1.0,
    ):
        """Holds out combinations of perturbations to test the ability of a
          model to learn the combined effect of multiple perturbations from
          individual perturbations.

        Args
            print_split: Whether to print the split summary
            seed: Random seed for reproducibility
            max_heldout_fraction_per_covariate: Maximum fraction of
              perturbations to hold out for each unique set of covariates
            train_control_fraction: Fraction of control cells to include in the
              training set

        Returns
            split: Split of the data into train/val/test as a pd.Series
        """
        split_key = "combo_split_seed" + str(seed)  ## Unique key for this split
        self.split_params[split_key] = {
            "max_heldout_fraction_per_cov": max_heldout_fraction_per_covariate,
            "train_control_fraction": train_control_fraction,
        }

        if downsample_fraction < 1.0:
            perturbations = self._downsample_combinatorial_perturbations(
                seed, downsample_fraction
            )
        else:
            perturbations = self.obs_dataframe[self.perturbation_key].unique()

        train_perturbation_covariates = []
        heldout_perturbation_covariates = []
        for covariates, df in self.obs_dataframe.groupby(self.covariate_keys):
            covariates = frozenset(covariates)
            cov_perturbations = [
                x for x in df[self.perturbation_key].unique() if x in perturbations
            ]

            random.seed(seed)
            combo_perturbations = [
                x
                for x in cov_perturbations
                if self.perturbation_combination_delimiter in x
            ]
            num_heldout_cov = min(
                int(max_heldout_fraction_per_covariate * len(combo_perturbations)),
                max_heldout_perturbations_per_covariate,
            )
            heldout_combo_perturbations = random.sample(
                combo_perturbations, num_heldout_cov
            )
            heldout_perturbation_covariates.extend(
                [
                    (perturbation, covariates)
                    for perturbation in heldout_combo_perturbations
                ]
            )
            train_perturbation_covariates.extend(
                [
                    (perturbation, covariates)
                    for perturbation in cov_perturbations
                    if perturbation not in heldout_combo_perturbations
                ]
            )

        self._assign_split(
            seed,
            train_perturbation_covariates,
            heldout_perturbation_covariates,
            split_key,
            test_fraction=test_fraction,
        )
        self._split_controls(seed, split_key, train_control_fraction)

        ## Print split
        split_summary_df = self._summarize_split(split_key)
        self.summary_dataframes[split_key] = split_summary_df
        if print_split:
            print(split_summary_df)

        split = self.obs_dataframe[split_key]
        return split

    def split_combinations_inverse(
        self,
        print_split: bool = True,
        seed: int = 54,
        max_heldout_fraction_per_covariate: float = 0.3,
        max_heldout_perturbations_per_covariate: int = 200,
        train_control_fraction: float = 0.5,
        test_fraction: float = 0.5,
        downsample_fraction: float = 1.0,
    ):
        """Holds out single perturbations to test the ability of a model to
          learn single perturbation effects given a combination of perturbation
          and other single perturbations.

        Args
            print_split: Whether to print the split summary
            seed: Random seed for reproducibility
            max_heldout_fraction_per_covariate: Maximum fraction of
              perturbations to hold out for each unique set of covariates
            train_control_fraction: Fraction of control cells to include in the
              training set

        Returns
            split: Split of the data into train/val/test as a pd.Series
        """
        split_key = "inverse_combo_split_seed" + str(seed)  ## Unique key for this split
        self.split_params[split_key] = {
            "max_heldout_fraction_per_cov": max_heldout_fraction_per_covariate,
            "train_control_fraction": train_control_fraction,
        }

        if downsample_fraction < 1.0:
            perturbations = self._downsample_combinatorial_perturbations(
                seed, downsample_fraction
            )
        else:
            perturbations = self.obs_dataframe[self.perturbation_key].unique()

        train_perturbation_covariates = []
        heldout_perturbation_covariates = []
        for covariates, df in self.obs_dataframe.groupby(self.covariate_keys):
            covariates = frozenset(covariates)
            cov_perturbations = [
                x for x in perturbations if x in df[self.perturbation_key].unique()
            ]

            random.seed(seed)
            combo_to_single_dict = {
                x: x.split(self.perturbation_combination_delimiter)
                for x in cov_perturbations
                if self.perturbation_combination_delimiter in x
            }
            single_to_combo_dict = defaultdict(list)
            for combo, singles in combo_to_single_dict.items():
                for single in singles:
                    single_to_combo_dict[single].append(combo)

            single_perturbations = [
                x for x in cov_perturbations if x not in combo_to_single_dict
            ]
            single_perturbation_pool = random.sample(
                single_perturbations, len(single_perturbations)
            )
            total_perturbations_heldout = min(
                int(max_heldout_fraction_per_covariate * len(single_perturbations)),
                max_heldout_perturbations_per_covariate,
            )
            heldout_perturbations = []
            while len(heldout_perturbations) < total_perturbations_heldout:
                perturbation = single_perturbation_pool.pop()
                heldout_perturbations.append(perturbation)

                perturbation_combos = single_to_combo_dict[perturbation]
                perturbations_remove = []
                for combo in perturbation_combos:
                    if combo in combo_to_single_dict:
                        perturbations_remove.extend(combo_to_single_dict[combo])
                single_perturbation_pool = [
                    x for x in single_perturbation_pool if x not in perturbations_remove
                ]

                if len(single_perturbation_pool) == 0:
                    break

            heldout_perturbation_covariates.extend(
                [(perturbation, covariates) for perturbation in heldout_perturbations]
            )
            train_perturbation_covariates.extend(
                [
                    (perturbation, covariates)
                    for perturbation in cov_perturbations
                    if perturbation not in heldout_perturbations
                ]
            )

        self._assign_split(
            seed,
            train_perturbation_covariates,
            heldout_perturbation_covariates,
            split_key,
            test_fraction=test_fraction,
        )
        self._split_controls(seed, split_key, train_control_fraction)

        ## Print split
        split_summary_df = self._summarize_split(split_key)
        self.summary_dataframes[split_key] = split_summary_df
        if print_split:
            print(split_summary_df)

        split = self.obs_dataframe[split_key]
        return split
