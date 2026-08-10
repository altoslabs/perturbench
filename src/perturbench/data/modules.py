import builtins
from enum import StrEnum
from collections.abc import Callable
from typing import Sequence
from pathlib import Path

from omegaconf import DictConfig
import lightning as L
import scanpy as sc
import pandas as pd
from torch.utils.data import DataLoader
from perturbench.modelcore.utils import (
    instantiate_with_context,
    EmptyContextError
)

from .loaders import BatchedDataLoader
from .collate import noop_collate
from .utils import load_dataframe_from_h5, parse_perturbation_combinations

import inspect
import logging

log = logging.getLogger(__name__)


class DataLitModule(L.LightningDataModule):
    """Base Data Module for Perturbation Prediction Models."""

    class Mode(StrEnum):
        """Data access mode."""

        EXAMPLE = "example"
        BATCH = "batch"

    accessor: Callable  # an instance variable with type declared at the class level
    obs_accessor: Callable
    loaders: dict[Mode, DataLoader] = {}
    data_handle: sc.AnnData | None = None
    splits: dict[str, Sequence[int]] | None = None
    control_transform: Callable | None = None

    def __init__(
            self,
            data: DictConfig,
            data_iter_factory: Callable,
            mode: Mode,
            loader: DictConfig | None = None,
            transform: DictConfig | None = None,
            splitter: DictConfig | None = None,
            collate: DictConfig | None = None,
            evaluation: DictConfig | None = None,
            prediction: DictConfig | None = None,
            filter_split_perturbations: bool = True,
    ):
        super().__init__()

        # Set attributes
        self.data = data
        self.data_iter_factory = data_iter_factory
        self.mode = mode
        self.loader = loader or {}
        self.splitter = splitter
        self.collate = collate
        self.evaluation = evaluation
        self.prediction = prediction
        self.filter_split_perturbations = filter_split_perturbations

        # Initialize context
        self.context = {}

        # data_iter_factory is a functools.partial
        # control the pipeline behaviour from here:
        # if data contains pert embeddings, the pipeline will too handle embedding
        data_iter_factory_params = inspect.signature(self.data_iter_factory).parameters
        if 'perturbation_embeddings_path' in data_iter_factory_params and \
                data_iter_factory_params['perturbation_embeddings_path'].default is not None:
            assert data_iter_factory_params['perturbation_embeddings_path'].default.endswith('.parquet')  # standardized
            self.context['use_perturbation_embedding'] = True
        else:
            self.context['use_perturbation_embedding'] = False

        # Try to build an example/batch transform pipeline without context
        if transform is not None:
            try:
                self.transform = instantiate_with_context(transform, self.context)
            except (EmptyContextError, KeyError):
                # Context needed, defer instantiation to train stage
                self.transform = transform

        self.train_iterator = self.val_iterator = self.test_iterator = self.predict_iterator = None

        # Track removed perturbations by split for logging
        self.removed_perturbations: dict[str, set[str]] = {}

    @property
    def mode(self) -> Mode:
        """Data access mode."""
        return self._mode

    @mode.setter
    def mode(self, mode: Mode):
        if mode not in self.loaders:
            raise ValueError(f"Mode {mode} not supported.")
        self._mode = mode

    @property
    def num_genes(self) -> int:
        """Number of genes in the dataset."""
        return len(self.train_iterator.gene_names)

    @property
    def embedding_width(self) -> int | None:
        """Width of the embeddings."""
        if hasattr(self.train_iterator, 'embeddings'):
            if self.train_iterator.embeddings is None:
                return None
            else:
                return self.train_iterator.embeddings.shape[1]

        elif hasattr(self.train_iterator, 'embedding_key'):
            if self.train_iterator.embedding_key is None:
                return None
            else:
                return self.train_iterator.embedding_width

        else:
            raise ValueError("Neither embeddings nor embedding_key is set, so embedding_width cannot be determined.")

    @property
    def num_perturbations(self) -> int:
        """Number of perturbations in the dataset."""
        return len(self.context["perturbation_uniques"])

    @property
    def perturbation_embedding_width(self) -> int | None:
        """Width of the embeddings."""
        if self.train_iterator.perturbation_embeddings is None:
            return None
        else:
            # get embedding from a random key and look at its size
            return len(self.train_iterator.perturbation_embeddings.columns)

    def _filter_split_perturbations(self, split_key: str) -> set[str]:
        """Filter split to remove perturbations not in train.

        Args:
            split_key: The key of the split to filter ("val" or "test")

        Returns:
            Set of perturbations that were removed from the split.
        """
        train_perturbations = set(self.context["perturbation_uniques"])
        split_indices = self.splits[split_key]

        # Get perturbation info from context (populated from train iterator)
        perturbation_key = self.context["perturbation_key"]
        delimiter = self.context["perturbation_combination_delimiter"]
        control_value = self.context["perturbation_control_value"]

        # Get obs_df rows for the split indices
        split_obs = self.obs_df.iloc[split_indices]

        # Parse perturbations for split cells
        perturbation_col = split_obs[perturbation_key].astype('category')
        parsed_perturbations, split_perturbation_uniques, _ = parse_perturbation_combinations(
            perturbation_col,
            delimiter,
            control_value,
        )

        # Find perturbations in split but not in train
        removed_perturbations = set(split_perturbation_uniques) - train_perturbations

        if not removed_perturbations:
            return set()

        # Filter indices: keep only cells where ALL perturbations are in train
        filtered_indices = []
        for idx, cell_perturbations in zip(split_indices, parsed_perturbations):
            # cell_perturbations is a list of individual perturbations for this cell
            # Empty list means control cell - always keep
            if len(cell_perturbations) == 0 or all(p in train_perturbations for p in cell_perturbations):
                filtered_indices.append(idx)

        # Update the split in place
        self.splits[split_key] = filtered_indices

        return removed_perturbations

    def _verify_split(self, split_context: dict, split: str | None = None, split_key: str | None = None) -> bool:
        """Verify that the split dataset contains the same perturbations and
        covariates as the train dataset.

        Args:
            split_context: Context dict from the split iterator
            split: Name used for logging messages (e.g., "val", "test")
            split_key: Key in self.splits to filter. Defaults to split if not provided.

        Returns:
            True if the split was modified and iterator needs to be recreated, False otherwise.
        """
        if split_key is None:
            split_key = split

        needs_recreation = False
        if split_context is not None:
            # Check if split has perturbations not in train
            if not self.context['use_perturbation_embedding'] and \
                    not set(self.context["perturbation_uniques"]) >= set(
                        split_context["perturbation_uniques"]
                    ):
                if self.filter_split_perturbations:
                    removed_perts = self._filter_split_perturbations(split_key)
                    log.warning(
                        f"Removed {len(removed_perts)} perturbation(s) from {split} split "
                        f"that were not in train: {removed_perts}"
                    )
                    self.removed_perturbations[split] = removed_perts
                    needs_recreation = True
                else:
                    extra_perts = set(split_context["perturbation_uniques"]) - set(self.context["perturbation_uniques"])
                    self.context["perturbation_uniques"] = (
                            set(self.context["perturbation_uniques"]) | extra_perts
                    )
                    self.context["num_perturbations"] = len(self.context["perturbation_uniques"])
                    log.info(
                        f"Perturbation filtering disabled: added {len(extra_perts)} "
                        f"perturbation(s) from {split} split to context: {extra_perts}"
                    )
                    needs_recreation = True

            if set(self.context["perturbation_uniques"]) != set(
                    split_context["perturbation_uniques"]
            ):
                log.warning(
                    f"{split} dataset is missing perturbations from train dataset."
                )

            # Filter out continuous covariates (None values) for validation
            # Only validate categorical covariates
            categorical_train = {k: v for k, v in self.context["covariate_uniques"].items() if v is not None}
            categorical_split = {k: v for k, v in split_context["covariate_uniques"].items() if v is not None}

            if any([set(categorical_train[k]) < set(categorical_split[k])
                    for k in categorical_train if k in categorical_split]):
                raise RuntimeError(
                    f"Train dataset must contain all covariates in {split} dataset."
                )

            if set(self.context["covariate_uniques"]) != set(
                    split_context["covariate_uniques"]
            ):
                log.warning(
                    f"{split} dataset is missing covariates from train dataset."
                )

        return needs_recreation

    def _get_output_path(self) -> str | None:
        """Get the output path from the splitter, if available.

        Returns:
            The output path string, or None if not available.
        """
        # Check if splitter has save attribute and it's False
        if hasattr(self.splitter, 'save') and not self.splitter.save:
            return None

        # Check if splitter has output_path
        if not hasattr(self.splitter, 'output_path'):
            return None

        output_path = self.splitter.output_path

        # Handle None, "None" string, or empty output_path
        if output_path is None or output_path == "None" or not output_path:
            return None

        return output_path

    def _save_removed_perturbations(self) -> None:
        """Save the removed perturbations to a file in the output directory.

        Creates a removed_perturbations.txt file with val and test sections.
        """
        if not self.removed_perturbations:
            return

        output_path = self._get_output_path()
        if output_path is None:
            log.debug("No output path available, skipping removed perturbations save.")
            return

        filepath = output_path + 'removed_perturbations.txt'
        with builtins.open(filepath, 'w') as f:
            f.write("# Perturbations removed from val/test splits (not present in train)\n")
            f.write("# These perturbations were filtered out to ensure all test perturbations are in training data\n\n")

            for split_name in ['val', 'test']:
                if split_name in self.removed_perturbations:
                    removed = self.removed_perturbations[split_name]
                    f.write(f"[{split_name}]\n")
                    for pert in sorted(removed):
                        f.write(f"{pert}\n")
                    f.write("\n")

        log.info(f"Saved removed perturbations to {filepath}")

    def _save_splits(self) -> None:
        """Save the current splits to the output directory.

        Saves three files:
        - train_test_split.csv: The split assignments for each cell
        - perturbation_covariate_split_map.csv: Deduplicated mapping of perturbation/covariate to split
        - removed_perturbations.txt: List of perturbations removed from val/test (if any)
        """
        output_path = self._get_output_path()
        if output_path is None:
            log.debug("No output path available, skipping split save.")
            return

        # Convert dict-based splits to pandas Series
        # Create a Series with index matching obs_df and values being split names
        split_series = pd.Series(index=self.obs_df.index, dtype=str)
        for split_name, indices in self.splits.items():
            for idx in indices:
                split_series.iloc[idx] = split_name

        # Save the split
        split_series.to_csv(output_path + 'train_test_split.csv', index=True, header=False)
        log.info(f"Saved splits to {output_path}train_test_split.csv")

        # Save the perturbation-covariate-split mapping (deduplicated)
        perturbation_key = self.context.get("perturbation_key")
        covariate_keys = self.context.get("covariate_keys", [])

        if perturbation_key and covariate_keys:
            unique_obs_df = self.obs_df[covariate_keys + [perturbation_key]].copy()
            unique_obs_df['split'] = split_series
            unique_obs_df = unique_obs_df.drop_duplicates().reset_index(drop=True)
            unique_obs_df.to_csv(
                output_path + 'perturbation_covariate_split_map.csv',
                index=False
            )
            log.info(f"Saved perturbation-covariate-split mapping to {output_path}perturbation_covariate_split_map.csv")

        # Also save removed perturbations if any
        self._save_removed_perturbations()

    def setup(self, stage: str | None = None) -> None:
        if self.data_handle is None:
            self.data_handle, obs_df = self.accessor(**self.data)
            self.obs_df = obs_df  # Store for filtering

        if self.splits is None:
            # self.splits = self.splitter.split(self.obs_df)
            import perturbench.data.datasplitter as datasplitter
            self.splits = datasplitter.PerturbationDataSplitter.split_dataset(
                splitter_config=self.splitter,
                obs_dataframe=self.obs_df,
                perturbation_key=inspect.signature(self.data_iter_factory).parameters['perturbation_key'].default ,
                perturbation_combination_delimiter=inspect.signature(self.data_iter_factory).parameters['perturbation_combination_delimiter'].default,
                perturbation_control_value=inspect.signature(self.data_iter_factory).parameters['perturbation_control_value'].default,
            )

        if stage == "fit" and self.train_iterator is None:
            self.train_iterator, train_context = self.data_iter_factory(
                self.data_handle,
                split=self.splits["train"],
            )
            self.context.update(train_context)
            self.context.update(
                {"num_perturbations": len(train_context["perturbation_uniques"])}
            )
            if isinstance(self.transform, DictConfig):
                self._transform_config = self.transform
                self.transform = instantiate_with_context(
                    self.transform,
                    self.context,
                )
            self.train_iterator.transform = self.transform  # triggers setter function

        if stage in {"validate", "fit"}:
            if self.val_iterator is None:
                self.val_iterator, val_context = self.data_iter_factory(
                    self.data_handle,
                    split=self.splits["val"],
                )
                needs_recreation = self._verify_split(val_context, split="val")

                if needs_recreation:
                    self.val_iterator, val_context = self.data_iter_factory(
                        self.data_handle,
                        split=self.splits["val"],
                    )
                    if not self.filter_split_perturbations and hasattr(self, '_transform_config'):
                        self.transform = instantiate_with_context(self._transform_config, self.context)
                        self.train_iterator.transform = self.transform

                self.val_iterator.transform = self.transform

                # Save splits after validation setup (includes any filtering done)
                self._save_splits()

        if stage == "test":  # this will be called automatically when you do trainer.test
            if self.test_iterator is None:
                if self.evaluation.split_value_to_evaluate == "test":
                    split_use = "test"
                elif self.evaluation.split_value_to_evaluate == "val":
                    split_use = "val"
                elif self.evaluation.split_value_to_evaluate == "train":
                    log.warning("'split_value_to_evaluate' is set to 'train'")
                    split_use = "train"
                else:
                    raise ValueError(
                        "split_value_to_evaluate must be either 'train', 'test' or 'val'."
                    )

                if split_use not in self.splits:
                    raise ValueError(f"'{split_use}' split is not present in the data.")

                data_iter_factory_params = inspect.signature(self.data_iter_factory).parameters
                if 'perturbation_embeddings_path' in data_iter_factory_params:
                    perturbation_embeddings_path = data_iter_factory_params['perturbation_embeddings_path'].default
                else:
                    perturbation_embeddings_path = None

                self.test_iterator, test_context = self.evaluation.test_data_iter_factory(
                    self.data_handle,
                    split=self.splits[split_use],
                    use_synthetic_controls=self.evaluation.use_synthetic_controls,
                    perturbation_embeddings_path=perturbation_embeddings_path,
                )
                needs_recreation = self._verify_split(test_context, split="test", split_key=split_use)

                if needs_recreation:
                    self.test_iterator, test_context = self.evaluation.test_data_iter_factory(
                        self.data_handle,
                        split=self.splits[split_use],
                        use_synthetic_controls=self.evaluation.use_synthetic_controls,
                        perturbation_embeddings_path=perturbation_embeddings_path,
                    )
                    if not self.filter_split_perturbations and hasattr(self, '_transform_config'):
                        self.transform = instantiate_with_context(self._transform_config, self.context)
                        if self.train_iterator is not None:
                            self.train_iterator.transform = self.transform

                self.test_iterator.transform = self.transform

                # Save splits after test setup (includes any filtering done)
                self._save_splits()

        if stage == "predict":
            if self.predict_iterator is None:
                prediction_dataframe = pd.read_csv(
                    self.prediction.prediction_dataframe_path
                )
                self.predict_iterator, _ = self.prediction.predict_data_iter_factory(
                    self.data_handle,
                    prediction_dataframe,
                )
                self.predict_iterator.transform = self.transform

    def teardown(self, stage: str) -> None:
        if stage == "fit":
            del self.train_iterator
            del self.val_iterator

    def train_dataloader(self) -> DataLoader:
        dataloader = self.loaders[self.mode](
            self.train_iterator,
            shuffle=True,
            collate_fn=self.collate if self.mode == self.Mode.EXAMPLE else noop_collate(),
            **self.loader,
        )
        # Cache the dataloader for state management
        self._cached_train_dataloader = dataloader

        # Restore state if resuming from checkpoint
        if hasattr(self, '_dataloader_state_to_restore') and self._dataloader_state_to_restore is not None:
            if hasattr(dataloader, 'load_state_dict'):
                dataloader.load_state_dict(self._dataloader_state_to_restore)
                log.info("Restored dataloader state from checkpoint")
            self._dataloader_state_to_restore = None

        return dataloader

    def val_dataloader(self) -> DataLoader | None:
        if self.val_iterator is None:
            return None
        else:
            return self.loaders[self.mode](
                self.val_iterator,
                shuffle=False,
                **self.loader,
            )

    def test_dataloader(self) -> DataLoader | None:
        if self.test_iterator is None:
            return None
        else:
            return BatchedDataLoader(
                self.test_iterator,
                shuffle=False,
                batch_size=self.evaluation.chunk_size,
                num_workers=self.loader.num_workers,
                ddp_mode=getattr(self.loader, 'ddp_mode', False),
            )

    def predict_dataloader(self) -> DataLoader | None:
        if self.predict_iterator is None:
            return None
        else:
            return BatchedDataLoader(
                self.predict_iterator,
                shuffle=False,
                batch_size=self.prediction.chunk_size,
                num_workers=self.loader.num_workers,
                ddp_mode=getattr(self.loader, 'ddp_mode', False),
            )

    def state_dict(self) -> dict:
        """Save dataloader state for checkpoint resumption.

        Lightning calls this when saving checkpoints. We save the state of
        the train dataloader so training can resume from the exact position.
        """
        state = {}
        if hasattr(self, '_cached_train_dataloader') and self._cached_train_dataloader is not None:
            if hasattr(self._cached_train_dataloader, 'state_dict'):
                state['train_dataloader'] = self._cached_train_dataloader.state_dict()
        return state

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore dataloader state from checkpoint.

        Lightning calls this when loading checkpoints. We store the state
        to be applied when the dataloader is next created.
        """
        self._dataloader_state_to_restore = state_dict.get('train_dataloader')


# Aliases for mode enumerations
EXAMPLE = DataLitModule.Mode.EXAMPLE
BATCH = DataLitModule.Mode.BATCH


class AnnDataLitModule(DataLitModule):

    @staticmethod
    def accessor(**kwargs):
        adata = sc.read_h5ad(
            **kwargs,
            backed='r',
        )
        adata.uns["filename"] = kwargs['filename']
        return (adata, adata.obs)

    loaders = {
        EXAMPLE: DataLoader,
        BATCH: BatchedDataLoader,
    }


class H5LitModule(DataLitModule):
    @staticmethod
    def accessor(filename: str | list[str], append_dataset_to_index: bool = False, **kwargs):
        """Load observation data from one or more h5ad files.

        Args:
            filename: path to h5ad file(s). Can be a single path or a list
              of paths for multi-file loading.
            append_dataset_to_index: if True, append the 'dataset' column value
              to each cell index (e.g., 'AAACCTG-1' -> 'AAACCTG-1-feng24').
              Requires a 'dataset' column in obs. Useful for matching split files
              that were created from merged datasets.

        Returns:
            Tuple of (filenames, combined_obs_df) where filenames is a list
            and combined_obs_df contains all observations with a '_file_idx'
            column indicating the source file.
        """
        # Normalize to list
        if isinstance(filename, (str, Path)):
            filenames = [str(filename)]
        else:
            filenames = [str(f) for f in filename]

        if len(filenames) == 1:
            # Single file - original behavior (return single path, not list)
            obs_df = load_dataframe_from_h5(filenames[0], 'obs')
            return (filenames[0], obs_df)

        # Multiple files - concatenate obs dataframes
        obs_dfs = []
        cumulative_count = 0
        for i, f in enumerate(filenames):
            df = load_dataframe_from_h5(f, 'obs')
            df = df.copy()

            # Append dataset name and file index to index if requested
            # File index is needed to handle cases where multiple files share the same
            # dataset name (e.g., nadig24_hepg2 and nadig24_jurkat both have dataset='nadig24')
            if append_dataset_to_index:
                if 'dataset' not in df.columns:
                    raise ValueError(
                        f"append_dataset_to_index=True requires a 'dataset' column in obs, "
                        f"but file '{f}' does not have one. "
                        f"Available columns: {list(df.columns)}"
                    )
                dataset_name = df['dataset'].iloc[0]  # All rows have same dataset
                df.index = df.index.astype(str) + f'-f{i}-' + dataset_name

            df['_file_idx'] = i
            df['_global_idx'] = range(cumulative_count, cumulative_count + len(df))
            cumulative_count += len(df)
            obs_dfs.append(df)

        combined_obs = pd.concat(obs_dfs, ignore_index=False)
        return (filenames, combined_obs)

    loaders = {
        BATCH: BatchedDataLoader,
    }
