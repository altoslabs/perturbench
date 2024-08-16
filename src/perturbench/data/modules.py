import warnings
from pathlib import Path
from collections.abc import Callable, Mapping
import gc

from omegaconf import DictConfig
import lightning as L
import scanpy as sc
from torch.utils.data import DataLoader
from perturbench.modelcore.utils import (
    instantiate_with_context,
)

from .datasets import (
    CounterfactualWithReference,
    SingleCellPerturbation,
    SingleCellPerturbationWithControls,
)
from .utils import batch_dataloader
from .collate import noop_collate
import perturbench.data.datasplitter as datasplitter


class AnnDataLitModule(L.LightningDataModule):
    """AnnData Data Module for Perturbation Prediction Models."""

    def __init__(
        self,
        datapath: Path,
        add_controls: str,
        perturbation_key: str,
        perturbation_control_value: str,
        batch_size: int,
        perturbation_combination_delimiter: str | None,
        covariate_keys: list[str] | None = None,
        splitter: DictConfig | None = None,
        num_workers: int = 0,
        num_val_workers: int | None = None,
        num_test_workers: int | None = None,
        transform: DictConfig | None = None,
        collate: Mapping[str, Callable] | None = None,
        batch_sample: bool = True,
        evaluation: DictConfig | None = None,
        use_counts: bool = False,
        embedding_key: str | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_val_workers = (
            num_val_workers if num_val_workers is not None else num_workers
        )
        self.num_test_workers = (
            num_test_workers if num_test_workers is not None else num_workers
        )
        self.batch_sample = batch_sample
        self.split = None  ## Data split as a pandas series

        # Select appropriate dataset class and common arguments
        if add_controls:
            dataset_class = SingleCellPerturbationWithControls
        else:
            dataset_class = SingleCellPerturbation

        data_kwargs = {
            "perturbation_key": perturbation_key,
            "perturbation_combination_delimiter": perturbation_combination_delimiter,
            "covariate_keys": covariate_keys,
            "perturbation_control_value": perturbation_control_value,
            "embedding_key": embedding_key,
        }

        # Load AnnData object
        adata = sc.read_h5ad(datapath)
        if use_counts:
            adata.X = adata.layers["counts"]
        adata.raw = None
        if "counts" in adata.layers:
            del adata.layers["counts"]

        # Split intro train, val, test datasets
        if splitter is not None:
            # AnnData Split
            split_dict = datasplitter.PerturbationDataSplitter.split_dataset(
                splitter_config=splitter,
                obs_dataframe=adata.obs,
                perturbation_key=perturbation_key,
                perturbation_combination_delimiter=perturbation_combination_delimiter,
                perturbation_control_value=perturbation_control_value,
            )

            train_adata = adata[split_dict["train"]]
            val_adata = adata[split_dict["val"]]
            test_adata = adata[split_dict["test"]]

        else:
            train_adata = adata
            val_adata = None
            test_adata = None

        # Create datasets
        self.train_dataset, train_context = dataset_class.from_anndata(
            train_adata, **data_kwargs
        )

        self.val_dataset, val_context = (
            dataset_class.from_anndata(val_adata, **data_kwargs)
            if val_adata is not None
            else (None, None)
        )

        if evaluation.split_value_to_evaluate == "train":
            test_adata = train_adata
        elif evaluation.split_value_to_evaluate == "val":
            test_adata = val_adata
        self.test_dataset, test_context = (
            CounterfactualWithReference.from_anndata(
                test_adata,
                seed=seed,
                max_control_cells_per_covariate=evaluation.max_control_cells_per_covariate,
                **data_kwargs,
            )
            if test_adata is not None
            else (None, None)
        )
        self.train_context = train_context
        self.evaluation = evaluation

        # Verify that train, val, test datasets have the same perturbations and covariates
        self._verify_splits(train_context, val_context, test_context)

        # Build an example/batch transform pipeline from train dataset context
        if transform is not None:
            transform_pipeline = instantiate_with_context(transform, train_context)

            # Set transform pipeline for each dataset
            self.train_dataset.transform = transform_pipeline
            self.val_dataset.transform = transform_pipeline
            self.test_dataset.transform = transform_pipeline

        # Build example collation function
        if self.batch_sample is True:
            self.example_collate_fn = noop_collate()
        else:
            self.example_collate_fn = collate

        self.num_perturbations = len(train_context["perturbation_uniques"])

        # Cleanup
        del adata, train_adata
        if val_adata is not None:
            del val_adata
        if test_adata is not None:
            del test_adata
        gc.collect()

    @property
    def num_genes(self) -> int:
        """Number of genes in the dataset."""
        return len(self.train_dataset.gene_names)

    @property
    def embedding_width(self) -> int | None:
        """Width of the embeddings."""
        if self.train_dataset.embeddings is None:
            return None
        else:
            return self.train_dataset.embeddings.shape[1]

    @staticmethod
    def _verify_splits(train_info: dict, val_info: dict | None, test_info: dict | None):
        for split, info in [("val", val_info), ("test", test_info)]:
            if info is not None:
                if not set(train_info["perturbation_uniques"]) >= set(
                    info["perturbation_uniques"]
                ):
                    raise RuntimeError(
                        f"Train dataset must contain all perturbations in {split} dataset."
                    )
                if set(train_info["perturbation_uniques"]) != set(
                    info["perturbation_uniques"]
                ):
                    warnings.warn(
                        f"{split} dataset is missing perturbations from train dataset."
                    )

                if not set(train_info["covariate_uniques"]) >= set(
                    info["covariate_uniques"]
                ):
                    raise RuntimeError(
                        f"Train dataset must contain all covariates in {split} dataset."
                    )
                if set(train_info["covariate_uniques"]) != set(
                    info["covariate_uniques"]
                ):
                    warnings.warn(
                        f"{split} dataset is missing covariates from train dataset."
                    )

    def train_dataloader(self) -> DataLoader:
        if self.batch_sample:
            return batch_dataloader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                collate_fn=self.example_collate_fn,
            )
        else:
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                collate_fn=self.example_collate_fn,
                shuffle=True,
            )

    def val_dataloader(self) -> DataLoader | None:
        if self.val_dataset is None:
            return None
        else:
            if self.batch_sample:
                return batch_dataloader(
                    self.val_dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.num_val_workers,
                    collate_fn=self.example_collate_fn,
                )
            else:
                return DataLoader(
                    self.val_dataset,
                    batch_size=self.batch_size,
                    num_workers=self.num_val_workers,
                    collate_fn=self.example_collate_fn,
                    shuffle=False,
                )

    def test_dataloader(self) -> DataLoader | None:
        if self.test_dataset is None:
            return None
        else:
            return batch_dataloader(
                self.test_dataset,
                batch_size=self.evaluation.chunk_size,
                num_workers=self.num_test_workers,
                shuffle=False,
                collate_fn=self.example_collate_fn,
            )
