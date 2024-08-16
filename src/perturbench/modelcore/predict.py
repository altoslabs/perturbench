import pandas as pd
from omegaconf import DictConfig
import lightning as L
import logging
import hydra
import os

from perturbench.data.datasets import Counterfactual
from perturbench.data.utils import batch_dataloader
from perturbench.data.collate import noop_collate
from .models.base import PerturbationModel

log = logging.getLogger(__name__)


def predict(
    cfg: DictConfig,
):
    """Predict counterfactual perturbation effects"""
    # Set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info("Instantiating datamodule <%s>", cfg.data._target_)
    datamodule: L.LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info("Instantiating model <%s>", cfg.model._target_)
    model_class: PerturbationModel = hydra.utils.get_class(cfg.model._target_)

    # Load trained model
    if not os.path.exists(cfg.ckpt_path):
        raise ValueError(f"Checkpoint path {cfg.ckpt_path} does not exist")
    if not os.path.exists(cfg.ckpt_path):
        raise ValueError(f"Checkpoint path {cfg.ckpt_path} does not exist")

    trained_model: PerturbationModel = model_class.load_from_checkpoint(
        cfg.ckpt_path,
        datamodule=datamodule,
    )

    # Load prediction dataframe
    if not os.path.exists(cfg.prediction_dataframe_path):
        raise ValueError(
            f"Prediction dataframe path {cfg.prediction_dataframe_path} does not exist"
        )
    pred_df = pd.read_csv(cfg.prediction_dataframe_path)

    if cfg.data.perturbation_key not in pred_df.columns:
        raise ValueError(
            f"Prediction dataframe must contain column {cfg.data.perturbation_key}"
        )
    for covariate_key in cfg.data.covariate_keys:
        if covariate_key not in pred_df.columns:
            raise ValueError(
                f"Prediction dataframe must contain column {covariate_key}"
            )

    # Create inference dataloader
    test_adata = datamodule.test_dataset.reference_adata
    control_adata = test_adata[
        test_adata.obs[cfg.data.perturbation_key] == cfg.data.perturbation_control_value
    ]
    del test_adata

    inference_dataset, _ = Counterfactual.from_anndata(
        control_adata,
        pred_df,
        cfg.data.perturbation_key,
        perturbation_combination_delimiter=cfg.data.perturbation_combination_delimiter,
        covariate_keys=cfg.data.covariate_keys,
        perturbation_control_value=cfg.data.perturbation_control_value,
        seed=cfg.seed,
        max_control_cells_per_covariate=cfg.data.evaluation.max_control_cells_per_covariate,
    )
    inference_dataset.transform = trained_model.training_record["transform"]
    inference_dataloader = batch_dataloader(
        inference_dataset,
        batch_size=cfg.chunk_size,
        num_workers=cfg.data.num_workers,
        shuffle=False,
        collate_fn=noop_collate(),
    )

    log.info("Instantiating trainer <%s>", cfg.trainer._target_)
    trainer: L.Trainer = hydra.utils.instantiate(cfg.trainer)

    log.info("Generating predictions")
    if not os.path.exists(cfg.output_path):
        os.makedirs(cfg.output_path)
    trained_model.prediction_output_path = cfg.output_path
    trainer.predict(model=trained_model, dataloaders=inference_dataloader)


@hydra.main(version_base="1.3", config_path="../configs", config_name="predict.yaml")
def main(cfg: DictConfig):
    predict(cfg)


if __name__ == "__main__":
    main()
