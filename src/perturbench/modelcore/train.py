import logging
from typing import List
import hydra
import lightning as L
from omegaconf import DictConfig
from lightning.pytorch.loggers import Logger, WandbLogger
from perturbench.modelcore.utils import multi_instantiate
from perturbench.modelcore.models import PerturbationModel
from hydra.core.hydra_config import HydraConfig
import torch
from omegaconf import OmegaConf

torch.set_float32_matmul_precision("medium")

log = logging.getLogger(__name__)


def train(runtime_context: dict):

    cfg = runtime_context["cfg"]

    # Set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info("Instantiating datamodule <%s>", cfg.data._target_)
    datamodule: L.LightningDataModule = hydra.utils.instantiate(
        cfg.data,
        seed=cfg.seed,
    )

    log.info("Instantiating model <%s>", cfg.model._target_)
    model: PerturbationModel = hydra.utils.instantiate(cfg.model, datamodule=datamodule)

    log.info("Instantiating callbacks...")
    callbacks: List[L.Callback] = multi_instantiate(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    loggers: List[Logger] = multi_instantiate(cfg.get("logger"))

    for logger in loggers:

        if isinstance(logger, WandbLogger):
            flat_cfg = OmegaConf.to_container(cfg, resolve=True)
            if isinstance(flat_cfg, dict):
                model_cfg = flat_cfg.get("model")
                if isinstance(model_cfg, dict) and "_target_" in model_cfg:
                    model_cfg["model_name"] = model_cfg["_target_"].split(".")[-1]
                data_cfg = flat_cfg.get("data")
                if isinstance(data_cfg, dict) and "datapath" in data_cfg:
                    data_cfg["dataset_name"] = data_cfg["datapath"].split("/")[-1]
                
                if hasattr(logger, "experiment") and hasattr(logger.experiment, "config"):
                    logger.experiment.config.update(flat_cfg)

                # Add Optuna study name to wandb config if running HPO
                try:
                    hydra_cfg = HydraConfig.get()
                    print("Hydra config: ", hydra_cfg)
                    if hasattr(hydra_cfg, "sweeper") and hasattr(hydra_cfg.sweeper, "study_name"):
                        study_name = hydra_cfg.sweeper.study_name
                        logger.experiment.config["optuna_study_name"] = study_name
                        print("Found study name: ", study_name)
                except Exception as e:
                    log.debug(f"Could not get study name from config: {e}")

    log.info("Instantiating trainer <%s>", cfg.trainer._target_)
    trainer: L.Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=loggers
    )

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    summary_metrics_dict = {}
    if cfg.get("test"):
        log.info("Starting testing!")
        if cfg.get("train"):
            if (
                trainer.checkpoint_callback is None
                or trainer.checkpoint_callback.best_model_path == ""
            ):
                ckpt_path = None
            else:
                ckpt_path = "best"
        else:
            ckpt_path = cfg.get("ckpt_path")
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        summary_metrics_dict = model.summary_metrics.to_dict()[
            model.summary_metrics.columns[0]
        ]

    test_metrics = trainer.callback_metrics
    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics, **summary_metrics_dict}

    return metric_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> float | None:

    runtime_context = {"cfg": cfg, "trial_number": HydraConfig.get().job.get("num")}

    ## Train the model
    global metric_dict
    metric_dict = train(runtime_context)

    ## Combined metric
    metrics_use = cfg.get("metrics_to_optimize")
    if metrics_use:
        combined_metric = sum(
            [metric_dict.get(metric) * weight for metric, weight in metrics_use.items()]
        )
        return combined_metric


if __name__ == "__main__":
    main()
