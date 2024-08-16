import logging
from typing import List
import hydra
import lightning as L
from omegaconf import DictConfig
from lightning.pytorch.loggers import Logger
from perturbench.modelcore.utils import multi_instantiate
from perturbench.modelcore.models import PerturbationModel
from hydra.core.hydra_config import HydraConfig


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
