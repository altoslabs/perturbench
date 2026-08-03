import os
import warnings
from sklearn.exceptions import DataConversionWarning
import logging
from typing import List
import hydra
import lightning as L
import mlflow
from omegaconf import DictConfig
from lightning.pytorch.loggers import Logger
from perturbench.modelcore.utils import multi_instantiate, instantiate_with_context
from perturbench.modelcore.models import PerturbationModel
from perturbench.data.modules import DataLitModule
from hydra.core.hydra_config import HydraConfig

import torch

# Set HDF5 file locking to False to avoid locking issues
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
torch.set_num_threads(1)

warnings.simplefilter("ignore", DataConversionWarning)

if workspace := os.environ.get("MLFLOW_WORKSPACE"):
    mlflow.set_workspace(workspace)

log = logging.getLogger(__name__)


def train(cfg: DictConfig):
    runtime_context = {
        "cfg": cfg,
        "trial_number": HydraConfig.get().job.get("num"),
        "trial": cfg.get("trial")
    }

    # Set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info("Instantiating datamodule <%s>", cfg.data._target_)
    datamodule: DataLitModule = hydra.utils.instantiate(
        cfg.data,
    )
    datamodule.setup(stage='fit')

    log.info("Instantiating model <%s>", cfg.model._target_)
    dataset_type = cfg.data.data_iter_factory._target_.split('.')[-2]
    if dataset_type not in PerturbationModel.COMPATIBLE_DATASETS:
        raise ValueError(
            f"Model {cfg.model._target_} is not compatible with dataset {cfg.data.data_iter_factory._target_}")

    # Ensure datamodule uses the same transform as the model
    if cfg.get("ckpt_path"):
        model_class = hydra.utils.get_class(cfg.model._target_)
        update_kwargs = {
            'evaluation': datamodule.evaluation,
            'map_location': 'cpu',
            'weights_only': False,
        }
        if hasattr(cfg.model, 'lr'):
            update_kwargs['lr'] = cfg.model['lr']
        if hasattr(cfg.model, 'lr_scheduler'):
            update_kwargs['lr_scheduler'] = hydra.utils.instantiate(cfg.model.lr_scheduler)
        model: PerturbationModel = model_class.load_from_checkpoint(cfg.ckpt_path, **update_kwargs)

        transform = model.training_record['transform']
        if isinstance(transform, DictConfig):
            transform = instantiate_with_context(hydra.utils.instantiate(cfg.data.transform),
                                                 model.training_record['train_context'])
        datamodule.transform = transform
        datamodule.train_iterator.transform = transform
        datamodule.val_iterator.transform = transform
        datamodule.context = model.training_record['train_context']

        checkpoint_perts = set(model.training_record['train_context']['perturbation_uniques'])
        for split_name in ['val', 'test']:
            if split_name in datamodule.splits:
                split_obs = datamodule.obs_df.iloc[datamodule.splits[split_name]]
                split_perts = set(split_obs[datamodule.context['perturbation_key']].unique())
                split_perts.discard(datamodule.context.get('perturbation_control_value', 'control'))
                unknown = split_perts - checkpoint_perts
                if unknown:
                    log.warning(
                        f"{len(unknown)} perturbation(s) in {split_name} split are not in the "
                        f"pretrained encoder and will fail during evaluation: {unknown}"
                    )

    else:
        model: PerturbationModel = hydra.utils.instantiate(
            cfg.model,
            n_genes=datamodule.num_genes,
            n_perts=datamodule.num_perturbations,
            transform=datamodule.transform,
            context=datamodule.context,
            evaluation=datamodule.evaluation,
            embedding_width=datamodule.embedding_width,
            perturbation_embedding_width=datamodule.perturbation_embedding_width,
            gene_names=datamodule.train_iterator.gene_names,
        )

    log.info("Instantiating callbacks...")
    callbacks: List[L.Callback] = multi_instantiate(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    loggers: List[Logger] = multi_instantiate(
        cfg.get("logger"), context=runtime_context
    )

    log.info("Instantiating trainer <%s>", cfg.trainer._target_)
    trainer: L.Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=loggers,
    )

    if cfg.get("train"):
        log.info("Starting training!")
        # in finetuning mode, avoid loading full model state (e.g., optimizers),
        # so do not pass `ckpt_path`; to restore full model training, pass `ckpt_path`.
        trainer.fit(
            model=model,
            datamodule=datamodule,
            ckpt_path=None if getattr(cfg, 'finetune', False) else getattr(cfg, 'ckpt_path', None),
            weights_only=False,
        )

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

        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path, weights_only=False)
        if model.summary_metrics is not None:  # Only rank 0 has summary_metrics
            summary_metrics_dict = model.summary_metrics.to_dict()[
                model.summary_metrics.columns[0]
            ]

    test_metrics = trainer.callback_metrics
    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics, **summary_metrics_dict}

    return metric_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> float | None:
    ## Train the model
    global metric_dict
    metric_dict = train(cfg)

    ## Combined metric
    metrics_use = cfg.get("metrics_to_optimize")
    if metrics_use:
        combined_metric = sum(
            [metric_dict.get(metric) * weight for metric, weight in metrics_use.items()]
        )
        return combined_metric


if __name__ == "__main__":
    main()
