from omegaconf import DictConfig, open_dict
import lightning as L
import logging
import hydra
import os
import warnings

from perturbench.data.modules import DataLitModule
from perturbench.modelcore.models.base import PerturbationModel
from perturbench.modelcore.utils import instantiate_with_context

# Set HDF5 file locking to False to avoid locking issues
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

log = logging.getLogger(__name__)
warnings.simplefilter(action='ignore', category=UserWarning)


def predict(
        cfg: DictConfig,
):
    """Predict counterfactual perturbation effects"""
    # Set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info("Instantiating datamodule <%s>", cfg.data._target_)
    datamodule: DataLitModule = hydra.utils.instantiate(cfg.data)
    datamodule.setup("fit")

    log.info("Instantiating model <%s>", cfg.model._target_)
    model_class: PerturbationModel = hydra.utils.get_class(cfg.model._target_)

    # Load trained model
    if ckpt_path := cfg.get("ckpt_path"):
        log.info(f"Using checkpoint path from config: {ckpt_path}")

    update_kwargs = {
        'map_location': 'cpu',
        'weights_only': False,
    }

    trained_model: PerturbationModel = model_class.load_from_checkpoint(
        ckpt_path, **update_kwargs,
    )
    trained_model.prediction_config = cfg

    transform = trained_model.training_record['transform']
    if isinstance(transform, DictConfig):
        transform = instantiate_with_context(
            hydra.utils.instantiate(cfg.data.transform),
            trained_model.training_record['train_context']
        )

    # Auto-detect distributed environment and enable ddp_mode so the
    # predict_dataloader uses DistributedSampler to split work across ranks.
    if int(os.environ.get('WORLD_SIZE', 1)) > 1:
        log.info(
            "Detected distributed environment (WORLD_SIZE=%s), "
            "enabling ddp_mode for prediction dataloader",
            os.environ['WORLD_SIZE'],
        )
        with open_dict(cfg.data.loader):
            cfg.data.loader.ddp_mode = True

    # Set up datamodule for prediction
    datamodule.setup(stage='predict')
    datamodule.transform = transform
    datamodule.predict_iterator.transform = transform
    datamodule.context = trained_model.training_record['train_context']
    inference_dataloader = datamodule.predict_dataloader()

    log.info("Instantiating trainer <%s>", cfg.trainer._target_)
    trainer: L.Trainer = hydra.utils.instantiate(
        cfg.trainer, logger=False
    )

    log.info("Generating predictions")
    os.makedirs(cfg.output_path, exist_ok=True)
    trainer.predict(model=trained_model, dataloaders=inference_dataloader)

    if trainer.world_size > 1:
        trainer.strategy.barrier()


@hydra.main(version_base="1.3", config_path="../configs", config_name="predict.yaml")
def main(cfg: DictConfig):
    predict(cfg)


if __name__ == "__main__":
    main()
