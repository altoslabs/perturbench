from omegaconf import DictConfig, open_dict
import lightning as L
import logging
import hydra
import os
import pandas as pd
import warnings

from perturbench.data.modules import DataLitModule
from perturbench.modelcore.models.base import PerturbationModel
from perturbench.modelcore.utils import instantiate_with_context

# Set HDF5 file locking to False to avoid locking issues
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

log = logging.getLogger(__name__)
warnings.simplefilter(action='ignore', category=UserWarning)


def predict_chunk(
    cfg: DictConfig,
    chunk_idx: int,
    total_chunks: int,
):
    """Predict counterfactual perturbation effects for a single chunk."""

    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info("Processing chunk %d of %d", chunk_idx, total_chunks)

    log.info("Instantiating datamodule <%s>", cfg.data._target_)
    datamodule: DataLitModule = hydra.utils.instantiate(cfg.data)
    datamodule.setup("fit")

    log.info("Instantiating model <%s>", cfg.model._target_)
    model_class: PerturbationModel = hydra.utils.get_class(cfg.model._target_)

    # Load trained model
    if ckpt_path := cfg.get("ckpt_path"):
        log.info("Using checkpoint path from config: %s", ckpt_path)

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

    # chunk the prediction dataframe
    prediction_dataframe = pd.read_csv(cfg.data.prediction.prediction_dataframe_path)

    chunk_size = len(prediction_dataframe) // total_chunks
    remainder = len(prediction_dataframe) % total_chunks

    if chunk_idx < remainder:
        start_idx = chunk_idx * (chunk_size + 1)
        end_idx = start_idx + chunk_size + 1
    else:
        start_idx = remainder * (chunk_size + 1) + (chunk_idx - remainder) * chunk_size
        end_idx = start_idx + chunk_size

    chunk_dataframe = prediction_dataframe.iloc[start_idx:end_idx].copy()
    log.info(
        "Chunk %d: rows %d to %d (%d rows)",
        chunk_idx, start_idx, end_idx - 1, len(chunk_dataframe),
    )

    chunk_prediction_path = os.path.join(cfg.output_path, f"temp_prediction_chunk_{chunk_idx}.csv")
    chunk_dataframe.to_csv(chunk_prediction_path, index=False)

    with open_dict(cfg.data.prediction):
        cfg.data.prediction.prediction_dataframe_path = chunk_prediction_path
    with open_dict(datamodule.prediction):
        datamodule.prediction.prediction_dataframe_path = chunk_prediction_path

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

    log.info("Generating predictions for chunk %d", chunk_idx)
    os.makedirs(cfg.output_path, exist_ok=True)
    trainer.predict(model=trained_model, dataloaders=inference_dataloader)

    if trainer.world_size > 1:
        trainer.strategy.barrier()

    log.info("Completed chunk %d", chunk_idx)


@hydra.main(version_base="1.3", config_path="../configs", config_name="predict.yaml")
def main(cfg: DictConfig):
    with open_dict(cfg):
        cfg.output_path = f"{cfg.output_path}/chunk_{cfg.chunk_idx}"
        os.makedirs(cfg.output_path, exist_ok=True)
    predict_chunk(cfg, cfg.chunk_idx, cfg.total_chunks)


if __name__ == "__main__":
    main()
