import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from ..nn.mlp import MLP
from .base import PerturbationModel
from perturbench.data.types import Batch
from perturbench.data.transforms.base import Dispatch


class DecoderOnly(PerturbationModel):
    """
    A latent additive model for predicting perturbation effects
    """

    COMPATIBLE_DATASETS = (
        'SingleCellPerturbationWithControls',
    )

    def __init__(
            self,
            n_genes: int,
            n_perts: int,
            transform: Dispatch,
            context: dict,
            evaluation: DictConfig,
            perturbation_embedding_width: int | None = None,
            n_layers=2,
            encoder_width=128,
            softplus_output=True,
            use_covariates=True,
            use_perturbations=True,
            lr: float | None = None,
            wd: float | None = None,
            lr_scheduler: DictConfig | None = None,
            **kwargs,
    ) -> None:

        """
        The constructor for the DecoderOnly class.

        Args:
            n_genes (int): Number of genes to use for prediction
            n_perts (int): Number of perturbations in the dataset (not including controls)
            n_layers (int): Number of layers in the encoder/decoder
            lr (float): Learning rate
            wd (float): Weight decay
            softplus_output (bool): Whether to apply a softplus activation to the output of the decoder to enforce non-negativity
        """

        super(DecoderOnly, self).__init__(
            n_genes=n_genes,
            n_perts=n_perts,
            transform=transform,
            context=context,
            evaluation=evaluation,
            perturbation_embedding_width=perturbation_embedding_width,
            lr=lr,
            wd=wd,
            lr_scheduler=lr_scheduler,
        )
        self.save_hyperparameters()

        if not (use_covariates or use_perturbations):
            raise ValueError(
                "'use_covariates' and 'use_perturbations' can not both be false. \
                    Either covariates or perturbations have to be used."
            )

        n_perts = self.n_input_perturbation_features if use_perturbations else 0

        decoder_input_dim = self.n_total_covariates + n_perts

        self.decoder = MLP(decoder_input_dim, encoder_width, self.n_genes, n_layers)
        self.softplus_output = softplus_output
        self.use_covariates = use_covariates
        self.use_perturbations = use_perturbations

    def forward(
            self,
            perturbation: torch.Tensor,
            covariates: dict[str, torch.Tensor],
    ):
        if self.use_covariates and self.use_perturbations:
            embedding = torch.cat([cov if cov.ndim == 2 else cov.squeeze() for cov in covariates.values()], dim=1)
            embedding = torch.cat([perturbation, embedding], dim=1)
        elif self.use_covariates:
            embedding = torch.cat([cov if cov.ndim == 2 else cov.squeeze() for cov in covariates.values()], dim=1)
        elif self.use_perturbations:
            embedding = perturbation

        predicted_perturbed_expression = self.decoder(embedding)

        if self.softplus_output:
            predicted_perturbed_expression = F.softplus(predicted_perturbed_expression)
        return predicted_perturbed_expression

    def training_step(self, batch: Batch, batch_idx: int):
        predicted_perturbed_expression = self.forward(
            batch.perturbations.squeeze(), batch.covariates
        )
        loss = F.mse_loss(predicted_perturbed_expression, batch.gene_expression.squeeze())
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            logger=True,
            batch_size=len(batch)
        )
        return loss

    def validation_step(self, batch: Batch, batch_idx: int):
        predicted_perturbed_expression = self.forward(
            batch.perturbations.squeeze(), batch.covariates
        )
        val_loss = F.mse_loss(predicted_perturbed_expression, batch.gene_expression.squeeze())
        self.log(
            "val_loss",
            val_loss,
            on_step=True,
            prog_bar=True,
            logger=True,
            batch_size=len(batch),
        )
        return val_loss

    def predict(self, batch):
        perturbation = batch.perturbations.squeeze().to(self.device)
        covariates = {
            k: v.to(self.device) for k, v in batch.covariates.items()
        }

        predicted_perturbed_expression = self.forward(
            perturbation,
            covariates,
        )
        return predicted_perturbed_expression
