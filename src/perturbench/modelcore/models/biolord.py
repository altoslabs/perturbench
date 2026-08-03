import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from ..nn.mlp import MLP
from .base import PerturbationModel
from perturbench.data.types import Batch
from perturbench.data.transforms.base import Dispatch


class BiolordStar(PerturbationModel):
    """
    A version of Biolord
    """
    COMPATIBLE_DATASETS = (
        'SingleCellPerturbation',
    )

    def __init__(
            self,
            n_genes: int,
            n_perts: int,
            transform: Dispatch,
            context: dict,
            evaluation: DictConfig,
            embedding_width: int | None = None,
            n_layers: int = 2,
            encoder_width: int = 128,
            latent_dim: int = 32,
            penalty_weight: float = 10000.0,
            noise: float = 0.1,
            lr: float | None = None,
            wd: float | None = None,
            lr_scheduler: DictConfig | None = None,
            dropout: float | None = None,
            softplus_output: bool = True,
            **kwargs,
    ):
        """
        The constructor for the BiolordStar class.

        Args:
            n_genes: Number of genes to use for prediction
            n_perts: Number of perturbations in the dataset
                (not including controls)
            n_layers: Number of layers in the encoder/decoder
            encoder_width: Width of the hidden layers in the encoder/decoder
            latent_dim: Dimension of the latent space
            lr: Learning rate
            wd: Weight decay
            lr_scheduler_freq: How often the learning rate scheduler checks
                val_loss
            lr_scheduler_interval: Whether the learning rate scheduler checks
                every epoch or step
            lr_scheduler_patience: Learning rate scheduler patience
            lr_scheduler_factor: Factor by which to reduce learning rate when
                learning rate scheduler triggers
            dropout: Dropout rate or None for no dropout.
            softplus_output: Whether to apply a softplus activation to the
                output of the decoder to enforce non-negativity
            datamodule: The datamodule used to train the model
        """
        super(BiolordStar, self).__init__(
            n_genes=n_genes,
            n_perts=n_perts,
            transform=transform,
            context=context,
            evaluation=evaluation,
            lr=lr,
            wd=wd,
            lr_scheduler=lr_scheduler,
        )
        self.save_hyperparameters()

        if n_genes is not None:
            self.n_genes = n_genes
        if n_perts is not None:
            self.n_perts = n_perts

        # Check for continuous covariates (None uniques) which are not supported
        continuous_covs = [k for k, v in context["covariate_uniques"].items() if v is None]
        if continuous_covs:
            raise ValueError(
                f"BioLord does not support continuous covariates. "
                f"Found continuous covariates: {continuous_covs}"
            )

        self.lord_embedding = torch.nn.ParameterDict()
        for cov, unique_covs in context["covariate_uniques"].items():
            self.lord_embedding[cov] = torch.nn.Parameter(torch.randn(latent_dim, len(unique_covs)))

        decoder_input_dim = 3 * latent_dim
        self.gene_encoder = MLP(self.n_genes, encoder_width, latent_dim, n_layers, dropout)
        self.decoder = MLP(decoder_input_dim, encoder_width, self.n_genes, n_layers, dropout)
        self.pert_encoder = MLP(self.n_perts, encoder_width, latent_dim, n_layers, dropout)

        self.penalty_weight = penalty_weight
        self.noise = noise
        self.dropout = dropout
        self.softplus_output = softplus_output

    def forward(
            self,
            observed_perturbed_expression: torch.Tensor,
            perturbation: torch.Tensor,
            covariates: dict[str, torch.Tensor],
    ):
        latent_observed_perturbed_expression = self.gene_encoder(observed_perturbed_expression)
        latent_observed_perturbed_expression += self.noise * torch.randn_like(latent_observed_perturbed_expression)
        latent_perturbation = self.pert_encoder(perturbation)

        latent_covariates_list = []
        for cov, cov_values in covariates.items():
            latent_covariates_list.append(torch.vstack(
                [self.lord_embedding[cov][:, val.bool()].T for val in cov_values]
            ))
        latent_covariates = torch.hstack(latent_covariates_list)

        latent_perturbed_expression = torch.cat([
            latent_observed_perturbed_expression, latent_perturbation, latent_covariates
        ], dim=-1)

        predicted_perturbed_expression = self.decoder(latent_perturbed_expression)

        if self.softplus_output:
            predicted_perturbed_expression = F.softplus(predicted_perturbed_expression)
        return predicted_perturbed_expression, (latent_covariates ** 2).sum()

    def training_step(self, batch: Batch, batch_idx: int):
        observed_perturbed_expression = batch.gene_expression.squeeze()
        perturbation = batch.perturbations.squeeze()
        covariates = batch.covariates

        predicted_perturbed_expression, penalty = self.forward(
            observed_perturbed_expression, perturbation, covariates
        )
        loss = F.mse_loss(predicted_perturbed_expression, observed_perturbed_expression, reduction='none').sum(
            axis=1).mean()
        self.log("train_loss", loss, prog_bar=True, logger=True, batch_size=len(batch))
        return loss + self.penalty_weight * penalty

    def validation_step(self, batch: Batch, batch_idx: int):
        observed_perturbed_expression = batch.gene_expression.squeeze()
        perturbation = batch.perturbations.squeeze()
        covariates = batch.covariates

        predicted_perturbed_expression, penalty = self.forward(
            observed_perturbed_expression, perturbation, covariates
        )
        val_loss = F.mse_loss(predicted_perturbed_expression, observed_perturbed_expression, reduction='none').sum(
            axis=1).mean()
        self.log(
            "val_loss",
            val_loss,
            on_step=True,
            prog_bar=True,
            logger=True,
            batch_size=len(batch),
        )
        return val_loss + self.penalty_weight * penalty

    def predict(self, batch):
        control_expression = batch.gene_expression.squeeze().to(self.device)
        perturbation = batch.perturbations.squeeze().to(self.device)
        covariates = {k: v.to(self.device) for k, v in batch.covariates.items()}

        predicted_perturbed_expression, _ = self.forward(
            control_expression,
            perturbation,
            covariates,
        )
        return predicted_perturbed_expression
