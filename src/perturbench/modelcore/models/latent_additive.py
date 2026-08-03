import torch
from typing import Literal
from omegaconf import DictConfig
import logging

from perturbench.data.types import Batch
from perturbench.data.transforms.base import Dispatch
from .base import PerturbationModel
from ..nn.mlp import MLP
from ..nn.decoders import (
    DeepGaussian,
    DeepIsotropicGaussian,
    DeepPoisson,
    DeepPoissonGamma,
    ZeroInflatedPoissonGamma,
)

log = logging.getLogger(__name__)


class LatentAdditive(PerturbationModel):
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
            embedding_width: int | None = None,
            perturbation_embedding_width: int | None = None,
            n_layers: int = 2,
            encoder_width: int = 128,
            decoder_width: int | None = None,
            latent_dim: int = 32,
            decoder_distribution: str = 'IsotropicGaussian',
            library_size: Literal['learned', 'observed'] | None = None,
            lr: float | None = None,
            wd: float | None = None,
            lr_scheduler: DictConfig | None = None,
            dropout: float | None = None,
            inject_covariates_encoder: bool = False,
            inject_covariates_decoder: bool = False,
            softplus_output: bool = False,
            use_legacy_negative_binomial: bool = False,
            count_based_input_expression: bool = False,
            dispersion_by_gene_cell: bool = False,
            use_legacy_mse: bool = False,
            embedding_decoder_path: str | None = None,
            **kwargs,
    ) -> None:
        """
        The constructor for the LatentAdditive class.

        Args:
            n_genes: Number of genes to use for prediction
            n_perts: Number of perturbations in the dataset
                (not including controls)
            n_layers: Number of layers in the encoder/decoder
            encoder_width: Width of the hidden layers in the encoder
            decoder_width: Width of the hidden layers in the decoder. If None, defaults to encoder_width
            latent_dim: Dimension of the latent space
            lr: Learning rate
            wd: Weight decay
            dropout: Dropout rate or None for no dropout.
            softplus_output: Whether to apply a softplus activation to the
                output of the decoder to enforce non-negativity
            inject_covariates_encoder: Whether to condition the encoder on
                covariates
            inject_covariates_decoder: Whether to condition the decoder on
                covariates
            loss: The loss function to use (either "nll" or "mse")
            count_based_input_expression: The input gene expressions (control/perturbed cells)
                are either count-based or already normalized
            embedding_decoder_path: Path to a saved embedding decoder model to load
        """
        super(LatentAdditive, self).__init__(
            n_genes=n_genes,
            n_perts=n_perts,
            transform=transform,
            context=context,
            evaluation=evaluation,
            embedding_width=embedding_width,
            perturbation_embedding_width=perturbation_embedding_width,
            lr=lr,
            wd=wd,
            lr_scheduler=lr_scheduler,
            count_based_input_expression=count_based_input_expression,
            embedding_decoder_path=embedding_decoder_path,
        )

        self.save_hyperparameters()

        # Set decoder_width to encoder_width if not specified
        if decoder_width is None:
            decoder_width = encoder_width

        if decoder_distribution in [dist.__name__ for dist in self.COUNT_DISTRIBUTIONS]:
            if library_size is None:
                raise ValueError(f"library_size must be set to 'learned' or 'observed' "
                                 f"if decoder_distribution is in {self.COUNT_DISTRIBUTIONS}")
            elif library_size == 'learned':
                log.warning("library_size is set to 'learned' but in the current implementation "
                            "it is not treated as a random variable")

        encoder_input_dim = (
            self.n_input_features + self.n_total_covariates if inject_covariates_encoder else self.n_input_features
        )
        decoder_input_dim = (
            latent_dim + self.n_total_covariates if inject_covariates_decoder else latent_dim
        )

        self.gene_encoder = MLP(
            encoder_input_dim, encoder_width, latent_dim, n_layers, dropout
        )
        self.pert_encoder = MLP(self.n_input_perturbation_features, encoder_width, latent_dim, n_layers, dropout)

        if decoder_distribution == "Gaussian":
            self.decoder = DeepGaussian(decoder_input_dim, decoder_width, self.n_output_features, n_layers, dropout)
        elif decoder_distribution == "IsotropicGaussian":
            self.decoder = DeepIsotropicGaussian(decoder_input_dim, decoder_width, self.n_output_features, n_layers,
                                                 dropout, softplus_output)
        elif decoder_distribution == "Poisson":
            self.decoder = DeepPoisson(decoder_input_dim, decoder_width, self.n_output_features, n_layers, dropout,
                                       library_size)
        elif decoder_distribution == "PoissonGamma":
            self.decoder = DeepPoissonGamma(decoder_input_dim, decoder_width, self.n_output_features, n_layers, dropout,
                                            library_size, use_legacy_negative_binomial)
        elif decoder_distribution == "ZeroInflatedPoissonGamma":
            self.decoder = ZeroInflatedPoissonGamma(
                decoder_input_dim, decoder_width, self.n_output_features, n_layers, dropout, library_size,
                use_legacy_negative_binomial, dispersion_by_gene_cell)
        else:
            raise ValueError(
                "decoder_distribution must be one of 'Gaussian', 'IsotropicGaussian', 'Poisson', 'PoissonGamma', 'ZeroInflatedPoissonGamma'"
            )

        self.decoder_distribution = decoder_distribution
        self.dropout = dropout
        self.softplus_output = softplus_output
        self.inject_covariates_encoder = inject_covariates_encoder
        self.inject_covariates_decoder = inject_covariates_decoder
        self.use_legacy_mse = use_legacy_mse

    def forward(
            self,
            control_input: torch.Tensor,
            perturbation: torch.Tensor,
            covariates: dict[str, torch.Tensor],
            observed_perturbed_input: torch.Tensor | None = None,
    ):
        if self.inject_covariates_encoder or self.inject_covariates_decoder:
            merged_covariates = torch.cat(
                [cov if cov.ndim == 2 else cov.squeeze() for cov in covariates.values()], dim=1
            )

        if type(self.decoder) in self.COUNT_DISTRIBUTIONS:
            library_size = (
                observed_perturbed_input.sum(axis=-1).reshape(-1, 1) \
                    if self.training or self.trainer.validating \
                    else control_input.sum(axis=-1).reshape(-1, 1)
            )
            if self.embedding_width is None and self.count_based_input_expression:
                # todo, check if normalization is necessary (check if scvi did that)
                # control_input in theory can be embeddings
                control_input = torch.log1p(control_input)

        if self.inject_covariates_encoder:
            control_input = torch.cat(
                [control_input, merged_covariates], dim=1
            )

        latent_control = self.gene_encoder(control_input)
        latent_perturbation = self.pert_encoder(perturbation)

        latent_perturbed = latent_control + latent_perturbation

        if self.inject_covariates_decoder:
            latent_perturbed = torch.cat(
                [latent_perturbed, merged_covariates], dim=1
            )
        predictions = self.decoder(
            latent_perturbed,
            library_size=library_size if type(self.decoder) in self.COUNT_DISTRIBUTIONS else None
        )

        return predictions

    def training_step(self, batch: Batch, batch_idx: int):
        if batch.control_embeddings is not None:
            control_input = batch.control_embeddings.squeeze()
        else:
            control_input = batch.controls.squeeze()

        if self.embedding_decoder is not None:
            observed_perturbed_input = batch.embeddings.squeeze()
        else:
            observed_perturbed_input = batch.gene_expression.squeeze()

        predictions = self.forward(
            control_input,
            batch.perturbations.squeeze(),
            batch.covariates,
            observed_perturbed_input
        )
        if self.decoder_distribution == 'IsotropicGaussian' and self.use_legacy_mse:
            train_loss = self.decoder.reconstruction_loss(predictions, observed_perturbed_input,
                                                          reduction='featurewise_mean')
        else:
            train_loss = self.decoder.reconstruction_loss(predictions, observed_perturbed_input)

        self.log(
            "train_loss",
            train_loss,
            prog_bar=True,
            logger=True,
            batch_size=len(batch),
            sync_dist=True,
        )
        return train_loss

    def validation_step(self, batch: Batch, batch_idx: int):
        if batch.control_embeddings is not None:
            control_input = batch.control_embeddings.squeeze()
        else:
            control_input = batch.controls.squeeze()

        if self.embedding_decoder is not None:
            observed_perturbed_input = batch.embeddings.squeeze()
        else:
            observed_perturbed_input = batch.gene_expression.squeeze()

        predictions = self.forward(
            control_input,
            batch.perturbations.squeeze(),
            batch.covariates,
            observed_perturbed_input
        )

        if self.decoder_distribution == 'IsotropicGaussian' and self.use_legacy_mse:
            val_loss = self.decoder.reconstruction_loss(predictions, observed_perturbed_input,
                                                        reduction='featurewise_mean')
        else:
            val_loss = self.decoder.reconstruction_loss(predictions, observed_perturbed_input)

        self.log(
            "val_loss",
            val_loss,
            on_step=True,
            prog_bar=True,
            logger=True,
            batch_size=len(batch),
            sync_dist=True,
        )
        return val_loss

    def predict(self, batch: Batch):
        if batch.embeddings is not None:
            control_input = batch.embeddings.squeeze()
        else:
            control_input = batch.gene_expression.squeeze()

        perturbation = batch.perturbations.squeeze().to(self.device)
        covariates = {k: v.to(self.device) for k, v in batch.covariates.items()}

        prediction = self.forward(
            control_input,
            perturbation,
            covariates,
        )
        if self.embedding_decoder is not None:
            library_size = batch.gene_expression.sum(axis=1).unsqueeze(1)
            prediction = self.embedding_decoder.decode(
                prediction,
                library=torch.log(library_size)  ## log library size
            )

        return prediction
