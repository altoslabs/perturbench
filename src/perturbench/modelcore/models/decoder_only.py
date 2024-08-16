"""
BSD 3-Clause License

Copyright (c) 2024, <anonymized authors of NeurIPS submission #1306>

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import torch
import torch.nn.functional as F
import lightning as L
import numpy as np

from ..nn.mlp import MLP
from .base import PerturbationModel
from perturbench.data.types import Batch


class DecoderOnly(PerturbationModel):
    """
    A latent additive model for predicting perturbation effects
    """

    def __init__(
        self,
        n_genes: int,
        n_perts: int,
        n_layers=2,
        encoder_width=128,
        softplus_output=True,
        use_covariates=True,
        use_perturbations=True,
        lr: float | None = None,
        wd: float | None = None,
        lr_scheduler_freq: int | None = None,
        lr_scheduler_patience: int | None = None,
        lr_scheduler_factor: float | None = None,
        datamodule: L.LightningDataModule | None = None,
    ) -> None:
        """
        The constructor for the DecoderOnly class.

        Args:
            n_genes (int): Number of genes to use for prediction
            n_perts (int): Number of perturbations in the dataset (not including controls)
            n_layers (int): Number of layers in the encoder/decoder
            lr (float): Learning rate
            wd (float): Weight decay
            lr_scheduler_freq (int): How often the learning rate scheduler checks val_loss
            lr_scheduler_patience (int): Learning rate scheduler patience
            lr_scheduler_factor (float): Factor by which to reduce learning rate when learning rate scheduler triggers
            softplus_output (bool): Whether to apply a softplus activation to the output of the decoder to enforce non-negativity
        """

        super(DecoderOnly, self).__init__(
            datamodule=datamodule,
            lr=lr,
            wd=wd,
            lr_scheduler_freq=lr_scheduler_freq,
            lr_scheduler_patience=lr_scheduler_patience,
            lr_scheduler_factor=lr_scheduler_factor,
        )
        self.save_hyperparameters(ignore=["datamodule"])

        if not (use_covariates or use_perturbations):
            raise ValueError(
                "'use_covariates' and 'use_perturbations' can not both be false. Either covariates or perturbations have to be used."
            )

        if n_genes is not None:
            self.n_genes = n_genes
        if n_perts is not None:
            self.n_perts = n_perts

        n_total_covariates = (
            np.sum(
                [
                    len(unique_covs)
                    for unique_covs in datamodule.train_context[
                        "covariate_uniques"
                    ].values()
                ]
            )
            if use_covariates
            else 0
        )

        n_perts = self.n_perts if use_perturbations else 0

        decoder_input_dim = n_total_covariates + n_perts

        self.decoder = MLP(decoder_input_dim, encoder_width, self.n_genes, n_layers)
        self.softplus_output = softplus_output
        self.use_covariates = use_covariates
        self.use_perturbations = use_perturbations

    def forward(
        self,
        control_expression: torch.Tensor,
        perturbation: torch.Tensor,
        covariates: dict[str, torch.Tensor],
    ):
        if self.use_covariates and self.use_perturbations:
            embedding = torch.cat([cov.squeeze() for cov in covariates.values()], dim=1)
            embedding = torch.cat([perturbation, embedding], dim=1)
        elif self.use_covariates:
            embedding = torch.cat([cov.squeeze() for cov in covariates.values()], dim=1)
        elif self.use_perturbations:
            embedding = perturbation

        predicted_perturbed_expression = self.decoder(embedding)

        if self.softplus_output:
            predicted_perturbed_expression = F.softplus(predicted_perturbed_expression)
        return predicted_perturbed_expression

    def training_step(self, batch: Batch, batch_idx: int):
        (
            observed_perturbed_expression,
            control_expression,
            perturbation,
            covariates,
            _,
        ) = self.unpack_batch(batch)
        predicted_perturbed_expression = self.forward(
            control_expression, perturbation, covariates
        )
        loss = F.mse_loss(predicted_perturbed_expression, observed_perturbed_expression)
        self.log("train_loss", loss, prog_bar=True, logger=True, batch_size=len(batch))
        return loss

    def validation_step(self, batch: Batch, batch_idx: int):
        (
            observed_perturbed_expression,
            control_expression,
            perturbation,
            covariates,
            _,
        ) = self.unpack_batch(batch)
        predicted_perturbed_expression = self.forward(
            control_expression, perturbation, covariates
        )
        val_loss = F.mse_loss(
            predicted_perturbed_expression, observed_perturbed_expression
        )
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
        control_expression = batch.gene_expression.squeeze().to(self.device)
        perturbation = batch.perturbations.squeeze().to(self.device)
        covariates = {k: v.to(self.device) for k, v in batch.covariates.items()}

        predicted_perturbed_expression = self.forward(
            control_expression,
            perturbation,
            covariates,
        )
        return predicted_perturbed_expression
