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

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

from perturbench.data.types import Batch
from .base import PerturbationModel
from ..nn.mlp import MLP

class EmbeddingModel(PerturbationModel):
    """
    A latent additive model for predicting perturbation effects
    """

    def __init__(
        self,
        n_genes: int,
        n_perts: int,
        inject_covariates: False,
        lr: float | None = None,
        wd: float | None = None,
        embedding_path: str = None,
        dropout: float = 0.0,
        n_layers: int = 1,
        latent_dim: int = 64,
        encoder_width: int = 128,
        lr_scheduler_freq: int | None = None,
        lr_scheduler_interval: str | None = None,
        lr_scheduler_patience: int | None = None,
        lr_scheduler_factor: int | None = None,
        softplus_output: bool = True,
        datamodule: L.LightningDataModule | None = None,
    ) -> None:
        """
        The constructor for the LinearAdditive class.

        Args:
            n_genes (int): Number of genes in the dataset
            n_perts (int): Number of perturbations in the dataset (not including controls)
            lr (float): Learning rate
            wd (float): Weight decay
            lr_scheduler_freq (int): How often the learning rate scheduler checks val_loss
            lr_scheduler_interval (str): Whether to check val_loss every epoch or every step
            lr_scheduler_patience (int): Learning rate scheduler patience
            lr_scheduler_factor (float): Factor by which to reduce learning rate when learning rate scheduler triggers
            inject_covariates: Whether to condition the linear layer on
                covariates
            softplus_output: Whether to apply a softplus activation to the
                output of the decoder to enforce non-negativity
            datamodule: The datamodule used to train the model
        """
        super(EmbeddingModel, self).__init__(
            datamodule=datamodule,
            lr=lr,
            wd=wd,
            lr_scheduler_interval=lr_scheduler_interval,
            lr_scheduler_freq=lr_scheduler_freq,
            lr_scheduler_patience=lr_scheduler_patience,
            lr_scheduler_factor=lr_scheduler_factor,
        )
        self.save_hyperparameters(ignore=["datamodule"])
        self.softplus_output = softplus_output

        if n_genes is not None:
            self.n_genes = n_genes
        if n_perts is not None:
            self.n_perts = n_perts

        embedding_df = pd.read_parquet(embedding_path)
        self.datamodule = datamodule
        
        self._init_embedding(embedding_df, self.datamodule.train_context["perturbation_uniques"])
        
        self.pert_encoder = MLP(
            self.embedding.shape[1], encoder_width, latent_dim // 2, n_layers, dropout
        )

        self.gex_encoder = MLP(
            self.n_genes, encoder_width, latent_dim // 2, n_layers, dropout
        )
        
        self.decoder = MLP(
            latent_dim, encoder_width, self.n_genes, n_layers, dropout
        )


    def _init_embedding(self, embedding_df: pd.DataFrame, perturbations: list[str]):
        gene_map = {
            "FOXL2NB": "C3orf72",
            "MIDEAS": "ELMSAN1",
            "CBARP": "C19orf26",
            "MAP3K21": "KIAA1804",
            "NUP50-DT": "NUP50-AS1",
            "SNHG29": "LRRC75A-AS1",
            "STING1": "TMEM173",
            "ATP5MK": "ATP5MD"
        }
        
        embedding_df = embedding_df.rename(index=gene_map)
        embedding_df = embedding_df.loc[perturbations]
        
        self.register_buffer("embedding", torch.Tensor(embedding_df.values))   
    
    def forward(
        self,
        control_expression: torch.Tensor,
        perturbation: torch.Tensor,
        covariates: dict,
    ):

        latent_control = self.gex_encoder(control_expression)
        perturbation_embedding = torch.matmul(perturbation, self.embedding)
        latent_perturbation = self.pert_encoder(perturbation_embedding)
        latent = torch.cat([latent_control, latent_perturbation], dim=1)
        
        predicted_perturbed_expression = self.decoder(latent)
        
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

    def predict(self, batch: Batch):
        control_expression = batch.gene_expression.squeeze().to(self.device)
        perturbation = batch.perturbations.squeeze().to(self.device)
        covariates = {k: v.to(self.device) for k, v in batch.covariates.items()}
        predicted_perturbed_expression = self.forward(
            control_expression,
            perturbation,
            covariates,
        )
        return predicted_perturbed_expression
