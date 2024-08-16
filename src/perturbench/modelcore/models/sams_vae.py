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

import numpy as np
import torch
import torch.nn.functional as F
import lightning as L
from torch.distributions.kl import kl_divergence
import torch.distributions as dist
from ..nn.vae import BaseEncoder, PerturbationConditionalEncoder
from ..nn.mlp import MLP
from .base import PerturbationModel

from perturbench.data.types import Batch
from typing import Tuple


class SparseAdditiveVAE(PerturbationModel):
    """
    Sparse Additive Variational Autoencoder (VAE) model, following the model proposed in the paper:

    Bereket, Michael, and Theofanis Karaletsos.
    "Modelling Cellular Perturbations with the Sparse Additive Mechanism Shift Variational Autoencoder."
    Advances in Neural Information Processing Systems 36 (2024).

    Attributes:
        n_genes (int): Number of genes.
        n_perts (int): Number of perturbations.
        lr (int): Learning rate.
        wd (int): Weight decay.
        lr_scheduler_freq (int): Frequency of the learning rate scheduler.
        lr_scheduler_patience (int): Patience of the learning rate scheduler.
        lr_scheduler_factor (int): Factor of the learning rate scheduler.
        latent_dim (int): Latent dimension.
        sparse_additive_mechanism (bool): Whether to use sparse additive mechanism.
        mean_field_encoding (bool): Whether to use mean field encoding.
        inject_covariates_encoder (bool): Whether to inject covariates in the encoder.
        inject_covariates_decoder (bool): Whether to inject covariates in the decoder.
        mean_field_encoding (bool): Whether to use mean field encoding.
        sparse_additive_mechanism (bool): Whether to use sparse additive mechanism.
        mask_prior_probability (float): The target probability for the masks.
        simple_latent (bool): Whether to use a simple encoding such as CPA or a full encoding for all perturbations (as SAMS paper suggests).
        datamodule (L.LightningDataModule | None): LightningDataModule for data loading.

    Methods:
        reparameterize(mu, log_var): Reparametrizes the Gaussian distribution.
        training_step(batch, batch_idx): Performs a training step.
        validation_step(batch, batch_idx): Performs a validation step.
        configure_optimizers(): Configures the optimizers.

    """

    def __init__(
        self,
        n_genes: int,
        n_perts: int,
        n_layers_encoder_x: int = 2,
        n_layers_encoder_e: int = 2,
        n_layers_decoder: int = 3,
        hidden_dim_x: int = 850,
        hidden_dim_cond: int = 128,
        latent_dim: int = 40,
        dropout: float = 0.2,
        inject_covariates_encoder: bool = False,
        inject_covariates_decoder: bool = False,
        mean_field_encoding: bool = False,
        sparse_additive_mechanism: bool = True,
        mask_prior_probability: float = 0.01,
        simple_latent: bool = True,
        lr: int | None = None,
        wd: int | None = None,
        lr_scheduler_freq: int | None = None,
        lr_scheduler_patience: int | None = None,
        lr_scheduler_factor: int | None = None,
        softplus_output: bool = True,
        datamodule: L.LightningDataModule | None = None,
    ) -> None:
        """
        Initializes the SparseAdditiveVAE model.

        Args:
            n_genes (int): Number of genes.
            n_perts (int): Number of perturbations.
            n_layers_encoder_x (int): Number of layers in the encoder for x.
            n_layers_encoder_e (int): Number of layers in the encoder for e.
            n_layers_decoder (int): Number of layers in the decoder.
            hidden_dim_x (int): Hidden dimension for x.
            hidden_dim_cond (int): Hidden dimension for the conditional input.
            latent_dim (int): Latent dimension.
            lr (int): Learning rate.
            wd (int): Weight decay.
            lr_scheduler_freq (int): Frequency of the learning rate scheduler.
            lr_scheduler_patience (int): Patience of the learning rate scheduler.
            lr_scheduler_factor (int): Factor of the learning rate scheduler.
            inject_covariates_encoder (bool): Whether to inject covariates in the encoder.
            inject_covariates_decoder (bool): Whether to inject covariates in the decoder.
            mean_field_encoding (bool): Whether to use mean field encoding.
            sparse_additive_mechanism (bool): Whether to use sparse additive mechanism.
            mask_prior_probability (float): The target probability for the masks.
            simple_latent (bool): Whether to use a simple encoding such as CPA or a full encoding for all perturbations (as SAMS paper suggests).
            softplus_output: Whether to apply a softplus activation to the
                output of the decoder to enforce non-negativity
            datamodule (L.LightningDataModule | None): LightningDataModule for data loading.

        Returns:
            None
        """

        super(SparseAdditiveVAE, self).__init__(
            datamodule=datamodule,
            lr=lr,
            wd=wd,
            lr_scheduler_freq=lr_scheduler_freq,
            lr_scheduler_patience=lr_scheduler_patience,
            lr_scheduler_factor=lr_scheduler_factor,
        )
        self.save_hyperparameters(ignore=["datamodule"])

        if n_genes is not None:
            self.n_genes = n_genes
        if n_perts is not None:
            self.n_perts = n_perts

        self.latent_dim = latent_dim
        self.latent_dim_pert = (
            latent_dim if simple_latent else latent_dim * self.n_perts
        )
        self.sparse_additive_mechanism = sparse_additive_mechanism
        self.mean_field_encoding = mean_field_encoding
        self.inject_covariates_encoder = inject_covariates_encoder
        self.inject_covariates_decoder = inject_covariates_decoder
        self.simple_latent = simple_latent
        self.mask_prior_probability = mask_prior_probability
        self.softplus_output = softplus_output

        if inject_covariates_encoder or inject_covariates_decoder:
            n_total_covs = np.sum(
                [
                    len(unique_covs)
                    for unique_covs in datamodule.train_context[
                        "covariate_uniques"
                    ].values()
                ]
            )

        encoder_input_dim = (
            self.n_genes + n_total_covs
            if self.inject_covariates_encoder
            else self.n_genes
        )
        decoder_input_dim = (
            latent_dim + n_total_covs if self.inject_covariates_decoder else latent_dim
        )

        if self.mean_field_encoding:
            self.encoder_x = BaseEncoder(
                input_dim=encoder_input_dim,
                hidden_dim=hidden_dim_x,
                latent_dim=latent_dim,
                n_layers=n_layers_encoder_x,
            )
            self.encoder_e = BaseEncoder(
                input_dim=self.n_perts,
                hidden_dim=self.n_perts,
                latent_dim=self.latent_dim_pert,
                n_layers=n_layers_encoder_e,
            )
        elif self.sparse_additive_mechanism:
            self.encoder_x = PerturbationConditionalEncoder(
                input_dim=encoder_input_dim,
                condition_dim=latent_dim,
                hidden_dim_x=hidden_dim_x,
                hidden_dim_cond=hidden_dim_cond,
                output_dim=latent_dim,
                n_layers=n_layers_encoder_x,
            )
            self.encoder_e = PerturbationConditionalEncoder(
                input_dim=latent_dim,
                condition_dim=self.n_perts,
                hidden_dim_x=hidden_dim_cond,
                hidden_dim_cond=hidden_dim_cond,
                output_dim=latent_dim,
                n_layers=n_layers_encoder_e,
            )
        else:
            self.encoder_x = PerturbationConditionalEncoder(
                input_dim=encoder_input_dim,
                condition_dim=latent_dim,
                hidden_dim_x=hidden_dim_x,
                hidden_dim_cond=hidden_dim_cond,
                output_dim=latent_dim,
                n_layers=n_layers_encoder_x,
            )
            self.encoder_e = BaseEncoder(
                input_dim=self.n_perts,
                hidden_dim=self.n_perts,
                latent_dim=self.latent_dim_pert,
                n_layers=n_layers_encoder_e,
            )
        if self.sparse_additive_mechanism:
            self.encoder_m = MLP(
                self.n_perts,
                hidden_dim_cond,
                output_dim=self.latent_dim_pert,
                n_layers=n_layers_encoder_e,
            )
        self.decoder = MLP(
            decoder_input_dim,
            hidden_dim_x,
            self.n_genes,
            n_layers_decoder,
            dropout=dropout,
        )

    def forward(
        self,
        observed_perturbed_expression: torch.Tensor,
        perturbation: torch.Tensor,
        covariates: dict,
    ) -> Tuple:

        batch_size = observed_perturbed_expression.shape[0]

        if self.inject_covariates_encoder or self.inject_covariates_decoder:
            merged_covariates = torch.cat(
                [cov.squeeze() for cov in covariates.values()], dim=1
            )

        if self.inject_covariates_encoder:
            observed_expression_with_covariates = torch.cat(
                [observed_perturbed_expression, merged_covariates.to(self.device)],
                dim=1,
            )
        else:
            observed_expression_with_covariates = observed_perturbed_expression

        if self.sparse_additive_mechanism:
            m_logits = self.encoder_m(perturbation)
            m_probs = 1 / (1 + torch.exp(-m_logits))
            m = torch.bernoulli(m_probs) - m_probs.detach() + m_probs

        if self.sparse_additive_mechanism and not self.mean_field_encoding:
            if self.simple_latent:
                e_mu, e_log_var = self.encoder_e(m, perturbation)
                e = self.reparameterize(e_mu, e_log_var)
            else:
                perturbation_flattened = torch.flatten(
                    perturbation.unsqueeze(1).expand(-1, self.n_perts, -1), end_dim=-2
                )
                m_flattened = torch.flatten(
                    m.reshape(batch_size, self.n_perts, self.latent_dim), end_dim=-2
                )
                e_mu, e_log_var = self.encoder_e(m_flattened, perturbation_flattened)
                e = self.reparameterize(e_mu, e_log_var)
        else:
            e_mu, e_log_var = self.encoder_e(perturbation)
            e = self.reparameterize(e_mu, e_log_var)

        if self.sparse_additive_mechanism:
            if self.simple_latent:
                z_p = m * e
            else:
                z_p = torch.bmm(
                    perturbation.reshape(batch_size, 1, self.n_perts),
                    m.reshape(batch_size, self.n_perts, self.latent_dim)
                    * e.reshape(batch_size, self.n_perts, self.latent_dim),
                ).squeeze()
        else:
            if self.simple_latent:
                z_p = e
            else:
                z_p = torch.bmm(
                    perturbation.reshape(batch_size, 1, self.n_perts),
                    e.reshape(batch_size, self.n_perts, self.latent_dim),
                ).squeeze()

        if self.mean_field_encoding:
            z_mu_x, z_log_var_x = self.encoder_x(observed_expression_with_covariates)
        else:
            z_mu_x, z_log_var_x = self.encoder_x(
                observed_expression_with_covariates, z_p
            )

        z_basal = self.reparameterize(z_mu_x, z_log_var_x)
        z = z_basal + z_p

        if self.inject_covariates_decoder:
            z = torch.cat([z, merged_covariates], dim=1)

        if self.softplus_output:
            x_sample = F.softplus(self.decoder(z))
        else:
            x_sample = F.relu(self.decoder(z))

        # Define distributons for kl_divergence
        q_z = dist.Normal(loc=z_mu_x, scale=torch.exp(0.5 * z_log_var_x))
        q_e = dist.Normal(loc=e_mu, scale=torch.exp(0.5 * e_log_var))
        p_z = dist.Normal(loc=torch.zeros_like(z_mu_x), scale=torch.ones_like(z_mu_x))
        p_e = dist.Normal(loc=torch.zeros_like(e_mu), scale=torch.ones_like(e_mu))

        mae = (
            torch.nn.L1Loss(reduction="none")(observed_perturbed_expression, x_sample)
            .sum(axis=1)
            .mean()
        )
        # mse = nn.MSELoss(reduction='none')(observed_perturbed_expression, x_sample).sum(axis=1).mean() # could use reduction = 'sum', but currently set up to highlight the summation over different dimensions
        kl_qz_pz = kl_divergence(q_z, p_z).sum(axis=1).mean()
        kl_qe_pe = kl_divergence(q_e, p_e).sum(axis=1).mean()
        kl_sum = kl_qz_pz + kl_qe_pe

        if self.sparse_additive_mechanism:
            q_m = dist.Bernoulli(probs=m_probs)
            p_m = dist.Bernoulli(probs=self.mask_prior_probability * torch.ones_like(m))
            kl_qm_pm = kl_divergence(q_m, p_m).sum(axis=1).mean()
            kl_sum += (
                kl_qm_pm  # add missing kl_divergence term for sparse additive mechanism
            )

        return x_sample, mae, kl_sum

    def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:

        observed_perturbed_expression = batch.gene_expression.squeeze()
        perturbation = batch.perturbations.squeeze()
        covariates = batch.covariates

        _, mse, kl_sum = self(observed_perturbed_expression, perturbation, covariates)
        loss = mse + kl_sum
        self.log(
            "recon_loss", mse, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "kl_div", kl_sum, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        return loss

    def validation_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:

        observed_perturbed_expression = batch.gene_expression.squeeze()
        perturbation = batch.perturbations.squeeze()
        covariates = batch.covariates

        _, mse, kl_sum = self(observed_perturbed_expression, perturbation, covariates)
        val_loss = mse + kl_sum
        self.log(
            "val_loss",
            val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return val_loss

    def predict(self, batch: Batch) -> torch.Tensor:

        observed_perturbed_expression = batch.gene_expression.squeeze().to(self.device)
        perturbation = batch.perturbations.squeeze().to(self.device)
        covariates = batch.covariates

        x_sample, mse, kl_sum = self(
            observed_perturbed_expression, perturbation, covariates
        )
        return x_sample

    def reparameterize(
        self,
        mu: torch.Tensor,
        log_var: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reparametrizes the Gaussian distribution so (stochastic) backpropagation can be applied.
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return mu + eps * std
