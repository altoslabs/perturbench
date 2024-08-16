"""
BSD 3-Clause License

Copyright (c) 2024, Mohammad Lotfollahi, Theislab

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
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence as kl
from torchmetrics.functional import accuracy
from typing import Optional
import lightning as L

from ..nn.vae import VariationalEncoder, VariationalDecoder
from ..nn.mlp import MLP
from perturbench.data.types import Batch
from .base import PerturbationModel


class CPA(PerturbationModel):
    """
    CPA module using Gaussian/NegativeBinomial/Zero-InflatedNegativeBinomial Likelihood
    """

    def __init__(
        self,
        n_genes: int | None = None,
        n_perts: int | None = None,
        n_latent: int = 128,
        recon_loss: str = "gauss",
        hidden_dim: int = 256,
        n_layers_encoder: int = 3,
        n_layers_pert_emb: int = 2,
        n_layers_covar_emb: int = 1,
        adv_classifier_hidden_dim: int = 128,
        adv_classifier_n_layers: int = 2,
        variational: bool = True,
        ## TODO do we want to separate out optimizers, lr, wd etc. for adversary, autoencoder and dosers?
        ## Alternatively, this can be achieved by balancing losses for different components and adding
        ## an explicit loss term for regularization (instead of using weight decay).
        lr: float = 1e-3,
        wd: float = 1e-8,
        lr_scheduler_freq: int | None = None,
        lr_scheduler_interval: str | None = None,
        lr_scheduler_patience: int | None = None,
        lr_scheduler_factor: float | None = None,
        kl_weight: float = 1.0,
        adv_weight: float = 1.0,
        dropout: float = 0.1,
        penalty_weight: float = 10.0,
        datamodule: L.LightningDataModule | None = None,
        adv_steps: int = 7,
        use_adversary: bool = True,
        use_covariates: bool = True,
        softplus_output: bool = False,
        elementwise_affine: bool = False,
    ):
        """The constructor for the CPA module class.
        Args:
            n_genes: Number of genes.
            n_perts: Number of perturbations.
            drug_embeddings: Drug embeddings.
            n_latent: Number of latent variables.
            recon_loss: Reconstruction loss type.
            hidden_dim: Hidden dimension.
            n_layers_encoder: Number of encoder layers.
            n_layers_decoder: Number of decoder layers.
            n_layers_pert_emb: Number of perturbation embedding layers.
            n_layers_covar_emb: Number of covariate embedding layers.
            adv_classifier_hidden_dim: Adversarial classifier hidden dimension.
            adv_classifier_n_layers: Number of adversarial classifier layers.
            variational: Whether to use variational autoencoder.
            seed: Random seed.
            lr: Learning rate.
            wd: Weight decay.
            kl_weight: KL divergence weight.
            adv_weight: Adversarial weight.
            dropout: Dropout rate.
            penalty_weight: Penalty weight.
            datamodule: Data module.
            adv_steps: Number of adversarial steps.
            use_adversary: Whether to use the adversarial component.
            use_covariates: Whether to use additive covariate conditioning.
            softplus_output: Whether to apply a softplus activation to the output.
            elementwise_affine: Whether to use elementwise affine in the layer norms
        """
        super(CPA, self).__init__(
            datamodule=datamodule,
            lr=lr,
            wd=wd,
            lr_scheduler_freq=lr_scheduler_freq,
            lr_scheduler_patience=lr_scheduler_patience,
            lr_scheduler_factor=lr_scheduler_factor,
            lr_scheduler_interval=lr_scheduler_interval,
        )
        self.save_hyperparameters(ignore=["datamodule"])

        recon_loss = recon_loss.lower()
        assert recon_loss in ["gauss", "zinb", "nb"]

        if n_genes is not None:
            self.n_genes = n_genes
        if n_perts is not None:
            self.n_perts = n_perts

        self.n_latent = n_latent
        self.recon_loss = recon_loss
        self.variational = variational
        self.hidden_dim = hidden_dim
        self.n_layers_pert_emb = n_layers_pert_emb
        self.n_layers_covar_emb = n_layers_covar_emb
        self.n_layers_encoder = n_layers_encoder
        self.kl_weight = kl_weight
        self.adv_weight = adv_weight
        self.penalty_weight = penalty_weight
        self.adv_classifier_hidden_dim = adv_classifier_hidden_dim
        self.adv_classifier_n_layers = adv_classifier_n_layers
        self.dropout = dropout
        self.adv_steps = adv_steps
        self.softplus_output = softplus_output
        self.adv_loss_drugs = nn.CrossEntropyLoss()
        self.adv_loss_fn = nn.CrossEntropyLoss()
        self.use_adversary = use_adversary
        self.use_covariates = use_covariates

        self.encoder = VariationalEncoder(
            input_dim=self.n_input_features,
            hidden_dim=self.hidden_dim,
            latent_dim=self.n_latent,
            n_layers=self.n_layers_encoder,
            dropout=self.dropout,
        )

        if self.use_covariates:
            self.covars_encoder = {
                covar: uniques
                for covar, uniques in datamodule.train_context[
                    "covariate_uniques"
                ].items()
                if len(uniques) > 1
            }
        else:
            self.covars_encoder = {}

        self.decoder = VariationalDecoder(
            output_dim=self.n_genes,
            hidden_dim=self.hidden_dim,
            latent_dim=self.n_latent,
            n_layers=self.n_layers_encoder,  # ∞ For now set this to same as encoder (same as dropout and hidden dim)
            dropout=self.dropout,
        )

        self.pert_network = MLP(
            input_dim=self.n_perts,
            hidden_dim=self.hidden_dim,
            output_dim=self.n_latent,
            n_layers=self.n_layers_pert_emb,
            dropout=self.dropout,
            elementwise_affine=elementwise_affine,
        )

        self.perturbation_adversary_classifier = MLP(
            self.n_latent,
            self.adv_classifier_hidden_dim,
            self.n_perts,
            self.adv_classifier_n_layers,
            self.dropout,
            elementwise_affine=elementwise_affine,
        )

        if self.use_covariates:
            self.covars_embeddings = nn.ModuleDict(
                {
                    key: MLP(
                        input_dim=len(unique_covars),
                        output_dim=n_latent,
                        hidden_dim=hidden_dim,
                        n_layers=self.n_layers_covar_emb,
                        dropout=dropout,
                        elementwise_affine=elementwise_affine,
                    )
                    for key, unique_covars in self.covars_encoder.items()
                    if len(unique_covars) > 1
                }
            )

            self.covars_adversary_classifiers = dict()
            for covar, unique_covars in self.covars_encoder.items():
                self.covars_adversary_classifiers[covar] = MLP(
                    self.n_latent,
                    self.adv_classifier_hidden_dim,
                    len(unique_covars),
                    self.adv_classifier_n_layers,
                    self.dropout,
                    elementwise_affine=elementwise_affine,
                )
            self.covars_adversary_classifiers = nn.ModuleDict(
                self.covars_adversary_classifiers
            )

        else:
            self.covars_embeddings = None
            self.covars_adversary_classifiers = None

    def unpack_batch(self, batch: Batch):
        if batch.embeddings is not None:
            embeddings = batch.embeddings.squeeze()
        else:
            embeddings = None

        x = batch.gene_expression.squeeze()  # batch_size, n_genes
        perts = batch.perturbations.squeeze()  # batch_size, n_perts
        covars_dict = batch.covariates

        return dict(
            embeddings=embeddings,
            x=x,
            perts=perts,
            covars_dict=covars_dict,
        )

    def inference(
        self,
        x: torch.Tensor,
        perts: torch.Tensor,
        covars_dict: dict[str, torch.Tensor],
        embeddings: torch.Tensor | None = None,
        n_samples: int = 1,
        covars_to_add: Optional[list] = None,
    ):
        if embeddings is not None:
            x_ = embeddings
            library = None, None
        elif self.recon_loss in ["nb", "zinb"]:
            # log the input to the variational distribution for numerical stability
            x_ = torch.log(1 + x)
            library = torch.log(x.sum(1)).unsqueeze(1)
        else:
            x_ = x
            library = None, None

        if self.variational:
            qz, z_basal = self.encoder(x_)
        else:
            qz, z_basal = None, self.encoder(x_)

        if self.variational and n_samples > 1:
            sampled_z = qz.sample((n_samples,))
            z_basal = self.encoder.z_transformation(sampled_z)
            if self.recon_loss in ["nb", "zinb"]:
                library = library.unsqueeze(0).expand(
                    (n_samples, library.size(0), library.size(1))
                )

        z_pert_true = self.pert_network(perts)  # perturbation encoder
        z_pert = z_pert_true
        z_covs_dict = torch.zeros_like(z_basal)  # ([n_samples,] batch_size, n_latent)

        if covars_to_add is None:
            covars_to_add = list(self.covars_encoder.keys())

        z_covs_dict = {}
        for covar in self.covars_encoder:
            if covar in covars_to_add:
                covars_input = covars_dict[covar]
                z_cov = self.covars_embeddings[covar](covars_input)
                z_covs_dict[covar] = z_cov

        if len(z_covs_dict) > 0:
            z_covs = torch.stack(list(z_covs_dict.values()), dim=0).sum(dim=0)
        else:
            z_covs = torch.zeros_like(z_basal)

        z = z_basal + z_pert + z_covs
        z_no_pert = z_basal + z_covs

        return dict(
            z=z,
            z_no_pert=z_no_pert,
            z_basal=z_basal,
            z_covs=z_covs,
            z_pert=z_pert.sum(dim=1),
            library=library,
            qz=qz,
        )

    def generative(
        self,
        z: torch.Tensor,
    ):
        px_mean, px_var = self.decoder(z)
        if self.softplus_output:
            px_mean = F.softplus(px_mean)
        px = Normal(loc=px_mean, scale=px_var.sqrt())
        pz = Normal(torch.zeros_like(z), torch.ones_like(z))
        return dict(px=px, pz=pz)

    def loss(
        self,
        x: torch.Tensor,
        perturbations: torch.Tensor,
        covariates: dict[str, torch.Tensor],
        inference_outputs: dict[str, torch.Tensor],
        generative_outputs: dict[str, torch.Tensor],
        batch_idx: int,
    ):
        """Computes the reconstruction loss (AE) or the ELBO (VAE)"""
        px = generative_outputs["px"]
        recon_loss = -px.log_prob(x).sum(dim=-1).mean()

        if self.variational:
            qz = inference_outputs["qz"]
            pz = generative_outputs["pz"]  # just a standard gaussian

            kl_divergence_z = kl(qz, pz).sum(dim=1)
            kl_loss = kl_divergence_z.mean()
        else:
            kl_loss = torch.zeros_like(recon_loss)

        if self.use_adversary:
            adv_loss = self.adversarial_loss(
                perturbations, covariates, inference_outputs["z_basal"], self.training
            )
        else:
            adv_loss = {
                "adv_loss": torch.zeros_like(recon_loss),
                "penalty_adv": torch.zeros_like(recon_loss),
                "penalty_covars": torch.zeros_like(recon_loss),
                "penalty_perts": torch.zeros_like(recon_loss),
                "acc_perts": torch.zeros_like(recon_loss),
                "covariate_classfier_loss": torch.zeros_like(recon_loss),
                "perturbation_classifier_loss": torch.zeros_like(recon_loss),
            }

        total_loss = (
            recon_loss
            + kl_loss * self.kl_weight
            + self.adv_weight * adv_loss["adv_loss"]
        )
        if batch_idx % self.adv_steps != 0:
            total_loss += self.penalty_weight * adv_loss["penalty_adv"]

        return {
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "adv_loss": adv_loss["adv_loss"],
            "covariate_classfier_loss": adv_loss["covariate_classfier_loss"],
            "perturbation_classifier_loss": adv_loss["perturbation_classifier_loss"],
            "penalty_adv": adv_loss["penalty_adv"],
            "penalty_covars": adv_loss["penalty_covars"],
            "penalty_perts": adv_loss["penalty_perts"],
            "acc_perts": adv_loss["acc_perts"],
        }

    def adversarial_loss(
        self,
        perturbations: torch.Tensor,
        covariates: dict[str, torch.Tensor],
        z_basal: torch.Tensor,
        compute_penalty: bool = True,
    ):
        """Computes adversarial classification losses and regularizations"""
        if compute_penalty:
            z_basal = z_basal.requires_grad_(True)

        covars_pred_logits = {}
        for covar in self.covars_encoder.keys():
            if self.covars_adversary_classifiers[covar] is not None:
                covars_pred_logits[covar] = self.covars_adversary_classifiers[covar](
                    z_basal
                )
            else:
                covars_pred_logits[covar] = None

        adv_results = {}

        # Classification losses for different covariates
        for covar, covars in self.covars_encoder.items():
            adv_results[f"adv_{covar}"] = (
                self.adv_loss_fn(  # ∞ we've removed mixup for now
                    covars_pred_logits[covar],
                    covariates[covar],
                )
                if covars_pred_logits[covar] is not None
                else torch.as_tensor(0.0).to(self.device)
            )

            adv_results[f"acc_{covar}"] = (
                accuracy(
                    covars_pred_logits[covar].argmax(1),
                    covariates[covar].argmax(1),
                    task="multiclass",
                    num_classes=len(covars),
                )
                if covars_pred_logits[covar] is not None
                else torch.as_tensor(0.0).to(self.device)
            )

        if len(self.covars_encoder) > 0:
            adv_results["covariate_classfier_loss"] = sum(
                [adv_results[f"adv_{key}"] for key in self.covars_encoder.keys()]
            )
        else:
            adv_results["covariate_classfier_loss"] = torch.as_tensor(0.0).to(
                self.device
            )

        # TODO Not using mixups for now.
        perturbations_pred_logits = self.perturbation_adversary_classifier(z_basal)

        adv_results["perturbation_classifier_loss"] = self.adv_loss_drugs(
            perturbations_pred_logits, perturbations
        )

        adv_results["acc_perts"] = accuracy(
            perturbations_pred_logits.argmax(1),
            perturbations.argmax(1),
            average="macro",
            num_classes=self.n_perts,
            task="multiclass",
        )

        adv_results["adv_loss"] = (
            adv_results["covariate_classfier_loss"]
            + adv_results["perturbation_classifier_loss"]
        )

        if compute_penalty:
            # Penalty losses
            for covar in self.covars_encoder.keys():
                adv_results[f"penalty_{covar}"] = (
                    (
                        torch.autograd.grad(
                            covars_pred_logits[covar].sum(),
                            z_basal,
                            create_graph=True,
                            retain_graph=True,
                            only_inputs=True,
                        )[0]
                        .pow(2)
                        .mean()
                    )
                    if covars_pred_logits[covar] is not None
                    else torch.as_tensor(0.0).to(self.device)
                )

            if len(self.covars_encoder) > 0:
                adv_results["penalty_covars"] = sum(
                    [
                        adv_results[f"penalty_{covar}"]
                        for covar in self.covars_encoder.keys()
                    ]
                )
            else:
                adv_results["penalty_covars"] = torch.as_tensor(0.0).to(self.device)

            adv_results["penalty_perts"] = (
                torch.autograd.grad(
                    perturbations_pred_logits.sum(),
                    z_basal,
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True,
                )[0]
                .pow(2)
                .mean()
            )

            adv_results["penalty_adv"] = (
                adv_results["penalty_perts"] + adv_results["penalty_covars"]
            )
        else:
            for covar in self.covars_encoder.keys():
                adv_results[f"penalty_{covar}"] = torch.as_tensor(0.0).to(self.device)

            adv_results["penalty_covars"] = torch.as_tensor(0.0).to(self.device)
            adv_results["penalty_perts"] = torch.as_tensor(0.0).to(self.device)
            adv_results["penalty_adv"] = torch.as_tensor(0.0).to(self.device)

        return adv_results

    def _get_dict_if_none(param):
        param = {} if not isinstance(param, dict) else param

        return param

    def forward(
        self,
        batch: Batch,
    ):
        embeddings, x, perts, covars_dict = self.unpack_batch(batch).values()
        inference_outputs = self.inference(x, perts, covars_dict, embeddings)

        generative_outputs = self.generative(inference_outputs["z"])
        return inference_outputs, generative_outputs

    def training_step(self, batch: Batch, batch_idx: int):
        inference_outputs, generative_outputs = self.forward(batch)
        losses = self.loss(
            batch.gene_expression,
            batch.perturbations,
            batch.covariates,
            inference_outputs,
            generative_outputs,
            batch_idx,
        )
        if self.training:
            for key, value in losses.items():
                self.log(
                    "train_" + key,
                    value,
                    prog_bar=True,
                    logger=True,
                    batch_size=len(batch),
                )
        return losses["total_loss"]

    def validation_step(self, batch: Batch, batch_idx: int):
        loss = self.training_step(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True, logger=True, batch_size=len(batch))
        return loss

    def predict(self, batch: Batch):
        _, generative_distributions = self.forward(batch)
        return generative_distributions["px"].mean
