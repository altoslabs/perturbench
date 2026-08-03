'''
BSD 3-Clause License

Copyright (c) 2023, Mohammad Lotfollahi, Theislab
All rights reserved.

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

Used Theis lab latest code _module.py
Removed CPA_REGISTRY_KEYS
For now removed mix_up & disentanglement
Simplified code so that the forward step consists of inference (encoder) and generative (decoder)
Need to consider zinb, nb or gaussian
Need to handle doses separately
removed var_activation

'''
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence as kl
from torchmetrics.functional import accuracy
from typing import Optional, Literal
from omegaconf import DictConfig, OmegaConf
import logging

from ..nn.vae import VariationalEncoder
from ..nn.decoders import (
    DeepGaussian,
    DeepIsotropicGaussian,
    DeepPoisson,
    DeepPoissonGamma,
    ZeroInflatedPoissonGamma,
)
from ..nn.mlp import MLP
from perturbench.data.types import Batch
from perturbench.data.transforms.base import Dispatch

from .base import PerturbationModel
from perturbench.modelcore.utils import instantiate_with_context

log = logging.getLogger(__name__)


class CPA(PerturbationModel):
    """
    CPA module using Gaussian/NegativeBinomial/Zero-InflatedNegativeBinomial Likelihood
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
            n_latent: int = 128,
            decoder_distribution: str = 'IsotropicGaussian',
            library_size: Literal['learned', 'observed'] | None = None,
            hidden_dim: int = 256,
            n_layers_encoder: int = 3,
            n_layers_pert_emb: int = 2,
            n_layers_covar_emb: int = 1,
            adv_classifier_hidden_dim: int = 128,
            adv_classifier_n_layers: int = 2,
            variational: bool = True,
            lr: float = 1e-3,
            wd: float = 1e-8,
            lr_scheduler: DictConfig | None = None,
            kl_weight: float = 1.0,
            adv_weight: float = 1.0,
            dropout: float = 0.1,
            penalty_weight: float = 10.0,
            adv_steps: int = 7,
            n_warmup_epochs: int = 5,
            use_adversary: bool = True,
            use_covariates: bool = True,
            softplus_output: bool = False,
            elementwise_affine: bool = False,
            use_legacy_negative_binomial: bool = False,
            count_based_input_expression: bool = False,
            dispersion_by_gene_cell: bool = False,
            **kwargs,
    ):
        """The constructor for the CPA module class.
        Args:
            n_genes: Number of genes. 
            n_perts: Number of perturbations. 
            drug_embeddings: Drug embeddings. 
            n_latent: Number of latent variables.
            decoder_distribution: Distribution of the decoder.
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
            n_warmup_epochs: Number of warmup epochs for the autoencoder
            use_adversary: Whether to use the adversarial component.
            use_covariates: Whether to use additive covariate conditioning.
            softplus_output: Whether to apply a softplus activation to the output.
            elementwise_affine: Whether to use elementwise affine in the layer norms
        """
        super(CPA, self).__init__(
            n_genes=n_genes,
            n_perts=n_perts,
            transform=transform,
            context=context,
            evaluation=evaluation,
            lr=lr,
            wd=wd,
            lr_scheduler=lr_scheduler,
            embedding_width=embedding_width,
        )
        self.save_hyperparameters()
        self.automatic_optimization = False

        if decoder_distribution in [dist.__name__ for dist in self.COUNT_DISTRIBUTIONS]:
            if library_size is None:
                raise ValueError(f"library_size must be set to 'learned' or 'observed' "
                                 f"if decoder_distribution is in {self.COUNT_DISTRIBUTIONS}")
            elif library_size == 'learned':
                log.warning("library_size is set to 'learned' but in the current implementation "
                            "it is not treated as a random variable")

        self.n_genes = n_genes
        self.n_perts = n_perts
        self.n_latent = n_latent
        self.decoder_distribution = decoder_distribution
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
        self.n_warmup_epochs = n_warmup_epochs
        self.softplus_output = softplus_output
        self.adv_loss_drugs = nn.CrossEntropyLoss()
        self.adv_loss_fn = nn.CrossEntropyLoss()
        self.use_adversary = use_adversary
        self.use_covariates = use_covariates
        self.use_legacy_negative_binomial = use_legacy_negative_binomial
        self.count_based_input_expression = count_based_input_expression
        self.dispersion_by_gene_cell = dispersion_by_gene_cell
        self.library_size = library_size

        self.encoder = VariationalEncoder(
            input_dim=self.n_input_features,
            hidden_dim=self.hidden_dim,
            latent_dim=self.n_latent,
            n_layers=self.n_layers_encoder,
            dropout=self.dropout
        )

        if self.use_covariates:
            # Check for continuous covariates (None uniques) which are not supported
            continuous_covs = [k for k, v in context['covariate_uniques'].items() if v is None]
            if continuous_covs:
                raise ValueError(
                    f"CPA does not support continuous covariates. "
                    f"Found continuous covariates: {continuous_covs}"
                )
            self.covars_encoder = {
                covar: uniques for covar, uniques in context['covariate_uniques'].items()
                if len(uniques) > 1
            }
        else:
            self.covars_encoder = {}

        if decoder_distribution == "Gaussian":
            self.decoder = DeepGaussian(self.n_latent, self.hidden_dim, self.n_genes, self.n_layers_encoder,
                                        self.dropout)
        elif decoder_distribution == "IsotropicGaussian":
            self.decoder = DeepIsotropicGaussian(self.n_latent, self.hidden_dim, self.n_genes, self.n_layers_encoder,
                                                 self.dropout, self.softplus_output)
        elif decoder_distribution == "Poisson":
            self.decoder = DeepPoisson(self.n_latent, self.hidden_dim, self.n_genes, self.n_layers_encoder,
                                       self.dropout, self.library_size)
        elif decoder_distribution == "PoissonGamma":
            self.decoder = DeepPoissonGamma(self.n_latent, self.hidden_dim, self.n_genes, self.n_layers_encoder,
                                            self.dropout, self.library_size, self.use_legacy_negative_binomial)
        elif decoder_distribution == "ZeroInflatedPoissonGamma":
            self.decoder = ZeroInflatedPoissonGamma(
                self.n_latent, self.hidden_dim, self.n_genes, self.n_layers_encoder, self.dropout, self.library_size,
                self.use_legacy_negative_binomial, self.dispersion_by_gene_cell)
        else:
            raise ValueError(
                "decoder_distribution must be one of 'Gaussian', 'IsotropicGaussian', 'Poisson', 'PoissonGamma', 'ZeroInflatedPoissonGamma'"
            )

        self.pert_network = MLP(
            input_dim=self.n_perts,
            hidden_dim=self.hidden_dim,
            output_dim=self.n_latent,
            n_layers=self.n_layers_pert_emb,
            dropout=self.dropout,
            elementwise_affine=elementwise_affine
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
                        elementwise_affine=elementwise_affine
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
            self.covars_adversary_classifiers = nn.ModuleDict(self.covars_adversary_classifiers)

        else:
            self.covars_embeddings = None
            self.covars_adversary_classifiers = None

        self.generative_modules = nn.ModuleList([self.encoder, self.decoder, self.pert_network, self.covars_embeddings])
        self.adversary_modules = nn.ModuleList(
            [self.perturbation_adversary_classifier, self.covars_adversary_classifiers])

    @property
    def start_adv_training(self):
        if self.n_warmup_epochs:
            return self.current_epoch > self.n_warmup_epochs
        else:
            return True

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
            library = None
        elif type(self.decoder) in self.COUNT_DISTRIBUTIONS:
            library = x.sum(axis=-1).reshape(-1, 1)
            if self.embedding_width is None and self.count_based_input_expression:
                # todo, check if normalization is necessary (check if scvi did that)
                # control_input in theory can be embeddings
                x_ = torch.log1p(x)
        else:
            x_ = x
            library = None

        if self.variational:
            qz, z_basal = self.encoder(x_)
        else:
            qz, z_basal = None, self.encoder(x_)

        if self.variational and n_samples > 1:
            sampled_z = qz.sample((n_samples,))
            z_basal = self.encoder.z_transformation(sampled_z)
            if type(self.decoder) in self.COUNT_DISTRIBUTIONS:
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
                covars_input = (covars_dict[covar])
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
            library: torch.Tensor | None = None,
    ):
        if type(self.decoder) in self.COUNT_DISTRIBUTIONS:
            predictions = self.decoder(z, library)
        else:
            predictions = self.decoder(z)
        return {
            'predictions': predictions,
            'pz': Normal(torch.zeros_like(z), torch.ones_like(z)),
        }

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
        recon_loss = self.decoder.reconstruction_loss(generative_outputs['predictions'], x)

        if self.variational:
            qz = inference_outputs["qz"]
            pz = generative_outputs['pz']  # just a standard gaussian

            kl_divergence_z = kl(qz, pz).sum(dim=1)
            kl_loss = kl_divergence_z.mean()
        else:
            kl_loss = torch.zeros_like(recon_loss)

        if self.use_adversary:
            adv_loss = self.adversarial_loss(perturbations, covariates, inference_outputs['z_basal'], self.training)
        else:
            adv_loss = {
                'adv_loss': torch.zeros_like(recon_loss, requires_grad=True),
                'penalty_adv': torch.zeros_like(recon_loss, requires_grad=True),
                'penalty_covars': torch.zeros_like(recon_loss, requires_grad=True),
                'penalty_perts': torch.zeros_like(recon_loss, requires_grad=True),
                'acc_perts': torch.zeros_like(recon_loss, requires_grad=True),
                'covariate_classifier_loss': torch.zeros_like(recon_loss, requires_grad=True),
                'perturbation_classifier_loss': torch.zeros_like(recon_loss, requires_grad=True),
            }

        total_loss = recon_loss + self.kl_weight * kl_loss - self.adv_weight * adv_loss["adv_loss"]

        return {
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "adv_loss": adv_loss["adv_loss"],
            "covariate_classifier_loss": adv_loss["covariate_classifier_loss"],
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
                covars_pred_logits[covar] = self.covars_adversary_classifiers[covar](z_basal)
            else:
                covars_pred_logits[covar] = None

        adv_results = {}

        # Classification losses for different covariates
        for covar, covars in self.covars_encoder.items():
            adv_results[f'adv_{covar}'] = self.adv_loss_fn(  # ∞ we've removed mixup for now
                covars_pred_logits[covar],
                covariates[covar],
            ) if covars_pred_logits[covar] is not None else torch.as_tensor(0.0).to(self.device)

            adv_results[f'acc_{covar}'] = accuracy(
                covars_pred_logits[covar].argmax(1), covariates[covar].argmax(1), task='multiclass',
                num_classes=len(covars)) \
                if covars_pred_logits[covar] is not None else torch.as_tensor(0.0).to(self.device)

        if len(self.covars_encoder) > 0:
            adv_results['covariate_classifier_loss'] = sum(
                [adv_results[f'adv_{key}'] for key in self.covars_encoder.keys()])
        else:
            adv_results['covariate_classifier_loss'] = torch.as_tensor(0.0).to(self.device)

        # TODO Not using mixups for now.
        perturbations_pred_logits = self.perturbation_adversary_classifier(z_basal)

        adv_results['perturbation_classifier_loss'] = self.adv_loss_drugs(perturbations_pred_logits, perturbations)

        adv_results['acc_perts'] = accuracy(
            perturbations_pred_logits.argmax(1), perturbations.argmax(1), average='macro',
            num_classes=self.n_perts, task='multiclass',
        )

        adv_results['adv_loss'] = adv_results['covariate_classifier_loss'] + adv_results['perturbation_classifier_loss']

        if compute_penalty:
            # Penalty losses
            for covar in self.covars_encoder.keys():
                adv_results[f'penalty_{covar}'] = (
                    torch.autograd.grad(
                        covars_pred_logits[covar].sum(),
                        z_basal,
                        create_graph=True,
                        retain_graph=True,
                        only_inputs=True,
                    )[0].pow(2).mean()
                ) if covars_pred_logits[covar] is not None else torch.as_tensor(0.0).to(self.device)

            if len(self.covars_encoder) > 0:
                adv_results['penalty_covars'] = sum(
                    [adv_results[f'penalty_{covar}'] for covar in self.covars_encoder.keys()])
            else:
                adv_results['penalty_covars'] = torch.as_tensor(0.0).to(self.device)

            adv_results['penalty_perts'] = (
                torch.autograd.grad(
                    perturbations_pred_logits.sum(),
                    z_basal,
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True,
                )[0].pow(2).mean()
            )

            adv_results['penalty_adv'] = adv_results['penalty_perts'] + adv_results['penalty_covars']
        else:
            for covar in self.covars_encoder.keys():
                adv_results[f'penalty_{covar}'] = torch.as_tensor(0.0).to(self.device)

            adv_results['penalty_covars'] = torch.as_tensor(0.0).to(self.device)
            adv_results['penalty_perts'] = torch.as_tensor(0.0).to(self.device)
            adv_results['penalty_adv'] = torch.as_tensor(0.0).to(self.device)

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

        generative_outputs = self.generative(inference_outputs['z'], inference_outputs['library'])
        return inference_outputs, generative_outputs

    def training_step(self, batch: Batch, batch_idx: int):
        optimizer_generative, optimizer_adversary = self.optimizers()

        inference_outputs, generative_outputs = self.forward(batch)
        losses = self.loss(batch.gene_expression, batch.perturbations, batch.covariates, inference_outputs,
                           generative_outputs, batch_idx)

        total_loss = losses["total_loss"]
        adv_loss = self.adv_weight * losses["adv_loss"] + self.penalty_weight * losses["penalty_adv"]

        if self.start_adv_training:
            if batch_idx % self.adv_steps == 0:
                self.toggle_optimizer(optimizer_generative)
                optimizer_generative.zero_grad()
                self.manual_backward(total_loss)
                optimizer_generative.step()
                self.untoggle_optimizer(optimizer_generative)

            else:
                self.toggle_optimizer(optimizer_adversary)
                optimizer_adversary.zero_grad()
                self.manual_backward(adv_loss)
                optimizer_adversary.step()
                self.untoggle_optimizer(optimizer_adversary)

        else:
            gen_loss = losses["recon_loss"] + self.kl_weight * losses["kl_loss"]
            self.toggle_optimizer(optimizer_generative)
            optimizer_generative.zero_grad()
            self.manual_backward(gen_loss)
            optimizer_generative.step()
            self.untoggle_optimizer(optimizer_generative)

        if self.training:
            for key, value in losses.items():
                self.log("train_" + key, value, prog_bar=True, logger=True, batch_size=len(batch))

    def validation_step(self, batch: Batch, batch_idx: int):
        inference_outputs, generative_outputs = self.forward(batch)
        losses = self.loss(batch.gene_expression, batch.perturbations, batch.covariates, inference_outputs,
                           generative_outputs, batch_idx)
        total_loss = losses["total_loss"]
        self.log("val_loss", total_loss, prog_bar=True, logger=True, batch_size=len(batch))
        return total_loss

    def predict(self, batch: Batch):
        _, generative_outputs = self.forward(batch)
        return generative_outputs["predictions"]

    def configure_optimizers(self):
        optimizer_generative = torch.optim.Adam(self.generative_modules.parameters(), lr=self.lr, weight_decay=self.wd)
        optimizer_adversary = torch.optim.Adam(self.adversary_modules.parameters(), lr=self.lr, weight_decay=self.wd)

        if self.lr_scheduler is not None:
            scheduler_generative = instantiate_with_context(self.lr_scheduler,
                                                            context={'optimizer': optimizer_generative})
            scheduler_adversary = instantiate_with_context(self.lr_scheduler,
                                                           context={'optimizer': optimizer_adversary})
            lr_scheduler_generative = {
                "scheduler": scheduler_generative,
            }
            lr_scheduler_adversary = {
                "scheduler": scheduler_adversary,
            }
            if 'extras' in self.lr_scheduler:
                lr_scheduler_generative.update(OmegaConf.to_object(self.lr_scheduler.extras))
                lr_scheduler_adversary.update(OmegaConf.to_object(self.lr_scheduler.extras))

            return [{"optimizer": optimizer_generative, "lr_scheduler": lr_scheduler_generative},
                    {"optimizer": optimizer_adversary, "lr_scheduler": lr_scheduler_adversary}]
        else:
            return [{"optimizer": optimizer_generative}, {"optimizer": optimizer_adversary}]

