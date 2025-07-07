"""
Copyright (C) 2024  <anonymized authors of NeurIPS submission #1306>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import torch
import torch.nn.functional as F
import lightning as L
import numpy as np
import json

from ..nn.mlp import MLP, MaskNet
from .base import PerturbationModel
from perturbench.data.types import Batch


class LatentAdditiveEmbedding(PerturbationModel):
    """
    A latent additive model for predicting perturbation effects
    """

    def __init__(
        self,
        n_genes: int,
        n_perts: int,
        n_layers: int = 2,
        encoder_width: int = 128,
        pert_encoder_width: int = 128,
        latent_dim: int = 32,
        embed_cell: bool = True,
        embed_gene: bool = True,
        debug_zero_embeddings: bool = False,
        lr: float | None = None,
        wd: float | None = None,
        lr_scheduler_freq: int | None = None,
        lr_scheduler_interval: str | None = None,
        lr_scheduler_patience: int | None = None,
        lr_scheduler_factor: float | None = None,
        dropout: float | None = None,
        softplus_output: bool = True,
        sparse_additive_mechanism: bool = False,
        inject_covariates_encoder: bool = False,
        inject_covariates_decoder: bool = False,
        n_total_covariates: int | None = None,
        datamodule: L.LightningDataModule | None = None,
    ) -> None:
        """
        The constructor for the LatentAdditive class.

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
            inject_covariates_encoder: Whether to condition the encoder on
                covariates
            inject_covariates_decoder: Whether to condition the decoder on
                covariates
            datamodule: The datamodule used to train the model
        """
        super(LatentAdditiveEmbedding, self).__init__(
            datamodule=datamodule,
            lr=lr,
            wd=wd,
            lr_scheduler_freq=lr_scheduler_freq,
            lr_scheduler_interval=lr_scheduler_interval,
            lr_scheduler_patience=lr_scheduler_patience,
            lr_scheduler_factor=lr_scheduler_factor,
        )

        self.save_hyperparameters(ignore=["datamodule"])
        if n_genes is not None:
            self.n_genes = n_genes
        if n_perts is not None:
            self.n_perts = n_perts
        self.embed_gene = embed_gene
        self.debug_zero_embeddings = debug_zero_embeddings


        if inject_covariates_encoder or inject_covariates_decoder:
            if datamodule is None or datamodule.train_context is None:
                raise ValueError(
                    "If inject_covariates is True, datamodule must be provided"
                )
            n_total_covariates = np.sum(
                [
                    len(unique_covs)
                    for unique_covs in datamodule.train_context[
                        "covariate_uniques"
                    ].values()
                ]
            )

        if embed_gene:
            if datamodule is None or datamodule.train_context is None:
                raise ValueError(
                    "If embed_gene is True, datamodule must be provided"
                )
            pert_names = {i: name for i, name in enumerate(datamodule.train_context["perturbation_uniques"])}
            self.gene_embedding = torch.nn.Embedding(60697, 512, padding_idx=60694)
            self.gene_embedding.load_state_dict(torch.load("/cluster/scratch/fluebeck/perturbench_data/pretrained_models/scGPT_human/gene_embedding.pth"))
            
            # Freeze the gene embedding parameters to make them non-trainable
            self.gene_embedding.requires_grad_(False)
            print("Gene embedding parameters frozen (non-trainable)")
            
            # Add a learnable scaling parameter for the embeddings
            self.embedding_scale = torch.nn.Parameter(torch.tensor(10.0))
            print("Added learnable embedding scale parameter")

            with open("/cluster/scratch/fluebeck/perturbench_data/pretrained_models/scGPT_human/vocab.json", "r") as f:
                vocab = json.load(f)
            gene_vocab = vocab

            pert_name_map = {
                # name in perturbench : name in scgpt
                "C3orf72": "FOXL2NB",
                "ELMSAN1": "MIDEAS",
                "C19orf26": "CBARP",
                "KIAA1804": "MAP3K21",
                "NUP50-AS1": "NUP50-DT",
                "LRRC75A-AS1": "SNHG29",
                "TMEM173": "STING1",
                "ATP5MD": "ATP5MK",
            }

            self.pert_id_to_scgpt_id = {}
            for i, pert_name in pert_names.items():
                scgpt_pert_name = pert_name_map.get(pert_name, pert_name)
                if scgpt_pert_name not in gene_vocab:
                    print(f"Perturbation {pert_name} not in vocab.")
                else:
                    self.pert_id_to_scgpt_id[i] = gene_vocab[scgpt_pert_name]
            max_pert_id = max(self.pert_id_to_scgpt_id.keys())
            mapping_tensor = torch.full((max_pert_id + 1,), fill_value=0, dtype=torch.long)
            for pert_id, scgpt_id in self.pert_id_to_scgpt_id.items():
                mapping_tensor[pert_id] = scgpt_id
            self.register_buffer('pert_to_scgpt_tensor', mapping_tensor)
            print(f"Perturbation to scGPT ID mapping: {self.pert_id_to_scgpt_id}")
            

        encoder_input_dim = (
            self.n_input_features + n_total_covariates
            if inject_covariates_encoder
            else self.n_input_features
        )
        gene_encoder_input_dim = (
            512
            if embed_gene
            else self.n_perts
        )
        decoder_input_dim = (
            latent_dim + (n_total_covariates or 0) if inject_covariates_decoder else latent_dim
        )
        

        self.gene_encoder = MLP(
            encoder_input_dim, encoder_width, latent_dim, n_layers, dropout
        )
        self.decoder = MLP(
            decoder_input_dim, encoder_width, self.n_genes, n_layers, dropout
        )
        self.pert_encoder = MLP(
            gene_encoder_input_dim, pert_encoder_width, latent_dim, n_layers, dropout
        )

        if sparse_additive_mechanism:
            self.mask_encoder = MaskNet(
                self.n_perts, encoder_width, latent_dim, n_layers
            )

        self.dropout = dropout
        self.softplus_output = softplus_output
        self.sparse_additive_mechanism = sparse_additive_mechanism
        self.inject_covariates_encoder = inject_covariates_encoder
        self.inject_covariates_decoder = inject_covariates_decoder

    def get_perturbation_embedding(self, perturbation: torch.Tensor) -> torch.Tensor:
        """
        Given a (batch_size, n_perts) multi-hot perturbation tensor, returns the summed gene embeddings for each cell.
        """

        if perturbation.dim() == 1:
            perturbation = perturbation.unsqueeze(0)
        batch_size, n_perts = perturbation.shape
        device = perturbation.device
        # Get indices of nonzero perturbations for each cell (allows multiple perturbations per cell)
        pert_indices = (perturbation > 0).nonzero(as_tuple=False)  # (num_nonzero, 2): [cell_idx, pert_idx]
        cell_ids = pert_indices[:, 0]
        pert_ids = pert_indices[:, 1]
        
        
        # Map pert_idx to scGPT vocab id
        scgpt_ids = self.pert_to_scgpt_tensor[pert_ids].to(device)
                
        # Get embeddings for all scGPT ids
        all_embeddings = self.gene_embedding(scgpt_ids)  # (num_nonzero, emb_dim)
        
        # REPLACE WITH RANDOM VECTORS FOR TESTING
        if self.debug_zero_embeddings:
            all_embeddings = torch.zeros_like(all_embeddings)
        
        
        # Check if embeddings are all zeros or very similar
        if all_embeddings.abs().max() < 1e-6:
            print("WARNING: All embeddings are essentially zero!")
        elif all_embeddings.std() < 1e-3:
            print("WARNING: Embeddings have very low variance - they might not be meaningful!")
        
        # Sum embeddings for each cell
        emb_dim = all_embeddings.shape[1]
        perturbation_emb = torch.zeros((batch_size, emb_dim), device=device)
        perturbation_emb.index_add_(0, cell_ids, all_embeddings)

        # Optionally: warn if any cell has more than one perturbation
        per_cell_counts = torch.bincount(cell_ids, minlength=batch_size)
        if (per_cell_counts > 1).any():
            raise ValueError("Multiple perturbations per cell are not yet supported")
        
        return perturbation_emb

    def forward(
        self,
        control_input: torch.Tensor,
        perturbation: torch.Tensor,
        covariates: dict[str, torch.Tensor],
    ):
        if self.inject_covariates_encoder or self.inject_covariates_decoder:
            merged_covariates = torch.cat(
                [cov.squeeze() for cov in covariates.values()], dim=1
            )

        # Store original perturbation for sparse additive mechanism
        original_perturbation = perturbation
        
        
        if self.embed_gene:
            perturbation = self.get_perturbation_embedding(perturbation)

        if self.inject_covariates_encoder:
            control_input = torch.cat([control_input, merged_covariates], dim=1)

        latent_control = self.gene_encoder(control_input)
        latent_perturbation = self.pert_encoder(perturbation)

        if self.sparse_additive_mechanism:
            # Use original perturbation tensor for mask encoder
            mask = self.mask_encoder(original_perturbation)
            latent_perturbation = mask * latent_perturbation

        latent_perturbed = latent_control + latent_perturbation

        if self.inject_covariates_decoder:
            latent_perturbed = torch.cat([latent_perturbed, merged_covariates], dim=1)
        predicted_perturbed_expression = self.decoder(latent_perturbed)

        if self.softplus_output:
            predicted_perturbed_expression = F.softplus(predicted_perturbed_expression)
        return predicted_perturbed_expression

    def training_step(self, batch: Batch, batch_idx: int):
        (
            observed_perturbed_expression,
            control_expression,
            perturbation,
            covariates,
            embeddings,
        ) = self.unpack_batch(batch)

        if embeddings is not None:
            control_input = embeddings
        else:
            control_input = control_expression

        predicted_perturbed_expression = self.forward(
            control_input, perturbation, covariates
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
            embeddings,
        ) = self.unpack_batch(batch)

        if embeddings is not None:
            control_input = embeddings
        else:
            control_input = control_expression

        predicted_perturbed_expression = self.forward(
            control_input, perturbation, covariates
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
        if batch.embeddings is not None:
            control_input = batch.embeddings.squeeze()
        else:
            control_input = batch.gene_expression.squeeze()

        perturbation = batch.perturbations.squeeze().to(self.device)
        covariates = {k: v.to(self.device) for k, v in batch.covariates.items()}

        predicted_perturbed_expression = self.forward(
            control_input,
            perturbation,
            covariates,
        )
        return predicted_perturbed_expression
