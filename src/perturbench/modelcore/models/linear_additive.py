import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from perturbench.data.types import Batch
from perturbench.data.transforms.base import Dispatch

from .base import PerturbationModel


class LinearAdditive(PerturbationModel):
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
            inject_covariates: bool = False,
            lr: float | None = None,
            wd: float | None = None,
            lr_scheduler: DictConfig | None = None,
            softplus_output: bool = True,
            **kwargs,
    ) -> None:
        """
        The constructor for the LinearAdditive class.

        Args:
            n_genes (int): Number of genes in the dataset
            n_perts (int): Number of perturbations in the dataset (not including controls)
            lr (float): Learning rate
            wd (float): Weight decay
            inject_covariates: Whether to condition the linear layer on
                covariates
            softplus_output: Whether to apply a softplus activation to the
                output of the decoder to enforce non-negativity
        """
        super(LinearAdditive, self).__init__(
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
        self.softplus_output = softplus_output

        self.inject_covariates = inject_covariates
        if inject_covariates:
            self.fc_pert = nn.Linear(self.n_input_perturbation_features + self.n_total_covariates, self.n_genes)
        else:
            self.fc_pert = nn.Linear(self.n_input_perturbation_features, self.n_genes)

    def forward(
            self,
            control_expression: torch.Tensor,
            perturbation: torch.Tensor,
            covariates: dict,
    ):
        if self.inject_covariates:
            merged_covariates = torch.cat([cov if cov.ndim == 2 else cov.squeeze() for cov in covariates.values()],
                                          dim=1)
            perturbation = torch.cat([perturbation, merged_covariates], dim=1)

        predicted_perturbed_expression = control_expression + self.fc_pert(perturbation)
        if self.softplus_output:
            predicted_perturbed_expression = F.softplus(predicted_perturbed_expression)
        return predicted_perturbed_expression

    def training_step(self, batch: Batch, batch_idx: int):
        predicted_perturbed_expression = self.forward(
            batch.gene_expression.squeeze(), batch.perturbations.squeeze(), batch.covariates
        )
        loss = F.mse_loss(predicted_perturbed_expression, batch.gene_expression.squeeze())
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            logger=True,
            batch_size=len(batch),
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch: Batch, batch_idx: int):
        predicted_perturbed_expression = self.forward(
            batch.gene_expression.squeeze(), batch.perturbations.squeeze(), batch.covariates
        )
        val_loss = F.mse_loss(predicted_perturbed_expression, batch.gene_expression.squeeze())
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
        control_expression = batch.gene_expression.squeeze().to(self.device)
        perturbation = batch.perturbations.squeeze().to(self.device)
        covariates = {
            k: v.to(self.device) for k, v in batch.covariates.items()
        }
        predicted_perturbed_expression = self.forward(
            control_expression,
            perturbation,
            covariates,
        )
        return predicted_perturbed_expression
