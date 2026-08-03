import lightning as L
import torch
from torch import optim
from perturbench.data.types import Batch
from .base import PerturbationModel


class Average(PerturbationModel):
    """
    A perturbation prediction baseline model that returns the average expression of each perturbation in the training data.
    """

    def __init__(
        self,
        n_genes: int,
        n_perts: int,
        datamodule: L.LightningDataModule | None = None,
    ) -> None:
        """
        The constructor for the Average class.

        Args:
            n_genes (int): Number of genes in the dataset
            n_perts (int): Number of perturbations in the dataset (not including controls)
        """
        super(Average, self).__init__(datamodule)
        self.save_hyperparameters(ignore=["datamodule"])

        if n_genes is None:
            n_genes = datamodule.num_genes

        if n_perts is None:
            n_perts = datamodule.num_perturbations

        self.n_genes = n_genes
        self.n_perts = n_perts
        self.average_expression = torch.nn.Parameter(
            torch.zeros(n_perts, n_genes), requires_grad=False
        )
        self.sum_expression = torch.zeros(n_perts, n_genes)
        self.num_cells = torch.zeros(n_perts)
        self.dummy_nn = torch.nn.Linear(1, 1)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters())
        return optimizer

    def backward(self, use_amp, loss, optimizer):
        return

    def on_train_start(self):
        self.sum_expression = self.sum_expression.to(self.device)
        self.num_cells = self.num_cells.to(self.device)

    def training_step(self, batch: Batch, batch_idx: int | list[int]):
        # Unpack the batch
        observed_perturbed_expression = batch.gene_expression.squeeze()
        perturbation = batch.perturbations.squeeze()
        self.sum_expression += torch.matmul(
            perturbation.t(), observed_perturbed_expression
        )
        self.num_cells += perturbation.sum(0)

    def on_train_epoch_end(self):
        average_expression = self.sum_expression.t() / self.num_cells
        self.average_expression = torch.nn.Parameter(
            average_expression.t(), requires_grad=False
        )

        self.sum_expression = torch.zeros(self.n_perts, self.n_genes)
        self.num_cells = torch.zeros(self.n_perts)

    def predict(self, batch: Batch):
        perturbation = batch.perturbations.squeeze()
        perturbation = perturbation.to(self.device)
        predicted_perturbed_expression = torch.matmul(
            perturbation, self.average_expression
        )
        predicted_perturbed_expression = (
            predicted_perturbed_expression.t() / perturbation.sum(1)
        )
        return predicted_perturbed_expression.t()
