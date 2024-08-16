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
