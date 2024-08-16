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

from typing import Dict, Any
import lightning as L
import torch
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import anndata as ad
from omegaconf import DictConfig
import os
import gc
from perturbench.data.types import Batch
from perturbench.analysis.benchmarks.evaluation import Evaluation, merge_evals


class PerturbationModel(L.LightningModule, ABC):
    """A base model class for perturbation prediction models.

    Attributes:
        training_record: A record of the transforms and training context used
            to train the model
        evaluation_config: A configuration object containing the evaluation
            parameters
        summary_metrics: A DataFrame containing the summary metrics of the
            model evaluation
        training_record: A record of the transforms and training context used
            to train the model
        evaluation_config: A configuration object containing the evaluation
            parameters
        summary_metrics: A DataFrame containing the summary metrics of the
            model evaluation
    """

    training_record: dict = {
        "transform": None,
        "train_context": None,
        "n_total_covs": None,
    }
    evaluation_config: DictConfig | None = None
    summary_metrics: pd.DataFrame | None = None
    prediction_output_path: str | None = None

    def __init__(
        self,
        datamodule: L.LightningDataModule | None = None,
        lr: float | None = None,
        wd: float | None = None,
        lr_scheduler_freq: float | None = None,
        lr_scheduler_interval: str | None = None,
        lr_scheduler_patience: float | None = None,
        lr_scheduler_factor: float | None = None,
        lr_monitor_key: str | None = None,
    ):

        super(PerturbationModel, self).__init__()

        self.lr = 1e-3 if lr is None else lr
        self.wd = 1e-5 if wd is None else wd
        self.lr_scheduler_freq = 1 if lr_scheduler_freq is None else lr_scheduler_freq
        self.lr_scheduler_interval = (
            "epoch" if lr_scheduler_interval is None else lr_scheduler_interval
        )
        self.lr_scheduler_patience = (
            15 if lr_scheduler_patience is None else lr_scheduler_patience
        )
        self.lr_scheduler_factor = (
            0.2 if lr_scheduler_factor is None else lr_scheduler_factor
        )
        self.lr_monitor_key = "val_loss" if lr_monitor_key is None else lr_monitor_key

        if datamodule is not None:
            self.training_record["transform"] = datamodule.train_dataset.transform
            self.training_record["train_context"] = datamodule.train_context
            self.evaluation_config = datamodule.evaluation

            self.training_record["train_context"] = datamodule.train_context
            self.evaluation_config = datamodule.evaluation

            self.n_genes = datamodule.num_genes
            self.n_perts = datamodule.num_perturbations

            embedding_width = datamodule.embedding_width
            if embedding_width is not None:
                self.n_input_features = embedding_width
            else:
                self.n_input_features = self.n_genes

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.wd
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=self.lr_scheduler_factor,
            patience=self.lr_scheduler_patience,
        )
        lr_scheduler = {
            "scheduler": scheduler,
            "monitor": self.lr_monitor_key,
            "frequency": self.lr_scheduler_freq,
            "interval": self.lr_scheduler_interval,
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def unpack_batch(self, batch: Batch):
        observed_perturbed_expression = batch.gene_expression.squeeze()
        control_expression = (
            batch.controls.squeeze() if batch.controls is not None else None
        )
        perturbation = batch.perturbations.squeeze()
        covariates = batch.covariates if batch.covariates is not None else None
        embeddings = (
            batch.embeddings.squeeze() if batch.embeddings is not None else None
        )
        return (
            observed_perturbed_expression,
            control_expression,
            perturbation,
            covariates,
            embeddings,
        )

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint["training_record"] = self.training_record

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.training_record = checkpoint["training_record"]

    def predict_step(
        self,
        data_tuple: tuple[Batch, pd.DataFrame],
        batch_idx: int,
    ) -> ad.AnnData | None:
        """Given a batch of data, predict the counterfactual perturbed expression
           as an AnnData object.

        Args:
            data_tuple: A tuple containing the counterfactual batch and a
                pandas DataFrame containing the counterfactual cell level
                metadata.

        Returns:
            ad.AnnData: The predicted counterfactual perturbed expression.
        """
        counterfactual_batch, counterfactual_obs = data_tuple
        predicted_expression = (
            self.predict(counterfactual_batch).squeeze().cpu().detach().numpy()
        )
        predicted_adata = ad.AnnData(
            X=predicted_expression,
            obs=counterfactual_obs,
        )
        predicted_adata.var_names = counterfactual_batch.gene_names
        if self.prediction_output_path is not None:
            predicted_adata.write_h5ad(
                self.prediction_output_path + f"/prediction_chunk_{batch_idx}.h5ad"
            )
        else:
            return predicted_adata

    def on_test_start(self) -> None:
        super().on_test_start()
        self.evaluation_list = []
        self.unique_aggregations = set()
        for eval_dict in self.evaluation_config.evaluation_pipelines:
            self.unique_aggregations.add(eval_dict["aggregation"])

    def test_step(
        self,
        data_tuple: tuple[Batch, pd.DataFrame, ad.AnnData],
        batch_idx: int,
    ):
        counterfactual_batch, counterfactual_obs, reference_adata = data_tuple
        train_context = self.training_record["train_context"]
        model_name = str(self.__class__).split(".")[-1].replace("'>", "")

        ## Build predicted anndata object
        predicted_adata = self.predict_step(
            (counterfactual_batch, counterfactual_obs),
            batch_idx,
        )

        ## Evaluate the predicted anndata object
        control_adata = reference_adata[
            reference_adata.obs[train_context["perturbation_key"]]
            == train_context["perturbation_control_value"]
        ]
        assert control_adata.shape[0] > 0

        predicted_adata = predicted_adata[
            predicted_adata.obs[train_context["perturbation_key"]]
            != train_context["perturbation_control_value"]
        ]
        predicted_adata = ad.concat([predicted_adata, control_adata])
        predicted_adata.obs_names_make_unique()

        ev = Evaluation(
            model_adatas=[predicted_adata],
            model_names=[model_name],
            ref_adata=reference_adata,
            pert_col=train_context["perturbation_key"],
            cov_cols=train_context["covariate_keys"],
            ctrl=train_context["perturbation_control_value"],
        )
        for aggr in self.unique_aggregations:
            ev.aggregate(aggr_method=aggr)
        del ev.adatas
        ev.adatas = None
        self.evaluation_list.append(ev)

        # Cleanup
        gc.collect()

    def on_test_end(self) -> None:
        super().on_test_end()
        ev = merge_evals(self.evaluation_list)

        summary_metrics_dict = {}
        for eval_dict in self.evaluation_config.evaluation_pipelines:
            aggr = eval_dict["aggregation"]
            metric = eval_dict["metric"]
            ev.evaluate(aggr_method=aggr, metric=metric)

            df = ev.evals[aggr][metric].copy()
            avg = df.groupby("model").mean("metric")
            summary_metrics_dict[metric + "_" + aggr] = avg["metric"]

            if eval_dict.get("rank"):
                ev.evaluate_pairwise(aggr_method=aggr, metric=metric)
                ev.evaluate_rank(aggr_method=aggr, metric=metric)

                rank_df = ev.rank_evals[aggr][metric].copy()
                avg_rank = rank_df.groupby("model").mean("rank")
                summary_metrics_dict[metric + "_rank_" + aggr] = avg_rank["rank"]

        summary_metrics = pd.DataFrame(summary_metrics_dict).T.applymap(
            lambda x: float(
                np.format_float_positional(
                    x, precision=4, unique=False, fractional=False, trim="k"
                )
            ),
        )
        if self.evaluation_config.print_summary:
            print(summary_metrics)

        if not os.path.exists(self.evaluation_config.save_dir):
            os.makedirs(self.evaluation_config.save_dir)
        ev.save(self.evaluation_config.save_dir)

        summary_metrics.to_csv(
            self.evaluation_config.save_dir + "/summary.csv",
            index_label="metric",
        )
        ## TODO: Figure out a better way to return summary metrics, seems to be an issue with lightning test step
        self.summary_metrics = summary_metrics

        # Logging to metrics (better for comparison on mlflow)
        # Convert DataFrame to a dictionary with row-wise (metric_name: value) pairs
        if self.logger is not None:
            for _, row in summary_metrics.T.iterrows():
                metrics_dict = row.to_dict()

                for key, value in metrics_dict.items():
                    # Ensure value is a scalar for logging
                    value = float(value) if pd.notnull(value) else None
                    if value is not None:
                        ## TODO: Figure out what happens if there are multiple loggers
                        self.logger.log_metrics({key: value})
            if isinstance(self.logger, L.pytorch.loggers.MLFlowLogger):
                self.logger.experiment.log_artifact(
                    local_path=self.evaluation_config.save_dir,
                    run_id=self.logger.run_id,
                )

        # Cleanup
        gc.collect()

    @abstractmethod
    def predict(self, counterfactual_batch: Batch) -> torch.Tensor:
        """Given a counterfactual_batch of data, predicted the counterfactual perturbed expression.

        Example implementation:
        ```
        def predict(self, counterfactual_batch):
            control_expression = counterfactual_batch.gene_expression.squeeze()
            perturbation = counterfactual_batch.perturbations.squeeze()
            covariates = counterfactual_batch.covariates.squeeze()

            predicted_perturbed_expression = self.forward(
                control_expression,
                perturbation,
                covariates,
            )
            return predicted_perturbed_expression
        ```
        """
        pass
