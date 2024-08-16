## Evaluation wrapper class
import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
from .evaluation import Evaluation
import os

# import subprocess as sp


class Evaluator:
    """A class for benchmarking model predictions on a specific task."""

    @staticmethod
    def list_tasks():
        """List the tasks that the Evaluator class can evaluate models on."""
        return [
            "srivatsan20-transfer",
            "norman19-combo",
            "mcfaline23-transfer",
        ]

    @staticmethod
    def get_task_data(
        task: str,
        data_paths: dict,
        local_data_cache: str = "/perturbench_data",
    ):
        """Pulls the anndata object for a specific task into the local cache."""
        if (task not in Evaluator.list_tasks()) or (task not in data_paths):
            raise ValueError(f"Task {task} is not supported.")

        if not os.path.exists(local_data_cache):
            os.makedirs(local_data_cache)
        # TODO: wrap the curation notebooks as functions and call with this method
        raise NotImplementedError(
            "Data curation functions are not yet implemented. Please run the \
            scripts and notebooks in the notebooks/neurips2024/data_curation/ \
            directory to download and preprocess the data."
        )
        # sp.call(f'cp {data_paths[task]} {local_data_cache}', shell=True)

    @staticmethod
    def get_task_metadata(task: str):
        """Returns the metadata columns for a specific task."""
        metadata_dict = {
            "perturbation": "condition",
        }
        if task in ["srivatsan20-transfer", "norman19-combo"]:
            metadata_dict["covariates"] = ["cell_type"]
        elif task == "mcfaline23-transfer":
            metadata_dict["covariates"] = ["cell_type", "treatment"]
        else:
            raise ValueError(f"Task {task} is not supported.")

        metadata_dict["control_value"] = "control"
        return metadata_dict

    def __init__(
        self,
        task: str,
        local_data_cache: str = "/perturbench_data",
    ):
        """The constructor for the Evaluation class.

        Args:
            task: The task that the model is being evaluated on. Must be one of
              "srivatsan20-transfer", "norman19-combo", "mcfaline23-transfer".
            local_data_cache: The local directory where the task data is stored.
        """
        if task not in Evaluator.list_tasks():
            raise ValueError(f"Task {task} is not supported.")

        # TODO: Change to figshare URLs once datasets are uploaded
        self.data_paths = {
            "srivatsan20-transfer": f"{local_data_cache}/srivatsan20_processed.h5ad",
            "norman19-combo": f"{local_data_cache}/norman19_processed.h5ad",
            "mcfaline23-transfer": f"{local_data_cache}/mcfaline23_gxe_processed.h5ad",
        }
        local_data_path = local_data_cache + "/" + self.data_paths[task].split("/")[-1]
        # if not os.path.exists(local_data_path):
        #     Evaluator.get_task_data(task, self.data_paths, local_data_cache)

        # Load observed anndata object
        ref_adata = sc.read_h5ad(local_data_path)
        task_metadata = Evaluator.get_task_metadata(task)

        self.ref_adata = ref_adata
        self.task_metadata = task_metadata

    def evaluate(
        self,
        model_predictions: dict[str, ad.AnnData],
        return_metrics_dataframe: bool = False,
        print_metrics: bool = False,
    ):
        """Evaluates the model predictions on the task.

        Args:
            model_predictions: A dictionary mapping model names to their predictions
              as anndata objects.
            return_metrics_dataframe: Whether to return the summarized metrics
              as a pandas dataframe.
            print_metrics: Whether to print the summarized metrics to the console.
        """

        ev = Evaluation(
            model_adatas=model_predictions,
            ref_adata=self.ref_adata,
            pert_col=self.task_metadata["perturbation"],
            cov_cols=self.task_metadata["covariates"],
            ctrl=self.task_metadata["control_value"],
        )

        summary_metrics_dict = {}
        for aggr in ["average", "logfc"]:
            ev.aggregate(aggr_method=aggr)

            if aggr == "average":
                metric = "rmse"
            elif aggr == "logfc":
                metric = "cosine"

            ev.evaluate(aggr_method=aggr, metric=metric)
            ev.evaluate_pairwise(aggr_method=aggr, metric=metric)
            ev.evaluate_rank(aggr_method=aggr, metric=metric)

            df = ev.evals[aggr][metric].copy()
            avg = df.groupby("model").mean("metric")
            summary_metrics_dict[metric + "_" + aggr] = avg["metric"]

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
        self.summary_metrics = summary_metrics
        self.ev = ev

        if print_metrics:
            print(summary_metrics)

        if return_metrics_dataframe:
            return summary_metrics
