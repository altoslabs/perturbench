from typing import Dict, Any
import lightning as L
import torch
import torch.distributions as dist
from abc import ABC, abstractmethod
import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
import anndata as ad
import scanpy as sc
from omegaconf import DictConfig, OmegaConf
import os
import gc
import logging
import subprocess
import sys
import warnings
from glob import glob
from pathlib import Path

from perturbench.modelcore.utils import instantiate_with_context
from .embeddings import EmbeddingModel
from ..nn.decoders import (
    DeepPoisson,
    DeepPoissonGamma,
    ZeroInflatedPoissonGamma
)

from perturbench.data.types import Batch
from perturbench.data.transforms.base import Dispatch
from perturbench.analysis.benchmarks.evaluation import Evaluation

_logger = logging.getLogger(__name__)

_CELL_EVAL_WORKER = str(Path(__file__).resolve().parent.parent / "cell_eval_worker.py")
warnings.filterwarnings("ignore", message=r"The feature ([^\s]+) is currently marked under review")
warnings.filterwarnings("ignore", message=r"In the future ([^\s]+) will be defined as the corresponding NumPy scalar")

original_filterwarnings = warnings.filterwarnings


def _filterwarnings(*args, **kwargs):
    return original_filterwarnings(*args, **{**kwargs, 'append': True})


warnings.filterwarnings = _filterwarnings


def _run_cell_eval_subprocess(
        pred_adata: ad.AnnData,
        real_adata: ad.AnnData,
        control_pert: str,
        pert_col: str,
        outdir: str,
        num_threads: int = 12,
        batch_size: int = 2000,
        profile: str = "vcc",
        timeout: int = 900,
) -> pd.DataFrame | None:
    """Run cell-eval in an isolated subprocess to avoid multiprocessing deadlocks.

    Returns the agg_results DataFrame (indexed by 'statistic') on success, or
    None on timeout / subprocess failure.
    """
    os.makedirs(outdir, exist_ok=True)
    pred_path = os.path.join(outdir, "_tmp_pred.h5ad")
    real_path = os.path.join(outdir, "_tmp_real.h5ad")

    non_serializable_cols = [c for c in pred_adata.obs.columns
                             if pred_adata.obs[c].map(type).eq(frozenset).any()]
    if non_serializable_cols:
        pred_adata = pred_adata.copy()
        pred_adata.obs = pred_adata.obs.drop(columns=non_serializable_cols)
    non_serializable_cols = [c for c in real_adata.obs.columns
                             if real_adata.obs[c].map(type).eq(frozenset).any()]
    if non_serializable_cols:
        real_adata = real_adata.copy()
        real_adata.obs = real_adata.obs.drop(columns=non_serializable_cols)

    pred_adata.write_h5ad(pred_path)
    real_adata.write_h5ad(real_path)

    cmd = [
        sys.executable, _CELL_EVAL_WORKER,
        "--pred", pred_path,
        "--real", real_path,
        "--control-pert", control_pert,
        "--pert-col", pert_col,
        "--num-threads", str(num_threads),
        "--batch-size", str(batch_size),
        "--outdir", outdir,
        "--profile", profile,
    ]

    try:
        _logger.info(f"Launching cell-eval subprocess for {outdir}")
        result = subprocess.run(
            cmd,
            timeout=timeout,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            _logger.warning(
                f"cell-eval subprocess failed (exit {result.returncode}) for {outdir}.\n"
                f"stderr: {result.stderr[-2000:] if result.stderr else '(empty)'}"
            )
            return None
    except subprocess.TimeoutExpired:
        _logger.warning(f"cell-eval subprocess timed out after {timeout}s for {outdir}")
        return None
    finally:
        for p in (pred_path, real_path):
            try:
                os.remove(p)
            except OSError:
                pass

    agg_path = os.path.join(outdir, "agg_results.csv")
    if not os.path.exists(agg_path):
        _logger.warning(f"cell-eval subprocess succeeded but {agg_path} not found")
        return None

    return pd.read_csv(agg_path).set_index("statistic", drop=True)


def _save_and_log_metrics(
        summary_metrics: pd.DataFrame,
        summary_metrics_by_cov_dict: dict,
        ev: Evaluation,
        evaluation_config,
        logger_obj,
        print_summary: bool = False,
):
    """Save evaluation results to disk and log to MLFlow.

    Designed to be called twice: once with core metrics (before cell-eval) and
    once with the full set (after cell-eval succeeds).
    """
    summary_metrics_by_cov = pd.DataFrame(summary_metrics_by_cov_dict).T.map(
        lambda x: float(np.format_float_positional(x, precision=4, unique=False, fractional=False, trim='k')),
    )

    if print_summary:
        print(summary_metrics)
        print(summary_metrics_by_cov)

    save_dir = evaluation_config.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    ev.save(save_dir)

    summary_metrics_by_cov.to_csv(
        save_dir + '/summary_by_cov.csv',
        index_label='metric',
    )

    if logger_obj is not None:
        for _, row in summary_metrics.T.iterrows():
            metrics_dict = row.to_dict()
            for key, value in metrics_dict.items():
                value = float(value) if pd.notnull(value) else None
                if value is not None:
                    logger_obj.log_metrics({key: value})

        if isinstance(logger_obj, L.pytorch.loggers.MLFlowLogger):
            logger_obj.experiment.log_artifact(
                local_path=save_dir,
                run_id=logger_obj.run_id,
            )


class PerturbationModel(L.LightningModule, ABC):
    """A base model class for perturbation prediction models.

    Attributes:
        training_record: A record of the transforms and training context used
            to train the model
        evaluation_config: A configuration object containing the evaluation
            parameters
        summary_metrics: A DataFrame containing the summary metrics of the
            model evaluation
        COUNT_DISTRIBUTIONS: A tuple of string names for count distributions
    """

    training_record: dict = None
    evaluation_config: DictConfig | None = None
    summary_metrics: pd.DataFrame | None = None
    prediction_config: DictConfig | None = None
    control_adata: ad.AnnData | None = None
    n_genes: int | None = None
    n_perts: int | None = None
    n_total_covariates: int | None = None
    embedding_decoder: EmbeddingModel | None = None

    COUNT_DISTRIBUTIONS = (
        DeepPoisson,
        DeepPoissonGamma,
        ZeroInflatedPoissonGamma
    )

    COMPATIBLE_DATASETS = (
        'SingleCellPerturbation',
        'SingleCellPerturbationWithControls',
    )

    def __init__(
            self,
            n_genes: int,
            n_perts: int,
            transform: Dispatch,
            context: dict,
            evaluation: DictConfig,
            embedding_width: int | None = None,  # gene expression embedding
            perturbation_embedding_width: int | None = None,
            lr: float | None = None,
            wd: float | None = None,
            lr_scheduler: DictConfig | None = None,
            count_based_input_expression: bool = False,
            embedding_decoder_path: str | None = None,
            **kwargs,
    ):
        super(PerturbationModel, self).__init__()

        self.lr = 1e-3 if lr is None else lr
        self.wd = 1e-5 if wd is None else wd
        self.lr_scheduler = lr_scheduler

        self.training_record = {
            'transform': transform,
            'train_context': context,
        }
        self.evaluation_config = evaluation

        self.n_genes = n_genes
        self.n_perts = n_perts
        # Continuous covariates (None) contribute 1 to the dimension, categorical contribute len(uniques)
        self.n_total_covariates = np.sum([
            1 if unique_covs is None else len(unique_covs)
            for unique_covs in context['covariate_uniques'].values()
        ])

        # gex embedding
        self.embedding_width = embedding_width
        if embedding_width is not None:
            self.n_input_features = embedding_width
        else:
            self.n_input_features = self.n_genes

        # perturbation embedding
        self.perturbation_embedding_width = perturbation_embedding_width
        if perturbation_embedding_width is not None:
            self.n_input_perturbation_features = perturbation_embedding_width
        else:
            self.n_input_perturbation_features = self.n_perts

        self.count_based_input_expression = count_based_input_expression

        # Load embedding decoder if path is provided
        self.embedding_decoder = None
        if embedding_decoder_path is not None:
            print(f"Loading embedding decoder from {embedding_decoder_path}")
            self.embedding_decoder = EmbeddingModel.load(embedding_decoder_path)
            self.n_output_features = embedding_width
        else:
            self.n_output_features = self.n_genes

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)

        if self.lr_scheduler is not None:
            scheduler = instantiate_with_context(self.lr_scheduler, context={'optimizer': optimizer})
            lr_scheduler = {
                "scheduler": scheduler,
            }
            if 'extras' in self.lr_scheduler:
                lr_scheduler.update(OmegaConf.to_object(self.lr_scheduler.extras))

            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
        else:
            return {"optimizer": optimizer}

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint['training_record'] = self.training_record

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.training_record = checkpoint['training_record']
        # checkpoint["optimizer_states"] = []
        # checkpoint["lr_schedulers"] = []

    def on_train_start(self) -> None:
        # Log train/test split
        if isinstance(self.logger, L.pytorch.loggers.MLFlowLogger):
            output_path = self.evaluation_config.save_dir.split('evaluation')[0]
            split_path = output_path + 'train_test_split.csv'
            if os.path.exists(split_path):
                self.logger.experiment.log_artifact(
                    local_path=split_path,
                    run_id=self.logger.run_id,
                )

            pert_cov_path = output_path + 'perturbation_covariate_split_map.csv'
            if os.path.exists(pert_cov_path):
                self.logger.experiment.log_artifact(
                    local_path=pert_cov_path,
                    run_id=self.logger.run_id,
                )

        if self.embedding_decoder is not None and self.embedding_decoder.decoder is not None:
            # pytorch lightning WILL silently change `self.device`
            self.embedding_decoder.device = self.device
            self.embedding_decoder.decoder.to(self.device)

    def predict_anndata(self, data_tuple: tuple[Batch, pd.DataFrame], ) -> ad.AnnData:
        counterfactual_batch, counterfactual_obs = data_tuple
        predictions = self.predict(counterfactual_batch)

        if isinstance(predictions, dist.Distribution):
            predicted_expression = predictions.mean
            predicted_variance = predictions.variance.float().squeeze().cpu().detach().numpy()
            if predicted_expression.ndim == 3:
                # in cases with a sample_size dimension the shape will be: [sample_size, batch_size, n_genes]
                predicted_expression = predicted_expression.mean(dim=0)
                predicted_variance = predicted_variance.mean(axis=0)  # formula for independent variables
        elif isinstance(predictions, torch.Tensor):
            predicted_expression, predicted_variance = predictions, None
        else:
            raise ValueError(f"Unexpected prediction type: {type(predictions)}")

        predicted_expression = predicted_expression.float().squeeze().cpu().detach().numpy()
        predicted_adata = ad.AnnData(
            X=predicted_expression,
            obs=counterfactual_obs,
        )

        if predicted_variance is not None:
            predicted_adata.layers['variance'] = predicted_variance

        predicted_adata.var_names = counterfactual_batch.gene_names

        return predicted_adata

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
        predicted_adata = self.predict_anndata(data_tuple)

        if self.prediction_config is None:
            raise ValueError("Prediction config is not set, cannot run differential expression")

        if self.prediction_config.output_path is not None:
            predicted_adata.write_h5ad(
                self.prediction_config.output_path + f"/prediction_rank{self.global_rank}_chunk_{batch_idx}.h5ad"
            )

        del predicted_adata
        gc.collect()

    def on_test_start(self) -> None:
        super().on_test_start()
        self.saved_prediction_files = []
        self.saved_reference_files = []
        self.unique_aggregations = set()
        for _, eval_dict in enumerate(self.evaluation_config.evaluation_pipelines):
            aggr_name = eval_dict['aggregation']
            self.unique_aggregations.add(aggr_name)

        # Create temporary directory for saving predictions
        # Broadcast rank 0's directory to all ranks so predictions land in one place
        # even if each rank has a different Hydra output directory.
        self.predicted_adata_dir = os.path.join(self.evaluation_config.save_dir, 'predicted_anndata')
        if self.trainer.world_size > 1:
            import torch.distributed as tdist
            dir_list = [self.predicted_adata_dir]
            tdist.broadcast_object_list(dir_list, src=0)
            self.predicted_adata_dir = dir_list[0]
        os.makedirs(self.predicted_adata_dir, exist_ok=True)

        if self.embedding_decoder is not None and self.embedding_decoder.decoder is not None:
            self.embedding_decoder.device = self.device
            self.embedding_decoder.decoder.to(self.device)

    def test_step(
            self,
            data_tuple: tuple[Batch, pd.DataFrame],
            batch_idx: int,
    ):
        # counterfactual_batch are all control cells and are contained in reference_adata
        # reference_adata first have some perturbed cells then control cells
        # order of the control cells between the two are not the same
        counterfactual_batch, counterfactual_obs = data_tuple

        ## Build predicted anndata object
        predicted_adata = self.predict_anndata(
            (counterfactual_batch, counterfactual_obs),
        )

        ## Save predicted anndata to disk (include global_rank for multi-GPU/node support)
        prediction_file = os.path.join(self.predicted_adata_dir,
                                       f'predicted_rank{self.global_rank}_batch{batch_idx}.h5ad')
        predicted_adata.write_h5ad(prediction_file)
        self.saved_prediction_files.append(prediction_file)

        # Cleanup
        del predicted_adata
        gc.collect()

    def on_test_end(self) -> None:
        super().on_test_end()

        # Synchronize all ranks before loading files (ensures all writes are complete)
        if self.trainer.world_size > 1:
            self.trainer.strategy.barrier()

        # Only rank 0 performs merging and evaluation
        if self.global_rank != 0:
            return

        train_context = self.training_record['train_context']  ## Training context

        if not os.path.exists(self.evaluation_config.save_dir):
            os.makedirs(self.evaluation_config.save_dir)

        # Load all predicted files from all ranks, deduplicating conditions
        # that were duplicated by DistributedSampler padding
        all_prediction_files = sorted(glob(
            os.path.join(self.predicted_adata_dir, 'predicted_rank*_batch*.h5ad')
        ))

        seen_condition_indices = set()
        predicted_adatas = []
        for pred_file in all_prediction_files:
            adata = ad.read_h5ad(pred_file)
            if '_condition_idx' in adata.obs.columns:
                new_conditions = set(adata.obs['_condition_idx'].unique()) - seen_condition_indices
                if new_conditions:
                    mask = adata.obs['_condition_idx'].isin(new_conditions)
                    predicted_adatas.append(adata[mask].copy())
                    seen_condition_indices.update(new_conditions)
            else:
                predicted_adatas.append(adata)
            del adata

        # Merge all predictions
        predicted_adata = ad.concat(predicted_adatas, axis=0)
        predicted_adata.obs_names_make_unique()
        if '_condition_idx' in predicted_adata.obs.columns:
            predicted_adata.obs.drop(columns=['_condition_idx'], inplace=True)

        del predicted_adatas
        gc.collect()

        ## Load reference anndata
        reference_adata = self.trainer.datamodule.test_iterator.reference_adata

        # Even if our models use NB/ZINB/poisson which are count-based observation models,
        # the data for training may have already been normalized and log1p
        if self.count_based_input_expression:
            sc.pp.normalize_total(reference_adata, inplace=True, target_sum=1e4)
            sc.pp.log1p(reference_adata)
            sc.pp.normalize_total(predicted_adata, inplace=True, target_sum=1e4)
            sc.pp.log1p(predicted_adata)

        # If we are not using synthetic controls, we need to add the control data to the predicted data
        if not self.evaluation_config.use_synthetic_controls:
            control_adata = reference_adata[
                reference_adata.obs[train_context['perturbation_key']] == train_context['perturbation_control_value']
                ]
            predicted_adata = predicted_adata[
                predicted_adata.obs[train_context['perturbation_key']] != train_context['perturbation_control_value']
                ]
            # control data is needed for computing deg, logfc and others
            predicted_adata = ad.concat([predicted_adata, control_adata])
            predicted_adata.obs_names_make_unique()
        else:
            assert train_context['perturbation_control_value'] in predicted_adata.obs[
                train_context['perturbation_key']].unique()

        # Get model name
        model_name = str(self.__class__).split('.')[-1].replace('\'>', '')

        # Create evaluation object with merged data
        ev = Evaluation(
            model_adatas=[predicted_adata],
            model_names=[model_name],
            ref_adata=reference_adata,
            pert_col=train_context['perturbation_key'],
            cov_cols=train_context['covariate_keys'],
            ctrl=train_context['perturbation_control_value'],
        )

        # Run aggregations
        for aggr in self.unique_aggregations:
            ev.aggregate(
                aggr_method=aggr,
                use_control_variance=self.evaluation_config.use_control_variance
            )

        summary_metrics_dict = {}
        summary_metrics_by_cov_dict = {}
        for eval_dict in self.evaluation_config.evaluation_pipelines:
            aggr = eval_dict['aggregation']
            metric = eval_dict['metric']
            ev.evaluate(aggr_method=aggr, metric=metric)

            df = ev.evals[aggr][metric].copy()
            df['covariate'] = ['_'.join(x.split('_')[:-1]) for x in df.cov_pert]
            avg = df.groupby('model').mean('metric')
            avg_by_cov = df.groupby(['covariate']).mean('metric')
            summary_metrics_dict[metric + '_' + aggr] = avg['metric']
            summary_metrics_by_cov_dict[metric + '_' + aggr] = avg_by_cov['metric']

            if eval_dict.get('rank'):
                ev.evaluate_pairwise(aggr_method=aggr, metric=metric)
                ev.evaluate_rank(aggr_method=aggr, metric=metric)

                rank_df = ev.rank_evals[aggr][metric].copy()
                rank_df['covariate'] = ['_'.join(x.split('_')[:-1]) for x in rank_df.cov_pert]
                avg_rank = rank_df.groupby('model').mean('rank')
                avg_rank_by_cov = rank_df.groupby(['covariate']).mean('rank')
                summary_metrics_dict[metric + '_rank_' + aggr] = avg_rank['rank']
                summary_metrics_by_cov_dict[metric + '_rank_' + aggr] = avg_rank_by_cov['rank']

        summary_metrics = pd.DataFrame(summary_metrics_dict).T.map(
            lambda x: float(np.format_float_positional(x, precision=4, unique=False, fractional=False, trim='k')),
        )

        # ---- Early save: persist non-cell-eval metrics before running cell-eval ----
        self.summary_metrics = summary_metrics
        _save_and_log_metrics(
            summary_metrics=summary_metrics,
            summary_metrics_by_cov_dict=summary_metrics_by_cov_dict,
            ev=ev,
            evaluation_config=self.evaluation_config,
            logger_obj=self.logger,
            print_summary=self.evaluation_config.print_summary,
        )

        # ---- Cell-eval metrics (run in isolated subprocess) ----
        run_cell_eval = self.evaluation_config.get('run_cell_eval', True)

        if run_cell_eval and train_context['covariate_keys']:
            continuous_covs = [
                k for k in train_context['covariate_keys']
                if is_numeric_dtype(reference_adata.obs[k])
            ]
            if continuous_covs:
                warnings.warn(
                    f"Skipping cell_eval: continuous covariates detected ({continuous_covs}). "
                    f"cell_eval does not support continuous covariates for control matching.",
                    stacklevel=1
                )
                run_cell_eval = False

        if run_cell_eval:
            covariate_keys = train_context['covariate_keys']

            if self.evaluation_config.get('baseline_metrics', False):
                baseline_metrics = dict(self.evaluation_config.baseline_metrics)
            else:
                baseline_metrics = None

            if covariate_keys:
                predicted_adata.obs['covariate_combo'] = predicted_adata.obs[covariate_keys].apply(
                    lambda row: '_'.join(row.astype(str)), axis=1
                )
                reference_adata.obs['covariate_combo'] = reference_adata.obs[covariate_keys].apply(
                    lambda row: '_'.join(row.astype(str)), axis=1
                )
                unique_covariates = predicted_adata.obs['covariate_combo'].unique()
            else:
                predicted_adata.obs['covariate_combo'] = 'all'
                reference_adata.obs['covariate_combo'] = 'all'
                unique_covariates = ['all']

            cell_eval_metrics_by_cov = {}
            cell_eval_weighted_metrics = {}
            total_perturbations = 0
            cell_eval_timeout = self.evaluation_config.get('cell_eval_timeout', 900)

            for cov_combo in unique_covariates:
                pred_cov_adata = predicted_adata[predicted_adata.obs['covariate_combo'] == cov_combo].copy()
                ref_cov_adata = reference_adata[reference_adata.obs['covariate_combo'] == cov_combo].copy()

                n_perturbations = len(pred_cov_adata.obs[
                                          pred_cov_adata.obs[train_context['perturbation_key']] != train_context[
                                              'perturbation_control_value']
                                          ][train_context['perturbation_key']].unique())

                if n_perturbations == 0:
                    continue

                cell_eval_agg_results = _run_cell_eval_subprocess(
                    pred_adata=pred_cov_adata,
                    real_adata=ref_cov_adata,
                    control_pert=train_context['perturbation_control_value'],
                    pert_col=train_context['perturbation_key'],
                    outdir=self.evaluation_config.save_dir + f'/cell_eval/{cov_combo}',
                    num_threads=12,
                    batch_size=2000,
                    timeout=cell_eval_timeout,
                )
                if cell_eval_agg_results is None:
                    _logger.warning(f"Skipping cell-eval metrics for covariate {cov_combo}")
                    continue

                cov_metrics = {}
                for metric_name in cell_eval_agg_results.columns:
                    metric_value = float(cell_eval_agg_results[metric_name].loc["mean"])
                    cov_metrics[metric_name] = metric_value

                    if baseline_metrics is not None and metric_name in baseline_metrics:
                        baseline_metric_value = baseline_metrics[metric_name]
                        if metric_name == 'mae':
                            scaled_metric_value = (baseline_metric_value - metric_value) / baseline_metric_value
                        else:
                            scaled_metric_value = (metric_value - baseline_metric_value) / (1 - baseline_metric_value)
                        cov_metrics[f'{metric_name}_scaled'] = max(0.0, scaled_metric_value)
                    else:
                        scaled_metric_value = None

                    if metric_name not in cell_eval_weighted_metrics:
                        cell_eval_weighted_metrics[metric_name] = 0
                    if f'{metric_name}_scaled' not in cell_eval_weighted_metrics:
                        cell_eval_weighted_metrics[f'{metric_name}_scaled'] = 0
                    cell_eval_weighted_metrics[metric_name] += metric_value * n_perturbations
                    if scaled_metric_value is not None:
                        cell_eval_weighted_metrics[f'{metric_name}_scaled'] += scaled_metric_value * n_perturbations

                if cov_metrics:
                    cell_eval_metrics_by_cov[cov_combo] = cov_metrics

                total_perturbations += n_perturbations

            cell_eval_metrics = {}
            if total_perturbations > 0:
                for metric_name, weighted_sum in cell_eval_weighted_metrics.items():
                    cell_eval_metrics[metric_name] = weighted_sum / total_perturbations

            for cov_combo, cov_metrics in cell_eval_metrics_by_cov.items():
                for metric_name, metric_value in cov_metrics.items():
                    if metric_name not in summary_metrics_by_cov_dict:
                        summary_metrics_by_cov_dict[metric_name] = {}
                    summary_metrics_by_cov_dict[metric_name][cov_combo] = metric_value

            if cell_eval_metrics:
                cell_eval_df = pd.DataFrame([cell_eval_metrics], index=[model_name])
                summary_metrics = pd.concat([summary_metrics, cell_eval_df.T])

            # ---- Update save: overwrite with full metrics including cell-eval ----
            self.summary_metrics = summary_metrics
            _save_and_log_metrics(
                summary_metrics=summary_metrics,
                summary_metrics_by_cov_dict=summary_metrics_by_cov_dict,
                ev=ev,
                evaluation_config=self.evaluation_config,
                logger_obj=self.logger,
                print_summary=self.evaluation_config.print_summary,
            )

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
