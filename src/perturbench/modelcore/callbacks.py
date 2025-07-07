"""
Callbacks for the perturbench model training.
"""

import logging
from typing import List, Optional
import lightning as L
import torch
import numpy as np
import pandas as pd
import anndata as ad
import gc
import os
from perturbench.analysis.benchmarks.evaluation import Evaluation, merge_evals

log = logging.getLogger(__name__)


class EvaluationCallback(L.Callback):
    """
    Callback to run evaluation during training at fixed intervals.
    
    This callback uses the full evaluation pipeline to compute comprehensive
    metrics on validation data during training, similar to the test evaluation.
    """
    
    def __init__(
        self,
        eval_interval: int = 5,
        log_prefix: str = "eval",
        max_batches: int = 10,
    ):
        """
        Initialize the evaluation callback.
        
        Args:
            eval_interval: Run evaluation every N epochs
            log_prefix: Prefix for logged metrics
            max_batches: Maximum validation batches to evaluate
        """
        super().__init__()
        self.eval_interval = eval_interval
        self.log_prefix = log_prefix
        self.max_batches = max_batches
        self.current_epoch = 0
        
    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Track current epoch."""
        self.current_epoch = trainer.current_epoch
        
    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Run evaluation at the end of training epochs if interval is reached."""
        if (self.current_epoch + 1) % self.eval_interval == 0:
            log.info(f"Running evaluation at epoch {self.current_epoch + 1}")
            self._run_evaluation(trainer, pl_module)
    
    def _run_evaluation(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Run evaluation on validation data using the full evaluation pipeline."""
        pl_module.eval()
        
        val_dataloader = trainer.val_dataloaders
        if not val_dataloader:
            log.warning("No validation dataloaders found")
            pl_module.train()
            return
                
        
        # Collect predictions and create AnnData objects
        all_predictions = []
        all_targets = []
        all_obs = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader):
                
                # Move batch to the same device as the model
                batch = {k: v.to(pl_module.device) if hasattr(v, 'to') else v for k, v in batch.items()}
                
                observed_perturbed_expression, control_expression, perturbation, covariates, _ = pl_module.unpack_batch(batch)
                predicted_perturbed_expression = pl_module.forward(control_expression, perturbation, covariates)
                    
                # Convert to numpy and collect
                pred_np = predicted_perturbed_expression.cpu().numpy()
                target_np = observed_perturbed_expression.cpu().numpy()
                    
                all_predictions.append(pred_np)
                all_targets.append(target_np)
                
                # Create observation metadata
                batch_obs = pd.DataFrame({
                    'batch_idx': [batch_idx] * len(pred_np),
                    'sample_idx': range(len(pred_np))
                })
                all_obs.append(batch_obs)

                # Limit evaluation to avoid slowdown
                if batch_idx >= self.max_batches:
                    break
                
        
        if all_predictions and all_targets:
            predictions = np.vstack(all_predictions)
            targets = np.vstack(all_targets)
            obs_df = pd.concat(all_obs, ignore_index=True)
            
            # Create AnnData objects for evaluation
            predicted_adata = ad.AnnData(
                X=predictions,
                obs=obs_df,
                var=pd.DataFrame(index=[f"gene_{i}" for i in range(predictions.shape[1])])
            )
            
            reference_adata = ad.AnnData(
                X=targets,
                obs=obs_df,
                var=pd.DataFrame(index=[f"gene_{i}" for i in range(targets.shape[1])])
            )
            
            # Get training context from model
            train_context = pl_module.training_record.get("train_context", {})
            model_name = str(pl_module.__class__).split(".")[-1].replace("'>", "")
            
            # Create evaluation object
            ev = Evaluation(
                model_adatas=[predicted_adata],
                model_names=[model_name],
                ref_adata=reference_adata,
                pert_col=train_context.get("perturbation_key", "condition"),
                cov_cols=train_context.get("covariate_keys", []),
                ctrl=train_context.get("perturbation_control_value", "ctrl"),
            )
            
            # Run evaluation pipelines
            summary_metrics_dict = {}
            
            # Use the model's evaluation configuration
            if hasattr(pl_module, 'evaluation_config') and pl_module.evaluation_config:
                evaluation_pipelines = pl_module.evaluation_config.get('evaluation_pipelines', [])
                log.info(f"Using {len(evaluation_pipelines)} evaluation pipelines from model config")
            else:
                log.warning("No evaluation configuration found in model, skipping evaluation")
                pl_module.train()
                return
            
            for eval_dict in evaluation_pipelines:
                aggr = eval_dict["aggregation"]
                metric = eval_dict["metric"]
                
                # Aggregate
                ev.aggregate(aggr_method=aggr)
                
                # Evaluate
                ev.evaluate(aggr_method=aggr, metric=metric)
                
                # Get average metric
                df = ev.evals[aggr][metric].copy()
                if len(df) > 0:
                    avg = df.groupby("model").mean("metric")
                    summary_metrics_dict[f"{metric}_{aggr}"] = avg["metric"].iloc[0]
                
                # Add rank metrics if requested
                if eval_dict.get("rank"):
                    ev.evaluate_pairwise(aggr_method=aggr, metric=metric)
                    ev.evaluate_rank(aggr_method=aggr, metric=metric)
                    
                    rank_df = ev.rank_evals[aggr][metric].copy()
                    if len(rank_df) > 0:
                        avg_rank = rank_df.groupby("model").mean("rank")
                        summary_metrics_dict[f"{metric}_rank_{aggr}"] = avg_rank["rank"].iloc[0]
            
            # Log metrics
            for metric_name, value in summary_metrics_dict.items():
                if trainer.logger is not None:
                    trainer.logger.log_metrics({
                        f"{self.log_prefix}_{metric_name}": float(value)
                    }, step=trainer.global_step)
                
            log.info(f"Evaluation metrics at epoch {self.current_epoch + 1}: {summary_metrics_dict}")
            
        else:
            log.warning("No valid predictions and targets collected for evaluation")
        
        pl_module.train() 