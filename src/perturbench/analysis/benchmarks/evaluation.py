## Model evaluation module
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from anndata import AnnData
from collections import defaultdict
import pickle
import os

from .aggregation import aggregate_adata
from .metrics import compare_perts, pairwise_metric_helper, rank_helper
from ..plotting import boxplot_jitter, scatter_labels

class Evaluation:
    """A class for evaluating perturbation prediction models.

    Attributes:
        adatas (dict)
            Dictionary of model predictions as AnnData objects. The `ref` key is reserved for reference perturbation expression.
        pert_col (str)
            Name of perturbation column in AnnData obs DataFrames
        cov_cols (list)
            Name(s) of covariate column(s) in AnnData obs DataFrames
        ctrl (str)
            Name of control perturbation
        aggr (dict)
            Dictionary of model predictions/reference with cells aggregated by perturbation. Access pattern is `aggr[aggr_method][model_name]` which returns an AnnData object
        evals (dict)
            Dictionary of prediction evaluations. Access pattern is `evals[aggr_method][metric]` which returns a dataframe of evaluation scores
        pairwise_evals (dict)
            Dictionary of pairwise prediction evaluations. Access pattern is `pairwise_evals[aggr_method][metric][model_name]` which returns a dataframe of pairwise evaluation scores
        rank_evals (dict)
            Dictionary of rank prediction evaluations computed from the pairwise_evals matrices. Access pattern is `rank_evals[aggr_method][metric]` which returns a dataframe of rank evaluation scores
        deg_dict (dict)
            Dictionary of differentially expressed genes per perturbation/covariate
        use_degs (dict)
            Dictionary of whether differentially expressed genes are used for evaluation
        ref_uns (dict)
            Dictionary of reference AnnData uns keys
    """

    def __init__(
        self,
        model_adatas: list[AnnData] | dict,
        ref_adata: AnnData,
        pert_col: str,
        model_names: list = None,
        ctrl: str | list | None = None,
        cov_cols: list | str | None = None,
        features: list | None = None,
    ):
        """The constructor for the Evaluation class.

        Args:
            model_adatas (list, dict): List or dict of predicted perturbation responses as AnnData objects
            ref_adata (anndata.AnnData): Reference perturbation response as AnnData object
            pert_col (str): Name of perturbation column in predicted/reference AnnData obs DataFrames
            model_names (list): List of model names (optional if model_adatas is a dict)
            ctrl (str): Name of control perturbation
            cov_cols (list): Name(s) of covariate column(s) in predicted/reference AnnData obs DataFrames
            features (list): Subset of features to use for evaluation (default: use all features)

        Note:
            This constructor modifies the input AnnData objects in-place
            by removing 'X_pca' from obsm if present.
        """
        if features is None:
            features = list(ref_adata.var_names)

        if isinstance(model_adatas, dict):
            model_names = list(model_adatas.keys())
            model_adatas = list(model_adatas.values())
        else:
            if model_names is None:
                raise ValueError(
                    "Please specify model names if model_adatas is a list of AnnData objects"
                )
            if not len(model_names) == len(model_adatas):
                raise ValueError(
                    "Number of model names does not match number of model adatas"
                )

        for name in model_names:
            if name == "ref":
                raise ValueError("Model name 'ref' is reserved for reference perturbation expression")

        adata_dict = {}
        for i, k in enumerate(model_names):
            adata_i = model_adatas[i][:, features]
            if adata_i.obs[pert_col].dtype.name != "category":
                adata_i.obs[pert_col] = adata_i.obs[pert_col].astype("category")
            if 'X_pca' in adata_i.obsm.keys():
                del adata_i.obsm['X_pca']
            adata_dict[k] = adata_i
        
        if 'X_pca' in ref_adata.obsm.keys():
            del ref_adata.obsm['X_pca']
        adata_dict["ref"] = ref_adata

        if isinstance(ctrl, list):
            for k, adata in adata_dict.items():
                adata[adata.obs[pert_col].isin(ctrl)].obs[pert_col] = "ctrl"
                adata_dict[k] = adata
            self.ctrl = "ctrl"

        else:
            self.ctrl = ctrl

        if isinstance(cov_cols, str):
            cov_cols = [cov_cols]

        if cov_cols is None:
            cov_cols = []

        self.adatas = adata_dict
        self.pert_col = pert_col
        self.cov_cols = cov_cols
        self.aggr = {}
        self.evals = {}
        self.pairwise_evals = {}
        self.rank_evals = {}
        self.deg_dict = None
        self.use_degs = defaultdict(dict)
        self.ref_uns = self.adatas["ref"].uns.copy()

        self.mmd_df = None

    def aggregate(
        self,
        aggr_method: str = "average",
        delim: str = "_",
        pseudocount: float = 0.1,
        adjust_var: bool = True,
        de_method: str = "t-test_overestim_var",
        use_control_variance: bool = False,
        n_comps: int = 100,
        pca_model=None,
        **kwargs,
    ):
        """Aggregate cells per perturbation

        Args:
            aggr_method (str): Method used to aggregate cells per perturbation (logFC, average, scaled avg, pca, default: logFC)
            delim (str): Delimiter separating covariates (default: '_')
            pseudocount (float): Pseudocount to add to fold-changes to avoid undefined log fold-changes
            adjust_var (bool): If `aggr_method` is `variance`, use variances adjusted by average expression (default: True)
            de_method (str): If `aggr_method` is `logp`, use this differential expression method for computing p-values
            n_comps (int): Number of PCA components to use when aggr_method is 'pca' (default: 30)
        """
        if aggr_method not in self.list_available_aggregations():
            raise ValueError(f"Invalid aggregation method {aggr_method}")
        
        agg_adatas = {}
        ref_adata = self.adatas["ref"] if aggr_method in ['pca', 'pca_average'] else None
        for model_name in self.adatas:
            if model_name == 'ref':
                use_control_variance_model = False
            else:
                use_control_variance_model = use_control_variance
            
            agg_adatas[model_name] = aggregate_adata(
                self.adatas[model_name],
                aggr_method=aggr_method,
                pert_col=self.pert_col,
                cov_cols=self.cov_cols,
                ctrl=self.ctrl,
                pseudocount=pseudocount,
                delim=delim,
                adjust_var=adjust_var,
                de_method=de_method,
                use_control_variance=use_control_variance_model,
                ref_adata=ref_adata,
                n_comps=n_comps,
                pca_model=pca_model,
                **kwargs,
            )

        self.aggr[aggr_method] = agg_adatas
        self.evals[aggr_method] = {}
        self.pairwise_evals[aggr_method] = {}
        self.rank_evals[aggr_method] = {}

    def get_aggr(self, aggr_method: str, model: str):
        """Returns perturbation expression aggregated per perturbation

        Args:
            aggr_method (str): Method used to aggregate cells per perturbation (logFC, average, scaled avg)
            model (str): Name of model to return aggregated expression for (specify `ref` to return reference expression)

        Returns:
            anndata.AnnData: perturbation expression aggregated per perturbation
        """
        return self.aggr[aggr_method][model]


    def evaluate(
        self,
        aggr_method: str = "logfc",
        metric: str = "pearson",
        perts: list | None = None,
        plot: bool = False,
        plot_size: tuple | None = None,
        return_df: bool = False,
        n_top_degs: int | None = None,
    ):
        """Evaluate predicted perturbation effect against reference perturbation effect

        Args:
            aggr_method (str): Method used to aggregate cells per perturbation (logFC, average, scaled avg, none, default: logFC). Set to none if using energy distance as the evaluation metric
            metric (str): Metric used to measure prediction accuracy (pearson, spearman, r2_score, dcor, cosine, mse, energy, deg_recall, default: pearson)
            perts (list): Subset of perturbations to evaluate (default: evaluate all perturbations)
            return_df (bool): If True, return evaluation dataframe
            plot (bool): If True, plot evaluation results as boxplots
            plot_size (tuple): Size of plot (default: automatically set depending on number of plots)
            deg_key: Key in `adata.uns` where differentially expressed genes are stored (default: None)
            n_top_genes (int): Number of top genes to plot if subsetting to differentially expressed genes (DEGs) (default: 100)

        Returns:
            None or pandas.DataFrame: Per-perturbation evaluation of model predictions
        """

        if metric not in self.list_available_metrics():
            raise ValueError("Metric not implemented")

        if n_top_degs is not None:
            if self.deg_dict is None:
                self.deg_dict = aggregate_adata(
                    self.adatas["ref"],
                    aggr_method="scores",
                    pert_col=self.pert_col,
                    cov_cols=self.cov_cols,
                    ctrl=self.ctrl,
                    pseudocount=0.1,
                    delim="_",
                )
            self.use_degs[aggr_method][metric] = True
        else:
            self.use_degs[aggr_method][metric] = False

        if aggr_method not in self.aggr.keys():
            self.aggregate(aggr_method=aggr_method)

        aggr_ref = self.aggr[aggr_method]["ref"]

        evals = []
        for k, aggr in self.aggr[aggr_method].items():
            if k not in ["ref", "target"]:
                for cov, cov_aggr in aggr.items():
                    ref_perts = list(aggr_ref[cov].keys())
                    perts = list(set(cov_aggr.keys()).intersection(ref_perts))
                    if len(perts) == 0:
                        raise ValueError(f"No perturbations in common between {k} and ref for covariate {cov}")
                    
                    if n_top_degs is not None:
                        deg_mask = {}
                        for p in perts:
                            p_scores = np.abs(self.deg_dict[cov][p])
                            indices = np.argsort(p_scores)[-n_top_degs:]
                            indices = indices[::-1]
                            deg_mask[p] = indices
                    else:
                        deg_mask = None
                    
                    scores = compare_perts(
                        cov_aggr,
                        aggr_ref[cov],
                        perts=perts,
                        metric=metric,
                        deg_mask=deg_mask,
                    )
                    df = pd.DataFrame(
                        index=perts,
                        data={
                            "cov_pert": [f"{cov}_{p}" for p in perts],
                            "model": [k] * len(perts),
                            "metric": scores,
                        },
                    )
                    evals.append(df)

        evals = pd.concat(evals, axis=0)
        evals.reset_index(drop=True, inplace=True)

        self.evals[aggr_method][metric] = evals

        if plot:
            self.summary_plots(
                aggr_method=aggr_method, metrics=[metric], figsize=plot_size
            )

        if return_df:
            return evals

    def get_eval(
        self,
        aggr_method: str,
        metric: str,
        melt: bool = False,
    ):
        """Returns per-perturbation evaluation of model predictions

        Args:
            aggr_method (str): Method used to aggregate cells per perturbation (logFC, average, scaled avg, default: logFC)
            metric (str): Metric used to measure prediction accuracy (pearson, spearman, r2_score, dcor, default: pearson)
            melt (bool): If True, return DataFrame with one column per model. Otherwise each model will have its own column

        Returns:
            pandas.DataFrame: Per-perturbation evaluation of model predictions
        """
        eval_df = self.evals[aggr_method][metric]
        if not melt:
            eval_df = eval_df.pivot(index="cov_pert", columns="model", values="metric")
        return eval_df


    def evaluate_pairwise(
        self,
        aggr_method: str,
        metric: str,
        models: list | None = None,
    ):
        """Evaluate every predicted perturbation effect against every reference perturbation effect

        Args:
            aggr_method (str): Method used to aggregate cells per perturbation (logFC, average, scaled avg)
            metric (str): Metric used to measure prediction accuracy (pearson, spearman, r2_score, dcor, cosine, mse)
            models (list,None): List of models evaluate (default: evaluate all models)
            deg_key (str | None, optional): Key in `adata.uns` where differentially expressed genes are stored (default: None)
            verbose (bool, optional): If True, print evaluation progress (default: False)

        Returns:
            Stashes matrix of predicted perturbation effect vs reference perturbation effect evaluation scores per unique set of covariates in `self.pairwise_evals`
        """
        self.pairwise_evals[aggr_method][metric] = {}
        ref_aggr = self.aggr[aggr_method]["ref"]
        
        if models is None:
            models = [
                k for k in self.aggr[aggr_method].keys() if k not in ["ref", "target"]
            ]

        for model_name in models:
            if model_name not in self.aggr[aggr_method].keys():
                raise ValueError("Model %s not found" % model_name)

            aggr = self.aggr[aggr_method][model_name]

            if (self.cov_cols is not None) and (len(self.cov_cols) > 0):
                cov_unique = list(ref_aggr.keys())
                mat_dict = {}
                for cov in cov_unique:
                    aggr_cov = aggr[cov]
                    ref_aggr_cov = ref_aggr[cov]

                    perts_common = list(
                        set(aggr_cov.keys()).intersection(ref_aggr_cov.keys())
                    )

                    mat = pairwise_metric_helper(
                        aggr_cov,
                        ref_aggr_cov,
                        perts=perts_common,
                        metric=metric,
                    )
                    mat_dict[cov] = mat
                self.pairwise_evals[aggr_method][metric][model_name] = mat_dict

            else:
                perts_common = list(
                    set(aggr.keys()).intersection(ref_aggr.keys())
                )
                mat = pairwise_metric_helper(
                    aggr,
                    ref_aggr,
                    perts=perts_common,
                    metric=metric,
                )
                self.pairwise_evals[aggr_method][metric][model_name] = mat

    def evaluate_rank(
        self,
        aggr_method: str,
        metric: str,
        models: list | None = None,
        return_df: bool=False,
        transpose: bool=False,
    ):
        """Evaluate rank ordering of predicted perturbation effects vs a given reference perturbation effect.
           A rank of 0 indicates predictions are ordered perfectly. A rank of 0.5 indicates predictions are ordered randomly.

        Args:
            aggr_method (str): Method used to aggregate cells per perturbation (logFC, average, scaled avg)
            metric (str): Metric used to measure prediction accuracy (pearson, spearman, r2_score, dcor, cosine, mse)
            models (list,None): List of models evaluate (default: evaluate all models)

        Returns:
            Stashes a dataframe of rank evaluation scores per unique set of covariates in `self.rank_evals`
        """
        if (self.pairwise_evals is None) or (
            self.pairwise_evals[aggr_method][metric] is None
        ):
            raise ValueError(
                "Please run pairwise evaluation using aggregation method %s and metric %s first"
                % (aggr_method, metric)
            )

        if models is None:
            models = [
                k for k in self.aggr[aggr_method].keys() if k not in ["ref", "target"]
            ]

        if metric in ["pearson", "spearman", "dcor", "r2_score", "cosine", "top_k_recall"]:
            metric_type = "similarity"
        elif metric in ["mse", "rmse", "mae", "energy", "mmd"]:
            metric_type = "distance"
        else:
            raise ValueError("Invalid metric")

        ref_aggr = self.aggr[aggr_method]["ref"]
        rank_df_list = []
        for model in models:
            if (self.cov_cols is not None) and (len(self.cov_cols) > 0):
                cov_unique = list(ref_aggr.keys())
                for cov in cov_unique:
                    mat = self.pairwise_evals[aggr_method][metric][model][cov]
                    cov_ranks = rank_helper(mat, metric_type=metric_type)
                    cov_df = pd.DataFrame(
                        {
                            "model": [model] * len(cov_ranks),
                            "cov_pert": [f"{cov}_{p}" for p in cov_ranks.index.values],
                            "rank": cov_ranks.values,
                        }
                    )
                    
                    if transpose:
                        cov_ranks_transpose = rank_helper(mat.T, metric_type=metric_type)
                        cov_df['rank_transpose'] = cov_ranks_transpose.values
                    
                    rank_df_list.append(cov_df)
            else:
                mat = self.pairwise_evals[aggr_method][metric][model]
                ranks = rank_helper(mat, metric_type=metric_type)
                df = pd.DataFrame(
                    {
                        "model": [model] * len(ranks),
                        "cov_pert": ranks.index,
                        "rank": ranks.values,
                    }
                )
                
                if transpose:
                    ranks_transpose = rank_helper(mat.T, metric_type=metric_type)
                    df['rank_transpose'] = ranks_transpose.values
                
                rank_df_list.append(df)

        rank_df = pd.concat(rank_df_list)
        self.rank_evals[aggr_method][metric] = rank_df
        
        if return_df:
            return rank_df

    def get_rank_eval(
        self,
        aggr_method: str,
        metric: str,
        melt: bool = False,
    ):
        """Returns per-perturbation evaluation of model predictions

        Args:
            aggr_method (str): Method used to aggregate cells per perturbation (logFC, average, scaled avg, default: logFC)
            metric (str): Metric used to measure prediction accuracy (pearson, spearman, r2_score, dcor, default: pearson)
            melt (bool): If True, return DataFrame with one column per model. Otherwise each model will have its own column

        Returns:
            pandas.DataFrame: Per-perturbation evaluation of model predictions
        """
        eval_df = self.rank_evals[aggr_method][metric]
        if not melt:
            eval_df = eval_df.pivot(index="cov_pert", columns="model", values="rank")
        return eval_df

    def prediction_scatter(
        self,
        perts: list,
        features: list | None = None,
        aggr_method: str = "logfc",
        models: list | None = None,
        metric: str = "pearson",
        x_title: str = "pred expr",
        y_title: str = "ref expr",
        axis_title_size: float = 15,
        figsize: tuple | None = None,
        show_metric=True,
        n_top_genes: int = 100,
        quadrants=True,
        **kwargs,
    ):
        """Scatterplots of true vs predicted expression for each model prediction

        Args:
            perts (list): Perturbations to plot
            features (list,None): Features to plot (default: all features)
            aggr_method (str): Method used to aggregate cells per perturbation (logFC, average, scaled avg, default: logFC)
            models (list,None): List of models to plot (default: plot all models)
            metric (str): Metric used to measure prediction accuracy (pearson, spearman, r2_score, dcor, default: pearson)
            x_title (str): X-axis title (default: pred expr)
            y_title (str): Y-axis title (default: ref expr)
            axis_title_size (float): Size of x/y-axis titles (default: 15)
            title_size (float): Size of plot title (default: 16)
            figsize (tuple, None): Figure size (default: automatically set depending on number of plots)
            show_metric (bool): If true, show metric on plot
            n_top_genes (int): Number of top genes to plot if subsetting to differentially expressed genes (DEGs) (default: 100)
            **kwargs: Additional keyword arguments passed onto `scatter_labels`
        """
        assert aggr_method in self.aggr.keys()
        assert metric in self.evals[aggr_method].keys()

        adata_ref = self.aggr[aggr_method]["ref"]

        if features is None:
            features = list(adata_ref.var_names)

        if models is None:
            models = [
                k for k in self.aggr[aggr_method].keys() if k not in ["ref", "target"]
            ]

        num_models = len(models)
        num_perts = len(perts)
        if figsize is None:
            figsize = (num_models * 3 + 0.25, num_perts * 3)

        eval_df = self.evals[aggr_method][metric]

        fig, axs = plt.subplots(num_perts, num_models, figsize=figsize, squeeze=False)
        for i, p in enumerate(perts):
            use_degs = self.use_degs[aggr_method][metric]
            if use_degs:
                features_use = self.deg_dict[p][:n_top_genes]
            else:
                features_use = features

            for j, k in enumerate(models):
                ax = axs[i, j]

                scatter_labels(
                    x=self.aggr[aggr_method][k].to_df().loc[p].loc[features_use],
                    y=adata_ref.to_df().loc[p].loc[features_use],
                    axis_title_size=axis_title_size,
                    ax=ax,
                    quadrants=quadrants,
                    **kwargs,
                )

                if show_metric:
                    score = eval_df.loc[
                        (eval_df.model == k) & (eval_df.cov_pert == p), "metric"
                    ]
                    score = str(np.round(score.iloc[0], decimals=3))
                    ax.text(0.05, 0.9, metric + "=" + score, transform=ax.transAxes)

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        for ax, model in zip(axs[0], models, strict=True):
            ax.set_title(model)

        for ax, pert in zip(axs[:, 0], perts, strict=True):
            ax.set_ylabel(pert, size="large")

        fig.supxlabel(x_title, size=axis_title_size)
        fig.supylabel(y_title, size=axis_title_size)
        plt.show()

    def summary_plots(
        self,
        aggr_method: str = "logfc",
        metrics: str | list[str] | None = None,
        models: list | None = None,
        figsize: tuple | None = None,
        alpha: float = 0.8,
        title: str | None = None,
        title_size: int = 16,
        violin: bool = False,
        ylim: tuple | None = None,
        **kwargs,
    ):
        """Box or violin plots summarizing model performance per perturbation

        Args:
            aggr_method (str): Method used to aggregate cells per perturbation (logFC, average, scaled avg, default: logFC)
            metrics (str, list): Metric(s) used to measure prediction accuracy
                                 (pearson, spearman, r2_score, dcor, default: pearson)
            models (list, None): List of models to plot (default: plot all models)
            figsize (tuple, None): Figure size (default: automatically set depending on number of plots)
            alpha (float): Transparency of plots
            title (str): Overall plot title
            title_size (float): Size of plot title (default: 16)
            violin (bool): If true, generate violin plots. Otherwise generate boxplots.
            ylim (tuple, None): Y-axis limits
            **kwargs: Additional keyword arguments passed onto `boxplot_jitter`
        """
        if metrics is None:
            metrics = ["pearson"]

        if isinstance(metrics, str):
            metrics = [metrics]

        ncols = len(metrics)

        nplots = len(metrics)
        nrows = int(np.ceil(float(nplots) / ncols))
        if figsize is None:
            figsize = (4 * ncols, 4 * nrows)

        fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
        for i, m in enumerate(metrics):
            df = self.evals[aggr_method][m]
            if models is not None:
                df = df.loc[df.model.isin(models), :]

            if nplots > 1:
                ax = axs[i]
            else:
                ax = axs

            boxplot_jitter(
                "model",
                "metric",
                df,
                y_title=m,
                axis_title_size=14,
                axis_text_size=14,
                alpha=alpha,
                ax=ax,
                violin=violin,
                **kwargs,
            )
            if ylim is not None:
                ax.set_ylim(ylim)

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.suptitle(title, size=title_size)
        plt.show()

    @classmethod
    def list_available_aggregations(cls):
        return [
            "average",
            "scaled",
            "logfc",
            "logp",
            "logp_adjusted",
            "var",
            "var-logfc",
            "scores",
            "pvals",
            "none",
            "pca",
            "pca_average",
        ]
    
    @classmethod
    def list_available_metrics(cls):
        return [
            "pearson",
            "r2_score",
            "mse",
            "rmse",
            "mae",
            "cosine",
            "cosine_weighted",
            "energy",
            "top_k_recall",
            "mmd",
        ]
    
    def save(
        self,
        save_path: str,
        save_adatas: bool = False,
    ):
        """Save evaluation object to disk"""
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if not save_adatas:
            self.adatas = None

        with open(save_path + "/eval.pkl", "wb") as f:
            pickle.dump(self, f)

    def copy(
        self,
    ):
        """Copy evaluation object"""
        return pickle.loads(pickle.dumps(self))


def load_eval(
    save_path: str,
):
    """ "Load evaluation object from disk"""
    if not os.path.exists(save_path):
        raise ValueError("Path does not exist")
    with open(save_path + "/eval.pkl", "rb") as f:
        ev = pickle.load(f)

    return ev


def merge_evals(
    evals: list[Evaluation],
) -> Evaluation:
    """Merge evaluation objects"""
    for ev in evals:
        if not isinstance(ev, Evaluation):
            raise ValueError("All objects must be of type Evaluation")

    ev_anchor = evals[0].copy()
    ev_anchor.adatas = None

    if len(ev_anchor.aggr) == 0:
        raise ValueError("Please run aggregation on all objects first")
    if len(ev_anchor.evals) == 0:
        raise ValueError("Please run evaluation on all objects first")

    for ev in evals[1:]:
        if len(ev.aggr) == 0:
            raise ValueError("Please run aggregation on all objects first")
        if len(ev.evals) == 0:
            raise ValueError("Please run evaluation on all objects first")

        if ev.pert_col != ev_anchor.pert_col:
            raise ValueError("Perturbation column names do not match")
        for col in ev.cov_cols:
            if col not in ev_anchor.cov_cols:
                raise ValueError("Covariate column names do not match")
        if ev.ctrl != ev_anchor.ctrl:
            raise ValueError("Control perturbations do not match")

    for ev in evals[1:]:
        for aggr_method, aggr_dict in ev.aggr.items():
            if aggr_method not in ev_anchor.aggr:
                ev_anchor.aggr[aggr_method] = aggr_dict
            else:
                for model_name, aggr in aggr_dict.items():
                    if model_name not in ev_anchor.aggr[aggr_method]:
                        ev_anchor.aggr[aggr_method][model_name] = aggr
                    else:
                        for cov in aggr.keys():
                            if cov not in ev_anchor.aggr[aggr_method][model_name]:
                                ev_anchor.aggr[aggr_method][model_name][cov] = aggr[cov]
                            else:
                                for p in aggr[cov].keys():
                                    if p not in ev_anchor.aggr[aggr_method][model_name][cov]:
                                        ev_anchor.aggr[aggr_method][model_name][cov][p] = aggr[cov][p]

        for aggr_method, eval_dict in ev.evals.items():
            if aggr_method not in ev_anchor.evals:
                ev_anchor.evals[aggr_method] = eval_dict
            else:
                for metric, eval_df in eval_dict.items():
                    if metric not in ev_anchor.evals[aggr_method]:
                        ev_anchor.evals[aggr_method][metric] = eval_df
                    else:
                        ev_anchor.evals[aggr_method][metric] = pd.concat(
                            [ev_anchor.evals[aggr_method][metric], eval_df]
                        )

    return ev_anchor
