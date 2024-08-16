## Model evaluation module

import scipy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import anndata as ad
from anndata import AnnData
from ..plotting import boxplot_jitter, scatter_labels
from collections import defaultdict
import pickle
import os
from .aggregation import aggr_helper
from ..utils import merge_cols
from .metrics import compare_perts, pairwise_metric_helper, rank_helper


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
        reduction (str)
            Type of dimensional reduction applied (currently only pca is supported)
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
        ctrl: str | None = None,
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
            assert name != "ref"

        adata_dict = {}
        for i, k in enumerate(model_names):
            adata_i = model_adatas[i][:, features]
            if adata_i.obs[pert_col].dtype.name != "category":
                adata_i.obs[pert_col] = adata_i.obs[pert_col].astype("category")
            if ctrl not in adata_i.obs[pert_col].cat.categories:
                raise ValueError(
                    "Control perturbation not found in model predictions %s" % k
                )
            adata_dict[k] = adata_i

        adata_dict["ref"] = ref_adata
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
        self.reduction = None
        self.deg_dict = None
        self.use_degs = defaultdict(dict)
        self.ref_uns = self.adatas["ref"].uns.copy()

    def aggregate(
        self,
        aggr_method: str = "logfc",
        delim: str = "_",
        pseudocount: float = 0.1,
        adjust_var: bool = True,
        de_method: str = "wilcoxon",
        **kwargs,
    ):
        """Aggregate cells per perturbation

        Args:
            aggr_method (str): Method used to aggregate cells per perturbation (logFC, average, scaled avg, default: logFC)
            delim (str): Delimiter separating covariates (default: '_')
            pseudocount (float): Pseudocount to add to fold-changes to avoid undefined log fold-changes
            adjust_var (bool): If `aggr_method` is `variance`, use variances adjusted by average expression (default: True)
            de_method (str): If `aggr_method` is `logp`, use this differential expression method for computing p-values
        """
        if aggr_method not in [
            "average",
            "scaled",
            "logfc",
            "logp",
            "var",
            "var-logfc",
        ]:
            raise ValueError("Invalid aggregation method")

        agg_adatas = {}
        for k in self.adatas:
            agg_adatas[k] = aggr_helper(
                self.adatas[k],
                aggr_method=aggr_method,
                pert_col=self.pert_col,
                cov_cols=self.cov_cols,
                ctrl=self.ctrl,
                pseudocount=pseudocount,
                delim=delim,
                adjust_var=adjust_var,
                de_method=de_method,
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

    def reduce(
        self,
        reduction: str = "pca",
        n_comps: int = 30,
    ):
        """Reduce dimensionality of expression by projecting predicted expression onto reference expression PCs

        Args:
            reduction (str): Reduction method (currently only pca is supported)
            n_comps (int): Number of embedding components (default: 30)

        Returns:
            PC embeddings are stored in self.adatas
        """

        ref = self.adatas["ref"]
        if "X_pca" not in ref.obsm.keys():
            sc.tl.pca(ref, n_comps=n_comps)
            sc.pp.neighbors(ref)
            sc.tl.umap(ref)

        for k, adata in self.adatas.items():
            if k != "ref":
                sc.tl.ingest(adata, ref, embedding_method="pca")
            adata_pca = ad.AnnData(adata.obsm["X_pca"].copy(), obs=adata.obs.copy())
            self.adatas[k] = adata_pca

        self.reduction = reduction

    def evaluate(
        self,
        aggr_method: str = "logfc",
        metric: str = "pearson",
        perts: list | None = None,
        plot: bool = False,
        plot_size: tuple | None = None,
        return_df: bool = False,
        deg_key: str | None = None,
        n_top_genes: int = 100,
    ):
        """Evaluate predicted perturbation effect against reference perturbation effect

        Args:
            aggr_method (str): Method used to aggregate cells per perturbation (logFC, average, scaled avg, none, default: logFC). Set to none if using energy distance as the evaluation metric
            metric (str): Metric used to measure prediction accuracy (pearson, spearman, r2_score, dcor, cosine, mse, energy, default: pearson)
            perts (list): Subset of perturbations to evaluate (default: evaluate all perturbations)
            return_df (bool): If True, return evaluation dataframe
            plot (bool): If True, plot evaluation results as boxplots
            plot_size (tuple): Size of plot (default: automatically set depending on number of plots)
            deg_key: Key in `adata.uns` where differentially expressed genes are stored (default: None)
            n_top_genes (int): Number of top genes to plot if subsetting to differentially expressed genes (DEGs) (default: 100)

        Returns:
            None or pandas.DataFrame: Per-perturbation evaluation of model predictions
        """

        if metric not in [
            "pearson",
            # 'spearman',
            "r2_score",
            "mse",
            "rmse",
            "mae",
            "cosine",
            "energy",
        ]:
            raise ValueError("Metric not implemented")

        if deg_key is not None:
            deg_dict = self.ref_uns[deg_key]
            self.deg_dict = deg_dict
            self.use_degs[aggr_method][metric] = True
        else:
            self.use_degs[aggr_method][metric] = False

        if metric == "energy":
            try:
                from scperturb import edist_to_control
            except ImportError as exc:
                raise ValueError(
                    "Please install scperturb (https://github.com/sanderlab/scPerturb/) to use energy distance as an evaluation metric"
                ) from exc

            if aggr_method != "none":
                raise ValueError(
                    "Please set aggr_method to none to use energy distance as an evaluation metric"
                )

            if perts is None:
                perts = list(self.adatas["ref"].obs[self.pert_col].unique())

            perts = [x for x in perts if x != self.ctrl]
            for k in self.adatas:
                self.adatas[k] = self.adatas[k][
                    self.adatas[k].obs[self.pert_col].isin(perts)
                ]

            if self.cov_cols is not None:
                for k in self.adatas:
                    self.adatas[k].obs["cov_pert"] = merge_cols(
                        self.adatas[k].obs, self.cov_cols + [self.pert_col], delim="_"
                    )
            else:
                for k in self.adatas:
                    self.adatas[k].obs["cov_pert"] = self.adatas[k].obs[self.pert_col]

            adata_ref = self.adatas["ref"]
            pert_cov_unique = adata_ref.obs["cov_pert"].unique()

            if (deg_key is not None) and (deg_key not in adata_ref.uns.keys()):
                raise ValueError(
                    "Please run `sc.tl.rank_genes_groups` to compute differentially expressed genes"
                )

            edist_df_list = []
            for p in pert_cov_unique:
                pert_ad_list = []
                for k, v in self.adatas.items():
                    v = v[v.obs["cov_pert"] == p]
                    v.obs["model"] = k
                    pert_ad_list.append(v)

                pert_ad = ad.concat(pert_ad_list)
                if scipy.sparse.issparse(pert_ad.X):
                    pert_ad.X = pert_ad.X.toarray()

                if deg_key is not None:
                    p_degs = deg_dict[p]
                    pert_ad = pert_ad[:, p_degs[:n_top_genes]]

                pert_ad.obsm["X_edist"] = pert_ad.X
                pert_df = edist_to_control(
                    pert_ad,
                    obs_key="model",
                    obsm_key="X_edist",
                    control="ref",
                    verbose=False,
                )
                pert_df.columns = [p]
                edist_df_list.append(pert_df)

            evals = pd.concat(edist_df_list, axis=1)
            evals["model"] = evals.index
            evals = evals.melt(
                id_vars=["model"], var_name="cov_pert", value_name="metric"
            )
            evals = evals.loc[evals.model != "ref", :]
            evals = evals.loc[:, ["cov_pert", "model", "metric"]]
            self.evals[aggr_method] = {}

        else:
            if aggr_method not in self.aggr.keys():
                self.aggregate(aggr_method=aggr_method)

            aggr_ref = self.aggr[aggr_method]["ref"]
            ref_perts = list(aggr_ref.obs_names)

            if deg_key is not None:
                deg_dict = self.deg_dict
            else:
                deg_dict = None

            evals = []
            for k, aggr in self.aggr[aggr_method].items():
                if k not in ["ref", "target"]:
                    perts = list(set(aggr.obs_names).intersection(ref_perts))
                    if len(perts) < len(ref_perts):
                        missing_perts = [x for x in ref_perts if x not in perts]
                        print(
                            "Warning: missing perturbations for model %s: %s"
                            % (k, str(missing_perts))
                        )

                    scores = compare_perts(
                        aggr.to_df(),
                        aggr_ref.to_df(),
                        perts=perts,
                        metric=metric,
                        deg_dict=deg_dict,
                    )
                    df = pd.DataFrame(
                        index=perts,
                        data={
                            "cov_pert": perts,
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
        melt=False,
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
        title_size: float = 16,
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

        for ax, model in zip(axs[0], models):
            ax.set_title(model)

        for ax, pert in zip(axs[:, 0], perts):
            ax.set_ylabel(pert, size="large")

        # fig.supxlabel(x_title, size=axis_title_size)
        # fig.supylabel(y_title, size=axis_title_size)
        plt.show()

    def summary_plots(
        self,
        aggr_method: str = "logfc",
        metrics: str | list[str] = ["pearson"],
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

    def evaluate_pairwise(
        self,
        aggr_method: str,
        metric: str,
        models: list | None = None,
        deg_key: str | None = None,
        verbose: bool = False,
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

        if deg_key is not None:
            if self.deg_dict is not None:
                deg_dict = self.deg_dict
            else:
                deg_dict = self.ref_uns[deg_key]

            if self.pairwise_deg_dict is None:
                pairwise_deg_dict = {}
                if (self.cov_cols is None) or (len(self.cov_cols) == 0):
                    for p1 in ref_aggr.obs_names:
                        for p2 in ref_aggr.obs_names:
                            genes1 = deg_dict[p1]
                            genes2 = deg_dict[p2]
                            pairwise_deg_dict[frozenset([p1, p2])] = list(
                                set(genes1).union(genes2)
                            )
                else:
                    for cov in ref_aggr.obs["cov_merged"].cat.categories:
                        ref_aggr_cov = ref_aggr[ref_aggr.obs["cov_merged"] == cov, :]
                        pairwise_deg_dict[cov] = {}
                        for p1 in ref_aggr_cov.obs_names:
                            for p2 in ref_aggr_cov.obs_names:
                                genes1 = deg_dict[p1]
                                genes2 = deg_dict[p2]
                                pairwise_deg_dict[cov][frozenset([p1, p2])] = list(
                                    set(genes1).union(genes2)
                                )
                self.pairwise_deg_dict = pairwise_deg_dict

            else:
                pairwise_deg_dict = self.pairwise_deg_dict

        else:
            pairwise_deg_dict = None

        for model_name in models:
            if model_name not in self.aggr[aggr_method].keys():
                raise ValueError("Model %s not found" % model_name)

            aggr = self.aggr[aggr_method][model_name]
            aggr = aggr[list(set(aggr.obs_names).intersection(ref_aggr.obs_names)), :]

            if (self.cov_cols is not None) and (len(self.cov_cols) > 0):
                cov_unique = ref_aggr.obs["cov_merged"].cat.categories
                mat_dict = {}
                for cov in cov_unique:
                    if verbose:
                        print(cov)
                    aggr_cov_df = aggr[aggr.obs["cov_merged"] == cov, :].to_df()
                    ref_aggr_cov_df = ref_aggr[
                        ref_aggr.obs["cov_merged"] == cov, :
                    ].to_df()

                    idx_common = list(
                        set(aggr_cov_df.index).intersection(ref_aggr_cov_df.index)
                    )
                    aggr_cov_df = aggr_cov_df.loc[idx_common, :]
                    ref_aggr_cov_df = ref_aggr_cov_df.loc[idx_common, :]

                    if pairwise_deg_dict is not None:
                        pairwise_deg_dict_cov = pairwise_deg_dict[cov]
                    else:
                        pairwise_deg_dict_cov = None

                    mat = pairwise_metric_helper(
                        aggr_cov_df,
                        df2=ref_aggr_cov_df,
                        metric=metric,
                        pairwise_deg_dict=pairwise_deg_dict_cov,
                        verbose=verbose,
                    )
                    mat_dict[cov] = mat
                self.pairwise_evals[aggr_method][metric][model_name] = mat_dict

            else:
                aggr_df = aggr.to_df()
                ref_aggr_df = ref_aggr.to_df()

                idx_common = list(set(aggr_df.index).intersection(ref_aggr_df.index))
                aggr_df = aggr_df.loc[idx_common, :]
                ref_aggr_df = ref_aggr_df.loc[idx_common, :]

                mat = pairwise_metric_helper(
                    aggr_df,
                    df2=ref_aggr_df,
                    metric=metric,
                    pairwise_deg_dict=pairwise_deg_dict,
                    verbose=verbose,
                )
                self.pairwise_evals[aggr_method][metric][model_name] = mat

    def evaluate_rank(
        self,
        aggr_method: str,
        metric: str,
        models: list | None = None,
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

        if metric in ["pearson", "spearman", "dcor", "r2_score", "cosine"]:
            metric_type = "similarity"
        elif metric in ["mse", "rmse", "mae", "energy"]:
            metric_type = "distance"
        else:
            raise ValueError("Invalid metric")

        ref_aggr = self.aggr[aggr_method]["ref"]
        rank_df_list = []
        for model in models:
            if (self.cov_cols is not None) and (len(self.cov_cols) > 0):
                cov_unique = ref_aggr.obs["cov_merged"].cat.categories
                for cov in cov_unique:
                    mat = self.pairwise_evals[aggr_method][metric][model][cov]
                    cov_ranks = rank_helper(mat, metric_type=metric_type)
                    cov_df = pd.DataFrame(
                        {
                            "model": [model] * len(cov_ranks),
                            "cov_pert": cov_ranks.index,
                            "rank": cov_ranks.values,
                        }
                    )
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
                rank_df_list.append(df)

        rank_df = pd.concat(rank_df_list)
        self.rank_evals[aggr_method][metric] = rank_df

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
        for model_name, aggr_dict in ev.aggr.items():
            if model_name not in ev_anchor.aggr:
                ev_anchor.aggr[model_name] = aggr_dict
            else:
                for aggr_method, aggr_adata in aggr_dict.items():
                    if aggr_method not in ev_anchor.aggr[model_name]:
                        ev_anchor.aggr[model_name][aggr_method] = aggr_adata
                    else:
                        aggr_ids_merge = [
                            x
                            for x in aggr_adata.obs_names
                            if x
                            not in list(
                                ev_anchor.aggr[model_name][aggr_method].obs_names
                            )
                        ]

                        if len(aggr_ids_merge) > 0:
                            ev_anchor.aggr[model_name][aggr_method] = ad.concat(
                                [
                                    ev_anchor.aggr[model_name][aggr_method],
                                    aggr_adata[aggr_ids_merge],
                                ],
                            )

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
