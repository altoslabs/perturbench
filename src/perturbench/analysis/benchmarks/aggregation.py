import scanpy as sc
import pandas as pd
import numpy as np
import scipy
from scipy.sparse import csr_matrix, issparse
from ..utils import merge_cols


def average_adata(
    adata, cols, delim="_", mode="average", scale=False, return_h5=False, max_scale=10
):
    """Compute average expression values"""

    for col in cols:
        assert col in adata.obs.columns

    if scale:
        adata = sc.pp.scale(adata, copy=True, max_value=max_scale)

    avg_col = merge_cols(adata.obs, cols, delim=delim)
    avg_df = pd.DataFrame(0.0, columns=adata.var_names, index=avg_col.cat.categories)
    for i, group in enumerate(avg_col.cat.categories):
        if mode == "average":
            avg_df.iloc[i] = adata[avg_col.isin([group]), :].X.mean(0)
        elif mode == "sum":
            avg_df.iloc[i] = adata[avg_col.isin([group]), :].X.sum(0)

    if return_h5:
        avg_ad = sc.AnnData(avg_df)
        avg_ad.X = avg_ad.X.astype("float32")
        for i, col in enumerate(cols):
            avg_ad.obs[col] = [x.split(delim)[i] for x in avg_ad.obs_names]
            avg_ad.obs[col] = avg_ad.obs[col].astype("category")

        avg_ad.var = adata.var.copy()
        return avg_ad

    else:
        return avg_df


def var_adata(adata, cols, delim="_", adjust_var=True, return_h5=False):
    """Compute gene-wise variance within each group"""

    for col in cols:
        assert col in adata.obs.columns

    var_col = merge_cols(adata.obs, cols, delim=delim)

    var_df = pd.DataFrame(columns=adata.var_names, index=var_col.cat.categories)
    for group in var_col.cat.categories:
        if scipy.sparse.issparse(adata.X):
            raw_var = adata[var_col.isin([group]), :].X.A.var(0)
        else:
            raw_var = adata[var_col.isin([group]), :].X.var(0)

        if adjust_var:
            var_df.loc[group] = np.sqrt(raw_var) / adata[
                var_col.isin([group]), :
            ].X.mean(0)
        else:
            var_df.loc[group] = raw_var

    var_df = var_df.fillna(0.0)
    if return_h5:
        var_ad = sc.AnnData(var_df)
        for i, col in enumerate(cols):
            var_ad.obs[col] = [x.split(delim)[i] for x in var_ad.obs_names]
            var_ad.obs[col] = var_ad.obs[col].astype("category")

        return var_ad

    else:
        return var_df


def logFC_helper(adata, pert_col, delim, pseudocount, ctrl=None, exp=False):
    """Compute average log2 fold-changes against control"""
    unique_perts = adata.obs[pert_col].cat.categories

    if exp:
        adata_exp = adata.copy()
        if not issparse(adata_exp.X):
            adata_exp.X = csr_matrix(adata_exp.X)
            adata_exp.X.eliminate_zeros()
        adata_exp.X.data = np.expm1(adata_exp.X.data)
    else:
        adata_exp = adata

    if ctrl is not None:
        avg = average_adata(adata_exp, cols=[pert_col], delim=delim, return_h5=False)
        unique_perts = [x for x in unique_perts if x != ctrl]

    logfc = pd.DataFrame(0.0, index=unique_perts, columns=adata_exp.var_names)
    for i, p in enumerate(unique_perts):
        if ctrl is not None:
            avg_p = avg.loc[p]
            avg_ctrl = avg.loc[ctrl]
        else:
            avg_p = adata_exp[adata_exp.obs[pert_col] == p, :].X.mean(0)
            avg_ctrl = adata_exp[adata_exp.obs[pert_col] != p, :].X.mean(0)

        avg_p = np.log2(avg_p + pseudocount)
        avg_ctrl = np.log2(avg_ctrl + pseudocount)

        logfc.iloc[i] = avg_p - avg_ctrl

    logfc.fillna(0.0, inplace=True)
    return logfc


def logFC_adata(
    adata,
    pert_col,
    ctrl=None,
    cov_cols=[],
    delim="_",
    pseudocount=0.1,
    exp=False,
    return_h5=False,
):
    """Compute average log2 fold-changes per perturbation/covariate"""
    assert pert_col in adata.obs.columns

    if ctrl is not None:
        assert ctrl in adata.obs[pert_col].cat.categories

    if len(cov_cols) > 0:
        for col in cov_cols:
            assert col in adata.obs.columns
        covs = merge_cols(adata.obs, cov_cols, delim=delim)

    if len(cov_cols) <= 0:
        logfc_full = logFC_helper(
            adata, pert_col, delim, pseudocount=pseudocount, ctrl=ctrl, exp=exp
        )

    else:
        logfc_list = []
        for cov in covs.cat.categories:
            logfc = logFC_helper(
                adata[covs == cov, :],
                pert_col,
                delim,
                pseudocount=pseudocount,
                ctrl=ctrl,
                exp=exp,
            )
            logfc.index = str(cov) + delim + logfc.index.astype(str)
            logfc_list.append(logfc)

        logfc_full = pd.concat(logfc_list)

    if return_h5:
        logfc_ad = sc.AnnData(logfc_full)
        cols = cov_cols + [pert_col]
        for i, col in enumerate(cols):
            logfc_ad.obs[col] = [x.split(delim)[i] for x in logfc_ad.obs_names]
            logfc_ad.obs[col] = logfc_ad.obs[col].astype("category")
        return logfc_ad

    else:
        return logfc_full


def logp_helper(
    adata,
    pert_col,
    ctrl,
    de_method,
    deg_key,
    min_p=1e-24,
):
    sc.tl.rank_genes_groups(
        adata, groupby=pert_col, reference=ctrl, method=de_method, key_added=deg_key
    )
    deg_df = sc.get.rank_genes_groups_df(adata, group=None, key=deg_key)
    deg_df["logp"] = -1 * np.log(deg_df.pvals + min_p)
    deg_df["logp"] = np.sign(deg_df.scores) * deg_df.logp
    logp = deg_df.pivot(index="group", columns="names", values="logp")
    return logp


def logp_adata(
    adata,
    pert_col,
    ctrl=None,
    cov_cols=[],
    de_method="t-test-overestim-var",
    delim="_",
    return_h5=False,
    min_p=1e-24,
):
    """Compute log(p-values) for each perturbation/covariate"""
    assert pert_col in adata.obs.columns
    adata.obs[pert_col] = adata.obs[pert_col].astype("category")

    if ctrl is not None:
        assert ctrl in adata.obs[pert_col].cat.categories

    if len(cov_cols) > 0:
        for col in cov_cols:
            assert col in adata.obs.columns
        covs = merge_cols(adata.obs, cov_cols, delim=delim)

    deg_key = pert_col + delim + de_method
    if len(cov_cols) <= 0:
        logp_full = logp_helper(
            adata,
            pert_col,
            ctrl,
            de_method,
            deg_key,
            min_p=min_p,
        )

    else:
        logp_list = []
        for cov in covs.cat.categories:
            adata_cov = adata[covs == cov].copy()
            logp = logp_helper(
                adata_cov,
                pert_col,
                ctrl,
                de_method,
                deg_key,
                min_p=min_p,
            )
            logp.index = str(cov) + delim + logp.index.astype(str)
            logp_list.append(logp)

        logp_full = pd.concat(logp_list)

    if return_h5:
        logp_ad = sc.AnnData(logp_full)
        cols = cov_cols + [pert_col]
        for i, col in enumerate(cols):
            logp_ad.obs[col] = [x.split(delim)[i] for x in logp_ad.obs_names]
            logp_ad.obs[col] = logp_ad.obs[col].astype("category")
        return logp_ad

    else:
        return logp_full


def aggr_helper(
    adata,
    aggr_method,
    pert_col,
    cov_cols,
    ctrl,
    pseudocount,
    delim,
    adjust_var,
    de_method,
    **kwargs,
):
    """Aggregate adata"""
    adata.obs[pert_col] = adata.obs[pert_col].astype("category")

    if aggr_method == "average":
        agg_adata = average_adata(
            adata,
            cols=cov_cols + [pert_col],
            scale=False,
            delim=delim,
            return_h5=True,
            **kwargs,
        )
    elif aggr_method == "scaled":
        agg_adata = average_adata(
            adata,
            cols=cov_cols + [pert_col],
            scale=True,
            delim=delim,
            return_h5=True,
            **kwargs,
        )
    elif aggr_method == "logfc":
        agg_adata = logFC_adata(
            adata,
            pert_col=pert_col,
            ctrl=ctrl,
            cov_cols=cov_cols,
            pseudocount=pseudocount,
            delim=delim,
            return_h5=True,
            **kwargs,
        )

    elif aggr_method == "var":
        agg_adata = var_adata(
            adata,
            cols=cov_cols + [pert_col],
            delim=delim,
            return_h5=True,
            adjust_var=adjust_var,
            **kwargs,
        )

    elif aggr_method == "logp":
        agg_adata = logp_adata(
            adata,
            pert_col=pert_col,
            ctrl=ctrl,
            cov_cols=cov_cols,
            de_method=de_method,
            delim=delim,
            return_h5=True,
            **kwargs,
        )

    if cov_cols is not None and len(cov_cols) > 0:
        agg_adata.obs["cov_merged"] = merge_cols(agg_adata.obs, cov_cols, delim=delim)

    agg_adata = agg_adata[agg_adata.obs[pert_col] != ctrl, :]
    return agg_adata
