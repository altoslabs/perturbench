import scanpy as sc
import pandas as pd
import numpy as np
import scipy
from scipy.sparse import csr_matrix, issparse
import anndata as ad
from pandas.api.types import is_numeric_dtype

from ..utils import merge_cols
from ._rank_genes_helpers import rank_genes_groups_control_var

import warnings
from pandas.errors import PerformanceWarning
from anndata import ImplicitModificationWarning
warnings.filterwarnings('ignore', category=PerformanceWarning)
warnings.filterwarnings('ignore', category=ImplicitModificationWarning)

def _average_adata(
    adata, 
    cols, 
    delim='_', 
    mode='average', 
    scale=False, 
    return_h5=False, 
    max_scale=10
):
    """Compute average expression values"""
    
    for col in cols:
        assert col in adata.obs.columns
    
    if scale:
        adata = sc.pp.scale(adata, copy=True, max_value=max_scale)
    
    avg_col = merge_cols(adata.obs, cols, delim=delim)
    avg_df = pd.DataFrame(0.0, 
                          columns=adata.var_names, 
                          index=avg_col.cat.categories)                                                                                                 
    for i,group in enumerate(avg_col.cat.categories):
        if mode == 'average':
            avg_df.iloc[i] = adata[avg_col.isin([group]),:].X.mean(0)
        elif mode == 'sum':
            avg_df.iloc[i] = adata[avg_col.isin([group]),:].X.sum(0)
    
    if return_h5:
        avg_ad = sc.AnnData(avg_df)
        avg_ad.X = avg_ad.X.astype('float32')
        for i,col in enumerate(cols):
            avg_ad.obs[col] = [x.split(delim)[i] for x in avg_ad.obs_names]
            avg_ad.obs[col] = avg_ad.obs[col].astype('category')
            
        avg_ad.var = adata.var.copy()
        return(avg_ad)
    
    else:
        return(avg_df)


def _var_adata(adata, cols, delim='_', adjust_var=True, return_h5=False):
    """Compute gene-wise variance within each group"""
    
    for col in cols:
        assert col in adata.obs.columns
    
    var_col = merge_cols(adata.obs, cols, delim=delim)
    
    var_df = pd.DataFrame(columns=adata.var_names, index=var_col.cat.categories)                                                                                                 
    for group in var_col.cat.categories:
        if scipy.sparse.issparse(adata.X):
            raw_var = adata[var_col.isin([group]),:].X.A.var(0)
        else:
            raw_var = adata[var_col.isin([group]),:].X.var(0)
            
        if adjust_var:
            var_df.loc[group] = np.sqrt(raw_var)/adata[var_col.isin([group]),:].X.mean(0)
        else:
            var_df.loc[group] = raw_var
                
    var_df = var_df.fillna(0.0)
    if return_h5:
        var_ad = sc.AnnData(var_df)
        for i,col in enumerate(cols):
            var_ad.obs[col] = [x.split(delim)[i] for x in var_ad.obs_names]
            var_ad.obs[col] = var_ad.obs[col].astype('category')
        
        return(var_ad)
    
    else:
        return(var_df)


def _logfc_helper(adata, pert_col, delim, pseudocount, ctrl=None, exp=False):
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
        avg = _average_adata(
            adata_exp, 
            cols=[pert_col],
            delim=delim,
            return_h5=False,
        )
        unique_perts = [x for x in unique_perts if x != ctrl]
    
    logfc = pd.DataFrame(0.0, index=unique_perts, columns=adata_exp.var_names)
    for i,p in enumerate(unique_perts):
        if ctrl is not None:
            avg_p = avg.loc[p]
            avg_ctrl = avg.loc[ctrl]
        else:
            avg_p = adata_exp[adata_exp.obs[pert_col] == p,:].X.mean(0)
            avg_ctrl = adata_exp[adata_exp.obs[pert_col] != p,:].X.mean(0)

        avg_p = np.log2(avg_p + pseudocount)
        avg_ctrl = np.log2(avg_ctrl + pseudocount)
            
        logfc.iloc[i] = avg_p - avg_ctrl
    
    logfc.fillna(0.0, inplace=True)
    return(logfc)


def _differential_expression_helper(
    adata,
    pert_col,
    ctrl,
    de_method,
    deg_key,
    use_control_variance=False,
):
    if use_control_variance:
        rank_genes_groups_control_var(
            adata, 
            groupby=pert_col, 
            reference=ctrl, 
            method=de_method, 
            key_added=deg_key,
        )
    else:
        sc.tl.rank_genes_groups(
            adata, 
            groupby=pert_col, 
            reference=ctrl, 
            method=de_method, 
            key_added=deg_key,
        )
    deg_df = sc.get.rank_genes_groups_df(adata, group=None, key=deg_key)
    if 'group' not in deg_df.columns:
        unique_groups = [x for x in adata.obs[pert_col].unique().tolist() if x != ctrl]
        assert len(unique_groups) == 1
        deg_df['group'] = unique_groups[0]
    
    return(deg_df)


def _pca_helper(adata, ref_adata, n_comps=30, pca_model=None):
    """Apply PCA dimensionality reduction by projecting onto reference PCs"""
    from ..embed_cells import train_pca, embed_pca
    
    # Compute PCA on reference if not already done
    if pca_model is None:
        pca_model = train_pca(ref_adata, n_comps=n_comps)
    
    X_pca = embed_pca(adata, pca_model)
    adata_pca = ad.AnnData(X_pca, obs=adata.obs.copy())
    
    return adata_pca


def aggregate_adata(
    adata, 
    aggr_method, 
    pert_col, 
    cov_cols, 
    ctrl, 
    pseudocount=0.1,
    delim='_',
    adjust_var=False,
    de_method='t-test_overestim_var',
    use_control_variance=False,
    ref_adata=None,
    n_comps=30,
    pca_model=None,
    **kwargs
):
    """Aggregate adata and return dictionary with perturbation/covariate keys"""
    adata.obs[pert_col] = adata.obs[pert_col].astype('category')
    
    if ctrl is not None:
        assert ctrl in adata.obs[pert_col].cat.categories
    
    if len(cov_cols) == 0:
        adata.obs['_dummy_cov'] = '1'
        cov_cols = ['_dummy_cov']
    
    for col in cov_cols:
        assert col in adata.obs.columns
    covs = merge_cols(adata.obs, cov_cols, delim=delim)

    # Identify categorical vs continuous covariates for control matching
    # Controls are matched only on categorical covariates (not continuous ones like dose/time)
    categorical_cov_cols = [c for c in cov_cols if not is_numeric_dtype(adata.obs[c])]

    # Create categorical-only covariate column for control matching
    if len(categorical_cov_cols) > 0:
        categorical_covs = merge_cols(adata.obs, categorical_cov_cols, delim=delim)
    else:
        # No categorical covariates - all cells share the same control pool
        categorical_covs = pd.Series('_all_', index=adata.obs.index).astype('category')

    # Build mapping: full_cov -> categorical_cov for control lookup
    cov_to_categorical = {}
    for cov in covs.cat.categories:
        # Find the first row with this covariate combo to get categorical values
        idx = (covs == cov).idxmax()
        cov_to_categorical[cov] = categorical_covs[idx]
    
    # Handle PCA aggregation separately
    if aggr_method in ['pca', 'pca_average']:
        if ref_adata is None:
            raise ValueError("ref_adata must be provided for PCA aggregation")
        pca_adata = _pca_helper(adata, ref_adata, n_comps=n_comps, pca_model=pca_model)
        
    result_dict = {}
    for cov in covs.cat.categories:
        cov_adata_full = adata[covs == cov, :]
        # Skip covariate combinations with only control cells (no perturbations to aggregate)
        cov_perts = cov_adata_full.obs[pert_col].unique()
        non_ctrl_perts = [p for p in cov_perts if p != ctrl]
        if len(non_ctrl_perts) == 0:
            continue  # Skip control-only covariate combinations

        result_dict[cov] = {}
        if len(cov_perts) > 1:  # Has both controls and perturbations
            ## Handle reductions that result in a matrix of cell expression per perturbation
            if aggr_method in ['pca', 'pca_average', 'none']:
                if aggr_method in ['pca', 'pca_average']:
                    cov_pca_adata = pca_adata[covs == cov,:]
                    for pert_name in cov_pca_adata.obs[pert_col].unique():
                        if pert_name != ctrl:
                            pert_pca_adata = cov_pca_adata[cov_pca_adata.obs[pert_col] == pert_name,:]
                            if aggr_method == 'pca_average':
                                result_dict[cov][pert_name] = pert_pca_adata.X.toarray().mean(0)
                            elif aggr_method == 'pca':
                                result_dict[cov][pert_name] = pert_pca_adata.X.toarray()

                elif aggr_method == 'none':
                    cov_adata = adata[covs == cov,:]
                    for pert_name in cov_adata.obs[pert_col].unique():
                        if pert_name != ctrl:
                            pert_adata = cov_adata[cov_adata.obs[pert_col] == pert_name,:]
                            result_dict[cov][pert_name] = pert_adata.X
            
            ## Reductions that result in an aggregate vector per perturbation
            else:
                if aggr_method == 'average':
                    aggr_cov = _average_adata(
                        adata[covs == cov,:], 
                        cols=[pert_col], 
                        delim=delim,
                        return_h5=False,
                        **kwargs,
                    )
                elif aggr_method == 'scaled':
                    aggr_cov = _average_adata(
                        adata[covs == cov,:], 
                        cols=[pert_col], 
                        delim=delim,
                        scale=True,
                        return_h5=False,
                        **kwargs,
                    )
                elif aggr_method == 'logfc':
                    # For logfc, combine perturbed cells (exact cov match) with controls (categorical match)
                    categorical_cov = cov_to_categorical[cov]
                    perturbed_mask = (covs == cov) & (adata.obs[pert_col] != ctrl)
                    control_mask = (categorical_covs == categorical_cov) & (adata.obs[pert_col] == ctrl)
                    subset_adata = adata[perturbed_mask | control_mask, :]

                    aggr_cov = _logfc_helper(
                        subset_adata,
                        pert_col=pert_col,
                        delim=delim,
                        pseudocount=pseudocount,
                        ctrl=ctrl,
                        **kwargs,
                    )
                elif aggr_method == 'var':
                    aggr_cov = _var_adata(
                        adata[covs == cov,:],
                        cols=[pert_col],
                        delim=delim,
                        adjust_var=adjust_var,
                        return_h5=False,
                        **kwargs,
                    )
                
                elif aggr_method in ['scores', 'pvals', 'logp']:
                    # For DE analysis, combine perturbed cells (exact cov match) with controls (categorical match)
                    categorical_cov = cov_to_categorical[cov]
                    perturbed_mask = (covs == cov) & (adata.obs[pert_col] != ctrl)
                    control_mask = (categorical_covs == categorical_cov) & (adata.obs[pert_col] == ctrl)
                    subset_adata = adata[perturbed_mask | control_mask, :]

                    deg_df = _differential_expression_helper(
                        subset_adata,
                        pert_col=pert_col,
                        ctrl=ctrl,
                        de_method=de_method,
                        deg_key=pert_col + delim + de_method,
                        use_control_variance=use_control_variance,
                        **kwargs,
                    )
                    inf_rows = (deg_df.scores == np.inf) | (deg_df.scores == -np.inf)
                    deg_df.loc[inf_rows, 'scores'] = 0.0
                    deg_df.loc[inf_rows, 'pvals'] = 1.0
                    deg_df.loc[inf_rows, 'pvals_adj'] = 1.0
                    

                    if np.sum(inf_rows)/len(inf_rows) > 0.5:
                        warnings.warn(
                            f'{cov} has a large fraction ({np.sum(inf_rows)/len(inf_rows)}) of inf t-scores',
                            stacklevel=1
                        )
                    
                    if aggr_method == 'scores':
                        aggr_cov = deg_df.pivot(index='group', columns='names', values='scores')
                    elif aggr_method == 'pvals':
                        aggr_cov = deg_df.pivot(index='group', columns='names', values='pvals')
                    elif aggr_method == 'logp':
                        deg_df['pval_transformed'] = -1*np.log(deg_df['pvals_adj'] + 1e-37)
                        deg_df['pval_transformed'] = np.sign(deg_df.scores)*deg_df.pval_transformed
                        aggr_cov = deg_df.pivot(index='group', columns='names', values='pval_transformed')
                
                # Convert DataFrame to dictionary with proper key naming
                for pert_name in aggr_cov.index:
                    if pert_name != ctrl:
                        # Get the expression vector, ensuring it's 1D
                        expr_vector = aggr_cov.loc[pert_name].values
                        result_dict[cov][pert_name] = expr_vector
    
    return result_dict
