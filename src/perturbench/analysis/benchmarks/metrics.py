from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import numpy as np
from numpy.linalg import norm
import pandas as pd
import tqdm
import matplotlib.pyplot as plt
import torch
from geomloss import SamplesLoss


def top_k_recall(x: np.ndarray, y: np.ndarray, k: int = 20) -> float:
    """
    Compute the top k recall between x and y where x is the reference and y is the prediction
    """
    x = np.argsort(np.abs(x))[::-1]
    y = np.argsort(np.abs(y))[::-1]
    
    recall = len(set(x[:k]).intersection(set(y[:k]))) / k
    return recall

def compute_metric(x, y, metric):
    """Compute specified similarity/distance metric between x and y vectors"""
    if metric == 'mmd':
        if x.ndim == 1 or y.ndim == 1:
            raise ValueError('MMD requires 2D arrays - check your aggregation method (must be pca or none)')
        if type(x) is not np.ndarray:
            x = x.toarray()
        if type(y) is not np.ndarray:
            y = y.toarray()
        score_fn = SamplesLoss(loss="energy", p=2)
        score = score_fn(torch.from_numpy(x).float(), torch.from_numpy(y).float()).numpy()
        score = float(score)
    
    else:
        if x.ndim == 2 or y.ndim == 2:
            raise ValueError('Metrics require 1D arrays - check your aggregation method')
        
        if metric == 'pearson':
            score = pearsonr(x, y)[0]
        elif metric == 'r2_score':
            score = r2_score(x, y)
        elif metric == 'cosine':
            score = np.dot(x, y)/(norm(x)*norm(y))
        elif metric == 'mse':
            score = np.mean(np.square(x - y))
        elif metric == 'rmse':
            score = np.sqrt(np.mean(np.square(x - y)))
        elif metric == 'mae':
            score = np.mean(np.abs(x - y))
        elif metric == 'top_k_recall':
            score = top_k_recall(x, y, k=50)
        else:
            raise ValueError(f"Unknown metric: {metric}. Supported metrics: pearson, r2_score, cosine, mse, rmse, mae, top_k_recall")
    
    return(score)


def compare_perts(pred_aggr_dict, ref_aggr_dict, perts=None, metric='pearson', deg_mask=None):
    """Compare expression similarities between `pred` and `ref` DataFrames using the specified metric"""
    
    if perts is None:
        perts = list(set(pred_aggr_dict.keys()).intersection(ref_aggr_dict.keys()))
        if len(perts) == 0:
            raise ValueError('No perturbations in common between pred and ref')
    else:
        perts = list(perts)
    
    eval_metric = []
    for p in perts:
        pred_p = pred_aggr_dict[p]
        ref_p = ref_aggr_dict[p]
        
        if deg_mask is not None:
            if ref_p.ndim == 1:
                gene_idxs = deg_mask[p]
                ref_p = ref_p[gene_idxs]
                pred_p = pred_p[gene_idxs]
            else:
                gene_idxs = deg_mask[p]
                ref_p = ref_p[:,gene_idxs]
                pred_p = pred_p[:,gene_idxs]
        
        eval_metric.append(
            compute_metric(pred_p, ref_p, metric)
        )
    
    eval_scores = pd.Series(index=perts, data=eval_metric, dtype=np.float64)
    return(eval_scores)


def pairwise_metric_helper(
    aggr_dict1, 
    aggr_dict2=None,
    perts=None,
    metric='rmse', 
    verbose=False,
):
    if aggr_dict2 is None:
        aggr_dict2 = aggr_dict1
    
    if perts is not None:
        perts_unique1 = perts
        perts_unique2 = perts
    else:
        perts_unique1 = aggr_dict1.keys()
        perts_unique2 = aggr_dict2.keys()
    
    mat = pd.DataFrame(0.0, index=perts_unique1, columns=perts_unique2)
    for p1 in tqdm.tqdm(perts_unique1, disable=not verbose):
        for p2 in perts_unique2:
            m = compute_metric(
                aggr_dict1[p1],
                aggr_dict2[p2],
                metric=metric,
            )
            mat.at[p1,p2] = m
    
    return(mat)


def rank_helper(pred_ref_mat, metric_type):
    rel_ranks = pd.Series(1.0, index=pred_ref_mat.columns)
    for p in pred_ref_mat.columns:
        pred_metrics = pred_ref_mat.loc[:,p]
        pred_metrics = pred_metrics.sample(frac=1.0) ## Shuffle to avoid ties
        if metric_type == 'distance':
            pred_metrics = pred_metrics.sort_values(ascending=True)
        elif metric_type == 'similarity':
            pred_metrics = pred_metrics.sort_values(ascending=False)
        else:
            raise ValueError('Invalid metric_type, should be either distance or similarity')

        rel_ranks.loc[p] = np.where(pred_metrics.index == p)[0][0]

    if len(rel_ranks) > 1:
        rel_ranks = rel_ranks / (len(rel_ranks) - 1)
    return rel_ranks


def deg_pairwise_jaccard_similarity_helper(
    eval,  # an Evaluation objective
    model_name: str,
    k: int = 50,
    visualize_heatmaps: bool = False,
) -> np.array:

    # only works when aggr=scores and metrics=top_k_recall
    pred = np.argsort(eval.aggr['scores'][model_name].X, axis=-1)[:, ::-1][:, :k]
    truth = np.argsort(eval.aggr['scores']['ref'].X, axis=-1)[:, ::-1][:, :k]

    gene_names = eval.aggr['scores']['ref'].var_names
    obs_names = eval.aggr['scores']['ref'].obs_names
    deg_recalled_list = [set(gene_names[indices_pred]).intersection(gene_names[indices_truth])
           for indices_pred, indices_truth in zip(pred, truth, strict=True)]
    deg_recalled_pd = pd.DataFrame.from_dict({name: pd.Series(list(li)) for name, li in zip(obs_names, deg_recalled_list, strict=True)})
    deg_recalled_pd = deg_recalled_pd.T

    pairwise_jaccard_similarity = np.zeros((len(deg_recalled_pd), len(deg_recalled_pd)), dtype=np.float32)
    pairwise_jaccard_similarity.fill(np.nan)

    for i in range(len(deg_recalled_pd)):
        for j in range(i + 1, len(deg_recalled_pd)):
            pert_i = deg_recalled_pd.iloc[i][~pd.isna(deg_recalled_pd.iloc[i])].values
            pert_j = deg_recalled_pd.iloc[j][~pd.isna(deg_recalled_pd.iloc[j])].values
            pert_intersection = set(pert_i).intersection(pert_j)
            pert_union = set(pert_i).union(pert_j)
            if len(pert_union) > 0:
                pairwise_jaccard_similarity[i][j] = len(pert_intersection) / len(pert_union)

    if not visualize_heatmaps:
        return deg_recalled_pd, pairwise_jaccard_similarity
    else:
        f, ax = plt.subplots()
        f.set_figheight(80)
        f.set_figwidth(80)
        im = ax.imshow(pairwise_jaccard_similarity, vmin=0, vmax=1)
        cbar = ax.figure.colorbar(im, ax=ax, shrink=0.5)
        cbar.ax.tick_params(labelsize=18)
        # Show all ticks and label them with the respective list entries
        ax.set_xticks(range(len(deg_recalled_pd)), labels=deg_recalled_pd.index, rotation=45, va='bottom',
                      ha='left')
        ax.set_yticks(range(len(deg_recalled_pd)), ha="left", labels=deg_recalled_pd.index)
        ax.yaxis.tick_right()
        ax.xaxis.tick_top()
        ax.grid(visible=False, which='major')

        return deg_recalled_pd, pairwise_jaccard_similarity, f


