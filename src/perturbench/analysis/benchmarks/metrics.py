from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import numpy as np
from numpy.linalg import norm
import pandas as pd
import tqdm


def compute_metric(x, y, metric):
    """Compute specified similarity/distance metric between x and y vectors"""

    if metric == "pearson":
        score = pearsonr(x, y)[0]
    # elif metric == 'spearman':
    #     score = spearmanr(x, y)[0]
    elif metric == "r2_score":
        score = r2_score(x, y)
    elif metric == "cosine":
        score = np.dot(x, y) / (norm(x) * norm(y))
    elif metric == "mse":
        score = np.mean(np.square(x - y))
    elif metric == "rmse":
        score = np.sqrt(np.mean(np.square(x - y)))
    elif metric == "mae":
        score = np.mean(np.abs(x - y))

    return score


def compare_perts(
    pred, ref, features=None, perts=None, metric="pearson", deg_dict=None
):
    """Compare expression similarities between `pred` and `ref` DataFrames using the specified metric"""

    if perts is None:
        perts = list(set(pred.index).intersection(ref.index))
        assert len(perts) > 0
    else:
        perts = list(perts)

    if features is not None:
        pred = pred.loc[:, features]
        ref = ref.loc[:, features]

    pred = pred.loc[perts, :]
    ref = ref.loc[perts, :]

    pred = pred.replace([np.inf, -np.inf], 0)
    ref = ref.replace([np.inf, -np.inf], 0)

    eval_metric = []
    for p in perts:
        if deg_dict is not None:
            genes = deg_dict[p]
        else:
            genes = ref.columns

        eval_metric.append(
            compute_metric(pred.loc[p, genes], ref.loc[p, genes], metric)
        )

    eval_scores = pd.Series(index=perts, data=eval_metric)
    return eval_scores


def pairwise_metric_helper(
    df,
    df2=None,
    metric="rmse",
    pairwise_deg_dict=None,
    verbose=False,
):
    if df2 is None:
        df2 = df

    mat = pd.DataFrame(0.0, index=df.index, columns=df2.index)
    for p1 in tqdm.tqdm(df.index, disable=not verbose):
        for p2 in df2.index:
            if pairwise_deg_dict is not None:
                pp = frozenset([p1, p2])
                genes_ix = pairwise_deg_dict[pp]

                m = compute_metric(
                    df.loc[p1, genes_ix],
                    df2.loc[p2, genes_ix],
                    metric=metric,
                )
            else:
                m = compute_metric(
                    df.loc[p1],
                    df2.loc[p2],
                    metric=metric,
                )
            mat.at[p1, p2] = m

    return mat


def rank_helper(pred_ref_mat, metric_type):
    rel_ranks = pd.Series(1.0, index=pred_ref_mat.columns)
    for p in pred_ref_mat.columns:
        pred_metrics = pred_ref_mat.loc[:, p]
        pred_metrics = pred_metrics.sample(frac=1.0)  ## Shuffle to avoid ties
        if metric_type == "distance":
            pred_metrics = pred_metrics.sort_values(ascending=True)
        elif metric_type == "similarity":
            pred_metrics = pred_metrics.sort_values(ascending=False)
        else:
            raise ValueError(
                "Invalid metric_type, should be either distance or similarity"
            )

        rel_ranks.loc[p] = np.where(pred_metrics.index == p)[0][0]

    rel_ranks = rel_ranks / len(rel_ranks)
    return rel_ranks
