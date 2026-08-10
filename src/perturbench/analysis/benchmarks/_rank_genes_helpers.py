"""
Adapted from scanpy (https://github.com/scverse/scanpy) and 
modified to only use the variance from the control group.

BSD 3-Clause License

Copyright (c) 2017 F. Alexander Wolf, P. Angerer, Theis Lab
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
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

from __future__ import annotations
from typing import Literal

import numpy as np
import pandas as pd

import scanpy._utils as _utils
import scanpy.logging as logg
from scanpy._compat import old_positionals
from scanpy._utils import (
    check_nonnegative_integers,
)
from scanpy.get import _check_mask
from scanpy.tools._rank_genes_groups import (
    _RankGenes,
)

from collections.abc import Generator, Iterable
from anndata import AnnData
from numpy.typing import NDArray

_CorrMethod = Literal["benjamini-hochberg", "bonferroni"]
_Method = Literal["t-test", "t-test_overestim_var"]

class _RankGenesControlVar(_RankGenes):

    def t_test(
        self, method: Literal["t-test", "t-test_overestim_var"]
    ) -> Generator[tuple[int, NDArray[np.floating], NDArray[np.floating]], None, None]:
        from scipy import stats

        self._basic_stats()

        for group_index, (mask_obs, mean_group, var_group) in enumerate(  # noqa: B007
            zip(self.groups_masks_obs, self.means, self.vars, strict=True)
        ):
            if self.ireference is not None and group_index == self.ireference:
                continue

            ns_group = np.count_nonzero(mask_obs)

            if self.ireference is not None:
                mean_rest = self.means[self.ireference]
                var_rest = self.vars[self.ireference]
                ns_other = np.count_nonzero(self.groups_masks_obs[self.ireference])
            else:
                mean_rest = self.means_rest[group_index]
                var_rest = self.vars_rest[group_index]
                ns_other = self.X.shape[0] - ns_group

            # TODO: Come up with better solution. Mask unexpressed genes?
            # See https://github.com/scipy/scipy/issues/10269
            with np.errstate(invalid="ignore"):
                scores, pvals = stats.ttest_ind_from_stats(
                    mean1=mean_group,
                    std1=np.sqrt(var_rest),
                    nobs1=ns_group,
                    mean2=mean_rest,
                    std2=np.sqrt(var_rest),
                    nobs2=ns_other,
                    equal_var=True,
                )

            # I think it's only nan when means are the same and vars are 0
            scores[np.isnan(scores)] = 0
            # This also has to happen for Benjamini Hochberg
            pvals[np.isnan(pvals)] = 1

            yield group_index, scores, pvals


@old_positionals(
    "mask",
    "use_raw",
    "groups",
    "reference",
    "n_genes",
    "rankby_abs",
    "pts",
    "key_added",
    "copy",
    "method",
    "corr_method",
    "tie_correct",
    "layer",
)
def rank_genes_groups_control_var(
    adata: AnnData,
    groupby: str,
    *,
    mask_var: NDArray[np.bool_] | str | None = None,
    use_raw: bool | None = None,
    groups: Literal["all"] | Iterable[str] = "all",
    reference: str = "rest",
    n_genes: int | None = None,
    rankby_abs: bool = False,
    pts: bool = False,
    key_added: str | None = None,
    copy: bool = False,
    method: _Method | None = None,
    corr_method: _CorrMethod = "benjamini-hochberg",
    tie_correct: bool = False,
    layer: str | None = None,
    **kwds,
) -> AnnData | None:
    """\
    Rank genes for characterizing groups.

    Expects logarithmized data.

    Parameters
    ----------
    adata
        Annotated data matrix.
    groupby
        The key of the observations grouping to consider.
    mask_var
        Select subset of genes to use in statistical tests.
    use_raw
        Use `raw` attribute of `adata` if present.
    layer
        Key from `adata.layers` whose value will be used to perform tests on.
    groups
        Subset of groups, e.g. [`'g1'`, `'g2'`, `'g3'`], to which comparison
        shall be restricted, or `'all'` (default), for all groups. Note that if
        `reference='rest'` all groups will still be used as the reference, not
        just those specified in `groups`.
    reference
        If `'rest'`, compare each group to the union of the rest of the group.
        If a group identifier, compare with respect to this group.
    n_genes
        The number of genes that appear in the returned tables.
        Defaults to all genes.
    method
        The default method is `'t-test'`,
        `'t-test_overestim_var'` overestimates variance of each group,
        `'wilcoxon'` uses Wilcoxon rank-sum,
        `'logreg'` uses logistic regression. See :cite:t:`Ntranos2019`,
        `here <https://github.com/scverse/scanpy/issues/95>`__ and `here
        <https://www.nxn.se/valent/2018/3/5/actionable-scrna-seq-clusters>`__,
        for why this is meaningful.
    corr_method
        p-value correction method.
        Used only for `'t-test'`, `'t-test_overestim_var'`, and `'wilcoxon'`.
    tie_correct
        Use tie correction for `'wilcoxon'` scores.
        Used only for `'wilcoxon'`.
    rankby_abs
        Rank genes by the absolute value of the score, not by the
        score. The returned scores are never the absolute values.
    pts
        Compute the fraction of cells expressing the genes.
    key_added
        The key in `adata.uns` information is saved to.
    copy
        Whether to copy `adata` or modify it inplace.
    kwds
        Are passed to test methods. Currently this affects only parameters that
        are passed to :class:`sklearn.linear_model.LogisticRegression`.
        For instance, you can pass `penalty='l1'` to try to come up with a
        minimal set of genes that are good predictors (sparse solution meaning
        few non-zero fitted coefficients).

    Returns
    -------
    Returns `None` if `copy=False`, else returns an `AnnData` object. Sets the following fields:

    `adata.uns['rank_genes_groups' | key_added]['names']` : structured :class:`numpy.ndarray` (dtype `object`)
        Structured array to be indexed by group id storing the gene
        names. Ordered according to scores.
    `adata.uns['rank_genes_groups' | key_added]['scores']` : structured :class:`numpy.ndarray` (dtype `object`)
        Structured array to be indexed by group id storing the z-score
        underlying the computation of a p-value for each gene for each
        group. Ordered according to scores.
    `adata.uns['rank_genes_groups' | key_added]['logfoldchanges']` : structured :class:`numpy.ndarray` (dtype `object`)
        Structured array to be indexed by group id storing the log2
        fold change for each gene for each group. Ordered according to
        scores. Only provided if method is 't-test' like.
        Note: this is an approximation calculated from mean-log values.
    `adata.uns['rank_genes_groups' | key_added]['pvals']` : structured :class:`numpy.ndarray` (dtype `float`)
        p-values.
    `adata.uns['rank_genes_groups' | key_added]['pvals_adj']` : structured :class:`numpy.ndarray` (dtype `float`)
        Corrected p-values.
    `adata.uns['rank_genes_groups' | key_added]['pts']` : :class:`pandas.DataFrame` (dtype `float`)
        Fraction of cells expressing the genes for each group.
    `adata.uns['rank_genes_groups' | key_added]['pts_rest']` : :class:`pandas.DataFrame` (dtype `float`)
        Only if `reference` is set to `'rest'`.
        Fraction of cells from the union of the rest of each group
        expressing the genes.

    Notes
    -----
    There are slight inconsistencies depending on whether sparse
    or dense data are passed. See `here <https://github.com/scverse/scanpy/blob/main/scanpy/tests/test_rank_genes_groups.py>`__.

    Examples
    --------
    >>> import scanpy as sc
    >>> adata = sc.datasets.pbmc68k_reduced()
    >>> sc.tl.rank_genes_groups(adata, 'bulk_labels', method='wilcoxon')
    >>> # to visualize the results
    >>> sc.pl.rank_genes_groups(adata)
    """
    if mask_var is not None:
        mask_var = _check_mask(adata, mask_var, "var")

    if use_raw is None:
        use_raw = adata.raw is not None
    elif use_raw is True and adata.raw is None:
        raise ValueError("Received `use_raw=True`, but `adata.raw` is empty.")

    if method is None:
        method = "t-test"

    if "only_positive" in kwds:
        rankby_abs = not kwds.pop("only_positive")  # backwards compat

    start = logg.info("ranking genes")
    avail_methods = ['t-test', 't-test_overestim_var']
    if method not in avail_methods:
        raise ValueError(f"Method must be one of {avail_methods}.")

    avail_corr = {"benjamini-hochberg", "bonferroni"}
    if corr_method not in avail_corr:
        raise ValueError(f"Correction method must be one of {avail_corr}.")

    adata = adata.copy() if copy else adata
    _utils.sanitize_anndata(adata)
    # for clarity, rename variable
    if groups == "all":
        groups_order = "all"
    elif isinstance(groups, (str, int)):
        raise ValueError("Specify a sequence of groups")
    else:
        groups_order = list(groups)
        if isinstance(groups_order[0], int):
            groups_order = [str(n) for n in groups_order]
        if reference != "rest" and reference not in set(groups_order):
            groups_order += [reference]
    if reference != "rest" and reference not in adata.obs[groupby].cat.categories:
        cats = adata.obs[groupby].cat.categories.tolist()
        raise ValueError(
            f"reference = {reference} needs to be one of groupby = {cats}."
        )

    if key_added is None:
        key_added = "rank_genes_groups"
    adata.uns[key_added] = {}
    adata.uns[key_added]["params"] = dict(
        groupby=groupby,
        reference=reference,
        method=method,
        use_raw=use_raw,
        layer=layer,
        corr_method=corr_method,
    )

    test_obj = _RankGenesControlVar(
        adata,
        groups_order,
        groupby,
        mask_var=mask_var,
        reference=reference,
        use_raw=use_raw,
        layer=layer,
        comp_pts=pts,
    )

    if check_nonnegative_integers(test_obj.X) and method != "logreg":
        logg.warning(
            "It seems you use rank_genes_groups on the raw count data. "
            "Please logarithmize your data before calling rank_genes_groups."
        )

    # for clarity, rename variable
    n_genes_user = n_genes
    # make sure indices are not OoB in case there are less genes than n_genes
    # defaults to all genes
    if n_genes_user is None or n_genes_user > test_obj.X.shape[1]:
        n_genes_user = test_obj.X.shape[1]

    logg.debug(f"consider {groupby!r} groups:")
    logg.debug(f"with sizes: {np.count_nonzero(test_obj.groups_masks_obs, axis=1)}")

    test_obj.compute_statistics(
        method,
        corr_method=corr_method,
        n_genes_user=n_genes_user,
        rankby_abs=rankby_abs,
        tie_correct=tie_correct,
        **kwds,
    )

    if test_obj.pts is not None:
        groups_names = [str(name) for name in test_obj.groups_order]
        adata.uns[key_added]["pts"] = pd.DataFrame(
            test_obj.pts.T, index=test_obj.var_names, columns=groups_names
        )
    if test_obj.pts_rest is not None:
        adata.uns[key_added]["pts_rest"] = pd.DataFrame(
            test_obj.pts_rest.T, index=test_obj.var_names, columns=groups_names
        )

    test_obj.stats.columns = test_obj.stats.columns.swaplevel()

    dtypes = {
        "names": "O",
        "scores": "float32",
        "logfoldchanges": "float32",
        "pvals": "float64",
        "pvals_adj": "float64",
    }

    for col in test_obj.stats.columns.levels[0]:
        adata.uns[key_added][col] = test_obj.stats[col].to_records(
            index=False, column_dtypes=dtypes[col]
        )

    logg.info(
        "    finished",
        time=start,
        deep=(
            f"added to `.uns[{key_added!r}]`\n"
            "    'names', sorted np.recarray to be indexed by group ids\n"
            "    'scores', sorted np.recarray to be indexed by group ids\n"
            + (
                "    'logfoldchanges', sorted np.recarray to be indexed by group ids\n"
                "    'pvals', sorted np.recarray to be indexed by group ids\n"
                "    'pvals_adj', sorted np.recarray to be indexed by group ids"
                if method in {"t-test", "t-test_overestim_var", "wilcoxon"}
                else ""
            )
        ),
    )
    return adata if copy else None
