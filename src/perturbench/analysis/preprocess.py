import scanpy as sc
import anndata as ad
from .utils import merge_cols
import pandas as pd
import warnings


def differential_expression_by_covariate(
    adata,
    perturbation_key: str,
    perturbation_control_value: str,
    covariate_keys: list[str] = [],
    n_differential_genes=25,
    rankby_abs=True,
    key_added="rank_genes_groups_cov",
    return_dict=False,
    delim="_",
):
    """Finds the top `n_differential_genes` differentially expressed genes for
       each perturbation. The differential expression is run separately for
       each unique covariate category in the dataset.

    Args:
        adata: AnnData dataset
        perturbation_key: the key in adata.obs that contains the perturbations
        perturbation_combination_delimiter: the delimiter used to separate
            perturbations in the perturbation_key
        covariate_keys: a list of keys in adata.obs that contain the covariates
        perturbation_control_value: the value in adata.obs[perturbation_key] that
            corresponds to control cells
        n_differential_genes: number of top differentially expressed genes for
            each perturbation
        rankby_abs: if True, rank genes by absolute values of the score, thus including
            top downregulated genes in the top N genes. If False, the ranking will
            have only upregulated genes at the top.
        key_added: key used when adding the DEG dictionary to adata.uns
        return_dict: if True, return the DEG dictionary

    Returns:
        Adds the DEG dictionary to adata.uns

        If return_dict is True returns:
            gene_dict : dict
                Dictionary where groups are stored as keys, and the list of DEGs
                are the corresponding values
    """
    if "base" not in adata.uns["log1p"]:
        adata.uns["log1p"]["base"] = None

    gene_dict = {}
    if len(covariate_keys) == 0:
        sc.tl.rank_genes_groups(
            adata,
            groupby=perturbation_key,
            reference=perturbation_control_value,
            rankby_abs=rankby_abs,
            n_genes=n_differential_genes,
        )

        top_de_genes = pd.DataFrame(adata.uns["rank_genes_groups"]["names"])
        for group in top_de_genes:
            gene_dict[group] = top_de_genes[group].tolist()

    else:
        merged_covariates = merge_cols(adata.obs, covariate_keys, delim=delim)
        for unique_covariate in merged_covariates.unique():
            adata_cov = adata[merged_covariates == unique_covariate]
            sc.tl.rank_genes_groups(
                adata_cov,
                groupby=perturbation_key,
                reference=perturbation_control_value,
                rankby_abs=rankby_abs,
                n_genes=n_differential_genes,
            )

            top_de_genes = pd.DataFrame(adata_cov.uns["rank_genes_groups"]["names"])
            for group in top_de_genes:
                cov_group = unique_covariate + delim + group
                gene_dict[cov_group] = top_de_genes[group].tolist()

    adata.uns[key_added] = gene_dict

    if return_dict:
        return gene_dict


def preprocess(
    adata: ad.AnnData,
    perturbation_key: str,
    covariate_keys: list[str],
    control_value: str = "control",
    combination_delimiter: str = "+",
    highly_variable: int = 4000,
    degs: int = 25,
):
    adata.raw = None
    adata.var_names_make_unique()
    adata.obs_names_make_unique()

    ## Merge covariate columns
    adata.obs["cov_merged"] = merge_cols(adata.obs, covariate_keys)

    ## Preprocess
    print("Preprocessing ...")
    sc.pp.filter_genes(adata, min_cells=10)
    sc.pp.calculate_qc_metrics(adata, inplace=True)

    ## Normalize if needed
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)

    ## Pull out perturbed genes
    unique_perturbations = set()
    for comb in adata.obs[perturbation_key].unique():
        unique_perturbations.update(comb.split(combination_delimiter))
    unique_perturbations = unique_perturbations.intersection(adata.var_names)

    ## Subset to highly variable or differentially expressed genes
    if highly_variable > 0:
        print(
            "Filtering for highly variable genes or differentially expressed genes ..."
        )
        sc.pp.highly_variable_genes(
            adata,
            batch_key="cov_merged",
            flavor="seurat_v3",
            layer="counts",
            n_top_genes=int(highly_variable),
            subset=False,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            deg_gene_dict = differential_expression_by_covariate(
                adata,
                perturbation_key,
                control_value,
                covariate_keys,
                n_differential_genes=degs,
                rankby_abs=True,
                key_added="rank_genes_groups_cov",
                return_dict=True,
                delim="_",
            )
        deg_genes = set()
        for genes in deg_gene_dict.values():
            deg_genes.update(genes)

        var_genes = list(adata.var_names[adata.var["highly_variable"]])
        var_genes = list(unique_perturbations.union(var_genes).union(deg_genes))
        adata = adata[:, var_genes]

    print("Processed dataset summary:")
    print(adata)

    return adata
