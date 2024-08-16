import random
from collections import defaultdict


def merge_cols(obs_df, cols, delim="_"):
    """Merge columns in DataFrame"""

    covs = obs_df[cols[0]]
    if len(cols) > 1:
        for i in range(1, len(cols)):
            covs = covs.astype(str) + delim + obs_df[cols[i]].astype(str)
    covs = covs.astype("category")
    return covs


def subsample_by_group(adata, obs_key, max_cells=10000):
    """
    Downsample anndata on a per group basis
    """
    cells_keep = []
    for k in list(adata.obs[obs_key].unique()):
        cells = list(adata[adata.obs[obs_key] == k].obs_names)
        if len(cells) > max_cells:
            cells = random.sample(cells, k=max_cells)
        cells_keep.extend(cells)

    print("Input cells: " + str(adata.shape[0]))
    print("Sampled cells: " + str(len(cells_keep)))

    adata_downsampled = adata[cells_keep, :]
    return adata_downsampled


def pert_cluster_filter(adata, pert_key, cluster_key, delim="_", min_cells=20):
    """
    Filter anndata to only include perturbations that have at least min_cells in all clusters
    """

    adata.obs["pert_cluster"] = (
        adata.obs[pert_key].astype(str) + delim + adata.obs[cluster_key].astype(str)
    )
    pert_cl_counts = adata.obs["pert_cluster"].value_counts()
    pert_cl_keep = pert_cl_counts.loc[[x >= min_cells for x in pert_cl_counts]].index

    pert_counts_dict = defaultdict(int)
    for x in list(pert_cl_keep):
        pert_counts_dict[x.split(delim)[0]] += 1

    perts_keep = [
        x
        for x, n in pert_counts_dict.items()
        if n == len(adata.obs[cluster_key].unique())
    ]

    print("Original perturbations: " + str(len(adata.obs[pert_key].unique())))
    print("Filtered perturbations: " + str(len(perts_keep)))

    adata_filtered = adata[adata.obs[pert_key].isin(perts_keep), :]
    return adata_filtered


def get_ensembl_mappings():
    try:
        from pybiomart import Dataset
    except ImportError:
        raise ImportError("Please install the pybiomart package to use this function")

    # Set up connection to server
    dataset = Dataset(name="hsapiens_gene_ensembl", host="http://www.ensembl.org")

    id_gene_df = dataset.query(attributes=["ensembl_gene_id", "hgnc_symbol"])

    ensembl_to_genesymbol = {}
    # Store the data in a dict
    for gene_id, gene_symbol in zip(
        id_gene_df["Gene stable ID"], id_gene_df["HGNC symbol"]
    ):
        ensembl_to_genesymbol[gene_id] = gene_symbol

    return ensembl_to_genesymbol
