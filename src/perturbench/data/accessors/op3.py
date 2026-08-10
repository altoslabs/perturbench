import scanpy as sc
import os

from perturbench.data.accessors.base import (
    download_file,
    Accessor,
)


class OP3(Accessor):
    def __init__(self, data_cache_dir="../perturbench_data"):
        super().__init__(
            data_cache_dir=data_cache_dir,
            dataset_hf_url="https://huggingface.co/datasets/altoslabs/perturbench/resolve/main/op3_processed.h5ad.gz",
            split_hf_url="https://huggingface.co/datasets/altoslabs/perturbench/resolve/main/op3_split.csv",
            dataset_orig_url=None,
            dataset_name="op3",
        )
        self.split_data_path = f"{self.data_cache_dir}/{self.dataset_name}_split.csv"
        self.split_error_message = "Try using the notebooks/neurips2025/build_op3_splits.ipynb to generate the split."

    def get_anndata(self):
        """
        Downloads, curates, and preprocesses the OP3 dataset from Hugging
        Face. Saves the preprocessed data to disk and
        returns it in-memory.

        Returns:
            adata (anndata.AnnData): Anndata object containing the processed data.

        """
        self.processed_data_path = (
            f"{self.data_cache_dir}/{self.dataset_name}_processed.h5ad"
        )
        if os.path.exists(self.processed_data_path):
            print("Loading processed data from:", self.processed_data_path)
            adata = sc.read_h5ad(self.processed_data_path)

        else:
            try:
                hf_filename = f"{self.dataset_name}_processed.h5ad.gz"
                download_file(self.dataset_hf_url, self.data_cache_dir, hf_filename)
                adata = sc.read_h5ad(self.processed_data_path)
                
            except Exception as e:
                print(f"Error downloading file from {self.dataset_hf_url}: {e}.\
                        Use the notebooks/neurips2025/data_curation/curate_op3.ipynb to download and preprocess the data."
                )
                

        return adata
