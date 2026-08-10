import scanpy as sc
import os
import tarfile
import pandas as pd

from perturbench.data.accessors.base import (
    download_file,
    Accessor,
)


class McFaline23(Accessor):
    def __init__(self, data_cache_dir="../perturbench_data"):
        super().__init__(
            data_cache_dir=data_cache_dir,
            dataset_hf_url="https://huggingface.co/datasets/altoslabs/perturbench/resolve/main/mcfaline23_gxe_processed.h5ad.gz",
            split_hf_url="https://huggingface.co/datasets/altoslabs/perturbench/resolve/main/mcfaline23_gxe_splits.tar.gz",
            dataset_orig_url=None,
            dataset_name="mcfaline23",
        )
        self.split_data_path = f"{self.data_cache_dir}/{self.dataset_name}_gxe_splits.tar.gz"
        self.split_error_message = "Try using the notebooks/neurips2025/build_data_scaling_splits.ipynb to generate the split."

    def get_anndata(self):
        """
        Downloads, curates, and preprocesses the McFalineFigueroa23 dataset from
        Hugging Face. Saves the preprocessed data to disk and returns it in-memory.

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
                print(f"Error downloading file from {self.dataset_hf_url}: {e}")
                raise ValueError(
                    "Automatic data curation not available for this dataset. \
                     Use the notebooks in notebooks/neurips2025/data_curation \
                     to download and preprocess the data."
                ) from e
            
            print("Saved processed data to:", self.processed_data_path)

        return adata

    def get_split(self):
        if not os.path.exists(self.split_data_path):
            print("Downloading split from:", self.split_hf_url)
            try:
                download_file(self.split_hf_url, self.data_cache_dir, f"{self.dataset_name}_gxe_splits.tar.gz")
            except Exception as e:
                print(
                    f"Error downloading file from {self.split_hf_url}: {e}." + self.split_error_message
                )

        print("Loading split from:", self.split_data_path)
        with tarfile.open(self.split_data_path, 'r:gz') as tar:
            tar.extractall(path=self.data_cache_dir)
        
        split_files = {}
        extracted_dir = self.split_data_path.replace('.tar.gz', '')
        if os.path.exists(extracted_dir):
            for filename in os.listdir(extracted_dir):
                if filename.endswith('.csv'):
                    csv_path = os.path.join(extracted_dir, filename)
                    split_name = filename.replace('.csv', '')
                    split_files[split_name] = pd.read_csv(csv_path, index_col=0, header=None).iloc[:, 0]
        
        return split_files
