import scanpy as sc
import os
from abc import abstractmethod
import gzip
import shutil
from urllib.request import urlretrieve
import pandas as pd
import subprocess as sp

from perturbench.data.datasets import (
    Counterfactual,
    CounterfactualWithReference,
    SingleCellPerturbation,
    SingleCellPerturbationWithControls,
)
from perturbench.data.transforms.pipelines import LinearModelPipeline


def download_scperturb_adata(data_url, data_cache_dir, filename):
    """
    Helper function to download and cache anndata files. Returns an in-memory
    anndata object as-is with no curation.
    """
    if not os.path.exists(data_cache_dir):
        os.makedirs(data_cache_dir)

    tmp_data_path = f"{data_cache_dir}/{filename}"

    if not os.path.exists(tmp_data_path):
        urlretrieve(data_url, tmp_data_path)

    adata = sc.read_h5ad(tmp_data_path)
    return adata


def download_file(url: str, output_dir: str, output_filename: str) -> str:
    """
    Downloads a file from a URL to the specified output directory.
    If the file is gzipped, it will be automatically decompressed.
    
    Args:
        url (str): The URL of the file to download
        output_dir (str): The directory where the file should be saved
        output_filename (str): The filename for the downloaded file
    
    Returns:
        str: The path to the final file (decompressed if it was .gz)
    """
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = f"{output_dir}/{output_filename}"
    try:
        urlretrieve(url, output_path)
    except Exception as e:
        print(f"Error downloading file from {url} with urlretrieve: {e}. Trying wget instead.")
        try:
            sp.run(["wget", url, "-O", output_path], check=True)
        except Exception as e:
            print(f"Error downloading file from {url} with wget: {e}")
            raise ValueError(f"Error downloading file from {url}: {e}") from e
    
    if "h5ad.gz" in output_filename:
        # Decompress the .gz file using native gzip module
        decompressed_path = output_path.replace(".gz", "")
        with gzip.open(output_path, 'rb') as f_in:
            with open(decompressed_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        # Remove the original .gz file
        os.remove(output_path)
        return decompressed_path
    
    return output_path    


class Accessor:
    data_cache_dir: str
    dataset_hf_url: str
    dataset_orig_url: str
    dataset_name: str
    processed_data_path: str
    split_hf_url: str | None = None
    split_data_path: str | None = None

    def __init__(
        self,
        dataset_hf_url: str,
        dataset_orig_url: str,
        dataset_name: str,
        split_hf_url: str | None = None,
        split_data_path: str | None = None,
        data_cache_dir: str = "../perturbench_data",
    ):
        self.dataset_hf_url = dataset_hf_url
        self.dataset_orig_url = dataset_orig_url
        self.dataset_name = dataset_name
        self.data_cache_dir = data_cache_dir
        self.split_hf_url = split_hf_url
        self.split_data_path = split_data_path

    def get_dataset(
        self,
        dataset_class=SingleCellPerturbation,
        add_default_transforms=True,
        **dataset_kwargs,
    ):
        if dataset_class not in [
            SingleCellPerturbation,
            SingleCellPerturbationWithControls,
            Counterfactual,
            CounterfactualWithReference,
        ]:
            raise ValueError("Invalid dataset class.")

        ## Instantiate datamodule with Hydra using the config with the datapath changed
        adata = self.get_anndata()

        if "perturbation_key" not in dataset_kwargs:
            dataset_kwargs["perturbation_key"] = "condition"
        if "covariate_keys" not in dataset_kwargs:
            dataset_kwargs["covariate_keys"] = ["cell_type"]
        if "perturbation_control_value" not in dataset_kwargs:
            dataset_kwargs["perturbation_control_value"] = "control"

        dataset, context = dataset_class.from_anndata(
            adata=adata,
            **dataset_kwargs,
        )

        if add_default_transforms:
            dataset.transform = LinearModelPipeline(
                perturbation_uniques=context["perturbation_uniques"],
                covariate_uniques=context["covariate_uniques"],
            )

        return dataset, context

    
    def get_split(self):
        if os.path.exists(self.split_data_path):
            print("Loading split from:", self.split_data_path)
            split = pd.read_csv(self.split_data_path, index_col=0, header=None).iloc[:, 0]
        else:
            print("Downloading split from:", self.split_hf_url)
            try:
                download_file(self.split_hf_url, self.data_cache_dir, f"{self.dataset_name}_split.csv")
                split = pd.read_csv(self.split_data_path, index_col=0, header=None).iloc[:, 0]
            except Exception as e:
                print(
                    f"Error downloading file from {self.split_hf_url}: {e}." + self.split_error_message
                )

        return split
    
    @abstractmethod
    def get_anndata(self):
        pass
