import scanpy as sc
import os
from abc import abstractmethod
import subprocess as sp

from perturbench.data.datasets import (
    Counterfactual,
    CounterfactualWithReference,
    SingleCellPerturbation,
    SingleCellPerturbationWithControls,
)
from perturbench.data.transforms.pipelines import SingleCellPipeline


def download_scperturb_adata(data_url, data_cache_dir, filename):
    """
    Helper function to download and cache anndata files. Returns an in-memory
    anndata object as-is with no curation.
    """
    if not os.path.exists(data_cache_dir):
        os.makedirs(data_cache_dir)

    tmp_data_path = f'{data_cache_dir}/{filename}'

    if not os.path.exists(tmp_data_path):
        sp.call(f'wget {data_url} -O {tmp_data_path}', shell=True)
    
    adata = sc.read_h5ad(tmp_data_path)
    return adata



class Accessor():
    
    data_cache_dir: str
    dataset_url: str
    dataset_name: str
    processed_data_path: str
    
    def __init__(
        self,
        dataset_url,
        dataset_name,
        data_cache_dir='../perturbench_data',
    ):
        self.dataset_url = dataset_url
        self.dataset_name = dataset_name
        self.data_cache_dir = data_cache_dir
    
    def get_dataset(
        self,
        dataset_class = SingleCellPerturbation,
        add_default_transforms = True,
        **dataset_kwargs,
    ):
        if dataset_class not in [
            SingleCellPerturbation,
            SingleCellPerturbationWithControls,
            Counterfactual,
            CounterfactualWithReference,
        ]:
            raise ValueError('Invalid dataset class.')
        
        ## Instantiate datamodule with Hydra using the config with the datapath changed
        adata = self.get_anndata()
        
        if 'perturbation_key' not in dataset_kwargs:
            dataset_kwargs['perturbation_key'] = 'condition'
        if 'covariate_keys' not in dataset_kwargs:
            dataset_kwargs['covariate_keys'] = ['cell_type']
        if 'perturbation_control_value' not in dataset_kwargs:
            dataset_kwargs['perturbation_control_value'] = 'control'
        
        dataset, context = dataset_class.from_anndata(
            adata=adata,
            **dataset_kwargs,
        )
        
        if add_default_transforms:
            dataset.transform = SingleCellPipeline(
                perturbation_uniques=context['perturbation_uniques'],
                covariate_uniques=context['covariate_uniques'],
            )
            
        return dataset, context
            
    
    @abstractmethod
    def get_anndata(self):
        pass
