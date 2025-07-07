import scanpy as sc
import os
from scipy.sparse import csr_matrix

from perturbench.analysis.preprocess import preprocess
from perturbench.data.accessors.base import download_scperturb_adata
from perturbench.data.accessors.base import Accessor


class Frangieh21(Accessor):
    def __init__(self, data_cache_dir='/cluster/scratch/fluebeck/perturbench_data'):
        super().__init__(
            data_cache_dir=data_cache_dir,
            dataset_url='https://zenodo.org/records/7041849/files/FrangiehIzar2021_RNA.h5ad?download=1',
            dataset_name='frangieh21',
        )
        
    def get_anndata(self):
        """
        Downloads, curates, and preprocesses the sciplex3 dataset from the scPerturb 
        database. Saves the preprocessed data to disk and returns it in-memory.
        
        Returns:
            adata (anndata.AnnData): Anndata object containing the processed data.
        
        """
        self.processed_data_path = f'{self.data_cache_dir}/{self.dataset_name}_processed.h5ad'        
        if os.path.exists(self.processed_data_path):
            print('Loading processed data from:', self.processed_data_path)
            adata = sc.read_h5ad(self.processed_data_path)
        
        else:    
            adata = download_scperturb_adata(
                self.dataset_url, 
                self.data_cache_dir, 
                filename=f'{self.dataset_name}_downloaded.h5ad',
            )
            
            ## Format column names
            treatment_map = {
                'Co-culture': 'co-culture',
                'Control': 'none',
            }
            adata.obs['treatment'] = [treatment_map[x] if x in treatment_map else x for x in adata.obs.perturbation_2]
            adata.obs['cell_type'] = 'melanocyte'
            adata.obs['condition'] = adata.obs.perturbation.copy()
            adata.obs['perturbation_type'] = 'CRISPRi'
            adata.obs['dataset'] = 'frangieh21' 
            
            adata.X = csr_matrix(adata.X)
            adata = preprocess(
                adata,
                perturbation_key='condition',
                covariate_keys=['treatment'],
            )
            
            adata = adata.copy()
            adata.write_h5ad(self.processed_data_path)
            print('Saved processed data to:', self.processed_data_path)
        
        return adata