import scanpy as sc
import os
from scipy.sparse import csr_matrix

from perturbench.analysis.preprocess import preprocess
from perturbench.analysis.utils import get_ensembl_mappings
from perturbench.data.accessors.base import download_scperturb_adata
from perturbench.data.accessors.base import Accessor


class Sciplex3(Accessor):
    def __init__(self, data_cache_dir='../perturbench_data'):
        super().__init__(
            data_cache_dir=data_cache_dir,
            dataset_url='https://zenodo.org/records/7041849/files/SrivatsanTrapnell2020_sciplex3.h5ad?download=1',
            dataset_name='sciplex3',
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
            
            unique_genes = ~adata.var.ensembl_id.duplicated()
            adata = adata[:,unique_genes]

            ## Map ENSEMBL IDs to gene symbols
            adata.var_names = adata.var.ensembl_id.astype(str)
            human_ids = [x for x in adata.var_names if 'ENSG' in x]

            adata = adata[:,human_ids]
            gene_mappings = get_ensembl_mappings()
            gene_mappings = {k:v for k,v in gene_mappings.items() if isinstance(v, str) and v != ''}
            adata = adata[:,[x in gene_mappings for x in adata.var_names]]
            adata.var['gene_symbol'] = [gene_mappings[x] for x in adata.var_names]
            adata.var_names = adata.var['gene_symbol']
            adata.var_names_make_unique()
            adata.var.index.name = None
            
            ## Format column names
            adata.obs.rename(columns = {
                'n_genes': 'ngenes',
                'n_counts': 'ncounts',
            }, inplace=True)

            ## Format cell line names
            adata.obs['cell_type'] = adata.obs['cell_line'].copy()
            adata = adata[adata.obs.cell_type.isin(['MCF7', 'A549', 'K562'])]
            adata.obs['cell_type'] = [x.lower() for x in adata.obs.cell_type]
            
            ## Rename some chemicals with the "+" symbol
            perturbation_remap = {
                '(+)-JQ1': 'JQ1',
                'ENMD-2076 L-(+)-Tartaric acid': 'ENMD-2076',
            }
            adata.obs['perturbation'] = [perturbation_remap.get(x, x) for x in adata.obs.perturbation.astype(str)]
            adata.obs['condition'] = adata.obs['perturbation'].copy()
            
            ## Subset to highest dose only
            adata = adata[(adata.obs.dose_value == 10000) | (adata.obs.condition == 'control')].copy()
            
            adata.X = csr_matrix(adata.X)
            adata = preprocess(
                adata,
                perturbation_key='condition',
                covariate_keys=['cell_type'],
            )
            
            adata = adata.copy()
            adata.write_h5ad(self.processed_data_path)
            print('Saved processed data to:', self.processed_data_path)
        
        return adata