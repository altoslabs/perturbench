#!/usr/bin/env python
# coding: utf-8

# In[10]:


import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import anndata as ad
import gc
from perturbench.analysis.preprocess import preprocess

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[11]:


data_directory = '../perturbench_data/'


# In[ ]:


adata_paths = [
    'Seurat_object_IFNB_Perturb_seq.h5ad',
    'Seurat_object_IFNG_Perturb_seq.h5ad',
    'Seurat_object_INS_Perturb_seq.h5ad',
    'Seurat_object_TGFB_Perturb_seq.h5ad',
    'Seurat_object_TNFA_Perturb_seq.h5ad',
]
adata_paths = [data_directory + path for path in adata_paths]


# In[ ]:


adata_list = []
for adata_path in adata_paths:
    adata = sc.read_h5ad(adata_path)
    adata.X = adata.raw.X.copy()
    adata.raw = None

    adata.obs.cell_type = [x.lower() for x in adata.obs.cell_type]
    adata.obs.cell_type.value_counts()

    adata.obs['treatment'] = adata_path.split('/')[-1].split('_')[2]
    adata.obs.treatment.value_counts()

    condition_remap = {
        'NT': 'control',
    }
    adata.obs['condition'] = adata.obs.gene.copy()
    adata.obs.condition = [condition_remap.get(x, x) for x in adata.obs.condition]
    adata.obs['condition'] = adata.obs.condition.astype('category')
    adata.obs['perturbation'] = adata.obs.condition.copy()

    adata.obs['ncounts'] = adata.obs['nCount_RNA'].copy()
    adata.obs['ngenes'] = adata.obs['nFeature_RNA'].copy()
    adata.obs['perturbation_type'] = 'CRISPRi'
    
    adata_list.append(adata)
    del adata
    gc.collect()


# In[ ]:


adata_merged = ad.concat(adata_list)
adata_merged.obs_names_make_unique()

del adata_list
gc.collect()

adata_merged


# In[ ]:


adata_merged.obs['dataset'] = 'jiang24'


# In[ ]:


required_cols = [
    'condition',
    'cell_type',
    'treatment',
    'perturbation_type',
    'dataset',
    'ngenes',
    'ncounts',
]

for col in required_cols:
    assert col in adata_merged.obs.columns
    if np.any(adata_merged.obs[col].isnull()):
        print(col)
    if np.any(adata_merged.obs[col].isna()):
        print(col)


# In[ ]:


adata_merged.obs.condition.value_counts()


# In[ ]:


adata_merged = preprocess(
    adata_merged,
    perturbation_key='condition',
    covariate_keys=['cell_type', 'treatment'],
)
adata_merged


# In[ ]:


adata_merged.write_h5ad(data_directory + 'jiang24_processed.h5ad')

