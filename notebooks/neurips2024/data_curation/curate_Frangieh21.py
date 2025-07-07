#!/usr/bin/env python
# coding: utf-8

# In[1]:


import scanpy as sc
import numpy as np
import os
import subprocess as sp
from scipy.sparse import csr_matrix

# In[ ]:


data_url = 'https://zenodo.org/records/7041849/files/FrangiehIzar2021_RNA.h5ad?download=1'
data_cache_dir = '/cluster/scratch/fluebeck/perturbench_data' ## Change this to your local data directory

if not os.path.exists(data_cache_dir):
    os.makedirs(data_cache_dir)

tmp_data_dir = f'{data_cache_dir}/frangieh21_downloaded.h5ad'

if not os.path.exists(tmp_data_dir):
    sp.call(f'wget {data_url} -O {tmp_data_dir}', shell=True)


# In[3]:


adata = sc.read_h5ad(tmp_data_dir)
adata


# In[4]:


adata.obs.celltype.value_counts()


# In[5]:


adata.obs['cell_type'] = 'melanocyte'


# In[6]:


adata.obs.perturbation_2.value_counts()


# In[7]:


treatment_map = {
    'Co-culture': 'co-culture',
    'Control': 'none',
}

adata.obs['treatment'] = [treatment_map[x] if x in treatment_map else x for x in adata.obs.perturbation_2]
adata.obs.treatment.value_counts()


# In[8]:


adata.obs.perturbation.value_counts()


# In[9]:


adata.obs['condition'] = adata.obs.perturbation.copy()
adata.obs['perturbation_type'] = 'CRISPRi'
adata.obs['dataset'] = 'frangieh21'


# In[10]:


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
    assert col in adata.obs.columns
    if np.any(adata.obs[col].isnull()):
        print(col)
    if np.any(adata.obs[col].isna()):
        print(col)


# In[11]:


adata.var.head()


# In[12]:


adata.X = csr_matrix(adata.X)


# In[13]:


adata.write_h5ad(f'{data_cache_dir}/frangieh21_processed.h5ad')

