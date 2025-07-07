#!/usr/bin/env python
# coding: utf-8

# # 2023-04-17-Curation: Norman19 Combo Screen

# Add cross validation splits to Norman19

# In[1]:


import scanpy as sc
import os
import subprocess as sp
from perturbench.analysis.preprocess import preprocess
from scipy.sparse import csr_matrix



# Download from: https://zenodo.org/records/7041849/files/NormanWeissman2019_filtered.h5ad?download=1

# In[2]:


data_url = 'https://zenodo.org/records/7041849/files/NormanWeissman2019_filtered.h5ad?download=1'
data_cache_dir = '/cluster/scratch/fluebeck/perturbench_data' ## Change this to your local data directory

if not os.path.exists(data_cache_dir):
    os.makedirs(data_cache_dir)

tmp_data_dir = f'{data_cache_dir}/norman19_downloaded.h5ad'

if not os.path.exists(tmp_data_dir):
    sp.call(f'wget {data_url} -O {tmp_data_dir}', shell=True)


# In[3]:


adata = sc.read_h5ad(tmp_data_dir)
adata


# In[5]:


adata.obs.rename(columns = {
    'nCount_RNA': 'ncounts',
    'nFeature_RNA': 'ngenes',
    'percent.mt': 'percent_mito',
    'cell_line': 'cell_type',
}, inplace=True)


# In[6]:


adata.obs.perturbation.unique()


# In[7]:


adata.obs['perturbation'] = adata.obs['perturbation'].str.replace('_', '+')
adata.obs['perturbation'] = adata.obs['perturbation'].astype('category')
adata.obs.perturbation.value_counts()


# In[8]:


adata.obs['condition'] = adata.obs.perturbation.copy()


# In[9]:


adata.X = csr_matrix(adata.X)


# In[10]:


adata = preprocess(
    adata,
    perturbation_key='condition',
    covariate_keys=['cell_type'],
)


# In[11]:


adata = adata.copy()
output_data_path = f'{data_cache_dir}/norman19_processed.h5ad'
adata.write_h5ad(output_data_path)


# In[12]:


adata


# In[22]:


for col in adata.obs.columns:
    print(col)


# In[30]:


adata.obs["perturbation"].head()


# In[31]:


adata.obs["nperts"].head()


# In[15]:


print(adata.var.head())


# In[32]:


adata.X.layers


# In[ ]:




