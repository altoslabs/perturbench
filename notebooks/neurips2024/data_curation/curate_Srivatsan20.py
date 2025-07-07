#!/usr/bin/env python
# coding: utf-8

# # 2023-07-26-Curation: Srivatsan20 Chemical Perturbation Screen

# In[2]:


import scanpy as sc
import numpy as np
import subprocess as sp
import os
from perturbench.analysis.utils import get_ensembl_mappings
from perturbench.analysis.preprocess import preprocess


# Download from: https://zenodo.org/records/7041849/files/SrivatsanTrapnell2020_sciplex3.h5ad?download=1

# In[3]:


data_url = 'https://zenodo.org/records/7041849/files/SrivatsanTrapnell2020_sciplex3.h5ad?download=1'
data_cache_dir = '/cluster/scratch/fluebeck/perturbench_data' ## Change this to your local data directory

if not os.path.exists(data_cache_dir):
    os.makedirs(data_cache_dir)

tmp_data_dir = f'{data_cache_dir}/srivatsan20_downloaded.h5ad'

if not os.path.exists(tmp_data_dir):
    sp.call(f'wget {data_url} -O {tmp_data_dir}', shell=True)


# In[4]:


adata = sc.read_h5ad(tmp_data_dir)
adata


# In[5]:


adata.obs.cell_line.value_counts()


# In[6]:


adata.var.head()


# In[7]:


unique_genes = ~adata.var.ensembl_id.duplicated()
np.sum(unique_genes)


# In[8]:


adata = adata[:,unique_genes]
adata.var_names = adata.var.ensembl_id.astype(str)


# In[9]:


human_ids = [x for x in adata.var_names if 'ENSG' in x]
len(human_ids)


# In[10]:


adata = adata[:,human_ids]
adata.shape


# In[12]:


gene_mappings = get_ensembl_mappings()


# In[13]:


len(gene_mappings.keys())


# In[14]:


gene_mappings = {k:v for k,v in gene_mappings.items() if isinstance(v, str) and v != ''}
len(gene_mappings.keys())


# In[15]:


np.sum([x in gene_mappings for x in adata.var_names])


# In[16]:


adata = adata[:,[x in gene_mappings for x in adata.var_names]]
adata


# In[17]:


adata.var['gene_symbol'] = [gene_mappings[x] for x in adata.var_names]
adata.var_names = adata.var['gene_symbol']


# In[18]:


adata.var_names[0:5]


# In[19]:


adata.obs.perturbation.value_counts()


# In[20]:


adata.obs.rename(columns = {
    'n_genes': 'ngenes',
    'n_counts': 'ncounts',
}, inplace=True)

adata.obs['perturbation_type'] = 'drug'
adata.obs['dataset'] = 'srivatsan20'
adata.obs['cell_type'] = adata.obs['cell_line'].copy()
adata.obs['treatment'] = 'none'
adata.obs['condition'] = adata.obs['perturbation'].copy()

adata


# In[21]:


adata.obs.cell_type.value_counts()


# In[ ]:


adata = adata[adata.obs.cell_type.isin(['MCF7', 'A549', 'K562'])]
adata.obs.cell_type = [x.lower() for x in adata.obs.cell_type]
adata.obs.cell_type.unique()


# Ensure doses are in micromolars

# In[23]:


adata.obs['dose'] = adata.obs['dose_value'].copy() * 1/1000
adata.obs.dose.value_counts()


# In[24]:


adata.obs['dose_unit'] = 'uM'


# Ensure no perturbation name has a "+" in it since we use "+" as the perturbation delimiter

# In[25]:


for p in adata.obs.condition.unique():
    if "+" in p:
        print(p)


# In[26]:


perturbation_remap = {
    '(+)-JQ1': 'JQ1',
    'ENMD-2076 L-(+)-Tartaric acid': 'ENMD-2076',
}

adata.obs['perturbation'] = [perturbation_remap.get(x, x) for x in adata.obs.perturbation.astype(str)]
adata.obs['condition'] = adata.obs['perturbation'].copy()


# Subset to highest dose

# In[27]:


adata.obs['dose'].value_counts()


# In[28]:


adata.shape


# In[29]:


np.sum((adata.obs.dose == 10.0) | (adata.obs.condition == 'control'))


# In[ ]:


adata = adata[(adata.obs.dose == 10.0) | (adata.obs.condition == 'control')].copy()
adata.shape


# In[31]:


import gc
gc.collect()


# Run preprocessing

# In[32]:


adata = preprocess(
    adata,
    perturbation_key='condition',
    covariate_keys=['cell_type'],
)


# In[33]:


adata = adata.copy()
adata

gc.collect()


# In[37]:


adata.var.head()


# In[36]:


adata.var.index.name = None


# In[38]:


output_data_path = f'{data_cache_dir}/srivatsan20_processed.h5ad'
adata.write_h5ad(output_data_path)

