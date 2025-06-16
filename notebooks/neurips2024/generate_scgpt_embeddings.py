#!/usr/bin/env python
# coding: utf-8

# # 2024-05-25-Preprocessing: Generating scGPT embeddings using the pretrained scGPT model

# In[ ]:


import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mode
import scanpy as sc
import sklearn
import warnings

import scgpt as scg


# In[ ]:


data_cache_dir = '../perturbench_data' ## Change this to your local data directory


# In[ ]:


# whole-human pretrained model is downloaded from:
# https://github.com/bowang-lab/scGPT?tab=readme-ov-file#pretrained-scgpt-model-zoo

model_dir = Path(f"{data_cache_dir}/pretrained_models/scGPT_human")


# ### norman19

# In[8]:


datapath = f'{data_cache_dir}/norman19_processed.h5ad'


# In[24]:


adata = sc.read_h5ad(datapath)


# In[11]:


adata_copy = adata.copy()


# In[12]:


adata_copy.X = adata_copy.layers['counts']


# In[ ]:





# In[ ]:


adata_copy_embeddings = scg.tasks.embed_data(
    adata_copy,
    model_dir,
    gene_col='gene_symbol',
    batch_size=128,
    return_new_adata=True,
)


# In[20]:


adata_copy_embeddings.X.shape


# In[25]:


adata.obsm['scgpt_embbeddings'] = adata_copy_embeddings.X


# In[26]:


adata


# In[29]:


outfile = f'{data_cache_dir}/norman19_preprocessed_with_embeddings.h5ad'


# In[30]:


adata.write_h5ad(outfile)


# In[ ]:





# ### mcfaline23

# In[3]:


datapath = f'{data_cache_dir}/mcfaline23_gxe_processed.h5ad'


# In[4]:


adata = sc.read_h5ad(datapath)


# In[5]:


adata


# In[ ]:





# In[6]:


adata_copy = adata.copy()


# In[7]:


adata_copy.X = adata_copy.layers['counts']


# In[ ]:


adata_copy_embeddings = scg.tasks.embed_data(
    adata_copy,
    model_dir,
    gene_col='index',
    batch_size=128,
    return_new_adata=True,
)


# In[9]:


adata.obsm['scgpt_embbeddings'] = adata_copy_embeddings.X


# In[10]:


adata


# In[11]:


outfile = f'{data_cache_dir}/mcfaline23_gxe_processed_with_embeddings.h5ad'


# In[12]:


adata.write_h5ad(outfile)


# In[ ]:





# ### srivatsan20

# In[13]:


datapath = f'{data_cache_dir}/srivatsan20_processed.h5ad'


# In[14]:


adata = sc.read_h5ad(datapath)


# In[15]:


adata


# In[ ]:





# In[16]:


adata_copy = adata.copy()


# In[17]:


adata_copy.X = adata_copy.layers['counts']


# In[ ]:


adata_copy_embeddings = scg.tasks.embed_data(
    adata_copy,
    model_dir,
    gene_col='index',
    batch_size=128,
    return_new_adata=True,
)


# In[19]:


adata.obsm['scgpt_embbeddings'] = adata_copy_embeddings.X


# In[20]:


adata


# In[21]:


outfile = f'{data_cache_dir}/srivatsan20_highest_processed_with_embeddings.h5ad'


# In[22]:


adata.write_h5ad(outfile)

