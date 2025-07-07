#!/usr/bin/env python
# coding: utf-8

# # Generate manual splits for Jiang24 and Frangieh21

# In[9]:


import scanpy as sc
import pandas as pd
from sklearn.model_selection import train_test_split

from perturbench.data.datasplitter import PerturbationDataSplitter

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# In[10]:


data_cache_dir = '/cluster/scratch/fluebeck/perturbench_data'


# ## Jiang24

# In[12]:


adata = sc.read_h5ad(f'{data_cache_dir}/jiang24_processed.h5ad', backed='r')
adata


# ### Create a manual split

# In[14]:


jiang24_heldout_covariates = []
cytokines_holdout = ['IFNG', 'INS', 'TGFB']
cell_lines_holdout = ['k562', 'mcf7', 'ht29', 'hap1']
for cytokine in cytokines_holdout:
    for cell_line in cell_lines_holdout:
        jiang24_heldout_covariates.append({cytokine, cell_line})
jiang24_heldout_covariates


# In[13]:


manual_splitter = PerturbationDataSplitter(
    adata.obs.copy(),
    perturbation_key='condition',
    covariate_keys=['cell_type', 'treatment'],
    perturbation_control_value='control',
)
manual_splitter


# Holdout 70% of perturbations in 4 cytokine treatments for 4 cell lines

# In[15]:


jiang24_split = manual_splitter.split_covariates_manual(
    seed=0, 
    covariates_holdout=jiang24_heldout_covariates,
    max_heldout_fraction_per_covariate=0.7, ## Maximum fraction of perturbations held out per covariate
)


# In[16]:


jiang24_split.to_csv(f'{data_cache_dir}/jiang24_split.csv', header=False)


# ## frangieh21

# In[17]:


adata = sc.read_h5ad(f'{data_cache_dir}/frangieh21_processed.h5ad')
adata


# In[18]:


adata.obs.treatment.value_counts()


# In[19]:


manual_splitter = PerturbationDataSplitter(
    adata.obs.copy(),
    perturbation_key='condition',
    covariate_keys=['treatment'],
    perturbation_control_value='control',
)
manual_splitter


# Holdout 70% of perturbations in the Co-culture treatment

# In[20]:


frangieh21_split = manual_splitter.split_covariates_manual(
    seed=0, 
    covariates_holdout=[{'co-culture'}],
    max_heldout_fraction_per_covariate=0.7, ## Maximum fraction of perturbations held out per covariate
)


# In[21]:


frangieh21_split.to_csv(f'{data_cache_dir}/frangieh21_split.csv', header=False)

