#!/usr/bin/env python
# coding: utf-8

# # 2024-04-24-Preprocessing: Generating nested mcfaline23 subsets

# PerturbSeq screen of interactions between chemical and genetic perturbations

# In[1]:


import scanpy as sc
import pandas as pd
import numpy as np
import random
import os
from sklearn.model_selection import train_test_split

from perturbench.data.datasplitter import PerturbationDataSplitter



# ## Load data and generate subsets

# In[ ]:


data_cache_dir = './perturbench_data' ## Change this to your local data directory

splits_dir = f'{data_cache_dir}/mcfaline23_gxe_splits'
if not os.path.exists(splits_dir):
    os.makedirs(splits_dir)


# In[ ]:


adata = sc.read_h5ad(f'{data_cache_dir}/mcfaline23_gxe_processed.h5ad', backed='r')
adata


# ## Scale across covariates

# In[3]:


adata.obs['cell_type_treat'] = adata.obs['cell_type'].astype(str) + '_' + adata.obs['treatment'].astype(str)


# ### Generate subsets

# In[4]:


unique_covariates = [x for x in adata.obs.cell_type_treat.unique() if x != 'control']
unique_covariates_cell_type = [x.split('_')[0] for x in unique_covariates]
len(unique_covariates)


# In[5]:


unique_covariates


# In[6]:


small_covariates_holdout = [
    'a172_nintedanib',
    't98g_lapatinib',
    'u87mg_none',
]
small_covariates_train = [
    'a172_none',
    't98g_nintedanib',
    'u87mg_lapatinib',
]

small_covariates = small_covariates_holdout + small_covariates_train


# ### Generate splits

# #### Small

# In[7]:


adata_small = adata[adata.obs.cell_type_treat.isin(small_covariates)].to_memory()
adata_small


# In[8]:


unique_perturbations = [x for x in adata_small.obs.condition.unique() if x!= 'control']
len(unique_perturbations)


# In[10]:


rng = np.random.default_rng(12345)

control_val_ix = []
control_test_ix = []
heldout_pert_covs = []
for cov in small_covariates_holdout:
    random.seed(int(rng.integers(0, 2**16)))
    sampled_perts = random.sample(unique_perturbations, int(0.7*len(unique_perturbations)))
    heldout_pert_covs += [p + '_' + cov for p in sampled_perts]
    
    cov_controls = adata_small[
        (adata_small.obs.cell_type_treat == cov) & (adata_small.obs.condition == 'control')
    ].obs_names.to_list()
    cov_controls_heldout = random.sample(cov_controls, int(0.5*len(cov_controls)))
    cov_controls_heldout_val, cov_controls_heldout_test = train_test_split(
        cov_controls_heldout, test_size=0.5, random_state=int(rng.integers(0, 2**16))
    )
    control_val_ix += cov_controls_heldout_val
    control_test_ix += cov_controls_heldout_test

random.seed(int(rng.integers(0, 2**16)))
val_pert_covs = random.sample(heldout_pert_covs, int(0.5*len(heldout_pert_covs)))
test_pert_covs = [x for x in heldout_pert_covs if x not in val_pert_covs]

len(val_pert_covs), len(test_pert_covs)


# In[11]:


split = pd.Series('train', index=adata_small.obs.index)

adata_small.obs['pert_cov'] = adata_small.obs.condition.astype(str) + '_' + adata_small.obs.cell_type_treat.astype(str)
val_ix = adata_small[adata_small.obs.pert_cov.isin(val_pert_covs)].obs_names.tolist()
test_ix = adata_small[adata_small.obs.pert_cov.isin(test_pert_covs)].obs_names.tolist()

split.loc[val_ix + control_val_ix] = 'val'
split.loc[test_ix + control_test_ix] = 'test'

split.value_counts()


# In[15]:


small_obs = adata_small.obs.copy()
small_obs['split'] = split
small_obs = small_obs.loc[:,['split', 'condition', 'cell_type_treat']].drop_duplicates()

for cov in small_covariates:
    print(cov)
    cov_obs = small_obs[small_obs.cell_type_treat == cov]
    for spl in ['train', 'val', 'test']:
        cov_obs_spl = cov_obs.loc[cov_obs.split == spl]
        if cov_obs_spl.shape[0] > 0:
            assert 'control' in cov_obs_spl.condition.unique()
        else:
            assert spl != 'train'
    
    print(cov_obs.split.value_counts())


# In[16]:


split_padded = pd.Series(None, index=adata.obs.index)
split_padded.loc[split.index] = split
split_padded.value_counts()


# In[17]:


len(split_padded)


# In[18]:


split_padded.to_csv(f'{data_cache_dir}/mcfaline23_gxe_splits/small_covariate_split.csv', header=False)


# #### Medium

# In[19]:


medium_covariates = [
    'a172_lapatinib',
    't98g_none',
    'u87mg_nintedanib',
] + small_covariates
medium_covariates


# In[ ]:


adata_medium = adata[adata.obs.cell_type_treat.isin(medium_covariates)]
adata_medium


# In[ ]:


medium_split = pd.Series('train', index=adata_medium.obs.index)
for split_val in split.unique():
    split_idx = split[split == split_val].index
    medium_split.loc[split_idx] = split_val
medium_split.value_counts()


# In[ ]:


medium_split_padded = pd.Series(None, index=adata.obs.index)
medium_split_padded.loc[medium_split.index] = medium_split
len(medium_split_padded)


# In[ ]:


medium_split_padded.to_csv(f'{data_cache_dir}/mcfaline23_gxe_splits/medium_covariate_split.csv', header=False)


# #### Full

# In[ ]:


full_split = pd.Series('train', index=adata.obs.index)
for split_val in split.unique():
    split_idx = split[split == split_val].index
    full_split.loc[split_idx] = split_val
full_split.value_counts()


# In[ ]:


full_split.to_csv(f'{data_cache_dir}/mcfaline23_gxe_splits/full_covariate_split.csv', header=False)


# ## Scale across perturbations

# ### Generate subsets

# In[ ]:


unique_covariates = [x for x in adata.obs.condition.unique() if x != 'control']
len(unique_covariates)


# In[ ]:


random.seed(84)
medium_perturbations = random.sample(unique_covariates, int(len(unique_covariates)/2))
small_perturbations = random.sample(medium_perturbations, int(len(medium_perturbations)/2))
len(small_perturbations), len(medium_perturbations) 


# ### Generate splits

# #### Small

# In[ ]:


adata_small = adata[adata.obs.condition.isin(small_perturbations + ['control'])]
adata_small


# In[6]:


splitter = PerturbationDataSplitter(
    adata_small.obs.copy(),
    perturbation_key='condition',
    covariate_keys=['cell_type', 'treatment'],
    perturbation_control_value='control',
)


# In[ ]:


split = splitter.split_covariates(
    seed=57,
    print_split=True, ## Print a summary of the split if True
    max_heldout_covariates=7, ## Maximum number of held out covariates (in this case cell types)
    max_heldout_fraction_per_covariate=0.3, ## Maximum fraction of perturbations held out per covariate
)


# In[8]:


split.value_counts()


# In[9]:


split_padded = pd.Series(None, index=adata.obs.index)
split_padded.loc[split.index] = split
split_padded.value_counts()


# In[10]:


len(split_padded)


# In[11]:


split_padded.to_csv(f'{data_cache_dir}/mcfaline23_gxe_splits/small_split.csv', header=False)


# #### Medium

# In[ ]:


adata_medium = adata[adata.obs.condition.isin(medium_perturbations + ['control'])]
adata_medium


# In[13]:


medium_split = pd.Series('train', index=adata_medium.obs.index)
for split_val in split.unique():
    split_idx = split[split == split_val].index
    medium_split.loc[split_idx] = split_val
medium_split.value_counts()


# In[14]:


medium_split_padded = pd.Series(None, index=adata.obs.index)
medium_split_padded.loc[medium_split.index] = medium_split
medium_split_padded.value_counts()


# In[15]:


medium_split_padded.to_csv(f'{data_cache_dir}/mcfaline23_gxe_splits/medium_split.csv', header=False)


# #### Full

# In[16]:


full_split = pd.Series('train', index=adata.obs.index)
for split_val in split.unique():
    split_idx = split[split == split_val].index
    full_split.loc[split_idx] = split_val
full_split.value_counts()


# In[17]:


full_split.to_csv(f'{data_cache_dir}/mcfaline23_gxe_splits/full_split.csv', header=False)

