#!/usr/bin/env python
# coding: utf-8

# # Investigate data imbalance
# Start with Srivatsan balanced data (189 perturbationsn seen in three cell types). Create imbalanced splits through downsampling.  
# 

# In[14]:


import scanpy as sc
import pandas as pd
import numpy as np
from perturbench.data.datasplitter import PerturbationDataSplitter
import warnings
import os
# from scipy.stats import entropy

# Suppress FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)



# In[3]:


def get_num_train_cell_types(
    splitter,
    split_key,
):
    """Returns the number of training cell types per perturbation"""
    num_train_cell_types = []
    for pert in splitter.obs_dataframe.condition.unique():
        pert_df = splitter.obs_dataframe[splitter.obs_dataframe.condition == pert]
        pert_df = pert_df.loc[:,['cell_type', split_key]].drop_duplicates()
        num_train_cell_types.append(pert_df.loc[pert_df[split_key] == 'train', 'cell_type'].nunique())

    num_train_cell_types = pd.Series(num_train_cell_types, index=splitter.obs_dataframe.condition.unique())
    return num_train_cell_types


# In[4]:


data_cache_dir = './perturbench_data' ## Change this to your local data directory


# In[7]:


balanced_transfer_adata = sc.read_h5ad(f'{data_cache_dir}/srivatsan20_processed.h5ad')
balanced_transfer_adata


# In[8]:


unique_perturbations = balanced_transfer_adata.obs.groupby('cell_type')['perturbation'].nunique()
print(unique_perturbations)


# In[9]:


balanced_transfer_splitter = PerturbationDataSplitter(
    balanced_transfer_adata.obs.copy(),
    perturbation_key='condition',
    covariate_keys=['cell_type'],
    perturbation_control_value='control',
)
balanced_transfer_splitter


# In[10]:


def balanced_entropy(counts): 
    counts = counts + 1e-8 # to avoid -inf
    N = np.sum(counts) 
    probabilities = counts / N 
    log_probabilities = np.log(probabilities) 
    entropy = -np.sum(probabilities * log_probabilities) 
    normalized_entropy = entropy / np.log(len(counts)) 
    return normalized_entropy 

def find_array_with_entropy(target_entropy, max_value=189, tolerance=0.01, max_iter=10000, seed=42):  
    np.random.seed(seed)
    for _ in range(max_iter):  
        # Generate two random values less than or equal to max_value  
        # np.random.seed(0) 
        x1, x2 = np.random.randint(30, max_value+1, 2)  
        counts = np.array([189, x1, x2])  
        current_entropy = balanced_entropy(counts)  
        # Check if the current entropy is within the tolerance of the target entropy  
        if np.abs(current_entropy - target_entropy) < tolerance:  
            return counts  
    return None  # Return None if no solution is found within max_iter iterations


# In[11]:


# Minimum possible entropy requiring at least 30 perts per cell type
x = np.array([189, 30, 30])
x.shape
balanced_entropy(x)


# In[12]:


# Example usage and deciding reasonable values for target entropy (too low and no such distribution can be found)  
# Set a target entropy value  
target_entropy = 0.7  # Example value  
target_distribution = find_array_with_entropy(target_entropy)  
if target_distribution is not None:  
    print('Found array:', target_distribution)  
    print(balanced_entropy(target_distribution))
else:  
    print('No array found that meets the criteria within the given iterations.')


# In[13]:


def downsample_adata(balanced_transfer_adata, desired_unique_perturbations, seed=42): 
    np.random.seed(seed)   # Group the perturbations by cell type and count the number of unique perturbations 
    
    unique_perturbations = balanced_transfer_adata.obs.groupby('cell_type')['perturbation'].nunique() 
 
    # Create a mask to filter the data 
    mask = np.zeros(len(balanced_transfer_adata), dtype=bool) 
 
    # Iterate over each cell type and select the desired number of unique perturbations 
    for cell_type, num_unique_perturbations in zip(unique_perturbations.index, desired_unique_perturbations): 
        cell_type_mask = (balanced_transfer_adata.obs['cell_type'] == cell_type) 
        perturbations = balanced_transfer_adata.obs.loc[cell_type_mask, 'perturbation'].unique()
        
        np.random.shuffle(perturbations) 
        
        selected_perturbations = list(perturbations[:num_unique_perturbations]) 
        if 'control' not in selected_perturbations:
            selected_perturbations.append('control')
        mask |= (cell_type_mask & balanced_transfer_adata.obs['perturbation'].isin(selected_perturbations)) 
 
    # Apply the mask to create the downsampled AnnData object 
    adata_downsampled = balanced_transfer_adata[mask]#.copy() 
    # adata_downsampled.obs.reset_index(drop=True, inplace=True)  # Reset index to avoid any indexing issues
    adata_downsampled = adata_downsampled.copy()  # Ensure that the data matrix is realigned

    return adata_downsampled 


# In[15]:


split_dir = f'{data_cache_dir }/downsampled_imbalance/'

if not os.path.exists(split_dir):
    os.makedirs(split_dir)


# In[16]:


entropy_list =  [0.7,0.8,0.9]
for target_entropy in entropy_list:
    target_distribution = find_array_with_entropy(target_entropy) 
    adata_downsampled = downsample_adata(balanced_transfer_adata, target_distribution)
    
    splitter = PerturbationDataSplitter(
        adata_downsampled.obs.copy(),
        perturbation_key='condition',
        covariate_keys=['cell_type', 'treatment'],
        perturbation_control_value='control',
    )

    split = splitter.split_covariates(
    seed=57,
    print_split=True, ## Print a summary of the split if True
    max_heldout_covariates=1, ## Maximum number of held out covariates (in this case cell types)
    max_heldout_fraction_per_covariate=0.3, ## Maximum fraction of perturbations held out per covariate
)
    # Cells to be excluded in the downsample are given 'None' as their label so they don't go in train/validation/test
    split_padded = pd.Series(None, index=balanced_transfer_adata.obs.index)
    split_padded.loc[split.index] = split
    split_padded.to_csv(f'{split_dir}/srivatsan_downsampled_entropy_' + str(target_entropy) + '_splits.csv', header=False)
    print(target_entropy,target_distribution)
    print(split_padded.value_counts())


# ### Fully balanced split

# In[17]:


balanced_transfer_splitter = PerturbationDataSplitter(
    balanced_transfer_adata.obs.copy(),
    perturbation_key='condition',
    covariate_keys=['cell_type','treatment'],
    perturbation_control_value='control',
)


balanced_transfer_split = balanced_transfer_splitter.split_covariates(
            seed=57, 
            print_split=True, ## Print a summary of the split if True
            max_heldout_covariates=1, ## Maximum number of held out covariates (in this case cell types)
            max_heldout_fraction_per_covariate=0.3, ## Maximum fraction of perturbations held out per covariate
        )

balanced_transfer_split.to_csv(f'{split_dir}/srivatsan_downsampled_entropy_' + str(1) + '_splits.csv', header=False)

