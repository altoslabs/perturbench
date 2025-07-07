#!/usr/bin/env python
# coding: utf-8

# # 2023-07-26-Curation: McFaline-Figuroa23

# PerturbSeq screen of interactions between chemical and genetic perturbations

# In[1]:


import scanpy as sc
import numpy as np
import anndata as ad
import gc
from scipy.sparse import csr_matrix

from perturbench.analysis.utils import get_ensembl_mappings
from perturbench.analysis.preprocess import preprocess


# Get gene names from ENSEMBL IDs

# In[3]:


id_to_gene = get_ensembl_mappings()
id_to_gene = {k:v for k,v in id_to_gene.items() if isinstance(v, str) and v != ''}
len(id_to_gene.keys())


# ## Load data

# In[ ]:


data_cache_dir = '/cluster/scratch/fluebeck/perturbench_data'


# In[4]:


data_paths = [
    f'{data_cache_dir}/gxe1.h5ad',
    f'{data_cache_dir}/gxe2_A172.h5ad',
    f'{data_cache_dir}/gxe2_T98G.h5ad',
    f'{data_cache_dir}/gxe2_U87MG.h5ad',
]


# In[ ]:


adata_list = []
for path in data_paths:
    adata_list.append(sc.read_h5ad(path))
adata = ad.concat(adata_list)
adata


# In[ ]:


adata.X = csr_matrix(adata.X)


# In[6]:


adata.obs_names_make_unique()


# In[7]:


del adata_list
gc.collect()


# In[8]:


adata.obs.dose.value_counts()


# In[9]:


adata.obs['drug_dose'] = adata.obs.dose.copy()


# In[10]:


adata.obs.cell_type = [x.lower() for x in adata.obs.cell_type]
adata.obs.cell_type = adata.obs.cell_type.astype('category')
adata.obs.cell_type.value_counts()


# In[11]:


adata.obs.treatment.value_counts()


# In[12]:


adata.obs.gene_id.value_counts()


# ## Rename metadata columns

# In[13]:


adata.obs.rename(columns = {
    'nCount_RNA': 'ncounts',
    'nFeature_RNA': 'ngenes',
}, inplace=True)
adata.obs['perturbation_type'] = 'CRISPRi'
adata.obs['dataset'] = 'mcfaline23'


# ## Rename perturbations

# In[14]:


adata.obs['gene_id'] = [x.replace(',', '+') for x in adata.obs.gene_id]

gene_controls = ['NA', 'NTC', 'random']
for ctrl in gene_controls:
    adata.obs['gene_id'] = [x.replace(ctrl, 'control') for x in adata.obs['gene_id']]
adata.obs.gene_id.value_counts()


# In[15]:


single_gene_perts = [x for x in adata.obs.gene_id.unique() if '+' not in x]
adata = adata[adata.obs.gene_id.isin(single_gene_perts),:]
adata


# In[16]:


drug_controls = ['vehicle', 'dmso']
for ctrl in drug_controls:
    adata.obs['treatment'] = [x.replace(ctrl, 'none') for x in adata.obs['treatment']]
adata.obs.treatment.value_counts()


# In[18]:


gene_dose = []
for gene in adata.obs.gene_id:
    ngenes = len(gene.split('+'))
    dose = '+'.join(['1']*ngenes)
    gene_dose.append(dose)
    
adata.obs['gene_dose'] = gene_dose
adata.obs.gene_dose.value_counts()


# In[19]:


adata.obs['perturbation'] = adata.obs.gene_id.astype('category').copy()
adata.obs.perturbation.value_counts()


# In[20]:


adata.obs['pert_cl_tr'] = adata.obs['perturbation'].astype(str) + '_' + adata.obs['cell_type'].astype(str) + '_' + adata.obs['treatment'].astype(str)
pert_cl_tr_counts = adata.obs.pert_cl_tr.value_counts()
pert_cl_tr_keep = list(pert_cl_tr_counts.loc[pert_cl_tr_counts >= 20].index)
print(len(pert_cl_tr_keep))


# In[21]:


adata.shape


# In[22]:


adata = adata[adata.obs.pert_cl_tr.isin(pert_cl_tr_keep)]
adata.shape


# In[26]:


adata.obs['condition'] = adata.obs['perturbation'].copy()
adata.obs.condition = adata.obs.condition.astype('category')
adata.obs.perturbation = adata.obs.perturbation.astype('category')


# In[27]:


adata


# In[29]:


adata.var['gene_id'] = [x.split('.')[0] for x in adata.var_names]
adata = adata[:,[x in id_to_gene for x in adata.var['gene_id']]]
adata.shape


# In[30]:


adata.var['gene_name'] = [str(id_to_gene[x]) for x in adata.var['gene_id']]
adata = adata[:,[x != '' for x in adata.var['gene_name']]]
adata.shape


# In[31]:


adata.var_names = adata.var.gene_name.astype(str).copy()


# In[32]:


adata.var.head()


# In[33]:


adata = adata[:,['nan' not in x for x in adata.var_names]]
adata.shape


# In[34]:


duplicated_genes = adata.var.index.duplicated()
adata = adata[:,~duplicated_genes]
adata.shape


# In[35]:


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


# In[38]:


adata = adata.copy()


# In[39]:


gc.collect()


# In[5]:


condition_plus_treatment = []
for condition, treatment in zip(adata.obs.condition, adata.obs.treatment):
    if treatment == 'none':
        condition_plus_treatment.append(str(condition))
    else:
        condition_plus_treatment.append(str(condition) + '+' + str(treatment))

adata.obs['condition_plus_treatment'] = condition_plus_treatment
adata.obs['condition_plus_treatment'] = adata.obs['condition_plus_treatment'].astype('category')
adata.obs.condition_plus_treatment.value_counts()


# In[3]:


unique_obs = adata.obs.loc[:,['condition', 'cell_type', 'treatment']].drop_duplicates()
unique_obs.treatment.value_counts()


# In[5]:


treatments_remove = [
    'temozolomide',
    'thioguanine'
]

adata = adata[~adata.obs.treatment.isin(treatments_remove)].to_memory()
adata


# In[ ]:


adata = preprocess(
    adata,
    perturbation_key='condition',
    covariate_keys=['cell_type', 'treatment'],
)


# In[6]:


adata.write_h5ad(f'{data_cache_dir}/mcfaline23_gxe_processed.h5ad')

