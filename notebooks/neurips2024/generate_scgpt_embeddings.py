#!/usr/bin/env python
# coding: utf-8

# # 2024-05-25-Preprocessing: Generating scGPT embeddings using the pretrained scGPT model

# In[1]:


import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mode
import scanpy as sc
import sklearn
import warnings

import scgpt as scg


# In[2]:


data_cache_dir = './perturbench_data' ## Change this to your local data directory


# In[3]:


# whole-human pretrained model is downloaded from:
# https://github.com/bowang-lab/scGPT?tab=readme-ov-file#pretrained-scgpt-model-zoo

model_dir = Path(f"{data_cache_dir}/pretrained_models/scGPT_human")


# In[4]:


import os
os.listdir("perturbench_data/pretrained_models/scGPT_human")


# ### norman19

# In[6]:


datapath = f'{data_cache_dir}/norman19_processed.h5ad'


# In[7]:


adata = sc.read_h5ad(datapath)


# In[8]:


adata_copy = adata.copy()


# In[9]:


adata_copy.X = adata_copy.layers['counts']


# In[10]:


adata.var.head()


# In[58]:


adata_copy_embeddings = scg.tasks.embed_data(
    adata_copy,
    model_dir,
    gene_col="index",#,'gene_symbol',
    batch_size=128,
    return_new_adata=True,
)


# In[59]:


adata_copy_embeddings.X.shape


# In[60]:


adata.obsm['scgpt_embeddings'] = adata_copy_embeddings.X


# In[61]:


adata


# In[62]:


outfile = f'{data_cache_dir}/norman19_preprocessed_with_embeddings.h5ad'


# In[63]:


adata.write_h5ad(outfile)


# In[ ]:





# ### mcfaline23

# In[18]:


datapath = f'{data_cache_dir}/frangieh21_processed.h5ad'


# In[19]:


adata = sc.read_h5ad(datapath)


# In[20]:


adata


# In[ ]:





# In[21]:


adata_copy = adata.copy()


# In[7]:


adata_copy.X = adata_copy.layers['counts']


# In[23]:


adata_copy_embeddings = scg.tasks.embed_data(
    adata_copy,
    model_dir,
    gene_col="index",
    batch_size=128,
    return_new_adata=True,
)


# In[24]:


adata.obsm['scgpt_embeddings'] = adata_copy_embeddings.X


# In[25]:


adata


# In[26]:


outfile = f'{data_cache_dir}/frangieh21_processed_with_embeddings.h5ad'


# In[27]:


adata.write_h5ad(outfile)


# In[31]:


adata


# In[32]:


import scanpy as sc
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import anndata

embedding_data = anndata.AnnData(X=adata.obsm['scgpt_embeddings'])

# Run UMAP
sc.pp.neighbors(embedding_data, use_rep='X')
sc.tl.umap(embedding_data)

# Plot UMAP
sc.pl.umap(embedding_data, size=5, title="UMAP of scGPT Embeddings")


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


adata.obsm['scgpt_embeddings'] = adata_copy_embeddings.X


# In[20]:


adata


# In[21]:


outfile = f'{data_cache_dir}/srivatsan20_highest_processed_with_embeddings.h5ad'


# In[22]:


adata.write_h5ad(outfile)


# In[ ]:





# In[15]:


import json
import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import scanpy as sc
import torch
from anndata import AnnData
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

from scgpt import logger
from scgpt.data_collator import DataCollator
from scgpt.model import TransformerModel
from scgpt.tokenizer import GeneVocab
from scgpt.utils import load_pretrained

gene_col = "index"
if gene_col == "index":
    adata.var["index"] = adata.var.index

PathLike = Union[str, os.PathLike]

# LOAD MODEL
model_dir = Path(model_dir)
vocab_file = model_dir / "vocab.json"
model_config_file = model_dir / "args.json"
model_file = model_dir / "best_model.pt"
pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]

# vocabulary
vocab = GeneVocab.from_file(vocab_file)
for s in special_tokens:
    if s not in vocab:
        vocab.append_token(s)
adata.var["id_in_vocab"] = [
    vocab[gene] if gene in vocab else -1 for gene in adata.var[gene_col]
]
gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
logger.info(
    f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
    f"in vocabulary of size {len(vocab)}."
)
adata = adata[:, adata.var["id_in_vocab"] >= 0]

with open(model_config_file, "r") as f:
    model_configs = json.load(f)

# Binning will be applied after tokenization. A possible way to do is to use the unified way of binning in the data collator.

vocab.set_default_index(vocab["<pad>"])
genes = adata.var[gene_col].tolist()
gene_ids = np.array(vocab(genes), dtype=int)

# all_counts = adata.layers["counts"]
# num_of_non_zero_genes = [
#     np.count_nonzero(all_counts[i]) for i in range(all_counts.shape[0])
# ]
# max_length = min(max_length, np.max(num_of_non_zero_genes) + 1)

model = TransformerModel(
    ntoken=len(vocab),
    d_model=model_configs["embsize"],
    nhead=model_configs["nheads"],
    d_hid=model_configs["d_hid"],
    nlayers=model_configs["nlayers"],
    nlayers_cls=model_configs["n_layers_cls"],
    n_cls=1,
    vocab=vocab,
    dropout=model_configs["dropout"],
    pad_token=model_configs["pad_token"],
    pad_value=model_configs["pad_value"],
    do_mvc=True,
    do_dab=False,
    use_batch_labels=False,
    domain_spec_batchnorm=False,
    explicit_zero_prob=False,
    use_fast_transformer=False,
    fast_transformer_backend="flash",
    pre_norm=False,
)
load_pretrained(model, torch.load(model_file, map_location="cuda"), verbose=False)
model.to("cuda")
model.eval()


# In[17]:


model


# In[ ]:




