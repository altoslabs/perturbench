
import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path

filepath = '/cluster/scratch/fluebeck/perturbench_data/frangieh21_processed_with_embeddings.h5ad'
adata = sc.read_h5ad(filepath)
print('Before renaming:')
print('obsm keys:', list(adata.obsm.keys()))

# Rename the embedding
adata.obsm['scgpt_embeddings'] = adata.obsm['scgpt_embbeddings']
del adata.obsm['scgpt_embbeddings']

print('After renaming:')
print('obsm keys:', list(adata.obsm.keys()))

# Save the updated file
adata.write_h5ad(filepath)
print('File updated successfully!')
