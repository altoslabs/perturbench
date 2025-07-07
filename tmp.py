
import json
import torch
import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path

filepath= "/cluster/scratch/fluebeck/perturbench_data/norman19_preprocessed_with_embeddings.h5ad"

adata = sc.read_h5ad(filepath)


gene_embedding = torch.nn.Embedding(60697, 512, padding_idx=60694)
gene_embedding.load_state_dict(torch.load("/cluster/scratch/fluebeck/perturbench_data/pretrained_models/scGPT_human/gene_embedding.pth"))

with open("/cluster/scratch/fluebeck/perturbench_data/pretrained_models/scGPT_human/vocab.json", "r") as f:
    vocab = json.load(f)
gene_vocab = vocab