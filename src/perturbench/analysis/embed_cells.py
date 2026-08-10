from pathlib import Path
import scanpy as sc
import sys
import argparse
import numpy as np
import pandas as pd
import anndata as ad
import torch

from perturbench.modelcore.models.embeddings import PCAEmbeddingModel


def download_scgpt_model(model_dir):
    """Download scGPT pretrained model files from Google Drive."""
    print(f"Downloading scGPT model to {model_dir}...")
    
    # Create model directory if it doesn't exist
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Try to import gdown for Google Drive downloads
    try:
        import gdown
    except ImportError as err:
        raise ImportError("gdown is required for scGPT embedding. Install with: pip install gdown") from err
    
    # Google Drive folder ID from the provided URL
    folder_id = "1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y"
    folder_url = f"https://drive.google.com/drive/folders/{folder_id}"
    
    # Required files and their expected sizes (for verification)
    required_files = {
        'args.json': 1024,  # ~1 KB
        'best_model.pt': 195.9 * 1024 * 1024,  # ~195.9 MB  
        'vocab.json': 1.3 * 1024 * 1024  # ~1.3 MB
    }
    
    print("Attempting to download from Google Drive folder...")
    try:
        # Download entire folder
        gdown.download_folder(folder_url, output=str(model_dir), quiet=False, use_cookies=False)
        
        # Check if files were downloaded successfully
        downloaded_files = []
        for filename in required_files.keys():
            file_path = model_dir / filename
            if file_path.exists():
                downloaded_files.append(filename)
                print(f"  ✓ {filename} downloaded successfully")
            else:
                print(f"  ✗ {filename} not found after download")
        
        if len(downloaded_files) == len(required_files):
            print("All required files downloaded successfully!")
            return True
            
    except Exception as e:
        print(f"Error downloading with gdown: {e}")
    
    # Fallback: Manual download instructions
    print("\nAutomatic download failed or gdown not available.")
    print("Please manually download the scGPT model files:")
    print(f"1. Visit: {folder_url}")
    print("2. Download the following files to the model directory:")
    for filename in required_files.keys():
        file_path = model_dir / filename
        status = "✓ Already exists" if file_path.exists() else "✗ Missing"
        print(f"   - {filename} ({status})")
    print(f"3. Place all files in: {model_dir.absolute()}")
    
    # Check what files we have after manual download instructions
    missing_files = [f for f in required_files.keys() if not (model_dir / f).exists()]
    
    if not missing_files:
        print("All required files are present!")
        return True
    else:
        print(f"Still missing files: {missing_files}")
        return False


def load_data_split(split_path):
    """Load data split information from file as a pandas Series."""
    if split_path.endswith('.csv'):
        split_df = pd.read_csv(split_path, header=None, names=['cell_id', 'split'])
    elif split_path.endswith('.txt'):
        split_df = pd.read_csv(split_path, sep='\t', header=None, names=['cell_id', 'split'])
    else:
        raise ValueError("Split file must be .csv or .txt format")
    
    # Convert to Series with cell_id as index and split as values
    split_series = split_df.set_index('cell_id')['split']
    return split_series


def embed_scgpt(adata, model_dir=None):
    """Embed data using scGPT."""
    try:
        import scgpt as scg
    except ImportError as err:
        raise ImportError("scgpt is required for scGPT embedding. Install with: pip install scgpt") from err
    
    if model_dir is None:
        model_dir = Path("pretrained_models/scGPT_human")
    
    model_dir = Path(model_dir)
    
    # Ensure the model directory exists (create if it doesn't)
    model_dir.mkdir(parents=True)
    print(f"Model directory: {model_dir.absolute()}")
    
    # Check if model directory exists and contains required files
    required_files = ['args.json', 'best_model.pt', 'vocab.json']
    missing_files = [f for f in required_files if not (model_dir / f).exists()]
    
    if not model_dir.exists() or missing_files:
        print(f"Model directory {model_dir} does not exist or is missing required files: {missing_files}")
        print("Attempting to download scGPT pretrained model...")
        
        success = download_scgpt_model(model_dir)
        if not success:
            print("Failed to download scGPT model. Please download manually.")
            sys.exit(1)
        
        # Verify download was successful
        missing_files = [f for f in required_files if not (model_dir / f).exists()]
        if missing_files:
            print(f"Download incomplete. Missing files: {missing_files}")
            sys.exit(1)
    
    # Prepare data for scGPT
    adata_copy = adata.copy()
    adata_copy.var["gene_symbol"] = adata_copy.var_names.values
    
    # Store normalized counts and use raw counts for scGPT
    if "counts" in adata_copy.layers:
        adata_copy.X = adata_copy.layers["counts"]
    
    embeddings = scg.tasks.embed_data(
        adata_copy,
        model_dir,
        gene_col="gene_symbol",
        batch_size=100,
        return_new_adata=True,
    )
    
    return embeddings.X


def train_pca(adata, train_test_split=None, n_comps=256):
    """Train PCA model on data."""
    if train_test_split is not None:
        # Fit PCA only on training data
        train_cells = train_test_split.loc[train_test_split == 'train'].index.values
        train_mask = adata.obs_names.isin(train_cells)
        train_adata = adata[train_mask].copy()
        print('Training on {} cells'.format(train_adata.shape[0]))
        
        # Fit PCA on training data
        sc.tl.pca(train_adata, n_comps=n_comps)
        
        # Get PCA components and mean
        pca_components = train_adata.varm['PCs'].T  # Transpose to (n_components, n_features)
        pca_mean = train_adata.X.mean(0)
        
    else:
        # Fit PCA on all data
        sc.tl.pca(adata, n_comps=n_comps)
        pca_components = adata.varm['PCs'].T  # Transpose to (n_components, n_features)
        pca_mean = adata.X.mean(0)
    
    pca_mean = np.array(pca_mean).squeeze()
    return PCAEmbeddingModel(
        feature_loadings=pca_components.copy(),
        mean_centering_vector=pca_mean.copy(),
        model_name='pca'
    )


def embed_pca(adata: ad.AnnData, pca_model: PCAEmbeddingModel) -> np.ndarray:
    """Embed data using PCA model."""
    # Use the PCAEmbeddingModel's encode method
    embeddings = pca_model.encode(adata.X)
    if isinstance(embeddings, torch.Tensor):
        if embeddings.device.type == 'cuda':
            embeddings = embeddings.cpu()
        embeddings = embeddings.to(torch.float32).numpy()
    return embeddings


def embed_data(
    datapath,
    method='pca',
    obsm_key=None,
    split_path=None,
    output_suffix='embeddings',
    latent_dim=256,
    output_path=None,
    model=None,  # Allow passing pre-trained models
    save_model_path=None,  # Path to save trained models
):
    """Main function to embed data using specified method."""
    print(f"Loading data from {datapath}")
    adata = sc.read_h5ad(datapath)
    print("Data summary:")
    print(adata)
    
    # Load data split if provided
    train_test_split = None
    if split_path:
        print(f"Loading data split from {split_path}")
        train_test_split = load_data_split(split_path)
        print(f"Split contains {len(train_test_split)} cells")
    
    # Generate embeddings based on method
    if method == 'scgpt':
        print("Generating scGPT embeddings...")
        embeddings = embed_scgpt(adata)
    
    elif method == 'pca':
        print("Generating PCA embeddings...")
        if model is None:
            pca_model = train_pca(adata, train_test_split, latent_dim)
        else:
            pca_model = model
        embeddings = embed_pca(adata, pca_model)
        
        # Save PCA model if path provided
        if save_model_path:
            pca_model.save(save_model_path)
            print(f"PCA model saved to: {save_model_path}")
    else:
        raise ValueError(f"Unknown embedding method: {method}")
    
    # Store embeddings
    if obsm_key is None:
        obsm_key = 'X_' + method
    adata.obsm[obsm_key] = embeddings
    print(f"Embeddings saved with key '{obsm_key}'")
    
    # Save results
    if output_suffix:
        output_path = datapath.replace(".h5ad", f"_{output_suffix}.h5ad")
    else:
        output_path = datapath
    
    print(f"Saving results to {output_path}")
    adata.write_h5ad(output_path)


def main():
    parser = argparse.ArgumentParser(description="Generate single-cell embeddings using various methods")
    parser.add_argument("--anndata", "-a", help="Path to input h5ad file")
    parser.add_argument("--method", "-m", help="Embedding method to use (pca, scgpt)")
    parser.add_argument("--split-path", "-s", help="Path to data split file (two-column CSV/TSV: cell_id, split)")
    parser.add_argument("--obsm-key", "-l", help="Key to save embeddings to (default: use the method name)")
    parser.add_argument("--output-suffix", "-o", help="Custom suffix for output file (default: method name)")
    parser.add_argument("--latent-dim", "-d", type=int, default=256,
                        help="Number of latent embedding dimensions (default: 256)")
    parser.add_argument("--model-path", help="Path to pre-trained model (optional)")
    parser.add_argument("--save-model-path", help="Path to save trained model to (optional)")
    
    args = parser.parse_args()
    
    # Load model if path provided
    model = None
    if args.model_path:
        if args.method == 'pca':
            model = PCAEmbeddingModel.load(args.model_path)
        else:
            print(f"Pre-trained models not supported for method: {args.method}")
    
    embed_data(
        datapath=args.anndata,
        method=args.method,
        split_path=args.split_path,
        obsm_key=args.obsm_key,
        output_suffix=args.output_suffix,
        latent_dim=args.latent_dim,
        model=model,
        save_model_path=args.save_model_path,
    )


if __name__ == "__main__":
    main()
