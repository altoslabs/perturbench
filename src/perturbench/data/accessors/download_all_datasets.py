#!/usr/bin/env python3
"""
Script to download all PerturbBench datasets to a specified data cache directory.

This script instantiates all available dataset accessors and downloads their data
to the specified cache directory. Each dataset will be processed and saved as
{dataset_name}_processed.h5ad in the cache directory.

Usage:
    python download_all_datasets.py [--data_cache_dir PATH]

Example:
    python download_all_datasets.py --data_cache_dir /path/to/data
    python download_all_datasets.py  # Uses default ../perturbench_data
"""

import argparse
import sys
import os

# Import all dataset accessors
from perturbench.data.accessors.norman19 import Norman19
from perturbench.data.accessors.srivatsan20 import Sciplex3
from perturbench.data.accessors.frangieh21 import Frangieh21
from perturbench.data.accessors.mcfaline23 import McFaline23
from perturbench.data.accessors.jiang24 import Jiang24
from perturbench.data.accessors.op3 import OP3


def download_all_datasets(data_cache_dir="../perturbench_data"):
    """
    Download all available PerturbBench datasets to the specified cache directory.
    
    Args:
        data_cache_dir (str): Directory where datasets will be cached
        
    Returns:
        dict: Summary of download results for each dataset
    """
    
    # Define all available dataset accessors
    datasets = {
        "norman19": Norman19,
        "sciplex3": Sciplex3,
        "frangieh21": Frangieh21,
        "mcfaline23": McFaline23,
        "jiang24": Jiang24,
        "op3": OP3,
    }
    
    results = {}
    
    print(f"Starting download of {len(datasets)} datasets to: {data_cache_dir}")
    print("=" * 60)
    
    for dataset_name, accessor_class in datasets.items():
        print(f"\n[{dataset_name.upper()}] Starting download...")
        
        try:
            # Instantiate the accessor
            accessor = accessor_class(data_cache_dir=data_cache_dir)
            
            # Download and process the data
            adata = accessor.get_anndata()
            
            # Report success
            print(f"[{dataset_name.upper()}] ✓ Successfully downloaded and processed")
            print(f"[{dataset_name.upper()}] Shape: {adata.shape[0]} cells × {adata.shape[1]} genes")
            print(f"[{dataset_name.upper()}] Saved to: {accessor.processed_data_path}")
            
            results[dataset_name] = {
                "status": "success",
                "shape": adata.shape,
                "path": accessor.processed_data_path
            }
            
            if accessor.split_hf_url is not None:
                accessor.get_split()
                print(f"[{dataset_name.upper()}] Split saved to: {accessor.split_data_path}")
                results[dataset_name]["split_path"] = accessor.split_data_path
            
        except Exception as e:
            print(f"[{dataset_name.upper()}] ✗ Failed to download: {str(e)}")
            results[dataset_name] = {
                "status": "failed",
                "error": str(e)
            }
    
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    
    successful = 0
    failed = 0
    
    for dataset_name, result in results.items():
        if result["status"] == "success":
            print(f"✓ {dataset_name}: {result['shape'][0]} cells × {result['shape'][1]} genes")
            successful += 1
        else:
            print(f"✗ {dataset_name}: {result['error']}")
            failed += 1
    
    print(f"\nTotal: {successful} successful, {failed} failed")
    
    if failed > 0:
        print("\nCache directory contents:")
        if os.path.exists(data_cache_dir):
            for file in os.listdir(data_cache_dir):
                if file.endswith('.h5ad'):
                    file_path = os.path.join(data_cache_dir, file)
                    size_mb = os.path.getsize(file_path) / (1024 * 1024)
                    print(f"  {file}: {size_mb:.1f} MB")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Download all PerturbBench datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
                python download_all_datasets.py
                python download_all_datasets.py --data_cache_dir /path/to/data
                python download_all_datasets.py --data_cache_dir ~/perturbench_data
        """
    )
    
    parser.add_argument(
        "--data-cache-dir",
        type=str,
        default="../perturbench_data",
        help="Directory to cache downloaded datasets (default: ../perturbench_data)"
    )
    
    args = parser.parse_args()
    
    # Expand user path if needed
    data_cache_dir = os.path.expanduser(args.data_cache_dir)
    
    # Create directory if it doesn't exist
    os.makedirs(data_cache_dir, exist_ok=True)
    
    print("PerturbBench Dataset Downloader")
    print(f"Cache directory: {os.path.abspath(data_cache_dir)}")
    
    # Download all datasets
    results = download_all_datasets(data_cache_dir)
    
    # Exit with error code if any downloads failed
    failed_count = sum(1 for r in results.values() if r["status"] == "failed")
    if failed_count > 0:
        sys.exit(1)
    else:
        print("\n🎉 All datasets downloaded successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
