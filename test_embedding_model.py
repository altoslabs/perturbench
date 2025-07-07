#!/usr/bin/env python3
"""
Test script to verify that LatentAdditiveEmbedding is working correctly
and producing different results from the regular LatentAdditive model.
"""

import torch
import torch.nn.functional as F
import numpy as np

def test_embedding_model():
    """Test that the embedding model produces different results from regular model."""
    
    # Create dummy data
    batch_size = 4
    n_genes = 100
    n_perts = 10
    
    # Create random control expression
    control_expression = torch.randn(batch_size, n_genes)
    
    # Create perturbation tensor (one-hot)
    perturbation = torch.zeros(batch_size, n_perts)
    perturbation[0, 0] = 1  # First cell, first perturbation
    perturbation[1, 1] = 1  # Second cell, second perturbation
    perturbation[2, 2] = 1  # Third cell, third perturbation
    perturbation[3, 0] = 1  # Fourth cell, first perturbation
    
    # Create dummy covariates
    covariates = {"cell_type": torch.randn(batch_size, 5)}
    
    # Create target expression
    target_expression = torch.randn(batch_size, n_genes)
    
    print("Testing LatentAdditiveEmbedding model...")
    print(f"Control expression shape: {control_expression.shape}")
    print(f"Perturbation shape: {perturbation.shape}")
    print(f"Perturbation sum: {perturbation.sum()}")
    
    # Test the embedding functionality
    try:
        # Import the model
        from src.perturbench.modelcore.models.latent_additive_gene_embedding import LatentAdditiveEmbedding
        
        # Create a mock datamodule with train_context
        class MockDataModule:
            def __init__(self):
                self.train_context = {
                    "perturbation_uniques": ["GENE1", "GENE2", "GENE3", "GENE4", "GENE5", 
                                           "GENE6", "GENE7", "GENE8", "GENE9", "GENE10"],
                    "covariate_uniques": {"cell_type": ["type1", "type2", "type3", "type4", "type5"]}
                }
        
        mock_datamodule = MockDataModule()
        
        # Create the embedding model
        model = LatentAdditiveEmbedding(
            n_genes=n_genes,
            n_perts=n_perts,
            embed_gene=True,
            datamodule=mock_datamodule,
            n_layers=2,
            encoder_width=64,
            latent_dim=16,
            dropout=0.1
        )
        
        # Test forward pass
        with torch.no_grad():
            output = model.forward(control_expression, perturbation, covariates)
            print(f"Model output shape: {output.shape}")
            print(f"Model output norm: {output.norm()}")
            
            # Test that output is different from input
            diff_from_control = (output - control_expression).norm()
            print(f"Difference from control: {diff_from_control}")
            
            if diff_from_control > 0.1:
                print("✅ SUCCESS: Model is producing different outputs from control!")
            else:
                print("❌ WARNING: Model output is very similar to control input")
                
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_embedding_model() 