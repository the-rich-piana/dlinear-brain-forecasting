#!/usr/bin/env python3
"""
Simple POCO embedding visualization - minimal dependencies version.
Run this if the full version has import issues.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
import sys

def simple_visualize():
    """Simple visualization with minimal imports."""
    
    # Path to your trained model
    checkpoint_path = "./checkpoints/ActivityLongBehavioral7000N/POCO_48_16_POCO_ActivityBehavioral_ftM_sl48_ll16_pl16_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_Exp_0/checkpoint.pth"
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at: {checkpoint_path}")
        print("Please update the path in the script.")
        return
    
    # Load checkpoint
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract state dict
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # Look for embedding weights
    unit_emb_key = None
    session_emb_key = None
    
    for key in state_dict.keys():
        if 'unit_emb.weight' in key:
            unit_emb_key = key
        elif 'session_emb.weight' in key:
            session_emb_key = key
    
    if unit_emb_key is None:
        print("Could not find unit embedding weights in checkpoint.")
        print("Available keys:", list(state_dict.keys())[:10])
        return
    
    # Extract embeddings
    unit_embeddings = state_dict[unit_emb_key].cpu().numpy()
    
    if session_emb_key:
        session_embeddings = state_dict[session_emb_key].cpu().numpy()
        print(f"Session embeddings shape: {session_embeddings.shape}")
    
    print(f"Unit embeddings shape: {unit_embeddings.shape}")
    
    # Create visualizations
    os.makedirs('./embedding_viz', exist_ok=True)
    
    # PCA visualization of unit embeddings
    if unit_embeddings.shape[0] >= 2:
        print("Creating PCA visualization...")
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(unit_embeddings)
        
        plt.figure(figsize=(10, 8))
        plt.scatter(reduced[:, 0], reduced[:, 1], s=1, alpha=0.6)
        plt.title(f'Unit Embeddings PCA (n={unit_embeddings.shape[0]} neurons)')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.grid(True, alpha=0.3)
        plt.savefig('./embedding_viz/unit_embeddings_pca.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
        print(f"Total variance explained: {pca.explained_variance_ratio_.sum():.3f}")
    
    # Basic statistics
    print(f"\nUnit Embedding Statistics:")
    print(f"Mean: {unit_embeddings.mean():.4f}")
    print(f"Std: {unit_embeddings.std():.4f}")
    print(f"Min: {unit_embeddings.min():.4f}")
    print(f"Max: {unit_embeddings.max():.4f}")
    
    # Sample some neuron embeddings for visualization
    if unit_embeddings.shape[0] > 100:
        sample_idx = np.random.choice(unit_embeddings.shape[0], 100, replace=False)
        sample_embeddings = unit_embeddings[sample_idx]
        
        # Plot embedding distribution
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.hist(unit_embeddings.flatten(), bins=50, alpha=0.7)
        plt.title('Embedding Value Distribution')
        plt.xlabel('Embedding Value')
        plt.ylabel('Frequency')
        
        plt.subplot(1, 3, 2)
        plt.imshow(sample_embeddings[:50].T, aspect='auto', cmap='coolwarm')
        plt.title('Sample Embeddings Heatmap')
        plt.xlabel('Neuron Index')
        plt.ylabel('Embedding Dimension')
        plt.colorbar()
        
        plt.subplot(1, 3, 3)
        norms = np.linalg.norm(unit_embeddings, axis=1)
        plt.hist(norms, bins=50, alpha=0.7)
        plt.title('Embedding Norm Distribution')
        plt.xlabel('L2 Norm')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('./embedding_viz/embedding_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print("\nVisualization complete! Check ./embedding_viz/ for results.")

if __name__ == "__main__":
    simple_visualize()