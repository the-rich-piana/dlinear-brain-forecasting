#!/usr/bin/env python3
"""
Visualization script for POCO unit embeddings from your trained model.
Adapted from the original POCO embedding visualization code.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import argparse
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from pathlib import Path

# Import your POCO model
from models.POCO import POCO, NeuralPredictionConfig

def reduce_dimensionality(data: np.ndarray, method: str) -> np.ndarray:
    """Reduce data to 2D using specified method."""
    if method == 'PCA':
        return PCA(n_components=2).fit_transform(data)
    elif method == 'TSNE':
        perplexity = max(5, min(data.shape[0] // 10, 30))
        return TSNE(n_components=2, random_state=42, perplexity=perplexity).fit_transform(data)
    elif method == 'UMAP':
        return umap.UMAP(n_components=2, random_state=42).fit_transform(data)
    else:
        raise ValueError(f"Unknown method {method}")

def load_poco_model(checkpoint_path, num_neurons=7000):
    """Load trained POCO model from checkpoint."""
    # Create config matching your training setup
    config = NeuralPredictionConfig()
    config.seq_length = 64  # 48 + 16 from your script
    config.pred_length = 16
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # POCO-specific parameters (using defaults from standalone)
    config.compression_factor = 16
    config.conditioning_dim = 1024
    config.decoder_hidden_size = 128
    config.decoder_num_layers = 1
    config.decoder_num_heads = 16
    config.poyo_num_latents = 8
    
    # Single session with your neuron count
    input_size = [[num_neurons]]
    
    # Load model
    model = POCO(config, input_size)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Load state dict, handling potential key mismatches
    model_dict = model.state_dict()
    filtered_dict = {}
    
    for k, v in state_dict.items():
        # Remove any module prefix if it exists
        key = k.replace('module.', '') if k.startswith('module.') else k
        if key in model_dict and v.shape == model_dict[key].shape:
            filtered_dict[key] = v
        else:
            print(f"Skipping {key}: shape mismatch or key not found")
    
    model.load_state_dict(filtered_dict, strict=False)
    model.eval()
    
    return model, config

def extract_embeddings(model):
    """Extract unit and session embeddings from the POCO model."""
    # Get unit embeddings from the decoder
    unit_embeddings = model.decoder.unit_emb.weight.detach().cpu().numpy()
    session_embeddings = model.decoder.session_emb.weight.detach().cpu().numpy()
    
    print(f"Unit embeddings shape: {unit_embeddings.shape}")
    print(f"Session embeddings shape: {session_embeddings.shape}")
    
    return unit_embeddings, session_embeddings

def visualize_embeddings(embeddings, method='PCA', title_prefix="", save_path=None):
    """Visualize embeddings in 2D."""
    if embeddings.shape[0] < 2:
        print(f"Not enough data points for {method} visualization")
        return
    
    # Reduce to 2D
    reduced = reduce_dimensionality(embeddings, method)
    
    plt.figure(figsize=(10, 8))
    
    # For many neurons, use small points and transparency
    if embeddings.shape[0] > 1000:
        plt.scatter(reduced[:, 0], reduced[:, 1], s=1, alpha=0.6, c='blue')
    else:
        plt.scatter(reduced[:, 0], reduced[:, 1], s=20, alpha=0.7, c='blue')
    
    plt.title(f"{title_prefix} Embeddings ({method})")
    plt.xlabel(f"{method} Dimension 1")
    plt.ylabel(f"{method} Dimension 2")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()

def analyze_embedding_structure(embeddings, name="Embeddings"):
    """Analyze basic properties of the embeddings."""
    print(f"\n=== {name} Analysis ===")
    print(f"Shape: {embeddings.shape}")
    print(f"Mean: {embeddings.mean(axis=0)[:5]}...")  # Show first 5 dims
    print(f"Std: {embeddings.std(axis=0)[:5]}...")   # Show first 5 dims
    print(f"Min: {embeddings.min():.4f}")
    print(f"Max: {embeddings.max():.4f}")
    
    # Compute pairwise distances (sample subset if too large)
    if embeddings.shape[0] > 1000:
        idx = np.random.choice(embeddings.shape[0], 1000, replace=False)
        sample_embeddings = embeddings[idx]
    else:
        sample_embeddings = embeddings
    
    from scipy.spatial.distance import pdist
    distances = pdist(sample_embeddings)
    print(f"Pairwise distances - Mean: {distances.mean():.4f}, Std: {distances.std():.4f}")

def main():
    parser = argparse.ArgumentParser(description='Visualize POCO unit embeddings')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to POCO model checkpoint')
    parser.add_argument('--num_neurons', type=int, default=7000,
                        help='Number of neurons in the dataset')
    parser.add_argument('--methods', nargs='+', default=['PCA', 'UMAP'],
                        choices=['PCA', 'TSNE', 'UMAP'],
                        help='Dimensionality reduction methods to use')
    parser.add_argument('--output_dir', type=str, default='./embedding_visualizations',
                        help='Directory to save visualizations')
    parser.add_argument('--analyze_only', action='store_true',
                        help='Only analyze embeddings without visualization')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading POCO model from: {args.checkpoint}")
    
    try:
        # Load model
        model, config = load_poco_model(args.checkpoint, args.num_neurons)
        
        # Extract embeddings
        unit_embeddings, session_embeddings = extract_embeddings(model)
        
        # Analyze embeddings
        analyze_embedding_structure(unit_embeddings, "Unit Embeddings")
        analyze_embedding_structure(session_embeddings, "Session Embeddings")
        
        if not args.analyze_only:
            # Visualize unit embeddings
            for method in args.methods:
                print(f"\nCreating {method} visualization for unit embeddings...")
                save_path = output_dir / f"unit_embeddings_{method.lower()}.png"
                visualize_embeddings(unit_embeddings, method, "Unit", save_path)
            
            # Visualize session embeddings if more than one session
            if session_embeddings.shape[0] > 1:
                for method in args.methods:
                    print(f"\nCreating {method} visualization for session embeddings...")
                    save_path = output_dir / f"session_embeddings_{method.lower()}.png"
                    visualize_embeddings(session_embeddings, method, "Session", save_path)
            else:
                print("Only one session - skipping session embedding visualization")
        
        print("\nEmbedding analysis complete!")
        
    except Exception as e:
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    main()