#!/usr/bin/env python3
"""
POCO embedding visualization with brain region coloring.
Uses your region labels CSV file.
"""
# run from root dir

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from pathlib import Path
import seaborn as sns

def load_region_labels(csv_path, max_neurons=7000):
    """Load region labels from your CSV file."""
    print(f"Loading region labels from: {csv_path}")
    
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Take only first 7000 neurons (matching your training data)
    df = df.head(max_neurons)
    
    # Get unique regions and assign integer IDs
    unique_regions = sorted(df['region_label'].unique())
    region_to_id = {region: i for i, region in enumerate(unique_regions)}
    
    # Create region ID array
    region_ids = df['region_label'].map(region_to_id).values
    
    print(f"Found {len(unique_regions)} brain regions:")
    for i, region in enumerate(unique_regions):
        count = (region_ids == i).sum()
        print(f"  {region}: {count} neurons")
    
    return region_ids, unique_regions

def reduce_dimensionality(data: np.ndarray, method: str) -> np.ndarray:
    """Reduce data to 2D using specified method."""
    if method == 'PCA':
        return PCA(n_components=5, random_state=42).fit_transform(data)
    elif method == 'TSNE':
        perplexity = max(5, min(data.shape[0] // 10, 30))
        return TSNE(n_components=5, random_state=42, perplexity=perplexity).fit_transform(data)
    elif method == 'UMAP':
        return umap.UMAP(n_components=2, random_state=42).fit_transform(data)
    else:
        raise ValueError(f"Unknown method {method}")

def load_poco_embeddings(checkpoint_path):
    """Load unit embeddings from POCO checkpoint."""
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract state dict
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # Find unit embedding key
    unit_emb_key = None
    for key in state_dict.keys():
        if 'unit_emb.weight' in key:
            unit_emb_key = key
            break
    
    if unit_emb_key is None:
        raise ValueError("Could not find unit embedding weights in checkpoint")
    
    # Extract embeddings
    unit_embeddings = state_dict[unit_emb_key].cpu().numpy()
    print(f"Unit embeddings shape: {unit_embeddings.shape}")
    
    return unit_embeddings

def create_color_palette(n_regions):
    """Create a distinct color palette for brain regions."""
    if n_regions <= 10:
        # Use matplotlib default colors
        return [f'C{i}' for i in range(n_regions)]
    elif n_regions <= 20:
        # Use tab20 colormap
        cmap = plt.cm.hsv
        return [cmap(i) for i in np.linspace(0, 1, n_regions)]
    else:
        # Use hsv colormap for many regions
        cmap = plt.cm.hsv
        return [cmap(i) for i in np.linspace(0, 1, n_regions)]

def visualize_embeddings_by_region(embeddings, region_ids, region_names, method='PCA', save_path=None):
    """Visualize embeddings colored by brain region."""
    
    # Reduce to 2D
    reduced = reduce_dimensionality(embeddings, method)
    
    # Create colors
    colors = create_color_palette(len(region_names))
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: All regions together
    for i, region in enumerate(region_names):
        mask = region_ids == i
        if mask.sum() > 0:
            ax1.scatter(reduced[mask, 0], reduced[mask, 1], 
                       c=[colors[i]], label=region, s=10, alpha=0.6)
    
    ax1.set_title(f'Unit Embeddings by Brain Region ({method})')
    ax1.set_xlabel(f'{method} Dimension 1')
    ax1.set_ylabel(f'{method} Dimension 2')
    ax1.grid(True, alpha=0.3)
    
    # Add legend outside plot
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Plot 2: Region density heatmap
    from scipy.stats import gaussian_kde
    
    # Create a density plot
    ax2.scatter(reduced[:, 0], reduced[:, 1], c='lightgray', s=2, alpha=0.5)
    
    # Highlight a few major regions with density contours
    region_counts = [(i, (region_ids == i).sum()) for i in range(len(region_names))]
    region_counts.sort(key=lambda x: x[1], reverse=True)
    
    for rank, (i, count) in enumerate(region_counts[:5]):  # Top 5 regions
        if count > 10:  # Only if enough neurons
            mask = region_ids == i
            points = reduced[mask]
            if len(points) > 5:
                try:
                    kde = gaussian_kde(points.T)
                    x_min, x_max = reduced[:, 0].min(), reduced[:, 0].max()
                    y_min, y_max = reduced[:, 1].min(), reduced[:, 1].max()
                    xx, yy = np.mgrid[x_min:x_max:50j, y_min:y_max:50j]
                    positions = np.vstack([xx.ravel(), yy.ravel()])
                    f = kde(positions).reshape(xx.shape)
                    ax2.contour(xx, yy, f, levels=3, colors=[colors[i]], alpha=0.6)
                except:
                    pass  # Skip if KDE fails
    
    ax2.set_title(f'Regional Density Patterns ({method})')
    ax2.set_xlabel(f'{method} Dimension 1')
    ax2.set_ylabel(f'{method} Dimension 2')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()

def analyze_regional_embedding_structure(embeddings, region_ids, region_names):
    """Analyze embedding structure by brain region."""
    print("\n=== Regional Embedding Analysis ===")
    
    # Compute within-region and between-region distances
    from scipy.spatial.distance import pdist, cdist
    
    region_stats = []
    
    for i, region in enumerate(region_names):
        mask = region_ids == i
        if mask.sum() < 2:
            continue
            
        region_embeddings = embeddings[mask]
        
        # Within-region distances
        within_distances = pdist(region_embeddings)
        
        # Between-region distances (to all other regions)
        other_mask = ~mask
        if other_mask.sum() > 0:
            between_distances = cdist(region_embeddings, embeddings[other_mask]).flatten()
        else:
            between_distances = []
        
        region_stats.append({
            'region': region,
            'n_neurons': mask.sum(),
            'within_dist_mean': within_distances.mean() if len(within_distances) > 0 else 0,
            'within_dist_std': within_distances.std() if len(within_distances) > 0 else 0,
            'between_dist_mean': between_distances.mean() if len(between_distances) > 0 else 0,
            'between_dist_std': between_distances.std() if len(between_distances) > 0 else 0,
        })
    
    # Print results
    print(f"{'Region':<15} {'N':<6} {'Within Dist':<12} {'Between Dist':<12} {'Separation':<10}")
    print("-" * 70)
    
    for stats in sorted(region_stats, key=lambda x: x['n_neurons'], reverse=True):
        separation = stats['between_dist_mean'] / stats['within_dist_mean'] if stats['within_dist_mean'] > 0 else 0
        print(f"{stats['region']:<15} {stats['n_neurons']:<6} "
              f"{stats['within_dist_mean']:<6.3f}±{stats['within_dist_std']:<5.3f} "
              f"{stats['between_dist_mean']:<6.3f}±{stats['between_dist_std']:<5.3f} "
              f"{separation:<6.3f}")

def main():
    # Configuration
    #PASSIVE:
    # checkpoint_path = "/cs/student/msc/aibh/2024/gcosta/DLinear/checkpoints/ActivityLongPassive11392N/POCO_48_16_POCO_ActivityBehavioral_ftM_sl48_ll16_pl16_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_Exp_0/checkpoint.pth"
    #ACTIVE:
    checkpoint_path = "/cs/student/msc/aibh/2024/gcosta/DLinear/checkpoints/ActivityLongActive11392N/POCO_48_16_POCO_Activity_ftM_sl48_ll16_pl16_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_Exp_0/checkpoint.pth"
    region_csv_path = "/cs/student/msc/aibh/2024/gcosta/DLinear/Experiment_Results/Exp_2/regions_5ea6bb9b-6163-4e8a-816b-efe7002666b0_validation.csv"
    output_dir = "/cs/student/msc/aibh/2024/gcosta/DLinear/Experiment_Results/figures"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading POCO EMBEDDINGS")
    embeddings = load_poco_embeddings(checkpoint_path)

    print("Loading brain region labels...")
    region_ids, region_names = load_region_labels(region_csv_path, max_neurons=embeddings.shape[0])
    
    # Verify dimensions match
    if len(region_ids) != embeddings.shape[0]:
        print(f"Warning: Mismatch between embeddings ({embeddings.shape[0]}) and regions ({len(region_ids)})")
        min_size = min(len(region_ids), embeddings.shape[0])
        embeddings = embeddings[:min_size]
        region_ids = region_ids[:min_size]
    
    # Analyze embedding structure by region
    analyze_regional_embedding_structure(embeddings, region_ids, region_names)
    
    # Create visualizations
    methods = ['PCA', 'UMAP']  # Skip TSNE for speed with 7000 neurons
    
    for method in methods:
        print(f"\nCreating {method} visualization...")
        save_path = os.path.join(output_dir, f"embeddings_by_region_{method.lower()}.png")
        visualize_embeddings_by_region(embeddings, region_ids, region_names, method, save_path)
    
    print(f"\nVisualization complete! Check {output_dir}/ for results.")

if __name__ == "__main__":
    main()