import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional


def parse_training_log(log_file_path: str) -> Optional[pd.DataFrame]:
    """
    Parse a training log file to extract epoch, train loss, and validation loss.
    
    Args:
        log_file_path: Path to the log file
        
    Returns:
        DataFrame with columns ['epoch', 'train_loss', 'vali_loss', 'test_loss'] or None if parsing fails
    """
    if not os.path.exists(log_file_path):
        print(f"Warning: Log file not found: {log_file_path}")
        return None
    
    try:
        with open(log_file_path, 'r') as f:
            lines = f.readlines()
        
        # Pattern to match: "Epoch: X, Steps: Y | Train Loss: Z Vali Loss: W Test Loss: V"
        pattern = r'Epoch:\s+(\d+),\s+Steps:\s+\d+\s+\|\s+Train Loss:\s+([\d.]+)\s+Vali Loss:\s+([\d.]+)\s+Test Loss:\s+([\d.]+)'
        
        epochs = []
        train_losses = []
        vali_losses = []
        test_losses = []
        
        for line in lines:
            match = re.search(pattern, line)
            if match:
                epochs.append(int(match.group(1)))
                train_losses.append(float(match.group(2)))
                vali_losses.append(float(match.group(3)))
                test_losses.append(float(match.group(4)))
        
        if not epochs:
            print(f"Warning: No training data found in {log_file_path}")
            return None
            
        return pd.DataFrame({
            'epoch': epochs,
            'train_loss': train_losses,
            'vali_loss': vali_losses,
            'test_loss': test_losses
        })
        
    except Exception as e:
        print(f"Error parsing {log_file_path}: {str(e)}")
        return None


def plot_training_curves(experiment_name: str, 
                        models: List[str] = None,
                        logs_dir: str = './logs',
                        figsize: Tuple[int, int] = (20, 4),
                        save_path: Optional[str] = None) -> None:
    """
    Plot training and validation curves for multiple models side by side.
    
    Args:
        experiment_name: Name of the experiment (folder name in logs directory)
        models: List of model names to plot. If None, defaults to ['DLinear', 'Informer', 'POCO', 'Transformer', 'TSMixer']
        logs_dir: Directory containing log folders
        figsize: Figure size as (width, height)
        save_path: Optional path to save the plot
    """
    
    if models is None:
        models = ['DLinear', 'Informer', 'POCO', 'Transformer', 'TSMixer']
    
    experiment_dir = os.path.join(logs_dir, experiment_name)
    
    if not os.path.exists(experiment_dir):
        print(f"Error: Experiment directory not found: {experiment_dir}")
        return
    
    # Set up the plot
    fig, axes = plt.subplots(1, len(models), figsize=figsize)
    if len(models) == 1:
        axes = [axes]
    
    # Color palette
    colors = {'train': '#2E86AB', 'validation': '#A23B72', 'test': '#F18F01'}
    
    for i, model in enumerate(models):
        # Find log file (flexible naming)
        log_files = [f for f in os.listdir(experiment_dir) if f.startswith(model) and f.endswith('.log')]
        
        if not log_files:
            print(f"Warning: No log file found for {model} in {experiment_dir}")
            axes[i].text(0.5, 0.5, f'No data\navailable\nfor {model}', 
                        ha='center', va='center', transform=axes[i].transAxes,
                        fontsize=12, alpha=0.6)
            axes[i].set_title(f'{model}', fontsize=14)
            continue
        
        log_file = log_files[0]  # Use first matching file
        log_path = os.path.join(experiment_dir, log_file)
        
        # Parse the log file
        df = parse_training_log(log_path)
        
        if df is None or df.empty:
            axes[i].text(0.5, 0.5, f'Failed to parse\nlog for {model}', 
                        ha='center', va='center', transform=axes[i].transAxes,
                        fontsize=12, alpha=0.6)
            axes[i].set_title(f'{model}', fontsize=14)
            continue
        
        # Plot training and validation curves
        axes[i].plot(df['epoch'], df['train_loss'], 
                    color=colors['train'], linewidth=2.5, 
                    label='Train Loss', marker='o', markersize=4)
        axes[i].plot(df['epoch'], df['vali_loss'], 
                    color=colors['validation'], linewidth=2.5, 
                    label='Validation Loss', marker='s', markersize=4)
        
        # Styling
        axes[i].set_title(f'{model}', fontsize=14, pad=10)
        axes[i].set_xlabel('Epoch', fontsize=11)
        if i == 0:  # Only add y-label to first subplot
            axes[i].set_ylabel('Loss (MAE)', fontsize=11)
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(fontsize=10)
        
        # Add final loss values as text
        final_train = df['train_loss'].iloc[-1]
        final_val = df['vali_loss'].iloc[-1]
        axes[i].text(0.02, 0.98, f'Final Train: {final_train:.4f}\nFinal Val: {final_val:.4f}',
                    transform=axes[i].transAxes, fontsize=9, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Overall title
    fig.suptitle(f'Training Curves - {experiment_name}', 
                fontsize=16, y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


def plot_loss_comparison(experiment_names: List[str], 
                        model: str = 'DLinear',
                        logs_dir: str = './logs',
                        figsize: Tuple[int, int] = (12, 6),
                        save_path: Optional[str] = None) -> None:
    """
    Compare training curves for a single model across multiple experiments.
    
    Args:
        experiment_names: List of experiment names to compare
        model: Model name to plot
        logs_dir: Directory containing log folders
        figsize: Figure size as (width, height)
        save_path: Optional path to save the plot
    """
    
    plt.figure(figsize=figsize)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(experiment_names)))
    
    for i, exp_name in enumerate(experiment_names):
        experiment_dir = os.path.join(logs_dir, exp_name)
        
        if not os.path.exists(experiment_dir):
            print(f"Warning: Experiment directory not found: {experiment_dir}")
            continue
        
        # Find log file
        log_files = [f for f in os.listdir(experiment_dir) if f.startswith(model) and f.endswith('.log')]
        
        if not log_files:
            print(f"Warning: No log file found for {model} in {experiment_dir}")
            continue
        
        log_path = os.path.join(experiment_dir, log_files[0])
        df = parse_training_log(log_path)
        
        if df is None or df.empty:
            continue
        
        # Plot both train and validation
        plt.plot(df['epoch'], df['train_loss'], 
                color=colors[i], linewidth=2, linestyle='-',
                label=f'{exp_name} - Train', alpha=0.8)
        plt.plot(df['epoch'], df['vali_loss'], 
                color=colors[i], linewidth=2, linestyle='--',
                label=f'{exp_name} - Validation', alpha=0.8)
    
    plt.title(f'{model} Training Curves Comparison', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MAE)', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


# Example usage:
if __name__ == "__main__":
    # Plot training curves for ActivityShort7000N experiment
    plot_training_curves('ActivityShort7000N')
    
    # Compare DLinear across different neuron counts
    # plot_loss_comparison(['ActivityShort70N', 'ActivityShort700N', 'ActivityShort7000N'], 
    #                     model='DLinear')