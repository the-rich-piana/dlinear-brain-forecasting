import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

sns.set_theme(style="white")
plt.rcParams.update({'figure.dpi': 300, 'savefig.dpi': 300})

# Load all datasets
datasets = {
    'short': {
        70: pd.read_csv("ActivityShort70N.csv"),
        700: pd.read_csv("ActivityShort700N.csv"),
        7000: pd.read_csv("ActivityShort7000N.csv")
    },
    'medium': {
        70: pd.read_csv("ActivityLong70N.csv"),
        700: pd.read_csv("ActivityLong700N.csv"), 
        7000: pd.read_csv("ActivityLong7000N.csv")
    },
    'long': {
        70: pd.read_csv("ActivityExtraLong70N.csv"),
        700: pd.read_csv("ActivityExtraLong700N.csv"),
        7000: pd.read_csv("ActivityExtraLong7000N.csv")
    }
}

# Models to plot (exclude Mean, include Naive reference)
models_to_plot = ['DLinear', 'Linear', 'POCO', 'TSMixer', 'Informer', 'Transformer']
neuron_labels = ['70 Neurons', '700 Neurons', '7000 Neurons']
prediction_labels = ['8 steps ahead', '16 steps ahead', '32 steps ahead']

# Create 3x3 subplot grid
fig, axes = plt.subplots(3, 3, figsize=(15, 12))

# Define color palette and create consistent color mapping
colors = sns.color_palette("Set2", n_colors=len(models_to_plot))
color_map = dict(zip(models_to_plot, colors))

# Plot data
neuron_counts = [70, 700, 7000]
context_names = ['short', 'medium', 'long']

for i, neurons in enumerate(neuron_counts):
    for j, (context, pred_label) in enumerate(zip(context_names, prediction_labels)):
        ax = axes[i, j]
        
        # Get data for this specific neuron count and prediction horizon
        plot_data = datasets[context][neurons]
        
        # Calculate mean and std for each model (handling multiple training runs)
        model_stats = plot_data.groupby('model')['mae'].agg(['mean', 'std']).reset_index()
        naive_baseline = model_stats[model_stats['model'] == 'Naive']['mean'].iloc[0]
        
        # Filter to models we want to plot
        model_data = model_stats[model_stats['model'].isin(models_to_plot)].copy()
        model_data = model_data.sort_values('mean')
        
        # Look up consistent colors for each model
        model_colors = [color_map[model] for model in model_data['model']]
        
        # Create bar plot with error bars
        bars = ax.bar(range(len(model_data)), model_data['mean'], 
                     yerr=model_data['std'], capsize=3, width=1.0,
                     color=model_colors, alpha=0.8, 
                     error_kw={'elinewidth': 0.5, 'capthick': 2, 'ecolor': 'black', 'alpha': 0.5})
        
        # Add dark horizontal lines at top of each bar
        for k, (bar, color) in enumerate(zip(bars, model_colors)):
            bar_height = bar.get_height()
            bar_x = bar.get_x()
            bar_width = bar.get_width()
            ax.plot([bar_x, bar_x + bar_width], [bar_height, bar_height], 
                   color=color, linewidth=3, solid_capstyle='butt', zorder=1)
        
        # Add Naive baseline as dotted line
        ax.axhline(y=naive_baseline, color='black', linestyle=':', linewidth=1.5, alpha=0.8)
        
        # Formatting
        ax.set_xticks(range(len(model_data)))
        ax.set_xticklabels(model_data['model'], rotation=45, ha='right', fontsize=9)
        
        # Only add y-axis label on leftmost column
        if j == 0:
            ax.set_ylabel('MAE', fontsize=10)
        
        # Add neuron count label on leftmost plots (moved further left)
        if j == 0:
            ax.text(-0.25, 0.5, neuron_labels[i], rotation=90, verticalalignment='center',
                   transform=ax.transAxes, fontsize=11, fontweight='bold')
        
        # Add prediction label on top plots
        if i == 0:
            ax.set_title(pred_label, fontsize=11, fontweight='bold', pad=15)
        
        # Set y-axis limits per row (same neuron count)
        if j == 0:  # Calculate row limits only once per row
            row_maes = []
            for ctx in context_names:
                ctx_data = datasets[ctx][neurons]
                row_maes.extend(ctx_data['mae'].tolist())
            
            row_y_min = min(row_maes) * 0.95
            row_y_max = max(row_maes) * 1.05
        
        ax.set_ylim(row_y_min, row_y_max)
        
        ax.grid(axis='y', alpha=0.3)
        
        # Make subplot borders thinner and more transparent
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_alpha(0.5)

plt.tight_layout()
plt.subplots_adjust(left=0.1, hspace=0.3, wspace=0.3)
plt.savefig('Experiment_Results/figures/grand_average_neurons_prediction.png', 
            bbox_inches='tight', dpi=300)
plt.show()

# Print summary statistics
print("\nMAE Results by Neuron Count and Prediction Horizon")
print("=" * 60)
for i, neurons in enumerate(neuron_counts):
    print(f"\n{neurons} NEURONS:")
    for j, context in enumerate(context_names):
        print(f"  {prediction_labels[j]}:")
        data = datasets[context][neurons]
        naive_mae = data[data['model'] == 'Naive']['mae'].iloc[0]
        model_stats = data.groupby('model')['mae'].agg(['mean', 'std']).reset_index()
        for model in models_to_plot:
            model_row = model_stats[model_stats['model'] == model]
            if len(model_row) > 0:
                mae_mean = model_row['mean'].iloc[0]
                mae_std = model_row['std'].iloc[0]
                improvement = ((naive_mae - mae_mean) / naive_mae) * 100
                print(f"    {model:12}: {mae_mean:.4f}±{mae_std:.4f} ({improvement:+5.1f}% vs Naive)")
        print(f"    {'Naive':12}: {naive_mae:.4f}")
        print()