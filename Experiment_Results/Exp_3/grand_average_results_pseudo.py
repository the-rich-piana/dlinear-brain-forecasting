import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

sns.set_theme(style="white")
plt.rcParams.update({'figure.dpi': 150, 'savefig.dpi': 150})

# Load the three datasets for comparison
datasets = {
    'Delta F/F': pd.read_csv("/cs/student/msc/aibh/2024/gcosta/DLinear/ActivityLong7000N.csv"),
    'Pseudo': pd.read_csv("/cs/student/msc/aibh/2024/gcosta/DLinear/ActivityLong7000NPseudo.csv"),
    'ZScore': pd.read_csv("/cs/student/msc/aibh/2024/gcosta/DLinear/ActivityLong7000NZscore.csv")
}

# Models to plot (exclude Mean, include Naive reference)
models_to_plot = ['DLinear', 'Linear', 'POCO', 'TSMixer', 'Informer', 'Transformer']
dataset_labels = ['Delta F/F', 'Pseudo', 'ZScore']

# Create 1x3 subplot grid for comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Define color palette and create consistent color mapping
colors = sns.color_palette("Set2", n_colors=len(models_to_plot))
color_map = dict(zip(models_to_plot, colors))

# Function to calculate prediction score: 1 - L(model)/L(naive)
def calculate_prediction_score(data):
    naive_mse = data[data['model'] == 'Naive']['mse'].iloc[0]
    model_stats = []
    
    for model in models_to_plot:
        model_data = data[data['model'] == model]
        if len(model_data) > 0:
            model_mse = model_data['mse'].mean()
            pred_score = 1 - (model_mse / naive_mse)
            model_stats.append({'model': model, 'prediction_score': pred_score})
    
    return pd.DataFrame(model_stats)

# Calculate global y-axis limits across all datasets
all_scores = []
for data in datasets.values():
    pred_scores = calculate_prediction_score(data)
    all_scores.extend(pred_scores['prediction_score'].tolist())

global_y_min = min(all_scores) * 1.1 if min(all_scores) < 0 else min(all_scores) * 0.9
global_y_max = max(all_scores) * 1.1

# Plot data for all three datasets
for i, (dataset_name, data) in enumerate(datasets.items()):
    ax = axes[i]
    
    # Calculate prediction scores for all models
    pred_scores = calculate_prediction_score(data)
    pred_scores = pred_scores.sort_values('prediction_score', ascending=True)
    
    # Look up consistent colors for each model
    model_colors = [color_map[model] for model in pred_scores['model']]
    
    # Create bar plot
    bars = ax.bar(range(len(pred_scores)), pred_scores['prediction_score'], 
                 color=model_colors, alpha=0.8, width=0.8)
    
    # Add dark horizontal lines at top of each bar
    for bar, color in zip(bars, model_colors):
        bar_height = bar.get_height()
        bar_x = bar.get_x()
        bar_width = bar.get_width()
        ax.plot([bar_x, bar_x + bar_width], [bar_height, bar_height], 
               color=color, linewidth=3, solid_capstyle='butt', zorder=1)
    
    # Add zero reference line
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.8)
    
    # Formatting
    ax.set_xticks(range(len(pred_scores)))
    ax.set_xticklabels(pred_scores['model'], rotation=45, ha='right', fontsize=10)
    
    # Only add y-axis label on leftmost plot
    if i == 0:
        ax.set_ylabel('Prediction Score', fontsize=14)
    
    ax.set_title(f'{dataset_name} Dataset', fontsize=14, fontweight='medium', pad=15, color="black")
    
    # Use global y-axis limits for all plots
    ax.set_ylim(global_y_min, global_y_max)
    
    ax.grid(axis='y', alpha=0.3)
    
    # Make subplot borders thinner and more transparent
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_alpha(0.5)

plt.tight_layout()
plt.subplots_adjust(wspace=0.1)
plt.savefig('Experiment_Results/figures/prediction_score_comparison_pseudo.jpeg', 
            bbox_inches='tight', dpi=150)
plt.show()

# Prepare table data
models_to_include = ['Naive', 'DLinear', 'Linear', 'POCO', 'TSMixer', 'Informer', 'Transformer']
table_rows = []

for dataset_name, data in datasets.items():
    # Calculate statistics for all models
    model_stats = data.groupby('model').agg({'mae': 'mean', 'mse': 'mean'}).reset_index()
    
    # Get Naive baseline for Prediction Score calculation
    naive_row = model_stats[model_stats['model'] == 'Naive']
    naive_mse = naive_row['mse'].iloc[0] if len(naive_row) > 0 else 1.0
    
    # Build row data
    row_data = {'Dataset': dataset_name}
    
    # Add each model's data
    for model in models_to_include:
        model_row = model_stats[model_stats['model'] == model]
        if len(model_row) > 0:
            mae = model_row['mae'].iloc[0]
            mse = model_row['mse'].iloc[0]
            
            # Store MAE for all models
            row_data[f'{model}_MAE'] = float(mae)
            
            # Store Prediction Score only for non-Naive models
            if model != 'Naive':
                prediction_score = 1 - (mse / naive_mse)
                row_data[f'{model}_PredScore'] = float(prediction_score)
        else:
            # Model not found, set to NaN
            row_data[f'{model}_MAE'] = float('nan')
            if model != 'Naive':
                row_data[f'{model}_PredScore'] = float('nan')
    
    table_rows.append(row_data)

# Convert to DataFrame
results_df = pd.DataFrame(table_rows)

# Export to CSV
csv_file = '/cs/student/msc/aibh/2024/gcosta/DLinear/Experiment_Results/Exp_3/grand_average_results_pseudo.csv'
results_df.to_csv(csv_file, index=False)
print("Saving .csv")

# Generate LaTeX table
tex_file = '/cs/student/msc/aibh/2024/gcosta/DLinear/Experiment_Results/Exp_3/grand_average_results_pseudo.tex'

latex_content = []
latex_content.append("\\begin{table}[h!]")
latex_content.append("\\centering")
latex_content.append("\\caption{Prediction Score Comparison: Delta F/F vs Pseudo vs ZScore}")
latex_content.append("\\label{tab:prediction_score_comparison}")

# Create table header - Naive gets 1 column, others get 2
header = "\\begin{tabular}{|c|c|"  # Dataset, Naive_MAE
for model in models_to_include[1:]:  # Skip Naive
    header += "c|c|"  # MAE and PredScore columns for each model
header += "}"

latex_content.append(header)
latex_content.append("\\hline")

# Column headers
col_header = "\\multirow{2}{*}{Dataset} & {Naive} "
for model in models_to_include[1:]:  # Skip Naive
    col_header += f"& \\multicolumn{{2}}{{c|}}{{{model}}} "
col_header += "\\\\"
latex_content.append(col_header)

# Sub-headers for MAE and PredScore
sub_header = " & MAE "
for model in models_to_include[1:]:  # Skip Naive
    sub_header += "& MAE & PredScore "
sub_header += "\\\\"
latex_content.append(sub_header)
latex_content.append("\\hline")

# Add data rows with bold formatting
for i, row in results_df.iterrows():
    # Find best MAE (lowest) and best PredScore (highest) for this row
    mae_values = []
    pred_score_values = []
    
    for model in models_to_include:
        mae_col = f'{model}_MAE'
        if mae_col in row and not pd.isna(row[mae_col]):
            mae_values.append((model, row[mae_col]))
            
    for model in models_to_include[1:]:  # Skip Naive for PredScore
        pred_col = f'{model}_PredScore'
        if pred_col in row and not pd.isna(row[pred_col]):
            pred_score_values.append((model, row[pred_col]))
    
    # Find best values
    best_mae_model = min(mae_values, key=lambda x: x[1])[0] if mae_values else None
    best_pred_model = max(pred_score_values, key=lambda x: x[1])[0] if pred_score_values else None
    
    # Start row with dataset name
    row_str = f"{row['Dataset']}"
    
    # Naive MAE column
    naive_mae = row['Naive_MAE']
    if pd.isna(naive_mae):
        row_str += " & --"
    else:
        if best_mae_model == 'Naive':
            row_str += f" & \\textbf{{{naive_mae:.3f}}}"
        else:
            row_str += f" & {naive_mae:.3f}"
    
    # Other model columns
    for model in models_to_include[1:]:  # Skip Naive
        mae = row[f'{model}_MAE']
        pred_score = row[f'{model}_PredScore'] if f'{model}_PredScore' in row else float('nan')
        
        # Format MAE
        if pd.isna(mae):
            mae_str = "--"
        else:
            if best_mae_model == model:
                mae_str = f"\\textbf{{{mae:.3f}}}"
            else:
                mae_str = f"{mae:.3f}"
        
        # Format PredScore
        if pd.isna(pred_score):
            pred_str = "--"
        else:
            if best_pred_model == model:
                pred_str = f"\\textbf{{{pred_score:.3f}}}"
            else:
                pred_str = f"{pred_score:.3f}"
        
        row_str += f" & {mae_str} & {pred_str}"
    
    row_str += " \\\\"
    latex_content.append(row_str)

latex_content.append("\\hline")
latex_content.append("\\end{tabular}")
latex_content.append("\\end{table}")

# Write LaTeX file
with open(tex_file, 'w') as f:
    f.write('\n'.join(latex_content))

print("Saving .tex")