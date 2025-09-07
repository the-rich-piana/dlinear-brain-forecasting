import pandas as pd
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

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

# Models to include in the table (include statistical baselines)
models_to_include = ['Naive', 'Mean', 'DLinear', 'Linear', 'POCO', 'TSMixer', 'Informer', 'Transformer']
neuron_counts = [70, 700, 7000]
context_names = ['short', 'medium', 'long']
context_labels = ['16,8', '48,16', '96,32']  # Context,Prediction pairs

# Prepare table data
table_rows = []

for i, neurons in enumerate(neuron_counts):
    for j, (context, context_label) in enumerate(zip(context_names, context_labels)):
        # Get data for this specific neuron count and prediction horizon
        plot_data = datasets[context][neurons]
        
        # Calculate mean and std for each model
        model_stats = plot_data.groupby('model').agg({'mae': 'mean', 'mse': 'mean'}).reset_index()
        
        # Get Naive baseline for Prediction Score calculation
        naive_row = model_stats[model_stats['model'] == 'Naive']
        if len(naive_row) > 0:
            naive_mse = naive_row['mse'].iloc[0]
        else:
            naive_mse = 1.0  # fallback
        
        # Build row data
        row_data = {
            'Dataset': f'{neurons} Neurons',
            'Context_Pred': context_label
        }
        
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
                    # Calculate Prediction Score: 1 - (model_mse / naive_mse)
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
csv_file = './Experiment_Results/Exp_1/grand_average_results_table.csv'
results_df.to_csv(csv_file, index=False)
print("Saving .csv")

# Generate LaTeX table
tex_file = './Experiment_Results/Exp_1/grand_average_results_table.tex'

latex_content = []
latex_content.append("\\begin{table}[h!]")
latex_content.append("\\centering")
latex_content.append("\\caption{Grand Average Results by Population Size and Context Length}")
latex_content.append("\\label{tab:grand_average_results}")

# Create table header - Naive gets 1 column, others get 2
header = "\\begin{tabular}{|c|c|c|"  # Population, Context, Naive_MAE
for model in models_to_include[1:]:  # Skip Naive
    header += "c|c|"  # MAE and PredScore columns for each model
header += "}"

latex_content.append(header)
latex_content.append("\\hline")

# Column headers
col_header = "\\multirow{2}{*}{Population} & \\multirow{2}{*}{Context} & {Naive} "
for model in models_to_include[1:]:  # Skip Naive
    col_header += f"& \\multicolumn{{2}}{{c|}}{{{model}}} "
col_header += "\\\\"
latex_content.append(col_header)

# Sub-headers for MAE and PredScore
sub_header = " & & MAE "
for model in models_to_include[1:]:  # Skip Naive
    sub_header += "& MAE & PredScore "
sub_header += "\\\\"
latex_content.append(sub_header)
latex_content.append("\\hline")

# Add data rows with bold formatting
neuron_labels = ['70', '700', '7000']
for i, neurons in enumerate(neuron_counts):
    for j, (context, context_label) in enumerate(zip(context_names, context_labels)):
        row_idx = i * 3 + j
        row = results_df.iloc[row_idx]
        
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
        
        # First column: population size (only for first row of each group)
        if j == 0:
            row_str = f"\\multirow{{3}}{{*}}{{{neuron_labels[i]}}}"
        else:
            row_str = ""
        
        # Second column: context/prediction length
        row_str += f" & {context_label}"
        
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
        
        # Add horizontal line after each population group
        if j == 2 and i < 2:  # After last context of each population (except the last)
            latex_content.append("\\hline")

latex_content.append("\\hline")
latex_content.append("\\end{tabular}")
latex_content.append("\\end{table}")

# Write LaTeX file
with open(tex_file, 'w') as f:
    f.write('\n'.join(latex_content))

print("Saving .tex")