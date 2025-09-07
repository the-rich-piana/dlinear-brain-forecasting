import pandas as pd

def calculate_performance_metrics(results_df, dataset_name):
    """
    Given a dataframe (which is extracted from a model test results .csv)
    
    Calculate three performance metrics relative to Naive baseline:
    1. MAE improvement %: ((Naive_MAE - Model_MAE) / Naive_MAE) × 100
    2. MSE improvement %: ((Naive_MSE - Model_MSE) / Naive_MSE) × 100  
    3. Prediction Score: 1 − L(model)/L(naive) where L is MSE loss
    """
    
    # Get Naive baseline values
    naive_row = results_df[results_df['model'] == 'Naive']
    if len(naive_row) == 0:
        print(f"Warning: No Naive model found in {dataset_name}")
        return None
        
    naive_mae = naive_row['mae'].iloc[0]
    naive_mse = naive_row['mse'].iloc[0]
    
    # Calculate metrics for all models
    metrics_df = results_df.copy()
    
    # 1. MAE improvement percentage
    metrics_df['mae_improvement_pct'] = ((naive_mae - metrics_df['mae']) / naive_mae) * 100
    
    # 2. MSE improvement percentage  
    metrics_df['mse_improvement_pct'] = ((naive_mse - metrics_df['mse']) / naive_mse) * 100
    
    # 3. Prediction Score: 1 - L(model)/L(naive)
    metrics_df['prediction_score'] = 1 - (metrics_df['mse'] / naive_mse)
    
    # Add dataset identifier
    metrics_df['dataset'] = dataset_name
    
    return metrics_df

