# Neural Population Activity Forecasting with Linear and Transformer Models

This repository contains a PyTorch implementation for forecasting neural population activity in mice, originally forked from the DLinear LTSF-Linear project but now completely diverged into an experimental framework for neuroscience applications. The codebase includes DLinear and multiple transformer-based models adapted for neural time series forecasting. 


## Project Overview

This experimental framework evaluates the effectiveness of different forecasting models on neural population activity data from mice. The project compares linear models (Linear, DLinear) against transformer architectures (Informer, Transformer) and specialized models (POCO, TSMixer) for predicting neural firing patterns.

### Key Features
- **Neural Activity Forecasting**: Specialized for mouse neural population data with 70-7000 neurons
- **Behavioral Context Analysis**: Compares model performance across Active vs Passive behavioral states
- **Multiple Model Types**: Linear models, Transformers, and population-specific architectures
- **Parallel Training Infrastructure**: Efficient model comparison with concurrent training
- **Comprehensive Analysis**: Performance metrics, visualization tools, and statistical comparisons

### Recent Developments
- **Activity Dataset Integration**: Custom data loaders for neural population time series (`.h5` format)
- **Behavioral Data Analysis**: Active vs Passive behavioral state prediction comparison
- **POCO Model Integration**: Population-Conditioned Forecaster with memory-efficient attention
- **Parallel Training Scripts**: Model-pair parallelization for efficient experimentation
- **Performance Metrics Framework**: Three-metric evaluation system relative to Naive baseline



## Implemented Models

### Linear Models
- [x] **Linear**: Single linear layer for neural activity prediction
- [x] **DLinear**: Decomposition Linear with separate trend/seasonal components
- [x] **Mean**: Statistical baseline using historical mean
- [x] **Naive**: Statistical baseline using last observation

### Transformer Models  
- [x] **Transformer**: Standard transformer architecture adapted for neural forecasting
- [x] **Informer**: Sparse attention mechanism for long-sequence neural data
- [x] **TSMixer**: MLP-based mixer architecture for multivariate neural time series

### Specialized Models
- [x] **POCO**: Population-Conditioned Forecaster with memory-efficient attention
  - Designed specifically for neural population dynamics
  - Supports variable population sizes (70-7000 neurons)
  - Integrated with xformers for efficient attention computation

### Experimental Configurations
- [x] **Short Forecasting**: Context=16, Prediction=8 timesteps
- [x] **Long Forecasting**: Context=48, Prediction=16 timesteps  
- [x] **Multiple Population Sizes**: 70, 700, 7000 neuron experiments
- [x] **Behavioral Context**: Active vs Passive behavioral state analysis


## Experiment Structure

Neural activity forecasting experiments are organized in `./scripts/`:

| Directory | Description |
|-----------|-------------|
| **EXP_Activity/** | Core neural activity forecasting experiments |
| **EXP_Activity_Behavioral/** | Active vs Passive behavioral state analysis |
| **EXP-LongForecasting/DLinear/** | Original long-term forecasting (legacy) |
| **EXP-LookBackWindow/** | Context window size studies |

### Key Script Types
- **Parallel Training**: `*_parallel.sh` - Concurrent model training for efficiency
- **Sequential Training**: Standard single-threaded model training
- **Behavioral Analysis**: Comparison across Active/Passive behavioral contexts
- **Population Scaling**: Experiments across different neuron counts (70/700/7000)

## Data Pipeline

### Neural Activity Dataset
- **Format**: HDF5 files (`.h5`) containing neural population time series
- **Structure**: `[timepoints, neurons]` matrices with neural firing rates
- **Population Sizes**: 70, 700, 7000 neurons from the same recording session
- **Behavioral States**: Active and Passive behavioral task contexts

### Data Splits
- **Training**: First 70% of time series (chronologically)  
- **Validation**: Next 10% for hyperparameter tuning
- **Test**: Last 20% for final evaluation
- **Windowing**: Sliding window approach with 1-timestep stride

## Performance Analysis

### Evaluation Metrics
The framework uses a three-metric evaluation system relative to Naive baseline:

1. **MAE Improvement %**: `((Naive_MAE - Model_MAE) / Naive_MAE) × 100`
2. **MSE Improvement %**: `((Naive_MSE - Model_MSE) / Naive_MSE) × 100`  
3. **Prediction Score**: `1 - L(model)/L(naive)` where L = MSE loss

### Key Findings for Neural Activity Forecasting
- **Linear models** often outperform complex Transformers on neural population data
- **DLinear** effectively captures trend components in neural firing patterns
- **POCO** shows promise for population-specific neural dynamics modeling
- **Population size** significantly impacts model performance and computational requirements
- **Behavioral context** (Active vs Passive) affects model generalization

### Analysis Tools
- **Training Curves**: Automatic visualization of loss progression (`utils/plots.py`)
- **Performance Comparison**: Jupyter notebooks for comprehensive model analysis
- **Weight Visualization**: Interpretability tools for linear model weights
- **Statistical Analysis**: Cross-behavioral state performance comparisons

## Getting Started

### Environment Requirements
The framework requires Python 3.10+ with PyTorch and specialized neuroscience dependencies:

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install torch torchvision torchaudio
pip install pandas numpy matplotlib seaborn
pip install h5py scikit-learn

# For POCO model support
pip install torchtyping xformers einops
```

### Data Preparation
Neural activity data should be in HDF5 format and placed in the `./dataset/` directory:

```bash
mkdir dataset
# Place your neural activity .h5 files here
# Format: [timepoints, neurons] matrices
```

### Dataset Structure
```
dataset/
├── session_5ea6bb9b-6163-4e8a-816b-efe7002666b0_70.h5    # 70 neurons
├── session_5ea6bb9b-6163-4e8a-816b-efe7002666b0_700.h5   # 700 neurons  
├── session_5ea6bb9b-6163-4e8a-816b-efe7002666b0_7000.h5  # 7000 neurons
└── ...
```

### Training Examples

#### Single Model Training
Train a specific model on neural activity data:

```bash
# Train DLinear on 7000 neuron dataset
python run_longExp.py \
  --is_training 1 \
  --model DLinear \
  --data ActivityBehavioral \
  --data_path session_5ea6bb9b-6163-4e8a-816b-efe7002666b0_7000.h5 \
  --seq_len 48 \
  --pred_len 16 \
  --enc_in 7000 \
  --c_out 7000 \
  --experiment_name MyExperiment

# Train statistical baseline
python run_stat.py \
  --model Naive \
  --data ActivityBehavioral \
  --data_path session_5ea6bb9b-6163-4e8a-816b-efe7002666b0_7000.h5 \
  --seq_len 48 \
  --pred_len 16
```

#### Parallel Training (Recommended)
Use parallel scripts for efficient model comparison:

```bash
# Run all models on 7000 neurons with behavioral analysis
sh scripts/EXP_Activity_Behavioral/activity_long_behavorial_7000_neurons_parallel.sh

# Run short forecasting experiments
sh scripts/EXP_Activity/activity_short_70_700_7000_neurons_parallel.sh
```

#### Background Training with Screen
For long-running experiments:

```bash
# Start screen session
screen -S neural_training

# Run training script
sh scripts/EXP_Activity_Behavioral/activity_long_behavorial_7000_neurons_parallel.sh

# Detach: Ctrl+A, then D
# Reattach later: screen -r neural_training
``` 

### Results Analysis
After training, results are stored in structured directories:

```
./logs/{experiment_name}/           # Training logs
./checkpoints/{experiment_name}/    # Model checkpoints  
./results/                          # Prediction outputs
```

Use the provided Jupyter notebooks for comprehensive analysis:
- `results_plot_long_7000_behavioral.ipynb`: Active vs Passive behavioral comparison
- `results_tables_short_long_7N.ipynb`: Performance metrics across neuron counts

### Weight Visualization  
Linear model weights can reveal neural population dynamics patterns:

```bash
python weight_plot.py
# Set model_name variable to your checkpoint path
```

## Output Structure
- **Logs**: Training progress in `./logs/{experiment_name}/`
- **Checkpoints**: Model weights in `./checkpoints/{experiment_name}/`  
- **Results**: Predictions and metrics in `./results/`
- **Visualizations**: Training curves and analysis plots

## Acknowledgments
This codebase was originally forked from the DLinear LTSF-Linear project and has been extensively adapted for neural population forecasting. We acknowledge the original implementations:

- **Autoformer, Informer, Transformer**: https://github.com/thuml/Autoformer
- **Original DLinear**: https://github.com/cure-lab/DLinear
- **POCO Model**: Population-Conditioned Forecaster architecture  
- **TSMixer**: MLP-based mixer for time series
