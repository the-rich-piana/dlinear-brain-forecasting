# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## For Research/Codebase Agents

When understanding how experiments work in this codebase:

1. **Start at bash scripts**: Check `scripts/EXP_Activity/` or `scripts/EXP_Activity_Behavioral/` directories for experiment configurations and parameters
2. **Check dataloader class**: Examine the specific dataloader class used by that script (typically in `data_provider/data_loader.py`)  
3. **See results analysis**: Check the `Experiment_Results/` directory for scripts showing how we generated our research results

## Project Overview

This is a PyTorch implementation of DLinear for time series forecasting, based on the paper "Are Transformers Effective for Time Series Forecasting?". The repository includes DLinear and five transformer-based models (Transformer, Informer, Autoformer, Pyraformer, FEDformer) for comparison.

## Key Commands

### Training Models
- Main training entry point: `python run_longExp.py [args]`
- Run DLinear experiments: `sh scripts/EXP-LongForecasting/DLinear/{dataset}.sh`
- Run statistical models: `python run_stat.py [args]`

### Data Preparation
- Datasets should be placed in `./dataset/` directory
- Dataset sources available from Google Drive link in README.md

### Visualization
- Generate weight visualizations: `python weight_plot.py`
- Requires setting `model_name` variable in the script to match checkpoint path

## Architecture Overview

### Core Components

1. **Models** (`models/`):
   - `DLinear.py`: Main DLinear implementation with series decomposition
   - `Autoformer.py`, `Informer.py`, `Transformer.py`: Transformer variants
   - `Stat_models.py`: Statistical baseline models

2. **Data Pipeline** (`data_provider/`):
   - `data_factory.py`: Data provider factory with dataset mappings
   - `data_loader.py`: Dataset classes for different data formats
   - Supports ETT datasets, custom datasets, and Activity dataset

3. **Experiment Management** (`exp/`):
   - `exp_main.py`: Main experiment class with train/test/predict methods
   - `exp_basic.py`: Base experiment class
   - `exp_stat.py`: Statistical model experiments

4. **Utilities** (`utils/`):
   - `tools.py`: Early stopping, learning rate adjustment, visualization
   - `metrics.py`: Evaluation metrics
   - `timefeatures.py`: Time feature encoding

### DLinear Model Architecture

- **Series Decomposition**: Separates time series into trend and seasonal components using moving average
- **Individual Mode**: Option to use separate linear layers for each channel (`--individual` flag)
- **Dual Linear Layers**: 
  - `Linear_Seasonal`: Processes seasonal component
  - `Linear_Trend`: Processes trend component
- **Output**: Combines seasonal and trend predictions

### Experiment Scripts Structure

- `scripts/EXP-LongForecasting/DLinear/`: Dataset-specific training scripts
- `scripts/EXP-LookBackWindow/`: Look-back window size experiments
- `scripts/EXP-Embedding/`: Embedding strategy experiments
- `FEDformer/scripts/` and `Pyraformer/scripts/`: Model-specific experiments

## Key Configuration Parameters

- `seq_len`: Input sequence length (look-back window)
- `pred_len`: Prediction sequence length (forecasting horizon)
- `label_len`: Overlap between input and output sequences (for decoder initialization in Transformers)
- `enc_in`: Number of input channels/features
- `individual`: Use individual linear layers per channel
- `features`: Forecasting task type (M: multivariate, S: univariate, MS: multivariate to univariate)
- `data`: Dataset type (ETTh1, ETTh2, ETTm1, ETTm2, custom, Activity)
- `train_only`: Use 100% data for training (vs normal 70/10/20 split)

## Output Structure

- **Checkpoints**: Saved in `./checkpoints/` with auto-generated naming
- **Logs**: Training logs in `./logs/LongForecasting/`
- **Results**: Prediction arrays in `./results/`
- **Test Results**: Visualization plots in `./test_results/`

## Data Splitting & Windowing Details

### Dataset Splits (Dataset_Activity)
- **Training**: First 70% of timeseries (chronologically earliest)
- **Validation**: Next 10% of timeseries (middle)
- **Test**: Last 20% of timeseries (chronologically latest)
- Uses temporal splits to maintain chronological order (no data leakage)

### Sliding Window Mechanism
- Creates overlapping samples by sliding 1 timepoint at a time
- From N timepoints with seq_len=S, pred_len=P: generates N-S-P+1 samples
- Each sample: Input[i:i+S] → Target[i+S-label_len:i+S+P]
- Example: 100 timepoints, seq_len=32, pred_len=8 → 61 training samples

### Label Usage
- `label_len`: Overlap timepoints between input and output sequences
- **Transformers**: Use label portion for decoder initialization (known recent values)
- **Linear models**: Ignore label portion completely (only use seq_len input)
- **Loss computation**: Only computed on pred_len portion (label excluded from loss)

### Border Indices
- `border1s[i]`, `border2s[i]`: Start/end indices for each split (train/val/test)
- Test border starts seq_len earlier to ensure sufficient lookback context
- Format: border1s = [train_start, val_start, test_start], border2s = [train_end, val_end, test_end]

## Development Notes
- Uses PyTorch 1.9.0 with CUDA support
- Supports multi-GPU training with `--use_multi_gpu` flag
- Fixed seed (2021) for reproducibility
- Early stopping implemented with configurable patience
- Automatic mixed precision training available with `--use_amp`
- NEVER open any csv with shell commands. All the datasets are big >1 GB and will crash the terminal session. If you need info about the data, consult the user.

## Recent Development Updates

### Checkpoint Management System
- **Fixed checkpoint paths**: Updated `exp/exp_main.py` to use `self.args.checkpoints` instead of hardcoded `./checkpoints/`
- **Experiment organization**: Checkpoints now saved to `./checkpoints/{experiment_name}/{setting}/` structure
- **Consistent paths**: All functions (`train()`, `test()`, `predict()`) now use experiment_name in checkpoint paths
- **Issue resolved**: Multiple experiments no longer overwrite each other's checkpoints

### Parallel Training Infrastructure
- **Parallel scripts**: Created `activity_short_70_700_7000_neurons_parallel.sh` for concurrent model training
- **Background execution**: Uses bash `&` operator with `wait` synchronization for model groups
- **Error logging**: Added `2>&1` redirection to capture stderr in log files
- **Signal handling**: Implemented Ctrl+C cleanup to kill background processes
- **Performance**: Trains 3 datasets simultaneously per model type (70/700/7000 neurons)

### Activity Dataset Experiments
- **Dataset variants**: 70, 700, 7000 neuron subsets from Activity dataset
- **Short forecasting**: Context=16, Prediction=8 configuration
- **Long forecasting**: Context=48, Prediction=16 configuration  
- **Model coverage**: Tests DLinear, Informer, POCO, Transformer, TSMixer, Linear, Mean, Naive

### Visualization and Analysis Tools
- **Training curves**: `utils/plots.py` with `plot_training_curves(experiment_name)` function
- **Log parsing**: Automatic extraction of epoch/train_loss/vali_loss from log files
- **Performance metrics**: Three-metric evaluation system relative to Naive baseline:
  1. MAE improvement %: `((Naive_MAE - Model_MAE) / Naive_MAE) × 100`
  2. MSE improvement %: `((Naive_MSE - Model_MSE) / Naive_MSE) × 100`
  3. Prediction Score: `1 - L(model)/L(naive)` where L = MSE loss
- **Comprehensive notebooks**: 
  - `results_plot_7_neurons.ipynb`: Comparative analysis across neuron counts
  - `results_tables_short_long_7N.ipynb`: Performance metrics and statistical analysis

### Dependencies and Environment
- **Required packages**: `torchtyping`, `xformers`, `einops` for POCO model
- **Memory efficient attention**: xformers integration for handling large neuron counts
- **Virtual environment**: `/cs/student/projects1/aibh/2024/gcosta/venv/` with Python 3.10.14

### Key Script Locations
- **Parallel training**: `scripts/EXP_Activity/activity_short_70_700_7000_neurons_parallel.sh`
- **Sequential training**: `scripts/EXP_Activity/activity_short_70_700_7000_neurons.sh`
- **Visualization utilities**: `utils/plots.py`
- **Log directories**: `./logs/{experiment_name}/{model}_{seq_len}_{pred_len}.log`

### Performance Analysis Framework
- **Baseline comparison**: All metrics calculated relative to Naive model within each dataset
- **No cross-dataset comparison**: Each dataset evaluated independently
- **Statistical consistency**: Model performance variability analysis across datasets
- **Best performer identification**: Automated detection of top models per metric per dataset

### POCO Model Integration
- **POCO Model**: Population-Conditioned Forecaster integrated with framework adapter wrapper
- **Adapter Pattern**: `models/POCO.py` contains both original POCO implementation and framework-compatible `Model` class
- **Import Structure**: Uses `from . import POCO` in `models/__init__.py` to follow framework pattern (`POCO.Model()`)
- **Parameter Mapping**: Framework `configs` object automatically mapped to POCO's `NeuralPredictionConfig`
  - `seq_len + pred_len` → `seq_length` (total sequence length)  
  - `pred_len` → `pred_length`
  - `enc_in` → single session input_size `[[enc_in]]`
- **Tensor Reshaping**: Adapter handles conversion between framework `[B, L, D]` and POCO `[L, B, D]` formats
- **Dependencies**: Requires `torchtyping`, `xformers`, `einops` packages
- **Usage**: Standard script interface: `--model POCO --enc_in 7 --c_out 7`
- **Error Fix**: Import issue resolved - POCO module imported correctly to support `POCO.Model()` access pattern