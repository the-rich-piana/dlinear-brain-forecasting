# check_data_pipeline.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import h5py
import numpy as np
from utils.data_splitter import TimeSeriesSplitter # Your new, fixed splitter
# We will create a simplified data loader here to avoid torch dependency

# --- Configuration ---
HDF5_PATH = '/cs/student/projects1/aibh/2024/gcosta/mpci_data/session_61f260e7-b5d3-4865-a577-bcfc53fda8a8.h5'
SEQ_LEN = 48
PRED_LEN = 16
LABEL_LEN = 8

# --- The Sanity Check ---

# 1. Load the full, unfiltered data from the HDF5 file
print("Loading full dataset...")
with h5py.File(HDF5_PATH, 'r') as f:
    full_activity = f['activity'][:]
    full_covariates = f['covariate_matrix'][:]
    norm_q005 = f['normalization']['q005'][:]
    norm_range = f['normalization']['data_range'][:]
    top_indices_from_file = f['metadata']['top_neuron_indices'][:] # Assume you save this
print(f"Loaded {full_activity.shape[0]} timepoints.")

# 2. Create the Split Map using the STATIC method
print("\nCreating split map...")
split_map = TimeSeriesSplitter.create_stimulus_based_splits(full_covariates)

# 3. Instantiate the Splitter to get valid indices
print("\nGenerating safe sample indices...")
splitter = TimeSeriesSplitter(split_map, SEQ_LEN, PRED_LEN, LABEL_LEN)
train_indices = splitter.get_indices('train')

# 4. Manually perform a __getitem__ for the FIRST valid training sample
print("\n--- Performing Manual __getitem__ and Inverse Transform Check ---")
s_begin = train_indices[0]
s_end = s_begin + SEQ_LEN
r_begin = s_end - LABEL_LEN
r_end = r_begin + LABEL_LEN + PRED_LEN

# Get the SLICE of data that would be returned
seq_y_normalized = full_activity[r_begin:r_end, :]

# Apply the neuron selection that happens in the data loader
# NOTE: The top_indices used for normalization MUST be the same!
# Let's assume you saved the indices used during preprocessing.
selected_norm_q005 = norm_q005[:, top_indices_from_file]
selected_norm_range = norm_range[:, top_indices_from_file]

# 5. Perform the INVERSE TRANSFORM
print("Applying inverse transform...")
seq_y_rescaled = (seq_y_normalized * selected_norm_range) + selected_norm_q005

# 6. Load the ORIGINAL, UNFILTERED data for comparison
# You need to have a way to get the original data before normalization.
# Let's assume you have a function for that.
original_unfiltered_activity = ... # Load the raw dff matrix before normalization

# 7. THE FINAL CHECK
print("Comparing rescaled data to original data...")
original_slice = original_unfiltered_activity[r_begin:r_end, top_indices_from_file]

# Check if they are almost identical
if np.allclose(seq_y_rescaled, original_slice, atol=1e-5):
    print("✅ SUCCESS! Inverse transform is correct. The data pipeline is sound.")
else:
    print("❌ FAILURE! Inverse transform is incorrect. The bug is in the normalization/selection logic.")
    # Print out values to see the difference
    print("Example from rescaled:", seq_y_rescaled[0, 0])
    print("Example from original:", original_slice[0, 0])
