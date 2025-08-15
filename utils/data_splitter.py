import numpy as np
from typing import Tuple, List, Optional

class TimeSeriesSplitter:
    """
    Generates leak-free lists of valid starting indices for time series forecasting.

    A sample starting at index `i` is considered valid for a given split (e.g., 'train')
    if its entire required window of data falls completely within that split.
    """
    def __init__(self, split_map: np.ndarray, seq_len: int, pred_len: int, label_len: int):
        """
        Args:
            split_map (np.ndarray): An array of the same length as the data, where each
                                     element is an integer representing the split
                                     (e.g., 0=train, 1=val, 2=test, -1=ignore).
            seq_len (int): Length of the input sequence.
            pred_len (int): Length of the prediction sequence.
            label_len (int): Length of the overlap/decoder seed.
        """
        self.split_map = split_map
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.label_len = label_len

        # The total number of consecutive timepoints a single sample needs.
        self.total_window_len = self.seq_len + self.pred_len
        
        # --- These will store our final, clean lists of indices ---
        self.train_indices = []
        self.val_indices = []
        self.test_indices = []

        self._generate_safe_indices()

    def _generate_safe_indices(self):
        """
        The core logic to find all valid starting points based on the simple, unbreakable rule.
        
        A sample starting at index `i` is valid for a split if and only if its entire 
        window (from i to i + seq_len + pred_len) falls completely within that same split.
        Inter-trial periods (marked as -1) are ignored and cannot be used for samples.
        """
        print("Generating safe, leak-free sample indices...")
        
        # Iterate through every possible starting point in the entire dataset
        for i in range(len(self.split_map) - self.total_window_len + 1):
            
            # Get the splits for the entire window this sample would touch
            window_splits = self.split_map[i : i + self.total_window_len]
            
            # The split of the FIRST element determines which set this sample MIGHT belong to.
            first_split_type = window_splits[0]

            # If the first element is in an 'ignore' zone, skip immediately.
            if first_split_type == -1:
                continue
            
            # THE CRITICAL VALIDITY CHECK:
            # Does the entire window belong to the SAME split type?
            if np.all(window_splits == first_split_type):
                if first_split_type == 0:  # Train
                    self.train_indices.append(i)
                elif first_split_type == 1:  # Validation
                    self.val_indices.append(i)
                elif first_split_type == 2:  # Test
                    self.test_indices.append(i)
        
        print(f"Found {len(self.train_indices)} valid training samples.")
        print(f"Found {len(self.val_indices)} valid validation samples.")
        print(f"Found {len(self.test_indices)} valid test samples.")

    def get_indices(self, flag='train') -> List[int]:
        """
        Public method to get the clean index list for a given split.
        
        Args:
            flag (str): The split to get indices for ('train', 'val', or 'test')
            
        Returns:
            List[int]: List of valid starting indices for the specified split
        """
        if flag == 'train':
            return self.train_indices
        elif flag == 'val':
            return self.val_indices
        elif flag == 'test':
            return self.test_indices
        else:
            raise ValueError("Flag must be 'train', 'val', or 'test'.")
    
    def get_split_summary(self) -> dict:
        """
        Return summary statistics about the generated samples.
        
        Returns:
            dict: Dictionary containing sample counts and percentages for each split
        """
        total_samples = len(self.train_indices) + len(self.val_indices) + len(self.test_indices)
        if total_samples == 0:
            return {
                'train_samples': 0, 'val_samples': 0, 'test_samples': 0,
                'total_samples': 0, 'train_pct': 0, 'val_pct': 0, 'test_pct': 0
            }
        return {
            'train_samples': len(self.train_indices),
            'val_samples': len(self.val_indices),
            'test_samples': len(self.test_indices),
            'total_samples': total_samples,
            'train_pct': len(self.train_indices) / total_samples * 100,
            'val_pct': len(self.val_indices) / total_samples * 100,
            'test_pct': len(self.test_indices) / total_samples * 100
        }

    @staticmethod
    def create_stimulus_based_splits(covariate_matrix: np.ndarray, 
                                   train_pct: float = 0.7, val_pct: float = 0.1, 
                                   held_out_stimulus_types: Optional[List[int]] = None) -> np.ndarray:
        """
        Creates a robust train/val/test split map based on stimulus types and timepoints.
        
        This method splits data by dividing the timepoints of each stimulus type according
        to the specified percentages, ensuring balanced representation across splits.
        Inter-trial periods are marked as -1 (ignore) in the split map.
        
        Args:
            covariate_matrix (np.ndarray): [n_timepoints, 11] covariate matrix from preprocessed data.
                                          Columns 1-9 are stimulus types (catch + 4 left + 4 right).
                                          Column 10 is 'trial_phase' indicator.
            train_pct (float): Percentage for training (default 0.7)
            val_pct (float): Percentage for validation (default 0.1) 
            held_out_stimulus_types (Optional[List[int]]): List of stimulus types to hold out 
                                                         for test set (e.g., [1] for Left 100% contrast)
                                   
        Returns:
            np.ndarray: Split map where 0=train, 1=val, 2=test, -1=ignore (inter-trial periods)
        """
        if held_out_stimulus_types is None:
            held_out_stimulus_types = []
            
        n_timepoints = covariate_matrix.shape[0]
        
        # Extract stimulus encoding (columns 1-9 are the 9 stimulus types)
        # Assumes feature 11 (index 10) is 'trial_phase'
        stimulus_onehot = covariate_matrix[:, 1:10]
        is_trial_period = covariate_matrix[:, 10] == 1
        
        # Initialize split map. -1 means "ignore" (e.g., inter-trial periods).
        split_map = np.full(n_timepoints, -1, dtype=int)
        
        # Process each stimulus type individually
        for stim_type in range(9):
            # Find all timepoints belonging to this stimulus type
            type_indices = np.where((stimulus_onehot[:, stim_type] == 1) & is_trial_period)[0]

            if len(type_indices) == 0:
                continue

            # Handle held-out test stimuli
            if stim_type in held_out_stimulus_types:
                split_map[type_indices] = 2  # Assign all to test
                print(f"Held out stimulus type {stim_type} for testing: {len(type_indices)} timepoints")
                continue
            
            # Split the INDICES of the timepoints, not the blocks
            n_points = len(type_indices)
            train_end_idx_pos = int(n_points * train_pct)
            val_end_idx_pos = train_end_idx_pos + int(n_points * val_pct)
            
            # Use the positions to slice the actual index array
            train_indices_for_type = type_indices[:train_end_idx_pos]
            val_indices_for_type = type_indices[train_end_idx_pos:val_end_idx_pos]
            test_indices_for_type = type_indices[val_end_idx_pos:]

            # Assign splits in the master map
            split_map[train_indices_for_type] = 0  # Train
            split_map[val_indices_for_type] = 1   # Validation
            split_map[test_indices_for_type] = 2    # In-distribution Test
            
            print(f"Stimulus type {stim_type}: {n_points} pts -> {len(train_indices_for_type)} train, {len(val_indices_for_type)} val, {len(test_indices_for_type)} test")
        
        return split_map