import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from utils.data_splitter import TimeSeriesSplitter
import warnings
import h5py

warnings.filterwarnings('ignore')


class Dataset_Activity_Stimulus(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='activity_raw.csv', # Changed default data_path
                 target='OT', train_only=False,
                 scale=True, timeenc=0, freq='h' #UNUSED PARAMS
                 ):
        
        # size [seq_len, label_len, pred_len]
        if size == None:
            self.seq_len = 48
            self.label_len = 8
            self.pred_len = 16
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
            
        # Fix: label_len should not exceed seq_len for DLinear
        if self.label_len > self.seq_len:
            print(f"Warning: label_len ({self.label_len}) > seq_len ({self.seq_len}). Setting label_len = seq_len // 2 = {self.seq_len // 2}")
            self.label_len = max(1, self.seq_len // 2)
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target # This will be ignored but is kept for API compatibility
        self.scale = scale
        self.timeenc = timeenc # This will be ignored but is kept for API compatibility
        self.freq = freq # This will be ignored but is kept for API compatibility

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        # Read preprocessed HDF5 data
        file_path = os.path.join(self.root_path, self.data_path)
        
        with h5py.File(file_path, 'r') as f:
            # Load the already preprocessed data (ΔF/F normalized and filtered)
            processed_data = f['activity'][:]
            timestamps = f['timestamps'][:]
            
            # Load covariate matrix
            covariate_matrix = f['covariate_matrix'][:]
            feature_names = [name.decode('utf-8') for name in f['covariate_metadata']['feature_names'][:]]
            
            # Load metadata
            n_original_neurons = f['metadata'].attrs['n_original_neurons']
            # Load normalization parameters for inverse transform
            if 'normalization' in f:
                self.norm_q005 = f['normalization']['q005'][:]
                self.norm_q995 = f['normalization']['q995'][:]
                self.norm_range = f['normalization']['data_range'][:]
            else:
                print("WARNING: no normalization parameters")                
                self.norm_q005 = None
                self.norm_q995 = None
                self.norm_range = None
            
            print(f"Loaded preprocessed data: {processed_data.shape}")
            print(f"Loaded covariate matrix: {covariate_matrix.shape}")
            print(f"Original neurons: {n_original_neurons}, current: {processed_data.shape[1]}")
            print(f"Data range: {processed_data.min():.3f} to {processed_data.max():.3f}")
            print(f"Covariate features: {feature_names}")
        
        # === NEW SPLITTING LOGIC USING TIMESERIESPLITTER ===
        # Create stimulus-based split map using covariate matrix
        print("\nCreating stimulus-based splits...")
        # Define which stimulus type to hold out for out-of-distribution testing.
        # Type 1 corresponds to Left 100% contrast.
        held_out_stimulus = [1] 
        print(f"Designating stimulus type(s) {held_out_stimulus} as the held-out test set.")

        split_map = TimeSeriesSplitter.create_stimulus_based_splits(
            covariate_matrix=covariate_matrix,
            train_pct=0.7, 
            val_pct=0.1,
            held_out_stimulus_types=held_out_stimulus
        )
        
        # Create TimeSeriesSplitter to get leak-free sample indices
        splitter = TimeSeriesSplitter(
            split_map=split_map,
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            label_len=self.label_len
        )
        
        # Get valid starting indices for the current dataset flag
        flag_map = {0: 'train', 1: 'val', 2: 'test'}
        current_flag = flag_map[self.set_type]
        self.valid_indices = splitter.get_indices(flag=current_flag)
        
        print(f"Split summary:")
        summary = splitter.get_split_summary()
        for key, value in summary.items():
            if 'pct' in key:
                print(f"  {key}: {value:.1f}%")
            else:
                print(f"  {key}: {value}")
        
        if len(self.valid_indices) == 0:
            raise RuntimeError(f"No valid samples found for split '{current_flag}'. Check window sizes and data length.")
        
        # Store full data for indexing in __getitem__.
        # 'processed_data' is already the final, selected top-N neuron matrix.
        self.data_x = processed_data
        self.data_y = processed_data
        self.covariate_data = covariate_matrix
        self.feature_names = feature_names
        self.data_stamp = np.zeros((processed_data.shape[0], 1))
        
        print(f"Using pre-selected top {self.data_x.shape[1]} neurons from HDF5 file.")
        print(f"Full dataset shape: {self.data_x.shape}")
        print(f"Full covariate shape: {self.covariate_data.shape}")
        print(f"Valid {current_flag} samples: {len(self.valid_indices)}")
        print(f"Data statistics - Mean: {self.data_x.mean():.3f}, Std: {self.data_x.std():.3f}")

    def __getitem__(self, index):
        # CRITICAL CHANGE: 'index' now refers to an index into our list of SAFE starting points.
        s_begin = self.valid_indices[index]  # Get actual starting point from valid indices
        # print(f"Sample index: {index} -> Data index: {s_begin}")
        
        # Rest of sequence extraction logic remains the same
        s_end = s_begin + self.seq_len  # End of input sequence
        # print(f"Input sequence: {s_begin} to {s_end-1} ({self.seq_len} steps)")
        
        # Target sequence start (overlaps with end of input by label_len)
        r_begin = s_end - self.label_len  # Start of target sequence (includes overlap)
        # print(f"Target sequence start (r_begin): {r_begin} (overlap of {self.label_len} steps)")
        
        # Target sequence end (target start + label overlap + prediction length)
        r_end = r_begin + self.label_len + self.pred_len  # End of target sequence
        # print(f"Target sequence: {r_begin} to {r_end-1} (total target length: {self.label_len + self.pred_len})")
        
        # Bounds checking (should never fail due to TimeSeriesSplitter validation)
        if r_end > len(self.data_x):
            raise IndexError(f"Sequence would exceed data bounds. Data length: {len(self.data_x)}, required end: {r_end}")

        # Extract neural data sequences
        seq_x = self.data_x[s_begin:s_end]  # Input neural sequence
        seq_y = self.data_y[r_begin:r_end]  # Target neural sequence (with overlap)
        
        # Ensure consistent shapes and dtypes
        seq_x = np.array(seq_x, dtype=np.float32)
        seq_y = np.array(seq_y, dtype=np.float32)
        
        # Verify expected shapes
        if seq_x.shape[0] != self.seq_len:
            raise ValueError(f"seq_x has wrong length: {seq_x.shape[0]}, expected: {self.seq_len}")
        if seq_y.shape[0] != (self.label_len + self.pred_len):
            raise ValueError(f"seq_y has wrong length: {seq_y.shape[0]}, expected: {self.label_len + self.pred_len}")
        
        # Extract covariate sequences aligned to neural sequences
        seq_x_mark = self.covariate_data[s_begin:s_end]  # Input covariates (wheel, stimuli, trial phase)
        seq_y_mark = self.covariate_data[r_begin:r_end]  # Target covariates (known future values)
        
        # Ensure consistent dtypes
        seq_x_mark = np.array(seq_x_mark, dtype=np.float32)
        seq_y_mark = np.array(seq_y_mark, dtype=np.float32)

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        # The length is now the number of valid samples we can create (leak-free)
        return len(self.valid_indices)

    def inverse_transform(self, data):
        """Convert normalized predictions back to original scale"""
        # Reverse the robust normalization: unnormalized = (normalized * range) + q005
        return (data * self.norm_range) + self.norm_q005

class Dataset_Activity(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='activity_raw.csv', # Changed default data_path
                 target='OT', scale=True, timeenc=0, freq='h', train_only=False): # Target is now a placeholder
        
        # size [seq_len, label_len, pred_len]
        if size == None:
            self.seq_len = 48
            self.label_len = 8
            self.pred_len = 16
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
            
        # Fix: label_len should not exceed seq_len for DLinear
        if self.label_len > self.seq_len:
            print(f"Warning: label_len ({self.label_len}) > seq_len ({self.seq_len}). Setting label_len = seq_len // 2 = {self.seq_len // 2}")
            self.label_len = max(1, self.seq_len // 2)
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target # This will be ignored but is kept for API compatibility
        self.scale = scale
        self.timeenc = timeenc # This will be ignored but is kept for API compatibility
        self.freq = freq # This will be ignored but is kept for API compatibility
        self.train_only = train_only # This will be ignored but is kept for API compatibility

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        # Read preprocessed HDF5 data
        file_path = os.path.join(self.root_path, self.data_path)
        
        with h5py.File(file_path, 'r') as f:
            # Load the already preprocessed data (ΔF/F normalized and filtered)
            processed_data: np.ndarray = f['activity'][:]
            timestamps = f['timestamps'][:]
            
            # Load metadata
            n_original_neurons = f['metadata'].attrs['n_original_neurons']
            # Load normalization parameters for inverse transform
            if 'normalization' in f:
                self.norm_q005 = f['normalization']['q005'][:]
                self.norm_q995 = f['normalization']['q995'][:]
                self.norm_range = f['normalization']['data_range'][:]
            else:
                print("WARNING: no normalization parameters")
                self.norm_q005 = None
                self.norm_q995 = None
                self.norm_range = None
            
            print(f"Loaded preprocessed data: {processed_data.shape}")
            print(f"Original neurons: {n_original_neurons}, current: {processed_data.shape[1]}")
            print(f"Data range: {processed_data.min():.3f} to {processed_data.max():.3f}")
        
        num_train = int(len(processed_data) * 0.7)
        num_vali = int(len(processed_data) * 0.1)
        num_test = len(processed_data) - num_train - num_vali
        
        # Final data assignment (data is already normalized from preprocessing)
        self.data_stamp = np.zeros((processed_data.shape[0], 1))
        self.data_x = processed_data[0:1000]
        self.data_y = processed_data[0:1000]
        
        print(f"Using pre-selected top {self.data_x.shape[1]} neurons from HDF5 file.")
        print(f"Full dataset shape: {self.data_x.shape}")
        print(f"Data statistics - Mean: {self.data_x.mean():.3f}, Std: {self.data_x.std():.3f}")

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        # Bounds checking
        if r_end > len(self.data_x):
            raise IndexError(f"Sequence index {index} would exceed data bounds. Data length: {len(self.data_x)}, required end: {r_end}")

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        
        # Ensure consistent shapes and dtypes
        seq_x = np.array(seq_x, dtype=np.float32)
        seq_y = np.array(seq_y, dtype=np.float32)
        
        # Verify expected shapes
        if seq_x.shape[0] != self.seq_len:
            raise ValueError(f"seq_x has wrong length: {seq_x.shape[0]}, expected: {self.seq_len}")
        if seq_y.shape[0] != (self.label_len + self.pred_len):
            raise ValueError(f"seq_y has wrong length: {seq_y.shape[0]}, expected: {self.label_len + self.pred_len}")
        
        # --- MODIFICATION 4: Return dummy tensors for time features ---
        # The training loop expects four return values. We provide zeros
        # for seq_x_mark and seq_y_mark, as DLinear will ignore them.
        seq_x_mark = np.zeros((self.seq_len, 1), dtype=np.float32)
        seq_y_mark = np.zeros((self.label_len + self.pred_len, 1), dtype=np.float32)

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        # Ensure we have enough data for at least one sequence
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        """Convert normalized predictions back to original scale"""
        if hasattr(self, 'norm_q005') and self.norm_q005 is not None:
            # Reverse the robust normalization: unnormalized = (normalized * range) + q005
            return (data * self.norm_range) + self.norm_q005
        return data


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', train_only=False):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', train_only=False):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        #We subtract the label length so we have some overlapping.
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Activity_Ordered(Dataset):
    """
    Dataset for brain activity with proper temporal splits.
    Train: first 85%, Val: next 10%, Test: last 5%
    """
    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='session_0.h5', 
                 target='OT', scale=True, timeenc=0, freq='h',
                 n_neurons=5000, train_only=False):
        
        # Sequence parameters
        if size is None:
            self.seq_len = 336
            self.label_len = 48
            self.pred_len = 96
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        
        # Dataset parameters
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.n_neurons = n_neurons
        self.train_only = train_only
        
        self.root_path = root_path
        self.data_path = data_path
        
        self.__read_data__()
    
    def __read_data__(self):
        """Read and process brain data with temporal splits"""
        
        # Load preprocessed HDF5 data
        file_path = os.path.join(self.root_path, self.data_path)
        
        with h5py.File(file_path, 'r') as f:
            # Load preprocessed data
            processed_data = f['processed_data'][:]
            timestamps = f['timestamps'][:]
            
            # Load metadata
            n_original_neurons = f['metadata'].attrs['n_original_neurons']
            n_processed_neurons = f['metadata'].attrs['n_processed_neurons']
            
            print(f"Brain data loaded: {processed_data.shape}")
            print(f"Original neurons: {n_original_neurons}, Processed neurons: {n_processed_neurons}")
            print(f"Data range: {processed_data.min():.3f} to {processed_data.max():.3f}")
            
            # Load normalization parameters if available
            if 'normalization' in f:
                self.norm_q005 = f['normalization']['q005'][:]
                self.norm_q995 = f['normalization']['q995'][:]
                self.norm_range = f['normalization']['data_range'][:]
            else:
                self.norm_q005 = None
                self.norm_q995 = None
                self.norm_range = None
        
        # TEMPORAL SPLITS FOR BRAIN DATA
        # Train: first 85%, Val: next 10%, Test: last 5%
        total_time = len(processed_data)
        train_end = int(total_time * 0.85)
        val_end = int(total_time * 0.95)
        
        # Calculate borders for each split
        border1s = [
            0,                           # Train start
            train_end - self.seq_len,    # Val start (overlap for sequences)
            val_end - self.seq_len       # Test start (overlap for sequences)
        ]
        border2s = [
            train_end,                   # Train end
            val_end,                     # Val end
            total_time                   # Test end
        ]
        
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        print(f"Temporal split - Total time: {total_time}")
        print(f"Train: 0 to {train_end} ({train_end/total_time*100:.1f}%)")
        print(f"Val: {train_end} to {val_end} ({(val_end-train_end)/total_time*100:.1f}%)")
        print(f"Test: {val_end} to {total_time} ({(total_time-val_end)/total_time*100:.1f}%)")
        print(f"Current split ({['train', 'val', 'test'][self.set_type]}): {border1} to {border2}")
        
        # SELECT TOP N NEURONS BASED ON TRAINING DATA ACTIVITY
        train_data = processed_data[border1s[0]:border2s[0]]
        
        # Calculate activity metrics on training data only
        neuron_stds = np.std(train_data, axis=0)
        
        # Select top N most active neurons
        num_neurons_to_keep = min(self.n_neurons, processed_data.shape[1])
        top_indices = np.argsort(neuron_stds)[-num_neurons_to_keep:]
        
        print(f"Selected top {num_neurons_to_keep} neurons out of {processed_data.shape[1]} total")
        print(f"Selected neuron std range: {neuron_stds[top_indices].min():.3f} to {neuron_stds[top_indices].max():.3f}")
        
        # Apply neuron selection
        data_filtered = processed_data[:, top_indices]
        
        # Store normalization parameters for selected neurons
        if self.norm_q005 is not None:
            self.norm_q005 = self.norm_q005[:, top_indices]
            self.norm_q995 = self.norm_q995[:, top_indices]
            self.norm_range = self.norm_range[:, top_indices]
        
        # Use real timestamps for time features
        self.data_stamp = timestamps.reshape(-1, 1)
        self.data_x = data_filtered[border1:border2]
        self.data_y = data_filtered[border1:border2]
        
        # Store info
        self.top_indices = top_indices
        
        print(f"Final dataset shape: {self.data_x.shape}")
        print(f"Final data stats - Mean: {self.data_x.mean():.3f}, Std: {self.data_x.std():.3f}")
        print(f"Sequences available: {len(self)}")
    
    def __getitem__(self, index):
        """Get a sequence for forecasting"""
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        # Bounds checking
        if r_end > len(self.data_x):
            raise IndexError(f"Sequence index {index} would exceed data bounds. Data length: {len(self.data_x)}, required end: {r_end}")

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        
        # Ensure consistent shapes and dtypes
        seq_x = np.array(seq_x, dtype=np.float32)
        seq_y = np.array(seq_y, dtype=np.float32)
        
        # Verify expected shapes
        if seq_x.shape[0] != self.seq_len:
            raise ValueError(f"seq_x has wrong length: {seq_x.shape[0]}, expected: {self.seq_len}")
        if seq_y.shape[0] != (self.label_len + self.pred_len):
            raise ValueError(f"seq_y has wrong length: {seq_y.shape[0]}, expected: {self.label_len + self.pred_len}")
        
        # Return real time features
        seq_x_mark = self.data_stamp[s_begin:s_end].astype(np.float32)
        seq_y_mark = self.data_stamp[r_begin:r_end].astype(np.float32)
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        """Number of sequences available"""
        available_length = len(self.data_x) - self.seq_len - self.pred_len + 1
        return max(0, available_length)
    
    def inverse_transform(self, data):
        """Convert normalized predictions back to original scale"""
        if hasattr(self, 'norm_q005') and self.norm_q005 is not None:
            return (data * self.norm_range) + self.norm_q005
        return data

class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', train_only=False):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.train_only = train_only

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        if self.features == 'S':
            cols.remove(self.target)
        cols.remove('date')
        # print(cols)
        num_train = int(len(df_raw) * (0.7 if not self.train_only else 1))
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            df_raw = df_raw[['date'] + cols]
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_raw = df_raw[['date'] + cols + [self.target]]
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            # print(self.scaler.mean_)
            # exit()
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    

class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None, train_only=False):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
        else:
            cols = list(df_raw.columns)
            self.cols = cols.copy()
            cols.remove('date')
        if self.features == 'S':
            cols.remove(self.target)
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            df_raw = df_raw[['date'] + cols]
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_raw = df_raw[['date'] + cols + [self.target]]
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        self.future_dates = list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
