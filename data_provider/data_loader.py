import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings
import h5py

warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
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
                 target='OT', scale=True, timeenc=0, freq='t'):
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


class Dataset_Activity(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='activity_raw.csv', # Changed default data_path
                 target='OT', scale=True, timeenc=0, freq='h'): # Target is now a placeholder
        
        # size [seq_len, label_len, pred_len]
        if size == None:
            self.seq_len = 96
            self.label_len = 48
            self.pred_len = 96
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
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
            processed_data = f['processed_data'][:]
            
            # Load metadata
            n_original_neurons = f['metadata'].attrs['n_original_neurons']
            n_processed_neurons = f['metadata'].attrs['n_processed_neurons']
            
            # Load normalization parameters for inverse transform
            if 'normalization' in f:
                self.norm_q005 = f['normalization']['q005'][:]
                self.norm_q995 = f['normalization']['q995'][:]
                self.norm_range = f['normalization']['data_range'][:]
            else:
                self.norm_q005 = None
                self.norm_q995 = None
                self.norm_range = None
            
            print(f"Loaded preprocessed data: {processed_data.shape}")
            print(f"Original neurons: {n_original_neurons}, Processed neurons: {n_processed_neurons}")
            print(f"Data range: {processed_data.min():.3f} to {processed_data.max():.3f}")
        
        # Define splits (same as before)
        num_train = int(len(processed_data) * 0.7)
        num_vali = int(len(processed_data) * 0.1)
        num_test = len(processed_data) - num_train - num_vali
        
        border1s = [0, num_train - self.seq_len, len(processed_data) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(processed_data)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        # === SELECT TOP 1000 NEURONS FROM PREPROCESSED DATA ===
        train_data = processed_data[border1s[0]:border2s[0]]
        
        # Calculate activity metrics on preprocessed data
        neuron_stds = np.std(train_data, axis=0)
        neuron_means = np.mean(train_data, axis=0)
        neuron_maxs = np.max(train_data, axis=0)
        
        # Select top 1000 most active neurons
        num_neurons_to_keep = min(5000, processed_data.shape[1])
        top_indices = np.argsort(neuron_stds)[-num_neurons_to_keep:]
        
        print(f"Selected top {num_neurons_to_keep} neurons out of {processed_data.shape[1]} preprocessed neurons")
        
        # Apply neuron selection
        data_filtered = processed_data[:, top_indices]
        
        # Final data assignment (data is already normalized from preprocessing)
        self.data_stamp = np.zeros((data_filtered.shape[0], 1))
        self.data_x = data_filtered[border1:border2]
        self.data_y = data_filtered[border1:border2]
        
        # Store selection info and normalization parameters for selected neurons
        self.top_indices = top_indices
        if self.norm_q005 is not None:
            self.norm_q005 = self.norm_q005[:, top_indices]
            self.norm_q995 = self.norm_q995[:, top_indices]
            self.norm_range = self.norm_range[:, top_indices]
        
        print(f"Final dataset shape: {self.data_x.shape}")
        print(f"Final data statistics - Mean: {self.data_x.mean():.3f}, Std: {self.data_x.std():.3f}")

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        
        # --- MODIFICATION 4: Return dummy tensors for time features ---
        # The training loop expects four return values. We provide zeros
        # for seq_x_mark and seq_y_mark, as DLinear will ignore them.
        seq_x_mark = np.zeros((self.seq_len, 1))
        seq_y_mark = np.zeros((self.label_len + self.pred_len, 1))

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        # This logic remains correct
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        """Convert normalized predictions back to original scale"""
        if hasattr(self, 'norm_q005') and self.norm_q005 is not None:
            # Reverse the robust normalization: unnormalized = (normalized * range) + q005
            return (data * self.norm_range) + self.norm_q005
        return data

class Dataset_Activity_Ordered(Dataset):
    """
    Dataset for brain activity with proper temporal splits.
    Train: first 85%, Val: next 10%, Test: last 5%
    """
    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='session_0.h5', 
                 target='OT', scale=True, timeenc=0, freq='h',
                 n_neurons=5000):
        
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
        
        # Final data assignment
        self.data_stamp = np.zeros((data_filtered.shape[0], 1))
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
        
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        
        # Return dummy time features
        seq_x_mark = np.zeros((self.seq_len, 1))
        seq_y_mark = np.zeros((self.label_len + self.pred_len, 1))
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        """Number of sequences available"""
        return len(self.data_x) - self.seq_len - self.pred_len + 1
    
    def inverse_transform(self, data):
        """Convert normalized predictions back to original scale"""
        if hasattr(self, 'norm_q005') and self.norm_q005 is not None:
            return (data * self.norm_range) + self.norm_q005
        return data

class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
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

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        # print(cols)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
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
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
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
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
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
