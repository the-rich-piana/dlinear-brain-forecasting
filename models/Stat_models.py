import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
# import pmdarima as pm
import threading
from sklearn.ensemble import GradientBoostingRegressor

class Naive_repeat(nn.Module):
    def __init__(self, configs):
        super(Naive_repeat, self).__init__()
        self.pred_len = configs.pred_len
        
    def forward(self, x):
        B,L,D = x.shape
        x = x[:,-1,:].reshape(B,1,D)
        x = x.repeat(1, self.pred_len, 1)
        return x # [B, pred_len, D]

class Mean_repeat(nn.Module):
    """
    Mean baseline model that predicts by repeating the mean of the input sequence.
    
    This is a PyTorch implementation following the JAX MeanBaseline model structure.
    The model computes the mean across the time dimension of the input sequence
    and repeats it for the prediction length.
    """
    
    def __init__(self, configs):
        super(Mean_repeat, self).__init__()
        self.pred_len = configs.pred_len
        
        # Create an unused parameter to match the JAX implementation
        # This ensures compatibility with existing training code that expects parameters
        self.unused_param = nn.Parameter(torch.zeros(1))
        
    def forward(self, x: torch.Tensor):
        """
        Forward pass for mean baseline model.
        
        Args:
            x: Input tensor of shape [B, L, D] where:
                B = batch size
                L = sequence length (time steps)
                D = number of features/channels
                
        Returns:
            Output tensor of shape [B, pred_len, D] where each time step
            contains the mean of the input sequence across the time dimension.
        """
        batch_size, seq_len, num_features = x.shape
        
        # Compute mean across time dimension (dim=1), keeping dimensions
        # Shape: [batch_size, 1, num_features]
        mean_values = torch.mean(x, dim=1, keepdim=True)
        
        # Repeat the mean values for pred_len time steps
        # Shape: [batch_size, pred_len, num_features]
        predictions = mean_values.repeat(1, self.pred_len, 1)
        
        return predictions

# class Naive_thread(threading.Thread):
#     def __init__(self,func,args=()):
#         super(Naive_thread,self).__init__()
#         self.func = func
#         self.args = args

#     def run(self):
#         self.results = self.func(*self.args)
    
#     def return_result(self):
#         threading.Thread.join(self)
#         return self.results

# def _arima(seq,pred_len,bt,i):
#     model = pm.auto_arima(seq)
#     forecasts = model.predict(pred_len) 
#     return forecasts,bt,i

# class Arima(nn.Module):
#     """
#     Extremely slow, please sample < 0.1
#     """
#     def __init__(self, configs):
#         super(Arima, self).__init__()
#         self.pred_len = configs.pred_len
        
#     def forward(self, x):
#         result = np.zeros([x.shape[0],self.pred_len,x.shape[2]])
#         threads = []
#         for bt,seqs in tqdm(enumerate(x)):
#             for i in range(seqs.shape[-1]):
#                 seq = seqs[:,i]
#                 one_seq = Naive_thread(func=_arima,args=(seq,self.pred_len,bt,i))
#                 threads.append(one_seq)
#                 threads[-1].start()
#         for every_thread in tqdm(threads):
#             forcast,bt,i = every_thread.return_result()
#             result[bt,:,i] = forcast

#         return result # [B, L, D]

# def _sarima(season,seq,pred_len,bt,i):
#     model = pm.auto_arima(seq, seasonal=True, m=season)
#     forecasts = model.predict(pred_len) 
#     return forecasts,bt,i

# class SArima(nn.Module):
#     """
#     Extremely extremely slow, please sample < 0.01
#     """
#     def __init__(self, configs):
#         super(SArima, self).__init__()
#         self.pred_len = configs.pred_len
#         self.seq_len = configs.seq_len
#         self.season = 24
#         if 'Ettm' in configs.data_path:
#             self.season = 12
#         elif 'ILI' in configs.data_path:
#             self.season = 1
#         if self.season >= self.seq_len:
#             self.season = 1

#     def forward(self, x):
#         result = np.zeros([x.shape[0],self.pred_len,x.shape[2]])
#         threads = []
#         for bt,seqs in tqdm(enumerate(x)):
#             for i in range(seqs.shape[-1]):
#                 seq = seqs[:,i]
#                 one_seq = Naive_thread(func=_sarima,args=(self.season,seq,self.pred_len,bt,i))
#                 threads.append(one_seq)
#                 threads[-1].start()
#         for every_thread in tqdm(threads):
#             forcast,bt,i = every_thread.return_result()
#             result[bt,:,i] = forcast
#         return result # [B, L, D]

# def _gbrt(seq,seq_len,pred_len,bt,i):
#     model = GradientBoostingRegressor()
#     model.fit(np.arange(seq_len).reshape(-1,1),seq.reshape(-1,1))
#     forecasts = model.predict(np.arange(seq_len,seq_len+pred_len).reshape(-1,1))  
#     return forecasts,bt,i

# class GBRT(nn.Module):
#     def __init__(self, configs):
#         super(GBRT, self).__init__()
#         self.seq_len = configs.seq_len
#         self.pred_len = configs.pred_len
    
#     def forward(self, x):
#         result = np.zeros([x.shape[0],self.pred_len,x.shape[2]])
#         threads = []
#         for bt,seqs in tqdm(enumerate(x)):
#             for i in range(seqs.shape[-1]):
#                 seq = seqs[:,i]
#                 one_seq = Naive_thread(func=_gbrt,args=(seq,self.seq_len,self.pred_len,bt,i))
#                 threads.append(one_seq)
#                 threads[-1].start()
#         for every_thread in tqdm(threads):
#             forcast,bt,i = every_thread.return_result()
#             result[bt,:,i] = forcast
#         return result # [B, L, D]
