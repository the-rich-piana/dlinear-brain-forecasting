import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Model(nn.Module):
    """
    DummyLinear: A simple baseline that predicts the last timestep as the next timestep
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        
        # Since we're just copying the last timestep, we don't need any parameters
        # But we'll add a simple linear layer to make it trainable
        self.linear = nn.Linear(self.channels, self.channels)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, channels]
        # We only need the last timestep
        last_timestep = x[:, -1:, :]  # [batch_size, 1, channels]
        
        # Apply linear transformation (learnable)
        last_timestep = self.linear(last_timestep)
        
        # Repeat for pred_len timesteps
        prediction = last_timestep.repeat(1, self.pred_len, 1)  # [batch_size, pred_len, channels]
        
        return prediction