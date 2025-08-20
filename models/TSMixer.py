"""
TSMixer implementation in PyTorch for time series forecasting.

Converted from JAX implementation based on:
Paper: https://arxiv.org/abs/2303.06053
Reference: https://github.com/google-research/google-research/tree/master/tsmixer
"""

import torch
import torch.nn as nn


class ReversibleInstanceNorm(nn.Module):
    """Reversible Instance Normalization for TSMixer."""
    
    def __init__(self, eps=1e-5):
        super(ReversibleInstanceNorm, self).__init__()
        self.eps = eps
    
    def forward(self, x, stats=None):
        # x: [B, T, F]
        if stats is None:
            # Forward pass: normalize
            mean = x.mean(dim=1, keepdim=True)  # [B, 1, F]
            var = x.var(dim=1, keepdim=True, unbiased=False)  # [B, 1, F]
            stats = {'mean': mean, 'var': var}
            normalized = (x - mean) / torch.sqrt(var + self.eps)
            return normalized, stats
        else:
            # Reverse pass: denormalize
            mean, var = stats['mean'], stats['var']
            denormalized = x * torch.sqrt(var + self.eps) + mean
            return denormalized, stats


class TimeMix(nn.Module):
    """Time mixing block for TSMixer."""
    
    def __init__(self, seq_len, mlp_dim, activation_fn, dropout_prob, residual=True):
        super(TimeMix, self).__init__()
        self.seq_len = seq_len
        self.mlp_dim = mlp_dim
        self.residual = residual
        self.dropout_prob = dropout_prob
        
        # Layer normalization (applied to flattened input)
        self.norm = nn.LayerNorm(seq_len)  # Will be applied per feature
        
        # MLP for time dimension
        self.time_mlp = nn.Linear(seq_len, mlp_dim)
        
        # Activation function
        if activation_fn == 'relu':
            self.activation = nn.ReLU()
        elif activation_fn == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()  # default
            
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, x, training=True):
        # x: [B, T, F] -> [B, T, F]
        inputs = x
        B, T, F = x.shape
        
        # Normalize across time dimension for each feature separately
        # Reshape to [B*F, T] for normalization
        x_reshaped = x.permute(0, 2, 1).reshape(-1, T)  # [B*F, T]
        x_norm = self.norm(x_reshaped)  # [B*F, T]
        x_norm = x_norm.reshape(B, F, T).permute(0, 2, 1)  # [B, T, F]
        
        # Transpose to work on time dimension: [B, T, F] -> [B, F, T]
        x = x_norm.permute(0, 2, 1)
        
        # Apply time mixing MLP
        x = self.time_mlp(x)  # [B, F, mlp_dim]
        x = self.activation(x)
        
        # Transpose back: [B, F, mlp_dim] -> [B, mlp_dim, F]
        x = x.permute(0, 2, 1)
        
        # Apply dropout
        if training:
            x = self.dropout(x)
        
        # For residual connection, we need to match dimensions
        if self.residual and x.shape[1] == inputs.shape[1]:  # T == mlp_dim
            return x + inputs
        else:
            return x


class FeatureMix(nn.Module):
    """Feature mixing block for TSMixer."""
    
    def __init__(self, input_dim, mlp_dim_1, mlp_dim_2, activation_fn, dropout_prob, residual=True):
        super(FeatureMix, self).__init__()
        self.input_dim = input_dim
        self.mlp_dim_1 = mlp_dim_1
        self.mlp_dim_2 = mlp_dim_2
        self.residual = residual
        self.dropout_prob = dropout_prob
        
        # Layer normalization
        self.norm = nn.LayerNorm(input_dim)
        
        # MLP layers
        if mlp_dim_1 > 0:
            self.mlp1 = nn.Linear(input_dim, mlp_dim_1)
        else:
            self.mlp1 = None
            
        if mlp_dim_2 > 0:
            self.mlp2 = nn.Linear(mlp_dim_1 if mlp_dim_1 > 0 else input_dim, mlp_dim_2)
        else:
            self.mlp2 = None
        
        # Activation function
        if activation_fn == 'relu':
            self.activation = nn.ReLU()
        elif activation_fn == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()
            
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, x, training=True):
        # x: [B, T, F] -> [B, T, F]
        inputs = x
        B, T, F = x.shape
        
        # Normalize across feature dimension
        # Reshape to [B*T, F] for normalization
        x_reshaped = x.reshape(-1, F)  # [B*T, F]
        x_norm = self.norm(x_reshaped)  # [B*T, F]
        x = x_norm.reshape(B, T, F)  # [B, T, F]
        
        # Apply feature mixing MLPs
        if self.mlp1 is not None:
            x = self.mlp1(x)  # [B, T, mlp_dim_1]
            x = self.activation(x)
            if training:
                x = self.dropout(x)
                
        if self.mlp2 is not None:
            x = self.mlp2(x)  # [B, T, mlp_dim_2]
            if training:
                x = self.dropout(x)
        
        # Residual connection (only if output dimension matches input)
        if self.residual and x.shape[-1] == inputs.shape[-1]:
            return x + inputs
        else:
            return x


class MixerBlock(nn.Module):
    """Mixer block combining time and feature mixing."""
    
    def __init__(self, seq_len, num_features, 
                 time_mix_mlp_dim, time_mix_residual,
                 feature_mix_mlp_dim_1, feature_mix_mlp_dim_2, feature_mix_residual,
                 activation_fn, dropout_prob, time_mix_only=False, block_residual=False):
        super(MixerBlock, self).__init__()
        
        self.time_mix_only = time_mix_only
        self.block_residual = block_residual
        
        # Time mixing
        self.time_mix = TimeMix(
            seq_len=seq_len,
            mlp_dim=time_mix_mlp_dim,
            activation_fn=activation_fn,
            dropout_prob=dropout_prob,
            residual=time_mix_residual
        )
        
        # Feature mixing (only if not time_mix_only)
        if not time_mix_only:
            # Output dimension from time mix
            time_out_features = time_mix_mlp_dim if time_mix_mlp_dim != seq_len or not time_mix_residual else num_features
            
            self.feature_mix = FeatureMix(
                input_dim=time_out_features,
                mlp_dim_1=feature_mix_mlp_dim_1,
                mlp_dim_2=feature_mix_mlp_dim_2,
                activation_fn=activation_fn,
                dropout_prob=dropout_prob,
                residual=feature_mix_residual
            )
    
    def forward(self, x, training=True):
        # x: [B, T, F] -> [B, T, F]
        inputs = x
        
        # Time mixing
        x = self.time_mix(x, training=training)
        
        if self.time_mix_only:
            return x
            
        # Feature mixing
        x = self.feature_mix(x, training=training)
        
        # Block-level residual connection
        if self.block_residual and x.shape == inputs.shape:
            x = x + inputs
            
        return x


class Model(nn.Module):
    """
    TSMixer model for time series forecasting.
    Follows the DLinear project structure conventions.
    """
    
    def __init__(self, configs):
        super(Model, self).__init__()
        
        # Extract configuration
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in  # number of features
        
        # TSMixer specific configs with defaults
        self.n_blocks = getattr(2, 'n_blocks', 8)
        self.dropout = getattr(configs, 'dropout', 0.1)
        self.activation = getattr(configs, 'activation', 'relu')
        self.mlp_dim = getattr(configs, 'mlp_dim', 100)
        self.time_mix_mlp_dim = getattr(configs, 'time_mix_mlp_dim', -1)  # -1 means use seq_len
        
        # Normalization settings
        self.instance_norm = getattr(configs, 'instance_norm', True)
        self.revert_instance_norm = getattr(configs, 'revert_instance_norm', True)
        
        # Residual connection settings
        self.time_mix_residual = getattr(configs, 'time_mix_residual', True)
        self.feature_mix_residual = getattr(configs, 'feature_mix_residual', True)
        self.block_residual = getattr(configs, 'block_residual', False)
        self.time_mix_only = getattr(configs, 'time_mix_only', False)
        
        # Instance normalization
        if self.instance_norm:
            self.rev_norm = ReversibleInstanceNorm()
        
        # Determine actual dimensions
        actual_time_mix_mlp_dim = self.seq_len if self.time_mix_mlp_dim == -1 else self.time_mix_mlp_dim
        
        # Build mixer blocks
        self.mixer_blocks = nn.ModuleList()
        
        current_seq_len = self.seq_len
        current_features = self.enc_in
        
        for _ in range(self.n_blocks):
            block = MixerBlock(
                seq_len=current_seq_len,
                num_features=current_features,
                time_mix_mlp_dim=actual_time_mix_mlp_dim,
                time_mix_residual=self.time_mix_residual,
                feature_mix_mlp_dim_1=self.mlp_dim if self.mlp_dim > 0 else current_features,
                feature_mix_mlp_dim_2=current_features if self.mlp_dim > 0 else 0,
                feature_mix_residual=self.feature_mix_residual,
                activation_fn=self.activation,
                dropout_prob=self.dropout,
                time_mix_only=self.time_mix_only,
                block_residual=self.block_residual
            )
            self.mixer_blocks.append(block)
            
            # Update dimensions for next block (if no residual connections change dimensions)
            if not self.time_mix_residual:
                current_seq_len = actual_time_mix_mlp_dim
        
        # Final temporal projection to pred_len
        self.temporal_projection = nn.Linear(current_seq_len, self.pred_len)
    
    def forward(self, x):
        """
        Args:
            x: [Batch, seq_len, enc_in] - Input time series
        Returns:
            [Batch, pred_len, enc_in] - Predicted time series
        """
        # x: [B, T, F]
        
        # Instance normalization
        stats = None
        if self.instance_norm:
            x, stats = self.rev_norm(x)
        
        # Apply mixer blocks
        for block in self.mixer_blocks:
            x = block(x, training=self.training)
        
        # Temporal projection: [B, T, F] -> [B, pred_len, F]
        # Transpose to apply projection on time dimension
        x = x.permute(0, 2, 1)  # [B, F, T]
        x = self.temporal_projection(x)  # [B, F, pred_len]
        x = x.permute(0, 2, 1)  # [B, pred_len, F]
        
        # Revert instance normalization
        if self.instance_norm and self.revert_instance_norm and stats is not None:
            x, _ = self.rev_norm(x, stats)
        
        return x