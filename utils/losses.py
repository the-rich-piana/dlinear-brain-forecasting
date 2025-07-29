import torch
import torch.nn as nn
import numpy as np

class AntiMeanLoss(nn.Module):
    """
    Loss function that penalizes mean predictions to encourage learning patterns.
    
    Combines MSE loss with a penalty term that increases loss when predictions
    are too close to the mean of the input or target.
    """
    
    def __init__(self, alpha=0.5, beta=0.1):
        super(AntiMeanLoss, self).__init__()
        self.alpha = alpha  # Weight for MSE loss
        self.beta = beta    # Weight for mean penalty
        self.mse = nn.MSELoss()
        
    def forward(self, pred, target, input_seq=None):
        """
        Args:
            pred: Model predictions [batch, seq_len, features]
            target: Ground truth [batch, seq_len, features]
            input_seq: Input sequence [batch, seq_len, features] (optional)
        """
        # Standard MSE loss
        mse_loss = self.mse(pred, target)
        
        # Calculate mean penalty
        mean_penalty = 0.0
        
        # Penalty 1: Predictions too close to target mean
        target_mean = torch.mean(target, dim=1, keepdim=True)  # [batch, 1, features]
        mean_diff = torch.abs(pred - target_mean)
        mean_penalty += torch.exp(-mean_diff.mean()) * self.beta
        
        # Penalty 2: Predictions too close to input mean (if provided)
        if input_seq is not None:
            input_mean = torch.mean(input_seq, dim=1, keepdim=True)  # [batch, 1, features]
            input_mean_diff = torch.abs(pred - input_mean)
            mean_penalty += torch.exp(-input_mean_diff.mean()) * self.beta
        
        # Total loss
        total_loss = self.alpha * mse_loss + mean_penalty
        
        return total_loss

class VarianceLoss(nn.Module):
    """
    Loss function that encourages predictions to have similar variance to targets.
    """
    
    def __init__(self, alpha=0.8, beta=0.2):
        super(VarianceLoss, self).__init__()
        self.alpha = alpha  # Weight for MSE loss
        self.beta = beta    # Weight for variance matching
        self.mse = nn.MSELoss()
        
    def forward(self, pred, target):
        """
        Args:
            pred: Model predictions [batch, seq_len, features]
            target: Ground truth [batch, seq_len, features]
        """
        # Standard MSE loss
        mse_loss = self.mse(pred, target)
        
        # Variance matching loss
        pred_var = torch.var(pred, dim=1)  # [batch, features]
        target_var = torch.var(target, dim=1)  # [batch, features]
        var_loss = self.mse(pred_var, target_var)
        
        # Total loss
        total_loss = self.alpha * mse_loss + self.beta * var_loss
        
        return total_loss

class PatternLoss(nn.Module):
    """
    Loss function that encourages learning temporal patterns.
    """
    
    def __init__(self, alpha=0.7, beta=0.3):
        super(PatternLoss, self).__init__()
        self.alpha = alpha  # Weight for MSE loss
        self.beta = beta    # Weight for gradient matching
        self.mse = nn.MSELoss()
        
    def forward(self, pred, target):
        """
        Args:
            pred: Model predictions [batch, seq_len, features]
            target: Ground truth [batch, seq_len, features]
        """
        # Standard MSE loss
        mse_loss = self.mse(pred, target)
        
        # Gradient matching loss (encourages similar temporal patterns)
        if pred.size(1) > 1:  # Only if sequence length > 1
            pred_grad = pred[:, 1:, :] - pred[:, :-1, :]  # First derivative
            target_grad = target[:, 1:, :] - target[:, :-1, :]
            grad_loss = self.mse(pred_grad, target_grad)
        else:
            grad_loss = 0.0
        
        # Total loss
        total_loss = self.alpha * mse_loss + self.beta * grad_loss
        
        return total_loss

class SparsityLoss(nn.Module):
    """
    Loss function that encourages sparsity in predictions to avoid constant predictions.
    """
    
    def __init__(self, alpha=0.8, beta=0.2):
        super(SparsityLoss, self).__init__()
        self.alpha = alpha  # Weight for MSE loss
        self.beta = beta    # Weight for sparsity penalty
        self.mse = nn.MSELoss()
        
    def forward(self, pred, target):
        """
        Args:
            pred: Model predictions [batch, seq_len, features]
            target: Ground truth [batch, seq_len, features]
        """
        # Standard MSE loss
        mse_loss = self.mse(pred, target)
        
        # Sparsity penalty: penalize predictions that are all similar
        pred_flat = pred.view(pred.size(0), -1)  # [batch, seq_len * features]
        pred_std = torch.std(pred_flat, dim=1)  # [batch]
        sparsity_penalty = torch.exp(-pred_std.mean()) * self.beta
        
        # Total loss
        total_loss = self.alpha * mse_loss + sparsity_penalty
        
        return total_loss

class BrainActivityLoss(nn.Module):
    """
    Combined loss function specifically designed for brain activity data.
    
    Combines multiple penalties to encourage learning meaningful neural patterns:
    1. MSE loss for accuracy
    2. Variance matching to maintain activity levels
    3. Gradient matching for temporal patterns
    4. Anti-mean penalty to discourage constant predictions
    """
    
    def __init__(self, alpha=0.6, beta=0.2, gamma=0.1, delta=0.1):
        super(BrainActivityLoss, self).__init__()
        self.alpha = alpha  # Weight for MSE loss
        self.beta = beta    # Weight for variance matching
        self.gamma = gamma  # Weight for gradient matching
        self.delta = delta  # Weight for anti-mean penalty
        self.mse = nn.MSELoss()
        
    def forward(self, pred, target, input_seq=None):
        """
        Args:
            pred: Model predictions [batch, seq_len, features]
            target: Ground truth [batch, seq_len, features]
            input_seq: Input sequence [batch, seq_len, features] (optional)
        """
        # Standard MSE loss
        mse_loss = self.mse(pred, target)
        
        # Variance matching loss
        pred_var = torch.var(pred, dim=1)  # [batch, features]
        target_var = torch.var(target, dim=1)  # [batch, features]
        var_loss = self.mse(pred_var, target_var)
        
        # Gradient matching loss (temporal patterns)
        grad_loss = 0.0
        if pred.size(1) > 1:
            pred_grad = pred[:, 1:, :] - pred[:, :-1, :]
            target_grad = target[:, 1:, :] - target[:, :-1, :]
            grad_loss = self.mse(pred_grad, target_grad)
        
        # Anti-mean penalty
        target_mean = torch.mean(target, dim=1, keepdim=True)  # [batch, 1, features]
        mean_diff = torch.abs(pred - target_mean)
        anti_mean_penalty = torch.exp(-mean_diff.mean())
        
        # Total loss
        total_loss = (self.alpha * mse_loss + 
                     self.beta * var_loss + 
                     self.gamma * grad_loss + 
                     self.delta * anti_mean_penalty)
        
        return total_loss

def get_loss_function(loss_name='mse'):
    """
    Factory function to get loss functions.
    
    Args:
        loss_name: Name of the loss function
        
    Returns:
        Loss function instance
    """
    if loss_name == 'mse':
        return nn.MSELoss()
    elif loss_name == 'huber':
        return nn.HuberLoss(delta=1.0)  # Less sensitive to outliers
    elif loss_name == 'anti_mean':
        return AntiMeanLoss()
    elif loss_name == 'mae':
        return nn.L1Loss()
    elif loss_name == 'variance':
        return VarianceLoss()
    elif loss_name == 'pattern':
        return PatternLoss()
    elif loss_name == 'sparsity':
        return SparsityLoss()
    elif loss_name == 'brain_activity':
        return BrainActivityLoss()
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")