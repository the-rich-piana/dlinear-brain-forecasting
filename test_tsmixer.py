"""
Simple test script for TSMixer implementation
"""
import torch
import argparse
from models.TSMixer import Model

# Create mock config similar to DLinear project
class MockConfig:
    def __init__(self):
        self.seq_len = 32
        self.pred_len = 8  
        self.enc_in = 100  # 100 features like the Activity dataset uses
        
        # TSMixer specific
        self.n_blocks = 4
        self.dropout = 0.1
        self.activation = 'relu'
        self.mlp_dim = 50
        self.time_mix_mlp_dim = -1  # Use seq_len
        
        # Normalization
        self.instance_norm = True
        self.revert_instance_norm = True
        
        # Residual settings
        self.time_mix_residual = True
        self.feature_mix_residual = True
        self.block_residual = False
        self.time_mix_only = False

def test_tsmixer():
    print("Testing TSMixer implementation...")
    
    # Create config
    config = MockConfig()
    
    # Create model
    model = Model(config)
    model.eval()
    
    # Create test input [Batch, seq_len, features]
    batch_size = 4
    x = torch.randn(batch_size, config.seq_len, config.enc_in)
    
    print(f"Input shape: {x.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: [{batch_size}, {config.pred_len}, {config.enc_in}]")
    
    # Verify output shape
    expected_shape = (batch_size, config.pred_len, config.enc_in)
    if output.shape == expected_shape:
        print("✅ TSMixer test passed!")
        return True
    else:
        print(f"❌ Shape mismatch! Got {output.shape}, expected {expected_shape}")
        return False

if __name__ == "__main__":
    success = test_tsmixer()
    if success:
        print("\nTSMixer is ready to be integrated!")
    else:
        print("\nTSMixer needs debugging.")