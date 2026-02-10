#!/usr/bin/env python3
"""
Quick test to verify GPU usage in scPerb
"""
import torch
from options.option import options
from models.scperb_model import scperb

print("=" * 60)
print("Testing GPU Usage in scPerb")
print("=" * 60)

# Initialize options
Opt = options()
opt = Opt.init()

print(f"\nDevice setting: {opt.device}")
print(f"CUDA available: {torch.cuda.is_available()}")

if opt.device != 'cpu' and torch.cuda.is_available():
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")

# Create model
print("\nCreating model...")
model = scperb(opt)

# Check model device
if opt.device != 'cpu':
    model_device = next(model.model.parameters()).device
    print(f"Model device: {model_device}")
    print(f"CUDA memory after model creation: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    
    # Create dummy input
    print("\nCreating dummy input tensors...")
    batch_size = 10
    con = torch.randn(batch_size, opt.input_dim)
    sti = torch.randn(batch_size, opt.input_dim)
    sty = torch.randn(opt.input_dim)
    
    print(f"Input tensors device (before): con={con.device}, sti={sti.device}, sty={sty.device}")
    
    # Set input (should move to GPU)
    model.set_input(con, sti, sty)
    
    print(f"Input tensors device (after set_input): con={model.con.device}, sti={model.sti.device}, sty={model.sty.device}")
    print(f"CUDA memory after moving data: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    
    # Forward pass
    print("\nRunning forward pass...")
    model.forward()
    print(f"CUDA memory after forward: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    
    print("\n✓ GPU usage test completed!")
else:
    print("\n⚠ Running on CPU - GPU not available or not detected")

print("=" * 60)
