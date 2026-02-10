#!/usr/bin/env python3
"""
CUDA diagnostic script for scPerb
Run this to check if CUDA/GPU is properly configured.
"""
import sys

print("=" * 60)
print("CUDA/GPU Diagnostic Check")
print("=" * 60)

# Check PyTorch installation
try:
    import torch
    print(f"✓ PyTorch version: {torch.__version__}")
except ImportError:
    print("✗ PyTorch not installed!")
    sys.exit(1)

# Check CUDA availability
print(f"\nCUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"✓ CUDA is available!")
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  cuDNN version: {torch.backends.cudnn.version()}")
    print(f"  Number of GPUs: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"\n  GPU {i}:")
        print(f"    Name: {torch.cuda.get_device_name(i)}")
        print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        print(f"    Current device: {i == torch.cuda.current_device()}")
    
    # Test tensor creation on GPU
    try:
        x = torch.randn(3, 3).cuda()
        print(f"\n✓ Successfully created tensor on GPU: {x.device}")
        del x
    except Exception as e:
        print(f"\n✗ Failed to create tensor on GPU: {e}")
else:
    print("\n✗ CUDA is NOT available")
    print("\nPossible reasons:")
    print("  1. PyTorch was installed without CUDA support (CPU-only version)")
    print("  2. CUDA drivers are not installed")
    print("  3. CUDA version mismatch between PyTorch and drivers")
    print("\nTo fix:")
    print("  - Install CUDA-enabled PyTorch:")
    print("    Visit https://pytorch.org/get-started/locally/")
    print("  - Check CUDA drivers: nvidia-smi")
    print("  - Verify CUDA version matches PyTorch requirements")

print("\n" + "=" * 60)
