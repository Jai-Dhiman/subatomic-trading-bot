"""
Test script to verify class_weights device handling fix.
"""

import torch
import numpy as np
from src.models.trading_transformer_v2 import TradingLossV2, calculate_class_weights

print("=" * 70)
print("TESTING DEVICE HANDLING FOR CLASS_WEIGHTS")
print("=" * 70)

# Simulate imbalanced training data
decisions = np.random.choice([0, 1, 2], size=1000, p=[0.30, 0.54, 0.16])
class_weights = calculate_class_weights(decisions)
print(f"\nClass weights calculated: {[f'{w:.2f}' for w in class_weights]}")

# Create loss function with class weights
criterion = TradingLossV2(
    price_weight=0.10,
    decision_weight=0.10,
    profit_weight=0.80,
    class_weights=class_weights
)

print("\n1. Testing CPU (default)...")
# Create dummy tensors on CPU
batch_size = 4
predictions_cpu = {
    'predicted_price': torch.randn(batch_size, 48),
    'trading_decisions': torch.randn(batch_size, 48, 3),
    'trade_quantities': torch.rand(batch_size, 48) * 5
}
targets_cpu = {
    'price': torch.rand(batch_size) * 0.05 + 0.02,
    'decisions': torch.randint(0, 3, (batch_size,)),
    'quantities': torch.rand(batch_size) * 5,
    'consumption': torch.rand(batch_size) * 2
}

loss_cpu, loss_dict_cpu = criterion(predictions_cpu, targets_cpu)
print(f"   ✓ CPU loss: {loss_cpu.item():.4f}")
print(f"   ✓ Market profit: ${loss_dict_cpu['market_profit']:.4f}")

# Test if CUDA is available
if torch.cuda.is_available():
    print("\n2. Testing GPU (CUDA)...")
    
    # Move criterion to GPU
    criterion_gpu = criterion.cuda()
    
    # Move data to GPU
    predictions_gpu = {k: v.cuda() for k, v in predictions_cpu.items()}
    targets_gpu = {k: v.cuda() for k, v in targets_cpu.items()}
    
    # This should work without errors now
    loss_gpu, loss_dict_gpu = criterion_gpu(predictions_gpu, targets_gpu)
    print(f"   ✓ GPU loss: {loss_gpu.item():.4f}")
    print(f"   ✓ Market profit: ${loss_dict_gpu['market_profit']:.4f}")
    
    # Verify class_weights moved to GPU
    if hasattr(criterion_gpu, 'class_weights') and criterion_gpu.class_weights is not None:
        print(f"   ✓ Class weights device: {criterion_gpu.class_weights.device}")
        assert str(criterion_gpu.class_weights.device).startswith('cuda'), "Class weights should be on CUDA!"
        print(f"   ✓ Class weights correctly on GPU!")
    
    print("\n" + "=" * 70)
    print("✓ GPU TEST PASSED - No device mismatch errors!")
    print("=" * 70)
else:
    print("\n⚠ CUDA not available, skipping GPU test")
    print("   (This is expected on CPU-only machines)")
    print("\n" + "=" * 70)
    print("✓ CPU TEST PASSED")
    print("=" * 70)
