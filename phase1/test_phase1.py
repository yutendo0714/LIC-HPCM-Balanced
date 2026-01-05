"""
Phase 1 Implementation Test
Quick smoke test to verify the implementation works.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.optimizers.balanced import Balanced

def test_balanced_optimizer():
    """Test Balanced optimizer initialization and basic operations."""
    print("Testing Balanced optimizer...")
    
    # Create dummy model
    model = torch.nn.Linear(10, 5)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Initialize optimizer
    optimizer = Balanced(
        model.parameters(),
        lr=1e-3,
        n_tasks=2,
        gamma=0.003,
        w_lr=0.025,
        device=device
    )
    
    # Set min losses
    optimizer.set_min_losses(torch.tensor([-1.0, -1.0], device=device))
    
    # Simulate training step
    dummy_input = torch.randn(4, 10, device=device)
    dummy_target = torch.randn(4, 5, device=device)
    
    # Forward pass
    output = model(dummy_input)
    loss1 = torch.nn.functional.mse_loss(output, dummy_target)
    loss2 = output.abs().mean()
    
    # Task losses
    task_losses = torch.stack([loss1, loss2])
    
    # Backward with balancing
    weighted_loss = optimizer.backward_with_task_balancing(
        task_losses,
        shared_parameters=model.parameters()
    )
    
    # Step
    optimizer.step(task_losses=task_losses)
    
    # Update task weights
    optimizer.update_task_weights(task_losses.detach())
    
    print("✓ Balanced optimizer test passed!")
    print(f"  - Weighted loss: {weighted_loss.item():.4f}")
    print(f"  - Task weights: {torch.softmax(optimizer.w, -1).cpu().numpy()}")
    
    return True

def test_loss_function():
    """Test the extended RateDistortionLoss."""
    print("\nTesting RateDistortionLoss...")
    
    sys.path.insert(0, os.path.dirname(__file__))
    from train import RateDistortionLoss
    
    criterion = RateDistortionLoss(lmbda=0.013)
    
    # Dummy output
    dummy_output = {
        "x_hat": torch.randn(2, 3, 256, 256),
        "likelihoods": {
            "y": torch.rand(2, 320, 16, 16).clamp(1e-9, 1.0),
            "z": torch.rand(2, 256, 4, 4).clamp(1e-9, 1.0)
        }
    }
    dummy_target = torch.randn(2, 3, 256, 256)
    
    out = criterion(dummy_output, dummy_target)
    
    # Check all required keys
    required_keys = ["bpp_loss", "mse_loss", "distortion", "loss", "psnr", "y_bpp", "z_bpp"]
    for key in required_keys:
        assert key in out, f"Missing key: {key}"
    
    # Check distortion calculation
    expected_dist = 0.013 * 255 ** 2 * out["mse_loss"].item()
    actual_dist = out["distortion"].item()
    assert abs(expected_dist - actual_dist) < 1e-3, f"Distortion mismatch: {expected_dist} vs {actual_dist}"
    
    # Check loss composition
    expected_loss = out["distortion"].item() + out["bpp_loss"].item()
    actual_loss = out["loss"].item()
    assert abs(expected_loss - actual_loss) < 1e-3, f"Loss mismatch: {expected_loss} vs {actual_loss}"
    
    print("✓ RateDistortionLoss test passed!")
    print(f"  - Loss: {out['loss'].item():.4f}")
    print(f"  - Distortion: {out['distortion'].item():.4f}")
    print(f"  - BPP: {out['bpp_loss'].item():.4f}")
    
    return True

def test_imports():
    """Test that all imports work correctly."""
    print("\nTesting imports...")
    
    try:
        from src.optimizers import Balanced
        print("✓ Balanced optimizer import successful")
    except ImportError as e:
        print(f"✗ Failed to import Balanced optimizer: {e}")
        return False
    
    try:
        sys.path.insert(0, os.path.dirname(__file__))
        from train import RateDistortionLoss, AverageMeter, CustomDataParallel
        print("✓ Training modules import successful")
    except ImportError as e:
        print(f"✗ Failed to import training modules: {e}")
        return False
    
    return True

def main():
    """Run all tests."""
    print("=" * 60)
    print("Phase 1 Implementation Tests")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Balanced Optimizer", test_balanced_optimizer),
        ("Loss Function", test_loss_function),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ {test_name} failed with error: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(success for _, success in results)
    print("=" * 60)
    if all_passed:
        print("All tests passed! ✓")
        return 0
    else:
        print("Some tests failed! ✗")
        return 1

if __name__ == "__main__":
    sys.exit(main())
