#!/usr/bin/env python3
"""
Test adaptive gamma scheduler with different strategies.
"""
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt

# Add phase2 to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.adaptive_gamma import AdaptiveGammaScheduler, HPCMGammaScheduler

# Add phase1 to path
phase1_path = str(Path(__file__).parent.parent.parent / "phase1")
sys.path.insert(0, phase1_path)

def test_scheduler(strategy='cosine', epochs=3000):
    """Test a gamma scheduling strategy."""
    import torch
    from src.optimizers.balanced import Balanced
    
    # Create dummy optimizer
    model = torch.nn.Linear(10, 5)
    optimizer = Balanced(model.parameters(), lr=1e-3, gamma=0.006)
    
    # Create scheduler
    if strategy == 'hpcm':
        scheduler = HPCMGammaScheduler(optimizer, total_epochs=epochs)
    else:
        scheduler = AdaptiveGammaScheduler(
            optimizer,
            strategy=strategy,
            initial_gamma=0.006,
            final_gamma=0.001,
            total_epochs=epochs
        )
    
    # Simulate training
    gammas = []
    epoch_list = []
    
    for epoch in range(epochs):
        # Simulate decreasing loss
        loss = 1.0 - epoch / epochs * 0.5 + (epoch % 100) * 0.001
        gamma = scheduler.step(epoch, loss)
        
        if epoch % 10 == 0:
            gammas.append(gamma)
            epoch_list.append(epoch)
    
    return epoch_list, gammas

def main():
    """Test all strategies and plot results."""
    strategies = ['linear', 'cosine', 'step', 'adaptive', 'hpcm']
    
    plt.figure(figsize=(12, 6))
    
    for strategy in strategies:
        epochs, gammas = test_scheduler(strategy)
        plt.plot(epochs, gammas, marker='o' if strategy == 'step' else None, 
                label=strategy, linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Gamma', fontsize=12)
    plt.title('Gamma Scheduling Strategies Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = 'gamma_strategies_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    # Print sample values
    print("\nSample gamma values at key epochs:")
    print("-" * 70)
    print(f"{'Strategy':<12} {'Epoch 0':<10} {'Epoch 500':<10} {'Epoch 1500':<10} {'Epoch 2999':<10}")
    print("-" * 70)
    
    for strategy in strategies:
        epochs_sample = [0, 500, 1500, 2999]
        import torch
        from src.optimizers.balanced import Balanced
        
        model = torch.nn.Linear(10, 5)
        optimizer = Balanced(model.parameters(), lr=1e-3, gamma=0.006)
        
        if strategy == 'hpcm':
            scheduler = HPCMGammaScheduler(optimizer, total_epochs=3000)
        else:
            scheduler = AdaptiveGammaScheduler(
                optimizer, strategy=strategy,
                initial_gamma=0.006, final_gamma=0.001, total_epochs=3000
            )
        
        values = []
        for epoch in epochs_sample:
            gamma = scheduler.step(epoch, 0.5)
            values.append(f"{gamma:.4f}")
        
        print(f"{strategy:<12} {values[0]:<10} {values[1]:<10} {values[2]:<10} {values[3]:<10}")
    print("-" * 70)

if __name__ == "__main__":
    main()
