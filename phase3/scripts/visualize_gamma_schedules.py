#!/usr/bin/env python3
"""
Visualize scale-specific gamma scheduling strategies.
"""
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Add phase3 to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.scale_gamma_manager import ScaleGammaManager


def plot_gamma_schedules(total_epochs=3000):
    """Plot all gamma scheduling strategies."""
    strategies = ['fixed', 'linear', 'cosine', 'hierarchical']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, strategy in enumerate(strategies):
        ax = axes[idx]
        
        # Create manager
        manager = ScaleGammaManager(strategy=strategy, total_epochs=total_epochs)
        
        # Collect gamma values
        epochs = list(range(0, total_epochs, 10))
        s1_gammas = []
        s2_gammas = []
        s3_gammas = []
        
        for epoch in epochs:
            gammas = manager.step(epoch)
            s1_gammas.append(gammas['s1'])
            s2_gammas.append(gammas['s2'])
            s3_gammas.append(gammas['s3'])
        
        # Plot
        ax.plot(epochs, s1_gammas, label='s1 (coarse)', linewidth=2, marker='o', markersize=3)
        ax.plot(epochs, s2_gammas, label='s2 (middle)', linewidth=2, marker='s', markersize=3)
        ax.plot(epochs, s3_gammas, label='s3 (fine)', linewidth=2, marker='^', markersize=3)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Gamma', fontsize=12)
        ax.set_title(f'{strategy.capitalize()} Strategy', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = 'scale_gamma_strategies.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'Plot saved to: {output_path}')
    
    # Print sample values
    print("\n" + "="*80)
    print("Sample Gamma Values at Key Epochs")
    print("="*80)
    
    for strategy in strategies:
        print(f"\n{strategy.upper()} Strategy:")
        print("-" * 80)
        
        manager = ScaleGammaManager(strategy=strategy, total_epochs=total_epochs)
        key_epochs = [0, 500, 1000, 1500, 2000, 2500, 2999]
        
        print(f"{'Epoch':<10} {'s1':<12} {'s2':<12} {'s3':<12}")
        print("-" * 80)
        
        for epoch in key_epochs:
            gammas = manager.step(epoch)
            print(f"{epoch:<10} {gammas['s1']:<12.4f} {gammas['s2']:<12.4f} {gammas['s3']:<12.4f}")


if __name__ == "__main__":
    plot_gamma_schedules()
