#!/usr/bin/env python3
"""
Analyze layer-wise parameter statistics.
"""
import sys
from pathlib import Path
import torch

# Add phase4 to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models import HPCM_Base


def analyze_model_layers():
    """Analyze HPCM model layer structure."""
    model = HPCM_Base.HPCM(N=192, M=320)
    
    print("="*80)
    print("HPCM Model Layer Analysis")
    print("="*80)
    
    # Group by module
    module_params = {}
    
    for name, param in model.named_parameters():
        module_name = name.split('.')[0]
        
        if module_name not in module_params:
            module_params[module_name] = {
                'count': 0,
                'total_params': 0,
                'layers': []
            }
        
        module_params[module_name]['count'] += 1
        module_params[module_name]['total_params'] += param.numel()
        module_params[module_name]['layers'].append(name)
    
    # Print summary
    total_params = sum(info['total_params'] for info in module_params.values())
    
    print(f"\n{'Module':<25} {'Layers':<10} {'Parameters':<15} {'%':<10}")
    print("-"*80)
    
    for module, info in sorted(module_params.items()):
        pct = 100 * info['total_params'] / total_params
        params_m = info['total_params'] / 1e6
        print(f"{module:<25} {info['count']:<10} {params_m:>10.2f}M {pct:>8.1f}%")
    
    print("-"*80)
    print(f"{'TOTAL':<25} {sum(info['count'] for info in module_params.values()):<10} "
          f"{total_params/1e6:>10.2f}M {'100.0':>8}%")
    
    # Print detailed layer list for context
    print("\n" + "="*80)
    print("Context-related layers (for fine-tuning):")
    print("="*80)
    
    context_layers = [name for name in model.state_dict().keys() if 'context' in name.lower()]
    for layer in sorted(context_layers):
        print(f"  {layer}")
    
    # Print entropy-related layers
    print("\n" + "="*80)
    print("Entropy model layers (typically frozen):")
    print("="*80)
    
    entropy_layers = [name for name in model.state_dict().keys() 
                     if 'entropy' in name.lower() or 'gaussian' in name.lower() or 'quantile' in name.lower()]
    for layer in sorted(entropy_layers):
        print(f"  {layer}")


if __name__ == "__main__":
    analyze_model_layers()
