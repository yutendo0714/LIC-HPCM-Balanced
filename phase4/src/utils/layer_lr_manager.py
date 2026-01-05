"""
Layer-wise learning rate manager for context-aware fine-tuning.
Enables differential learning rates for different model components.
"""
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
import re


class LayerLRManager:
    """
    Manages layer-wise learning rates for fine-tuning.
    
    Features:
    - Freeze/unfreeze specific layers
    - Different learning rates per layer group
    - Gradual unfreezing (progressive fine-tuning)
    - Learning rate warmup/decay per layer
    
    Typical usage for HPCM:
    - Freeze entropy_bottleneck, gaussian_conditional (entropy model)
    - Fine-tune context prediction layers with lower LR
    - Fine-tune g_a/g_s with medium LR
    - Fine-tune h_a/h_s with higher LR
    """

    def __init__(
        self,
        model: nn.Module,
        base_lr: float = 1e-4,
        layer_lr_multipliers: Optional[Dict[str, float]] = None,
        freeze_patterns: Optional[List[str]] = None,
    ):
        """
        Initialize layer-wise LR manager.
        
        Args:
            model: PyTorch model.
            base_lr: Base learning rate.
            layer_lr_multipliers: Dictionary mapping layer name patterns to LR multipliers.
                Example: {'g_a': 1.0, 'context': 0.1, 'entropy': 0.0}
            freeze_patterns: List of regex patterns for layers to freeze.
                Example: ['entropy_bottleneck', 'gaussian_conditional']
        """
        self.model = model
        self.base_lr = base_lr
        
        # Default multipliers for HPCM
        if layer_lr_multipliers is None:
            layer_lr_multipliers = {
                'g_a': 1.0,           # Analysis transform: full LR
                'g_s': 1.0,           # Synthesis transform: full LR
                'h_a': 0.5,           # Hyper analysis: half LR
                'h_s': 0.5,           # Hyper synthesis: half LR
                'context': 0.1,       # Context prediction: 10% LR
                'entropy': 0.0,       # Entropy model: frozen
            }
        
        self.layer_lr_multipliers = layer_lr_multipliers
        
        # Default freeze patterns
        if freeze_patterns is None:
            freeze_patterns = [
                r'entropy_bottleneck',
                r'gaussian_conditional',
            ]
        
        self.freeze_patterns = [re.compile(p) for p in freeze_patterns]
        
        # Track frozen parameters
        self.frozen_params = set()
        
        print(f'LayerLRManager initialized:')
        print(f'  Base LR: {base_lr}')
        print(f'  Layer multipliers: {layer_lr_multipliers}')
        print(f'  Freeze patterns: {freeze_patterns}')

    def apply_freeze(self):
        """Freeze parameters matching freeze patterns."""
        frozen_count = 0
        
        for name, param in self.model.named_parameters():
            # Check if matches any freeze pattern
            should_freeze = any(pattern.search(name) for pattern in self.freeze_patterns)
            
            if should_freeze:
                param.requires_grad = False
                self.frozen_params.add(name)
                frozen_count += 1
        
        print(f'Froze {frozen_count} parameters')
        return frozen_count

    def get_parameter_groups(self) -> List[Dict]:
        """
        Create parameter groups with different learning rates.
        
        Returns:
            List of parameter group dictionaries for optimizer.
        """
        # Group parameters by layer pattern
        param_groups = {}
        default_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            # Find matching layer pattern
            matched = False
            for pattern, multiplier in self.layer_lr_multipliers.items():
                if pattern in name:
                    if pattern not in param_groups:
                        param_groups[pattern] = {
                            'params': [],
                            'lr': self.base_lr * multiplier,
                            'name': pattern,
                        }
                    param_groups[pattern]['params'].append(param)
                    matched = True
                    break
            
            if not matched:
                default_params.append(param)
        
        # Build final list
        groups = []
        
        # Add specific groups
        for pattern, group in param_groups.items():
            if group['params']:
                groups.append(group)
                print(f"  Layer '{pattern}': {len(group['params'])} params, LR={group['lr']:.2e}")
        
        # Add default group
        if default_params:
            groups.append({
                'params': default_params,
                'lr': self.base_lr,
                'name': 'default',
            })
            print(f"  Layer 'default': {len(default_params)} params, LR={self.base_lr:.2e}")
        
        return groups

    def unfreeze_layer(self, pattern: str):
        """
        Unfreeze layers matching pattern.
        
        Args:
            pattern: Regex pattern to match layer names.
        """
        pattern_re = re.compile(pattern)
        unfrozen_count = 0
        
        for name, param in self.model.named_parameters():
            if pattern_re.search(name) and name in self.frozen_params:
                param.requires_grad = True
                self.frozen_params.remove(name)
                unfrozen_count += 1
        
        print(f'Unfroze {unfrozen_count} parameters matching "{pattern}"')
        return unfrozen_count

    def progressive_unfreeze(
        self,
        epoch: int,
        unfreeze_schedule: Dict[int, List[str]]
    ):
        """
        Gradually unfreeze layers based on epoch.
        
        Args:
            epoch: Current epoch.
            unfreeze_schedule: Dictionary mapping epochs to layer patterns.
                Example: {100: ['h_a', 'h_s'], 200: ['context']}
        """
        if epoch in unfreeze_schedule:
            patterns = unfreeze_schedule[epoch]
            print(f'Epoch {epoch}: Progressive unfreezing...')
            for pattern in patterns:
                self.unfreeze_layer(pattern)

    def get_trainable_params_count(self) -> Tuple[int, int]:
        """
        Get count of trainable vs total parameters.
        
        Returns:
            (trainable_count, total_count)
        """
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        return trainable, total

    def print_freeze_status(self):
        """Print detailed freeze status."""
        print('\n=== Layer Freeze Status ===')
        
        # Group by module
        module_status = {}
        for name, param in self.model.named_parameters():
            module_name = name.split('.')[0]
            if module_name not in module_status:
                module_status[module_name] = {'frozen': 0, 'trainable': 0}
            
            if param.requires_grad:
                module_status[module_name]['trainable'] += 1
            else:
                module_status[module_name]['frozen'] += 1
        
        # Print
        for module, status in sorted(module_status.items()):
            total = status['frozen'] + status['trainable']
            frozen_pct = 100 * status['frozen'] / total if total > 0 else 0
            print(f"  {module:<20} | Frozen: {status['frozen']:>4}/{total:<4} ({frozen_pct:>5.1f}%)")
        
        # Overall
        trainable, total = self.get_trainable_params_count()
        trainable_pct = 100 * trainable / total if total > 0 else 0
        print(f"\n  {'TOTAL':<20} | Trainable: {trainable:>8}/{total:<8} ({trainable_pct:>5.1f}%)")
        print('=' * 50)

    @staticmethod
    def create_hpcm_fine_tuning_config(
        base_lr: float = 1e-5,
        freeze_entropy: bool = True,
        context_lr_ratio: float = 0.1,
    ) -> Tuple[Dict[str, float], Optional[List[str]]]:
        """
        Create standard fine-tuning configuration for HPCM.
        
        Args:
            base_lr: Base learning rate for main layers.
            freeze_entropy: Whether to freeze entropy model.
            context_lr_ratio: LR ratio for context prediction layers.
        
        Returns:
            (layer_lr_multipliers, freeze_patterns)
        """
        layer_lr_multipliers = {
            'g_a': 1.0,                    # Analysis: full LR
            'g_s': 1.0,                    # Synthesis: full LR
            'h_a': 0.5,                    # Hyper analysis: half LR
            'h_s': 0.5,                    # Hyper synthesis: half LR
            'context': context_lr_ratio,   # Context: reduced LR
        }
        
        freeze_patterns = None
        if freeze_entropy:
            freeze_patterns = [
                r'entropy_bottleneck',
                r'gaussian_conditional',
                r'\.quantiles$',  # Quantile parameters
            ]
        
        return layer_lr_multipliers, freeze_patterns


class DiscriminativeLRScheduler:
    """
    Learning rate scheduler with different schedules per layer group.
    
    Enables:
    - Different warmup durations per layer
    - Different decay rates per layer
    - Layer-specific plateau detection
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: Optional[Dict[str, int]] = None,
        decay_epochs: Optional[Dict[str, List[int]]] = None,
        decay_rate: float = 0.5,
    ):
        """
        Initialize discriminative LR scheduler.
        
        Args:
            optimizer: Optimizer with parameter groups.
            warmup_epochs: Warmup duration per group {'group_name': epochs}.
            decay_epochs: Decay epochs per group {'group_name': [epoch1, epoch2]}.
            decay_rate: LR decay factor.
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs or {}
        self.decay_epochs = decay_epochs or {}
        self.decay_rate = decay_rate
        
        # Store initial LRs
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

    def step(self, epoch: int):
        """Update learning rates for current epoch."""
        for i, param_group in enumerate(self.optimizer.param_groups):
            group_name = param_group.get('name', f'group_{i}')
            base_lr = self.base_lrs[i]
            
            # Warmup
            warmup = self.warmup_epochs.get(group_name, 0)
            if epoch < warmup:
                lr = base_lr * (epoch + 1) / warmup
            else:
                lr = base_lr
                
                # Decay
                decay_steps = self.decay_epochs.get(group_name, [])
                for decay_epoch in decay_steps:
                    if epoch >= decay_epoch:
                        lr *= self.decay_rate
            
            param_group['lr'] = lr

    def get_lr(self) -> List[float]:
        """Get current learning rates for all groups."""
        return [group['lr'] for group in self.optimizer.param_groups]
