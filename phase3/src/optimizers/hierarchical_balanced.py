"""
Hierarchical Balanced optimizer for HPCM with 5-task decomposition.
Extends Phase 1/2 Balanced optimizer to handle scale-specific losses.

Phase 3 introduces:
- 5-task decomposition: s1_distortion, s1_bpp, s2_distortion, s2_bpp, s3_bpp
- Scale-specific gamma values
- Hierarchical task weighting
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from typing import Callable, Iterable, Tuple, Optional, List, Dict, Union

# Import Phase 1 Balanced optimizer as base
phase1_path = str(Path(__file__).parent.parent.parent.parent / "phase1")
sys.path.insert(0, phase1_path)
from src.optimizers.balanced import Balanced


class HierarchicalBalanced(Balanced):
    """
    Hierarchical Balanced optimizer for HPCM with multi-scale optimization.
    
    Extends the base Balanced optimizer to support:
    - 5 tasks: s1_distortion, s1_bpp, s2_distortion, s2_bpp, s3_bpp
    - Scale-specific gamma values
    - Adaptive scale weighting
    
    Parameters:
        params: Iterable of parameters to optimize.
        lr: Learning rate (default: 1e-3).
        betas: Adam's betas (default: (0.9, 0.999)).
        eps: Epsilon for numerical stability (default: 1e-8).
        weight_decay: L2 penalty (default: 0.0).
        amsgrad: Whether to use AMSGrad (default: False).
        n_tasks: Number of tasks (must be 5 for hierarchical) (default: 5).
        gamma_s1: Gamma for scale 1 tasks (default: 0.008).
        gamma_s2: Gamma for scale 2 tasks (default: 0.006).
        gamma_s3: Gamma for scale 3 tasks (default: 0.004).
        w_lr: Learning rate for task weights (default: 0.025).
        max_norm: Max gradient norm (default: 1.0).
        scale_weights: Importance weights for each scale [s1, s2, s3] (default: [0.3, 0.4, 0.3]).
        device: Torch device.
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
        n_tasks: int = 5,
        gamma_s1: float = 0.008,
        gamma_s2: float = 0.006,
        gamma_s3: float = 0.004,
        w_lr: float = 0.025,
        max_norm: float = 1.0,
        scale_weights: Optional[List[float]] = None,
        device: torch.device = None,
    ):
        if n_tasks != 5:
            raise ValueError(f"HierarchicalBalanced requires n_tasks=5, got {n_tasks}")
        
        # Initialize base optimizer with average gamma
        avg_gamma = (gamma_s1 + gamma_s2 + gamma_s3) / 3.0
        super().__init__(
            params=params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            n_tasks=n_tasks,
            gamma=avg_gamma,
            w_lr=w_lr,
            max_norm=max_norm,
            device=device,
        )
        
        # Scale-specific gamma values
        self.gamma_s1 = gamma_s1
        self.gamma_s2 = gamma_s2
        self.gamma_s3 = gamma_s3
        self.scale_gammas = torch.tensor([gamma_s1, gamma_s1, gamma_s2, gamma_s2, gamma_s3], 
                                          device=self.device)
        
        # Scale importance weights (default: s2 > s1 = s3)
        if scale_weights is None:
            scale_weights = [0.3, 0.4, 0.3]  # [s1, s2, s3]
        self.scale_weights = torch.tensor(scale_weights, device=self.device)
        
        # Task indices for each scale
        self.task_indices = {
            's1': [0, 1],  # s1_distortion, s1_bpp
            's2': [2, 3],  # s2_distortion, s2_bpp
            's3': [4],     # s3_bpp
        }
        
        print(f'HierarchicalBalanced initialized:')
        print(f'  - 5 tasks: s1_dist, s1_bpp, s2_dist, s2_bpp, s3_bpp')
        print(f'  - Scale gammas: s1={gamma_s1}, s2={gamma_s2}, s3={gamma_s3}')
        print(f'  - Scale weights: s1={scale_weights[0]}, s2={scale_weights[1]}, s3={scale_weights[2]}')
        print(f'  - w_lr={w_lr}, max_norm={max_norm}')

    def get_weighted_loss(self, losses: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted loss using scale-specific gammas and FAMO.
        
        Args:
            losses: Tensor of shape [5] containing:
                [s1_distortion, s1_bpp, s2_distortion, s2_bpp, s3_bpp]
        
        Returns:
            Weighted scalar loss.
        """
        if losses.shape[0] != 5:
            raise ValueError(f"Expected 5 task losses, got {losses.shape[0]}")
        
        self.prev_loss = losses.detach().clone()
        
        # Compute task weights using FAMO
        z = F.softmax(self.w, -1)
        D = losses - self.min_losses + 1e-8
        c = (z / D).sum().detach()
        
        # Apply scale-specific modulation
        scale_modulation = self._compute_scale_modulation()
        z_modulated = z * scale_modulation
        z_modulated = z_modulated / z_modulated.sum()  # Renormalize
        
        # Weighted loss
        loss = (D.log() * z_modulated / c).sum()
        
        return loss

    def _compute_scale_modulation(self) -> torch.Tensor:
        """
        Compute scale-based modulation for task weights.
        
        Returns:
            Tensor of shape [5] with scale-specific modulation factors.
        """
        # Map scale weights to task weights
        # [s1_dist, s1_bpp, s2_dist, s2_bpp, s3_bpp]
        modulation = torch.zeros(5, device=self.device)
        modulation[0:2] = self.scale_weights[0]  # s1 tasks
        modulation[2:4] = self.scale_weights[1]  # s2 tasks
        modulation[4] = self.scale_weights[2]    # s3 task
        
        return modulation

    def update_task_weights(self, curr_loss: torch.Tensor):
        """
        Update task weights with scale-specific gamma regularization.
        
        Args:
            curr_loss: Current task losses [5].
        """
        if self.prev_loss is None:
            return
        
        # Compute delta (improvement signal)
        delta = (self.prev_loss - self.min_losses + 1e-8).log() - \
                (curr_loss - self.min_losses + 1e-8).log()
        
        with torch.enable_grad():
            d = torch.autograd.grad(F.softmax(self.w, -1),
                                    self.w,
                                    grad_outputs=delta.detach())[0]
        
        self.w_opt.zero_grad()
        
        # Apply scale-specific gamma regularization
        # weight_decay_term = gamma_i * w_i for each task i
        gamma_reg = self.scale_gammas * self.w
        d += gamma_reg
        
        self.w.grad = d
        self.w_opt.step()

    def get_task_weights(self) -> Dict[str, float]:
        """
        Get current task weights as a dictionary.
        
        Returns:
            Dictionary mapping task names to their weights.
        """
        z = F.softmax(self.w, -1)
        return {
            's1_distortion': z[0].item(),
            's1_bpp': z[1].item(),
            's2_distortion': z[2].item(),
            's2_bpp': z[3].item(),
            's3_bpp': z[4].item(),
        }

    def get_scale_contributions(self) -> Dict[str, float]:
        """
        Get contribution of each scale to the total loss.
        
        Returns:
            Dictionary mapping scale names to their contributions.
        """
        z = F.softmax(self.w, -1)
        contributions = {
            's1': (z[0] + z[1]).item(),
            's2': (z[2] + z[3]).item(),
            's3': z[4].item(),
        }
        return contributions

    def set_scale_weights(self, scale_weights: List[float]):
        """
        Dynamically update scale importance weights.
        
        Args:
            scale_weights: List of [s1_weight, s2_weight, s3_weight].
        """
        if len(scale_weights) != 3:
            raise ValueError(f"Expected 3 scale weights, got {len(scale_weights)}")
        
        self.scale_weights = torch.tensor(scale_weights, device=self.device)
        print(f'Updated scale weights: s1={scale_weights[0]:.3f}, s2={scale_weights[1]:.3f}, s3={scale_weights[2]:.3f}')

    def set_scale_gammas(self, gamma_s1: float, gamma_s2: float, gamma_s3: float):
        """
        Dynamically update scale-specific gamma values.
        
        Args:
            gamma_s1: Gamma for scale 1.
            gamma_s2: Gamma for scale 2.
            gamma_s3: Gamma for scale 3.
        """
        self.gamma_s1 = gamma_s1
        self.gamma_s2 = gamma_s2
        self.gamma_s3 = gamma_s3
        self.scale_gammas = torch.tensor([gamma_s1, gamma_s1, gamma_s2, gamma_s2, gamma_s3],
                                          device=self.device)
        print(f'Updated scale gammas: s1={gamma_s1:.4f}, s2={gamma_s2:.4f}, s3={gamma_s3:.4f}')
