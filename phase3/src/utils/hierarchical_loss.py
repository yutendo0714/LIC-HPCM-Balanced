"""
Hierarchical loss computation for HPCM with 5-task decomposition.
Handles multi-scale rate-distortion loss calculation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class HierarchicalLoss(nn.Module):
    """
    Hierarchical Rate-Distortion Loss for HPCM.
    
    Computes 5 separate task losses:
    - s1_distortion: Reconstruction error at scale 1
    - s1_bpp: Bits-per-pixel at scale 1
    - s2_distortion: Reconstruction error at scale 2
    - s2_bpp: Bits-per-pixel at scale 2
    - s3_bpp: Bits-per-pixel at scale 3 (hyperprior)
    
    The total loss is: λ * distortion + bpp
    But we decompose this into 5 tasks for hierarchical balancing.
    """

    def __init__(
        self,
        lmbda: float = 0.013,
        metric: str = 'mse',
        return_details: bool = False,
    ):
        """
        Initialize hierarchical loss.
        
        Args:
            lmbda: Rate-distortion trade-off parameter.
            metric: Distortion metric ('mse' or 'ms-ssim').
            return_details: Whether to return detailed loss breakdown.
        """
        super().__init__()
        self.lmbda = lmbda
        self.metric = metric
        self.return_details = return_details
        
        print(f'HierarchicalLoss initialized: λ={lmbda}, metric={metric}')

    def forward(
        self,
        output: Dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute hierarchical loss.
        
        Args:
            output: Model output dictionary containing:
                - 'x_hat': Reconstructed image
                - 'likelihoods': Dictionary of likelihoods for each scale
                    - 'y': Latent representation likelihood (scale 3)
                    - 'z': Hyperprior likelihood (scale 3)
                - Optional: intermediate reconstructions for progressive models
            target: Target image
        
        Returns:
            - total_loss: Scalar total loss
            - task_losses: Dictionary with individual task losses
        """
        batch_size = target.size(0)
        num_pixels = batch_size * target.size(2) * target.size(3)
        
        # 1. Compute distortion (reconstruction error)
        x_hat = output['x_hat']
        
        if self.metric == 'mse':
            distortion = F.mse_loss(x_hat, target, reduction='none')
            distortion = distortion.view(batch_size, -1).mean(dim=1).mean()
        elif self.metric == 'ms-ssim':
            # Note: ms-ssim computation would go here
            # For now, fall back to MSE
            distortion = F.mse_loss(x_hat, target)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
        
        # 2. Compute BPP (bits-per-pixel)
        likelihoods = output['likelihoods']
        
        # BPP for latent y (main representation)
        y_bpp = self._compute_bpp(likelihoods['y'], num_pixels)
        
        # BPP for hyperprior z
        z_bpp = self._compute_bpp(likelihoods['z'], num_pixels)
        
        # Total BPP
        total_bpp = y_bpp + z_bpp
        
        # 3. Decompose into 5 tasks
        # For HPCM, we approximate multi-scale decomposition:
        # - s1 (coarsest): handles ~40% of distortion, ~30% of BPP
        # - s2 (middle): handles ~35% of distortion, ~40% of BPP
        # - s3 (finest): handles ~25% of distortion, ~30% of BPP
        
        # Distortion distribution (learned empirically or heuristically)
        s1_distortion = distortion * 0.40
        s2_distortion = distortion * 0.35
        s3_distortion = distortion * 0.25
        
        # BPP distribution
        s1_bpp = total_bpp * 0.30
        s2_bpp = total_bpp * 0.40
        s3_bpp = total_bpp * 0.30
        
        # 4. Compute task losses (scaled by lambda)
        task_losses = {
            's1_distortion': self.lmbda * s1_distortion,
            's1_bpp': s1_bpp,
            's2_distortion': self.lmbda * s2_distortion,
            's2_bpp': s2_bpp,
            's3_bpp': s3_bpp,
        }
        
        # 5. Total loss (for reference)
        total_loss = self.lmbda * distortion + total_bpp
        
        # 6. Return as tensor for optimizer
        task_loss_tensor = torch.stack([
            task_losses['s1_distortion'],
            task_losses['s1_bpp'],
            task_losses['s2_distortion'],
            task_losses['s2_bpp'],
            task_losses['s3_bpp'],
        ])
        
        if self.return_details:
            details = {
                'total_loss': total_loss,
                'distortion': distortion,
                'bpp': total_bpp,
                'y_bpp': y_bpp,
                'z_bpp': z_bpp,
                **task_losses,
            }
            return task_loss_tensor, details
        else:
            return task_loss_tensor, task_losses

    def _compute_bpp(self, likelihood: torch.Tensor, num_pixels: int) -> torch.Tensor:
        """
        Compute bits-per-pixel from likelihood.
        
        Args:
            likelihood: Likelihood tensor from entropy model.
            num_pixels: Total number of pixels in batch.
        
        Returns:
            BPP as scalar tensor.
        """
        # Negative log-likelihood in bits
        bpp = -torch.log2(likelihood).sum() / num_pixels
        return bpp

    def compute_scale_losses(
        self,
        output: Dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Compute losses for each scale separately (for monitoring).
        
        Returns:
            Dictionary with scale-wise losses.
        """
        task_loss_tensor, task_losses = self.forward(output, target)
        
        scale_losses = {
            's1': (task_losses['s1_distortion'] + task_losses['s1_bpp']).item(),
            's2': (task_losses['s2_distortion'] + task_losses['s2_bpp']).item(),
            's3': task_losses['s3_bpp'].item(),
        }
        
        return scale_losses


class AdaptiveHierarchicalLoss(HierarchicalLoss):
    """
    Adaptive hierarchical loss with learned scale distributions.
    
    Instead of fixed 40-35-25 distortion split, learns the optimal
    distribution during training.
    """

    def __init__(
        self,
        lmbda: float = 0.013,
        metric: str = 'mse',
        return_details: bool = False,
        device: torch.device = None,
    ):
        super().__init__(lmbda, metric, return_details)
        
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Learnable distribution parameters (logits)
        self.distortion_logits = nn.Parameter(torch.tensor([0.4, 0.35, 0.25], device=self.device))
        self.bpp_logits = nn.Parameter(torch.tensor([0.3, 0.4, 0.3], device=self.device))
        
        print('AdaptiveHierarchicalLoss initialized with learnable scale distributions')

    def forward(
        self,
        output: Dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute loss with learned scale distributions."""
        batch_size = target.size(0)
        num_pixels = batch_size * target.size(2) * target.size(3)
        
        # Compute total distortion and BPP
        x_hat = output['x_hat']
        distortion = F.mse_loss(x_hat, target)
        
        likelihoods = output['likelihoods']
        y_bpp = self._compute_bpp(likelihoods['y'], num_pixels)
        z_bpp = self._compute_bpp(likelihoods['z'], num_pixels)
        total_bpp = y_bpp + z_bpp
        
        # Compute scale distributions (softmax to ensure sum to 1)
        distortion_dist = F.softmax(self.distortion_logits, dim=0)
        bpp_dist = F.softmax(self.bpp_logits, dim=0)
        
        # Decompose
        s1_distortion = distortion * distortion_dist[0]
        s2_distortion = distortion * distortion_dist[1]
        s3_distortion = distortion * distortion_dist[2]
        
        s1_bpp = total_bpp * bpp_dist[0]
        s2_bpp = total_bpp * bpp_dist[1]
        s3_bpp = total_bpp * bpp_dist[2]
        
        # Task losses
        task_losses = {
            's1_distortion': self.lmbda * s1_distortion,
            's1_bpp': s1_bpp,
            's2_distortion': self.lmbda * s2_distortion,
            's2_bpp': s2_bpp,
            's3_bpp': s3_bpp,
        }
        
        total_loss = self.lmbda * distortion + total_bpp
        
        task_loss_tensor = torch.stack([
            task_losses['s1_distortion'],
            task_losses['s1_bpp'],
            task_losses['s2_distortion'],
            task_losses['s2_bpp'],
            task_losses['s3_bpp'],
        ])
        
        if self.return_details:
            details = {
                'total_loss': total_loss,
                'distortion': distortion,
                'bpp': total_bpp,
                'distortion_dist': distortion_dist.detach().cpu().tolist(),
                'bpp_dist': bpp_dist.detach().cpu().tolist(),
                **task_losses,
            }
            return task_loss_tensor, details
        else:
            return task_loss_tensor, task_losses
