"""
Metrics Calculation Module

Compute compression metrics: PSNR, MS-SSIM, VMAF, etc.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """
    Calculate compression quality metrics.
    
    Supports: PSNR, MS-SSIM, VMAF (optional), pixel-level metrics
    """
    
    def __init__(
        self,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        """
        Args:
            device: Computation device
        """
        self.device = device
    
    def psnr(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
        max_val: float = 255.0,
    ) -> float:
        """
        Calculate PSNR between two images.
        
        Args:
            img1: Image tensor (B, C, H, W) or (C, H, W)
            img2: Image tensor (B, C, H, W) or (C, H, W)
            max_val: Maximum pixel value
        
        Returns:
            PSNR in dB
        """
        if img1.dim() == 3:
            img1 = img1.unsqueeze(0)
        if img2.dim() == 3:
            img2 = img2.unsqueeze(0)
        
        mse = F.mse_loss(img1, img2, reduction='mean')
        
        if mse == 0:
            return float('inf')
        
        psnr_val = 10 * torch.log10(max_val ** 2 / mse)
        return float(psnr_val.item())
    
    def ms_ssim(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
        max_val: float = 255.0,
        weights: Optional[torch.Tensor] = None,
    ) -> float:
        """
        Calculate MS-SSIM between two images.
        
        Args:
            img1: Image tensor (B, C, H, W) or (C, H, W)
            img2: Image tensor (B, C, H, W) or (C, H, W)
            max_val: Maximum pixel value
            weights: Multi-scale weights
        
        Returns:
            MS-SSIM value [0, 1]
        """
        if img1.dim() == 3:
            img1 = img1.unsqueeze(0)
        if img2.dim() == 3:
            img2 = img2.unsqueeze(0)
        
        # Default weights (from MS-SSIM paper)
        if weights is None:
            weights = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], device=self.device)
        
        levels = weights.size(0)
        mssim = []
        mcs = []
        
        for i in range(levels):
            ssim_val, cs_val = self._ssim(img1, img2, max_val=max_val)
            mssim.append(ssim_val)
            mcs.append(cs_val)
            
            # Downsample for next level
            if i < levels - 1:
                img1 = F.avg_pool2d(img1, kernel_size=2, stride=2)
                img2 = F.avg_pool2d(img2, kernel_size=2, stride=2)
        
        mssim = torch.stack(mssim)
        mcs = torch.stack(mcs)
        
        # MS-SSIM formula
        ms_ssim_val = torch.prod(mcs[:-1] ** weights[:-1]) * (mssim[-1] ** weights[-1])
        
        return float(ms_ssim_val.item())
    
    def _ssim(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
        max_val: float = 255.0,
        window_size: int = 11,
        sigma: float = 1.5,
        K1: float = 0.01,
        K2: float = 0.03,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate SSIM and contrast sensitivity.
        
        Returns:
            (ssim, contrast_sensitivity)
        """
        C1 = (K1 * max_val) ** 2
        C2 = (K2 * max_val) ** 2
        
        # Gaussian window
        window = self._gaussian_window(window_size, sigma, img1.size(1)).to(img1.device)
        
        # Compute means
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=img1.size(1))
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=img2.size(1))
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        # Compute variances and covariance
        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=img1.size(1)) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=img2.size(1)) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=img1.size(1)) - mu1_mu2
        
        # SSIM formula
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        # Contrast sensitivity
        cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
        
        return ssim_map.mean(), cs_map.mean()
    
    def _gaussian_window(
        self,
        window_size: int,
        sigma: float,
        channels: int,
    ) -> torch.Tensor:
        """Create Gaussian window for SSIM."""
        gauss = torch.Tensor([
            np.exp(-(x - window_size // 2) ** 2 / (2 * sigma ** 2))
            for x in range(window_size)
        ])
        gauss = gauss / gauss.sum()
        
        _1D_window = gauss.unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channels, 1, window_size, window_size).contiguous()
        
        return window
    
    def compute_all(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
        bpp: float,
    ) -> Dict[str, float]:
        """
        Compute all available metrics.
        
        Args:
            original: Original image
            reconstructed: Reconstructed image
            bpp: Bits per pixel
        
        Returns:
            Dict of metrics
        """
        metrics = {}
        
        # Ensure images are in [0, 255] range
        if original.max() <= 1.0:
            original = original * 255.0
            reconstructed = reconstructed * 255.0
        
        # PSNR
        metrics['psnr'] = self.psnr(original, reconstructed)
        
        # MS-SSIM
        try:
            metrics['ms_ssim'] = self.ms_ssim(original, reconstructed)
        except Exception as e:
            logger.warning(f"MS-SSIM calculation failed: {e}")
            metrics['ms_ssim'] = None
        
        # BPP
        metrics['bpp'] = bpp
        
        # Additional metrics
        metrics['mse'] = float(F.mse_loss(original, reconstructed).item())
        
        return metrics


def compute_metrics(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    bpp: float,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
) -> Dict[str, float]:
    """
    Convenience function to compute all metrics.
    
    Args:
        original: Original image
        reconstructed: Reconstructed image
        bpp: Bits per pixel
        device: Computation device
    
    Returns:
        Dict of metrics
    """
    calculator = MetricsCalculator(device=device)
    return calculator.compute_all(original, reconstructed, bpp)


if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create example images
    original = torch.rand(1, 3, 256, 256) * 255.0
    reconstructed = original + torch.randn_like(original) * 5.0  # Add noise
    
    # Calculate metrics
    calc = MetricsCalculator()
    
    psnr_val = calc.psnr(original, reconstructed)
    ms_ssim_val = calc.ms_ssim(original, reconstructed)
    
    print(f"PSNR: {psnr_val:.2f} dB")
    print(f"MS-SSIM: {ms_ssim_val:.4f}")
    
    # All metrics
    all_metrics = calc.compute_all(original, reconstructed, bpp=0.5)
    print(f"\nAll metrics: {all_metrics}")
