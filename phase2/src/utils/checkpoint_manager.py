"""
Checkpoint Manager for Phase 2
Handles checkpoint saving, loading, and best model tracking.
"""
import os
import shutil
import torch
from pathlib import Path
from typing import Dict, Optional
import json


class CheckpointManager:
    """Manage model checkpoints with automatic best model tracking."""
    
    def __init__(
        self,
        save_dir: str,
        keep_last_n: int = 3,
        keep_best: int = 3,
        metric_name: str = 'loss',
        mode: str = 'min'
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            save_dir: Directory to save checkpoints
            keep_last_n: Number of recent checkpoints to keep
            keep_best: Number of best checkpoints to keep
            metric_name: Metric name to track for best models
            mode: 'min' or 'max' for metric comparison
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.keep_last_n = keep_last_n
        self.keep_best = keep_best
        self.metric_name = metric_name
        self.mode = mode
        
        # Track checkpoints
        self.checkpoints = []  # [(epoch, metric_value, path), ...]
        self.best_checkpoints = []
        
        # Best metric value
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        
        print(f"CheckpointManager initialized at {save_dir}")
        print(f"  Tracking: {metric_name} ({mode})")
        print(f"  Keep last: {keep_last_n}, Keep best: {keep_best}")
    
    def is_better(self, new_value: float, old_value: float) -> bool:
        """Check if new metric value is better than old value."""
        if self.mode == 'min':
            return new_value < old_value
        else:
            return new_value > old_value
    
    def save_checkpoint(
        self,
        epoch: int,
        model_state: Dict,
        optimizer_state: Dict,
        metric_value: float,
        additional_info: Optional[Dict] = None
    ) -> str:
        """
        Save checkpoint and manage saved files.
        
        Args:
            epoch: Current epoch
            model_state: Model state dict
            optimizer_state: Optimizer state dict  
            metric_value: Metric value for this checkpoint
            additional_info: Additional information to save
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer_state,
            self.metric_name: metric_value,
        }
        
        if additional_info:
            checkpoint.update(additional_info)
        
        # Save checkpoint
        checkpoint_path = self.save_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Track this checkpoint
        self.checkpoints.append((epoch, metric_value, str(checkpoint_path)))
        
        # Check if this is a best checkpoint
        is_best = self.is_better(metric_value, self.best_metric)
        if is_best:
            self.best_metric = metric_value
            best_path = self.save_dir / 'checkpoint_best.pth'
            shutil.copy2(checkpoint_path, best_path)
            print(f"  → New best model! {self.metric_name}={metric_value:.4f}")
            
            # Track in best list
            self.best_checkpoints.append((epoch, metric_value, str(checkpoint_path)))
            self.best_checkpoints.sort(key=lambda x: x[1], 
                                      reverse=(self.mode == 'max'))
        
        # Clean up old checkpoints
        self._cleanup_checkpoints()
        
        # Save checkpoint info
        self._save_checkpoint_info()
        
        return str(checkpoint_path)
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints based on keep_last_n and keep_best."""
        if len(self.checkpoints) <= self.keep_last_n:
            return
        
        # Sort by epoch (most recent last)
        self.checkpoints.sort(key=lambda x: x[0])
        
        # Determine which to keep
        keep_paths = set()
        
        # Keep last N
        for epoch, metric, path in self.checkpoints[-self.keep_last_n:]:
            keep_paths.add(path)
        
        # Keep best N
        best_sorted = sorted(self.best_checkpoints, 
                           key=lambda x: x[1],
                           reverse=(self.mode == 'max'))
        for epoch, metric, path in best_sorted[:self.keep_best]:
            keep_paths.add(path)
        
        # Remove checkpoints not in keep list
        for epoch, metric, path in self.checkpoints:
            if path not in keep_paths and os.path.exists(path):
                try:
                    os.remove(path)
                    print(f"  Removed old checkpoint: {os.path.basename(path)}")
                except Exception as e:
                    print(f"  Warning: Could not remove {path}: {e}")
        
        # Update checkpoint list
        self.checkpoints = [(e, m, p) for e, m, p in self.checkpoints 
                           if p in keep_paths]
    
    def _save_checkpoint_info(self):
        """Save checkpoint information to JSON."""
        info = {
            'best_metric': self.best_metric,
            'metric_name': self.metric_name,
            'mode': self.mode,
            'checkpoints': [
                {'epoch': e, 'metric': m, 'path': p} 
                for e, m, p in self.checkpoints
            ],
            'best_checkpoints': [
                {'epoch': e, 'metric': m, 'path': p}
                for e, m, p in self.best_checkpoints[:self.keep_best]
            ]
        }
        
        info_path = self.save_dir / 'checkpoint_info.json'
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict:
        """Load checkpoint from path."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"  {self.metric_name}: {checkpoint.get(self.metric_name, 'unknown')}")
        
        return checkpoint
    
    def load_best_checkpoint(self) -> Dict:
        """Load the best checkpoint."""
        best_path = self.save_dir / 'checkpoint_best.pth'
        return self.load_checkpoint(str(best_path))
    
    def load_latest_checkpoint(self) -> Optional[Dict]:
        """Load the most recent checkpoint."""
        if not self.checkpoints:
            return None
        
        self.checkpoints.sort(key=lambda x: x[0])
        latest_path = self.checkpoints[-1][2]
        return self.load_checkpoint(latest_path)
    
    def get_best_metric(self) -> float:
        """Get best metric value."""
        return self.best_metric
    
    def list_checkpoints(self):
        """Print list of all checkpoints."""
        print("\nAvailable checkpoints:")
        print("-" * 70)
        for epoch, metric, path in sorted(self.checkpoints, key=lambda x: x[0]):
            marker = "  [BEST]" if path in [p for _, _, p in self.best_checkpoints[:1]] else ""
            print(f"Epoch {epoch:4d}: {self.metric_name}={metric:.4f}  {os.path.basename(path)}{marker}")
        print("-" * 70)


if __name__ == "__main__":
    # Test checkpoint manager
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(
            save_dir=tmpdir,
            keep_last_n=3,
            keep_best=2,
            metric_name='loss',
            mode='min'
        )
        
        # Simulate saving checkpoints
        dummy_model = {'param': torch.randn(10, 5)}
        dummy_optimizer = {'lr': 1e-3}
        
        losses = [1.0, 0.8, 0.9, 0.7, 0.75, 0.6, 0.65, 0.5]
        
        for epoch, loss in enumerate(losses):
            manager.save_checkpoint(
                epoch=epoch,
                model_state=dummy_model,
                optimizer_state=dummy_optimizer,
                metric_value=loss
            )
        
        manager.list_checkpoints()
        
        print(f"\nBest metric: {manager.get_best_metric():.4f}")
        
        # Test loading
        best_ckpt = manager.load_best_checkpoint()
        print(f"\nLoaded best checkpoint from epoch {best_ckpt['epoch']}")
