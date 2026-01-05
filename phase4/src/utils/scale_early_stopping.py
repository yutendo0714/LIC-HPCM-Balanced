"""
Scale-specific early stopping for hierarchical models.
Monitors each scale independently and can stop training per scale.
"""
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json


class ScaleEarlyStopping:
    """
    Early stopping with scale-specific monitoring.
    
    Features:
    - Independent patience per scale
    - Scale-specific metrics
    - Partial training stop (freeze converged scales)
    - Automatic best model selection per scale
    """

    def __init__(
        self,
        scales: List[str] = ['s1', 's2', 's3'],
        patience: int = 100,
        min_delta: float = 1e-4,
        mode: str = 'min',
        save_dir: Optional[str] = None,
    ):
        """
        Initialize scale-specific early stopping.
        
        Args:
            scales: List of scale names to monitor.
            patience: Number of epochs with no improvement before stopping.
            min_delta: Minimum change to qualify as improvement.
            mode: 'min' or 'max' (whether lower or higher is better).
            save_dir: Directory to save best models per scale.
        """
        self.scales = scales
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.save_dir = Path(save_dir) if save_dir else None
        
        # State per scale
        self.best_scores = {scale: None for scale in scales}
        self.best_epochs = {scale: 0 for scale in scales}
        self.counters = {scale: 0 for scale in scales}
        self.stopped = {scale: False for scale in scales}
        
        # History
        self.history = {scale: [] for scale in scales}
        
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f'ScaleEarlyStopping initialized:')
        print(f'  Scales: {scales}')
        print(f'  Patience: {patience} epochs')
        print(f'  Min delta: {min_delta}')
        print(f'  Mode: {mode}')

    def step(
        self,
        epoch: int,
        metrics: Dict[str, float],
        model_state: Optional[Dict] = None,
    ) -> Dict[str, bool]:
        """
        Check for early stopping and update state.
        
        Args:
            epoch: Current epoch.
            metrics: Dictionary of metrics per scale {'s1': loss, 's2': loss, ...}.
            model_state: Model state dict to save if improved.
        
        Returns:
            Dictionary of whether each scale should stop {'s1': False, 's2': True, ...}.
        """
        stop_signals = {}
        
        for scale in self.scales:
            if scale not in metrics:
                stop_signals[scale] = self.stopped[scale]
                continue
            
            if self.stopped[scale]:
                stop_signals[scale] = True
                continue
            
            score = metrics[scale]
            self.history[scale].append(score)
            
            # Check improvement
            improved = self._check_improvement(scale, score)
            
            if improved:
                self.best_scores[scale] = score
                self.best_epochs[scale] = epoch
                self.counters[scale] = 0
                
                # Save best model for this scale
                if model_state and self.save_dir:
                    self._save_best_model(scale, epoch, score, model_state)
                
                print(f'  [{scale}] Improved! Best: {score:.4f} (epoch {epoch})')
            
            else:
                self.counters[scale] += 1
                
                if self.counters[scale] >= self.patience:
                    self.stopped[scale] = True
                    print(f'  [{scale}] Early stopping triggered (patience={self.patience})')
                    print(f'  [{scale}] Best score: {self.best_scores[scale]:.4f} at epoch {self.best_epochs[scale]}')
            
            stop_signals[scale] = self.stopped[scale]
        
        return stop_signals

    def _check_improvement(self, scale: str, score: float) -> bool:
        """Check if current score is an improvement."""
        if self.best_scores[scale] is None:
            return True
        
        if self.mode == 'min':
            return score < (self.best_scores[scale] - self.min_delta)
        else:
            return score > (self.best_scores[scale] + self.min_delta)

    def _save_best_model(
        self,
        scale: str,
        epoch: int,
        score: float,
        model_state: Dict
    ):
        """Save best model for a scale."""
        filepath = self.save_dir / f'best_{scale}_epoch{epoch}.pth'
        torch.save({
            'epoch': epoch,
            'scale': scale,
            'score': score,
            'model_state_dict': model_state,
        }, filepath)
        print(f'  [{scale}] Saved best model to {filepath}')

    def should_stop_training(self) -> bool:
        """Check if all scales have stopped."""
        return all(self.stopped.values())

    def get_best_epochs(self) -> Dict[str, int]:
        """Get best epoch for each scale."""
        return self.best_epochs.copy()

    def get_best_scores(self) -> Dict[str, float]:
        """Get best score for each scale."""
        return self.best_scores.copy()

    def get_status_summary(self) -> Dict:
        """Get summary of current status."""
        return {
            'best_scores': self.get_best_scores(),
            'best_epochs': self.get_best_epochs(),
            'counters': self.counters.copy(),
            'stopped': self.stopped.copy(),
        }

    def save_history(self, filepath: str):
        """Save training history to JSON."""
        with open(filepath, 'w') as f:
            json.dump({
                'scales': self.scales,
                'history': self.history,
                'best_scores': {k: float(v) if v is not None else None 
                                for k, v in self.best_scores.items()},
                'best_epochs': self.best_epochs,
            }, f, indent=2)
        print(f'Saved early stopping history to {filepath}')

    def plot_history(self, output_path: str = 'early_stopping_history.png'):
        """Plot training history with early stopping markers."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print('matplotlib not available, skipping plot')
            return
        
        fig, axes = plt.subplots(len(self.scales), 1, figsize=(10, 4 * len(self.scales)))
        if len(self.scales) == 1:
            axes = [axes]
        
        for idx, scale in enumerate(self.scales):
            ax = axes[idx]
            history = self.history[scale]
            
            if not history:
                continue
            
            epochs = list(range(len(history)))
            ax.plot(epochs, history, label=f'{scale} loss', linewidth=2)
            
            # Mark best epoch
            if self.best_epochs[scale] < len(history):
                best_epoch = self.best_epochs[scale]
                best_score = history[best_epoch]
                ax.axvline(best_epoch, color='green', linestyle='--', 
                          label=f'Best (epoch {best_epoch})')
                ax.scatter([best_epoch], [best_score], color='green', s=100, zorder=5)
            
            # Mark stopping point
            if self.stopped[scale]:
                stop_epoch = self.best_epochs[scale] + self.patience
                if stop_epoch < len(history):
                    ax.axvline(stop_epoch, color='red', linestyle='--', 
                              label=f'Stopped (epoch {stop_epoch})')
            
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Loss', fontsize=12)
            ax.set_title(f'Scale {scale} Training History', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f'Saved early stopping plot to {output_path}')


class AdaptivePatienceEarlyStopping(ScaleEarlyStopping):
    """
    Early stopping with adaptive patience based on training progress.
    
    Increases patience if model is still improving gradually.
    """

    def __init__(
        self,
        scales: List[str] = ['s1', 's2', 's3'],
        initial_patience: int = 50,
        max_patience: int = 200,
        patience_increase: int = 20,
        improvement_threshold: float = 0.01,
        **kwargs
    ):
        """
        Initialize adaptive patience early stopping.
        
        Args:
            scales: List of scale names.
            initial_patience: Starting patience value.
            max_patience: Maximum patience allowed.
            patience_increase: How much to increase patience on gradual improvement.
            improvement_threshold: Threshold for gradual improvement detection.
        """
        super().__init__(scales=scales, patience=initial_patience, **kwargs)
        
        self.initial_patience = initial_patience
        self.max_patience = max_patience
        self.patience_increase = patience_increase
        self.improvement_threshold = improvement_threshold
        
        # Track recent improvements
        self.recent_improvements = {scale: [] for scale in scales}

    def step(self, epoch: int, metrics: Dict[str, float], model_state: Optional[Dict] = None):
        """Step with adaptive patience adjustment."""
        # Check for gradual improvements
        for scale in self.scales:
            if scale not in metrics or self.stopped[scale]:
                continue
            
            score = metrics[scale]
            
            # Track recent improvements
            if self.best_scores[scale] is not None:
                improvement = self.best_scores[scale] - score if self.mode == 'min' else score - self.best_scores[scale]
                self.recent_improvements[scale].append(improvement)
                
                # Keep last 10 improvements
                if len(self.recent_improvements[scale]) > 10:
                    self.recent_improvements[scale].pop(0)
                
                # Check if consistently improving (even if small)
                if len(self.recent_improvements[scale]) >= 5:
                    avg_improvement = np.mean(self.recent_improvements[scale])
                    
                    if avg_improvement > 0 and avg_improvement < self.improvement_threshold:
                        # Gradual improvement detected, increase patience
                        new_patience = min(self.patience + self.patience_increase, self.max_patience)
                        if new_patience > self.patience:
                            print(f'  [{scale}] Gradual improvement detected, increasing patience: {self.patience} → {new_patience}')
                            self.patience = new_patience
        
        # Continue with normal step
        return super().step(epoch, metrics, model_state)
