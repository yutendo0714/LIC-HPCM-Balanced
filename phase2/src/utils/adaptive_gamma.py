"""
Adaptive Gamma Scheduler for Phase 2
Dynamically adjusts gamma based on training progress.
"""
import torch
import numpy as np
from typing import Optional, Callable


class AdaptiveGammaScheduler:
    """
    Dynamically adjust gamma based on training epoch and loss dynamics.
    
    Strategies:
    - 'linear': Linear decay from initial to final gamma
    - 'cosine': Cosine annealing
    - 'step': Step-wise reduction
    - 'adaptive': Adapt based on loss convergence
    """
    
    def __init__(
        self,
        optimizer,
        strategy: str = 'cosine',
        initial_gamma: float = 0.006,
        final_gamma: float = 0.001,
        warmup_epochs: int = 100,
        total_epochs: int = 3000,
        step_epochs: Optional[list] = None,
        step_factor: float = 0.5
    ):
        """
        Initialize adaptive gamma scheduler.
        
        Args:
            optimizer: Balanced optimizer instance
            strategy: Scheduling strategy ('linear', 'cosine', 'step', 'adaptive')
            initial_gamma: Starting gamma value
            final_gamma: Ending gamma value
            warmup_epochs: Number of warmup epochs
            total_epochs: Total training epochs
            step_epochs: Epochs to reduce gamma (for 'step' strategy)
            step_factor: Reduction factor (for 'step' strategy)
        """
        self.optimizer = optimizer
        self.strategy = strategy
        self.initial_gamma = initial_gamma
        self.final_gamma = final_gamma
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.step_epochs = step_epochs or [1000, 2000, 2500]
        self.step_factor = step_factor
        
        # For adaptive strategy
        self.loss_history = []
        self.convergence_window = 50
        
        # Set initial gamma
        self.optimizer.gamma = initial_gamma
        self.current_gamma = initial_gamma
        
        print(f"AdaptiveGammaScheduler initialized:")
        print(f"  Strategy: {strategy}")
        print(f"  Initial gamma: {initial_gamma}")
        print(f"  Final gamma: {final_gamma}")
        print(f"  Warmup epochs: {warmup_epochs}")
    
    def step(self, epoch: int, loss: Optional[float] = None) -> float:
        """
        Update gamma for current epoch.
        
        Args:
            epoch: Current epoch number
            loss: Current loss value (for adaptive strategy)
            
        Returns:
            New gamma value
        """
        if epoch < self.warmup_epochs:
            # Warmup phase: use initial gamma
            gamma = self.initial_gamma
        elif self.strategy == 'linear':
            gamma = self._linear_schedule(epoch)
        elif self.strategy == 'cosine':
            gamma = self._cosine_schedule(epoch)
        elif self.strategy == 'step':
            gamma = self._step_schedule(epoch)
        elif self.strategy == 'adaptive':
            gamma = self._adaptive_schedule(epoch, loss)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        self.optimizer.gamma = gamma
        self.current_gamma = gamma
        
        return gamma
    
    def _linear_schedule(self, epoch: int) -> float:
        """Linear decay from initial to final gamma."""
        progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
        progress = min(progress, 1.0)
        gamma = self.initial_gamma - (self.initial_gamma - self.final_gamma) * progress
        return gamma
    
    def _cosine_schedule(self, epoch: int) -> float:
        """Cosine annealing schedule."""
        progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
        progress = min(progress, 1.0)
        gamma = self.final_gamma + (self.initial_gamma - self.final_gamma) * \
                0.5 * (1 + np.cos(np.pi * progress))
        return gamma
    
    def _step_schedule(self, epoch: int) -> float:
        """Step-wise reduction at specified epochs."""
        gamma = self.initial_gamma
        for step_epoch in self.step_epochs:
            if epoch >= step_epoch:
                gamma *= self.step_factor
        gamma = max(gamma, self.final_gamma)
        return gamma
    
    def _adaptive_schedule(self, epoch: int, loss: Optional[float]) -> float:
        """
        Adaptive schedule based on loss convergence.
        
        If loss is converging (small variance), reduce gamma for fine-tuning.
        If loss is unstable (large variance), increase gamma for exploration.
        """
        if loss is not None:
            self.loss_history.append(loss)
        
        # Use cosine as base schedule
        base_gamma = self._cosine_schedule(epoch)
        
        # Adjust based on loss dynamics
        if len(self.loss_history) >= self.convergence_window:
            recent_losses = self.loss_history[-self.convergence_window:]
            loss_std = np.std(recent_losses)
            loss_mean = np.mean(recent_losses)
            
            # Coefficient of variation
            cv = loss_std / (loss_mean + 1e-8)
            
            # If CV is small (converging), reduce gamma
            # If CV is large (unstable), increase gamma
            if cv < 0.01:
                # Converged: reduce gamma by 20%
                adjustment = 0.8
            elif cv > 0.05:
                # Unstable: increase gamma by 20%
                adjustment = 1.2
            else:
                # Normal: no adjustment
                adjustment = 1.0
            
            gamma = base_gamma * adjustment
            gamma = np.clip(gamma, self.final_gamma, self.initial_gamma * 1.5)
        else:
            gamma = base_gamma
        
        return gamma
    
    def get_current_gamma(self) -> float:
        """Get current gamma value."""
        return self.current_gamma
    
    def state_dict(self) -> dict:
        """Return state dictionary for checkpointing."""
        return {
            'current_gamma': self.current_gamma,
            'loss_history': self.loss_history,
            'strategy': self.strategy,
            'initial_gamma': self.initial_gamma,
            'final_gamma': self.final_gamma,
        }
    
    def load_state_dict(self, state_dict: dict):
        """Load state from dictionary."""
        self.current_gamma = state_dict['current_gamma']
        self.loss_history = state_dict['loss_history']
        self.optimizer.gamma = self.current_gamma


# Predefined schedules for HPCM
class HPCMGammaScheduler(AdaptiveGammaScheduler):
    """Predefined gamma schedule optimized for HPCM training."""
    
    def __init__(self, optimizer, total_epochs: int = 3000):
        """
        Initialize HPCM-specific gamma scheduler.
        
        Based on HPCM's training characteristics:
        - Early (0-500): Strong regularization (gamma=0.006)
        - Mid (500-1500): Balanced (gamma=0.003)
        - Late (1500-2750): Light regularization (gamma=0.001)
        - Final (2750+): Minimal regularization (gamma=0.0005)
        """
        super().__init__(
            optimizer=optimizer,
            strategy='step',
            initial_gamma=0.006,
            final_gamma=0.0005,
            warmup_epochs=100,
            total_epochs=total_epochs,
            step_epochs=[500, 1500, 2750],
            step_factor=0.5
        )
        
        print("HPCMGammaScheduler initialized with HPCM-optimized schedule")


if __name__ == "__main__":
    # Example usage
    import torch
    from pathlib import Path
    import sys
    
    # Add phase1 optimizer to path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "phase1"))
    from src.optimizers.balanced import Balanced
    
    # Create dummy optimizer
    model = torch.nn.Linear(10, 5)
    optimizer = Balanced(model.parameters(), lr=1e-3, gamma=0.003)
    
    # Test different strategies
    strategies = ['linear', 'cosine', 'step', 'adaptive']
    
    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"Testing {strategy} strategy")
        print('='*60)
        
        scheduler = AdaptiveGammaScheduler(
            optimizer,
            strategy=strategy,
            initial_gamma=0.006,
            final_gamma=0.001,
            total_epochs=3000
        )
        
        # Simulate training
        for epoch in [0, 100, 500, 1000, 1500, 2000, 2500, 2999]:
            loss = 0.5 - epoch / 10000  # Dummy decreasing loss
            gamma = scheduler.step(epoch, loss)
            print(f"Epoch {epoch:4d}: gamma = {gamma:.6f}")
