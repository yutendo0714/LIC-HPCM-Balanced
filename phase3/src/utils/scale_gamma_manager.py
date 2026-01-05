"""
Scale-specific gamma manager for Phase 3 hierarchical optimization.
Manages adaptive gamma values for each HPCM scale (s1, s2, s3).
"""
import torch
from typing import Dict, List, Optional, Tuple


class ScaleGammaManager:
    """
    Manages scale-specific gamma values with adaptive scheduling.
    
    Features:
    - Independent gamma values for each scale
    - Adaptive adjustment based on scale performance
    - Synchronized scheduling across scales
    - Dynamic rebalancing based on training progress
    """

    def __init__(
        self,
        initial_gammas: Optional[Dict[str, float]] = None,
        strategy: str = 'fixed',
        total_epochs: int = 3000,
        device: torch.device = None,
    ):
        """
        Initialize scale gamma manager.
        
        Args:
            initial_gammas: Initial gamma values {'s1': ..., 's2': ..., 's3': ...}.
                           Default: {'s1': 0.008, 's2': 0.006, 's3': 0.004}
            strategy: Scheduling strategy:
                - 'fixed': Keep gammas constant
                - 'linear': Linear decay
                - 'cosine': Cosine annealing
                - 'adaptive': Performance-based adaptation
                - 'hierarchical': HPCM-specific (emphasize different scales at different phases)
            total_epochs: Total training epochs.
            device: Torch device.
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.strategy = strategy
        self.total_epochs = total_epochs
        
        # Default initial gammas (higher for earlier scales)
        if initial_gammas is None:
            initial_gammas = {
                's1': 0.008,  # Scale 1: highest gamma (coarsest scale)
                's2': 0.006,  # Scale 2: medium gamma
                's3': 0.004,  # Scale 3: lowest gamma (finest scale)
            }
        
        self.initial_gammas = initial_gammas
        self.current_gammas = initial_gammas.copy()
        
        # Final gammas (typically 50-70% of initial)
        self.final_gammas = {
            's1': initial_gammas['s1'] * 0.5,
            's2': initial_gammas['s2'] * 0.5,
            's3': initial_gammas['s3'] * 0.5,
        }
        
        # Performance tracking for adaptive strategy
        self.scale_losses_history = {
            's1': [],
            's2': [],
            's3': [],
        }
        self.window_size = 50
        
        print(f'ScaleGammaManager initialized:')
        print(f'  Strategy: {strategy}')
        print(f'  Initial gammas: s1={initial_gammas["s1"]:.4f}, '
              f's2={initial_gammas["s2"]:.4f}, s3={initial_gammas["s3"]:.4f}')

    def step(
        self,
        epoch: int,
        scale_losses: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Update gammas for the current epoch.
        
        Args:
            epoch: Current epoch (0-indexed).
            scale_losses: Current losses for each scale (for adaptive strategy).
                         Format: {'s1': loss_s1, 's2': loss_s2, 's3': loss_s3}
        
        Returns:
            Updated gamma values as dictionary.
        """
        if self.strategy == 'fixed':
            return self.current_gammas
        
        elif self.strategy == 'linear':
            return self._linear_decay(epoch)
        
        elif self.strategy == 'cosine':
            return self._cosine_decay(epoch)
        
        elif self.strategy == 'adaptive':
            if scale_losses is None:
                raise ValueError("Adaptive strategy requires scale_losses")
            return self._adaptive_update(epoch, scale_losses)
        
        elif self.strategy == 'hierarchical':
            return self._hierarchical_schedule(epoch)
        
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _linear_decay(self, epoch: int) -> Dict[str, float]:
        """Linear decay from initial to final gammas."""
        progress = min(epoch / self.total_epochs, 1.0)
        
        gammas = {}
        for scale in ['s1', 's2', 's3']:
            initial = self.initial_gammas[scale]
            final = self.final_gammas[scale]
            gammas[scale] = initial - (initial - final) * progress
        
        self.current_gammas = gammas
        return gammas

    def _cosine_decay(self, epoch: int) -> Dict[str, float]:
        """Cosine annealing from initial to final gammas."""
        import math
        progress = min(epoch / self.total_epochs, 1.0)
        
        gammas = {}
        for scale in ['s1', 's2', 's3']:
            initial = self.initial_gammas[scale]
            final = self.final_gammas[scale]
            gammas[scale] = final + 0.5 * (initial - final) * (1 + math.cos(math.pi * progress))
        
        self.current_gammas = gammas
        return gammas

    def _adaptive_update(self, epoch: int, scale_losses: Dict[str, float]) -> Dict[str, float]:
        """
        Adaptively adjust gammas based on scale performance.
        
        Increases gamma for scales that are converging slowly,
        decreases for scales that have converged.
        """
        # Update loss history
        for scale, loss in scale_losses.items():
            self.scale_losses_history[scale].append(loss)
            if len(self.scale_losses_history[scale]) > self.window_size:
                self.scale_losses_history[scale].pop(0)
        
        # Compute convergence signals
        gammas = self.current_gammas.copy()
        
        if epoch > self.window_size:
            for scale in ['s1', 's2', 's3']:
                history = self.scale_losses_history[scale]
                
                # Check if scale has converged (low variance)
                recent_losses = history[-self.window_size:]
                std = torch.tensor(recent_losses).std().item()
                mean = torch.tensor(recent_losses).mean().item()
                cv = std / (mean + 1e-8)  # Coefficient of variation
                
                # Adjust gamma based on convergence
                if cv < 0.01:  # Converged -> reduce gamma
                    gammas[scale] *= 0.95
                elif cv > 0.05:  # Still varying -> increase gamma
                    gammas[scale] *= 1.02
                
                # Clamp to reasonable range
                initial = self.initial_gammas[scale]
                final = self.final_gammas[scale]
                gammas[scale] = max(final, min(initial * 1.5, gammas[scale]))
        
        self.current_gammas = gammas
        return gammas

    def _hierarchical_schedule(self, epoch: int) -> Dict[str, float]:
        """
        HPCM-specific hierarchical schedule.
        
        Phases:
        1. Warmup (0-300): Focus on s1 (coarse scale)
        2. Progressive (300-1500): Gradually shift focus to s2, s3
        3. Refinement (1500-2500): Balance all scales
        4. Fine-tuning (2500-3000): Emphasize s3 (fine details)
        """
        gammas = {}
        
        if epoch < 300:
            # Phase 1: Warmup - emphasize s1
            progress = epoch / 300
            gammas['s1'] = self.initial_gammas['s1'] * (1.0 + 0.2 * progress)
            gammas['s2'] = self.initial_gammas['s2'] * 0.8
            gammas['s3'] = self.initial_gammas['s3'] * 0.6
        
        elif epoch < 1500:
            # Phase 2: Progressive - balance s1, s2
            progress = (epoch - 300) / (1500 - 300)
            gammas['s1'] = self.initial_gammas['s1'] * (1.2 - 0.3 * progress)
            gammas['s2'] = self.initial_gammas['s2'] * (0.8 + 0.3 * progress)
            gammas['s3'] = self.initial_gammas['s3'] * (0.6 + 0.4 * progress)
        
        elif epoch < 2500:
            # Phase 3: Refinement - balance all
            progress = (epoch - 1500) / (2500 - 1500)
            gammas['s1'] = self.initial_gammas['s1'] * (0.9 - 0.2 * progress)
            gammas['s2'] = self.initial_gammas['s2'] * (1.1 - 0.3 * progress)
            gammas['s3'] = self.initial_gammas['s3'] * (1.0 + 0.1 * progress)
        
        else:
            # Phase 4: Fine-tuning - emphasize s3
            progress = (epoch - 2500) / (3000 - 2500)
            gammas['s1'] = self.initial_gammas['s1'] * (0.7 - 0.2 * progress)
            gammas['s2'] = self.initial_gammas['s2'] * (0.8 - 0.3 * progress)
            gammas['s3'] = self.initial_gammas['s3'] * (1.1 + 0.2 * progress)
        
        self.current_gammas = gammas
        return gammas

    def get_current_gammas(self) -> Dict[str, float]:
        """Get current gamma values."""
        return self.current_gammas.copy()

    def get_gamma_tensor(self) -> torch.Tensor:
        """
        Get current gammas as a tensor for optimizer.
        
        Returns:
            Tensor of shape [5]: [gamma_s1, gamma_s1, gamma_s2, gamma_s2, gamma_s3]
        """
        gammas = self.current_gammas
        return torch.tensor(
            [gammas['s1'], gammas['s1'], gammas['s2'], gammas['s2'], gammas['s3']],
            device=self.device
        )
