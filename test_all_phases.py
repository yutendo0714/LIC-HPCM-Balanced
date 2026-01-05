#!/usr/bin/env python3
"""
Comprehensive Test Script for All Phases (1-5)

Tests the implementation and integration of all phases.
"""

import sys
import importlib
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

# Color codes for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'


class TestResult:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def add_pass(self, test_name: str):
        self.passed += 1
        print(f"{GREEN}✓{RESET} {test_name}")
    
    def add_fail(self, test_name: str, error: str):
        self.failed += 1
        self.errors.append((test_name, error))
        print(f"{RED}✗{RESET} {test_name}")
        print(f"  {RED}Error:{RESET} {error}")
    
    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*80}")
        print(f"TEST SUMMARY")
        print(f"{'='*80}")
        print(f"Total: {total}")
        print(f"{GREEN}Passed: {self.passed}{RESET}")
        print(f"{RED}Failed: {self.failed}{RESET}")
        if total > 0:
            print(f"Success Rate: {self.passed/total*100:.1f}%")
        
        if self.errors:
            print(f"\n{RED}Failed Tests:{RESET}")
            for test_name, error in self.errors:
                print(f"  - {test_name}")
                print(f"    {error[:200]}")
        
        return self.failed == 0


def test_file_structure(result: TestResult):
    """Test that all expected files exist"""
    print(f"\n{BLUE}Testing File Structure{RESET}")
    
    base_path = Path(__file__).parent
    
    # Phase 1 files
    phase1_files = [
        'phase1/src/optimizers/balanced.py',
        'phase1/train.py',
        'phase1/README.md',
    ]
    
    for file_path in phase1_files:
        full_path = base_path / file_path
        if full_path.exists():
            result.add_pass(f"File exists: {file_path}")
        else:
            result.add_fail(f"File missing: {file_path}", "File not found")
    
    # Phase 2 files
    phase2_files = [
        'phase2/src/utils/adaptive_gamma.py',
        'phase2/src/utils/checkpoint_manager.py',
        'phase2/train.py',
        'phase2/README.md',
    ]
    
    for file_path in phase2_files:
        full_path = base_path / file_path
        if full_path.exists():
            result.add_pass(f"File exists: {file_path}")
        else:
            result.add_fail(f"File missing: {file_path}", "File not found")
    
    # Phase 3 files
    phase3_files = [
        'phase3/src/optimizers/hierarchical_balanced.py',
        'phase3/src/utils/scale_gamma_manager.py',
        'phase3/src/utils/hierarchical_loss.py',
        'phase3/train.py',
        'phase3/README.md',
    ]
    
    for file_path in phase3_files:
        full_path = base_path / file_path
        if full_path.exists():
            result.add_pass(f"File exists: {file_path}")
        else:
            result.add_fail(f"File missing: {file_path}", "File not found")
    
    # Phase 4 files
    phase4_files = [
        'phase4/src/utils/layer_lr_manager.py',
        'phase4/src/utils/scale_early_stopping.py',
        'phase4/train.py',
        'phase4/README.md',
    ]
    
    for file_path in phase4_files:
        full_path = base_path / file_path
        if full_path.exists():
            result.add_pass(f"File exists: {file_path}")
        else:
            result.add_fail(f"File missing: {file_path}", "File not found")
    
    # Phase 5 files
    phase5_files = [
        'phase5/src/evaluation/bd_rate.py',
        'phase5/src/evaluation/rd_curve.py',
        'phase5/src/evaluation/metrics.py',
        'phase5/src/evaluation/comparator.py',
        'phase5/evaluate.py',
        'phase5/README.md',
    ]
    
    for file_path in phase5_files:
        full_path = base_path / file_path
        if full_path.exists():
            result.add_pass(f"File exists: {file_path}")
        else:
            result.add_fail(f"File missing: {file_path}", "File not found")


def test_phase1_imports(result: TestResult):
    """Test Phase 1: Basic Balanced R-D imports"""
    print(f"\n{BLUE}Testing Phase 1: Basic Balanced R-D{RESET}")
    
    # Add phase1 to path
    phase1_path = Path(__file__).parent / 'phase1'
    if str(phase1_path) not in sys.path:
        sys.path.insert(0, str(phase1_path))
    
    # Test balanced optimizer import
    try:
        from src.optimizers.balanced import Balanced
        result.add_pass("Phase 1: Import Balanced optimizer")
    except ImportError as e:
        result.add_fail("Phase 1: Import Balanced optimizer", str(e))


def test_phase1_functionality(result: TestResult):
    """Test Phase 1: Basic functionality"""
    try:
        # Add phase1 to path
        phase1_path = Path(__file__).parent / 'phase1'
        if str(phase1_path) not in sys.path:
            sys.path.insert(0, str(phase1_path))
        
        from src.optimizers.balanced import Balanced
        import torch
        import torch.nn as nn
        
        # Create dummy model
        model = nn.Linear(10, 10)
        
        # Test optimizer creation
        optimizer = Balanced(
            model.parameters(),
            lr=1e-4,
            n_tasks=2,
        )
        
        result.add_pass("Phase 1: Create Balanced optimizer")
        
    except Exception as e:
        result.add_fail("Phase 1: Functionality test", str(e))


def test_phase2_imports(result: TestResult):
    """Test Phase 2: Adaptive Optimization imports"""
    print(f"\n{BLUE}Testing Phase 2: Adaptive Optimization{RESET}")
    
    # Add phase2 to path
    phase2_path = Path(__file__).parent / 'phase2'
    if str(phase2_path) not in sys.path:
        sys.path.insert(0, str(phase2_path))
    
    # Test adaptive gamma manager
    try:
        from src.utils.adaptive_gamma import AdaptiveGammaManager
        result.add_pass("Phase 2: Import AdaptiveGammaManager")
    except ImportError as e:
        result.add_fail("Phase 2: Import AdaptiveGammaManager", str(e))
    
    # Test checkpoint manager
    try:
        from src.utils.checkpoint_manager import CheckpointManager
        result.add_pass("Phase 2: Import CheckpointManager")
    except ImportError as e:
        result.add_fail("Phase 2: Import CheckpointManager", str(e))


def test_phase2_functionality(result: TestResult):
    """Test Phase 2: Adaptive functionality"""
    try:
        # Add phase2 to path
        phase2_path = Path(__file__).parent / 'phase2'
        if str(phase2_path) not in sys.path:
            sys.path.insert(0, str(phase2_path))
        
        from src.utils.adaptive_gamma import AdaptiveGammaScheduler
        import torch
        import torch.nn as nn
        
        # Create dummy model and optimizer
        model = nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters())
        
        # Create gamma scheduler
        gamma_scheduler = AdaptiveGammaScheduler(
            optimizer=optimizer,
            initial_gamma=1.0,
            strategy='cosine',
        )
        
        result.add_pass("Phase 2: Create AdaptiveGammaScheduler")
        
        # Test gamma update
        new_gamma = gamma_scheduler.step(epoch=10)
        
        result.add_pass("Phase 2: Gamma step works")
        
    except Exception as e:
        result.add_fail("Phase 2: Functionality test", str(e))


def test_phase3_imports(result: TestResult):
    """Test Phase 3: Hierarchical Balanced imports"""
    print(f"\n{BLUE}Testing Phase 3: Hierarchical Balanced{RESET}")
    
    # Add phase3 to path
    phase3_path = Path(__file__).parent / 'phase3'
    if str(phase3_path) not in sys.path:
        sys.path.insert(0, str(phase3_path))
    
    # Test hierarchical optimizer
    try:
        from src.optimizers.hierarchical_balanced import HierarchicalBalanced
        result.add_pass("Phase 3: Import HierarchicalBalanced")
    except ImportError as e:
        result.add_fail("Phase 3: Import HierarchicalBalanced", str(e))
    
    # Test scale gamma manager
    try:
        from src.utils.scale_gamma_manager import ScaleGammaManager
        result.add_pass("Phase 3: Import ScaleGammaManager")
    except ImportError as e:
        result.add_fail("Phase 3: Import ScaleGammaManager", str(e))
    
    # Test hierarchical loss
    try:
        from src.utils.hierarchical_loss import HierarchicalLoss
        result.add_pass("Phase 3: Import HierarchicalLoss")
    except ImportError as e:
        result.add_fail("Phase 3: Import HierarchicalLoss", str(e))


def test_phase3_functionality(result: TestResult):
    """Test Phase 3: Hierarchical functionality"""
    try:
        # Add phase3 to path
        phase3_path = Path(__file__).parent / 'phase3'
        if str(phase3_path) not in sys.path:
            sys.path.insert(0, str(phase3_path))
        
        from src.utils.scale_gamma_manager import ScaleGammaManager
        
        # Create manager
        manager = ScaleGammaManager(
            scales=['s1', 's2', 's3'],
            strategy='fixed',
        )
        
        result.add_pass("Phase 3: Create ScaleGammaManager")
        
        # Test gamma retrieval
        gammas = manager.get_gammas(epoch=10)
        
        if 's1_distortion' in gammas and 's1_bpp' in gammas:
            result.add_pass("Phase 3: Get scale gammas works")
        else:
            result.add_fail("Phase 3: Get scale gammas", "Missing expected gamma keys")
        
    except Exception as e:
        result.add_fail("Phase 3: Functionality test", str(e))


def test_phase4_imports(result: TestResult):
    """Test Phase 4: Context-Aware Fine-tuning imports"""
    print(f"\n{BLUE}Testing Phase 4: Context-Aware Fine-tuning{RESET}")
    
    # Add phase4 to path
    phase4_path = Path(__file__).parent / 'phase4'
    if str(phase4_path) not in sys.path:
        sys.path.insert(0, str(phase4_path))
    
    # Test layer LR manager
    try:
        from src.utils.layer_lr_manager import LayerLRManager
        result.add_pass("Phase 4: Import LayerLRManager")
    except ImportError as e:
        result.add_fail("Phase 4: Import LayerLRManager", str(e))
    
    # Test scale early stopping
    try:
        from src.utils.scale_early_stopping import ScaleEarlyStopping
        result.add_pass("Phase 4: Import ScaleEarlyStopping")
    except ImportError as e:
        result.add_fail("Phase 4: Import ScaleEarlyStopping", str(e))


def test_phase4_functionality(result: TestResult):
    """Test Phase 4: Context-aware functionality"""
    try:
        # Add phase4 to path
        phase4_path = Path(__file__).parent / 'phase4'
        if str(phase4_path) not in sys.path:
            sys.path.insert(0, str(phase4_path))
        
        from src.utils.layer_lr_manager import LayerLRManager
        import torch.nn as nn
        
        # Create dummy model
        model = nn.Sequential(
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
        )
        
        # Create manager
        lr_mgr = LayerLRManager(
            model=model,
            base_lr=1e-4,
        )
        
        result.add_pass("Phase 4: Create LayerLRManager")
        
        # Test parameter groups
        param_groups = lr_mgr.get_parameter_groups()
        
        if len(param_groups) > 0:
            result.add_pass("Phase 4: Get parameter groups works")
        else:
            result.add_fail("Phase 4: Get parameter groups", "No parameter groups returned")
        
    except Exception as e:
        result.add_fail("Phase 4: Functionality test", str(e))


def test_phase5_imports(result: TestResult):
    """Test Phase 5: Comprehensive Evaluation imports"""
    print(f"\n{BLUE}Testing Phase 5: Comprehensive Evaluation{RESET}")
    
    # Add phase5 to path
    phase5_path = Path(__file__).parent / 'phase5'
    if str(phase5_path) not in sys.path:
        sys.path.insert(0, str(phase5_path))
    
    # Test BD-rate calculator
    try:
        from src.evaluation.bd_rate import BDRateCalculator, compute_bd_rate
        result.add_pass("Phase 5: Import BDRateCalculator")
    except ImportError as e:
        result.add_fail("Phase 5: Import BDRateCalculator", str(e))
    
    # Test RD curve
    try:
        from src.evaluation.rd_curve import RDCurve, RDCurvePlotter
        result.add_pass("Phase 5: Import RDCurve")
    except ImportError as e:
        result.add_fail("Phase 5: Import RDCurve", str(e))
    
    # Test metrics calculator
    try:
        from src.evaluation.metrics import MetricsCalculator
        result.add_pass("Phase 5: Import MetricsCalculator")
    except ImportError as e:
        result.add_fail("Phase 5: Import MetricsCalculator", str(e))
    
    # Test comparator
    try:
        from src.evaluation.comparator import ModelComparator
        result.add_pass("Phase 5: Import ModelComparator")
    except ImportError as e:
        result.add_fail("Phase 5: Import ModelComparator", str(e))


def test_phase5_functionality(result: TestResult):
    """Test Phase 5: Evaluation functionality"""
    try:
        # Add phase5 to path
        phase5_path = Path(__file__).parent / 'phase5'
        if str(phase5_path) not in sys.path:
            sys.path.insert(0, str(phase5_path))
        
        from src.evaluation.bd_rate import compute_bd_rate
        
        # Test BD-rate calculation
        rate1 = [0.2, 0.4, 0.6, 0.8]
        psnr1 = [32.0, 35.0, 37.0, 38.5]
        rate2 = [0.25, 0.5, 0.75, 1.0]
        psnr2 = [31.5, 34.5, 36.5, 38.0]
        
        bd_rate = compute_bd_rate(rate1, psnr1, rate2, psnr2)
        
        if not (isinstance(bd_rate, float) and -50 < bd_rate < 50):
            result.add_fail("Phase 5: BD-rate calculation", f"Invalid BD-rate value: {bd_rate}")
        else:
            result.add_pass("Phase 5: BD-rate calculation works")
        
    except Exception as e:
        result.add_fail("Phase 5: Functionality test", str(e))
    
    # Test RD curve
    try:
        # Add phase5 to path
        phase5_path = Path(__file__).parent / 'phase5'
        if str(phase5_path) not in sys.path:
            sys.path.insert(0, str(phase5_path))
        
        from src.evaluation.rd_curve import RDCurve
        
        curve = RDCurve(
            name='Test',
            rates=[0.2, 0.4, 0.6],
            psnrs=[32.0, 35.0, 37.0],
        )
        
        if len(curve) == 3:
            result.add_pass("Phase 5: RDCurve creation works")
        else:
            result.add_fail("Phase 5: RDCurve creation", f"Expected 3 points, got {len(curve)}")
        
    except Exception as e:
        result.add_fail("Phase 5: RDCurve test", str(e))


def test_integration(result: TestResult):
    """Test integration between phases"""
    print(f"\n{BLUE}Testing Phase Integration{RESET}")
    
    try:
        # Add phase1 and phase3 to path
        phase1_path = Path(__file__).parent / 'phase1'
        phase3_path = Path(__file__).parent / 'phase3'
        if str(phase1_path) not in sys.path:
            sys.path.insert(0, str(phase1_path))
        if str(phase3_path) not in sys.path:
            sys.path.insert(0, str(phase3_path))
        
        # Test that Phase 3 can use Phase 1 components
        from src.optimizers.balanced import Balanced
        from src.optimizers.hierarchical_balanced import HierarchicalBalanced
        
        result.add_pass("Integration: Phase 3 can access Phase 1 components")
        
    except ImportError as e:
        result.add_fail("Integration: Phase 3 → Phase 1", str(e))
    
    try:
        # Add phase3 and phase4 to path
        phase3_path = Path(__file__).parent / 'phase3'
        phase4_path = Path(__file__).parent / 'phase4'
        if str(phase3_path) not in sys.path:
            sys.path.insert(0, str(phase3_path))
        if str(phase4_path) not in sys.path:
            sys.path.insert(0, str(phase4_path))
        
        # Test that Phase 4 can use Phase 3 components
        from src.utils.scale_gamma_manager import ScaleGammaManager
        from src.utils.layer_lr_manager import LayerLRManager
        
        result.add_pass("Integration: Phase 4 can access Phase 3 components")
        
    except ImportError as e:
        result.add_fail("Integration: Phase 4 → Phase 3", str(e))


def main():
    print(f"{BLUE}{'='*80}{RESET}")
    print(f"{BLUE}COMPREHENSIVE TEST SUITE FOR PHASES 1-5{RESET}")
    print(f"{BLUE}{'='*80}{RESET}")
    
    result = TestResult()
    
    # Test file structure
    test_file_structure(result)
    
    # Test Phase 1
    test_phase1_imports(result)
    test_phase1_functionality(result)
    
    # Test Phase 2
    test_phase2_imports(result)
    test_phase2_functionality(result)
    
    # Test Phase 3
    test_phase3_imports(result)
    test_phase3_functionality(result)
    
    # Test Phase 4
    test_phase4_imports(result)
    test_phase4_functionality(result)
    
    # Test Phase 5
    test_phase5_imports(result)
    test_phase5_functionality(result)
    
    # Test integration
    test_integration(result)
    
    # Print summary
    success = result.summary()
    
    print(f"\n{BLUE}{'='*80}{RESET}")
    
    if success:
        print(f"{GREEN}All tests passed! ✓{RESET}")
        return 0
    else:
        print(f"{RED}Some tests failed. Please review the errors above.{RESET}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
