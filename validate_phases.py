#!/usr/bin/env python3
"""
Simple structure validation for all phases (1-5)
Tests only file existence and basic Python syntax, no torch required.
"""

import sys
from pathlib import Path

# Color codes
GREEN = '\033[92m'
RED = '\033[91m'
BLUE = '\033[94m'
RESET = '\033[0m'


def check_files():
    """Check that all expected files exist"""
    print(f"\n{BLUE}Checking File Structure{RESET}\n")
    
    base = Path(__file__).parent
    
    files = {
        'Phase 1': [
            'phase1/src/optimizers/balanced.py',
            'phase1/train.py',
            'phase1/README.md',
            'phase1/COMPLETION_REPORT.md',
        ],
        'Phase 2': [
            'phase2/src/utils/adaptive_gamma.py',
            'phase2/src/utils/checkpoint_manager.py',
            'phase2/src/utils/hparam_analyzer.py',
            'phase2/train.py',
            'phase2/README.md',
            'phase2/COMPLETION_REPORT.md',
        ],
        'Phase 3': [
            'phase3/src/optimizers/hierarchical_balanced.py',
            'phase3/src/utils/scale_gamma_manager.py',
            'phase3/src/utils/hierarchical_loss.py',
            'phase3/train.py',
            'phase3/README.md',
            'phase3/COMPLETION_REPORT.md',
        ],
        'Phase 4': [
            'phase4/src/utils/layer_lr_manager.py',
            'phase4/src/utils/scale_early_stopping.py',
            'phase4/train.py',
            'phase4/README.md',
            'phase4/COMPLETION_REPORT.md',
        ],
        'Phase 5': [
            'phase5/src/evaluation/bd_rate.py',
            'phase5/src/evaluation/rd_curve.py',
            'phase5/src/evaluation/metrics.py',
            'phase5/src/evaluation/comparator.py',
            'phase5/evaluate.py',
            'phase5/compare_sota.py',
            'phase5/README.md',
            'phase5/COMPLETION_REPORT.md',
        ]
    }
    
    total = 0
    passed = 0
    
    for phase, file_list in files.items():
        print(f"{BLUE}{phase}:{RESET}")
        for file_path in file_list:
            full_path = base / file_path
            if full_path.exists():
                print(f"  {GREEN}✓{RESET} {file_path}")
                passed += 1
            else:
                print(f"  {RED}✗{RESET} {file_path} (NOT FOUND)")
            total += 1
        print()
    
    return passed, total


def check_syntax():
    """Check Python syntax of key files"""
    print(f"\n{BLUE}Checking Python Syntax{RESET}\n")
    
    base = Path(__file__).parent
    
    files = [
        'phase1/src/optimizers/balanced.py',
        'phase2/src/utils/adaptive_gamma.py',
        'phase3/src/optimizers/hierarchical_balanced.py',
        'phase4/src/utils/layer_lr_manager.py',
        'phase5/src/evaluation/bd_rate.py',
        'phase5/src/evaluation/rd_curve.py',
    ]
    
    total = 0
    passed = 0
    
    for file_path in files:
        full_path = base / file_path
        if not full_path.exists():
            print(f"  {RED}✗{RESET} {file_path} (NOT FOUND)")
            total += 1
            continue
        
        try:
            with open(full_path, 'r') as f:
                code = f.read()
            compile(code, str(full_path), 'exec')
            print(f"  {GREEN}✓{RESET} {file_path}")
            passed += 1
        except SyntaxError as e:
            print(f"  {RED}✗{RESET} {file_path} (SYNTAX ERROR: {e})")
        
        total += 1
    
    return passed, total


def check_documentation():
    """Check that documentation files are present and non-empty"""
    print(f"\n{BLUE}Checking Documentation{RESET}\n")
    
    base = Path(__file__).parent
    
    docs = [
        'phase1/README.md',
        'phase1/COMPLETION_REPORT.md',
        'phase2/README.md',
        'phase2/COMPLETION_REPORT.md',
        'phase3/README.md',
        'phase3/COMPLETION_REPORT.md',
        'phase4/README.md',
        'phase4/COMPLETION_REPORT.md',
        'phase5/README.md',
        'phase5/COMPLETION_REPORT.md',
    ]
    
    total = 0
    passed = 0
    
    for doc_path in docs:
        full_path = base / doc_path
        if not full_path.exists():
            print(f"  {RED}✗{RESET} {doc_path} (NOT FOUND)")
            total += 1
            continue
        
        size = full_path.stat().st_size
        if size > 1000:  # At least 1KB
            print(f"  {GREEN}✓{RESET} {doc_path} ({size:,} bytes)")
            passed += 1
        else:
            print(f"  {YELLOW}⚠{RESET} {doc_path} ({size} bytes - might be incomplete)")
            passed += 0.5
        
        total += 1
    
    return int(passed), total


def main():
    print(f"{BLUE}{'='*80}{RESET}")
    print(f"{BLUE}STRUCTURE VALIDATION FOR PHASES 1-5{RESET}")
    print(f"{BLUE}{'='*80}{RESET}")
    
    # Check files
    files_passed, files_total = check_files()
    
    # Check syntax
    syntax_passed, syntax_total = check_syntax()
    
    # Check documentation
    docs_passed, docs_total = check_documentation()
    
    # Summary
    total_passed = files_passed + syntax_passed + docs_passed
    total_tests = files_total + syntax_total + docs_total
    
    print(f"\n{BLUE}{'='*80}{RESET}")
    print(f"{BLUE}SUMMARY{RESET}")
    print(f"{BLUE}{'='*80}{RESET}")
    print(f"File Structure: {files_passed}/{files_total}")
    print(f"Python Syntax:  {syntax_passed}/{syntax_total}")
    print(f"Documentation:  {docs_passed}/{docs_total}")
    print(f"\nTotal: {total_passed}/{total_tests} ({total_passed/total_tests*100:.1f}%)")
    
    if total_passed == total_tests:
        print(f"\n{GREEN}✓ All checks passed!{RESET}")
        return 0
    else:
        print(f"\n{RED}✗ Some checks failed.{RESET}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
