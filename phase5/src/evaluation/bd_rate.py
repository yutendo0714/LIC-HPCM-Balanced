"""
BD-Rate Calculation Module

Implements Bjøntegaard Delta metrics for rate-distortion comparison.
Reference: G. Bjontegaard, "Calculation of average PSNR differences between RD-curves", 
           VCEG-M33, 2001.
"""

import numpy as np
from scipy import interpolate
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class BDRateCalculator:
    """
    Calculate Bjøntegaard Delta Rate (BD-Rate) and BD-PSNR.
    
    BD-Rate: Average percentage bitrate difference at same quality
    BD-PSNR: Average quality difference at same bitrate
    """
    
    def __init__(
        self,
        interpolation_type: str = 'cubic',
        integration_samples: int = 1000,
    ):
        """
        Args:
            interpolation_type: Interpolation method ('cubic', 'pchip', 'akima')
            integration_samples: Number of samples for numerical integration
        """
        self.interpolation_type = interpolation_type
        self.integration_samples = integration_samples
    
    def bd_rate(
        self,
        rate1: List[float],
        psnr1: List[float],
        rate2: List[float],
        psnr2: List[float],
    ) -> float:
        """
        Calculate BD-Rate between two RD curves.
        
        Args:
            rate1: Bitrates of curve 1 (bpp)
            psnr1: PSNRs of curve 1 (dB)
            rate2: Bitrates of curve 2 (reference)
            psnr2: PSNRs of curve 2 (reference)
        
        Returns:
            BD-Rate in percentage. Negative means curve 1 is better.
        
        Example:
            >>> calc = BDRateCalculator()
            >>> bd_rate = calc.bd_rate([0.2, 0.4, 0.6], [32, 35, 38],
            ...                         [0.25, 0.5, 0.75], [31, 34, 37])
            >>> print(f"BD-Rate: {bd_rate:.2f}%")
        """
        # Convert to numpy arrays
        rate1 = np.array(rate1)
        psnr1 = np.array(psnr1)
        rate2 = np.array(rate2)
        psnr2 = np.array(psnr2)
        
        # Sort by PSNR
        idx1 = np.argsort(psnr1)
        psnr1 = psnr1[idx1]
        rate1 = rate1[idx1]
        
        idx2 = np.argsort(psnr2)
        psnr2 = psnr2[idx2]
        rate2 = rate2[idx2]
        
        # Find common PSNR range
        min_psnr = max(psnr1.min(), psnr2.min())
        max_psnr = min(psnr1.max(), psnr2.max())
        
        if min_psnr >= max_psnr:
            logger.warning("No overlapping PSNR range. Cannot compute BD-Rate.")
            return float('nan')
        
        # Interpolate in log domain for rate
        log_rate1 = np.log(rate1)
        log_rate2 = np.log(rate2)
        
        # Create interpolation functions
        try:
            if self.interpolation_type == 'cubic':
                interp1 = interpolate.CubicSpline(psnr1, log_rate1)
                interp2 = interpolate.CubicSpline(psnr2, log_rate2)
            elif self.interpolation_type == 'pchip':
                interp1 = interpolate.PchipInterpolator(psnr1, log_rate1)
                interp2 = interpolate.PchipInterpolator(psnr2, log_rate2)
            elif self.interpolation_type == 'akima':
                interp1 = interpolate.Akima1DInterpolator(psnr1, log_rate1)
                interp2 = interpolate.Akima1DInterpolator(psnr2, log_rate2)
            else:
                raise ValueError(f"Unknown interpolation type: {self.interpolation_type}")
        except Exception as e:
            logger.error(f"Interpolation failed: {e}")
            return float('nan')
        
        # Integrate over common PSNR range
        psnr_samples = np.linspace(min_psnr, max_psnr, self.integration_samples)
        
        log_rate1_interp = interp1(psnr_samples)
        log_rate2_interp = interp2(psnr_samples)
        
        # BD-Rate = (integral(log_rate1) - integral(log_rate2)) / (max_psnr - min_psnr)
        avg_log_rate1 = np.trapz(log_rate1_interp, psnr_samples) / (max_psnr - min_psnr)
        avg_log_rate2 = np.trapz(log_rate2_interp, psnr_samples) / (max_psnr - min_psnr)
        
        # Convert to percentage
        bd_rate_percent = 100 * (np.exp(avg_log_rate1 - avg_log_rate2) - 1)
        
        return bd_rate_percent
    
    def bd_psnr(
        self,
        rate1: List[float],
        psnr1: List[float],
        rate2: List[float],
        psnr2: List[float],
    ) -> float:
        """
        Calculate BD-PSNR between two RD curves.
        
        Args:
            rate1: Bitrates of curve 1 (bpp)
            psnr1: PSNRs of curve 1 (dB)
            rate2: Bitrates of curve 2 (reference)
            psnr2: PSNRs of curve 2 (reference)
        
        Returns:
            BD-PSNR in dB. Positive means curve 1 is better.
        """
        # Convert to numpy arrays
        rate1 = np.array(rate1)
        psnr1 = np.array(psnr1)
        rate2 = np.array(rate2)
        psnr2 = np.array(psnr2)
        
        # Sort by rate
        idx1 = np.argsort(rate1)
        rate1 = rate1[idx1]
        psnr1 = psnr1[idx1]
        
        idx2 = np.argsort(rate2)
        rate2 = rate2[idx2]
        psnr2 = psnr2[idx2]
        
        # Find common rate range
        min_rate = max(rate1.min(), rate2.min())
        max_rate = min(rate1.max(), rate2.max())
        
        if min_rate >= max_rate:
            logger.warning("No overlapping rate range. Cannot compute BD-PSNR.")
            return float('nan')
        
        # Interpolate in log domain for rate
        log_rate1 = np.log(rate1)
        log_rate2 = np.log(rate2)
        
        # Create interpolation functions
        try:
            if self.interpolation_type == 'cubic':
                interp1 = interpolate.CubicSpline(log_rate1, psnr1)
                interp2 = interpolate.CubicSpline(log_rate2, psnr2)
            elif self.interpolation_type == 'pchip':
                interp1 = interpolate.PchipInterpolator(log_rate1, psnr1)
                interp2 = interpolate.PchipInterpolator(log_rate2, psnr2)
            elif self.interpolation_type == 'akima':
                interp1 = interpolate.Akima1DInterpolator(log_rate1, psnr1)
                interp2 = interpolate.Akima1DInterpolator(log_rate2, psnr2)
            else:
                raise ValueError(f"Unknown interpolation type: {self.interpolation_type}")
        except Exception as e:
            logger.error(f"Interpolation failed: {e}")
            return float('nan')
        
        # Integrate over common log-rate range
        log_rate_samples = np.linspace(np.log(min_rate), np.log(max_rate), 
                                       self.integration_samples)
        
        psnr1_interp = interp1(log_rate_samples)
        psnr2_interp = interp2(log_rate_samples)
        
        # BD-PSNR = (integral(psnr1) - integral(psnr2)) / (log(max_rate) - log(min_rate))
        avg_psnr1 = np.trapz(psnr1_interp, log_rate_samples) / (np.log(max_rate) - np.log(min_rate))
        avg_psnr2 = np.trapz(psnr2_interp, log_rate_samples) / (np.log(max_rate) - np.log(min_rate))
        
        bd_psnr_db = avg_psnr1 - avg_psnr2
        
        return bd_psnr_db
    
    def compute_multiple(
        self,
        test_curves: Dict[str, Tuple[List[float], List[float]]],
        reference_curve: Tuple[List[float], List[float]],
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute BD metrics for multiple test curves against a reference.
        
        Args:
            test_curves: Dict of {name: (rates, psnrs)}
            reference_curve: (rates, psnrs) of reference
        
        Returns:
            Dict of {name: {'bd_rate': float, 'bd_psnr': float}}
        
        Example:
            >>> calc = BDRateCalculator()
            >>> test_curves = {
            ...     'Method A': ([0.2, 0.4, 0.6], [32, 35, 38]),
            ...     'Method B': ([0.25, 0.5, 0.75], [31, 34, 37]),
            ... }
            >>> reference = ([0.3, 0.6, 0.9], [30, 33, 36])
            >>> results = calc.compute_multiple(test_curves, reference)
        """
        results = {}
        ref_rate, ref_psnr = reference_curve
        
        for name, (test_rate, test_psnr) in test_curves.items():
            bd_rate = self.bd_rate(test_rate, test_psnr, ref_rate, ref_psnr)
            bd_psnr = self.bd_psnr(test_rate, test_psnr, ref_rate, ref_psnr)
            
            results[name] = {
                'bd_rate': bd_rate,
                'bd_psnr': bd_psnr,
            }
            
            logger.info(f"{name} vs Reference: BD-Rate={bd_rate:.2f}%, BD-PSNR={bd_psnr:.3f}dB")
        
        return results


def compute_bd_rate(
    rate1: List[float],
    psnr1: List[float],
    rate2: List[float],
    psnr2: List[float],
    interpolation_type: str = 'cubic',
) -> float:
    """
    Convenience function to compute BD-Rate.
    
    Args:
        rate1: Bitrates of curve 1 (bpp)
        psnr1: PSNRs of curve 1 (dB)
        rate2: Bitrates of curve 2 (reference)
        psnr2: PSNRs of curve 2 (reference)
        interpolation_type: Interpolation method
    
    Returns:
        BD-Rate in percentage
    """
    calc = BDRateCalculator(interpolation_type=interpolation_type)
    return calc.bd_rate(rate1, psnr1, rate2, psnr2)


def compute_bd_psnr(
    rate1: List[float],
    psnr1: List[float],
    rate2: List[float],
    psnr2: List[float],
    interpolation_type: str = 'cubic',
) -> float:
    """
    Convenience function to compute BD-PSNR.
    
    Args:
        rate1: Bitrates of curve 1 (bpp)
        psnr1: PSNRs of curve 1 (dB)
        rate2: Bitrates of curve 2 (reference)
        psnr2: PSNRs of curve 2 (reference)
        interpolation_type: Interpolation method
    
    Returns:
        BD-PSNR in dB
    """
    calc = BDRateCalculator(interpolation_type=interpolation_type)
    return calc.bd_psnr(rate1, psnr1, rate2, psnr2)


def validate_rd_curve(
    rate: List[float],
    psnr: List[float],
) -> Tuple[bool, str]:
    """
    Validate RD curve data.
    
    Args:
        rate: Bitrates
        psnr: PSNRs
    
    Returns:
        (is_valid, error_message)
    """
    if len(rate) != len(psnr):
        return False, "Rate and PSNR arrays must have same length"
    
    if len(rate) < 3:
        return False, "Need at least 3 points for interpolation"
    
    if any(r <= 0 for r in rate):
        return False, "All rates must be positive"
    
    # Check monotonicity
    rate_arr = np.array(rate)
    psnr_arr = np.array(psnr)
    
    idx = np.argsort(rate_arr)
    rate_sorted = rate_arr[idx]
    psnr_sorted = psnr_arr[idx]
    
    if not np.all(np.diff(psnr_sorted) > 0):
        return False, "PSNR must increase with rate (non-monotonic curve)"
    
    return True, ""


if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Example RD curves
    method1_rate = [0.2, 0.4, 0.6, 0.8]
    method1_psnr = [32.5, 35.2, 37.1, 38.5]
    
    reference_rate = [0.25, 0.5, 0.75, 1.0]
    reference_psnr = [31.8, 34.5, 36.3, 37.8]
    
    # Validate curves
    valid1, msg1 = validate_rd_curve(method1_rate, method1_psnr)
    valid2, msg2 = validate_rd_curve(reference_rate, reference_psnr)
    
    print(f"Method 1 valid: {valid1}")
    print(f"Reference valid: {valid2}")
    
    if valid1 and valid2:
        # Calculate BD metrics
        calc = BDRateCalculator()
        
        bd_rate = calc.bd_rate(method1_rate, method1_psnr, 
                               reference_rate, reference_psnr)
        bd_psnr = calc.bd_psnr(method1_rate, method1_psnr,
                               reference_rate, reference_psnr)
        
        print(f"\nResults:")
        print(f"BD-Rate: {bd_rate:.2f}%")
        print(f"BD-PSNR: {bd_psnr:.3f} dB")
        
        if bd_rate < 0:
            print(f"Method 1 is {abs(bd_rate):.2f}% more efficient than reference")
        else:
            print(f"Method 1 is {bd_rate:.2f}% less efficient than reference")
