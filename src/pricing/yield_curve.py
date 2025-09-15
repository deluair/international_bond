"""
Yield curve construction and interpolation for bond pricing.
"""

import numpy as np
from datetime import date, datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from scipy import interpolate, optimize
from enum import Enum


class InterpolationMethod(Enum):
    """Yield curve interpolation methods."""
    LINEAR = "linear"
    CUBIC_SPLINE = "cubic_spline"
    NELSON_SIEGEL = "nelson_siegel"
    SVENSSON = "svensson"
    HERMITE = "hermite"


@dataclass
class YieldPoint:
    """Individual yield curve point."""
    maturity: float  # Years
    yield_rate: float  # Decimal (e.g., 0.05 for 5%)
    instrument_type: str = "bond"  # bond, bill, swap, etc.
    quote_date: Optional[date] = None
    
    def __post_init__(self):
        if self.quote_date is None:
            self.quote_date = date.today()


class YieldCurve:
    """
    Yield curve with multiple interpolation methods and analytics.
    """
    
    def __init__(self, currency: str, curve_date: date, 
                 points: List[YieldPoint] = None,
                 interpolation_method: InterpolationMethod = InterpolationMethod.CUBIC_SPLINE):
        """
        Initialize yield curve.
        
        Args:
            currency: Currency of the yield curve
            curve_date: Date of the curve
            points: List of yield points
            interpolation_method: Method for interpolation
        """
        self.currency = currency
        self.curve_date = curve_date
        self.points = points or []
        self.interpolation_method = interpolation_method
        
        # Interpolation objects
        self._interpolator = None
        self._nelson_siegel_params = None
        self._svensson_params = None
        
        if self.points:
            self._build_interpolator()
    
    def add_point(self, maturity: float, yield_rate: float, 
                  instrument_type: str = "bond"):
        """Add a yield point to the curve."""
        point = YieldPoint(
            maturity=maturity,
            yield_rate=yield_rate,
            instrument_type=instrument_type,
            quote_date=self.curve_date
        )
        self.points.append(point)
        self.points.sort(key=lambda x: x.maturity)
        self._build_interpolator()
    
    def _build_interpolator(self):
        """Build interpolation function based on selected method."""
        if len(self.points) < 2:
            return
        
        maturities = np.array([p.maturity for p in self.points])
        yields = np.array([p.yield_rate for p in self.points])
        
        # Remove duplicates
        unique_indices = np.unique(maturities, return_index=True)[1]
        maturities = maturities[unique_indices]
        yields = yields[unique_indices]
        
        if self.interpolation_method == InterpolationMethod.LINEAR:
            self._interpolator = interpolate.interp1d(
                maturities, yields, kind='linear',
                bounds_error=False, fill_value='extrapolate'
            )
        
        elif self.interpolation_method == InterpolationMethod.CUBIC_SPLINE:
            if len(maturities) >= 4:
                self._interpolator = interpolate.CubicSpline(
                    maturities, yields, bc_type='natural'
                )
            else:
                self._interpolator = interpolate.interp1d(
                    maturities, yields, kind='linear',
                    bounds_error=False, fill_value='extrapolate'
                )
        
        elif self.interpolation_method == InterpolationMethod.HERMITE:
            if len(maturities) >= 3:
                # Calculate derivatives for Hermite interpolation
                derivatives = np.gradient(yields, maturities)
                self._interpolator = interpolate.PchipInterpolator(
                    maturities, yields
                )
            else:
                self._interpolator = interpolate.interp1d(
                    maturities, yields, kind='linear',
                    bounds_error=False, fill_value='extrapolate'
                )
        
        elif self.interpolation_method == InterpolationMethod.NELSON_SIEGEL:
            self._fit_nelson_siegel(maturities, yields)
        
        elif self.interpolation_method == InterpolationMethod.SVENSSON:
            self._fit_svensson(maturities, yields)
    
    def _fit_nelson_siegel(self, maturities: np.ndarray, yields: np.ndarray):
        """Fit Nelson-Siegel model to yield curve."""
        def nelson_siegel(tau, beta0, beta1, beta2, lambda1):
            """Nelson-Siegel yield curve function."""
            term1 = beta1 * (1 - np.exp(-tau / lambda1)) / (tau / lambda1)
            term2 = beta2 * ((1 - np.exp(-tau / lambda1)) / (tau / lambda1) - np.exp(-tau / lambda1))
            return beta0 + term1 + term2
        
        def objective(params):
            beta0, beta1, beta2, lambda1 = params
            predicted = nelson_siegel(maturities, beta0, beta1, beta2, lambda1)
            return np.sum((yields - predicted) ** 2)
        
        # Initial parameter guess
        initial_guess = [yields[-1], yields[0] - yields[-1], 0.0, 2.0]
        
        # Bounds for parameters
        bounds = [(-0.1, 0.2), (-0.2, 0.2), (-0.2, 0.2), (0.1, 10.0)]
        
        try:
            result = optimize.minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
            if result.success:
                self._nelson_siegel_params = result.x
        except:
            # Fallback to simple parameters
            self._nelson_siegel_params = initial_guess
    
    def _fit_svensson(self, maturities: np.ndarray, yields: np.ndarray):
        """Fit Svensson model (extended Nelson-Siegel) to yield curve."""
        def svensson(tau, beta0, beta1, beta2, beta3, lambda1, lambda2):
            """Svensson yield curve function."""
            term1 = beta1 * (1 - np.exp(-tau / lambda1)) / (tau / lambda1)
            term2 = beta2 * ((1 - np.exp(-tau / lambda1)) / (tau / lambda1) - np.exp(-tau / lambda1))
            term3 = beta3 * ((1 - np.exp(-tau / lambda2)) / (tau / lambda2) - np.exp(-tau / lambda2))
            return beta0 + term1 + term2 + term3
        
        def objective(params):
            beta0, beta1, beta2, beta3, lambda1, lambda2 = params
            predicted = svensson(maturities, beta0, beta1, beta2, beta3, lambda1, lambda2)
            return np.sum((yields - predicted) ** 2)
        
        # Initial parameter guess
        initial_guess = [yields[-1], yields[0] - yields[-1], 0.0, 0.0, 2.0, 5.0]
        
        # Bounds for parameters
        bounds = [(-0.1, 0.2), (-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2), (0.1, 10.0), (0.1, 10.0)]
        
        try:
            result = optimize.minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
            if result.success:
                self._svensson_params = result.x
        except:
            # Fallback to Nelson-Siegel
            self._fit_nelson_siegel(maturities, yields)
    
    def get_yield(self, maturity: float) -> float:
        """Get interpolated yield for a given maturity."""
        if not self.points:
            return 0.0
        
        if len(self.points) == 1:
            return self.points[0].yield_rate
        
        # Handle edge cases
        min_maturity = min(p.maturity for p in self.points)
        max_maturity = max(p.maturity for p in self.points)
        
        if maturity <= min_maturity:
            return min(p.yield_rate for p in self.points if p.maturity == min_maturity)
        elif maturity >= max_maturity:
            return max(p.yield_rate for p in self.points if p.maturity == max_maturity)
        
        # Use appropriate interpolation method
        if self.interpolation_method == InterpolationMethod.NELSON_SIEGEL and self._nelson_siegel_params:
            beta0, beta1, beta2, lambda1 = self._nelson_siegel_params
            term1 = beta1 * (1 - np.exp(-maturity / lambda1)) / (maturity / lambda1)
            term2 = beta2 * ((1 - np.exp(-maturity / lambda1)) / (maturity / lambda1) - np.exp(-maturity / lambda1))
            return beta0 + term1 + term2
        
        elif self.interpolation_method == InterpolationMethod.SVENSSON and self._svensson_params:
            beta0, beta1, beta2, beta3, lambda1, lambda2 = self._svensson_params
            term1 = beta1 * (1 - np.exp(-maturity / lambda1)) / (maturity / lambda1)
            term2 = beta2 * ((1 - np.exp(-maturity / lambda1)) / (maturity / lambda1) - np.exp(-maturity / lambda1))
            term3 = beta3 * ((1 - np.exp(-maturity / lambda2)) / (maturity / lambda2) - np.exp(-maturity / lambda2))
            return beta0 + term1 + term2 + term3
        
        elif self._interpolator:
            return float(self._interpolator(maturity))
        
        else:
            # Fallback to linear interpolation
            maturities = [p.maturity for p in self.points]
            yields = [p.yield_rate for p in self.points]
            return float(np.interp(maturity, maturities, yields))
    
    def get_discount_factor(self, maturity: float) -> float:
        """Get discount factor for a given maturity."""
        yield_rate = self.get_yield(maturity)
        return np.exp(-yield_rate * maturity)
    
    def get_forward_rate(self, start_maturity: float, end_maturity: float) -> float:
        """Calculate forward rate between two maturities."""
        if start_maturity >= end_maturity:
            return self.get_yield(start_maturity)
        
        df_start = self.get_discount_factor(start_maturity)
        df_end = self.get_discount_factor(end_maturity)
        
        forward_rate = -np.log(df_end / df_start) / (end_maturity - start_maturity)
        return forward_rate
    
    def parallel_shift(self, shift: float) -> 'YieldCurve':
        """Create a new curve with parallel shift."""
        shifted_points = []
        for point in self.points:
            shifted_point = YieldPoint(
                maturity=point.maturity,
                yield_rate=point.yield_rate + shift,
                instrument_type=point.instrument_type,
                quote_date=point.quote_date
            )
            shifted_points.append(shifted_point)
        
        return YieldCurve(
            currency=self.currency,
            curve_date=self.curve_date,
            points=shifted_points,
            interpolation_method=self.interpolation_method
        )
    
    def steepen_flatten(self, short_shift: float, long_shift: float, 
                       pivot_maturity: float = 5.0) -> 'YieldCurve':
        """Create steepening/flattening scenario."""
        shifted_points = []
        for point in self.points:
            if point.maturity <= pivot_maturity:
                weight = point.maturity / pivot_maturity
                shift = short_shift + weight * (long_shift - short_shift)
            else:
                shift = long_shift
            
            shifted_point = YieldPoint(
                maturity=point.maturity,
                yield_rate=point.yield_rate + shift,
                instrument_type=point.instrument_type,
                quote_date=point.quote_date
            )
            shifted_points.append(shifted_point)
        
        return YieldCurve(
            currency=self.currency,
            curve_date=self.curve_date,
            points=shifted_points,
            interpolation_method=self.interpolation_method
        )
    
    def get_curve_points(self, num_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Get interpolated curve points for plotting."""
        if not self.points:
            return np.array([]), np.array([])
        
        min_maturity = min(p.maturity for p in self.points)
        max_maturity = max(p.maturity for p in self.points)
        
        maturities = np.linspace(min_maturity, max_maturity, num_points)
        yields = np.array([self.get_yield(m) for m in maturities])
        
        return maturities, yields
    
    @property
    def benchmark_yields(self) -> Dict[str, float]:
        """Get benchmark maturity yields."""
        benchmarks = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30]
        result = {}
        
        for maturity in benchmarks:
            if maturity <= max(p.maturity for p in self.points):
                yield_rate = self.get_yield(maturity)
                if maturity < 1:
                    key = f"{int(maturity * 12)}M"
                else:
                    key = f"{int(maturity)}Y"
                result[key] = yield_rate
        
        return result
    
    def calculate_duration(self, maturity: float, coupon_rate: float = 0.0) -> float:
        """Calculate modified duration for a bond on this curve."""
        yield_rate = self.get_yield(maturity)
        
        if coupon_rate == 0:  # Zero coupon bond
            return maturity / (1 + yield_rate)
        
        # Approximate duration for coupon bond
        # In practice, would calculate exact duration using cash flows
        macaulay_duration = (1 + yield_rate) / yield_rate - (1 + yield_rate + maturity * (coupon_rate - yield_rate)) / (coupon_rate * ((1 + yield_rate) ** maturity - 1) + yield_rate)
        modified_duration = macaulay_duration / (1 + yield_rate)
        
        return modified_duration
    
    def to_dict(self) -> Dict:
        """Convert yield curve to dictionary representation."""
        return {
            'currency': self.currency,
            'curve_date': self.curve_date.isoformat(),
            'interpolation_method': self.interpolation_method.value,
            'points': [
                {
                    'maturity': p.maturity,
                    'yield_rate': p.yield_rate,
                    'instrument_type': p.instrument_type,
                    'quote_date': p.quote_date.isoformat() if p.quote_date else None
                }
                for p in self.points
            ],
            'benchmark_yields': self.benchmark_yields
        }
    
    def __str__(self) -> str:
        """String representation of the yield curve."""
        return f"Yield Curve ({self.currency}) - {self.curve_date} - {len(self.points)} points"


class YieldCurveBuilder:
    """Builder class for constructing yield curves from market data."""
    
    @staticmethod
    def from_treasury_data(currency: str, curve_date: date, 
                          treasury_data: Dict[str, float]) -> YieldCurve:
        """Build yield curve from treasury data."""
        points = []
        
        for tenor, yield_rate in treasury_data.items():
            maturity = YieldCurveBuilder._parse_tenor(tenor)
            point = YieldPoint(
                maturity=maturity,
                yield_rate=yield_rate,
                instrument_type="treasury",
                quote_date=curve_date
            )
            points.append(point)
        
        return YieldCurve(currency, curve_date, points)
    
    @staticmethod
    def from_swap_data(currency: str, curve_date: date,
                      swap_data: Dict[str, float]) -> YieldCurve:
        """Build yield curve from swap data."""
        points = []
        
        for tenor, swap_rate in swap_data.items():
            maturity = YieldCurveBuilder._parse_tenor(tenor)
            point = YieldPoint(
                maturity=maturity,
                yield_rate=swap_rate,
                instrument_type="swap",
                quote_date=curve_date
            )
            points.append(point)
        
        return YieldCurve(currency, curve_date, points)
    
    @staticmethod
    def _parse_tenor(tenor: str) -> float:
        """Parse tenor string to years."""
        tenor = tenor.upper().strip()
        if tenor.endswith('Y'):
            return float(tenor[:-1])
        elif tenor.endswith('M'):
            return float(tenor[:-1]) / 12
        elif tenor.endswith('W'):
            return float(tenor[:-1]) / 52
        elif tenor.endswith('D'):
            return float(tenor[:-1]) / 365
        else:
            raise ValueError(f"Invalid tenor format: {tenor}")
    
    @staticmethod
    def bootstrap_curve(currency: str, curve_date: date,
                       instruments: List[Dict]) -> YieldCurve:
        """Bootstrap yield curve from various instruments."""
        # Simplified bootstrapping - in practice would be more sophisticated
        points = []
        
        # Sort instruments by maturity
        instruments.sort(key=lambda x: YieldCurveBuilder._parse_tenor(x['tenor']))
        
        for instrument in instruments:
            maturity = YieldCurveBuilder._parse_tenor(instrument['tenor'])
            
            if instrument['type'] == 'deposit' or instrument['type'] == 'bill':
                # Simple rate
                yield_rate = instrument['rate']
            elif instrument['type'] == 'swap':
                # For swaps, would need to bootstrap against existing curve
                yield_rate = instrument['rate']
            else:
                yield_rate = instrument['rate']
            
            point = YieldPoint(
                maturity=maturity,
                yield_rate=yield_rate,
                instrument_type=instrument['type'],
                quote_date=curve_date
            )
            points.append(point)
        
        return YieldCurve(currency, curve_date, points)