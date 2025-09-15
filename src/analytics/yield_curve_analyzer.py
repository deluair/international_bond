"""
Yield Curve Analyzer Module

This module provides comprehensive yield curve analysis capabilities including
curve fitting, metrics calculation, and shape analysis.
"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from datetime import datetime, date
import numpy as np
from scipy import interpolate


class CurveType(Enum):
    """Types of yield curves."""
    GOVERNMENT = "government"
    CORPORATE = "corporate"
    SWAP = "swap"
    MUNICIPAL = "municipal"


class InterpolationMethod(Enum):
    """Interpolation methods for yield curves."""
    LINEAR = "linear"
    CUBIC_SPLINE = "cubic_spline"
    NELSON_SIEGEL = "nelson_siegel"
    SVENSSON = "svensson"


@dataclass
class YieldCurveMetrics:
    """Metrics for yield curve analysis."""
    curve_id: str
    level: float
    slope: float
    curvature: float
    steepness_2y10y: float
    steepness_5y30y: float


@dataclass
class CurveShapeIndicators:
    """Indicators for curve shape analysis."""
    curve_id: str
    shape_classification: str
    inversion_points: List[float]
    hump_locations: List[float]


@dataclass
class CurveRiskMetrics:
    """Risk metrics for yield curves."""
    curve_id: str
    duration: float
    convexity: float
    var_95: float
    volatility: float


@dataclass
class FittedCurve:
    """Fitted yield curve object."""
    curve_id: str
    curve_type: CurveType
    maturities: List[float]
    yields: List[float]
    interpolation_method: InterpolationMethod
    curve_date: datetime


class YieldCurveAnalyzer:
    """Analyzer for yield curve operations and metrics."""
    
    def __init__(self, 
                 interpolation_method: InterpolationMethod = InterpolationMethod.CUBIC_SPLINE,
                 curve_smoothing: float = 0.1,
                 risk_free_rate: float = 0.02):
        """Initialize the yield curve analyzer."""
        self.interpolation_method = interpolation_method
        self.curve_smoothing = curve_smoothing
        self.risk_free_rate = risk_free_rate
    
    def fit_yield_curve(self, 
                       maturities: List[float], 
                       yields: List[float],
                       curve_date: datetime = None,
                       curve_type: CurveType = CurveType.GOVERNMENT,
                       country: str = "US") -> FittedCurve:
        """Fit a yield curve to the given data points."""
        if len(maturities) < 3:
            raise ValueError("At least 3 data points required for curve fitting")
        
        if len(maturities) != len(yields):
            raise ValueError("Maturities and yields must have same length")
        
        if any(np.isnan(yields)) or any(np.isinf(yields)):
            raise ValueError("Yields contain invalid values (NaN or Inf)")
        
        curve_id = f"{country}_{curve_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return FittedCurve(
            curve_id=curve_id,
            curve_type=curve_type,
            maturities=maturities,
            yields=yields,
            interpolation_method=self.interpolation_method,
            curve_date=curve_date or datetime.now()
        )
    
    def calculate_curve_metrics(self, fitted_curve: FittedCurve) -> YieldCurveMetrics:
        """Calculate comprehensive metrics for a fitted curve."""
        yields = np.array(fitted_curve.yields)
        maturities = np.array(fitted_curve.maturities)
        
        # Basic metrics
        level = np.mean(yields)
        slope = (yields[-1] - yields[0]) / (maturities[-1] - maturities[0])
        
        # Curvature (second derivative approximation)
        if len(yields) >= 3:
            curvature = np.mean(np.diff(yields, 2))
        else:
            curvature = 0.0
        
        # Steepness calculations
        steepness_2y10y = self._get_yield_at_maturity(fitted_curve, 10.0) - self._get_yield_at_maturity(fitted_curve, 2.0)
        steepness_5y30y = self._get_yield_at_maturity(fitted_curve, 30.0) - self._get_yield_at_maturity(fitted_curve, 5.0)
        
        return YieldCurveMetrics(
            curve_id=fitted_curve.curve_id,
            level=level,
            slope=slope,
            curvature=curvature,
            steepness_2y10y=steepness_2y10y,
            steepness_5y30y=steepness_5y30y
        )
    
    def analyze_curve_shape(self, fitted_curve: FittedCurve) -> CurveShapeIndicators:
        """Analyze the shape characteristics of a yield curve."""
        yields = np.array(fitted_curve.yields)
        maturities = np.array(fitted_curve.maturities)
        
        # Simple shape classification
        if yields[-1] > yields[0]:
            shape_classification = "normal"
        elif yields[-1] < yields[0]:
            shape_classification = "inverted"
        else:
            shape_classification = "flat"
        
        # Find inversion points (simplified)
        inversion_points = []
        for i in range(1, len(yields)):
            if yields[i] < yields[i-1]:
                inversion_points.append(maturities[i])
        
        # Find hump locations (simplified)
        hump_locations = []
        if len(yields) >= 3:
            for i in range(1, len(yields)-1):
                if yields[i] > yields[i-1] and yields[i] > yields[i+1]:
                    hump_locations.append(maturities[i])
        
        return CurveShapeIndicators(
            curve_id=fitted_curve.curve_id,
            shape_classification=shape_classification,
            inversion_points=inversion_points,
            hump_locations=hump_locations
        )
    
    def calculate_curve_risk_metrics(self, fitted_curve: FittedCurve) -> CurveRiskMetrics:
        """Calculate risk metrics for a yield curve."""
        yields = np.array(fitted_curve.yields)
        maturities = np.array(fitted_curve.maturities)
        
        # Simplified duration calculation (modified duration approximation)
        duration = np.average(maturities, weights=yields)
        
        # Simplified convexity
        convexity = np.average(maturities**2, weights=yields) / 2
        
        # Simplified VaR (95% confidence)
        volatility = np.std(yields) if len(yields) > 1 else 0.01
        var_95 = 1.645 * volatility  # 95% VaR
        
        return CurveRiskMetrics(
            curve_id=fitted_curve.curve_id,
            duration=duration,
            convexity=convexity,
            var_95=var_95,
            volatility=volatility
        )
    
    def calculate_zero_rates(self, fitted_curve: FittedCurve, maturities: List[float]) -> Dict[float, float]:
        """Calculate zero rates for given maturities."""
        zero_rates = {}
        
        for maturity in maturities:
            zero_rate = self._get_yield_at_maturity(fitted_curve, maturity)
            zero_rates[maturity] = zero_rate
        
        return zero_rates
    
    def compare_curves(self, curve1: FittedCurve, curve2: FittedCurve) -> Dict:
        """Compare two yield curves."""
        metrics1 = self.calculate_curve_metrics(curve1)
        metrics2 = self.calculate_curve_metrics(curve2)
        
        spread_metrics = {
            'level_spread': metrics1.level - metrics2.level,
            'slope_spread': metrics1.slope - metrics2.slope,
            'curvature_spread': metrics1.curvature - metrics2.curvature
        }
        
        shape_differences = {
            'steepness_2y10y_diff': metrics1.steepness_2y10y - metrics2.steepness_2y10y,
            'steepness_5y30y_diff': metrics1.steepness_5y30y - metrics2.steepness_5y30y
        }
        
        risk_differences = {
            'duration_diff': 0.0,  # Simplified
            'convexity_diff': 0.0  # Simplified
        }
        
        return {
            'spread_metrics': spread_metrics,
            'shape_differences': shape_differences,
            'risk_differences': risk_differences
        }
    
    def analyze_historical_curves(self, historical_curves: List[FittedCurve]) -> Dict:
        """Analyze historical evolution of yield curves."""
        if not historical_curves:
            return {}
        
        # Simplified historical analysis
        levels = [self.calculate_curve_metrics(curve).level for curve in historical_curves]
        slopes = [self.calculate_curve_metrics(curve).slope for curve in historical_curves]
        
        trend_analysis = {
            'level_trend': np.polyfit(range(len(levels)), levels, 1)[0] if len(levels) > 1 else 0.0,
            'slope_trend': np.polyfit(range(len(slopes)), slopes, 1)[0] if len(slopes) > 1 else 0.0
        }
        
        volatility_analysis = {
            'level_volatility': np.std(levels) if len(levels) > 1 else 0.0,
            'slope_volatility': np.std(slopes) if len(slopes) > 1 else 0.0
        }
        
        correlation_analysis = {
            'level_slope_correlation': np.corrcoef(levels, slopes)[0, 1] if len(levels) > 1 else 0.0
        }
        
        return {
            'trend_analysis': trend_analysis,
            'volatility_analysis': volatility_analysis,
            'correlation_analysis': correlation_analysis
        }
    
    def _get_yield_at_maturity(self, fitted_curve: FittedCurve, target_maturity: float) -> float:
        """Get interpolated yield at a specific maturity."""
        maturities = np.array(fitted_curve.maturities)
        yields = np.array(fitted_curve.yields)
        
        if target_maturity in maturities:
            idx = np.where(maturities == target_maturity)[0][0]
            return yields[idx]
        
        # Simple linear interpolation
        return np.interp(target_maturity, maturities, yields)