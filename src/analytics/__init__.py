"""
Analytics package for risk-adjusted comparison framework and relative value indicators.

This package provides comprehensive analytics tools for:
- Risk-adjusted performance comparison
- Relative value analysis
- Cross-market bond comparison
- Risk attribution and decomposition
"""

from .relative_value_analyzer import (
    RelativeValueAnalyzer,
    RelativeValueMetric,
    RiskAdjustmentMethod,
    RelativeValueInput,
    SpreadAnalysis,
    ComparisonResult
)

from .risk_adjusted_comparator import (
    RiskAdjustedComparator,
    RiskAdjustmentMethod as RiskMethod,
    PerformanceMetric,
    RiskAdjustedMetrics,
    PerformanceAttribution
)

from .cross_market_analyzer import (
    CrossMarketAnalyzer,
    ArbitrageOpportunity,
    MarketRegion,
    CrossMarketMetrics,
    MarketCorrelation
)

from .relative_value_calculator import (
    RelativeValueCalculator,
    SpreadType,
    SpreadResult
)

from .yield_curve_analyzer import (
    YieldCurveAnalyzer,
    CurveType,
    InterpolationMethod,
    YieldCurveMetrics,
    FittedCurve
)

__all__ = [
    # Relative Value Analysis
    'RelativeValueAnalyzer',
    'RelativeValueMetric',
    'RiskAdjustmentMethod',
    'RelativeValueInput',
    'SpreadAnalysis',
    'ComparisonResult',
    
    # Risk Adjusted Comparison
    'RiskAdjustedComparator',
    'RiskMethod',
    'PerformanceMetric',
    'RiskAdjustedMetrics',
    'PerformanceAttribution',
    
    # Cross Market Analysis
    'CrossMarketAnalyzer',
    'ArbitrageOpportunity',
    'MarketRegion',
    'CrossMarketMetrics',
    'MarketCorrelation',
    
    # Relative Value Calculator
    'RelativeValueCalculator',
    'SpreadType',
    'SpreadResult',
    
    # Yield Curve Analysis
    'YieldCurveAnalyzer',
    'CurveType',
    'InterpolationMethod',
    'YieldCurveMetrics',
    'FittedCurve'
]