"""Currency management and FX hedging module.

This module provides comprehensive currency management capabilities including:
- FX hedge calculation and optimization
- Currency overlay strategies
- FX risk management and monitoring
"""

from .fx_hedge_calculator import (
    FXHedgeCalculator,
    HedgeRatio,
    HedgeEffectiveness,
    FXRiskMetrics
)

from .currency_overlay import (
    CurrencyOverlay,
    OverlayStrategy,
    OverlayPosition,
    OverlayPerformance
)

from .fx_risk_manager import (
    FXRiskManager,
    RiskMetric,
    RiskAlert
)

__all__ = [
    'FXHedgeCalculator',
    'HedgeRatio',
    'HedgeEffectiveness', 
    'FXRiskMetrics',
    'CurrencyOverlay',
    'OverlayStrategy',
    'OverlayPosition',
    'OverlayPerformance',
    'FXRiskManager',
    'RiskMetric',
    'RiskAlert'
]