"""
Credit Default Swap (CDS) analytics and curve construction.

This module provides comprehensive CDS curve construction, interpolation,
and credit risk analytics for sovereign and corporate entities.
"""

from .cds_curve_builder import CDSCurveBuilder, CDSBootstrapper
from .credit_analytics import CreditAnalytics, CreditRiskMetrics
from .hazard_rate_model import HazardRateModel, SurvivalProbability
from .cds_pricer import CDSPricer, CDSValuation

__all__ = [
    'CDSCurveBuilder',
    'CDSBootstrapper', 
    'CreditAnalytics',
    'CreditRiskMetrics',
    'HazardRateModel',
    'SurvivalProbability',
    'CDSPricer',
    'CDSValuation'
]

__version__ = "1.0.0"