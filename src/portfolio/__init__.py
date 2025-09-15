"""Portfolio optimization and management module.

This module provides comprehensive portfolio optimization capabilities including:
- Duration-neutral optimization
- Currency-hedged strategies  
- Risk parity optimization
"""

from .duration_neutral_optimizer import (
    DurationNeutralOptimizer,
    OptimizationObjective,
    PortfolioConstraints,
    OptimizationResult
)

from .currency_hedged_strategy import (
    CurrencyHedgedStrategy,
    HedgeStrategy,
    HedgeConstraints,
    StrategyPerformance
)

from .risk_parity_optimizer import (
    RiskParityOptimizer,
    RiskParityMethod,
    RiskParityConstraints
)

__all__ = [
    'DurationNeutralOptimizer',
    'OptimizationObjective', 
    'PortfolioConstraints',
    'OptimizationResult',
    'CurrencyHedgedStrategy',
    'HedgeStrategy',
    'HedgeConstraints', 
    'StrategyPerformance',
    'RiskParityOptimizer',
    'RiskParityMethod',
    'RiskParityConstraints'
]