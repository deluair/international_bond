"""
Relative Value Calculator Module

This module provides simplified relative value calculation capabilities
for bond comparison and analysis.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from datetime import datetime, date
from dataclasses import dataclass
from enum import Enum

from ..models.bond import SovereignBond
from ..pricing.yield_curve import YieldCurve


class SpreadType(Enum):
    """Types of spread calculations."""
    YIELD_SPREAD = "yield_spread"
    Z_SPREAD = "z_spread"
    CREDIT_SPREAD = "credit_spread"
    DURATION_ADJUSTED_SPREAD = "duration_adjusted_spread"


@dataclass
class SpreadResult:
    """Result of spread calculation."""
    spread_type: SpreadType
    spread_value: float  # in basis points
    target_bond_id: str
    benchmark_bond_id: str
    calculation_date: datetime


class RelativeValueCalculator:
    """Calculator for relative value metrics between bonds."""
    
    def __init__(self):
        """Initialize the relative value calculator."""
        self.calculation_history: List[SpreadResult] = []
    
    def calculate_yield_spread(self, 
                              target_bond: SovereignBond, 
                              benchmark_bond: SovereignBond,
                              yield_curves: Dict[str, YieldCurve]) -> float:
        """Calculate simple yield spread between two bonds."""
        target_yield = self._get_bond_yield(target_bond, yield_curves)
        benchmark_yield = self._get_bond_yield(benchmark_bond, yield_curves)
        
        spread = (target_yield - benchmark_yield) * 10000  # Convert to basis points
        
        # Store result
        result = SpreadResult(
            spread_type=SpreadType.YIELD_SPREAD,
            spread_value=spread,
            target_bond_id=target_bond.isin,
            benchmark_bond_id=benchmark_bond.isin,
            calculation_date=datetime.now()
        )
        self.calculation_history.append(result)
        
        return spread
    
    def calculate_z_spread(self, 
                          target_bond: SovereignBond, 
                          benchmark_bond: SovereignBond,
                          yield_curves: Dict[str, YieldCurve]) -> float:
        """Calculate Z-spread (simplified implementation)."""
        # Simplified Z-spread calculation
        yield_spread = self.calculate_yield_spread(target_bond, benchmark_bond, yield_curves)
        
        # For simplicity, assume Z-spread is approximately yield spread + adjustment
        z_spread = yield_spread * 1.1  # Simple adjustment factor
        
        result = SpreadResult(
            spread_type=SpreadType.Z_SPREAD,
            spread_value=z_spread,
            target_bond_id=target_bond.isin,
            benchmark_bond_id=benchmark_bond.isin,
            calculation_date=datetime.now()
        )
        self.calculation_history.append(result)
        
        return z_spread
    
    def calculate_duration_adjusted_spread(self, 
                                         target_bond: SovereignBond, 
                                         benchmark_bond: SovereignBond,
                                         yield_curves: Dict[str, YieldCurve]) -> float:
        """Calculate duration-adjusted spread."""
        target_duration = self._get_bond_duration(target_bond, yield_curves)
        benchmark_duration = self._get_bond_duration(benchmark_bond, yield_curves)
        
        yield_spread = self.calculate_yield_spread(target_bond, benchmark_bond, yield_curves)
        
        # Adjust for duration differences
        if benchmark_duration > 0:
            duration_adjustment = target_duration / benchmark_duration
            adjusted_spread = yield_spread * duration_adjustment
        else:
            adjusted_spread = yield_spread
        
        result = SpreadResult(
            spread_type=SpreadType.DURATION_ADJUSTED_SPREAD,
            spread_value=adjusted_spread,
            target_bond_id=target_bond.isin,
            benchmark_bond_id=benchmark_bond.isin,
            calculation_date=datetime.now()
        )
        self.calculation_history.append(result)
        
        return adjusted_spread
    
    def calculate_multiple_spreads(self, 
                                  target_bond: SovereignBond, 
                                  benchmark_bond: SovereignBond,
                                  yield_curves: Dict[str, YieldCurve],
                                  spread_types: List[SpreadType] = None) -> Dict[SpreadType, float]:
        """Calculate multiple spread types at once."""
        if spread_types is None:
            spread_types = [SpreadType.YIELD_SPREAD, SpreadType.Z_SPREAD, SpreadType.DURATION_ADJUSTED_SPREAD]
        
        results = {}
        
        for spread_type in spread_types:
            if spread_type == SpreadType.YIELD_SPREAD:
                results[spread_type] = self.calculate_yield_spread(target_bond, benchmark_bond, yield_curves)
            elif spread_type == SpreadType.Z_SPREAD:
                results[spread_type] = self.calculate_z_spread(target_bond, benchmark_bond, yield_curves)
            elif spread_type == SpreadType.DURATION_ADJUSTED_SPREAD:
                results[spread_type] = self.calculate_duration_adjusted_spread(target_bond, benchmark_bond, yield_curves)
            elif spread_type == SpreadType.CREDIT_SPREAD:
                results[spread_type] = self._calculate_credit_spread(target_bond, benchmark_bond, yield_curves)
        
        return results
    
    def get_calculation_history(self) -> List[SpreadResult]:
        """Get history of all spread calculations."""
        return self.calculation_history.copy()
    
    def clear_history(self):
        """Clear calculation history."""
        self.calculation_history.clear()
    
    def _get_bond_yield(self, bond: SovereignBond, yield_curves: Dict[str, YieldCurve]) -> float:
        """Get bond yield from yield curves."""
        currency_key = bond.currency.value if hasattr(bond.currency, 'value') else str(bond.currency)
        
        if currency_key in yield_curves:
            # Calculate years to maturity
            days_to_maturity = (bond.maturity_date - date.today()).days
            years_to_maturity = days_to_maturity / 365.25
            
            # Get yield from curve
            return yield_curves[currency_key].get_yield(years_to_maturity)
        else:
            # Fallback to coupon rate if no yield curve available
            return bond.coupon_rate
    
    def _get_bond_duration(self, bond: SovereignBond, yield_curves: Dict[str, YieldCurve]) -> float:
        """Get bond duration (simplified calculation)."""
        # Simplified duration calculation
        years_to_maturity = (bond.maturity_date - date.today()).days / 365.25
        bond_yield = self._get_bond_yield(bond, yield_curves)
        
        # Modified duration approximation
        if bond_yield > 0:
            duration = years_to_maturity / (1 + bond_yield)
        else:
            duration = years_to_maturity
        
        return duration
    
    def _calculate_credit_spread(self, 
                               target_bond: SovereignBond, 
                               benchmark_bond: SovereignBond,
                               yield_curves: Dict[str, YieldCurve]) -> float:
        """Calculate credit spread (simplified)."""
        # For sovereign bonds, credit spread is often minimal
        # This is a simplified implementation
        yield_spread = self.calculate_yield_spread(target_bond, benchmark_bond, yield_curves)
        
        # Assume credit component is a fraction of yield spread
        credit_spread = yield_spread * 0.3  # 30% of yield spread attributed to credit
        
        result = SpreadResult(
            spread_type=SpreadType.CREDIT_SPREAD,
            spread_value=credit_spread,
            target_bond_id=target_bond.isin,
            benchmark_bond_id=benchmark_bond.isin,
            calculation_date=datetime.now()
        )
        self.calculation_history.append(result)
        
        return credit_spread