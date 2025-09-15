"""
Portfolio data model for managing collections of bonds and calculating metrics.
"""

from dataclasses import dataclass, field
from datetime import date
from typing import List, Dict, Optional, Tuple
import numpy as np
from .bond import SovereignBond
from .currency import CurrencyPair


@dataclass
class Position:
    """Individual bond position within a portfolio."""
    bond: SovereignBond
    quantity: float  # Face value amount
    weight: float = 0.0  # Portfolio weight (calculated)
    hedge_ratio: float = 0.0  # Currency hedge ratio
    
    @property
    def market_value(self) -> float:
        """Calculate market value of the position."""
        return self.quantity * (self.bond.current_price / 100)
    
    @property
    def duration_contribution(self) -> float:
        """Calculate duration contribution to portfolio."""
        return self.weight * self.bond.modified_duration
    
    @property
    def yield_contribution(self) -> float:
        """Calculate yield contribution to portfolio."""
        return self.weight * self.bond.yield_to_maturity


@dataclass
class Portfolio:
    """
    Portfolio of sovereign bonds with risk and performance analytics.
    """
    name: str
    base_currency: str
    positions: List[Position] = field(default_factory=list)
    rebalance_date: Optional[date] = None
    
    # Portfolio constraints
    max_country_weight: float = 0.3
    max_currency_weight: float = 0.4
    min_credit_rating: str = "BBB-"
    
    def __post_init__(self):
        """Initialize portfolio calculations."""
        self._update_weights()
    
    def add_position(self, bond: SovereignBond, quantity: float, hedge_ratio: float = 0.0):
        """Add a bond position to the portfolio."""
        position = Position(
            bond=bond,
            quantity=quantity,
            hedge_ratio=hedge_ratio
        )
        self.positions.append(position)
        self._update_weights()
    
    def remove_position(self, isin: str):
        """Remove a position by ISIN."""
        self.positions = [p for p in self.positions if p.bond.isin != isin]
        self._update_weights()
    
    def _update_weights(self):
        """Update position weights based on market values."""
        total_value = self.total_market_value
        if total_value > 0:
            for position in self.positions:
                position.weight = position.market_value / total_value
    
    @property
    def total_market_value(self) -> float:
        """Calculate total portfolio market value."""
        return sum(position.market_value for position in self.positions)
    
    @property
    def portfolio_duration(self) -> float:
        """Calculate portfolio modified duration."""
        return sum(position.duration_contribution for position in self.positions)
    
    @property
    def portfolio_yield(self) -> float:
        """Calculate portfolio yield to maturity."""
        return sum(position.yield_contribution for position in self.positions)
    
    @property
    def portfolio_convexity(self) -> float:
        """Calculate portfolio convexity."""
        return sum(position.weight * position.bond.convexity for position in self.positions)
    
    @property
    def average_credit_spread(self) -> float:
        """Calculate weighted average credit spread."""
        total_spread = 0.0
        for position in self.positions:
            if position.bond.credit_spread:
                total_spread += position.weight * position.bond.credit_spread
        return total_spread
    
    @property
    def average_cds_spread(self) -> float:
        """Calculate weighted average CDS spread."""
        total_spread = 0.0
        for position in self.positions:
            if position.bond.cds_spread:
                total_spread += position.weight * position.bond.cds_spread
        return total_spread
    
    def get_country_exposure(self) -> Dict[str, float]:
        """Get country exposure breakdown."""
        country_exposure = {}
        for position in self.positions:
            country = position.bond.country
            if country in country_exposure:
                country_exposure[country] += position.weight
            else:
                country_exposure[country] = position.weight
        return country_exposure
    
    def get_currency_exposure(self) -> Dict[str, float]:
        """Get currency exposure breakdown."""
        currency_exposure = {}
        for position in self.positions:
            currency = position.bond.currency
            if currency in currency_exposure:
                currency_exposure[currency] += position.weight
            else:
                currency_exposure[currency] = position.weight
        return currency_exposure
    
    def get_duration_buckets(self) -> Dict[str, float]:
        """Get duration bucket exposure."""
        buckets = {
            "0-2Y": 0.0,
            "2-5Y": 0.0,
            "5-10Y": 0.0,
            "10Y+": 0.0
        }
        
        for position in self.positions:
            years_to_maturity = position.bond.years_to_maturity
            if years_to_maturity <= 2:
                buckets["0-2Y"] += position.weight
            elif years_to_maturity <= 5:
                buckets["2-5Y"] += position.weight
            elif years_to_maturity <= 10:
                buckets["5-10Y"] += position.weight
            else:
                buckets["10Y+"] += position.weight
        
        return buckets
    
    def calculate_var(self, confidence_level: float = 0.95, time_horizon: int = 1) -> float:
        """Calculate Value at Risk (simplified)."""
        # Simplified VaR calculation using duration and yield volatility
        portfolio_value = self.total_market_value
        duration = self.portfolio_duration
        
        # Assume 1% daily yield volatility (in practice, use historical data)
        yield_volatility = 0.01
        
        # Scale for time horizon
        scaled_volatility = yield_volatility * np.sqrt(time_horizon)
        
        # Normal distribution assumption
        from scipy.stats import norm
        z_score = norm.ppf(1 - confidence_level)
        
        var = portfolio_value * duration * scaled_volatility * abs(z_score)
        return var
    
    def calculate_tracking_error(self, benchmark_yield: float, 
                               benchmark_duration: float) -> float:
        """Calculate tracking error vs benchmark."""
        yield_diff = self.portfolio_yield - benchmark_yield
        duration_diff = self.portfolio_duration - benchmark_duration
        
        # Simplified tracking error calculation
        tracking_error = np.sqrt(yield_diff**2 + (duration_diff * 0.01)**2)
        return tracking_error
    
    def check_constraints(self) -> Dict[str, bool]:
        """Check if portfolio meets defined constraints."""
        constraints = {}
        
        # Country concentration
        country_exposure = self.get_country_exposure()
        max_country_actual = max(country_exposure.values()) if country_exposure else 0
        constraints["country_concentration"] = max_country_actual <= self.max_country_weight
        
        # Currency concentration
        currency_exposure = self.get_currency_exposure()
        max_currency_actual = max(currency_exposure.values()) if currency_exposure else 0
        constraints["currency_concentration"] = max_currency_actual <= self.max_currency_weight
        
        # Credit quality
        investment_grade_weight = sum(
            position.weight for position in self.positions 
            if position.bond.is_investment_grade
        )
        constraints["credit_quality"] = investment_grade_weight >= 0.8  # 80% IG minimum
        
        return constraints
    
    def rebalance_to_target_duration(self, target_duration: float) -> List[Tuple[str, float]]:
        """Calculate trades needed to achieve target duration."""
        current_duration = self.portfolio_duration
        duration_diff = target_duration - current_duration
        
        if abs(duration_diff) < 0.1:  # Close enough
            return []
        
        trades = []
        total_value = self.total_market_value
        
        # Simple rebalancing: adjust positions proportionally
        for position in self.positions:
            bond_duration = position.bond.modified_duration
            if bond_duration > 0:
                # Calculate required quantity change
                target_contribution = position.weight * target_duration
                current_contribution = position.weight * bond_duration
                contribution_diff = target_contribution - current_contribution
                
                quantity_change = (contribution_diff / bond_duration) * total_value
                
                if abs(quantity_change) > 1000:  # Minimum trade size
                    trades.append((position.bond.isin, quantity_change))
        
        return trades
    
    def calculate_currency_hedged_return(self, fx_rates: Dict[str, CurrencyPair], 
                                       period_days: int = 30) -> float:
        """Calculate currency-hedged portfolio return."""
        hedged_return = 0.0
        
        for position in self.positions:
            bond_currency = position.bond.currency
            
            # Bond return in local currency
            bond_return = position.bond.yield_to_maturity * (period_days / 365)
            
            # Currency impact
            if bond_currency != self.base_currency and bond_currency in fx_rates:
                fx_pair = fx_rates[bond_currency]
                
                # Unhedged FX return
                fx_return = 0.0  # Simplified - would calculate from FX rates
                
                # Hedged return (remove FX impact based on hedge ratio)
                hedged_fx_return = fx_return * (1 - position.hedge_ratio)
                total_return = bond_return + hedged_fx_return
            else:
                total_return = bond_return
            
            hedged_return += position.weight * total_return
        
        return hedged_return
    
    def to_dict(self) -> Dict:
        """Convert portfolio to dictionary representation."""
        return {
            'name': self.name,
            'base_currency': self.base_currency,
            'total_market_value': self.total_market_value,
            'portfolio_duration': self.portfolio_duration,
            'portfolio_yield': self.portfolio_yield,
            'portfolio_convexity': self.portfolio_convexity,
            'average_credit_spread': self.average_credit_spread,
            'average_cds_spread': self.average_cds_spread,
            'country_exposure': self.get_country_exposure(),
            'currency_exposure': self.get_currency_exposure(),
            'duration_buckets': self.get_duration_buckets(),
            'constraint_compliance': self.check_constraints(),
            'positions': [
                {
                    'isin': pos.bond.isin,
                    'issuer': pos.bond.issuer,
                    'quantity': pos.quantity,
                    'weight': pos.weight,
                    'market_value': pos.market_value,
                    'hedge_ratio': pos.hedge_ratio
                }
                for pos in self.positions
            ]
        }
    
    def __str__(self) -> str:
        """String representation of the portfolio."""
        return (f"Portfolio: {self.name} | "
                f"Value: {self.total_market_value:,.0f} {self.base_currency} | "
                f"Duration: {self.portfolio_duration:.2f} | "
                f"Yield: {self.portfolio_yield:.2%} | "
                f"Positions: {len(self.positions)}")