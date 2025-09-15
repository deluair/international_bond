"""
Currency-hedged strategy implementation for FX-neutral bond strategies.
"""

import numpy as np
from datetime import date, datetime, timedelta
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import math

from ..models.bond import SovereignBond
from ..models.currency import CurrencyPair, CurrencyType
from ..models.portfolio import Portfolio, Position
from ..currency.fx_hedge_calculator import FXHedgeCalculator, HedgeType


class HedgeStrategy(Enum):
    """Currency hedge strategies."""
    FULL_HEDGE = "full_hedge"
    PARTIAL_HEDGE = "partial_hedge"
    DYNAMIC_HEDGE = "dynamic_hedge"
    SELECTIVE_HEDGE = "selective_hedge"
    NO_HEDGE = "no_hedge"
    OVERLAY_HEDGE = "overlay_hedge"


class HedgeFrequency(Enum):
    """Hedge rebalancing frequency."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    SEMI_ANNUAL = "semi_annual"
    ANNUAL = "annual"
    EVENT_DRIVEN = "event_driven"


@dataclass
class HedgeConstraints:
    """Currency hedge constraints."""
    # Hedge ratio constraints
    min_hedge_ratio: float = 0.0
    max_hedge_ratio: float = 1.0
    target_hedge_ratio: float = 1.0
    
    # Currency-specific constraints
    currency_hedge_ratios: Dict[str, Tuple[float, float]] = field(default_factory=dict)  # (min, max)
    excluded_currencies: List[str] = field(default_factory=list)
    
    # Cost constraints
    max_hedge_cost: float = 0.005  # 50 bps
    max_transaction_cost: float = 0.001  # 10 bps
    
    # Risk constraints
    max_tracking_error: float = 0.02  # 2%
    max_basis_risk: float = 0.01  # 1%
    
    # Operational constraints
    min_hedge_notional: float = 1000000  # $1M minimum
    hedge_frequency: HedgeFrequency = HedgeFrequency.MONTHLY
    
    # Hedge effectiveness constraints
    min_hedge_effectiveness: float = 0.8  # 80% minimum effectiveness
    
    def get_currency_hedge_bounds(self, currency: str) -> Tuple[float, float]:
        """Get hedge ratio bounds for specific currency."""
        if currency in self.currency_hedge_ratios:
            return self.currency_hedge_ratios[currency]
        else:
            return (self.min_hedge_ratio, self.max_hedge_ratio)


@dataclass
class StrategyPerformance:
    """Currency-hedged strategy performance metrics."""
    # Return metrics
    total_return: float
    hedged_return: float
    unhedged_return: float
    currency_return: float
    hedge_alpha: float
    
    # Risk metrics
    total_volatility: float
    hedged_volatility: float
    unhedged_volatility: float
    currency_volatility: float
    
    # Hedge effectiveness
    hedge_effectiveness: float
    hedge_ratio_realized: float
    basis_risk: float
    
    # Cost metrics
    hedge_costs: float
    transaction_costs: float
    total_costs: float
    
    # Risk-adjusted metrics
    sharpe_ratio: float
    information_ratio: float
    calmar_ratio: float
    
    def __str__(self) -> str:
        return (f"Currency-Hedged Strategy Performance:\n"
                f"Total Return: {self.total_return:.2%}\n"
                f"Hedged Return: {self.hedged_return:.2%}\n"
                f"Currency Return: {self.currency_return:.2%}\n"
                f"Hedge Alpha: {self.hedge_alpha:.2%}\n"
                f"Total Volatility: {self.total_volatility:.2%}\n"
                f"Hedge Effectiveness: {self.hedge_effectiveness:.2%}\n"
                f"Hedge Costs: {self.hedge_costs:.4f}\n"
                f"Sharpe Ratio: {self.sharpe_ratio:.2f}")


class CurrencyHedgedStrategy:
    """
    Comprehensive currency-hedged strategy implementation.
    """
    
    def __init__(self, 
                 base_currency: CurrencyType = CurrencyType.MAJOR,
                 fx_hedge_calculator: Optional[FXHedgeCalculator] = None):
        """
        Initialize currency-hedged strategy.
        
        Args:
            base_currency: Base currency for the strategy
            fx_hedge_calculator: FX hedge calculator instance
        """
        self.base_currency = base_currency
        self.fx_hedge_calculator = fx_hedge_calculator or FXHedgeCalculator(base_currency)
        self.hedge_history: List[Dict] = []
        self.performance_history: List[StrategyPerformance] = []
    
    def calculate_optimal_hedge_ratios(self, 
                                     portfolio: Portfolio,
                                     currency_pairs: List[CurrencyPair],
                                     constraints: HedgeConstraints,
                                     strategy: HedgeStrategy = HedgeStrategy.FULL_HEDGE,
                                     lookback_days: int = 252) -> Dict[str, float]:
        """
        Calculate optimal hedge ratios for portfolio currencies.
        
        Args:
            portfolio: Portfolio to hedge
            currency_pairs: Available currency pairs for hedging
            constraints: Hedge constraints
            strategy: Hedge strategy type
            lookback_days: Historical data lookback period
            
        Returns:
            Dictionary of optimal hedge ratios by currency
        """
        # Get currency exposures
        currency_exposures = self._calculate_currency_exposures(portfolio)
        
        if not currency_exposures:
            return {}
        
        hedge_ratios = {}
        
        for currency, exposure in currency_exposures.items():
            if currency in constraints.excluded_currencies:
                hedge_ratios[currency] = 0.0
                continue
            
            if abs(exposure) < constraints.min_hedge_notional:
                hedge_ratios[currency] = 0.0
                continue
            
            # Get currency pair for hedging
            currency_pair = self._find_currency_pair(currency, currency_pairs)
            if not currency_pair:
                hedge_ratios[currency] = 0.0
                continue
            
            # Calculate optimal hedge ratio based on strategy
            if strategy == HedgeStrategy.FULL_HEDGE:
                optimal_ratio = constraints.target_hedge_ratio
            elif strategy == HedgeStrategy.PARTIAL_HEDGE:
                optimal_ratio = constraints.target_hedge_ratio * 0.5  # 50% hedge
            elif strategy == HedgeStrategy.DYNAMIC_HEDGE:
                optimal_ratio = self._calculate_dynamic_hedge_ratio(
                    currency_pair, lookback_days, constraints
                )
            elif strategy == HedgeStrategy.SELECTIVE_HEDGE:
                optimal_ratio = self._calculate_selective_hedge_ratio(
                    currency_pair, exposure, constraints
                )
            elif strategy == HedgeStrategy.NO_HEDGE:
                optimal_ratio = 0.0
            elif strategy == HedgeStrategy.OVERLAY_HEDGE:
                optimal_ratio = self._calculate_overlay_hedge_ratio(
                    currency_pair, lookback_days, constraints
                )
            else:
                optimal_ratio = constraints.target_hedge_ratio
            
            # Apply constraints
            min_ratio, max_ratio = constraints.get_currency_hedge_bounds(currency)
            optimal_ratio = max(min_ratio, min(max_ratio, optimal_ratio))
            
            hedge_ratios[currency] = optimal_ratio
        
        return hedge_ratios
    
    def implement_hedge_strategy(self, 
                               portfolio: Portfolio,
                               hedge_ratios: Dict[str, float],
                               currency_pairs: List[CurrencyPair],
                               constraints: HedgeConstraints) -> Dict[str, Dict]:
        """
        Implement hedge strategy with specific hedge ratios.
        
        Args:
            portfolio: Portfolio to hedge
            hedge_ratios: Target hedge ratios by currency
            currency_pairs: Available currency pairs
            constraints: Hedge constraints
            
        Returns:
            Dictionary of hedge transactions by currency
        """
        hedge_transactions = {}
        currency_exposures = self._calculate_currency_exposures(portfolio)
        
        for currency, target_ratio in hedge_ratios.items():
            if target_ratio == 0.0:
                continue
            
            exposure = currency_exposures.get(currency, 0.0)
            if abs(exposure) < constraints.min_hedge_notional:
                continue
            
            # Find appropriate currency pair
            currency_pair = self._find_currency_pair(currency, currency_pairs)
            if not currency_pair:
                continue
            
            # Calculate hedge notional
            hedge_notional = abs(exposure) * target_ratio
            
            # Determine hedge direction
            hedge_direction = 1 if exposure > 0 else -1
            
            # Calculate hedge cost
            hedge_cost = self._calculate_hedge_cost(
                currency_pair, hedge_notional, constraints
            )
            
            # Create hedge transaction
            hedge_transactions[currency] = {
                'currency_pair': f"{currency}/{self.base_currency.value}",
                'hedge_notional': hedge_notional,
                'hedge_direction': hedge_direction,
                'hedge_ratio': target_ratio,
                'exposure': exposure,
                'hedge_cost': hedge_cost,
                'hedge_type': HedgeType.FORWARD,  # Default to forward
                'maturity_days': 30  # Default 1-month hedge
            }
        
        # Store hedge history
        self.hedge_history.append({
            'date': date.today(),
            'hedge_transactions': hedge_transactions,
            'total_exposure': sum(abs(exp) for exp in currency_exposures.values()),
            'total_hedge_notional': sum(
                trans['hedge_notional'] for trans in hedge_transactions.values()
            )
        })
        
        return hedge_transactions
    
    def calculate_hedge_effectiveness(self, 
                                    portfolio_returns: np.ndarray,
                                    currency_returns: np.ndarray,
                                    hedge_returns: np.ndarray) -> float:
        """
        Calculate hedge effectiveness using regression analysis.
        
        Args:
            portfolio_returns: Portfolio return series
            currency_returns: Currency return series
            hedge_returns: Hedge return series
            
        Returns:
            Hedge effectiveness ratio (0-1)
        """
        if len(portfolio_returns) != len(currency_returns) or len(portfolio_returns) != len(hedge_returns):
            return 0.0
        
        if len(portfolio_returns) < 2:
            return 0.0
        
        # Calculate hedged portfolio returns
        hedged_returns = portfolio_returns + hedge_returns
        
        # Calculate variance reduction
        unhedged_variance = np.var(portfolio_returns)
        hedged_variance = np.var(hedged_returns)
        
        if unhedged_variance > 0:
            effectiveness = 1.0 - (hedged_variance / unhedged_variance)
            return max(0.0, min(1.0, effectiveness))
        else:
            return 0.0
    
    def calculate_strategy_performance(self, 
                                     portfolio_returns: np.ndarray,
                                     currency_returns: np.ndarray,
                                     hedge_returns: np.ndarray,
                                     hedge_costs: np.ndarray,
                                     risk_free_rate: float = 0.02) -> StrategyPerformance:
        """
        Calculate comprehensive strategy performance metrics.
        
        Args:
            portfolio_returns: Portfolio return series
            currency_returns: Currency return series  
            hedge_returns: Hedge return series
            hedge_costs: Hedge cost series
            risk_free_rate: Risk-free rate for Sharpe ratio
            
        Returns:
            StrategyPerformance object
        """
        if len(portfolio_returns) == 0:
            return self._create_empty_performance()
        
        # Calculate return components
        unhedged_returns = portfolio_returns + currency_returns
        hedged_returns = portfolio_returns + hedge_returns - hedge_costs
        
        # Return metrics
        total_return = np.sum(hedged_returns)
        hedged_return = np.sum(portfolio_returns + hedge_returns)
        unhedged_return = np.sum(unhedged_returns)
        currency_return = np.sum(currency_returns)
        hedge_alpha = total_return - unhedged_return
        
        # Risk metrics
        total_volatility = np.std(hedged_returns) * math.sqrt(252) if len(hedged_returns) > 1 else 0.0
        hedged_volatility = np.std(portfolio_returns + hedge_returns) * math.sqrt(252) if len(hedged_returns) > 1 else 0.0
        unhedged_volatility = np.std(unhedged_returns) * math.sqrt(252) if len(unhedged_returns) > 1 else 0.0
        currency_volatility = np.std(currency_returns) * math.sqrt(252) if len(currency_returns) > 1 else 0.0
        
        # Hedge effectiveness
        hedge_effectiveness = self.calculate_hedge_effectiveness(
            portfolio_returns, currency_returns, hedge_returns
        )
        
        # Realized hedge ratio (simplified)
        if np.std(currency_returns) > 0:
            hedge_ratio_realized = -np.corrcoef(currency_returns, hedge_returns)[0, 1]
        else:
            hedge_ratio_realized = 0.0
        
        # Basis risk
        basis_risk = np.std(hedge_returns + currency_returns) if len(hedge_returns) > 1 else 0.0
        
        # Cost metrics
        total_hedge_costs = np.sum(hedge_costs)
        transaction_costs = total_hedge_costs * 0.1  # Assume 10% of hedge costs are transaction costs
        total_costs = total_hedge_costs
        
        # Risk-adjusted metrics
        if total_volatility > 0:
            sharpe_ratio = (total_return - risk_free_rate) / total_volatility
        else:
            sharpe_ratio = 0.0
        
        # Information ratio (vs unhedged)
        excess_returns = hedged_returns - unhedged_returns
        if np.std(excess_returns) > 0:
            information_ratio = np.mean(excess_returns) / np.std(excess_returns)
        else:
            information_ratio = 0.0
        
        # Calmar ratio (return/max drawdown)
        cumulative_returns = np.cumsum(hedged_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = cumulative_returns - running_max
        max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0.0
        
        if max_drawdown < 0:
            calmar_ratio = total_return / abs(max_drawdown)
        else:
            calmar_ratio = 0.0
        
        performance = StrategyPerformance(
            total_return=total_return,
            hedged_return=hedged_return,
            unhedged_return=unhedged_return,
            currency_return=currency_return,
            hedge_alpha=hedge_alpha,
            total_volatility=total_volatility,
            hedged_volatility=hedged_volatility,
            unhedged_volatility=unhedged_volatility,
            currency_volatility=currency_volatility,
            hedge_effectiveness=hedge_effectiveness,
            hedge_ratio_realized=hedge_ratio_realized,
            basis_risk=basis_risk,
            hedge_costs=total_hedge_costs,
            transaction_costs=transaction_costs,
            total_costs=total_costs,
            sharpe_ratio=sharpe_ratio,
            information_ratio=information_ratio,
            calmar_ratio=calmar_ratio
        )
        
        self.performance_history.append(performance)
        return performance
    
    def optimize_hedge_timing(self, 
                            currency_pair: CurrencyPair,
                            hedge_frequency: HedgeFrequency,
                            lookback_days: int = 252) -> Dict[str, float]:
        """
        Optimize hedge timing based on market conditions.
        
        Args:
            currency_pair: Currency pair to analyze
            hedge_frequency: Current hedge frequency
            lookback_days: Historical data lookback
            
        Returns:
            Dictionary with timing recommendations
        """
        # Simulate market data for timing analysis
        np.random.seed(hash(currency_pair.base_currency.value) % 2**32)
        
        # Generate synthetic FX returns
        fx_returns = np.random.normal(0, currency_pair.volatility or 0.15, lookback_days)
        
        # Calculate volatility regimes
        rolling_vol = np.array([
            np.std(fx_returns[max(0, i-20):i+1]) for i in range(len(fx_returns))
        ])
        
        high_vol_threshold = np.percentile(rolling_vol, 75)
        low_vol_threshold = np.percentile(rolling_vol, 25)
        
        current_vol = rolling_vol[-1] if len(rolling_vol) > 0 else currency_pair.volatility or 0.15
        
        # Timing recommendations
        if current_vol > high_vol_threshold:
            timing_score = 0.8  # High urgency to hedge
            recommended_frequency = HedgeFrequency.WEEKLY
        elif current_vol < low_vol_threshold:
            timing_score = 0.3  # Low urgency to hedge
            recommended_frequency = HedgeFrequency.QUARTERLY
        else:
            timing_score = 0.5  # Normal timing
            recommended_frequency = HedgeFrequency.MONTHLY
        
        return {
            'timing_score': timing_score,
            'current_volatility': current_vol,
            'volatility_regime': 'high' if current_vol > high_vol_threshold else 'low' if current_vol < low_vol_threshold else 'normal',
            'recommended_frequency': recommended_frequency,
            'days_to_next_hedge': self._calculate_days_to_hedge(recommended_frequency)
        }
    
    def _calculate_currency_exposures(self, portfolio: Portfolio) -> Dict[str, float]:
        """Calculate currency exposures from portfolio."""
        exposures = {}
        
        for position in portfolio.positions:
            # Get position currency (assuming it's stored in position)
            position_currency = getattr(position, 'currency', self.base_currency.value)
            
            if position_currency != self.base_currency.value:
                if position_currency in exposures:
                    exposures[position_currency] += position.market_value
                else:
                    exposures[position_currency] = position.market_value
        
        return exposures
    
    def _find_currency_pair(self, 
                          currency: str,
                          currency_pairs: List[CurrencyPair]) -> Optional[CurrencyPair]:
        """Find appropriate currency pair for hedging."""
        for pair in currency_pairs:
            if (pair.base_currency.value == currency and 
                pair.quote_currency == self.base_currency):
                return pair
            elif (pair.quote_currency.value == currency and 
                  pair.base_currency == self.base_currency):
                return pair
        
        return None
    
    def _calculate_dynamic_hedge_ratio(self, 
                                     currency_pair: CurrencyPair,
                                     lookback_days: int,
                                     constraints: HedgeConstraints) -> float:
        """Calculate dynamic hedge ratio based on market conditions."""
        # Simulate dynamic calculation
        volatility = currency_pair.volatility or 0.15
        
        # Higher volatility -> higher hedge ratio
        vol_factor = min(1.0, volatility / 0.20)  # Normalize to 20% vol
        
        # Calculate dynamic ratio
        dynamic_ratio = constraints.target_hedge_ratio * (0.5 + 0.5 * vol_factor)
        
        return dynamic_ratio
    
    def _calculate_selective_hedge_ratio(self, 
                                       currency_pair: CurrencyPair,
                                       exposure: float,
                                       constraints: HedgeConstraints) -> float:
        """Calculate selective hedge ratio based on exposure and market conditions."""
        # Larger exposures get higher hedge ratios
        exposure_factor = min(1.0, abs(exposure) / 10000000)  # Normalize to $10M
        
        # Market conditions factor (simplified)
        market_factor = 0.8  # Assume moderate market stress
        
        selective_ratio = constraints.target_hedge_ratio * exposure_factor * market_factor
        
        return selective_ratio
    
    def _calculate_overlay_hedge_ratio(self, 
                                     currency_pair: CurrencyPair,
                                     lookback_days: int,
                                     constraints: HedgeConstraints) -> float:
        """Calculate overlay hedge ratio for alpha generation."""
        # Overlay strategies may over/under hedge for alpha
        base_ratio = constraints.target_hedge_ratio
        
        # Add alpha overlay (simplified momentum signal)
        momentum_signal = 0.1  # Assume 10% momentum signal
        
        overlay_ratio = base_ratio + momentum_signal
        
        return overlay_ratio
    
    def _calculate_hedge_cost(self, 
                            currency_pair: CurrencyPair,
                            notional: float,
                            constraints: HedgeConstraints) -> float:
        """Calculate hedge cost."""
        # Simplified hedge cost calculation
        base_cost = constraints.max_hedge_cost * 0.5  # Assume 50% of max cost
        
        # Scale by notional (larger trades may get better pricing)
        notional_factor = max(0.5, min(1.0, notional / 10000000))  # $10M reference
        
        hedge_cost = base_cost * notional_factor * (notional / 1000000)  # Cost in dollars per million
        
        return hedge_cost
    
    def _calculate_days_to_hedge(self, frequency: HedgeFrequency) -> int:
        """Calculate days until next hedge based on frequency."""
        frequency_days = {
            HedgeFrequency.DAILY: 1,
            HedgeFrequency.WEEKLY: 7,
            HedgeFrequency.MONTHLY: 30,
            HedgeFrequency.QUARTERLY: 90,
            HedgeFrequency.SEMI_ANNUAL: 180,
            HedgeFrequency.ANNUAL: 365,
            HedgeFrequency.EVENT_DRIVEN: 0
        }
        
        return frequency_days.get(frequency, 30)
    
    def _create_empty_performance(self) -> StrategyPerformance:
        """Create empty performance object."""
        return StrategyPerformance(
            total_return=0.0,
            hedged_return=0.0,
            unhedged_return=0.0,
            currency_return=0.0,
            hedge_alpha=0.0,
            total_volatility=0.0,
            hedged_volatility=0.0,
            unhedged_volatility=0.0,
            currency_volatility=0.0,
            hedge_effectiveness=0.0,
            hedge_ratio_realized=0.0,
            basis_risk=0.0,
            hedge_costs=0.0,
            transaction_costs=0.0,
            total_costs=0.0,
            sharpe_ratio=0.0,
            information_ratio=0.0,
            calmar_ratio=0.0
        )