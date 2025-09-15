"""
Carry Trade Strategy Module

This module implements carry trade strategies for international bonds,
focusing on yield differentials and currency-hedged carry opportunities.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

class CarryTradeSignal(Enum):
    """Carry trade signal types"""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"

class CarryTradeType(Enum):
    """Types of carry trades"""
    CURRENCY_HEDGED = "currency_hedged"
    UNHEDGED = "unhedged"
    CROSS_CURRENCY = "cross_currency"
    INTRA_CURVE = "intra_curve"

class RiskAdjustmentMethod(Enum):
    """Risk adjustment methods for carry trades"""
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    SHARPE_RATIO = "sharpe_ratio"
    MAXIMUM_DRAWDOWN = "maximum_drawdown"
    VAR_ADJUSTED = "var_adjusted"

@dataclass
class CarryTradeOpportunity:
    """Represents a carry trade opportunity"""
    trade_id: str
    long_bond: str
    short_bond: str
    carry_yield: float
    funding_cost: float
    net_carry: float
    currency_hedge_cost: Optional[float]
    risk_adjusted_carry: float
    expected_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    trade_type: CarryTradeType
    signal_strength: CarryTradeSignal
    confidence_score: float
    entry_date: datetime
    recommended_holding_period: int  # days
    stop_loss_level: Optional[float]
    take_profit_level: Optional[float]
    risk_metrics: Dict[str, float]

@dataclass
class CarryTradePosition:
    """Represents an active carry trade position"""
    position_id: str
    opportunity: CarryTradeOpportunity
    entry_price_long: float
    entry_price_short: float
    position_size: float
    entry_date: datetime
    current_pnl: float
    unrealized_pnl: float
    carry_accrued: float
    funding_cost_paid: float
    hedge_cost_paid: float
    days_held: int
    current_signal: CarryTradeSignal
    exit_conditions: Dict[str, Any]

@dataclass
class CarryTradePerformance:
    """Performance metrics for carry trade strategies"""
    strategy_name: str
    total_return: float
    carry_return: float
    price_return: float
    currency_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    average_holding_period: float
    total_trades: int
    profitable_trades: int
    largest_win: float
    largest_loss: float
    calmar_ratio: float
    sortino_ratio: float

class CarryTradeStrategy:
    """
    Comprehensive carry trade strategy for international bonds
    """
    
    def __init__(
        self,
        strategy_name: str = "International Bond Carry",
        risk_adjustment_method: RiskAdjustmentMethod = RiskAdjustmentMethod.SHARPE_RATIO,
        min_carry_threshold: float = 0.005,  # 50 bps minimum carry
        max_position_size: float = 0.1,  # 10% max position
        currency_hedge_threshold: float = 0.002,  # 20 bps hedge cost threshold
        volatility_lookback: int = 252,  # 1 year
        rebalancing_frequency: int = 30  # monthly
    ):
        self.strategy_name = strategy_name
        self.risk_adjustment_method = risk_adjustment_method
        self.min_carry_threshold = min_carry_threshold
        self.max_position_size = max_position_size
        self.currency_hedge_threshold = currency_hedge_threshold
        self.volatility_lookback = volatility_lookback
        self.rebalancing_frequency = rebalancing_frequency
        
        # Strategy state
        self.active_positions: Dict[str, CarryTradePosition] = {}
        self.historical_opportunities: List[CarryTradeOpportunity] = []
        self.performance_history: List[CarryTradePerformance] = []
        
        # Market data cache
        self.yield_data: Dict[str, pd.Series] = {}
        self.fx_data: Dict[str, pd.Series] = {}
        self.volatility_data: Dict[str, pd.Series] = {}
    
    def identify_carry_opportunities(
        self,
        bond_universe: Dict[str, Dict[str, Any]],
        fx_rates: Dict[str, float],
        funding_rates: Dict[str, float],
        hedge_costs: Dict[str, float],
        analysis_date: datetime
    ) -> List[CarryTradeOpportunity]:
        """
        Identify carry trade opportunities across international bond markets
        """
        opportunities = []
        
        # Get all possible bond pairs
        bond_pairs = self._generate_bond_pairs(bond_universe)
        
        for long_bond, short_bond in bond_pairs:
            try:
                opportunity = self._evaluate_carry_opportunity(
                    long_bond, short_bond, bond_universe, fx_rates,
                    funding_rates, hedge_costs, analysis_date
                )
                
                if opportunity and self._meets_criteria(opportunity):
                    opportunities.append(opportunity)
                    
            except Exception as e:
                print(f"Error evaluating carry opportunity {long_bond}-{short_bond}: {str(e)}")
        
        # Sort by risk-adjusted carry
        opportunities.sort(key=lambda x: x.risk_adjusted_carry, reverse=True)
        
        return opportunities[:20]  # Return top 20 opportunities
    
    def generate_trade_signals(
        self,
        opportunities: List[CarryTradeOpportunity],
        market_conditions: Dict[str, Any],
        risk_budget: float = 1.0
    ) -> Dict[str, CarryTradeSignal]:
        """
        Generate trading signals based on carry opportunities and market conditions
        """
        signals = {}
        
        # Assess market regime
        market_regime = self._assess_market_regime(market_conditions)
        
        # Adjust signal generation based on market conditions
        risk_multiplier = self._get_risk_multiplier(market_regime)
        
        for opportunity in opportunities:
            signal = self._generate_signal_for_opportunity(
                opportunity, market_conditions, risk_multiplier
            )
            signals[opportunity.trade_id] = signal
        
        # Apply portfolio-level constraints
        signals = self._apply_portfolio_constraints(signals, opportunities, risk_budget)
        
        return signals
    
    def execute_carry_trades(
        self,
        signals: Dict[str, CarryTradeSignal],
        opportunities: List[CarryTradeOpportunity],
        current_prices: Dict[str, float],
        execution_date: datetime
    ) -> List[CarryTradePosition]:
        """
        Execute carry trades based on signals
        """
        new_positions = []
        
        for opportunity in opportunities:
            signal = signals.get(opportunity.trade_id, CarryTradeSignal.HOLD)
            
            if signal in [CarryTradeSignal.BUY, CarryTradeSignal.STRONG_BUY]:
                position = self._create_position(
                    opportunity, current_prices, execution_date, signal
                )
                
                if position:
                    self.active_positions[position.position_id] = position
                    new_positions.append(position)
        
        return new_positions
    
    def monitor_positions(
        self,
        current_prices: Dict[str, float],
        current_yields: Dict[str, float],
        fx_rates: Dict[str, float],
        monitoring_date: datetime
    ) -> Dict[str, CarryTradeSignal]:
        """
        Monitor active positions and generate exit signals
        """
        exit_signals = {}
        
        for position_id, position in self.active_positions.items():
            # Update position P&L
            self._update_position_pnl(position, current_prices, current_yields, fx_rates)
            
            # Check exit conditions
            exit_signal = self._check_exit_conditions(position, monitoring_date)
            
            if exit_signal != CarryTradeSignal.HOLD:
                exit_signals[position_id] = exit_signal
        
        return exit_signals
    
    def calculate_performance(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> CarryTradePerformance:
        """
        Calculate comprehensive performance metrics for the strategy
        """
        # Get all positions in the period
        period_positions = [
            pos for pos in self.active_positions.values()
            if start_date <= pos.entry_date <= end_date
        ]
        
        if not period_positions:
            return self._create_empty_performance()
        
        # Calculate returns
        total_return = sum(pos.current_pnl for pos in period_positions)
        carry_return = sum(pos.carry_accrued for pos in period_positions)
        price_return = total_return - carry_return
        
        # Calculate risk metrics
        returns_series = self._get_daily_returns(period_positions, start_date, end_date)
        volatility = np.std(returns_series) * np.sqrt(252) if len(returns_series) > 1 else 0
        
        sharpe_ratio = (np.mean(returns_series) * 252) / volatility if volatility > 0 else 0
        max_drawdown = self._calculate_max_drawdown(returns_series)
        
        # Calculate trade statistics
        closed_positions = [pos for pos in period_positions if pos.current_pnl != 0]
        win_rate = len([pos for pos in closed_positions if pos.current_pnl > 0]) / len(closed_positions) if closed_positions else 0
        
        return CarryTradePerformance(
            strategy_name=self.strategy_name,
            total_return=total_return,
            carry_return=carry_return,
            price_return=price_return,
            currency_return=0.0,  # Simplified
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            average_holding_period=np.mean([pos.days_held for pos in closed_positions]) if closed_positions else 0,
            total_trades=len(period_positions),
            profitable_trades=len([pos for pos in closed_positions if pos.current_pnl > 0]),
            largest_win=max([pos.current_pnl for pos in closed_positions]) if closed_positions else 0,
            largest_loss=min([pos.current_pnl for pos in closed_positions]) if closed_positions else 0,
            calmar_ratio=total_return / abs(max_drawdown) if max_drawdown != 0 else 0,
            sortino_ratio=self._calculate_sortino_ratio(returns_series)
        )
    
    def _generate_bond_pairs(self, bond_universe: Dict[str, Dict[str, Any]]) -> List[Tuple[str, str]]:
        """Generate all possible bond pairs for carry analysis"""
        bonds = list(bond_universe.keys())
        pairs = []
        
        for i, long_bond in enumerate(bonds):
            for short_bond in bonds[i+1:]:
                # Only pair bonds with different characteristics
                if self._are_suitable_for_carry(bond_universe[long_bond], bond_universe[short_bond]):
                    pairs.append((long_bond, short_bond))
        
        return pairs
    
    def _are_suitable_for_carry(self, bond1: Dict[str, Any], bond2: Dict[str, Any]) -> bool:
        """Check if two bonds are suitable for carry trade"""
        # Different currencies or different credit qualities
        return (bond1.get('currency') != bond2.get('currency') or
                bond1.get('credit_rating') != bond2.get('credit_rating') or
                abs(bond1.get('duration', 0) - bond2.get('duration', 0)) > 1.0)
    
    def _evaluate_carry_opportunity(
        self,
        long_bond: str,
        short_bond: str,
        bond_universe: Dict[str, Dict[str, Any]],
        fx_rates: Dict[str, float],
        funding_rates: Dict[str, float],
        hedge_costs: Dict[str, float],
        analysis_date: datetime
    ) -> Optional[CarryTradeOpportunity]:
        """Evaluate a specific carry trade opportunity"""
        
        long_data = bond_universe[long_bond]
        short_data = bond_universe[short_bond]
        
        # Calculate carry components
        carry_yield = long_data.get('yield', 0) - short_data.get('yield', 0)
        funding_cost = funding_rates.get(short_data.get('currency', 'USD'), 0)
        
        # Currency hedge cost if different currencies
        currency_hedge_cost = None
        if long_data.get('currency') != short_data.get('currency'):
            currency_pair = f"{long_data.get('currency')}{short_data.get('currency')}"
            currency_hedge_cost = hedge_costs.get(currency_pair, 0)
        
        net_carry = carry_yield - funding_cost
        if currency_hedge_cost:
            net_carry -= currency_hedge_cost
        
        # Risk adjustment
        volatility = self._estimate_trade_volatility(long_bond, short_bond)
        risk_adjusted_carry = net_carry / volatility if volatility > 0 else 0
        
        # Expected return (simplified)
        expected_return = net_carry * 0.8  # Assume 80% carry realization
        
        # Signal strength
        signal_strength = self._determine_signal_strength(risk_adjusted_carry, net_carry)
        
        # Confidence score
        confidence_score = self._calculate_confidence_score(
            long_data, short_data, net_carry, volatility
        )
        
        return CarryTradeOpportunity(
            trade_id=f"{long_bond}_{short_bond}_{analysis_date.strftime('%Y%m%d')}",
            long_bond=long_bond,
            short_bond=short_bond,
            carry_yield=carry_yield,
            funding_cost=funding_cost,
            net_carry=net_carry,
            currency_hedge_cost=currency_hedge_cost,
            risk_adjusted_carry=risk_adjusted_carry,
            expected_return=expected_return,
            volatility=volatility,
            sharpe_ratio=expected_return / volatility if volatility > 0 else 0,
            max_drawdown=volatility * 2,  # Simplified estimate
            trade_type=self._determine_trade_type(long_data, short_data),
            signal_strength=signal_strength,
            confidence_score=confidence_score,
            entry_date=analysis_date,
            recommended_holding_period=90,  # 3 months default
            stop_loss_level=-0.02,  # 2% stop loss
            take_profit_level=0.05,  # 5% take profit
            risk_metrics={
                'duration_risk': abs(long_data.get('duration', 0) - short_data.get('duration', 0)),
                'credit_risk': self._assess_credit_risk_differential(long_data, short_data),
                'currency_risk': 1.0 if long_data.get('currency') != short_data.get('currency') else 0.0
            }
        )
    
    def _meets_criteria(self, opportunity: CarryTradeOpportunity) -> bool:
        """Check if opportunity meets strategy criteria"""
        return (
            opportunity.net_carry >= self.min_carry_threshold and
            opportunity.confidence_score >= 0.6 and
            opportunity.volatility < 0.15  # Max 15% volatility
        )
    
    def _assess_market_regime(self, market_conditions: Dict[str, Any]) -> str:
        """Assess current market regime"""
        volatility = market_conditions.get('volatility', 0.1)
        trend = market_conditions.get('trend', 'neutral')
        
        if volatility > 0.2:
            return 'high_volatility'
        elif trend == 'rising_rates':
            return 'rising_rates'
        elif trend == 'falling_rates':
            return 'falling_rates'
        else:
            return 'neutral'
    
    def _get_risk_multiplier(self, market_regime: str) -> float:
        """Get risk multiplier based on market regime"""
        multipliers = {
            'high_volatility': 0.5,
            'rising_rates': 0.8,
            'falling_rates': 1.2,
            'neutral': 1.0
        }
        return multipliers.get(market_regime, 1.0)
    
    def _generate_signal_for_opportunity(
        self,
        opportunity: CarryTradeOpportunity,
        market_conditions: Dict[str, Any],
        risk_multiplier: float
    ) -> CarryTradeSignal:
        """Generate signal for a specific opportunity"""
        
        adjusted_carry = opportunity.risk_adjusted_carry * risk_multiplier
        
        if adjusted_carry > 0.15:
            return CarryTradeSignal.STRONG_BUY
        elif adjusted_carry > 0.08:
            return CarryTradeSignal.BUY
        elif adjusted_carry > -0.05:
            return CarryTradeSignal.HOLD
        elif adjusted_carry > -0.1:
            return CarryTradeSignal.SELL
        else:
            return CarryTradeSignal.STRONG_SELL
    
    def _apply_portfolio_constraints(
        self,
        signals: Dict[str, CarryTradeSignal],
        opportunities: List[CarryTradeOpportunity],
        risk_budget: float
    ) -> Dict[str, CarryTradeSignal]:
        """Apply portfolio-level constraints to signals"""
        
        # Count buy signals
        buy_signals = [k for k, v in signals.items() if v in [CarryTradeSignal.BUY, CarryTradeSignal.STRONG_BUY]]
        
        # Limit number of positions based on risk budget
        max_positions = int(risk_budget / self.max_position_size)
        
        if len(buy_signals) > max_positions:
            # Keep only the best opportunities
            opportunity_dict = {opp.trade_id: opp for opp in opportunities}
            buy_signals.sort(key=lambda x: opportunity_dict[x].risk_adjusted_carry, reverse=True)
            
            # Downgrade excess signals to HOLD
            for signal_id in buy_signals[max_positions:]:
                signals[signal_id] = CarryTradeSignal.HOLD
        
        return signals
    
    def _create_position(
        self,
        opportunity: CarryTradeOpportunity,
        current_prices: Dict[str, float],
        execution_date: datetime,
        signal: CarryTradeSignal
    ) -> Optional[CarryTradePosition]:
        """Create a new carry trade position"""
        
        position_size = self._calculate_position_size(opportunity, signal)
        
        return CarryTradePosition(
            position_id=f"POS_{opportunity.trade_id}_{execution_date.strftime('%Y%m%d_%H%M%S')}",
            opportunity=opportunity,
            entry_price_long=current_prices.get(opportunity.long_bond, 100.0),
            entry_price_short=current_prices.get(opportunity.short_bond, 100.0),
            position_size=position_size,
            entry_date=execution_date,
            current_pnl=0.0,
            unrealized_pnl=0.0,
            carry_accrued=0.0,
            funding_cost_paid=0.0,
            hedge_cost_paid=0.0,
            days_held=0,
            current_signal=signal,
            exit_conditions={
                'stop_loss': opportunity.stop_loss_level,
                'take_profit': opportunity.take_profit_level,
                'max_holding_period': opportunity.recommended_holding_period
            }
        )
    
    def _calculate_position_size(self, opportunity: CarryTradeOpportunity, signal: CarryTradeSignal) -> float:
        """Calculate position size based on opportunity and signal strength"""
        
        base_size = self.max_position_size
        
        # Adjust based on signal strength
        if signal == CarryTradeSignal.STRONG_BUY:
            size_multiplier = 1.0
        elif signal == CarryTradeSignal.BUY:
            size_multiplier = 0.7
        else:
            size_multiplier = 0.5
        
        # Adjust based on risk metrics
        risk_adjustment = 1.0 / (1.0 + opportunity.volatility)
        
        return base_size * size_multiplier * risk_adjustment
    
    def _update_position_pnl(
        self,
        position: CarryTradePosition,
        current_prices: Dict[str, float],
        current_yields: Dict[str, float],
        fx_rates: Dict[str, float]
    ):
        """Update position P&L"""
        
        # Price P&L
        long_price_pnl = (current_prices.get(position.opportunity.long_bond, position.entry_price_long) - 
                          position.entry_price_long) / position.entry_price_long
        short_price_pnl = (position.entry_price_short - 
                          current_prices.get(position.opportunity.short_bond, position.entry_price_short)) / position.entry_price_short
        
        price_pnl = (long_price_pnl + short_price_pnl) * position.position_size
        
        # Carry accrual (simplified)
        days_since_entry = (datetime.now() - position.entry_date).days
        carry_accrual = position.opportunity.net_carry * (days_since_entry / 365) * position.position_size
        
        position.current_pnl = price_pnl + carry_accrual
        position.carry_accrued = carry_accrual
        position.days_held = days_since_entry
    
    def _check_exit_conditions(self, position: CarryTradePosition, current_date: datetime) -> CarryTradeSignal:
        """Check if position should be exited"""
        
        # Stop loss
        if position.current_pnl / position.position_size <= position.exit_conditions.get('stop_loss', -0.02):
            return CarryTradeSignal.STRONG_SELL
        
        # Take profit
        if position.current_pnl / position.position_size >= position.exit_conditions.get('take_profit', 0.05):
            return CarryTradeSignal.STRONG_SELL
        
        # Max holding period
        if position.days_held >= position.exit_conditions.get('max_holding_period', 90):
            return CarryTradeSignal.SELL
        
        return CarryTradeSignal.HOLD
    
    def _estimate_trade_volatility(self, long_bond: str, short_bond: str) -> float:
        """Estimate volatility of the carry trade"""
        # Simplified volatility estimation
        return 0.08  # 8% annual volatility
    
    def _determine_signal_strength(self, risk_adjusted_carry: float, net_carry: float) -> CarryTradeSignal:
        """Determine signal strength based on carry metrics"""
        
        if risk_adjusted_carry > 0.1 and net_carry > 0.03:
            return CarryTradeSignal.STRONG_BUY
        elif risk_adjusted_carry > 0.05 and net_carry > 0.015:
            return CarryTradeSignal.BUY
        elif risk_adjusted_carry > -0.02:
            return CarryTradeSignal.HOLD
        elif risk_adjusted_carry > -0.05:
            return CarryTradeSignal.SELL
        else:
            return CarryTradeSignal.STRONG_SELL
    
    def _calculate_confidence_score(
        self,
        long_data: Dict[str, Any],
        short_data: Dict[str, Any],
        net_carry: float,
        volatility: float
    ) -> float:
        """Calculate confidence score for the trade"""
        
        # Base score from carry-to-volatility ratio
        base_score = min(abs(net_carry) / volatility, 1.0) if volatility > 0 else 0
        
        # Adjust for data quality and liquidity
        liquidity_adjustment = 0.9  # Assume good liquidity
        
        return base_score * liquidity_adjustment
    
    def _determine_trade_type(self, long_data: Dict[str, Any], short_data: Dict[str, Any]) -> CarryTradeType:
        """Determine the type of carry trade"""
        
        if long_data.get('currency') != short_data.get('currency'):
            return CarryTradeType.CROSS_CURRENCY
        elif long_data.get('maturity') != short_data.get('maturity'):
            return CarryTradeType.INTRA_CURVE
        else:
            return CarryTradeType.CURRENCY_HEDGED
    
    def _assess_credit_risk_differential(self, long_data: Dict[str, Any], short_data: Dict[str, Any]) -> float:
        """Assess credit risk differential between bonds"""
        
        # Simplified credit risk assessment
        credit_ratings = {'AAA': 1, 'AA': 2, 'A': 3, 'BBB': 4, 'BB': 5, 'B': 6}
        
        long_rating = credit_ratings.get(long_data.get('credit_rating', 'A'), 3)
        short_rating = credit_ratings.get(short_data.get('credit_rating', 'A'), 3)
        
        return abs(long_rating - short_rating) / 6.0
    
    def _create_empty_performance(self) -> CarryTradePerformance:
        """Create empty performance object"""
        
        return CarryTradePerformance(
            strategy_name=self.strategy_name,
            total_return=0.0,
            carry_return=0.0,
            price_return=0.0,
            currency_return=0.0,
            volatility=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            average_holding_period=0.0,
            total_trades=0,
            profitable_trades=0,
            largest_win=0.0,
            largest_loss=0.0,
            calmar_ratio=0.0,
            sortino_ratio=0.0
        )
    
    def _get_daily_returns(
        self,
        positions: List[CarryTradePosition],
        start_date: datetime,
        end_date: datetime
    ) -> np.ndarray:
        """Get daily returns series for performance calculation"""
        
        # Simplified daily returns calculation
        days = (end_date - start_date).days
        if days <= 0:
            return np.array([])
        
        # Generate synthetic daily returns based on position performance
        total_return = sum(pos.current_pnl for pos in positions)
        daily_return = total_return / days if days > 0 else 0
        
        return np.array([daily_return] * days)
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        
        if len(returns) == 0:
            return 0.0
        
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        
        return abs(np.min(drawdown))
    
    def _calculate_sortino_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sortino ratio"""
        
        if len(returns) == 0:
            return 0.0
        
        negative_returns = returns[returns < 0]
        if len(negative_returns) == 0:
            return float('inf') if np.mean(returns) > 0 else 0.0
        
        downside_deviation = np.std(negative_returns)
        
        return (np.mean(returns) * 252) / (downside_deviation * np.sqrt(252)) if downside_deviation > 0 else 0.0