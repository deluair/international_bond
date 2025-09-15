"""
Momentum Strategy Module

This module implements momentum-based trading strategies for international bonds,
including trend following, breakout strategies, and momentum factor models.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

class MomentumType(Enum):
    """Types of momentum strategies"""
    PRICE_MOMENTUM = "price_momentum"
    YIELD_MOMENTUM = "yield_momentum"
    SPREAD_MOMENTUM = "spread_momentum"
    CROSS_SECTIONAL = "cross_sectional"
    TIME_SERIES = "time_series"
    BREAKOUT = "breakout"
    TREND_FOLLOWING = "trend_following"

class MomentumSignal(Enum):
    """Momentum signal types"""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    NEUTRAL = "neutral"
    SELL = "sell"
    STRONG_SELL = "strong_sell"

class TrendDirection(Enum):
    """Trend direction"""
    UPTREND = "uptrend"
    DOWNTREND = "downtrend"
    SIDEWAYS = "sideways"
    REVERSAL = "reversal"

class MomentumTimeframe(Enum):
    """Momentum calculation timeframes"""
    SHORT_TERM = "short_term"    # 1-4 weeks
    MEDIUM_TERM = "medium_term"  # 1-3 months
    LONG_TERM = "long_term"      # 3-12 months

@dataclass
class MomentumIndicator:
    """Momentum indicator calculation"""
    indicator_name: str
    timeframe: MomentumTimeframe
    lookback_period: int
    value: float
    percentile: float
    z_score: float
    signal_strength: float

@dataclass
class MomentumSignalData:
    """Momentum signal data for a bond"""
    bond_id: str
    signal_date: datetime
    momentum_type: MomentumType
    signal: MomentumSignal
    signal_strength: float
    trend_direction: TrendDirection
    indicators: List[MomentumIndicator]
    risk_metrics: Dict[str, float]
    confidence_score: float

@dataclass
class MomentumPosition:
    """Momentum-based position"""
    bond_id: str
    position_size: float
    entry_date: datetime
    entry_price: float
    entry_yield: float
    momentum_score: float
    stop_loss_level: Optional[float]
    take_profit_level: Optional[float]
    trailing_stop: Optional[float]

@dataclass
class MomentumPortfolio:
    """Momentum strategy portfolio"""
    portfolio_id: str
    strategy_type: MomentumType
    positions: Dict[str, MomentumPosition]
    total_exposure: float
    net_exposure: float
    momentum_score: float
    risk_metrics: Dict[str, float]
    performance_metrics: Dict[str, float]

@dataclass
class MomentumPerformance:
    """Performance tracking for momentum strategies"""
    strategy_id: str
    start_date: datetime
    end_date: datetime
    total_return: float
    momentum_return: float
    carry_return: float
    transaction_costs: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    avg_holding_period: float

class MomentumStrategy:
    """
    Comprehensive momentum strategy for international bonds
    """
    
    def __init__(
        self,
        strategy_name: str = "Momentum Strategy",
        short_lookback: int = 21,      # 1 month
        medium_lookback: int = 63,     # 3 months
        long_lookback: int = 252,      # 1 year
        momentum_threshold: float = 0.5,  # Minimum momentum score
        rebalance_frequency: int = 5,   # Rebalance every 5 days
        max_positions: int = 20,        # Maximum number of positions
        position_sizing_method: str = "equal_weight"
    ):
        self.strategy_name = strategy_name
        self.short_lookback = short_lookback
        self.medium_lookback = medium_lookback
        self.long_lookback = long_lookback
        self.momentum_threshold = momentum_threshold
        self.rebalance_frequency = rebalance_frequency
        self.max_positions = max_positions
        self.position_sizing_method = position_sizing_method
        
        # Strategy state
        self.current_portfolio: Optional[MomentumPortfolio] = None
        self.signal_history: List[MomentumSignalData] = []
        self.performance_history: List[MomentumPerformance] = []
        
        # Market data
        self.price_history: pd.DataFrame = pd.DataFrame()
        self.yield_history: pd.DataFrame = pd.DataFrame()
        self.spread_history: pd.DataFrame = pd.DataFrame()
    
    def calculate_momentum_indicators(
        self,
        bond_id: str,
        price_data: pd.Series,
        calculation_date: datetime
    ) -> List[MomentumIndicator]:
        """
        Calculate various momentum indicators for a bond
        """
        
        indicators = []
        
        # Price momentum indicators
        indicators.extend(self._calculate_price_momentum(bond_id, price_data, calculation_date))
        
        # Trend strength indicators
        indicators.extend(self._calculate_trend_indicators(bond_id, price_data, calculation_date))
        
        # Volatility-adjusted momentum
        indicators.extend(self._calculate_risk_adjusted_momentum(bond_id, price_data, calculation_date))
        
        # Relative momentum (cross-sectional)
        indicators.extend(self._calculate_relative_momentum(bond_id, price_data, calculation_date))
        
        return indicators
    
    def generate_momentum_signals(
        self,
        bond_universe: Dict[str, Dict[str, Any]],
        market_data: Dict[str, pd.DataFrame],
        signal_date: datetime
    ) -> List[MomentumSignalData]:
        """
        Generate momentum signals for the bond universe
        """
        
        signals = []
        
        for bond_id, bond_info in bond_universe.items():
            # Get price data
            price_data = market_data.get('prices', pd.DataFrame()).get(bond_id)
            if price_data is None or len(price_data) < self.long_lookback:
                continue
            
            # Calculate momentum indicators
            indicators = self.calculate_momentum_indicators(bond_id, price_data, signal_date)
            
            # Generate composite signal
            signal_data = self._generate_composite_signal(
                bond_id, bond_info, indicators, signal_date
            )
            
            if signal_data:
                signals.append(signal_data)
        
        # Rank signals by strength
        signals.sort(key=lambda x: x.signal_strength, reverse=True)
        
        return signals
    
    def construct_momentum_portfolio(
        self,
        signals: List[MomentumSignalData],
        current_portfolio: Optional[MomentumPortfolio],
        target_exposure: float = 1.0
    ) -> MomentumPortfolio:
        """
        Construct momentum portfolio based on signals
        """
        
        # Filter signals by minimum threshold
        qualified_signals = [
            signal for signal in signals
            if signal.signal_strength >= self.momentum_threshold
        ]
        
        # Select top signals
        selected_signals = qualified_signals[:self.max_positions]
        
        # Calculate position sizes
        positions = {}
        
        for signal in selected_signals:
            position_size = self._calculate_position_size(
                signal, target_exposure, len(selected_signals)
            )
            
            # Create position
            position = MomentumPosition(
                bond_id=signal.bond_id,
                position_size=position_size,
                entry_date=signal.signal_date,
                entry_price=0.0,  # Would be filled from market data
                entry_yield=0.0,  # Would be filled from market data
                momentum_score=signal.signal_strength,
                stop_loss_level=self._calculate_stop_loss(signal),
                take_profit_level=self._calculate_take_profit(signal),
                trailing_stop=self._calculate_trailing_stop(signal)
            )
            
            positions[signal.bond_id] = position
        
        # Calculate portfolio metrics
        total_exposure = sum(abs(pos.position_size) for pos in positions.values())
        net_exposure = sum(pos.position_size for pos in positions.values())
        momentum_score = np.mean([pos.momentum_score for pos in positions.values()]) if positions else 0.0
        
        # Calculate risk metrics
        risk_metrics = self._calculate_portfolio_risk_metrics(positions)
        
        portfolio = MomentumPortfolio(
            portfolio_id=f"momentum_{datetime.now().strftime('%Y%m%d')}",
            strategy_type=MomentumType.CROSS_SECTIONAL,
            positions=positions,
            total_exposure=total_exposure,
            net_exposure=net_exposure,
            momentum_score=momentum_score,
            risk_metrics=risk_metrics,
            performance_metrics={}
        )
        
        return portfolio
    
    def update_momentum_positions(
        self,
        current_portfolio: MomentumPortfolio,
        market_data: Dict[str, Any],
        update_date: datetime
    ) -> Dict[str, Dict[str, Any]]:
        """
        Update momentum positions and check exit conditions
        """
        
        position_updates = {}
        
        for bond_id, position in current_portfolio.positions.items():
            # Get current market data
            current_price = market_data.get('prices', {}).get(bond_id, position.entry_price)
            current_yield = market_data.get('yields', {}).get(bond_id, position.entry_yield)
            
            # Calculate current P&L
            price_pnl = (current_price - position.entry_price) * position.position_size
            
            # Check exit conditions
            exit_signal = self._check_momentum_exit_conditions(
                position, current_price, current_yield, update_date
            )
            
            # Update trailing stop
            new_trailing_stop = self._update_trailing_stop(
                position, current_price, update_date
            )
            
            # Calculate momentum decay
            momentum_decay = self._calculate_momentum_decay(position, update_date)
            
            position_updates[bond_id] = {
                'current_price': current_price,
                'current_yield': current_yield,
                'price_pnl': price_pnl,
                'exit_signal': exit_signal,
                'trailing_stop': new_trailing_stop,
                'momentum_decay': momentum_decay,
                'days_held': (update_date - position.entry_date).days
            }
        
        return position_updates
    
    def calculate_momentum_performance(
        self,
        portfolio: MomentumPortfolio,
        start_date: datetime,
        end_date: datetime,
        market_data: Dict[str, pd.DataFrame]
    ) -> MomentumPerformance:
        """
        Calculate performance metrics for momentum strategy
        """
        
        # Calculate returns for each position
        position_returns = []
        holding_periods = []
        
        for bond_id, position in portfolio.positions.items():
            # Get price data
            price_data = market_data.get('prices', pd.DataFrame()).get(bond_id)
            if price_data is None:
                continue
            
            # Calculate return
            entry_price = position.entry_price
            exit_price = price_data.iloc[-1] if len(price_data) > 0 else entry_price
            
            position_return = (exit_price - entry_price) / entry_price * position.position_size
            position_returns.append(position_return)
            
            # Calculate holding period
            holding_period = (end_date - position.entry_date).days
            holding_periods.append(holding_period)
        
        # Calculate aggregate metrics
        total_return = sum(position_returns) if position_returns else 0.0
        avg_holding_period = np.mean(holding_periods) if holding_periods else 0.0
        
        # Calculate Sharpe ratio (simplified)
        returns_std = np.std(position_returns) if len(position_returns) > 1 else 0.01
        sharpe_ratio = total_return / returns_std if returns_std > 0 else 0.0
        
        # Calculate win rate
        winning_positions = sum(1 for ret in position_returns if ret > 0)
        win_rate = winning_positions / len(position_returns) if position_returns else 0.0
        
        performance = MomentumPerformance(
            strategy_id=portfolio.portfolio_id,
            start_date=start_date,
            end_date=end_date,
            total_return=total_return,
            momentum_return=total_return * 0.8,  # Simplified
            carry_return=total_return * 0.2,     # Simplified
            transaction_costs=0.001 * len(portfolio.positions),  # Simplified
            sharpe_ratio=sharpe_ratio,
            max_drawdown=0.05,  # Placeholder
            win_rate=win_rate,
            avg_holding_period=avg_holding_period
        )
        
        return performance
    
    def _calculate_price_momentum(
        self,
        bond_id: str,
        price_data: pd.Series,
        calculation_date: datetime
    ) -> List[MomentumIndicator]:
        """Calculate price-based momentum indicators"""
        
        indicators = []
        
        if len(price_data) < self.long_lookback:
            return indicators
        
        current_price = price_data.iloc[-1]
        
        # Short-term momentum (1 month)
        if len(price_data) >= self.short_lookback:
            short_return = (current_price / price_data.iloc[-self.short_lookback] - 1)
            short_momentum = MomentumIndicator(
                indicator_name="price_momentum_1m",
                timeframe=MomentumTimeframe.SHORT_TERM,
                lookback_period=self.short_lookback,
                value=short_return,
                percentile=self._calculate_percentile(short_return, price_data, self.short_lookback),
                z_score=self._calculate_z_score(short_return, price_data, self.short_lookback),
                signal_strength=min(abs(short_return) * 10, 1.0)
            )
            indicators.append(short_momentum)
        
        # Medium-term momentum (3 months)
        if len(price_data) >= self.medium_lookback:
            medium_return = (current_price / price_data.iloc[-self.medium_lookback] - 1)
            medium_momentum = MomentumIndicator(
                indicator_name="price_momentum_3m",
                timeframe=MomentumTimeframe.MEDIUM_TERM,
                lookback_period=self.medium_lookback,
                value=medium_return,
                percentile=self._calculate_percentile(medium_return, price_data, self.medium_lookback),
                z_score=self._calculate_z_score(medium_return, price_data, self.medium_lookback),
                signal_strength=min(abs(medium_return) * 5, 1.0)
            )
            indicators.append(medium_momentum)
        
        # Long-term momentum (12 months)
        if len(price_data) >= self.long_lookback:
            long_return = (current_price / price_data.iloc[-self.long_lookback] - 1)
            long_momentum = MomentumIndicator(
                indicator_name="price_momentum_12m",
                timeframe=MomentumTimeframe.LONG_TERM,
                lookback_period=self.long_lookback,
                value=long_return,
                percentile=self._calculate_percentile(long_return, price_data, self.long_lookback),
                z_score=self._calculate_z_score(long_return, price_data, self.long_lookback),
                signal_strength=min(abs(long_return) * 2, 1.0)
            )
            indicators.append(long_momentum)
        
        return indicators
    
    def _calculate_trend_indicators(
        self,
        bond_id: str,
        price_data: pd.Series,
        calculation_date: datetime
    ) -> List[MomentumIndicator]:
        """Calculate trend strength indicators"""
        
        indicators = []
        
        if len(price_data) < 50:
            return indicators
        
        # Moving average crossover
        ma_short = price_data.rolling(20).mean().iloc[-1]
        ma_long = price_data.rolling(50).mean().iloc[-1]
        current_price = price_data.iloc[-1]
        
        # Trend strength based on MA position
        ma_signal = 0.0
        if current_price > ma_short > ma_long:
            ma_signal = 1.0  # Strong uptrend
        elif current_price > ma_short and ma_short < ma_long:
            ma_signal = 0.5  # Weak uptrend
        elif current_price < ma_short < ma_long:
            ma_signal = -1.0  # Strong downtrend
        elif current_price < ma_short and ma_short > ma_long:
            ma_signal = -0.5  # Weak downtrend
        
        ma_indicator = MomentumIndicator(
            indicator_name="moving_average_trend",
            timeframe=MomentumTimeframe.MEDIUM_TERM,
            lookback_period=50,
            value=ma_signal,
            percentile=50.0 + ma_signal * 25,  # Convert to percentile
            z_score=ma_signal,
            signal_strength=abs(ma_signal)
        )
        indicators.append(ma_indicator)
        
        # Price vs moving average distance
        ma_distance = (current_price - ma_long) / ma_long
        ma_distance_indicator = MomentumIndicator(
            indicator_name="ma_distance",
            timeframe=MomentumTimeframe.MEDIUM_TERM,
            lookback_period=50,
            value=ma_distance,
            percentile=self._calculate_percentile(ma_distance, price_data, 50),
            z_score=self._calculate_z_score(ma_distance, price_data, 50),
            signal_strength=min(abs(ma_distance) * 20, 1.0)
        )
        indicators.append(ma_distance_indicator)
        
        return indicators
    
    def _calculate_risk_adjusted_momentum(
        self,
        bond_id: str,
        price_data: pd.Series,
        calculation_date: datetime
    ) -> List[MomentumIndicator]:
        """Calculate risk-adjusted momentum indicators"""
        
        indicators = []
        
        if len(price_data) < self.medium_lookback:
            return indicators
        
        # Calculate returns
        returns = price_data.pct_change().dropna()
        
        if len(returns) < 20:
            return indicators
        
        # Sharpe ratio momentum
        recent_returns = returns.tail(self.medium_lookback)
        mean_return = recent_returns.mean()
        return_std = recent_returns.std()
        
        sharpe_momentum = mean_return / return_std if return_std > 0 else 0.0
        
        sharpe_indicator = MomentumIndicator(
            indicator_name="sharpe_momentum",
            timeframe=MomentumTimeframe.MEDIUM_TERM,
            lookback_period=self.medium_lookback,
            value=sharpe_momentum,
            percentile=self._calculate_percentile(sharpe_momentum, returns, self.medium_lookback),
            z_score=self._calculate_z_score(sharpe_momentum, returns, self.medium_lookback),
            signal_strength=min(abs(sharpe_momentum) * 2, 1.0)
        )
        indicators.append(sharpe_indicator)
        
        # Volatility-adjusted momentum
        vol_adj_momentum = mean_return / (return_std ** 0.5) if return_std > 0 else 0.0
        
        vol_adj_indicator = MomentumIndicator(
            indicator_name="volatility_adjusted_momentum",
            timeframe=MomentumTimeframe.MEDIUM_TERM,
            lookback_period=self.medium_lookback,
            value=vol_adj_momentum,
            percentile=self._calculate_percentile(vol_adj_momentum, returns, self.medium_lookback),
            z_score=self._calculate_z_score(vol_adj_momentum, returns, self.medium_lookback),
            signal_strength=min(abs(vol_adj_momentum) * 3, 1.0)
        )
        indicators.append(vol_adj_indicator)
        
        return indicators
    
    def _calculate_relative_momentum(
        self,
        bond_id: str,
        price_data: pd.Series,
        calculation_date: datetime
    ) -> List[MomentumIndicator]:
        """Calculate relative (cross-sectional) momentum indicators"""
        
        indicators = []
        
        # This would require universe data for proper cross-sectional ranking
        # For now, using simplified relative momentum
        
        if len(price_data) < self.medium_lookback:
            return indicators
        
        # Relative momentum vs own history
        current_return = price_data.pct_change(self.medium_lookback).iloc[-1]
        historical_returns = price_data.pct_change(self.medium_lookback).dropna()
        
        if len(historical_returns) < 10:
            return indicators
        
        # Percentile rank of current return
        percentile_rank = (historical_returns < current_return).mean() * 100
        
        relative_indicator = MomentumIndicator(
            indicator_name="relative_momentum",
            timeframe=MomentumTimeframe.MEDIUM_TERM,
            lookback_period=self.medium_lookback,
            value=current_return,
            percentile=percentile_rank,
            z_score=(current_return - historical_returns.mean()) / historical_returns.std(),
            signal_strength=abs(percentile_rank - 50) / 50
        )
        indicators.append(relative_indicator)
        
        return indicators
    
    def _generate_composite_signal(
        self,
        bond_id: str,
        bond_info: Dict[str, Any],
        indicators: List[MomentumIndicator],
        signal_date: datetime
    ) -> Optional[MomentumSignalData]:
        """Generate composite momentum signal from indicators"""
        
        if not indicators:
            return None
        
        # Calculate weighted signal strength
        total_weight = 0.0
        weighted_signal = 0.0
        
        timeframe_weights = {
            MomentumTimeframe.SHORT_TERM: 0.3,
            MomentumTimeframe.MEDIUM_TERM: 0.5,
            MomentumTimeframe.LONG_TERM: 0.2
        }
        
        for indicator in indicators:
            weight = timeframe_weights.get(indicator.timeframe, 0.3)
            weighted_signal += indicator.signal_strength * np.sign(indicator.value) * weight
            total_weight += weight
        
        if total_weight == 0:
            return None
        
        composite_signal_strength = weighted_signal / total_weight
        
        # Determine signal type
        if composite_signal_strength > 0.7:
            signal = MomentumSignal.STRONG_BUY
        elif composite_signal_strength > 0.3:
            signal = MomentumSignal.BUY
        elif composite_signal_strength < -0.7:
            signal = MomentumSignal.STRONG_SELL
        elif composite_signal_strength < -0.3:
            signal = MomentumSignal.SELL
        else:
            signal = MomentumSignal.NEUTRAL
        
        # Determine trend direction
        avg_value = np.mean([ind.value for ind in indicators])
        if avg_value > 0.02:
            trend_direction = TrendDirection.UPTREND
        elif avg_value < -0.02:
            trend_direction = TrendDirection.DOWNTREND
        else:
            trend_direction = TrendDirection.SIDEWAYS
        
        # Calculate risk metrics
        risk_metrics = self._calculate_signal_risk_metrics(indicators, bond_info)
        
        # Calculate confidence score
        confidence_score = self._calculate_signal_confidence(indicators, composite_signal_strength)
        
        signal_data = MomentumSignalData(
            bond_id=bond_id,
            signal_date=signal_date,
            momentum_type=MomentumType.CROSS_SECTIONAL,
            signal=signal,
            signal_strength=abs(composite_signal_strength),
            trend_direction=trend_direction,
            indicators=indicators,
            risk_metrics=risk_metrics,
            confidence_score=confidence_score
        )
        
        return signal_data
    
    def _calculate_position_size(
        self,
        signal: MomentumSignalData,
        target_exposure: float,
        num_positions: int
    ) -> float:
        """Calculate position size based on signal strength and risk"""
        
        if self.position_sizing_method == "equal_weight":
            base_size = target_exposure / num_positions
        elif self.position_sizing_method == "signal_weighted":
            base_size = target_exposure * signal.signal_strength / num_positions
        else:
            base_size = target_exposure / num_positions
        
        # Adjust for risk
        risk_adjustment = 1.0 / (1.0 + signal.risk_metrics.get('volatility', 0.1))
        
        # Adjust for confidence
        confidence_adjustment = signal.confidence_score
        
        position_size = base_size * risk_adjustment * confidence_adjustment
        
        # Apply signal direction
        if signal.signal in [MomentumSignal.SELL, MomentumSignal.STRONG_SELL]:
            position_size *= -1
        
        return position_size
    
    def _calculate_stop_loss(self, signal: MomentumSignalData) -> Optional[float]:
        """Calculate stop loss level for momentum position"""
        
        # Base stop loss on volatility
        volatility = signal.risk_metrics.get('volatility', 0.02)
        
        # Tighter stops for weaker signals
        stop_multiplier = 2.0 - signal.confidence_score
        
        stop_loss_distance = volatility * stop_multiplier
        
        return stop_loss_distance
    
    def _calculate_take_profit(self, signal: MomentumSignalData) -> Optional[float]:
        """Calculate take profit level for momentum position"""
        
        # Base take profit on expected momentum
        expected_return = signal.signal_strength * 0.05  # 5% max expected return
        
        # Adjust for confidence
        take_profit = expected_return * signal.confidence_score
        
        return max(take_profit, 0.01)  # Minimum 1% take profit
    
    def _calculate_trailing_stop(self, signal: MomentumSignalData) -> Optional[float]:
        """Calculate trailing stop for momentum position"""
        
        # Base trailing stop on volatility
        volatility = signal.risk_metrics.get('volatility', 0.02)
        
        trailing_stop = volatility * 1.5
        
        return trailing_stop
    
    def _calculate_portfolio_risk_metrics(
        self,
        positions: Dict[str, MomentumPosition]
    ) -> Dict[str, float]:
        """Calculate portfolio-level risk metrics"""
        
        if not positions:
            return {}
        
        # Calculate concentration risk
        position_weights = [abs(pos.position_size) for pos in positions.values()]
        total_weight = sum(position_weights)
        
        if total_weight > 0:
            normalized_weights = [w / total_weight for w in position_weights]
            concentration = sum(w ** 2 for w in normalized_weights)  # Herfindahl index
        else:
            concentration = 0.0
        
        # Calculate momentum concentration
        momentum_scores = [pos.momentum_score for pos in positions.values()]
        avg_momentum = np.mean(momentum_scores) if momentum_scores else 0.0
        momentum_std = np.std(momentum_scores) if len(momentum_scores) > 1 else 0.0
        
        return {
            'concentration_risk': concentration,
            'avg_momentum_score': avg_momentum,
            'momentum_dispersion': momentum_std,
            'num_positions': len(positions),
            'gross_exposure': sum(abs(pos.position_size) for pos in positions.values()),
            'net_exposure': sum(pos.position_size for pos in positions.values())
        }
    
    def _check_momentum_exit_conditions(
        self,
        position: MomentumPosition,
        current_price: float,
        current_yield: float,
        update_date: datetime
    ) -> bool:
        """Check if momentum position should be exited"""
        
        # Calculate current P&L
        price_change = (current_price - position.entry_price) / position.entry_price
        
        # Check stop loss
        if position.stop_loss_level:
            if position.position_size > 0 and price_change <= -position.stop_loss_level:
                return True
            elif position.position_size < 0 and price_change >= position.stop_loss_level:
                return True
        
        # Check take profit
        if position.take_profit_level:
            if position.position_size > 0 and price_change >= position.take_profit_level:
                return True
            elif position.position_size < 0 and price_change <= -position.take_profit_level:
                return True
        
        # Check trailing stop
        if position.trailing_stop:
            # Simplified trailing stop logic
            days_held = (update_date - position.entry_date).days
            if days_held > 30:  # Only apply after 30 days
                if abs(price_change) < position.trailing_stop:
                    return True
        
        # Check momentum decay (time-based exit)
        days_held = (update_date - position.entry_date).days
        if days_held > 90:  # Maximum holding period
            return True
        
        return False
    
    def _update_trailing_stop(
        self,
        position: MomentumPosition,
        current_price: float,
        update_date: datetime
    ) -> Optional[float]:
        """Update trailing stop level"""
        
        if not position.trailing_stop:
            return None
        
        # Calculate current P&L
        price_change = (current_price - position.entry_price) / position.entry_price
        
        # Update trailing stop if position is profitable
        if position.position_size > 0 and price_change > 0:
            # Tighten trailing stop as profit increases
            new_trailing_stop = position.trailing_stop * (1 - price_change * 0.5)
            return max(new_trailing_stop, position.trailing_stop * 0.5)
        elif position.position_size < 0 and price_change < 0:
            # Tighten trailing stop for short positions
            new_trailing_stop = position.trailing_stop * (1 + price_change * 0.5)
            return max(new_trailing_stop, position.trailing_stop * 0.5)
        
        return position.trailing_stop
    
    def _calculate_momentum_decay(
        self,
        position: MomentumPosition,
        update_date: datetime
    ) -> float:
        """Calculate momentum decay factor"""
        
        days_held = (update_date - position.entry_date).days
        
        # Momentum decays exponentially over time
        decay_rate = 0.02  # 2% per day
        momentum_decay = np.exp(-decay_rate * days_held)
        
        return momentum_decay
    
    def _calculate_signal_risk_metrics(
        self,
        indicators: List[MomentumIndicator],
        bond_info: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate risk metrics for momentum signal"""
        
        # Estimate volatility from bond characteristics
        duration = bond_info.get('duration', 5.0)
        rating = bond_info.get('rating', 'BBB')
        
        # Base volatility by rating
        rating_vol = {
            'AAA': 0.01, 'AA': 0.015, 'A': 0.02,
            'BBB': 0.025, 'BB': 0.04, 'B': 0.06
        }.get(rating, 0.03)
        
        # Adjust for duration
        duration_adj_vol = rating_vol * (1 + duration * 0.1)
        
        # Signal consistency (lower is better)
        signal_values = [ind.value for ind in indicators]
        signal_consistency = np.std(signal_values) if len(signal_values) > 1 else 0.0
        
        return {
            'volatility': duration_adj_vol,
            'duration_risk': duration,
            'signal_consistency': signal_consistency,
            'num_indicators': len(indicators)
        }
    
    def _calculate_signal_confidence(
        self,
        indicators: List[MomentumIndicator],
        composite_signal_strength: float
    ) -> float:
        """Calculate confidence score for momentum signal"""
        
        if not indicators:
            return 0.0
        
        # Base confidence from signal strength
        strength_confidence = min(abs(composite_signal_strength), 1.0)
        
        # Consistency bonus (all indicators pointing same direction)
        signal_directions = [np.sign(ind.value) for ind in indicators]
        consistency = abs(np.mean(signal_directions))
        
        # Number of indicators bonus
        indicator_bonus = min(len(indicators) / 5.0, 1.0)
        
        # Z-score strength bonus
        avg_z_score = np.mean([abs(ind.z_score) for ind in indicators])
        z_score_bonus = min(avg_z_score / 2.0, 0.5)
        
        confidence = (
            strength_confidence * 0.4 +
            consistency * 0.3 +
            indicator_bonus * 0.2 +
            z_score_bonus * 0.1
        )
        
        return max(0.0, min(1.0, confidence))
    
    def _calculate_percentile(
        self,
        value: float,
        data_series: pd.Series,
        lookback: int
    ) -> float:
        """Calculate percentile rank of value in historical data"""
        
        if len(data_series) < lookback:
            return 50.0
        
        historical_data = data_series.tail(lookback)
        percentile = (historical_data < value).mean() * 100
        
        return percentile
    
    def _calculate_z_score(
        self,
        value: float,
        data_series: pd.Series,
        lookback: int
    ) -> float:
        """Calculate z-score of value vs historical data"""
        
        if len(data_series) < lookback:
            return 0.0
        
        historical_data = data_series.tail(lookback)
        mean_val = historical_data.mean()
        std_val = historical_data.std()
        
        if std_val == 0:
            return 0.0
        
        z_score = (value - mean_val) / std_val
        
        return z_score