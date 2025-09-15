"""
Currency overlay system for active FX management and alpha generation.
"""

import numpy as np
from datetime import date, datetime, timedelta
from typing import List, Dict, Tuple, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import math
from scipy import optimize, stats

from ..models.currency import CurrencyPair, CurrencyType
from .fx_hedge_calculator import FXHedgeCalculator, HedgeType


class OverlayStrategy(Enum):
    """Currency overlay strategies."""
    PASSIVE_HEDGE = "passive_hedge"
    ACTIVE_HEDGE = "active_hedge"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    CARRY_TRADE = "carry_trade"
    VOLATILITY_TARGET = "volatility_target"
    RISK_PARITY = "risk_parity"
    TACTICAL_ALLOCATION = "tactical_allocation"


class OverlaySignal(Enum):
    """Overlay trading signals."""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    NEUTRAL = "neutral"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


@dataclass
class OverlayPosition:
    """Currency overlay position."""
    currency_pair: str
    position_size: float  # Notional amount
    entry_rate: float
    current_rate: float
    entry_date: date
    strategy: OverlayStrategy
    signal_strength: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    @property
    def pnl(self) -> float:
        """Calculate position P&L."""
        return self.position_size * (self.current_rate - self.entry_rate)
    
    @property
    def pnl_percentage(self) -> float:
        """Calculate position P&L as percentage."""
        if self.entry_rate != 0:
            return (self.current_rate - self.entry_rate) / self.entry_rate
        return 0.0
    
    @property
    def days_held(self) -> int:
        """Calculate days position has been held."""
        return (date.today() - self.entry_date).days


@dataclass
class OverlayPerformance:
    """Currency overlay performance metrics."""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    average_win: float
    average_loss: float
    profit_factor: float
    calmar_ratio: float
    information_ratio: float
    tracking_error: float
    
    def __str__(self) -> str:
        return (f"Overlay Performance:\n"
                f"Total Return: {self.total_return:.2%}\n"
                f"Annualized Return: {self.annualized_return:.2%}\n"
                f"Volatility: {self.volatility:.2%}\n"
                f"Sharpe Ratio: {self.sharpe_ratio:.2f}\n"
                f"Max Drawdown: {self.max_drawdown:.2%}\n"
                f"Win Rate: {self.win_rate:.2%}\n"
                f"Information Ratio: {self.information_ratio:.2f}")


@dataclass
class OverlaySignalData:
    """Signal data for overlay strategies."""
    currency_pair: str
    signal: OverlaySignal
    signal_strength: float
    confidence: float
    expected_return: float
    expected_volatility: float
    time_horizon: int  # Days
    factors: Dict[str, float] = field(default_factory=dict)


class CurrencyOverlay:
    """
    Comprehensive currency overlay management system.
    """
    
    def __init__(self, 
                 base_currency: CurrencyType = CurrencyType.MAJOR,
                 risk_budget: float = 0.02,
                 max_position_size: float = 0.05):
        """
        Initialize currency overlay system.
        
        Args:
            base_currency: Base currency for calculations
            risk_budget: Maximum risk budget as fraction of portfolio
            max_position_size: Maximum position size as fraction of portfolio
        """
        self.base_currency = base_currency
        self.risk_budget = risk_budget
        self.max_position_size = max_position_size
        self.positions: List[OverlayPosition] = []
        self.performance_history: List[Dict] = []
        self.hedge_calculator = FXHedgeCalculator(base_currency)
    
    def generate_overlay_signals(self, 
                               currency_pairs: List[CurrencyPair],
                               strategy: OverlayStrategy,
                               lookback_days: int = 252) -> List[OverlaySignalData]:
        """
        Generate overlay trading signals for currency pairs.
        
        Args:
            currency_pairs: List of currency pairs to analyze
            strategy: Overlay strategy to use
            lookback_days: Historical data lookback period
            
        Returns:
            List of OverlaySignalData objects
        """
        signals = []
        
        for pair in currency_pairs:
            if strategy == OverlayStrategy.MOMENTUM:
                signal_data = self._generate_momentum_signal(pair, lookback_days)
            elif strategy == OverlayStrategy.MEAN_REVERSION:
                signal_data = self._generate_mean_reversion_signal(pair, lookback_days)
            elif strategy == OverlayStrategy.CARRY_TRADE:
                signal_data = self._generate_carry_trade_signal(pair)
            elif strategy == OverlayStrategy.VOLATILITY_TARGET:
                signal_data = self._generate_volatility_target_signal(pair, lookback_days)
            elif strategy == OverlayStrategy.TACTICAL_ALLOCATION:
                signal_data = self._generate_tactical_allocation_signal(pair, lookback_days)
            else:
                # Default neutral signal
                signal_data = OverlaySignalData(
                    currency_pair=f"{pair.base_currency.value}/{pair.quote_currency.value}",
                    signal=OverlaySignal.NEUTRAL,
                    signal_strength=0.0,
                    confidence=0.5,
                    expected_return=0.0,
                    expected_volatility=pair.volatility or 0.15,
                    time_horizon=30
                )
            
            signals.append(signal_data)
        
        return signals
    
    def execute_overlay_strategy(self, 
                               signals: List[OverlaySignalData],
                               portfolio_value: float,
                               current_positions: Optional[List[OverlayPosition]] = None) -> List[OverlayPosition]:
        """
        Execute overlay strategy based on signals.
        
        Args:
            signals: List of trading signals
            portfolio_value: Total portfolio value
            current_positions: Current overlay positions
            
        Returns:
            List of new/updated positions
        """
        if current_positions is None:
            current_positions = self.positions
        
        new_positions = []
        
        # Calculate available risk budget
        current_risk = self._calculate_portfolio_risk(current_positions)
        available_risk = max(0, self.risk_budget - current_risk)
        
        # Sort signals by strength and confidence
        sorted_signals = sorted(
            signals, 
            key=lambda s: abs(s.signal_strength) * s.confidence, 
            reverse=True
        )
        
        for signal in sorted_signals:
            if available_risk <= 0:
                break
            
            # Skip neutral signals
            if signal.signal == OverlaySignal.NEUTRAL:
                continue
            
            # Calculate position size
            position_size = self._calculate_position_size(
                signal, portfolio_value, available_risk
            )
            
            if abs(position_size) < portfolio_value * 0.001:  # Minimum position size
                continue
            
            # Create position
            position = OverlayPosition(
                currency_pair=signal.currency_pair,
                position_size=position_size,
                entry_rate=1.0,  # Would use current market rate
                current_rate=1.0,
                entry_date=date.today(),
                strategy=OverlayStrategy.TACTICAL_ALLOCATION,  # Default
                signal_strength=signal.signal_strength,
                stop_loss=self._calculate_stop_loss(signal),
                take_profit=self._calculate_take_profit(signal)
            )
            
            new_positions.append(position)
            
            # Update available risk
            position_risk = abs(position_size) * signal.expected_volatility
            available_risk -= position_risk
        
        return new_positions
    
    def calculate_overlay_performance(self, 
                                    positions: List[OverlayPosition],
                                    benchmark_returns: Optional[np.ndarray] = None,
                                    risk_free_rate: float = 0.02) -> OverlayPerformance:
        """
        Calculate overlay performance metrics.
        
        Args:
            positions: List of overlay positions
            benchmark_returns: Benchmark return series
            risk_free_rate: Risk-free rate for Sharpe ratio
            
        Returns:
            OverlayPerformance object
        """
        if not positions:
            return OverlayPerformance(
                total_return=0.0, annualized_return=0.0, volatility=0.0,
                sharpe_ratio=0.0, max_drawdown=0.0, win_rate=0.0,
                average_win=0.0, average_loss=0.0, profit_factor=0.0,
                calmar_ratio=0.0, information_ratio=0.0, tracking_error=0.0
            )
        
        # Calculate returns
        returns = np.array([pos.pnl_percentage for pos in positions])
        
        # Basic metrics
        total_return = np.sum(returns)
        
        # Annualized return (assuming positions held for average period)
        avg_days_held = np.mean([pos.days_held for pos in positions])
        if avg_days_held > 0:
            annualized_return = total_return * (365.25 / avg_days_held)
        else:
            annualized_return = 0.0
        
        volatility = np.std(returns) * math.sqrt(252) if len(returns) > 1 else 0.0
        
        # Sharpe ratio
        if volatility > 0:
            sharpe_ratio = (annualized_return - risk_free_rate) / volatility
        else:
            sharpe_ratio = 0.0
        
        # Max drawdown
        cumulative_returns = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = cumulative_returns - running_max
        max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0.0
        
        # Win/loss metrics
        winning_trades = returns[returns > 0]
        losing_trades = returns[returns < 0]
        
        win_rate = len(winning_trades) / len(returns) if len(returns) > 0 else 0.0
        average_win = np.mean(winning_trades) if len(winning_trades) > 0 else 0.0
        average_loss = np.mean(losing_trades) if len(losing_trades) > 0 else 0.0
        
        # Profit factor
        total_wins = np.sum(winning_trades) if len(winning_trades) > 0 else 0.0
        total_losses = abs(np.sum(losing_trades)) if len(losing_trades) > 0 else 0.0
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown < 0 else 0.0
        
        # Information ratio and tracking error
        if benchmark_returns is not None and len(benchmark_returns) == len(returns):
            excess_returns = returns - benchmark_returns
            tracking_error = np.std(excess_returns) * math.sqrt(252)
            information_ratio = np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) > 0 else 0.0
        else:
            tracking_error = 0.0
            information_ratio = 0.0
        
        return OverlayPerformance(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            average_win=average_win,
            average_loss=average_loss,
            profit_factor=profit_factor,
            calmar_ratio=calmar_ratio,
            information_ratio=information_ratio,
            tracking_error=tracking_error
        )
    
    def optimize_overlay_allocation(self, 
                                  signals: List[OverlaySignalData],
                                  portfolio_value: float,
                                  correlation_matrix: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Optimize overlay allocation across currency pairs.
        
        Args:
            signals: List of trading signals
            portfolio_value: Total portfolio value
            correlation_matrix: Currency correlation matrix
            
        Returns:
            Dictionary of optimal allocations by currency pair
        """
        n = len(signals)
        if n == 0:
            return {}
        
        # Expected returns and volatilities
        expected_returns = np.array([s.expected_return * s.confidence for s in signals])
        volatilities = np.array([s.expected_volatility for s in signals])
        
        # Use identity matrix if no correlation provided
        if correlation_matrix is None:
            correlation_matrix = np.eye(n)
        
        # Covariance matrix
        cov_matrix = np.outer(volatilities, volatilities) * correlation_matrix
        
        # Optimization objective: maximize Sharpe ratio
        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
            portfolio_vol = math.sqrt(max(0, portfolio_variance))
            
            if portfolio_vol > 0:
                sharpe_ratio = portfolio_return / portfolio_vol
                return -sharpe_ratio  # Minimize negative Sharpe
            else:
                return 0.0
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(np.abs(w)) - 1.0},  # Sum of absolute weights = 1
        ]
        
        # Bounds
        bounds = [(-self.max_position_size, self.max_position_size) for _ in range(n)]
        
        # Initial guess
        initial_guess = np.array([s.signal_strength * s.confidence for s in signals])
        if np.sum(np.abs(initial_guess)) > 0:
            initial_guess = initial_guess / np.sum(np.abs(initial_guess))
        else:
            initial_guess = np.ones(n) / n
        
        # Optimize
        try:
            result = optimize.minimize(
                objective,
                initial_guess,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if result.success:
                optimal_weights = result.x
            else:
                optimal_weights = initial_guess
                
        except Exception:
            optimal_weights = initial_guess
        
        # Convert to allocations
        allocations = {}
        for i, signal in enumerate(signals):
            allocation = optimal_weights[i] * portfolio_value * self.risk_budget
            allocations[signal.currency_pair] = allocation
        
        return allocations
    
    def _generate_momentum_signal(self, 
                                currency_pair: CurrencyPair,
                                lookback_days: int) -> OverlaySignalData:
        """Generate momentum-based signal."""
        # Simulate price data (in practice, would use real data)
        np.random.seed(hash(currency_pair.base_currency.value) % 2**32)
        
        # Generate returns with momentum
        returns = np.random.normal(0, currency_pair.volatility or 0.15, lookback_days)
        
        # Add momentum component
        momentum_factor = 0.1
        for i in range(1, len(returns)):
            returns[i] += momentum_factor * returns[i-1]
        
        # Calculate momentum indicators
        short_ma = np.mean(returns[-20:])  # 20-day average
        long_ma = np.mean(returns[-60:])   # 60-day average
        
        momentum_score = (short_ma - long_ma) / (currency_pair.volatility or 0.15)
        
        # Generate signal
        if momentum_score > 1.0:
            signal = OverlaySignal.STRONG_BUY
            signal_strength = min(1.0, momentum_score / 2.0)
        elif momentum_score > 0.5:
            signal = OverlaySignal.BUY
            signal_strength = momentum_score
        elif momentum_score < -1.0:
            signal = OverlaySignal.STRONG_SELL
            signal_strength = max(-1.0, momentum_score / 2.0)
        elif momentum_score < -0.5:
            signal = OverlaySignal.SELL
            signal_strength = momentum_score
        else:
            signal = OverlaySignal.NEUTRAL
            signal_strength = 0.0
        
        confidence = min(1.0, abs(momentum_score) / 2.0)
        expected_return = momentum_score * 0.02  # 2% per unit of momentum
        
        return OverlaySignalData(
            currency_pair=f"{currency_pair.base_currency.value}/{currency_pair.quote_currency.value}",
            signal=signal,
            signal_strength=signal_strength,
            confidence=confidence,
            expected_return=expected_return,
            expected_volatility=currency_pair.volatility or 0.15,
            time_horizon=30,
            factors={'momentum_score': momentum_score, 'short_ma': short_ma, 'long_ma': long_ma}
        )
    
    def _generate_mean_reversion_signal(self, 
                                      currency_pair: CurrencyPair,
                                      lookback_days: int) -> OverlaySignalData:
        """Generate mean reversion signal."""
        # Simulate price data
        np.random.seed(hash(currency_pair.quote_currency.value) % 2**32)
        
        prices = np.cumsum(np.random.normal(0, currency_pair.volatility or 0.15, lookback_days))
        
        # Calculate mean reversion indicators
        current_price = prices[-1]
        long_term_mean = np.mean(prices)
        std_dev = np.std(prices)
        
        z_score = (current_price - long_term_mean) / std_dev if std_dev > 0 else 0.0
        
        # Generate signal (contrarian)
        if z_score > 2.0:
            signal = OverlaySignal.STRONG_SELL
            signal_strength = -min(1.0, abs(z_score) / 3.0)
        elif z_score > 1.0:
            signal = OverlaySignal.SELL
            signal_strength = -z_score / 2.0
        elif z_score < -2.0:
            signal = OverlaySignal.STRONG_BUY
            signal_strength = min(1.0, abs(z_score) / 3.0)
        elif z_score < -1.0:
            signal = OverlaySignal.BUY
            signal_strength = -z_score / 2.0
        else:
            signal = OverlaySignal.NEUTRAL
            signal_strength = 0.0
        
        confidence = min(1.0, abs(z_score) / 2.0)
        expected_return = -z_score * 0.01  # Contrarian expectation
        
        return OverlaySignalData(
            currency_pair=f"{currency_pair.base_currency.value}/{currency_pair.quote_currency.value}",
            signal=signal,
            signal_strength=signal_strength,
            confidence=confidence,
            expected_return=expected_return,
            expected_volatility=currency_pair.volatility or 0.15,
            time_horizon=60,
            factors={'z_score': z_score, 'current_price': current_price, 'mean': long_term_mean}
        )
    
    def _generate_carry_trade_signal(self, currency_pair: CurrencyPair) -> OverlaySignalData:
        """Generate carry trade signal."""
        # Use interest rate differential
        base_rate = currency_pair.base_interest_rate or 0.02
        quote_rate = currency_pair.quote_interest_rate or 0.02
        
        carry = base_rate - quote_rate
        
        # Adjust for volatility (risk-adjusted carry)
        vol_adjusted_carry = carry / (currency_pair.volatility or 0.15)
        
        # Generate signal
        if vol_adjusted_carry > 0.5:
            signal = OverlaySignal.BUY
            signal_strength = min(1.0, vol_adjusted_carry)
        elif vol_adjusted_carry < -0.5:
            signal = OverlaySignal.SELL
            signal_strength = max(-1.0, vol_adjusted_carry)
        else:
            signal = OverlaySignal.NEUTRAL
            signal_strength = 0.0
        
        confidence = min(1.0, abs(vol_adjusted_carry))
        expected_return = carry * 0.8  # Carry minus some risk premium
        
        return OverlaySignalData(
            currency_pair=f"{currency_pair.base_currency.value}/{currency_pair.quote_currency.value}",
            signal=signal,
            signal_strength=signal_strength,
            confidence=confidence,
            expected_return=expected_return,
            expected_volatility=currency_pair.volatility or 0.15,
            time_horizon=90,
            factors={'carry': carry, 'vol_adjusted_carry': vol_adjusted_carry}
        )
    
    def _generate_volatility_target_signal(self, 
                                         currency_pair: CurrencyPair,
                                         lookback_days: int) -> OverlaySignalData:
        """Generate volatility targeting signal."""
        current_vol = currency_pair.volatility or 0.15
        target_vol = 0.12  # Target volatility
        
        vol_ratio = current_vol / target_vol
        
        # Scale position based on volatility
        if vol_ratio > 1.5:  # High volatility - reduce exposure
            signal = OverlaySignal.SELL
            signal_strength = -min(1.0, (vol_ratio - 1.0) / 2.0)
        elif vol_ratio < 0.7:  # Low volatility - increase exposure
            signal = OverlaySignal.BUY
            signal_strength = min(1.0, (1.0 - vol_ratio) / 0.5)
        else:
            signal = OverlaySignal.NEUTRAL
            signal_strength = 0.0
        
        confidence = 0.7  # Moderate confidence in vol targeting
        expected_return = 0.0  # Neutral return expectation
        
        return OverlaySignalData(
            currency_pair=f"{currency_pair.base_currency.value}/{currency_pair.quote_currency.value}",
            signal=signal,
            signal_strength=signal_strength,
            confidence=confidence,
            expected_return=expected_return,
            expected_volatility=target_vol,  # Target volatility
            time_horizon=30,
            factors={'current_vol': current_vol, 'target_vol': target_vol, 'vol_ratio': vol_ratio}
        )
    
    def _generate_tactical_allocation_signal(self, 
                                           currency_pair: CurrencyPair,
                                           lookback_days: int) -> OverlaySignalData:
        """Generate tactical allocation signal."""
        # Combine multiple factors
        momentum_signal = self._generate_momentum_signal(currency_pair, lookback_days)
        carry_signal = self._generate_carry_trade_signal(currency_pair)
        
        # Weighted combination
        combined_strength = (
            0.6 * momentum_signal.signal_strength + 
            0.4 * carry_signal.signal_strength
        )
        
        combined_confidence = (
            momentum_signal.confidence * carry_signal.confidence
        ) ** 0.5  # Geometric mean
        
        # Determine signal
        if combined_strength > 0.3:
            signal = OverlaySignal.BUY
        elif combined_strength < -0.3:
            signal = OverlaySignal.SELL
        else:
            signal = OverlaySignal.NEUTRAL
        
        expected_return = (
            0.6 * momentum_signal.expected_return + 
            0.4 * carry_signal.expected_return
        )
        
        return OverlaySignalData(
            currency_pair=f"{currency_pair.base_currency.value}/{currency_pair.quote_currency.value}",
            signal=signal,
            signal_strength=combined_strength,
            confidence=combined_confidence,
            expected_return=expected_return,
            expected_volatility=currency_pair.volatility or 0.15,
            time_horizon=45,
            factors={
                'momentum_component': momentum_signal.signal_strength,
                'carry_component': carry_signal.signal_strength,
                'combined_strength': combined_strength
            }
        )
    
    def _calculate_portfolio_risk(self, positions: List[OverlayPosition]) -> float:
        """Calculate current portfolio risk from positions."""
        if not positions:
            return 0.0
        
        # Simplified risk calculation
        total_risk = sum(abs(pos.position_size) * 0.15 for pos in positions)  # Assume 15% vol
        return total_risk
    
    def _calculate_position_size(self, 
                               signal: OverlaySignalData,
                               portfolio_value: float,
                               available_risk: float) -> float:
        """Calculate position size based on signal and risk budget."""
        # Risk-based position sizing
        signal_risk = signal.expected_volatility
        max_position_by_risk = available_risk / signal_risk if signal_risk > 0 else 0.0
        max_position_by_limit = portfolio_value * self.max_position_size
        
        max_position = min(max_position_by_risk, max_position_by_limit)
        
        # Scale by signal strength and confidence
        position_size = max_position * abs(signal.signal_strength) * signal.confidence
        
        # Apply sign
        if signal.signal in [OverlaySignal.SELL, OverlaySignal.STRONG_SELL]:
            position_size = -position_size
        
        return position_size
    
    def _calculate_stop_loss(self, signal: OverlaySignalData) -> Optional[float]:
        """Calculate stop loss level."""
        # Simple volatility-based stop loss
        stop_distance = 2.0 * signal.expected_volatility  # 2 standard deviations
        
        if signal.signal in [OverlaySignal.BUY, OverlaySignal.STRONG_BUY]:
            return 1.0 - stop_distance  # Stop below entry
        elif signal.signal in [OverlaySignal.SELL, OverlaySignal.STRONG_SELL]:
            return 1.0 + stop_distance  # Stop above entry
        else:
            return None
    
    def _calculate_take_profit(self, signal: OverlaySignalData) -> Optional[float]:
        """Calculate take profit level."""
        # Target based on expected return
        if abs(signal.expected_return) > 0.01:  # Only if meaningful expected return
            if signal.signal in [OverlaySignal.BUY, OverlaySignal.STRONG_BUY]:
                return 1.0 + abs(signal.expected_return)
            elif signal.signal in [OverlaySignal.SELL, OverlaySignal.STRONG_SELL]:
                return 1.0 - abs(signal.expected_return)
        
        return None