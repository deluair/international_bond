"""
Volatility Strategy Module

This module implements volatility-based trading strategies for international bonds,
including volatility arbitrage, volatility targeting, and volatility breakout strategies.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

class VolatilityStrategyType(Enum):
    """Types of volatility strategies"""
    VOLATILITY_TARGETING = "volatility_targeting"
    VOLATILITY_ARBITRAGE = "volatility_arbitrage"
    VOLATILITY_BREAKOUT = "volatility_breakout"
    VOLATILITY_MEAN_REVERSION = "volatility_mean_reversion"
    VOLATILITY_MOMENTUM = "volatility_momentum"
    VOLATILITY_CARRY = "volatility_carry"
    REGIME_SWITCHING = "regime_switching"

class VolatilitySignal(Enum):
    """Volatility signal types"""
    HIGH_VOL_BUY = "high_vol_buy"
    LOW_VOL_BUY = "low_vol_buy"
    NEUTRAL = "neutral"
    HIGH_VOL_SELL = "high_vol_sell"
    LOW_VOL_SELL = "low_vol_sell"
    VOL_BREAKOUT = "vol_breakout"

class VolatilityRegime(Enum):
    """Volatility regime classification"""
    LOW_VOLATILITY = "low_volatility"
    NORMAL_VOLATILITY = "normal_volatility"
    HIGH_VOLATILITY = "high_volatility"
    CRISIS_VOLATILITY = "crisis_volatility"
    TRANSITION = "transition"

class VolatilityMeasure(Enum):
    """Types of volatility measures"""
    REALIZED_VOLATILITY = "realized_volatility"
    GARCH_VOLATILITY = "garch_volatility"
    EWMA_VOLATILITY = "ewma_volatility"
    PARKINSON_VOLATILITY = "parkinson_volatility"
    GARMAN_KLASS_VOLATILITY = "garman_klass_volatility"
    RANGE_VOLATILITY = "range_volatility"

@dataclass
class VolatilityMetrics:
    """Comprehensive volatility metrics"""
    bond_id: str
    calculation_date: datetime
    realized_vol_1m: float
    realized_vol_3m: float
    realized_vol_6m: float
    realized_vol_1y: float
    garch_vol: float
    ewma_vol: float
    vol_of_vol: float
    vol_skew: float
    vol_percentile: float
    vol_z_score: float

@dataclass
class VolatilitySignalData:
    """Volatility signal data"""
    bond_id: str
    signal_date: datetime
    strategy_type: VolatilityStrategyType
    signal: VolatilitySignal
    volatility_regime: VolatilityRegime
    signal_strength: float
    volatility_metrics: VolatilityMetrics
    target_volatility: Optional[float]
    current_volatility: float
    vol_forecast: float
    confidence_score: float

@dataclass
class VolatilityPosition:
    """Volatility-based position"""
    bond_id: str
    position_size: float
    entry_date: datetime
    entry_price: float
    entry_volatility: float
    target_volatility: float
    volatility_beta: float
    vol_carry: float
    dynamic_hedge_ratio: float
    rebalance_threshold: float

@dataclass
class VolatilityPortfolio:
    """Volatility strategy portfolio"""
    portfolio_id: str
    strategy_type: VolatilityStrategyType
    positions: Dict[str, VolatilityPosition]
    target_portfolio_vol: float
    current_portfolio_vol: float
    vol_budget_allocation: Dict[str, float]
    total_exposure: float
    net_exposure: float
    volatility_metrics: Dict[str, float]
    performance_metrics: Dict[str, float]

@dataclass
class VolatilityPerformance:
    """Performance tracking for volatility strategies"""
    strategy_id: str
    start_date: datetime
    end_date: datetime
    total_return: float
    volatility_alpha: float
    vol_adjusted_return: float
    realized_volatility: float
    target_volatility: float
    vol_tracking_error: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    vol_timing_skill: float

class VolatilityStrategy:
    """
    Comprehensive volatility strategy for international bonds
    """
    
    def __init__(
        self,
        strategy_name: str = "Volatility Strategy",
        target_volatility: float = 0.10,      # 10% target volatility
        vol_lookback_short: int = 21,         # 1 month
        vol_lookback_medium: int = 63,        # 3 months
        vol_lookback_long: int = 252,         # 1 year
        rebalance_frequency: int = 5,         # Rebalance every 5 days
        vol_threshold: float = 0.02,          # 2% vol threshold for signals
        max_positions: int = 20,              # Maximum number of positions
        garch_params: Dict[str, float] = None,
        ewma_lambda: float = 0.94
    ):
        self.strategy_name = strategy_name
        self.target_volatility = target_volatility
        self.vol_lookback_short = vol_lookback_short
        self.vol_lookback_medium = vol_lookback_medium
        self.vol_lookback_long = vol_lookback_long
        self.rebalance_frequency = rebalance_frequency
        self.vol_threshold = vol_threshold
        self.max_positions = max_positions
        self.ewma_lambda = ewma_lambda
        
        # GARCH parameters
        self.garch_params = garch_params or {
            'omega': 0.000001,
            'alpha': 0.05,
            'beta': 0.90
        }
        
        # Strategy state
        self.current_portfolio: Optional[VolatilityPortfolio] = None
        self.signal_history: List[VolatilitySignalData] = []
        self.performance_history: List[VolatilityPerformance] = []
        
        # Volatility models
        self.garch_models: Dict[str, Dict[str, Any]] = {}
        self.vol_forecasts: Dict[str, pd.Series] = {}
        self.regime_models: Dict[str, Dict[str, Any]] = {}
    
    def calculate_volatility_metrics(
        self,
        bond_id: str,
        price_data: pd.Series,
        high_data: Optional[pd.Series] = None,
        low_data: Optional[pd.Series] = None,
        calculation_date: datetime = None
    ) -> VolatilityMetrics:
        """
        Calculate comprehensive volatility metrics for a bond
        """
        
        if len(price_data) < self.vol_lookback_long:
            # Return default metrics if insufficient data
            return self._create_default_vol_metrics(bond_id, calculation_date)
        
        # Calculate returns
        returns = price_data.pct_change().dropna()
        
        # Realized volatilities (annualized)
        realized_vol_1m = self._calculate_realized_volatility(returns, self.vol_lookback_short)
        realized_vol_3m = self._calculate_realized_volatility(returns, self.vol_lookback_medium)
        realized_vol_6m = self._calculate_realized_volatility(returns, min(126, len(returns)))
        realized_vol_1y = self._calculate_realized_volatility(returns, self.vol_lookback_long)
        
        # GARCH volatility
        garch_vol = self._calculate_garch_volatility(bond_id, returns)
        
        # EWMA volatility
        ewma_vol = self._calculate_ewma_volatility(returns)
        
        # Volatility of volatility
        vol_of_vol = self._calculate_vol_of_vol(returns)
        
        # Volatility skew
        vol_skew = self._calculate_vol_skew(returns)
        
        # Volatility percentile and z-score
        vol_percentile = self._calculate_vol_percentile(realized_vol_1m, returns)
        vol_z_score = self._calculate_vol_z_score(realized_vol_1m, returns)
        
        metrics = VolatilityMetrics(
            bond_id=bond_id,
            calculation_date=calculation_date or datetime.now(),
            realized_vol_1m=realized_vol_1m,
            realized_vol_3m=realized_vol_3m,
            realized_vol_6m=realized_vol_6m,
            realized_vol_1y=realized_vol_1y,
            garch_vol=garch_vol,
            ewma_vol=ewma_vol,
            vol_of_vol=vol_of_vol,
            vol_skew=vol_skew,
            vol_percentile=vol_percentile,
            vol_z_score=vol_z_score
        )
        
        return metrics
    
    def identify_volatility_regime(
        self,
        volatility_metrics: VolatilityMetrics,
        market_context: Dict[str, Any] = None
    ) -> VolatilityRegime:
        """
        Identify current volatility regime
        """
        
        current_vol = volatility_metrics.realized_vol_1m
        vol_percentile = volatility_metrics.vol_percentile
        vol_z_score = volatility_metrics.vol_z_score
        
        # Define regime thresholds
        if vol_percentile < 20 and vol_z_score < -1.0:
            return VolatilityRegime.LOW_VOLATILITY
        elif vol_percentile > 80 and vol_z_score > 2.0:
            if current_vol > 0.25:  # 25% volatility threshold for crisis
                return VolatilityRegime.CRISIS_VOLATILITY
            else:
                return VolatilityRegime.HIGH_VOLATILITY
        elif 20 <= vol_percentile <= 80:
            return VolatilityRegime.NORMAL_VOLATILITY
        else:
            return VolatilityRegime.TRANSITION
    
    def generate_volatility_signals(
        self,
        bond_universe: Dict[str, Dict[str, Any]],
        market_data: Dict[str, pd.DataFrame],
        signal_date: datetime
    ) -> List[VolatilitySignalData]:
        """
        Generate volatility-based signals for the bond universe
        """
        
        signals = []
        
        for bond_id, bond_info in bond_universe.items():
            # Get price data
            price_data = market_data.get('prices', pd.DataFrame()).get(bond_id)
            high_data = market_data.get('highs', pd.DataFrame()).get(bond_id)
            low_data = market_data.get('lows', pd.DataFrame()).get(bond_id)
            
            if price_data is None or len(price_data) < self.vol_lookback_long:
                continue
            
            # Calculate volatility metrics
            vol_metrics = self.calculate_volatility_metrics(
                bond_id, price_data, high_data, low_data, signal_date
            )
            
            # Identify volatility regime
            vol_regime = self.identify_volatility_regime(vol_metrics)
            
            # Generate signals for different strategies
            signals.extend(self._generate_vol_targeting_signals(bond_id, vol_metrics, vol_regime, signal_date))
            signals.extend(self._generate_vol_arbitrage_signals(bond_id, vol_metrics, vol_regime, signal_date))
            signals.extend(self._generate_vol_breakout_signals(bond_id, vol_metrics, vol_regime, signal_date))
            signals.extend(self._generate_vol_mean_reversion_signals(bond_id, vol_metrics, vol_regime, signal_date))
        
        # Sort by signal strength
        signals.sort(key=lambda x: x.signal_strength, reverse=True)
        
        return signals
    
    def construct_volatility_portfolio(
        self,
        signals: List[VolatilitySignalData],
        strategy_type: VolatilityStrategyType,
        current_portfolio: Optional[VolatilityPortfolio],
        target_exposure: float = 1.0
    ) -> VolatilityPortfolio:
        """
        Construct volatility-based portfolio
        """
        
        # Filter signals by strategy type
        strategy_signals = [s for s in signals if s.strategy_type == strategy_type]
        
        # Select top signals
        selected_signals = strategy_signals[:self.max_positions]
        
        # Calculate volatility budget allocation
        vol_budget = self._calculate_volatility_budget(selected_signals, strategy_type)
        
        # Create positions
        positions = {}
        
        for signal in selected_signals:
            position = self._create_volatility_position(
                signal, vol_budget, target_exposure, len(selected_signals)
            )
            positions[signal.bond_id] = position
        
        # Calculate portfolio metrics
        portfolio_metrics = self._calculate_portfolio_volatility_metrics(positions)
        
        portfolio = VolatilityPortfolio(
            portfolio_id=f"vol_{strategy_type.value}_{datetime.now().strftime('%Y%m%d')}",
            strategy_type=strategy_type,
            positions=positions,
            target_portfolio_vol=self.target_volatility,
            current_portfolio_vol=portfolio_metrics['portfolio_vol'],
            vol_budget_allocation=vol_budget,
            total_exposure=portfolio_metrics['total_exposure'],
            net_exposure=portfolio_metrics['net_exposure'],
            volatility_metrics=portfolio_metrics,
            performance_metrics={}
        )
        
        return portfolio
    
    def rebalance_volatility_portfolio(
        self,
        current_portfolio: VolatilityPortfolio,
        market_data: Dict[str, Any],
        rebalance_date: datetime
    ) -> Dict[str, Dict[str, Any]]:
        """
        Rebalance volatility portfolio based on current market conditions
        """
        
        rebalance_actions = {}
        
        # Update volatility metrics for each position
        for bond_id, position in current_portfolio.positions.items():
            current_price = market_data.get('prices', {}).get(bond_id, position.entry_price)
            
            # Recalculate volatility metrics
            price_data = market_data.get('price_history', {}).get(bond_id)
            if price_data is not None:
                vol_metrics = self.calculate_volatility_metrics(bond_id, price_data, calculation_date=rebalance_date)
                current_vol = vol_metrics.realized_vol_1m
            else:
                current_vol = position.entry_volatility
            
            # Calculate required position adjustment
            vol_adjustment = self._calculate_volatility_adjustment(
                position, current_vol, current_portfolio.target_portfolio_vol
            )
            
            # Check rebalance threshold
            if abs(vol_adjustment) > position.rebalance_threshold:
                new_position_size = position.position_size * (1 + vol_adjustment)
                
                rebalance_actions[bond_id] = {
                    'action': 'rebalance',
                    'current_size': position.position_size,
                    'new_size': new_position_size,
                    'adjustment': vol_adjustment,
                    'current_vol': current_vol,
                    'target_vol': position.target_volatility,
                    'vol_contribution': self._calculate_vol_contribution(position, current_vol)
                }
        
        # Check overall portfolio volatility
        current_portfolio_vol = self._estimate_current_portfolio_volatility(
            current_portfolio, market_data, rebalance_date
        )
        
        portfolio_vol_adjustment = (current_portfolio.target_portfolio_vol - current_portfolio_vol) / current_portfolio_vol
        
        # Apply portfolio-level adjustment if needed
        if abs(portfolio_vol_adjustment) > 0.1:  # 10% threshold
            for bond_id in rebalance_actions:
                rebalance_actions[bond_id]['portfolio_adjustment'] = portfolio_vol_adjustment
                rebalance_actions[bond_id]['new_size'] *= (1 + portfolio_vol_adjustment * 0.5)
        
        return rebalance_actions
    
    def calculate_volatility_performance(
        self,
        portfolio: VolatilityPortfolio,
        start_date: datetime,
        end_date: datetime,
        market_data: Dict[str, pd.DataFrame]
    ) -> VolatilityPerformance:
        """
        Calculate performance metrics for volatility strategy
        """
        
        # Calculate returns for each position
        position_returns = []
        realized_vols = []
        
        for bond_id, position in portfolio.positions.items():
            price_data = market_data.get('prices', pd.DataFrame()).get(bond_id)
            if price_data is None:
                continue
            
            # Calculate position return
            entry_price = position.entry_price
            exit_price = price_data.iloc[-1] if len(price_data) > 0 else entry_price
            
            position_return = (exit_price - entry_price) / entry_price * position.position_size
            position_returns.append(position_return)
            
            # Calculate realized volatility
            returns = price_data.pct_change().dropna()
            if len(returns) > 0:
                realized_vol = returns.std() * np.sqrt(252)
                realized_vols.append(realized_vol)
        
        # Calculate aggregate metrics
        total_return = sum(position_returns) if position_returns else 0.0
        avg_realized_vol = np.mean(realized_vols) if realized_vols else 0.0
        
        # Volatility tracking error
        vol_tracking_error = abs(avg_realized_vol - portfolio.target_portfolio_vol)
        
        # Volatility-adjusted return
        vol_adjusted_return = total_return / max(avg_realized_vol, 0.01)
        
        # Sharpe ratio
        returns_std = np.std(position_returns) if len(position_returns) > 1 else 0.01
        sharpe_ratio = total_return / returns_std if returns_std > 0 else 0.0
        
        # Sortino ratio (simplified)
        negative_returns = [r for r in position_returns if r < 0]
        downside_std = np.std(negative_returns) if negative_returns else returns_std
        sortino_ratio = total_return / downside_std if downside_std > 0 else 0.0
        
        # Volatility timing skill (simplified)
        vol_timing_skill = self._calculate_vol_timing_skill(portfolio, market_data)
        
        performance = VolatilityPerformance(
            strategy_id=portfolio.portfolio_id,
            start_date=start_date,
            end_date=end_date,
            total_return=total_return,
            volatility_alpha=total_return * 0.6,  # Simplified
            vol_adjusted_return=vol_adjusted_return,
            realized_volatility=avg_realized_vol,
            target_volatility=portfolio.target_portfolio_vol,
            vol_tracking_error=vol_tracking_error,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=0.05,  # Placeholder
            vol_timing_skill=vol_timing_skill
        )
        
        return performance
    
    def _calculate_realized_volatility(self, returns: pd.Series, lookback: int) -> float:
        """Calculate realized volatility (annualized)"""
        
        if len(returns) < lookback:
            lookback = len(returns)
        
        if lookback < 2:
            return 0.0
        
        recent_returns = returns.tail(lookback)
        volatility = recent_returns.std() * np.sqrt(252)  # Annualized
        
        return volatility
    
    def _calculate_garch_volatility(self, bond_id: str, returns: pd.Series) -> float:
        """Calculate GARCH(1,1) volatility forecast"""
        
        if len(returns) < 50:
            return self._calculate_realized_volatility(returns, len(returns))
        
        # Simplified GARCH(1,1) implementation
        # In practice, would use arch library
        
        omega = self.garch_params['omega']
        alpha = self.garch_params['alpha']
        beta = self.garch_params['beta']
        
        # Initialize
        returns_squared = returns ** 2
        
        # Use sample variance as initial condition
        long_run_var = returns_squared.mean()
        
        # Calculate conditional variance
        if bond_id in self.garch_models:
            prev_variance = self.garch_models[bond_id].get('last_variance', long_run_var)
            prev_return_sq = self.garch_models[bond_id].get('last_return_sq', long_run_var)
        else:
            prev_variance = long_run_var
            prev_return_sq = returns_squared.iloc[-1] if len(returns_squared) > 0 else long_run_var
        
        # GARCH(1,1) forecast
        next_variance = omega + alpha * prev_return_sq + beta * prev_variance
        
        # Store for next iteration
        self.garch_models[bond_id] = {
            'last_variance': next_variance,
            'last_return_sq': returns_squared.iloc[-1] if len(returns_squared) > 0 else long_run_var,
            'long_run_var': long_run_var
        }
        
        garch_vol = np.sqrt(next_variance * 252)  # Annualized
        
        return garch_vol
    
    def _calculate_ewma_volatility(self, returns: pd.Series) -> float:
        """Calculate EWMA volatility"""
        
        if len(returns) < 10:
            return self._calculate_realized_volatility(returns, len(returns))
        
        # EWMA calculation
        returns_squared = returns ** 2
        
        # Initialize with first observation
        ewma_var = returns_squared.iloc[0]
        
        # Update with EWMA
        for ret_sq in returns_squared.iloc[1:]:
            ewma_var = self.ewma_lambda * ewma_var + (1 - self.ewma_lambda) * ret_sq
        
        ewma_vol = np.sqrt(ewma_var * 252)  # Annualized
        
        return ewma_vol
    
    def _calculate_vol_of_vol(self, returns: pd.Series) -> float:
        """Calculate volatility of volatility"""
        
        if len(returns) < 50:
            return 0.0
        
        # Calculate rolling volatility
        vol_window = 20
        rolling_vol = returns.rolling(vol_window).std() * np.sqrt(252)
        
        # Calculate volatility of the rolling volatility
        vol_of_vol = rolling_vol.std()
        
        return vol_of_vol
    
    def _calculate_vol_skew(self, returns: pd.Series) -> float:
        """Calculate volatility skew"""
        
        if len(returns) < 50:
            return 0.0
        
        # Calculate rolling volatility
        vol_window = 20
        rolling_vol = returns.rolling(vol_window).std()
        
        # Calculate skewness of volatility distribution
        vol_skew = rolling_vol.skew()
        
        return vol_skew if not np.isnan(vol_skew) else 0.0
    
    def _calculate_vol_percentile(self, current_vol: float, returns: pd.Series) -> float:
        """Calculate volatility percentile rank"""
        
        if len(returns) < 50:
            return 50.0
        
        # Calculate historical volatilities
        vol_window = 20
        historical_vols = returns.rolling(vol_window).std() * np.sqrt(252)
        historical_vols = historical_vols.dropna()
        
        if len(historical_vols) == 0:
            return 50.0
        
        # Calculate percentile
        percentile = (historical_vols < current_vol).mean() * 100
        
        return percentile
    
    def _calculate_vol_z_score(self, current_vol: float, returns: pd.Series) -> float:
        """Calculate volatility z-score"""
        
        if len(returns) < 50:
            return 0.0
        
        # Calculate historical volatilities
        vol_window = 20
        historical_vols = returns.rolling(vol_window).std() * np.sqrt(252)
        historical_vols = historical_vols.dropna()
        
        if len(historical_vols) == 0:
            return 0.0
        
        mean_vol = historical_vols.mean()
        std_vol = historical_vols.std()
        
        if std_vol == 0:
            return 0.0
        
        z_score = (current_vol - mean_vol) / std_vol
        
        return z_score
    
    def _generate_vol_targeting_signals(
        self,
        bond_id: str,
        vol_metrics: VolatilityMetrics,
        vol_regime: VolatilityRegime,
        signal_date: datetime
    ) -> List[VolatilitySignalData]:
        """Generate volatility targeting signals"""
        
        signals = []
        
        current_vol = vol_metrics.realized_vol_1m
        vol_forecast = vol_metrics.garch_vol
        
        # Calculate volatility gap
        vol_gap = current_vol - self.target_volatility
        
        # Generate signal based on volatility gap
        if abs(vol_gap) > self.vol_threshold:
            if vol_gap > 0:
                # Current vol too high - reduce exposure
                signal = VolatilitySignal.HIGH_VOL_SELL
                signal_strength = min(abs(vol_gap) / self.target_volatility, 1.0)
            else:
                # Current vol too low - increase exposure
                signal = VolatilitySignal.LOW_VOL_BUY
                signal_strength = min(abs(vol_gap) / self.target_volatility, 1.0)
            
            confidence_score = self._calculate_vol_signal_confidence(vol_metrics, vol_regime)
            
            signal_data = VolatilitySignalData(
                bond_id=bond_id,
                signal_date=signal_date,
                strategy_type=VolatilityStrategyType.VOLATILITY_TARGETING,
                signal=signal,
                volatility_regime=vol_regime,
                signal_strength=signal_strength,
                volatility_metrics=vol_metrics,
                target_volatility=self.target_volatility,
                current_volatility=current_vol,
                vol_forecast=vol_forecast,
                confidence_score=confidence_score
            )
            
            signals.append(signal_data)
        
        return signals
    
    def _generate_vol_arbitrage_signals(
        self,
        bond_id: str,
        vol_metrics: VolatilityMetrics,
        vol_regime: VolatilityRegime,
        signal_date: datetime
    ) -> List[VolatilitySignalData]:
        """Generate volatility arbitrage signals"""
        
        signals = []
        
        realized_vol = vol_metrics.realized_vol_1m
        garch_vol = vol_metrics.garch_vol
        ewma_vol = vol_metrics.ewma_vol
        
        # Compare different volatility measures
        vol_spread_garch = realized_vol - garch_vol
        vol_spread_ewma = realized_vol - ewma_vol
        
        # Generate arbitrage signal
        if abs(vol_spread_garch) > self.vol_threshold:
            if vol_spread_garch > 0:
                # Realized vol > GARCH forecast - expect vol to decrease
                signal = VolatilitySignal.HIGH_VOL_SELL
            else:
                # Realized vol < GARCH forecast - expect vol to increase
                signal = VolatilitySignal.LOW_VOL_BUY
            
            signal_strength = min(abs(vol_spread_garch) / realized_vol, 1.0)
            confidence_score = self._calculate_vol_signal_confidence(vol_metrics, vol_regime)
            
            signal_data = VolatilitySignalData(
                bond_id=bond_id,
                signal_date=signal_date,
                strategy_type=VolatilityStrategyType.VOLATILITY_ARBITRAGE,
                signal=signal,
                volatility_regime=vol_regime,
                signal_strength=signal_strength,
                volatility_metrics=vol_metrics,
                target_volatility=None,
                current_volatility=realized_vol,
                vol_forecast=garch_vol,
                confidence_score=confidence_score
            )
            
            signals.append(signal_data)
        
        return signals
    
    def _generate_vol_breakout_signals(
        self,
        bond_id: str,
        vol_metrics: VolatilityMetrics,
        vol_regime: VolatilityRegime,
        signal_date: datetime
    ) -> List[VolatilitySignalData]:
        """Generate volatility breakout signals"""
        
        signals = []
        
        vol_z_score = vol_metrics.vol_z_score
        vol_percentile = vol_metrics.vol_percentile
        
        # Detect volatility breakouts
        if vol_z_score > 2.0 or vol_percentile > 90:
            # High volatility breakout
            signal = VolatilitySignal.VOL_BREAKOUT
            signal_strength = min(vol_z_score / 2.0, 1.0)
            
            confidence_score = self._calculate_vol_signal_confidence(vol_metrics, vol_regime)
            
            signal_data = VolatilitySignalData(
                bond_id=bond_id,
                signal_date=signal_date,
                strategy_type=VolatilityStrategyType.VOLATILITY_BREAKOUT,
                signal=signal,
                volatility_regime=vol_regime,
                signal_strength=signal_strength,
                volatility_metrics=vol_metrics,
                target_volatility=None,
                current_volatility=vol_metrics.realized_vol_1m,
                vol_forecast=vol_metrics.garch_vol,
                confidence_score=confidence_score
            )
            
            signals.append(signal_data)
        
        return signals
    
    def _generate_vol_mean_reversion_signals(
        self,
        bond_id: str,
        vol_metrics: VolatilityMetrics,
        vol_regime: VolatilityRegime,
        signal_date: datetime
    ) -> List[VolatilitySignalData]:
        """Generate volatility mean reversion signals"""
        
        signals = []
        
        vol_z_score = vol_metrics.vol_z_score
        current_vol = vol_metrics.realized_vol_1m
        long_term_vol = vol_metrics.realized_vol_1y
        
        # Mean reversion signal
        if abs(vol_z_score) > 1.5:
            if vol_z_score > 0:
                # Vol too high - expect reversion down
                signal = VolatilitySignal.HIGH_VOL_SELL
            else:
                # Vol too low - expect reversion up
                signal = VolatilitySignal.LOW_VOL_BUY
            
            signal_strength = min(abs(vol_z_score) / 2.0, 1.0)
            confidence_score = self._calculate_vol_signal_confidence(vol_metrics, vol_regime)
            
            signal_data = VolatilitySignalData(
                bond_id=bond_id,
                signal_date=signal_date,
                strategy_type=VolatilityStrategyType.VOLATILITY_MEAN_REVERSION,
                signal=signal,
                volatility_regime=vol_regime,
                signal_strength=signal_strength,
                volatility_metrics=vol_metrics,
                target_volatility=long_term_vol,
                current_volatility=current_vol,
                vol_forecast=vol_metrics.garch_vol,
                confidence_score=confidence_score
            )
            
            signals.append(signal_data)
        
        return signals
    
    def _calculate_volatility_budget(
        self,
        signals: List[VolatilitySignalData],
        strategy_type: VolatilityStrategyType
    ) -> Dict[str, float]:
        """Calculate volatility budget allocation"""
        
        vol_budget = {}
        
        if not signals:
            return vol_budget
        
        total_signal_strength = sum(signal.signal_strength for signal in signals)
        
        if total_signal_strength == 0:
            # Equal allocation
            allocation = 1.0 / len(signals)
            for signal in signals:
                vol_budget[signal.bond_id] = allocation
        else:
            # Signal-weighted allocation
            for signal in signals:
                vol_budget[signal.bond_id] = signal.signal_strength / total_signal_strength
        
        return vol_budget
    
    def _create_volatility_position(
        self,
        signal: VolatilitySignalData,
        vol_budget: Dict[str, float],
        target_exposure: float,
        num_positions: int
    ) -> VolatilityPosition:
        """Create volatility position from signal"""
        
        # Base position size
        budget_allocation = vol_budget.get(signal.bond_id, 1.0 / num_positions)
        base_size = target_exposure * budget_allocation
        
        # Adjust for volatility
        vol_adjustment = self.target_volatility / max(signal.current_volatility, 0.01)
        position_size = base_size * vol_adjustment
        
        # Apply signal direction
        if signal.signal in [VolatilitySignal.HIGH_VOL_SELL]:
            position_size *= -0.5  # Reduce exposure
        elif signal.signal in [VolatilitySignal.LOW_VOL_BUY]:
            position_size *= 1.5   # Increase exposure
        
        # Calculate volatility beta (simplified)
        volatility_beta = signal.current_volatility / self.target_volatility
        
        # Calculate volatility carry
        vol_carry = signal.current_volatility - signal.vol_forecast
        
        position = VolatilityPosition(
            bond_id=signal.bond_id,
            position_size=position_size,
            entry_date=signal.signal_date,
            entry_price=100.0,  # Would be filled from market data
            entry_volatility=signal.current_volatility,
            target_volatility=signal.target_volatility or self.target_volatility,
            volatility_beta=volatility_beta,
            vol_carry=vol_carry,
            dynamic_hedge_ratio=1.0,  # Initial hedge ratio
            rebalance_threshold=0.02  # 2% threshold for rebalancing
        )
        
        return position
    
    def _calculate_portfolio_volatility_metrics(
        self,
        positions: Dict[str, VolatilityPosition]
    ) -> Dict[str, float]:
        """Calculate portfolio-level volatility metrics"""
        
        if not positions:
            return {
                'portfolio_vol': 0.0,
                'total_exposure': 0.0,
                'net_exposure': 0.0,
                'vol_concentration': 0.0,
                'avg_vol_beta': 0.0
            }
        
        # Calculate exposures
        total_exposure = sum(abs(pos.position_size) for pos in positions.values())
        net_exposure = sum(pos.position_size for pos in positions.values())
        
        # Estimate portfolio volatility (simplified)
        weighted_vols = []
        position_weights = []
        
        for position in positions.values():
            weight = abs(position.position_size) / total_exposure if total_exposure > 0 else 0
            weighted_vols.append(position.entry_volatility * weight)
            position_weights.append(weight)
        
        # Simple weighted average (ignoring correlations)
        portfolio_vol = sum(weighted_vols)
        
        # Volatility concentration (Herfindahl index)
        vol_concentration = sum(w ** 2 for w in position_weights)
        
        # Average volatility beta
        avg_vol_beta = np.mean([pos.volatility_beta for pos in positions.values()])
        
        return {
            'portfolio_vol': portfolio_vol,
            'total_exposure': total_exposure,
            'net_exposure': net_exposure,
            'vol_concentration': vol_concentration,
            'avg_vol_beta': avg_vol_beta
        }
    
    def _calculate_volatility_adjustment(
        self,
        position: VolatilityPosition,
        current_vol: float,
        target_portfolio_vol: float
    ) -> float:
        """Calculate required volatility adjustment for position"""
        
        # Current volatility contribution
        current_vol_contribution = position.position_size * current_vol
        
        # Target volatility contribution
        target_vol_contribution = position.position_size * position.target_volatility
        
        # Required adjustment
        if abs(current_vol_contribution) > 0:
            vol_adjustment = (target_vol_contribution - current_vol_contribution) / abs(current_vol_contribution)
        else:
            vol_adjustment = 0.0
        
        return vol_adjustment
    
    def _calculate_vol_contribution(
        self,
        position: VolatilityPosition,
        current_vol: float
    ) -> float:
        """Calculate position's contribution to portfolio volatility"""
        
        vol_contribution = abs(position.position_size) * current_vol
        
        return vol_contribution
    
    def _estimate_current_portfolio_volatility(
        self,
        portfolio: VolatilityPortfolio,
        market_data: Dict[str, Any],
        estimation_date: datetime
    ) -> float:
        """Estimate current portfolio volatility"""
        
        total_vol_contribution = 0.0
        total_weight = 0.0
        
        for bond_id, position in portfolio.positions.items():
            # Get current volatility estimate
            price_data = market_data.get('price_history', {}).get(bond_id)
            if price_data is not None:
                vol_metrics = self.calculate_volatility_metrics(bond_id, price_data, calculation_date=estimation_date)
                current_vol = vol_metrics.realized_vol_1m
            else:
                current_vol = position.entry_volatility
            
            # Calculate contribution
            weight = abs(position.position_size)
            vol_contribution = weight * current_vol
            
            total_vol_contribution += vol_contribution
            total_weight += weight
        
        if total_weight > 0:
            portfolio_vol = total_vol_contribution / total_weight
        else:
            portfolio_vol = 0.0
        
        return portfolio_vol
    
    def _calculate_vol_timing_skill(
        self,
        portfolio: VolatilityPortfolio,
        market_data: Dict[str, pd.DataFrame]
    ) -> float:
        """Calculate volatility timing skill (simplified)"""
        
        # This would require more sophisticated analysis
        # For now, return a placeholder based on portfolio performance
        
        timing_skill = 0.0
        
        for bond_id, position in portfolio.positions.items():
            price_data = market_data.get('prices', pd.DataFrame()).get(bond_id)
            if price_data is None:
                continue
            
            # Simple timing skill: did we increase exposure before vol increased?
            returns = price_data.pct_change().dropna()
            if len(returns) > 20:
                recent_vol = returns.tail(20).std() * np.sqrt(252)
                entry_vol = position.entry_volatility
                
                # If we increased exposure and vol increased, that's good timing
                if position.position_size > 0 and recent_vol > entry_vol:
                    timing_skill += 0.1
                elif position.position_size < 0 and recent_vol < entry_vol:
                    timing_skill += 0.1
        
        # Normalize by number of positions
        if portfolio.positions:
            timing_skill /= len(portfolio.positions)
        
        return max(0.0, min(1.0, timing_skill))
    
    def _calculate_vol_signal_confidence(
        self,
        vol_metrics: VolatilityMetrics,
        vol_regime: VolatilityRegime
    ) -> float:
        """Calculate confidence score for volatility signal"""
        
        # Base confidence from z-score strength
        z_score_confidence = min(abs(vol_metrics.vol_z_score) / 2.0, 1.0)
        
        # Regime consistency bonus
        regime_bonus = {
            VolatilityRegime.LOW_VOLATILITY: 0.8,
            VolatilityRegime.NORMAL_VOLATILITY: 0.6,
            VolatilityRegime.HIGH_VOLATILITY: 0.8,
            VolatilityRegime.CRISIS_VOLATILITY: 0.9,
            VolatilityRegime.TRANSITION: 0.4
        }.get(vol_regime, 0.5)
        
        # Volatility model agreement
        garch_ewma_diff = abs(vol_metrics.garch_vol - vol_metrics.ewma_vol)
        model_agreement = max(0.0, 1.0 - garch_ewma_diff / vol_metrics.realized_vol_1m)
        
        # Volatility persistence (lower vol of vol = higher confidence)
        persistence_bonus = max(0.0, 1.0 - vol_metrics.vol_of_vol / vol_metrics.realized_vol_1m)
        
        confidence = (
            z_score_confidence * 0.4 +
            regime_bonus * 0.3 +
            model_agreement * 0.2 +
            persistence_bonus * 0.1
        )
        
        return max(0.0, min(1.0, confidence))
    
    def _create_default_vol_metrics(
        self,
        bond_id: str,
        calculation_date: Optional[datetime]
    ) -> VolatilityMetrics:
        """Create default volatility metrics when insufficient data"""
        
        default_vol = 0.05  # 5% default volatility
        
        return VolatilityMetrics(
            bond_id=bond_id,
            calculation_date=calculation_date or datetime.now(),
            realized_vol_1m=default_vol,
            realized_vol_3m=default_vol,
            realized_vol_6m=default_vol,
            realized_vol_1y=default_vol,
            garch_vol=default_vol,
            ewma_vol=default_vol,
            vol_of_vol=0.01,
            vol_skew=0.0,
            vol_percentile=50.0,
            vol_z_score=0.0
        )