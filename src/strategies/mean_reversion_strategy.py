"""
Mean Reversion Strategy Module

This module implements mean reversion trading strategies for international bonds,
including statistical arbitrage, pairs trading, and contrarian strategies.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

class ReversionType(Enum):
    """Types of mean reversion strategies"""
    YIELD_REVERSION = "yield_reversion"
    SPREAD_REVERSION = "spread_reversion"
    PRICE_REVERSION = "price_reversion"
    PAIRS_TRADING = "pairs_trading"
    STATISTICAL_ARBITRAGE = "statistical_arbitrage"
    VOLATILITY_REVERSION = "volatility_reversion"
    CURVE_REVERSION = "curve_reversion"

class ReversionSignal(Enum):
    """Mean reversion signal types"""
    STRONG_REVERT_BUY = "strong_revert_buy"
    REVERT_BUY = "revert_buy"
    NEUTRAL = "neutral"
    REVERT_SELL = "revert_sell"
    STRONG_REVERT_SELL = "strong_revert_sell"

class ReversionState(Enum):
    """Current reversion state"""
    OVERSOLD = "oversold"
    UNDERVALUED = "undervalued"
    FAIR_VALUE = "fair_value"
    OVERVALUED = "overvalued"
    OVERBOUGHT = "overbought"

class ReversionTimeframe(Enum):
    """Mean reversion timeframes"""
    INTRADAY = "intraday"        # Hours to days
    SHORT_TERM = "short_term"    # Days to weeks
    MEDIUM_TERM = "medium_term"  # Weeks to months
    LONG_TERM = "long_term"      # Months to quarters

@dataclass
class ReversionIndicator:
    """Mean reversion indicator"""
    indicator_name: str
    timeframe: ReversionTimeframe
    current_value: float
    mean_value: float
    std_deviation: float
    z_score: float
    percentile_rank: float
    reversion_probability: float
    half_life: float  # Expected time to revert (in days)

@dataclass
class ReversionSignalData:
    """Mean reversion signal data"""
    bond_id: str
    signal_date: datetime
    reversion_type: ReversionType
    signal: ReversionSignal
    reversion_state: ReversionState
    signal_strength: float
    indicators: List[ReversionIndicator]
    fair_value_estimate: float
    current_value: float
    expected_reversion: float
    time_to_revert: float
    confidence_score: float

@dataclass
class ReversionPosition:
    """Mean reversion position"""
    bond_id: str
    position_size: float
    entry_date: datetime
    entry_price: float
    entry_yield: float
    fair_value_target: float
    reversion_score: float
    stop_loss_level: Optional[float]
    take_profit_level: Optional[float]
    max_holding_period: int  # Maximum days to hold

@dataclass
class PairsTrade:
    """Pairs trading position"""
    pair_id: str
    long_bond_id: str
    short_bond_id: str
    long_position_size: float
    short_position_size: float
    entry_date: datetime
    entry_spread: float
    target_spread: float
    current_spread: float
    hedge_ratio: float
    correlation: float
    cointegration_score: float

@dataclass
class ReversionPortfolio:
    """Mean reversion strategy portfolio"""
    portfolio_id: str
    strategy_type: ReversionType
    positions: Dict[str, ReversionPosition]
    pairs_trades: Dict[str, PairsTrade]
    total_exposure: float
    net_exposure: float
    avg_reversion_score: float
    risk_metrics: Dict[str, float]
    performance_metrics: Dict[str, float]

@dataclass
class ReversionPerformance:
    """Performance tracking for mean reversion strategies"""
    strategy_id: str
    start_date: datetime
    end_date: datetime
    total_return: float
    reversion_alpha: float
    market_beta: float
    hit_rate: float
    avg_reversion_time: float
    max_drawdown: float
    sharpe_ratio: float
    calmar_ratio: float

class MeanReversionStrategy:
    """
    Comprehensive mean reversion strategy for international bonds
    """
    
    def __init__(
        self,
        strategy_name: str = "Mean Reversion Strategy",
        lookback_period: int = 252,        # 1 year for mean calculation
        reversion_threshold: float = 1.5,   # Z-score threshold for signals
        max_holding_period: int = 60,       # Maximum days to hold position
        rebalance_frequency: int = 5,       # Rebalance every 5 days
        max_positions: int = 15,            # Maximum number of positions
        pairs_correlation_threshold: float = 0.7,  # Minimum correlation for pairs
        position_sizing_method: str = "z_score_weighted"
    ):
        self.strategy_name = strategy_name
        self.lookback_period = lookback_period
        self.reversion_threshold = reversion_threshold
        self.max_holding_period = max_holding_period
        self.rebalance_frequency = rebalance_frequency
        self.max_positions = max_positions
        self.pairs_correlation_threshold = pairs_correlation_threshold
        self.position_sizing_method = position_sizing_method
        
        # Strategy state
        self.current_portfolio: Optional[ReversionPortfolio] = None
        self.signal_history: List[ReversionSignalData] = []
        self.performance_history: List[ReversionPerformance] = []
        
        # Market data
        self.price_history: pd.DataFrame = pd.DataFrame()
        self.yield_history: pd.DataFrame = pd.DataFrame()
        self.spread_history: pd.DataFrame = pd.DataFrame()
        
        # Statistical models
        self.mean_models: Dict[str, Dict[str, float]] = {}
        self.cointegration_pairs: Dict[str, Dict[str, Any]] = {}
    
    def calculate_reversion_indicators(
        self,
        bond_id: str,
        price_data: pd.Series,
        yield_data: Optional[pd.Series] = None,
        calculation_date: datetime = None
    ) -> List[ReversionIndicator]:
        """
        Calculate mean reversion indicators for a bond
        """
        
        indicators = []
        
        if len(price_data) < self.lookback_period:
            return indicators
        
        # Price reversion indicators
        indicators.extend(self._calculate_price_reversion_indicators(bond_id, price_data))
        
        # Yield reversion indicators (if available)
        if yield_data is not None and len(yield_data) >= self.lookback_period:
            indicators.extend(self._calculate_yield_reversion_indicators(bond_id, yield_data))
        
        # Volatility reversion indicators
        indicators.extend(self._calculate_volatility_reversion_indicators(bond_id, price_data))
        
        # Technical reversion indicators
        indicators.extend(self._calculate_technical_reversion_indicators(bond_id, price_data))
        
        return indicators
    
    def generate_reversion_signals(
        self,
        bond_universe: Dict[str, Dict[str, Any]],
        market_data: Dict[str, pd.DataFrame],
        signal_date: datetime
    ) -> List[ReversionSignalData]:
        """
        Generate mean reversion signals for the bond universe
        """
        
        signals = []
        
        for bond_id, bond_info in bond_universe.items():
            # Get market data
            price_data = market_data.get('prices', pd.DataFrame()).get(bond_id)
            yield_data = market_data.get('yields', pd.DataFrame()).get(bond_id)
            
            if price_data is None or len(price_data) < self.lookback_period:
                continue
            
            # Calculate reversion indicators
            indicators = self.calculate_reversion_indicators(
                bond_id, price_data, yield_data, signal_date
            )
            
            # Generate signal
            signal_data = self._generate_reversion_signal(
                bond_id, bond_info, indicators, signal_date
            )
            
            if signal_data:
                signals.append(signal_data)
        
        # Sort by signal strength
        signals.sort(key=lambda x: x.signal_strength, reverse=True)
        
        return signals
    
    def identify_pairs_trading_opportunities(
        self,
        bond_universe: Dict[str, Dict[str, Any]],
        market_data: Dict[str, pd.DataFrame],
        analysis_date: datetime
    ) -> List[PairsTrade]:
        """
        Identify pairs trading opportunities
        """
        
        pairs_opportunities = []
        
        # Get price data for all bonds
        price_data = market_data.get('prices', pd.DataFrame())
        
        if price_data.empty:
            return pairs_opportunities
        
        bond_ids = list(bond_universe.keys())
        
        # Find cointegrated pairs
        for i, bond1 in enumerate(bond_ids):
            for j, bond2 in enumerate(bond_ids[i+1:], i+1):
                
                if bond1 not in price_data.columns or bond2 not in price_data.columns:
                    continue
                
                # Get price series
                series1 = price_data[bond1].dropna()
                series2 = price_data[bond2].dropna()
                
                # Align series
                common_dates = series1.index.intersection(series2.index)
                if len(common_dates) < self.lookback_period:
                    continue
                
                aligned_series1 = series1.loc[common_dates]
                aligned_series2 = series2.loc[common_dates]
                
                # Test for pairs trading opportunity
                pairs_trade = self._test_pairs_opportunity(
                    bond1, bond2, aligned_series1, aligned_series2, analysis_date
                )
                
                if pairs_trade:
                    pairs_opportunities.append(pairs_trade)
        
        # Sort by cointegration score
        pairs_opportunities.sort(key=lambda x: x.cointegration_score, reverse=True)
        
        return pairs_opportunities[:10]  # Top 10 pairs
    
    def construct_reversion_portfolio(
        self,
        signals: List[ReversionSignalData],
        pairs_trades: List[PairsTrade],
        current_portfolio: Optional[ReversionPortfolio],
        target_exposure: float = 1.0
    ) -> ReversionPortfolio:
        """
        Construct mean reversion portfolio
        """
        
        # Filter signals by threshold
        qualified_signals = [
            signal for signal in signals
            if signal.signal_strength >= self.reversion_threshold
        ]
        
        # Select top signals
        selected_signals = qualified_signals[:self.max_positions]
        
        # Create positions
        positions = {}
        
        for signal in selected_signals:
            position_size = self._calculate_reversion_position_size(
                signal, target_exposure, len(selected_signals)
            )
            
            position = ReversionPosition(
                bond_id=signal.bond_id,
                position_size=position_size,
                entry_date=signal.signal_date,
                entry_price=signal.current_value,
                entry_yield=0.0,  # Would be filled from market data
                fair_value_target=signal.fair_value_estimate,
                reversion_score=signal.signal_strength,
                stop_loss_level=self._calculate_reversion_stop_loss(signal),
                take_profit_level=self._calculate_reversion_take_profit(signal),
                max_holding_period=self.max_holding_period
            )
            
            positions[signal.bond_id] = position
        
        # Add pairs trades (limited allocation)
        pairs_dict = {}
        pairs_allocation = min(0.3 * target_exposure, 0.3)  # Max 30% to pairs
        
        for i, pairs_trade in enumerate(pairs_trades[:5]):  # Max 5 pairs
            pairs_trade.long_position_size = pairs_allocation / 10  # 3% per pair
            pairs_trade.short_position_size = -pairs_allocation / 10
            pairs_dict[pairs_trade.pair_id] = pairs_trade
        
        # Calculate portfolio metrics
        total_exposure = sum(abs(pos.position_size) for pos in positions.values())
        total_exposure += sum(abs(pair.long_position_size) + abs(pair.short_position_size) 
                            for pair in pairs_dict.values())
        
        net_exposure = sum(pos.position_size for pos in positions.values())
        net_exposure += sum(pair.long_position_size + pair.short_position_size 
                          for pair in pairs_dict.values())
        
        avg_reversion_score = np.mean([pos.reversion_score for pos in positions.values()]) if positions else 0.0
        
        # Calculate risk metrics
        risk_metrics = self._calculate_reversion_portfolio_risk_metrics(positions, pairs_dict)
        
        portfolio = ReversionPortfolio(
            portfolio_id=f"reversion_{datetime.now().strftime('%Y%m%d')}",
            strategy_type=ReversionType.STATISTICAL_ARBITRAGE,
            positions=positions,
            pairs_trades=pairs_dict,
            total_exposure=total_exposure,
            net_exposure=net_exposure,
            avg_reversion_score=avg_reversion_score,
            risk_metrics=risk_metrics,
            performance_metrics={}
        )
        
        return portfolio
    
    def update_reversion_positions(
        self,
        current_portfolio: ReversionPortfolio,
        market_data: Dict[str, Any],
        update_date: datetime
    ) -> Dict[str, Dict[str, Any]]:
        """
        Update mean reversion positions and check exit conditions
        """
        
        position_updates = {}
        
        # Update single positions
        for bond_id, position in current_portfolio.positions.items():
            current_price = market_data.get('prices', {}).get(bond_id, position.entry_price)
            
            # Calculate reversion progress
            reversion_progress = self._calculate_reversion_progress(
                position, current_price, update_date
            )
            
            # Check exit conditions
            exit_signal = self._check_reversion_exit_conditions(
                position, current_price, update_date
            )
            
            # Calculate time decay
            days_held = (update_date - position.entry_date).days
            time_decay = self._calculate_time_decay(position, days_held)
            
            position_updates[bond_id] = {
                'current_price': current_price,
                'reversion_progress': reversion_progress,
                'exit_signal': exit_signal,
                'time_decay': time_decay,
                'days_held': days_held,
                'pnl': (current_price - position.entry_price) * position.position_size
            }
        
        # Update pairs trades
        for pair_id, pairs_trade in current_portfolio.pairs_trades.items():
            pairs_update = self._update_pairs_trade(pairs_trade, market_data, update_date)
            position_updates[f"pair_{pair_id}"] = pairs_update
        
        return position_updates
    
    def calculate_reversion_performance(
        self,
        portfolio: ReversionPortfolio,
        start_date: datetime,
        end_date: datetime,
        market_data: Dict[str, pd.DataFrame]
    ) -> ReversionPerformance:
        """
        Calculate performance metrics for mean reversion strategy
        """
        
        # Calculate returns for each position
        position_returns = []
        reversion_times = []
        hit_count = 0
        
        for bond_id, position in portfolio.positions.items():
            price_data = market_data.get('prices', pd.DataFrame()).get(bond_id)
            if price_data is None:
                continue
            
            # Calculate return
            entry_price = position.entry_price
            exit_price = price_data.iloc[-1] if len(price_data) > 0 else entry_price
            
            position_return = (exit_price - entry_price) / entry_price * position.position_size
            position_returns.append(position_return)
            
            # Check if reversion occurred
            target_reached = abs(exit_price - position.fair_value_target) < abs(entry_price - position.fair_value_target)
            if target_reached:
                hit_count += 1
            
            # Calculate reversion time
            reversion_time = (end_date - position.entry_date).days
            reversion_times.append(reversion_time)
        
        # Add pairs trading returns
        for pair_id, pairs_trade in portfolio.pairs_trades.items():
            pairs_return = self._calculate_pairs_return(pairs_trade, market_data, end_date)
            position_returns.append(pairs_return)
        
        # Calculate aggregate metrics
        total_return = sum(position_returns) if position_returns else 0.0
        hit_rate = hit_count / len(portfolio.positions) if portfolio.positions else 0.0
        avg_reversion_time = np.mean(reversion_times) if reversion_times else 0.0
        
        # Calculate Sharpe ratio
        returns_std = np.std(position_returns) if len(position_returns) > 1 else 0.01
        sharpe_ratio = total_return / returns_std if returns_std > 0 else 0.0
        
        # Estimate alpha and beta (simplified)
        reversion_alpha = total_return * 0.7  # Assume 70% is alpha
        market_beta = 0.3  # Low beta strategy
        
        performance = ReversionPerformance(
            strategy_id=portfolio.portfolio_id,
            start_date=start_date,
            end_date=end_date,
            total_return=total_return,
            reversion_alpha=reversion_alpha,
            market_beta=market_beta,
            hit_rate=hit_rate,
            avg_reversion_time=avg_reversion_time,
            max_drawdown=0.03,  # Placeholder
            sharpe_ratio=sharpe_ratio,
            calmar_ratio=sharpe_ratio / 0.03 if sharpe_ratio > 0 else 0.0
        )
        
        return performance
    
    def _calculate_price_reversion_indicators(
        self,
        bond_id: str,
        price_data: pd.Series
    ) -> List[ReversionIndicator]:
        """Calculate price-based reversion indicators"""
        
        indicators = []
        
        if len(price_data) < self.lookback_period:
            return indicators
        
        current_price = price_data.iloc[-1]
        historical_prices = price_data.tail(self.lookback_period)
        
        # Simple mean reversion
        mean_price = historical_prices.mean()
        std_price = historical_prices.std()
        
        if std_price > 0:
            z_score = (current_price - mean_price) / std_price
            percentile = (historical_prices < current_price).mean() * 100
            
            # Estimate half-life using AR(1) model
            returns = price_data.pct_change().dropna()
            if len(returns) > 10:
                half_life = self._estimate_half_life(returns)
            else:
                half_life = 30.0  # Default 30 days
            
            price_reversion = ReversionIndicator(
                indicator_name="price_mean_reversion",
                timeframe=ReversionTimeframe.MEDIUM_TERM,
                current_value=current_price,
                mean_value=mean_price,
                std_deviation=std_price,
                z_score=z_score,
                percentile_rank=percentile,
                reversion_probability=self._calculate_reversion_probability(z_score),
                half_life=half_life
            )
            indicators.append(price_reversion)
        
        # Bollinger Bands reversion
        bb_period = 20
        if len(price_data) >= bb_period:
            bb_mean = price_data.rolling(bb_period).mean().iloc[-1]
            bb_std = price_data.rolling(bb_period).std().iloc[-1]
            
            if bb_std > 0:
                bb_z_score = (current_price - bb_mean) / bb_std
                
                bb_reversion = ReversionIndicator(
                    indicator_name="bollinger_reversion",
                    timeframe=ReversionTimeframe.SHORT_TERM,
                    current_value=current_price,
                    mean_value=bb_mean,
                    std_deviation=bb_std,
                    z_score=bb_z_score,
                    percentile_rank=50 + bb_z_score * 15,  # Approximate percentile
                    reversion_probability=self._calculate_reversion_probability(bb_z_score),
                    half_life=10.0  # Shorter half-life for BB
                )
                indicators.append(bb_reversion)
        
        return indicators
    
    def _calculate_yield_reversion_indicators(
        self,
        bond_id: str,
        yield_data: pd.Series
    ) -> List[ReversionIndicator]:
        """Calculate yield-based reversion indicators"""
        
        indicators = []
        
        if len(yield_data) < self.lookback_period:
            return indicators
        
        current_yield = yield_data.iloc[-1]
        historical_yields = yield_data.tail(self.lookback_period)
        
        # Yield mean reversion
        mean_yield = historical_yields.mean()
        std_yield = historical_yields.std()
        
        if std_yield > 0:
            z_score = (current_yield - mean_yield) / std_yield
            percentile = (historical_yields < current_yield).mean() * 100
            
            # Estimate half-life
            yield_changes = yield_data.diff().dropna()
            if len(yield_changes) > 10:
                half_life = self._estimate_half_life(yield_changes)
            else:
                half_life = 45.0  # Default 45 days for yields
            
            yield_reversion = ReversionIndicator(
                indicator_name="yield_mean_reversion",
                timeframe=ReversionTimeframe.MEDIUM_TERM,
                current_value=current_yield,
                mean_value=mean_yield,
                std_deviation=std_yield,
                z_score=z_score,
                percentile_rank=percentile,
                reversion_probability=self._calculate_reversion_probability(z_score),
                half_life=half_life
            )
            indicators.append(yield_reversion)
        
        return indicators
    
    def _calculate_volatility_reversion_indicators(
        self,
        bond_id: str,
        price_data: pd.Series
    ) -> List[ReversionIndicator]:
        """Calculate volatility reversion indicators"""
        
        indicators = []
        
        if len(price_data) < self.lookback_period:
            return indicators
        
        # Calculate rolling volatility
        returns = price_data.pct_change().dropna()
        vol_window = 20
        
        if len(returns) < vol_window * 2:
            return indicators
        
        current_vol = returns.tail(vol_window).std() * np.sqrt(252)  # Annualized
        historical_vols = returns.rolling(vol_window).std().dropna() * np.sqrt(252)
        
        if len(historical_vols) < 50:
            return indicators
        
        mean_vol = historical_vols.mean()
        std_vol = historical_vols.std()
        
        if std_vol > 0:
            vol_z_score = (current_vol - mean_vol) / std_vol
            vol_percentile = (historical_vols < current_vol).mean() * 100
            
            vol_reversion = ReversionIndicator(
                indicator_name="volatility_reversion",
                timeframe=ReversionTimeframe.SHORT_TERM,
                current_value=current_vol,
                mean_value=mean_vol,
                std_deviation=std_vol,
                z_score=vol_z_score,
                percentile_rank=vol_percentile,
                reversion_probability=self._calculate_reversion_probability(vol_z_score),
                half_life=15.0  # Volatility reverts faster
            )
            indicators.append(vol_reversion)
        
        return indicators
    
    def _calculate_technical_reversion_indicators(
        self,
        bond_id: str,
        price_data: pd.Series
    ) -> List[ReversionIndicator]:
        """Calculate technical reversion indicators"""
        
        indicators = []
        
        if len(price_data) < 50:
            return indicators
        
        # RSI reversion
        rsi_period = 14
        if len(price_data) >= rsi_period * 2:
            rsi = self._calculate_rsi(price_data, rsi_period)
            current_rsi = rsi.iloc[-1]
            
            # RSI mean reversion (50 is neutral)
            rsi_deviation = current_rsi - 50
            rsi_z_score = rsi_deviation / 15  # RSI std is roughly 15
            
            rsi_reversion = ReversionIndicator(
                indicator_name="rsi_reversion",
                timeframe=ReversionTimeframe.SHORT_TERM,
                current_value=current_rsi,
                mean_value=50.0,
                std_deviation=15.0,
                z_score=rsi_z_score,
                percentile_rank=current_rsi,  # RSI is already a percentile
                reversion_probability=self._calculate_rsi_reversion_probability(current_rsi),
                half_life=7.0  # RSI reverts quickly
            )
            indicators.append(rsi_reversion)
        
        return indicators
    
    def _generate_reversion_signal(
        self,
        bond_id: str,
        bond_info: Dict[str, Any],
        indicators: List[ReversionIndicator],
        signal_date: datetime
    ) -> Optional[ReversionSignalData]:
        """Generate composite reversion signal"""
        
        if not indicators:
            return None
        
        # Calculate weighted signal strength
        total_weight = 0.0
        weighted_signal = 0.0
        
        timeframe_weights = {
            ReversionTimeframe.INTRADAY: 0.1,
            ReversionTimeframe.SHORT_TERM: 0.3,
            ReversionTimeframe.MEDIUM_TERM: 0.4,
            ReversionTimeframe.LONG_TERM: 0.2
        }
        
        for indicator in indicators:
            weight = timeframe_weights.get(indicator.timeframe, 0.25)
            # Use z-score and reversion probability
            signal_contribution = indicator.z_score * indicator.reversion_probability * weight
            weighted_signal += signal_contribution
            total_weight += weight
        
        if total_weight == 0:
            return None
        
        composite_signal_strength = abs(weighted_signal / total_weight)
        signal_direction = np.sign(weighted_signal / total_weight)
        
        # Determine signal type
        if composite_signal_strength > 2.0:
            if signal_direction > 0:
                signal = ReversionSignal.STRONG_REVERT_SELL  # Overbought, expect reversion down
            else:
                signal = ReversionSignal.STRONG_REVERT_BUY   # Oversold, expect reversion up
        elif composite_signal_strength > 1.0:
            if signal_direction > 0:
                signal = ReversionSignal.REVERT_SELL
            else:
                signal = ReversionSignal.REVERT_BUY
        else:
            signal = ReversionSignal.NEUTRAL
        
        # Determine reversion state
        avg_z_score = np.mean([ind.z_score for ind in indicators])
        if avg_z_score > 2.0:
            reversion_state = ReversionState.OVERBOUGHT
        elif avg_z_score > 1.0:
            reversion_state = ReversionState.OVERVALUED
        elif avg_z_score < -2.0:
            reversion_state = ReversionState.OVERSOLD
        elif avg_z_score < -1.0:
            reversion_state = ReversionState.UNDERVALUED
        else:
            reversion_state = ReversionState.FAIR_VALUE
        
        # Calculate fair value estimate
        price_indicators = [ind for ind in indicators if 'price' in ind.indicator_name]
        if price_indicators:
            fair_value_estimate = np.mean([ind.mean_value for ind in price_indicators])
            current_value = price_indicators[0].current_value
        else:
            fair_value_estimate = 100.0  # Placeholder
            current_value = 100.0
        
        # Calculate expected reversion
        expected_reversion = (fair_value_estimate - current_value) / current_value
        
        # Estimate time to revert
        avg_half_life = np.mean([ind.half_life for ind in indicators])
        time_to_revert = avg_half_life * np.log(2)  # Time to 50% reversion
        
        # Calculate confidence score
        confidence_score = self._calculate_reversion_confidence(indicators, composite_signal_strength)
        
        signal_data = ReversionSignalData(
            bond_id=bond_id,
            signal_date=signal_date,
            reversion_type=ReversionType.STATISTICAL_ARBITRAGE,
            signal=signal,
            reversion_state=reversion_state,
            signal_strength=composite_signal_strength,
            indicators=indicators,
            fair_value_estimate=fair_value_estimate,
            current_value=current_value,
            expected_reversion=expected_reversion,
            time_to_revert=time_to_revert,
            confidence_score=confidence_score
        )
        
        return signal_data
    
    def _test_pairs_opportunity(
        self,
        bond1_id: str,
        bond2_id: str,
        series1: pd.Series,
        series2: pd.Series,
        analysis_date: datetime
    ) -> Optional[PairsTrade]:
        """Test for pairs trading opportunity between two bonds"""
        
        # Calculate correlation
        correlation = series1.corr(series2)
        
        if abs(correlation) < self.pairs_correlation_threshold:
            return None
        
        # Test for cointegration (simplified)
        # In practice, would use Johansen test or Engle-Granger
        spread = series1 - series2
        spread_mean = spread.mean()
        spread_std = spread.std()
        
        if spread_std == 0:
            return None
        
        current_spread = spread.iloc[-1]
        spread_z_score = (current_spread - spread_mean) / spread_std
        
        # Only consider if spread is significantly away from mean
        if abs(spread_z_score) < 1.5:
            return None
        
        # Calculate hedge ratio (simplified)
        hedge_ratio = np.cov(series1, series2)[0, 1] / np.var(series2)
        
        # Estimate cointegration score (simplified)
        # Higher score = more mean reverting spread
        spread_adf_stat = self._calculate_adf_statistic(spread)  # Simplified
        cointegration_score = max(0, 5 + spread_adf_stat)  # Convert to positive score
        
        # Determine trade direction
        if spread_z_score > 0:
            # Spread too high: short bond1, long bond2
            long_bond_id = bond2_id
            short_bond_id = bond1_id
        else:
            # Spread too low: long bond1, short bond2
            long_bond_id = bond1_id
            short_bond_id = bond2_id
        
        pairs_trade = PairsTrade(
            pair_id=f"{bond1_id}_{bond2_id}",
            long_bond_id=long_bond_id,
            short_bond_id=short_bond_id,
            long_position_size=0.0,  # Will be set in portfolio construction
            short_position_size=0.0,
            entry_date=analysis_date,
            entry_spread=current_spread,
            target_spread=spread_mean,
            current_spread=current_spread,
            hedge_ratio=hedge_ratio,
            correlation=correlation,
            cointegration_score=cointegration_score
        )
        
        return pairs_trade
    
    def _calculate_reversion_position_size(
        self,
        signal: ReversionSignalData,
        target_exposure: float,
        num_positions: int
    ) -> float:
        """Calculate position size for reversion trade"""
        
        if self.position_sizing_method == "equal_weight":
            base_size = target_exposure / num_positions
        elif self.position_sizing_method == "z_score_weighted":
            # Size based on z-score strength
            avg_z_score = np.mean([abs(ind.z_score) for ind in signal.indicators])
            base_size = target_exposure * min(avg_z_score / 2.0, 1.0) / num_positions
        else:
            base_size = target_exposure / num_positions
        
        # Adjust for confidence
        confidence_adjustment = signal.confidence_score
        
        # Adjust for expected reversion
        reversion_adjustment = min(abs(signal.expected_reversion) * 10, 1.0)
        
        position_size = base_size * confidence_adjustment * reversion_adjustment
        
        # Apply signal direction (contrarian)
        if signal.signal in [ReversionSignal.REVERT_BUY, ReversionSignal.STRONG_REVERT_BUY]:
            position_size = abs(position_size)  # Long position
        elif signal.signal in [ReversionSignal.REVERT_SELL, ReversionSignal.STRONG_REVERT_SELL]:
            position_size = -abs(position_size)  # Short position
        
        return position_size
    
    def _calculate_reversion_stop_loss(self, signal: ReversionSignalData) -> Optional[float]:
        """Calculate stop loss for reversion position"""
        
        # Stop loss based on additional adverse movement
        avg_std = np.mean([ind.std_deviation for ind in signal.indicators if ind.std_deviation > 0])
        
        if avg_std > 0:
            # Stop if moves another 1 standard deviation against us
            stop_loss_distance = avg_std / signal.current_value
            return min(stop_loss_distance, 0.05)  # Max 5% stop loss
        
        return 0.03  # Default 3% stop loss
    
    def _calculate_reversion_take_profit(self, signal: ReversionSignalData) -> Optional[float]:
        """Calculate take profit for reversion position"""
        
        # Take profit at 50% of expected reversion
        expected_profit = abs(signal.expected_reversion) * 0.5
        
        return max(expected_profit, 0.01)  # Minimum 1% take profit
    
    def _calculate_reversion_progress(
        self,
        position: ReversionPosition,
        current_price: float,
        update_date: datetime
    ) -> float:
        """Calculate how much reversion has occurred"""
        
        entry_distance = abs(position.entry_price - position.fair_value_target)
        current_distance = abs(current_price - position.fair_value_target)
        
        if entry_distance == 0:
            return 1.0  # Already at target
        
        reversion_progress = (entry_distance - current_distance) / entry_distance
        
        return max(0.0, min(1.0, reversion_progress))
    
    def _check_reversion_exit_conditions(
        self,
        position: ReversionPosition,
        current_price: float,
        update_date: datetime
    ) -> bool:
        """Check if reversion position should be exited"""
        
        # Check maximum holding period
        days_held = (update_date - position.entry_date).days
        if days_held >= position.max_holding_period:
            return True
        
        # Check take profit
        price_change = (current_price - position.entry_price) / position.entry_price
        if position.take_profit_level:
            if position.position_size > 0 and price_change >= position.take_profit_level:
                return True
            elif position.position_size < 0 and price_change <= -position.take_profit_level:
                return True
        
        # Check stop loss
        if position.stop_loss_level:
            if position.position_size > 0 and price_change <= -position.stop_loss_level:
                return True
            elif position.position_size < 0 and price_change >= position.stop_loss_level:
                return True
        
        # Check if reversion target reached
        reversion_progress = self._calculate_reversion_progress(position, current_price, update_date)
        if reversion_progress >= 0.8:  # 80% of reversion achieved
            return True
        
        return False
    
    def _calculate_time_decay(self, position: ReversionPosition, days_held: int) -> float:
        """Calculate time decay factor for reversion position"""
        
        # Reversion probability decays over time
        decay_rate = 1.0 / position.max_holding_period
        time_decay = np.exp(-decay_rate * days_held)
        
        return time_decay
    
    def _update_pairs_trade(
        self,
        pairs_trade: PairsTrade,
        market_data: Dict[str, Any],
        update_date: datetime
    ) -> Dict[str, Any]:
        """Update pairs trade status"""
        
        # Get current prices
        long_price = market_data.get('prices', {}).get(pairs_trade.long_bond_id, 100.0)
        short_price = market_data.get('prices', {}).get(pairs_trade.short_bond_id, 100.0)
        
        # Calculate current spread
        current_spread = long_price - short_price
        pairs_trade.current_spread = current_spread
        
        # Calculate P&L
        spread_change = current_spread - pairs_trade.entry_spread
        
        # P&L depends on trade direction
        if pairs_trade.entry_spread > pairs_trade.target_spread:
            # We're short the spread (expecting it to narrow)
            pairs_pnl = -spread_change * abs(pairs_trade.long_position_size)
        else:
            # We're long the spread (expecting it to widen)
            pairs_pnl = spread_change * abs(pairs_trade.long_position_size)
        
        # Check exit conditions
        spread_normalized = (current_spread - pairs_trade.target_spread) / abs(pairs_trade.entry_spread - pairs_trade.target_spread)
        exit_signal = abs(spread_normalized) < 0.2  # Exit when spread is within 20% of target
        
        days_held = (update_date - pairs_trade.entry_date).days
        
        return {
            'current_spread': current_spread,
            'spread_change': spread_change,
            'pairs_pnl': pairs_pnl,
            'exit_signal': exit_signal,
            'days_held': days_held,
            'spread_normalized': spread_normalized
        }
    
    def _calculate_pairs_return(
        self,
        pairs_trade: PairsTrade,
        market_data: Dict[str, pd.DataFrame],
        end_date: datetime
    ) -> float:
        """Calculate return for pairs trade"""
        
        # Get price data
        price_data = market_data.get('prices', pd.DataFrame())
        
        if pairs_trade.long_bond_id not in price_data.columns or pairs_trade.short_bond_id not in price_data.columns:
            return 0.0
        
        # Get final prices
        long_final_price = price_data[pairs_trade.long_bond_id].iloc[-1]
        short_final_price = price_data[pairs_trade.short_bond_id].iloc[-1]
        
        # Calculate returns for each leg
        long_return = (long_final_price - 100.0) / 100.0 * pairs_trade.long_position_size  # Simplified
        short_return = (short_final_price - 100.0) / 100.0 * pairs_trade.short_position_size
        
        total_return = long_return + short_return
        
        return total_return
    
    def _calculate_reversion_portfolio_risk_metrics(
        self,
        positions: Dict[str, ReversionPosition],
        pairs_trades: Dict[str, PairsTrade]
    ) -> Dict[str, float]:
        """Calculate portfolio risk metrics for reversion strategy"""
        
        # Position concentration
        position_sizes = [abs(pos.position_size) for pos in positions.values()]
        total_size = sum(position_sizes)
        
        if total_size > 0:
            concentration = sum((size / total_size) ** 2 for size in position_sizes)
        else:
            concentration = 0.0
        
        # Average reversion score
        reversion_scores = [pos.reversion_score for pos in positions.values()]
        avg_reversion_score = np.mean(reversion_scores) if reversion_scores else 0.0
        
        # Time diversification
        holding_periods = [pos.max_holding_period for pos in positions.values()]
        avg_holding_period = np.mean(holding_periods) if holding_periods else 0.0
        
        return {
            'concentration_risk': concentration,
            'avg_reversion_score': avg_reversion_score,
            'avg_holding_period': avg_holding_period,
            'num_positions': len(positions),
            'num_pairs': len(pairs_trades),
            'gross_exposure': sum(abs(pos.position_size) for pos in positions.values()),
            'net_exposure': sum(pos.position_size for pos in positions.values())
        }
    
    def _calculate_reversion_confidence(
        self,
        indicators: List[ReversionIndicator],
        signal_strength: float
    ) -> float:
        """Calculate confidence score for reversion signal"""
        
        if not indicators:
            return 0.0
        
        # Base confidence from signal strength
        strength_confidence = min(signal_strength / 2.0, 1.0)
        
        # Consistency across indicators
        z_scores = [ind.z_score for ind in indicators]
        z_score_consistency = 1.0 - (np.std(z_scores) / (np.mean(np.abs(z_scores)) + 0.1))
        z_score_consistency = max(0.0, min(1.0, z_score_consistency))
        
        # Reversion probability
        avg_reversion_prob = np.mean([ind.reversion_probability for ind in indicators])
        
        # Number of indicators
        indicator_bonus = min(len(indicators) / 4.0, 1.0)
        
        confidence = (
            strength_confidence * 0.3 +
            z_score_consistency * 0.3 +
            avg_reversion_prob * 0.3 +
            indicator_bonus * 0.1
        )
        
        return max(0.0, min(1.0, confidence))
    
    def _calculate_reversion_probability(self, z_score: float) -> float:
        """Calculate probability of reversion based on z-score"""
        
        # Higher absolute z-score = higher reversion probability
        # Using sigmoid function
        prob = 1.0 / (1.0 + np.exp(-abs(z_score)))
        
        return prob
    
    def _calculate_rsi_reversion_probability(self, rsi: float) -> float:
        """Calculate reversion probability based on RSI"""
        
        if rsi > 70:
            # Overbought - high probability of reversion down
            return (rsi - 70) / 30.0
        elif rsi < 30:
            # Oversold - high probability of reversion up
            return (30 - rsi) / 30.0
        else:
            # Neutral zone - low reversion probability
            return 0.1
    
    def _estimate_half_life(self, series: pd.Series) -> float:
        """Estimate half-life of mean reversion using AR(1) model"""
        
        if len(series) < 10:
            return 30.0  # Default
        
        # Fit AR(1) model: x(t) = alpha * x(t-1) + epsilon
        lagged_series = series.shift(1).dropna()
        current_series = series[1:]
        
        # Align series
        common_index = lagged_series.index.intersection(current_series.index)
        if len(common_index) < 5:
            return 30.0
        
        lagged_values = lagged_series.loc[common_index].values
        current_values = current_series.loc[common_index].values
        
        # Simple linear regression
        if len(lagged_values) > 0 and np.var(lagged_values) > 0:
            alpha = np.cov(current_values, lagged_values)[0, 1] / np.var(lagged_values)
            
            # Half-life = ln(0.5) / ln(alpha)
            if 0 < alpha < 1:
                half_life = np.log(0.5) / np.log(alpha)
                return max(1.0, min(half_life, 365.0))  # Bound between 1 day and 1 year
        
        return 30.0  # Default half-life
    
    def _calculate_rsi(self, price_data: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI (Relative Strength Index)"""
        
        delta = price_data.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_adf_statistic(self, series: pd.Series) -> float:
        """Simplified ADF test statistic calculation"""
        
        # This is a very simplified version
        # In practice, would use statsmodels.tsa.stattools.adfuller
        
        if len(series) < 10:
            return 0.0
        
        # Calculate first difference
        diff_series = series.diff().dropna()
        
        if len(diff_series) < 5:
            return 0.0
        
        # Simple test: if variance of differences is much smaller than variance of levels
        level_var = series.var()
        diff_var = diff_series.var()
        
        if level_var > 0:
            adf_proxy = -np.log(diff_var / level_var)
            return max(-10.0, min(adf_proxy, 0.0))  # Bound the statistic
        
        return 0.0