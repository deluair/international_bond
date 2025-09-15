"""
Curve Strategy Module

This module implements yield curve trading strategies including steepeners,
flatteners, butterflies, and other curve-based relative value trades.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

class CurveTradeType(Enum):
    """Types of curve trades"""
    STEEPENER = "steepener"
    FLATTENER = "flattener"
    BUTTERFLY = "butterfly"
    CONDOR = "condor"
    BARBELL = "barbell"
    BULLET = "bullet"
    TWIST = "twist"

class CurveDirection(Enum):
    """Direction of curve trade"""
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"

class CurveSegment(Enum):
    """Yield curve segments"""
    SHORT_END = "short_end"  # 0-2 years
    BELLY = "belly"          # 2-7 years
    LONG_END = "long_end"    # 7+ years
    ULTRA_LONG = "ultra_long" # 20+ years

@dataclass
class CurvePosition:
    """Represents a position in a curve trade"""
    maturity: float  # years
    weight: float    # position weight
    duration: float  # modified duration
    bond_id: Optional[str] = None
    notional: Optional[float] = None

@dataclass
class CurveTradeSetup:
    """Defines a curve trade setup"""
    trade_id: str
    trade_type: CurveTradeType
    direction: CurveDirection
    positions: List[CurvePosition]
    target_duration: float
    expected_pnl: float
    risk_metrics: Dict[str, float]
    confidence_score: float
    entry_date: datetime
    recommended_holding_period: int
    stop_loss_level: Optional[float]
    take_profit_level: Optional[float]

@dataclass
class CurveAnalysis:
    """Curve analysis results"""
    curve_date: datetime
    curve_level: float
    curve_slope: float
    curve_curvature: float
    steepness_2s10s: float
    steepness_5s30s: float
    butterfly_5s10s30s: float
    volatility_term_structure: Dict[float, float]
    historical_percentiles: Dict[str, float]
    mean_reversion_signals: Dict[str, float]

@dataclass
class CurveTradePerformance:
    """Performance tracking for curve trades"""
    trade_id: str
    entry_date: datetime
    exit_date: Optional[datetime]
    total_pnl: float
    carry_pnl: float
    rolldown_pnl: float
    curve_pnl: float
    duration_pnl: float
    days_held: int
    max_favorable: float
    max_adverse: float
    realized_volatility: float

class CurveStrategy:
    """
    Comprehensive yield curve trading strategy
    """
    
    def __init__(
        self,
        strategy_name: str = "Yield Curve Strategy",
        curve_lookback_period: int = 252,  # 1 year
        volatility_lookback: int = 63,     # 3 months
        mean_reversion_threshold: float = 1.5,  # 1.5 standard deviations
        max_duration_risk: float = 0.5,    # 0.5 years max duration risk
        min_carry_threshold: float = 0.001  # 10 bps minimum carry
    ):
        self.strategy_name = strategy_name
        self.curve_lookback_period = curve_lookback_period
        self.volatility_lookback = volatility_lookback
        self.mean_reversion_threshold = mean_reversion_threshold
        self.max_duration_risk = max_duration_risk
        self.min_carry_threshold = min_carry_threshold
        
        # Strategy state
        self.active_trades: Dict[str, CurveTradeSetup] = {}
        self.historical_analysis: List[CurveAnalysis] = []
        self.performance_history: List[CurveTradePerformance] = {}
        
        # Market data
        self.yield_curves: Dict[datetime, Dict[float, float]] = {}
        self.curve_history: pd.DataFrame = pd.DataFrame()
    
    def analyze_curve(
        self,
        yield_curve: Dict[float, float],
        analysis_date: datetime,
        historical_curves: Optional[Dict[datetime, Dict[float, float]]] = None
    ) -> CurveAnalysis:
        """
        Perform comprehensive yield curve analysis
        """
        
        # Extract key points
        maturities = sorted(yield_curve.keys())
        yields = [yield_curve[mat] for mat in maturities]
        
        # Calculate curve metrics
        curve_level = np.mean(yields)
        curve_slope = self._calculate_curve_slope(maturities, yields)
        curve_curvature = self._calculate_curve_curvature(maturities, yields)
        
        # Calculate key spreads
        steepness_2s10s = self._get_spread(yield_curve, 10.0, 2.0)
        steepness_5s30s = self._get_spread(yield_curve, 30.0, 5.0)
        butterfly_5s10s30s = self._calculate_butterfly(yield_curve, 5.0, 10.0, 30.0)
        
        # Volatility term structure
        volatility_term_structure = self._estimate_volatility_term_structure(
            maturities, historical_curves
        )
        
        # Historical percentiles
        historical_percentiles = self._calculate_historical_percentiles(
            yield_curve, historical_curves
        )
        
        # Mean reversion signals
        mean_reversion_signals = self._calculate_mean_reversion_signals(
            yield_curve, historical_curves
        )
        
        analysis = CurveAnalysis(
            curve_date=analysis_date,
            curve_level=curve_level,
            curve_slope=curve_slope,
            curve_curvature=curve_curvature,
            steepness_2s10s=steepness_2s10s,
            steepness_5s30s=steepness_5s30s,
            butterfly_5s10s30s=butterfly_5s10s30s,
            volatility_term_structure=volatility_term_structure,
            historical_percentiles=historical_percentiles,
            mean_reversion_signals=mean_reversion_signals
        )
        
        self.historical_analysis.append(analysis)
        return analysis
    
    def identify_curve_opportunities(
        self,
        curve_analysis: CurveAnalysis,
        market_conditions: Dict[str, Any]
    ) -> List[CurveTradeSetup]:
        """
        Identify curve trading opportunities based on analysis
        """
        
        opportunities = []
        
        # Steepener/Flattener opportunities
        steepener_trades = self._identify_steepener_opportunities(curve_analysis)
        opportunities.extend(steepener_trades)
        
        # Butterfly opportunities
        butterfly_trades = self._identify_butterfly_opportunities(curve_analysis)
        opportunities.extend(butterfly_trades)
        
        # Barbell/Bullet opportunities
        barbell_trades = self._identify_barbell_opportunities(curve_analysis)
        opportunities.extend(barbell_trades)
        
        # Twist opportunities
        twist_trades = self._identify_twist_opportunities(curve_analysis)
        opportunities.extend(twist_trades)
        
        # Filter and rank opportunities
        opportunities = self._filter_opportunities(opportunities, market_conditions)
        opportunities.sort(key=lambda x: x.confidence_score, reverse=True)
        
        return opportunities[:10]  # Return top 10 opportunities
    
    def construct_curve_trade(
        self,
        trade_type: CurveTradeType,
        direction: CurveDirection,
        target_maturities: List[float],
        yield_curve: Dict[float, float],
        bond_universe: Dict[str, Dict[str, Any]],
        target_duration: float = 0.0
    ) -> Optional[CurveTradeSetup]:
        """
        Construct a specific curve trade
        """
        
        if trade_type == CurveTradeType.STEEPENER:
            return self._construct_steepener(
                direction, target_maturities, yield_curve, bond_universe, target_duration
            )
        elif trade_type == CurveTradeType.BUTTERFLY:
            return self._construct_butterfly(
                direction, target_maturities, yield_curve, bond_universe, target_duration
            )
        elif trade_type == CurveTradeType.BARBELL:
            return self._construct_barbell(
                direction, target_maturities, yield_curve, bond_universe, target_duration
            )
        else:
            return self._construct_generic_trade(
                trade_type, direction, target_maturities, yield_curve, bond_universe, target_duration
            )
    
    def calculate_curve_pnl(
        self,
        trade_setup: CurveTradeSetup,
        initial_curve: Dict[float, float],
        current_curve: Dict[float, float],
        days_elapsed: int
    ) -> Dict[str, float]:
        """
        Calculate P&L components for a curve trade
        """
        
        pnl_components = {
            'total_pnl': 0.0,
            'carry_pnl': 0.0,
            'rolldown_pnl': 0.0,
            'curve_pnl': 0.0,
            'duration_pnl': 0.0
        }
        
        for position in trade_setup.positions:
            # Get initial and current yields
            initial_yield = self._interpolate_yield(initial_curve, position.maturity)
            current_yield = self._interpolate_yield(current_curve, position.maturity)
            
            # Calculate yield change
            yield_change = current_yield - initial_yield
            
            # Duration P&L
            duration_pnl = -position.duration * yield_change * position.weight
            pnl_components['duration_pnl'] += duration_pnl
            
            # Carry P&L (simplified)
            carry_pnl = initial_yield * (days_elapsed / 365) * position.weight
            pnl_components['carry_pnl'] += carry_pnl
            
            # Rolldown P&L
            rolldown_pnl = self._calculate_rolldown_pnl(
                position, initial_curve, days_elapsed
            )
            pnl_components['rolldown_pnl'] += rolldown_pnl
        
        # Curve P&L (residual after duration, carry, rolldown)
        pnl_components['curve_pnl'] = (
            pnl_components['duration_pnl'] - 
            pnl_components['carry_pnl'] - 
            pnl_components['rolldown_pnl']
        )
        
        pnl_components['total_pnl'] = sum([
            pnl_components['duration_pnl'],
            pnl_components['carry_pnl'],
            pnl_components['rolldown_pnl']
        ])
        
        return pnl_components
    
    def monitor_curve_trades(
        self,
        current_curve: Dict[float, float],
        monitoring_date: datetime
    ) -> Dict[str, Dict[str, Any]]:
        """
        Monitor active curve trades and generate signals
        """
        
        monitoring_results = {}
        
        for trade_id, trade_setup in self.active_trades.items():
            # Calculate current P&L
            days_elapsed = (monitoring_date - trade_setup.entry_date).days
            
            # Get initial curve (simplified - would need to store this)
            initial_curve = current_curve  # Placeholder
            
            pnl_components = self.calculate_curve_pnl(
                trade_setup, initial_curve, current_curve, days_elapsed
            )
            
            # Check exit conditions
            exit_signal = self._check_curve_exit_conditions(
                trade_setup, pnl_components, days_elapsed
            )
            
            # Update performance tracking
            performance = CurveTradePerformance(
                trade_id=trade_id,
                entry_date=trade_setup.entry_date,
                exit_date=monitoring_date if exit_signal else None,
                total_pnl=pnl_components['total_pnl'],
                carry_pnl=pnl_components['carry_pnl'],
                rolldown_pnl=pnl_components['rolldown_pnl'],
                curve_pnl=pnl_components['curve_pnl'],
                duration_pnl=pnl_components['duration_pnl'],
                days_held=days_elapsed,
                max_favorable=pnl_components['total_pnl'],  # Simplified
                max_adverse=pnl_components['total_pnl'],   # Simplified
                realized_volatility=0.05  # Placeholder
            )
            
            monitoring_results[trade_id] = {
                'pnl_components': pnl_components,
                'exit_signal': exit_signal,
                'performance': performance,
                'days_held': days_elapsed
            }
        
        return monitoring_results
    
    def _calculate_curve_slope(self, maturities: List[float], yields: List[float]) -> float:
        """Calculate curve slope using linear regression"""
        
        if len(maturities) < 2:
            return 0.0
        
        # Simple slope calculation (30Y - 2Y)
        long_yield = yields[-1] if maturities[-1] >= 20 else yields[-1]
        short_yield = yields[0] if maturities[0] <= 3 else yields[1]
        
        return long_yield - short_yield
    
    def _calculate_curve_curvature(self, maturities: List[float], yields: List[float]) -> float:
        """Calculate curve curvature (second derivative)"""
        
        if len(yields) < 3:
            return 0.0
        
        # Simple curvature measure: 2*10Y - 5Y - 30Y
        try:
            # Find approximate indices for 5Y, 10Y, 30Y
            mid_idx = len(yields) // 2
            short_idx = len(yields) // 4
            long_idx = 3 * len(yields) // 4
            
            curvature = 2 * yields[mid_idx] - yields[short_idx] - yields[long_idx]
            return curvature
        except:
            return 0.0
    
    def _get_spread(self, yield_curve: Dict[float, float], long_mat: float, short_mat: float) -> float:
        """Get spread between two maturities"""
        
        long_yield = self._interpolate_yield(yield_curve, long_mat)
        short_yield = self._interpolate_yield(yield_curve, short_mat)
        
        return long_yield - short_yield
    
    def _calculate_butterfly(
        self,
        yield_curve: Dict[float, float],
        short_mat: float,
        mid_mat: float,
        long_mat: float
    ) -> float:
        """Calculate butterfly spread"""
        
        short_yield = self._interpolate_yield(yield_curve, short_mat)
        mid_yield = self._interpolate_yield(yield_curve, mid_mat)
        long_yield = self._interpolate_yield(yield_curve, long_mat)
        
        return short_yield + long_yield - 2 * mid_yield
    
    def _interpolate_yield(self, yield_curve: Dict[float, float], maturity: float) -> float:
        """Interpolate yield for a given maturity"""
        
        maturities = sorted(yield_curve.keys())
        
        if maturity in yield_curve:
            return yield_curve[maturity]
        
        # Linear interpolation
        for i in range(len(maturities) - 1):
            if maturities[i] <= maturity <= maturities[i + 1]:
                t1, t2 = maturities[i], maturities[i + 1]
                y1, y2 = yield_curve[t1], yield_curve[t2]
                
                weight = (maturity - t1) / (t2 - t1)
                return y1 + weight * (y2 - y1)
        
        # Extrapolation
        if maturity < maturities[0]:
            return yield_curve[maturities[0]]
        else:
            return yield_curve[maturities[-1]]
    
    def _estimate_volatility_term_structure(
        self,
        maturities: List[float],
        historical_curves: Optional[Dict[datetime, Dict[float, float]]]
    ) -> Dict[float, float]:
        """Estimate volatility term structure"""
        
        volatility_ts = {}
        
        for maturity in maturities:
            # Simplified volatility estimation
            if maturity <= 2:
                vol = 0.15  # 15% for short end
            elif maturity <= 10:
                vol = 0.12  # 12% for belly
            else:
                vol = 0.10  # 10% for long end
            
            volatility_ts[maturity] = vol
        
        return volatility_ts
    
    def _calculate_historical_percentiles(
        self,
        current_curve: Dict[float, float],
        historical_curves: Optional[Dict[datetime, Dict[float, float]]]
    ) -> Dict[str, float]:
        """Calculate historical percentiles for key metrics"""
        
        # Simplified percentile calculation
        percentiles = {
            'curve_level': 50.0,
            'curve_slope': 60.0,
            'steepness_2s10s': 45.0,
            'steepness_5s30s': 55.0,
            'butterfly_5s10s30s': 40.0
        }
        
        return percentiles
    
    def _calculate_mean_reversion_signals(
        self,
        current_curve: Dict[float, float],
        historical_curves: Optional[Dict[datetime, Dict[float, float]]]
    ) -> Dict[str, float]:
        """Calculate mean reversion signals"""
        
        # Simplified mean reversion signals
        signals = {
            'steepener_2s10s': 0.3,   # Positive = steepening expected
            'flattener_2s10s': -0.2,  # Negative = flattening expected
            'butterfly_5s10s30s': 0.1, # Positive = butterfly widening expected
            'twist_signal': 0.0       # Twist signal
        }
        
        return signals
    
    def _identify_steepener_opportunities(self, curve_analysis: CurveAnalysis) -> List[CurveTradeSetup]:
        """Identify steepener/flattener opportunities"""
        
        opportunities = []
        
        # Check 2s10s steepener
        if curve_analysis.historical_percentiles.get('steepness_2s10s', 50) < 25:
            # Curve is flat relative to history - steepener opportunity
            trade = CurveTradeSetup(
                trade_id=f"STEEP_2s10s_{curve_analysis.curve_date.strftime('%Y%m%d')}",
                trade_type=CurveTradeType.STEEPENER,
                direction=CurveDirection.LONG,
                positions=[
                    CurvePosition(maturity=10.0, weight=1.0, duration=8.5),
                    CurvePosition(maturity=2.0, weight=-0.85, duration=1.9)
                ],
                target_duration=0.0,
                expected_pnl=0.02,
                risk_metrics={'duration_risk': 0.3, 'curve_risk': 0.8},
                confidence_score=0.7,
                entry_date=curve_analysis.curve_date,
                recommended_holding_period=90,
                stop_loss_level=-0.01,
                take_profit_level=0.03
            )
            opportunities.append(trade)
        
        # Check 5s30s steepener
        if curve_analysis.historical_percentiles.get('steepness_5s30s', 50) < 30:
            trade = CurveTradeSetup(
                trade_id=f"STEEP_5s30s_{curve_analysis.curve_date.strftime('%Y%m%d')}",
                trade_type=CurveTradeType.STEEPENER,
                direction=CurveDirection.LONG,
                positions=[
                    CurvePosition(maturity=30.0, weight=1.0, duration=18.0),
                    CurvePosition(maturity=5.0, weight=-3.6, duration=4.5)
                ],
                target_duration=0.0,
                expected_pnl=0.025,
                risk_metrics={'duration_risk': 0.2, 'curve_risk': 1.0},
                confidence_score=0.65,
                entry_date=curve_analysis.curve_date,
                recommended_holding_period=120,
                stop_loss_level=-0.015,
                take_profit_level=0.04
            )
            opportunities.append(trade)
        
        return opportunities
    
    def _identify_butterfly_opportunities(self, curve_analysis: CurveAnalysis) -> List[CurveTradeSetup]:
        """Identify butterfly opportunities"""
        
        opportunities = []
        
        # Check 5s10s30s butterfly
        butterfly_percentile = curve_analysis.historical_percentiles.get('butterfly_5s10s30s', 50)
        
        if butterfly_percentile < 20:
            # Butterfly is cheap - buy butterfly
            trade = CurveTradeSetup(
                trade_id=f"FLY_5s10s30s_LONG_{curve_analysis.curve_date.strftime('%Y%m%d')}",
                trade_type=CurveTradeType.BUTTERFLY,
                direction=CurveDirection.LONG,
                positions=[
                    CurvePosition(maturity=5.0, weight=1.0, duration=4.5),
                    CurvePosition(maturity=10.0, weight=-2.0, duration=8.5),
                    CurvePosition(maturity=30.0, weight=1.0, duration=18.0)
                ],
                target_duration=0.0,
                expected_pnl=0.015,
                risk_metrics={'duration_risk': 0.1, 'curve_risk': 0.6},
                confidence_score=0.75,
                entry_date=curve_analysis.curve_date,
                recommended_holding_period=60,
                stop_loss_level=-0.008,
                take_profit_level=0.025
            )
            opportunities.append(trade)
        
        elif butterfly_percentile > 80:
            # Butterfly is rich - sell butterfly
            trade = CurveTradeSetup(
                trade_id=f"FLY_5s10s30s_SHORT_{curve_analysis.curve_date.strftime('%Y%m%d')}",
                trade_type=CurveTradeType.BUTTERFLY,
                direction=CurveDirection.SHORT,
                positions=[
                    CurvePosition(maturity=5.0, weight=-1.0, duration=4.5),
                    CurvePosition(maturity=10.0, weight=2.0, duration=8.5),
                    CurvePosition(maturity=30.0, weight=-1.0, duration=18.0)
                ],
                target_duration=0.0,
                expected_pnl=0.012,
                risk_metrics={'duration_risk': 0.1, 'curve_risk': 0.6},
                confidence_score=0.7,
                entry_date=curve_analysis.curve_date,
                recommended_holding_period=60,
                stop_loss_level=-0.008,
                take_profit_level=0.02
            )
            opportunities.append(trade)
        
        return opportunities
    
    def _identify_barbell_opportunities(self, curve_analysis: CurveAnalysis) -> List[CurveTradeSetup]:
        """Identify barbell/bullet opportunities"""
        
        opportunities = []
        
        # Barbell vs Bullet based on curve curvature
        if curve_analysis.curve_curvature > 0.01:  # Positive curvature
            # Favor barbell (short + long vs medium)
            trade = CurveTradeSetup(
                trade_id=f"BARBELL_{curve_analysis.curve_date.strftime('%Y%m%d')}",
                trade_type=CurveTradeType.BARBELL,
                direction=CurveDirection.LONG,
                positions=[
                    CurvePosition(maturity=2.0, weight=0.5, duration=1.9),
                    CurvePosition(maturity=30.0, weight=0.5, duration=18.0),
                    CurvePosition(maturity=10.0, weight=-1.0, duration=8.5)
                ],
                target_duration=0.0,
                expected_pnl=0.01,
                risk_metrics={'duration_risk': 0.2, 'curve_risk': 0.7},
                confidence_score=0.6,
                entry_date=curve_analysis.curve_date,
                recommended_holding_period=90,
                stop_loss_level=-0.008,
                take_profit_level=0.018
            )
            opportunities.append(trade)
        
        return opportunities
    
    def _identify_twist_opportunities(self, curve_analysis: CurveAnalysis) -> List[CurveTradeSetup]:
        """Identify twist opportunities"""
        
        opportunities = []
        
        # Twist based on relative steepness of different curve segments
        front_steepness = curve_analysis.steepness_2s10s
        back_steepness = curve_analysis.steepness_5s30s
        
        if front_steepness / back_steepness > 1.5:  # Front end relatively steep
            # Bear flattener front, bull steepener back
            trade = CurveTradeSetup(
                trade_id=f"TWIST_{curve_analysis.curve_date.strftime('%Y%m%d')}",
                trade_type=CurveTradeType.TWIST,
                direction=CurveDirection.LONG,
                positions=[
                    CurvePosition(maturity=2.0, weight=1.0, duration=1.9),
                    CurvePosition(maturity=5.0, weight=-0.8, duration=4.5),
                    CurvePosition(maturity=30.0, weight=0.3, duration=18.0)
                ],
                target_duration=0.0,
                expected_pnl=0.008,
                risk_metrics={'duration_risk': 0.15, 'curve_risk': 0.5},
                confidence_score=0.55,
                entry_date=curve_analysis.curve_date,
                recommended_holding_period=75,
                stop_loss_level=-0.006,
                take_profit_level=0.015
            )
            opportunities.append(trade)
        
        return opportunities
    
    def _filter_opportunities(
        self,
        opportunities: List[CurveTradeSetup],
        market_conditions: Dict[str, Any]
    ) -> List[CurveTradeSetup]:
        """Filter opportunities based on market conditions"""
        
        filtered = []
        
        for opportunity in opportunities:
            # Check minimum confidence
            if opportunity.confidence_score < 0.5:
                continue
            
            # Check duration risk limits
            total_duration_risk = sum(abs(pos.duration * pos.weight) for pos in opportunity.positions)
            if total_duration_risk > self.max_duration_risk:
                continue
            
            # Check expected carry
            if opportunity.expected_pnl < self.min_carry_threshold:
                continue
            
            filtered.append(opportunity)
        
        return filtered
    
    def _construct_steepener(
        self,
        direction: CurveDirection,
        target_maturities: List[float],
        yield_curve: Dict[float, float],
        bond_universe: Dict[str, Dict[str, Any]],
        target_duration: float
    ) -> CurveTradeSetup:
        """Construct steepener trade"""
        
        if len(target_maturities) < 2:
            raise ValueError("Steepener requires at least 2 maturities")
        
        long_mat = max(target_maturities)
        short_mat = min(target_maturities)
        
        # Duration neutral weights
        long_duration = self._estimate_duration(long_mat)
        short_duration = self._estimate_duration(short_mat)
        
        weight_ratio = long_duration / short_duration
        
        positions = [
            CurvePosition(maturity=long_mat, weight=1.0, duration=long_duration),
            CurvePosition(maturity=short_mat, weight=-weight_ratio, duration=short_duration)
        ]
        
        if direction == CurveDirection.SHORT:
            for pos in positions:
                pos.weight *= -1
        
        return CurveTradeSetup(
            trade_id=f"STEEP_{short_mat}s{long_mat}s_{datetime.now().strftime('%Y%m%d')}",
            trade_type=CurveTradeType.STEEPENER,
            direction=direction,
            positions=positions,
            target_duration=target_duration,
            expected_pnl=0.015,
            risk_metrics={'duration_risk': 0.2, 'curve_risk': 0.8},
            confidence_score=0.6,
            entry_date=datetime.now(),
            recommended_holding_period=90,
            stop_loss_level=-0.01,
            take_profit_level=0.025
        )
    
    def _construct_butterfly(
        self,
        direction: CurveDirection,
        target_maturities: List[float],
        yield_curve: Dict[float, float],
        bond_universe: Dict[str, Dict[str, Any]],
        target_duration: float
    ) -> CurveTradeSetup:
        """Construct butterfly trade"""
        
        if len(target_maturities) < 3:
            raise ValueError("Butterfly requires at least 3 maturities")
        
        target_maturities.sort()
        short_mat, mid_mat, long_mat = target_maturities[0], target_maturities[1], target_maturities[2]
        
        # Equal dollar duration weights
        short_duration = self._estimate_duration(short_mat)
        mid_duration = self._estimate_duration(mid_mat)
        long_duration = self._estimate_duration(long_mat)
        
        positions = [
            CurvePosition(maturity=short_mat, weight=1.0, duration=short_duration),
            CurvePosition(maturity=mid_mat, weight=-2.0, duration=mid_duration),
            CurvePosition(maturity=long_mat, weight=1.0, duration=long_duration)
        ]
        
        if direction == CurveDirection.SHORT:
            for pos in positions:
                pos.weight *= -1
        
        return CurveTradeSetup(
            trade_id=f"FLY_{short_mat}s{mid_mat}s{long_mat}s_{datetime.now().strftime('%Y%m%d')}",
            trade_type=CurveTradeType.BUTTERFLY,
            direction=direction,
            positions=positions,
            target_duration=target_duration,
            expected_pnl=0.01,
            risk_metrics={'duration_risk': 0.1, 'curve_risk': 0.6},
            confidence_score=0.65,
            entry_date=datetime.now(),
            recommended_holding_period=60,
            stop_loss_level=-0.008,
            take_profit_level=0.018
        )
    
    def _construct_barbell(
        self,
        direction: CurveDirection,
        target_maturities: List[float],
        yield_curve: Dict[float, float],
        bond_universe: Dict[str, Dict[str, Any]],
        target_duration: float
    ) -> CurveTradeSetup:
        """Construct barbell trade"""
        
        target_maturities.sort()
        short_mat = target_maturities[0]
        long_mat = target_maturities[-1]
        mid_mat = target_maturities[len(target_maturities)//2] if len(target_maturities) > 2 else (short_mat + long_mat) / 2
        
        positions = [
            CurvePosition(maturity=short_mat, weight=0.5, duration=self._estimate_duration(short_mat)),
            CurvePosition(maturity=long_mat, weight=0.5, duration=self._estimate_duration(long_mat)),
            CurvePosition(maturity=mid_mat, weight=-1.0, duration=self._estimate_duration(mid_mat))
        ]
        
        if direction == CurveDirection.SHORT:
            for pos in positions:
                pos.weight *= -1
        
        return CurveTradeSetup(
            trade_id=f"BARBELL_{datetime.now().strftime('%Y%m%d')}",
            trade_type=CurveTradeType.BARBELL,
            direction=direction,
            positions=positions,
            target_duration=target_duration,
            expected_pnl=0.008,
            risk_metrics={'duration_risk': 0.15, 'curve_risk': 0.7},
            confidence_score=0.6,
            entry_date=datetime.now(),
            recommended_holding_period=90,
            stop_loss_level=-0.006,
            take_profit_level=0.015
        )
    
    def _construct_generic_trade(
        self,
        trade_type: CurveTradeType,
        direction: CurveDirection,
        target_maturities: List[float],
        yield_curve: Dict[float, float],
        bond_universe: Dict[str, Dict[str, Any]],
        target_duration: float
    ) -> CurveTradeSetup:
        """Construct generic curve trade"""
        
        positions = []
        for i, maturity in enumerate(target_maturities):
            weight = 1.0 if i % 2 == 0 else -1.0
            if direction == CurveDirection.SHORT:
                weight *= -1
            
            positions.append(CurvePosition(
                maturity=maturity,
                weight=weight,
                duration=self._estimate_duration(maturity)
            ))
        
        return CurveTradeSetup(
            trade_id=f"{trade_type.value}_{datetime.now().strftime('%Y%m%d')}",
            trade_type=trade_type,
            direction=direction,
            positions=positions,
            target_duration=target_duration,
            expected_pnl=0.01,
            risk_metrics={'duration_risk': 0.2, 'curve_risk': 0.5},
            confidence_score=0.5,
            entry_date=datetime.now(),
            recommended_holding_period=90,
            stop_loss_level=-0.01,
            take_profit_level=0.02
        )
    
    def _estimate_duration(self, maturity: float) -> float:
        """Estimate modified duration for a given maturity"""
        
        # Simplified duration estimation
        if maturity <= 1:
            return maturity * 0.95
        elif maturity <= 10:
            return maturity * 0.85
        else:
            return maturity * 0.75
    
    def _calculate_rolldown_pnl(
        self,
        position: CurvePosition,
        initial_curve: Dict[float, float],
        days_elapsed: int
    ) -> float:
        """Calculate rolldown P&L for a position"""
        
        # Simplified rolldown calculation
        time_decay = days_elapsed / 365
        new_maturity = position.maturity - time_decay
        
        if new_maturity <= 0:
            return 0.0
        
        initial_yield = self._interpolate_yield(initial_curve, position.maturity)
        rolldown_yield = self._interpolate_yield(initial_curve, new_maturity)
        
        yield_change = rolldown_yield - initial_yield
        rolldown_pnl = -position.duration * yield_change * position.weight
        
        return rolldown_pnl
    
    def _check_curve_exit_conditions(
        self,
        trade_setup: CurveTradeSetup,
        pnl_components: Dict[str, float],
        days_elapsed: int
    ) -> bool:
        """Check if curve trade should be exited"""
        
        total_pnl_pct = pnl_components['total_pnl']
        
        # Stop loss
        if trade_setup.stop_loss_level and total_pnl_pct <= trade_setup.stop_loss_level:
            return True
        
        # Take profit
        if trade_setup.take_profit_level and total_pnl_pct >= trade_setup.take_profit_level:
            return True
        
        # Max holding period
        if days_elapsed >= trade_setup.recommended_holding_period:
            return True
        
        return False