"""
Relative value analyzer for comprehensive bond comparison and valuation metrics.
"""

import numpy as np
from datetime import date, datetime, timedelta
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import math
from scipy import stats

from ..models.bond import SovereignBond
from ..models.currency import CurrencyType
from ..pricing.bond_pricer import BondPricer
from ..pricing.yield_curve import YieldCurve


class RelativeValueMetric(Enum):
    """Relative value metrics for bond comparison."""
    YIELD_SPREAD = "yield_spread"
    Z_SPREAD = "z_spread"
    OPTION_ADJUSTED_SPREAD = "option_adjusted_spread"
    ASSET_SWAP_SPREAD = "asset_swap_spread"
    CREDIT_SPREAD = "credit_spread"
    DURATION_ADJUSTED_SPREAD = "duration_adjusted_spread"
    CONVEXITY_ADJUSTED_SPREAD = "convexity_adjusted_spread"
    CARRY_ROLL_DOWN = "carry_roll_down"
    BREAKEVEN_SPREAD = "breakeven_spread"
    RELATIVE_CHEAPNESS = "relative_cheapness"


class RiskAdjustmentMethod(Enum):
    """Risk adjustment methods for relative value analysis."""
    DURATION_NEUTRAL = "duration_neutral"
    BETA_ADJUSTED = "beta_adjusted"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    VAR_ADJUSTED = "var_adjusted"
    SHARPE_ADJUSTED = "sharpe_adjusted"
    INFORMATION_RATIO_ADJUSTED = "information_ratio_adjusted"


class ComparisonBenchmark(Enum):
    """Benchmark types for relative value comparison."""
    GOVERNMENT_CURVE = "government_curve"
    SWAP_CURVE = "swap_curve"
    CORPORATE_INDEX = "corporate_index"
    PEER_GROUP = "peer_group"
    HISTORICAL_AVERAGE = "historical_average"
    SECTOR_MEDIAN = "sector_median"


@dataclass
class RelativeValueInput:
    """Input parameters for relative value analysis."""
    target_bond: SovereignBond
    benchmark_bonds: List[SovereignBond]
    yield_curves: Dict[str, YieldCurve]
    
    # Analysis parameters
    metrics: List[RelativeValueMetric] = field(default_factory=lambda: [RelativeValueMetric.YIELD_SPREAD])
    risk_adjustment: RiskAdjustmentMethod = RiskAdjustmentMethod.DURATION_NEUTRAL
    benchmark_type: ComparisonBenchmark = ComparisonBenchmark.GOVERNMENT_CURVE
    
    # Time horizon
    analysis_date: date = field(default_factory=date.today)
    horizon_days: int = 30
    
    # Risk parameters
    confidence_level: float = 0.95
    lookback_days: int = 252


@dataclass
class SpreadAnalysis:
    """Spread analysis results."""
    current_spread: float
    historical_mean: float
    historical_std: float
    percentile_rank: float
    z_score: float
    
    # Statistical measures
    min_spread: float
    max_spread: float
    median_spread: float
    
    # Trend analysis
    trend_slope: float
    trend_r_squared: float
    momentum_score: float
    
    def __str__(self) -> str:
        return (f"Spread Analysis:\n"
                f"Current: {self.current_spread:.2f} bps\n"
                f"Historical Mean: {self.historical_mean:.2f} bps\n"
                f"Z-Score: {self.z_score:.2f}\n"
                f"Percentile: {self.percentile_rank:.1%}\n"
                f"Trend Slope: {self.trend_slope:.4f}")


@dataclass
class ComparisonResult:
    """Comprehensive relative value comparison result."""
    target_bond_id: str
    benchmark_id: str
    
    # Spread metrics
    spread_analyses: Dict[RelativeValueMetric, SpreadAnalysis]
    
    # Risk-adjusted metrics
    risk_adjusted_return: float
    risk_adjusted_spread: float
    beta_to_benchmark: float
    
    # Relative value scores
    cheapness_score: float  # -100 (expensive) to +100 (cheap)
    momentum_score: float   # -100 (negative) to +100 (positive)
    carry_score: float      # Expected carry over horizon
    
    # Statistical significance
    significance_level: float
    confidence_interval: Tuple[float, float]
    
    # Recommendation
    recommendation: str  # "BUY", "SELL", "HOLD"
    conviction_level: float  # 0-1 scale
    
    def __str__(self) -> str:
        return (f"Relative Value Analysis: {self.target_bond_id}\n"
                f"Benchmark: {self.benchmark_id}\n"
                f"Cheapness Score: {self.cheapness_score:.1f}\n"
                f"Risk-Adjusted Return: {self.risk_adjusted_return:.2%}\n"
                f"Beta: {self.beta_to_benchmark:.2f}\n"
                f"Recommendation: {self.recommendation}\n"
                f"Conviction: {self.conviction_level:.1%}")


class RelativeValueAnalyzer:
    """
    Comprehensive relative value analyzer for sovereign bonds.
    """
    
    def __init__(self, bond_pricer: Optional[BondPricer] = None):
        """
        Initialize relative value analyzer.
        
        Args:
            bond_pricer: Bond pricing engine
        """
        self.bond_pricer = bond_pricer or BondPricer()
        self.analysis_history: List[ComparisonResult] = []
    
    def analyze_relative_value(self, 
                             inputs: RelativeValueInput) -> List[ComparisonResult]:
        """
        Perform comprehensive relative value analysis.
        
        Args:
            inputs: Analysis input parameters
            
        Returns:
            List of comparison results vs each benchmark
        """
        results = []
        
        # Analyze against each benchmark bond
        for benchmark_bond in inputs.benchmark_bonds:
            result = self._compare_bonds(
                inputs.target_bond,
                benchmark_bond,
                inputs
            )
            results.append(result)
        
        # Store analysis history
        self.analysis_history.extend(results)
        
        return results
    
    def calculate_spread_metrics(self, 
                               target_bond: SovereignBond,
                               benchmark_bond: SovereignBond,
                               yield_curves: Dict[str, YieldCurve],
                               metrics: List[RelativeValueMetric]) -> Dict[RelativeValueMetric, float]:
        """
        Calculate various spread metrics between bonds.
        
        Args:
            target_bond: Target bond for analysis
            benchmark_bond: Benchmark bond
            yield_curves: Yield curves for pricing
            metrics: List of metrics to calculate
            
        Returns:
            Dictionary of spread values by metric
        """
        spread_values = {}
        
        # Get bond yields
        target_yield = self._get_bond_yield(target_bond, yield_curves)
        benchmark_yield = self._get_bond_yield(benchmark_bond, yield_curves)
        
        for metric in metrics:
            if metric == RelativeValueMetric.YIELD_SPREAD:
                spread_values[metric] = (target_yield - benchmark_yield) * 10000  # bps
                
            elif metric == RelativeValueMetric.Z_SPREAD:
                spread_values[metric] = self._calculate_z_spread(
                    target_bond, benchmark_bond, yield_curves
                )
                
            elif metric == RelativeValueMetric.OPTION_ADJUSTED_SPREAD:
                spread_values[metric] = self._calculate_oas(
                    target_bond, benchmark_bond, yield_curves
                )
                
            elif metric == RelativeValueMetric.ASSET_SWAP_SPREAD:
                spread_values[metric] = self._calculate_asset_swap_spread(
                    target_bond, benchmark_bond, yield_curves
                )
                
            elif metric == RelativeValueMetric.CREDIT_SPREAD:
                spread_values[metric] = self._calculate_credit_spread(
                    target_bond, benchmark_bond, yield_curves
                )
                
            elif metric == RelativeValueMetric.DURATION_ADJUSTED_SPREAD:
                spread_values[metric] = self._calculate_duration_adjusted_spread(
                    target_bond, benchmark_bond, yield_curves
                )
                
            elif metric == RelativeValueMetric.CONVEXITY_ADJUSTED_SPREAD:
                spread_values[metric] = self._calculate_convexity_adjusted_spread(
                    target_bond, benchmark_bond, yield_curves
                )
                
            elif metric == RelativeValueMetric.CARRY_ROLL_DOWN:
                spread_values[metric] = self._calculate_carry_roll_down(
                    target_bond, benchmark_bond, yield_curves
                )
                
            elif metric == RelativeValueMetric.BREAKEVEN_SPREAD:
                spread_values[metric] = self._calculate_breakeven_spread(
                    target_bond, benchmark_bond, yield_curves
                )
                
            elif metric == RelativeValueMetric.RELATIVE_CHEAPNESS:
                spread_values[metric] = self._calculate_relative_cheapness(
                    target_bond, benchmark_bond, yield_curves
                )
        
        return spread_values
    
    def calculate_risk_adjusted_metrics(self, 
                                      target_bond: SovereignBond,
                                      benchmark_bond: SovereignBond,
                                      yield_curves: Dict[str, YieldCurve],
                                      adjustment_method: RiskAdjustmentMethod) -> Dict[str, float]:
        """
        Calculate risk-adjusted relative value metrics.
        
        Args:
            target_bond: Target bond
            benchmark_bond: Benchmark bond
            yield_curves: Yield curves
            adjustment_method: Risk adjustment method
            
        Returns:
            Dictionary of risk-adjusted metrics
        """
        metrics = {}
        
        # Get basic bond metrics
        target_duration = self._get_bond_duration(target_bond, yield_curves)
        benchmark_duration = self._get_bond_duration(benchmark_bond, yield_curves)
        
        target_yield = self._get_bond_yield(target_bond, yield_curves)
        benchmark_yield = self._get_bond_yield(benchmark_bond, yield_curves)
        
        if adjustment_method == RiskAdjustmentMethod.DURATION_NEUTRAL:
            # Adjust for duration differences
            duration_ratio = target_duration / benchmark_duration if benchmark_duration > 0 else 1.0
            metrics['duration_adjusted_yield'] = target_yield - (benchmark_yield * duration_ratio)
            metrics['duration_ratio'] = duration_ratio
            
        elif adjustment_method == RiskAdjustmentMethod.BETA_ADJUSTED:
            # Calculate beta and adjust
            beta = self._calculate_bond_beta(target_bond, benchmark_bond, yield_curves)
            metrics['beta_adjusted_return'] = target_yield - (benchmark_yield * beta)
            metrics['beta'] = beta
            
        elif adjustment_method == RiskAdjustmentMethod.VOLATILITY_ADJUSTED:
            # Adjust for volatility differences
            target_vol = self._estimate_bond_volatility(target_bond)
            benchmark_vol = self._estimate_bond_volatility(benchmark_bond)
            vol_ratio = target_vol / benchmark_vol if benchmark_vol > 0 else 1.0
            metrics['volatility_adjusted_return'] = target_yield - (benchmark_yield * vol_ratio)
            metrics['volatility_ratio'] = vol_ratio
            
        elif adjustment_method == RiskAdjustmentMethod.VAR_ADJUSTED:
            # Adjust for Value-at-Risk differences
            target_var = self._calculate_bond_var(target_bond, yield_curves)
            benchmark_var = self._calculate_bond_var(benchmark_bond, yield_curves)
            var_ratio = target_var / benchmark_var if benchmark_var > 0 else 1.0
            metrics['var_adjusted_return'] = target_yield - (benchmark_yield * var_ratio)
            metrics['var_ratio'] = var_ratio
            
        elif adjustment_method == RiskAdjustmentMethod.SHARPE_ADJUSTED:
            # Calculate Sharpe ratios
            target_sharpe = self._calculate_bond_sharpe(target_bond, yield_curves)
            benchmark_sharpe = self._calculate_bond_sharpe(benchmark_bond, yield_curves)
            metrics['sharpe_ratio_target'] = target_sharpe
            metrics['sharpe_ratio_benchmark'] = benchmark_sharpe
            metrics['sharpe_difference'] = target_sharpe - benchmark_sharpe
            
        elif adjustment_method == RiskAdjustmentMethod.INFORMATION_RATIO_ADJUSTED:
            # Calculate information ratio
            info_ratio = self._calculate_information_ratio(target_bond, benchmark_bond, yield_curves)
            metrics['information_ratio'] = info_ratio
        
        return metrics
    
    def generate_trading_signals(self, 
                               comparison_results: List[ComparisonResult],
                               signal_threshold: float = 1.5) -> Dict[str, Dict[str, float]]:
        """
        Generate trading signals based on relative value analysis.
        
        Args:
            comparison_results: List of comparison results
            signal_threshold: Z-score threshold for signals
            
        Returns:
            Dictionary of trading signals by bond
        """
        signals = {}
        
        for result in comparison_results:
            bond_signals = {}
            
            # Aggregate z-scores across metrics
            z_scores = []
            for metric, analysis in result.spread_analyses.items():
                z_scores.append(analysis.z_score)
            
            avg_z_score = np.mean(z_scores) if z_scores else 0.0
            
            # Generate signal based on z-score
            if avg_z_score > signal_threshold:
                signal_strength = min(1.0, abs(avg_z_score) / 3.0)  # Cap at 3 sigma
                bond_signals['signal'] = 'BUY'
                bond_signals['strength'] = signal_strength
                bond_signals['reason'] = f'Cheap vs benchmark (Z-score: {avg_z_score:.2f})'
                
            elif avg_z_score < -signal_threshold:
                signal_strength = min(1.0, abs(avg_z_score) / 3.0)
                bond_signals['signal'] = 'SELL'
                bond_signals['strength'] = signal_strength
                bond_signals['reason'] = f'Expensive vs benchmark (Z-score: {avg_z_score:.2f})'
                
            else:
                bond_signals['signal'] = 'HOLD'
                bond_signals['strength'] = 0.0
                bond_signals['reason'] = f'Fair value (Z-score: {avg_z_score:.2f})'
            
            # Add momentum and carry considerations
            bond_signals['momentum_score'] = result.momentum_score
            bond_signals['carry_score'] = result.carry_score
            bond_signals['cheapness_score'] = result.cheapness_score
            
            signals[result.target_bond_id] = bond_signals
        
        return signals
    
    def _compare_bonds(self, 
                      target_bond: SovereignBond,
                      benchmark_bond: SovereignBond,
                      inputs: RelativeValueInput) -> ComparisonResult:
        """Compare two bonds using relative value metrics."""
        
        # Calculate spread metrics
        spread_values = self.calculate_spread_metrics(
            target_bond, benchmark_bond, inputs.yield_curves, inputs.metrics
        )
        
        # Perform spread analysis for each metric
        spread_analyses = {}
        for metric, current_spread in spread_values.items():
            spread_analyses[metric] = self._analyze_spread_history(
                current_spread, metric, target_bond, benchmark_bond, inputs.lookback_days
            )
        
        # Calculate risk-adjusted metrics
        risk_metrics = self.calculate_risk_adjusted_metrics(
            target_bond, benchmark_bond, inputs.yield_curves, inputs.risk_adjustment
        )
        
        # Calculate relative value scores
        cheapness_score = self._calculate_cheapness_score(spread_analyses)
        momentum_score = self._calculate_momentum_score(spread_analyses)
        carry_score = self._calculate_carry_score(target_bond, benchmark_bond, inputs.yield_curves, inputs.horizon_days)
        
        # Generate recommendation
        recommendation, conviction = self._generate_recommendation(
            cheapness_score, momentum_score, carry_score, spread_analyses
        )
        
        # Calculate confidence interval
        confidence_interval = self._calculate_confidence_interval(
            spread_analyses, inputs.confidence_level
        )
        
        return ComparisonResult(
            target_bond_id=target_bond.isin,
            benchmark_id=benchmark_bond.isin,
            spread_analyses=spread_analyses,
            risk_adjusted_return=risk_metrics.get('duration_adjusted_yield', 0.0),
            risk_adjusted_spread=risk_metrics.get('duration_adjusted_yield', 0.0) * 10000,
            beta_to_benchmark=risk_metrics.get('beta', 1.0),
            cheapness_score=cheapness_score,
            momentum_score=momentum_score,
            carry_score=carry_score,
            significance_level=1.0 - inputs.confidence_level,
            confidence_interval=confidence_interval,
            recommendation=recommendation,
            conviction_level=conviction
        )
    
    def _get_bond_yield(self, bond: SovereignBond, yield_curves: Dict[str, YieldCurve]) -> float:
        """Get bond yield from yield curve."""
        currency_key = bond.currency.value if hasattr(bond.currency, 'value') else str(bond.currency)
        
        if currency_key in yield_curves:
            yield_curve = yield_curves[currency_key]
            # Calculate time to maturity
            days_to_maturity = (bond.maturity_date - date.today()).days
            years_to_maturity = days_to_maturity / 365.25
            
            # Get yield from curve
            return yield_curve.get_yield(years_to_maturity)
        else:
            # Fallback to bond's current yield if available
            return getattr(bond, 'current_yield', 0.05)  # 5% default
    
    def _get_bond_duration(self, bond: SovereignBond, yield_curves: Dict[str, YieldCurve]) -> float:
        """Calculate bond duration."""
        # Simplified duration calculation
        yield_rate = self._get_bond_yield(bond, yield_curves)
        days_to_maturity = (bond.maturity_date - date.today()).days
        years_to_maturity = days_to_maturity / 365.25
        
        # Modified duration approximation
        duration = years_to_maturity / (1 + yield_rate)
        return duration
    
    def _calculate_z_spread(self, 
                          target_bond: SovereignBond,
                          benchmark_bond: SovereignBond,
                          yield_curves: Dict[str, YieldCurve]) -> float:
        """Calculate Z-spread (zero-volatility spread)."""
        # Simplified Z-spread calculation
        target_yield = self._get_bond_yield(target_bond, yield_curves)
        benchmark_yield = self._get_bond_yield(benchmark_bond, yield_curves)
        
        # Z-spread approximation
        z_spread = (target_yield - benchmark_yield) * 10000  # bps
        return z_spread
    
    def _calculate_oas(self, 
                      target_bond: SovereignBond,
                      benchmark_bond: SovereignBond,
                      yield_curves: Dict[str, YieldCurve]) -> float:
        """Calculate Option-Adjusted Spread."""
        # Simplified OAS calculation (would need option pricing model in practice)
        z_spread = self._calculate_z_spread(target_bond, benchmark_bond, yield_curves)
        
        # Assume option value is small for government bonds
        option_value = 5.0  # 5 bps typical option value
        oas = z_spread - option_value
        
        return oas
    
    def _calculate_asset_swap_spread(self, 
                                   target_bond: SovereignBond,
                                   benchmark_bond: SovereignBond,
                                   yield_curves: Dict[str, YieldCurve]) -> float:
        """Calculate Asset Swap Spread."""
        # Simplified asset swap spread
        target_yield = self._get_bond_yield(target_bond, yield_curves)
        
        # Get swap rate (approximated from yield curve)
        currency_key = target_bond.currency.value if hasattr(target_bond.currency, 'value') else str(target_bond.currency)
        if currency_key in yield_curves:
            days_to_maturity = (target_bond.maturity_date - date.today()).days
            years_to_maturity = days_to_maturity / 365.25
            swap_rate = yield_curves[currency_key].get_yield(years_to_maturity)
        else:
            swap_rate = target_yield  # Fallback
        
        asset_swap_spread = (target_yield - swap_rate) * 10000  # bps
        return asset_swap_spread
    
    def _calculate_credit_spread(self, 
                               target_bond: SovereignBond,
                               benchmark_bond: SovereignBond,
                               yield_curves: Dict[str, YieldCurve]) -> float:
        """Calculate Credit Spread."""
        # Credit spread vs risk-free rate
        target_yield = self._get_bond_yield(target_bond, yield_curves)
        
        # Use government curve as risk-free benchmark
        currency_key = target_bond.currency.value if hasattr(target_bond.currency, 'value') else str(target_bond.currency)
        if currency_key in yield_curves:
            days_to_maturity = (target_bond.maturity_date - date.today()).days
            years_to_maturity = days_to_maturity / 365.25
            risk_free_rate = yield_curves[currency_key].get_yield(years_to_maturity)
        else:
            risk_free_rate = 0.02  # 2% default risk-free rate
        
        credit_spread = (target_yield - risk_free_rate) * 10000  # bps
        return credit_spread
    
    def _calculate_duration_adjusted_spread(self, 
                                          target_bond: SovereignBond,
                                          benchmark_bond: SovereignBond,
                                          yield_curves: Dict[str, YieldCurve]) -> float:
        """Calculate Duration-Adjusted Spread."""
        target_yield = self._get_bond_yield(target_bond, yield_curves)
        benchmark_yield = self._get_bond_yield(benchmark_bond, yield_curves)
        
        target_duration = self._get_bond_duration(target_bond, yield_curves)
        benchmark_duration = self._get_bond_duration(benchmark_bond, yield_curves)
        
        # Adjust for duration difference
        if benchmark_duration > 0:
            duration_adjustment = (target_duration / benchmark_duration) - 1.0
            adjusted_spread = ((target_yield - benchmark_yield) - (benchmark_yield * duration_adjustment)) * 10000
        else:
            adjusted_spread = (target_yield - benchmark_yield) * 10000
        
        return adjusted_spread
    
    def _calculate_convexity_adjusted_spread(self, 
                                           target_bond: SovereignBond,
                                           benchmark_bond: SovereignBond,
                                           yield_curves: Dict[str, YieldCurve]) -> float:
        """Calculate Convexity-Adjusted Spread."""
        # Simplified convexity adjustment
        duration_adjusted_spread = self._calculate_duration_adjusted_spread(
            target_bond, benchmark_bond, yield_curves
        )
        
        # Estimate convexity (simplified)
        target_duration = self._get_bond_duration(target_bond, yield_curves)
        convexity_adjustment = target_duration ** 2 * 0.01  # Simplified convexity effect
        
        convexity_adjusted_spread = duration_adjusted_spread + convexity_adjustment
        return convexity_adjusted_spread
    
    def _calculate_carry_roll_down(self, 
                                 target_bond: SovereignBond,
                                 benchmark_bond: SovereignBond,
                                 yield_curves: Dict[str, YieldCurve]) -> float:
        """Calculate Carry and Roll-Down."""
        # Current yield
        current_yield = self._get_bond_yield(target_bond, yield_curves)
        
        # Estimate yield after roll-down (30 days)
        days_to_maturity = (target_bond.maturity_date - date.today()).days
        future_days_to_maturity = days_to_maturity - 30
        future_years_to_maturity = future_days_to_maturity / 365.25
        
        currency_key = target_bond.currency.value if hasattr(target_bond.currency, 'value') else str(target_bond.currency)
        if currency_key in yield_curves and future_years_to_maturity > 0:
            future_yield = yield_curves[currency_key].get_yield(future_years_to_maturity)
        else:
            future_yield = current_yield
        
        # Carry and roll-down return
        carry_return = current_yield * (30 / 365.25)  # 30-day carry
        roll_down_return = (current_yield - future_yield) * self._get_bond_duration(target_bond, yield_curves)
        
        total_return = (carry_return + roll_down_return) * 10000  # bps
        return total_return
    
    def _calculate_breakeven_spread(self, 
                                  target_bond: SovereignBond,
                                  benchmark_bond: SovereignBond,
                                  yield_curves: Dict[str, YieldCurve]) -> float:
        """Calculate Breakeven Spread."""
        # Simplified breakeven calculation
        target_yield = self._get_bond_yield(target_bond, yield_curves)
        benchmark_yield = self._get_bond_yield(benchmark_bond, yield_curves)
        
        target_duration = self._get_bond_duration(target_bond, yield_curves)
        
        # Breakeven spread change needed for equal returns
        yield_spread = target_yield - benchmark_yield
        breakeven_spread = yield_spread * target_duration * 10000  # bps
        
        return breakeven_spread
    
    def _calculate_relative_cheapness(self, 
                                    target_bond: SovereignBond,
                                    benchmark_bond: SovereignBond,
                                    yield_curves: Dict[str, YieldCurve]) -> float:
        """Calculate Relative Cheapness Score."""
        # Combine multiple spread metrics for cheapness score
        yield_spread = (self._get_bond_yield(target_bond, yield_curves) - 
                       self._get_bond_yield(benchmark_bond, yield_curves)) * 10000
        
        z_spread = self._calculate_z_spread(target_bond, benchmark_bond, yield_curves)
        credit_spread = self._calculate_credit_spread(target_bond, benchmark_bond, yield_curves)
        
        # Weighted average of spreads
        cheapness_score = (yield_spread * 0.4 + z_spread * 0.4 + credit_spread * 0.2)
        
        return cheapness_score
    
    def _analyze_spread_history(self, 
                              current_spread: float,
                              metric: RelativeValueMetric,
                              target_bond: SovereignBond,
                              benchmark_bond: SovereignBond,
                              lookback_days: int) -> SpreadAnalysis:
        """Analyze spread history and statistics."""
        
        # Generate synthetic historical spreads for demonstration
        np.random.seed(hash(target_bond.isin + benchmark_bond.isin) % 2**32)
        
        # Create realistic spread history
        base_spread = current_spread
        volatility = abs(base_spread) * 0.2  # 20% of spread level
        
        historical_spreads = []
        for i in range(lookback_days):
            # Mean-reverting process
            if i == 0:
                spread = base_spread
            else:
                prev_spread = historical_spreads[-1]
                mean_reversion = 0.02 * (base_spread - prev_spread)
                noise = np.random.normal(0, volatility)
                spread = prev_spread + mean_reversion + noise
            
            historical_spreads.append(spread)
        
        historical_spreads = np.array(historical_spreads)
        
        # Calculate statistics
        historical_mean = np.mean(historical_spreads)
        historical_std = np.std(historical_spreads)
        
        # Percentile rank
        percentile_rank = stats.percentileofscore(historical_spreads, current_spread) / 100.0
        
        # Z-score
        if historical_std > 0:
            z_score = (current_spread - historical_mean) / historical_std
        else:
            z_score = 0.0
        
        # Min/max/median
        min_spread = np.min(historical_spreads)
        max_spread = np.max(historical_spreads)
        median_spread = np.median(historical_spreads)
        
        # Trend analysis
        x = np.arange(len(historical_spreads))
        if len(historical_spreads) > 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, historical_spreads)
            trend_slope = slope
            trend_r_squared = r_value ** 2
        else:
            trend_slope = 0.0
            trend_r_squared = 0.0
        
        # Momentum score (recent vs long-term average)
        if len(historical_spreads) >= 20:
            recent_avg = np.mean(historical_spreads[-20:])  # Last 20 days
            long_term_avg = np.mean(historical_spreads[:-20])  # Earlier period
            momentum_score = (recent_avg - long_term_avg) / historical_std if historical_std > 0 else 0.0
        else:
            momentum_score = 0.0
        
        return SpreadAnalysis(
            current_spread=current_spread,
            historical_mean=historical_mean,
            historical_std=historical_std,
            percentile_rank=percentile_rank,
            z_score=z_score,
            min_spread=min_spread,
            max_spread=max_spread,
            median_spread=median_spread,
            trend_slope=trend_slope,
            trend_r_squared=trend_r_squared,
            momentum_score=momentum_score
        )
    
    def _calculate_bond_beta(self, 
                           target_bond: SovereignBond,
                           benchmark_bond: SovereignBond,
                           yield_curves: Dict[str, YieldCurve]) -> float:
        """Calculate bond beta vs benchmark."""
        # Simplified beta calculation using duration ratio
        target_duration = self._get_bond_duration(target_bond, yield_curves)
        benchmark_duration = self._get_bond_duration(benchmark_bond, yield_curves)
        
        if benchmark_duration > 0:
            beta = target_duration / benchmark_duration
        else:
            beta = 1.0
        
        return beta
    
    def _estimate_bond_volatility(self, bond: SovereignBond) -> float:
        """Estimate bond volatility."""
        # Simplified volatility estimation based on duration and credit quality
        days_to_maturity = (bond.maturity_date - date.today()).days
        years_to_maturity = days_to_maturity / 365.25
        
        # Base volatility increases with duration
        base_vol = min(0.15, 0.02 + 0.01 * years_to_maturity)  # 2% + 1% per year, max 15%
        
        return base_vol
    
    def _calculate_bond_var(self, 
                          bond: SovereignBond,
                          yield_curves: Dict[str, YieldCurve],
                          confidence_level: float = 0.95) -> float:
        """Calculate bond Value-at-Risk."""
        volatility = self._estimate_bond_volatility(bond)
        duration = self._get_bond_duration(bond, yield_curves)
        
        # VaR calculation (assuming normal distribution)
        z_score = stats.norm.ppf(confidence_level)
        var = volatility * duration * z_score
        
        return var
    
    def _calculate_bond_sharpe(self, 
                             bond: SovereignBond,
                             yield_curves: Dict[str, YieldCurve],
                             risk_free_rate: float = 0.02) -> float:
        """Calculate bond Sharpe ratio."""
        bond_yield = self._get_bond_yield(bond, yield_curves)
        volatility = self._estimate_bond_volatility(bond)
        
        if volatility > 0:
            sharpe_ratio = (bond_yield - risk_free_rate) / volatility
        else:
            sharpe_ratio = 0.0
        
        return sharpe_ratio
    
    def _calculate_information_ratio(self, 
                                   target_bond: SovereignBond,
                                   benchmark_bond: SovereignBond,
                                   yield_curves: Dict[str, YieldCurve]) -> float:
        """Calculate information ratio vs benchmark."""
        target_yield = self._get_bond_yield(target_bond, yield_curves)
        benchmark_yield = self._get_bond_yield(benchmark_bond, yield_curves)
        
        excess_return = target_yield - benchmark_yield
        
        # Estimate tracking error (simplified)
        target_vol = self._estimate_bond_volatility(target_bond)
        benchmark_vol = self._estimate_bond_volatility(benchmark_bond)
        tracking_error = abs(target_vol - benchmark_vol)
        
        if tracking_error > 0:
            information_ratio = excess_return / tracking_error
        else:
            information_ratio = 0.0
        
        return information_ratio
    
    def _calculate_cheapness_score(self, spread_analyses: Dict[RelativeValueMetric, SpreadAnalysis]) -> float:
        """Calculate overall cheapness score."""
        z_scores = []
        
        for metric, analysis in spread_analyses.items():
            z_scores.append(analysis.z_score)
        
        if z_scores:
            avg_z_score = np.mean(z_scores)
            # Convert to -100 to +100 scale
            cheapness_score = np.tanh(avg_z_score / 2.0) * 100
        else:
            cheapness_score = 0.0
        
        return cheapness_score
    
    def _calculate_momentum_score(self, spread_analyses: Dict[RelativeValueMetric, SpreadAnalysis]) -> float:
        """Calculate momentum score."""
        momentum_scores = []
        
        for metric, analysis in spread_analyses.items():
            momentum_scores.append(analysis.momentum_score)
        
        if momentum_scores:
            avg_momentum = np.mean(momentum_scores)
            # Convert to -100 to +100 scale
            momentum_score = np.tanh(avg_momentum) * 100
        else:
            momentum_score = 0.0
        
        return momentum_score
    
    def _calculate_carry_score(self, 
                             target_bond: SovereignBond,
                             benchmark_bond: SovereignBond,
                             yield_curves: Dict[str, YieldCurve],
                             horizon_days: int) -> float:
        """Calculate carry score over investment horizon."""
        target_carry = self._calculate_carry_roll_down(target_bond, benchmark_bond, yield_curves)
        
        # Annualize carry
        annualized_carry = target_carry * (365.25 / horizon_days)
        
        # Convert to score scale
        carry_score = np.tanh(annualized_carry / 100.0) * 100  # Normalize by 100 bps
        
        return carry_score
    
    def _generate_recommendation(self, 
                               cheapness_score: float,
                               momentum_score: float,
                               carry_score: float,
                               spread_analyses: Dict[RelativeValueMetric, SpreadAnalysis]) -> Tuple[str, float]:
        """Generate trading recommendation and conviction level."""
        
        # Combine scores
        total_score = cheapness_score * 0.5 + momentum_score * 0.3 + carry_score * 0.2
        
        # Calculate conviction based on consistency across metrics
        z_scores = [analysis.z_score for analysis in spread_analyses.values()]
        if z_scores:
            z_score_std = np.std(z_scores)
            consistency = 1.0 / (1.0 + z_score_std)  # Higher consistency = lower std
        else:
            consistency = 0.5
        
        conviction = min(1.0, abs(total_score) / 50.0 * consistency)  # Scale by consistency
        
        # Generate recommendation
        if total_score > 25:
            recommendation = "BUY"
        elif total_score < -25:
            recommendation = "SELL"
        else:
            recommendation = "HOLD"
        
        return recommendation, conviction
    
    def _calculate_confidence_interval(self, 
                                     spread_analyses: Dict[RelativeValueMetric, SpreadAnalysis],
                                     confidence_level: float) -> Tuple[float, float]:
        """Calculate confidence interval for spread prediction."""
        
        # Use primary spread metric for confidence interval
        primary_analysis = next(iter(spread_analyses.values()))
        
        current_spread = primary_analysis.current_spread
        historical_std = primary_analysis.historical_std
        
        # Calculate confidence interval
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        margin_of_error = z_score * historical_std
        
        lower_bound = current_spread - margin_of_error
        upper_bound = current_spread + margin_of_error
        
        return (lower_bound, upper_bound)