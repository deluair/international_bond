"""
Cross-Market Analyzer Module

This module provides comprehensive cross-market analysis capabilities for international bonds,
including arbitrage detection, market correlation analysis, and cross-currency opportunities.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Set
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class MarketRegion(Enum):
    """Major bond market regions"""
    US = "US"
    EUROPE = "EUR"
    JAPAN = "JPY"
    UK = "GBP"
    CANADA = "CAD"
    AUSTRALIA = "AUD"
    EMERGING_MARKETS = "EM"
    ASIA_PACIFIC = "APAC"

class ArbitrageType(Enum):
    """Types of arbitrage opportunities"""
    YIELD_CURVE = "yield_curve"
    CREDIT_SPREAD = "credit_spread"
    CURRENCY_HEDGED = "currency_hedged"
    CROSS_CURRENCY = "cross_currency"
    CALENDAR_SPREAD = "calendar_spread"
    BUTTERFLY_SPREAD = "butterfly_spread"
    BASIS_TRADE = "basis_trade"

class CorrelationPeriod(Enum):
    """Time periods for correlation analysis"""
    ONE_MONTH = "1M"
    THREE_MONTHS = "3M"
    SIX_MONTHS = "6M"
    ONE_YEAR = "1Y"
    TWO_YEARS = "2Y"
    FIVE_YEARS = "5Y"

@dataclass
class ArbitrageOpportunity:
    """Arbitrage opportunity identification"""
    opportunity_id: str
    arbitrage_type: ArbitrageType
    markets: List[MarketRegion]
    expected_return: float
    risk_level: float
    confidence_score: float
    entry_conditions: Dict[str, Any]
    exit_conditions: Dict[str, Any]
    hedge_requirements: Dict[str, float]
    estimated_duration: int  # days
    
@dataclass
class MarketCorrelation:
    """Market correlation analysis"""
    market_pair: Tuple[MarketRegion, MarketRegion]
    correlation_coefficient: float
    rolling_correlation: np.ndarray
    correlation_stability: float
    regime_changes: List[datetime]
    current_regime: str
    
@dataclass
class CrossMarketMetrics:
    """Cross-market performance metrics"""
    market: MarketRegion
    yield_level: float
    yield_volatility: float
    credit_spread: float
    duration: float
    convexity: float
    liquidity_score: float
    fx_volatility: float
    carry_potential: float

@dataclass
class MarketDivergence:
    """Market divergence analysis"""
    markets: List[MarketRegion]
    divergence_score: float
    divergence_drivers: Dict[str, float]
    mean_reversion_probability: float
    time_to_convergence: Optional[int]
    trading_opportunities: List[ArbitrageOpportunity]

class CrossMarketAnalyzer:
    """
    Comprehensive cross-market analysis engine for international bonds
    """
    
    def __init__(self):
        self.market_data = {}
        self.correlation_cache = {}
        self.arbitrage_history = []
        
    def analyze_market_correlations(
        self,
        market_returns: Dict[MarketRegion, np.ndarray],
        periods: List[CorrelationPeriod] = None
    ) -> Dict[Tuple[MarketRegion, MarketRegion], MarketCorrelation]:
        """Analyze correlations between different bond markets"""
        
        if periods is None:
            periods = [CorrelationPeriod.THREE_MONTHS, CorrelationPeriod.ONE_YEAR]
        
        correlations = {}
        markets = list(market_returns.keys())
        
        for i, market1 in enumerate(markets):
            for market2 in markets[i+1:]:
                returns1 = market_returns[market1]
                returns2 = market_returns[market2]
                
                # Static correlation
                correlation_coeff = np.corrcoef(returns1, returns2)[0, 1]
                
                # Rolling correlation
                window_size = min(60, len(returns1) // 4)  # 60-day or 1/4 of data
                rolling_corr = self._calculate_rolling_correlation(
                    returns1, returns2, window_size
                )
                
                # Correlation stability
                stability = 1 - np.std(rolling_corr)
                
                # Regime changes (simplified)
                regime_changes = self._detect_regime_changes(rolling_corr)
                current_regime = self._classify_current_regime(correlation_coeff)
                
                correlations[(market1, market2)] = MarketCorrelation(
                    market_pair=(market1, market2),
                    correlation_coefficient=correlation_coeff,
                    rolling_correlation=rolling_corr,
                    correlation_stability=stability,
                    regime_changes=regime_changes,
                    current_regime=current_regime
                )
        
        return correlations
    
    def identify_arbitrage_opportunities(
        self,
        market_data: Dict[MarketRegion, Dict[str, Any]],
        fx_rates: Dict[str, float],
        risk_tolerance: float = 0.05
    ) -> List[ArbitrageOpportunity]:
        """Identify cross-market arbitrage opportunities"""
        
        opportunities = []
        
        # Yield curve arbitrage
        yield_arb = self._find_yield_curve_arbitrage(market_data, risk_tolerance)
        opportunities.extend(yield_arb)
        
        # Credit spread arbitrage
        credit_arb = self._find_credit_spread_arbitrage(market_data, risk_tolerance)
        opportunities.extend(credit_arb)
        
        # Currency hedged arbitrage
        fx_arb = self._find_currency_hedged_arbitrage(market_data, fx_rates, risk_tolerance)
        opportunities.extend(fx_arb)
        
        # Cross-currency basis arbitrage
        basis_arb = self._find_basis_arbitrage(market_data, fx_rates, risk_tolerance)
        opportunities.extend(basis_arb)
        
        # Sort by expected return / risk ratio
        opportunities.sort(key=lambda x: x.expected_return / max(x.risk_level, 0.001), reverse=True)
        
        return opportunities
    
    def _find_yield_curve_arbitrage(
        self,
        market_data: Dict[MarketRegion, Dict[str, Any]],
        risk_tolerance: float
    ) -> List[ArbitrageOpportunity]:
        """Find yield curve arbitrage opportunities"""
        
        opportunities = []
        
        for market, data in market_data.items():
            yield_curve = data.get('yield_curve', {})
            
            # Look for curve steepening/flattening opportunities
            if len(yield_curve) >= 3:
                tenors = sorted(yield_curve.keys())
                yields = [yield_curve[tenor] for tenor in tenors]
                
                # Calculate curve slopes
                short_slope = yields[1] - yields[0]
                long_slope = yields[-1] - yields[-2]
                
                # Identify steep/flat regions
                if abs(short_slope - long_slope) > 0.5:  # 50bp difference
                    expected_return = abs(short_slope - long_slope) * 0.3  # 30% capture
                    risk_level = np.std(yields) * 2
                    
                    if risk_level <= risk_tolerance:
                        opportunities.append(ArbitrageOpportunity(
                            opportunity_id=f"YC_{market.value}_{datetime.now().strftime('%Y%m%d')}",
                            arbitrage_type=ArbitrageType.YIELD_CURVE,
                            markets=[market],
                            expected_return=expected_return,
                            risk_level=risk_level,
                            confidence_score=0.7,
                            entry_conditions={"curve_slope_diff": short_slope - long_slope},
                            exit_conditions={"target_convergence": 0.2},
                            hedge_requirements={"duration_neutral": True},
                            estimated_duration=90
                        ))
        
        return opportunities
    
    def _find_credit_spread_arbitrage(
        self,
        market_data: Dict[MarketRegion, Dict[str, Any]],
        risk_tolerance: float
    ) -> List[ArbitrageOpportunity]:
        """Find credit spread arbitrage opportunities"""
        
        opportunities = []
        markets = list(market_data.keys())
        
        for i, market1 in enumerate(markets):
            for market2 in markets[i+1:]:
                data1 = market_data[market1]
                data2 = market_data[market2]
                
                spread1 = data1.get('credit_spread', 0)
                spread2 = data2.get('credit_spread', 0)
                
                spread_diff = abs(spread1 - spread2)
                
                # Look for significant spread differences
                if spread_diff > 0.5:  # 50bp difference
                    expected_return = spread_diff * 0.4  # 40% capture
                    risk_level = max(
                        data1.get('credit_volatility', 0.1),
                        data2.get('credit_volatility', 0.1)
                    )
                    
                    if risk_level <= risk_tolerance:
                        opportunities.append(ArbitrageOpportunity(
                            opportunity_id=f"CS_{market1.value}_{market2.value}_{datetime.now().strftime('%Y%m%d')}",
                            arbitrage_type=ArbitrageType.CREDIT_SPREAD,
                            markets=[market1, market2],
                            expected_return=expected_return,
                            risk_level=risk_level,
                            confidence_score=0.6,
                            entry_conditions={"spread_difference": spread_diff},
                            exit_conditions={"convergence_threshold": 0.1},
                            hedge_requirements={"duration_matched": True, "currency_hedged": True},
                            estimated_duration=120
                        ))
        
        return opportunities
    
    def _find_currency_hedged_arbitrage(
        self,
        market_data: Dict[MarketRegion, Dict[str, Any]],
        fx_rates: Dict[str, float],
        risk_tolerance: float
    ) -> List[ArbitrageOpportunity]:
        """Find currency hedged arbitrage opportunities"""
        
        opportunities = []
        
        # Compare hedged yields across markets
        hedged_yields = {}
        for market, data in market_data.items():
            base_yield = data.get('yield', 0)
            fx_forward_points = fx_rates.get(f"{market.value}_forward", 0)
            hedged_yield = base_yield - fx_forward_points
            hedged_yields[market] = hedged_yield
        
        # Find yield pickup opportunities
        markets = list(hedged_yields.keys())
        for i, market1 in enumerate(markets):
            for market2 in markets[i+1:]:
                yield_diff = hedged_yields[market1] - hedged_yields[market2]
                
                if abs(yield_diff) > 0.3:  # 30bp pickup
                    expected_return = abs(yield_diff) * 0.8  # 80% capture (hedged)
                    fx_vol1 = market_data[market1].get('fx_volatility', 0.1)
                    fx_vol2 = market_data[market2].get('fx_volatility', 0.1)
                    risk_level = np.sqrt(fx_vol1**2 + fx_vol2**2) * 0.1  # Residual FX risk
                    
                    if risk_level <= risk_tolerance:
                        opportunities.append(ArbitrageOpportunity(
                            opportunity_id=f"FXH_{market1.value}_{market2.value}_{datetime.now().strftime('%Y%m%d')}",
                            arbitrage_type=ArbitrageType.CURRENCY_HEDGED,
                            markets=[market1, market2],
                            expected_return=expected_return,
                            risk_level=risk_level,
                            confidence_score=0.8,
                            entry_conditions={"hedged_yield_diff": yield_diff},
                            exit_conditions={"yield_convergence": 0.1},
                            hedge_requirements={"fx_hedge_ratio": 1.0},
                            estimated_duration=60
                        ))
        
        return opportunities
    
    def _find_basis_arbitrage(
        self,
        market_data: Dict[MarketRegion, Dict[str, Any]],
        fx_rates: Dict[str, float],
        risk_tolerance: float
    ) -> List[ArbitrageOpportunity]:
        """Find cross-currency basis arbitrage opportunities"""
        
        opportunities = []
        
        # Simplified basis calculation
        for market, data in market_data.items():
            if market == MarketRegion.US:
                continue
                
            usd_yield = market_data.get(MarketRegion.US, {}).get('yield', 0)
            local_yield = data.get('yield', 0)
            fx_swap_rate = fx_rates.get(f"{market.value}_swap", 0)
            
            basis = local_yield - usd_yield - fx_swap_rate
            
            if abs(basis) > 0.2:  # 20bp basis
                expected_return = abs(basis) * 0.6  # 60% capture
                risk_level = data.get('basis_volatility', 0.05)
                
                if risk_level <= risk_tolerance:
                    opportunities.append(ArbitrageOpportunity(
                        opportunity_id=f"BASIS_{market.value}_USD_{datetime.now().strftime('%Y%m%d')}",
                        arbitrage_type=ArbitrageType.BASIS_TRADE,
                        markets=[market, MarketRegion.US],
                        expected_return=expected_return,
                        risk_level=risk_level,
                        confidence_score=0.7,
                        entry_conditions={"basis_level": basis},
                        exit_conditions={"basis_normalization": 0.05},
                        hedge_requirements={"cross_currency_swap": True},
                        estimated_duration=180
                    ))
        
        return opportunities
    
    def analyze_market_divergence(
        self,
        market_metrics: Dict[MarketRegion, CrossMarketMetrics],
        historical_data: Dict[MarketRegion, np.ndarray]
    ) -> List[MarketDivergence]:
        """Analyze market divergence and convergence opportunities"""
        
        divergences = []
        markets = list(market_metrics.keys())
        
        # Calculate pairwise divergences
        for i, market1 in enumerate(markets):
            for market2 in markets[i+1:]:
                metrics1 = market_metrics[market1]
                metrics2 = market_metrics[market2]
                
                # Yield divergence
                yield_div = abs(metrics1.yield_level - metrics2.yield_level)
                
                # Spread divergence
                spread_div = abs(metrics1.credit_spread - metrics2.credit_spread)
                
                # Combined divergence score
                divergence_score = yield_div + spread_div
                
                # Historical correlation for mean reversion probability
                returns1 = historical_data[market1]
                returns2 = historical_data[market2]
                correlation = np.corrcoef(returns1, returns2)[0, 1]
                
                mean_reversion_prob = max(0, correlation) * 0.8  # Higher correlation = higher mean reversion
                
                # Estimate convergence time (simplified)
                volatility_avg = (metrics1.yield_volatility + metrics2.yield_volatility) / 2
                time_to_convergence = int(divergence_score / volatility_avg * 30) if volatility_avg > 0 else None
                
                # Generate trading opportunities
                trading_opps = []
                if divergence_score > 1.0 and mean_reversion_prob > 0.5:
                    # Create convergence trade opportunity
                    trading_opps.append(ArbitrageOpportunity(
                        opportunity_id=f"CONV_{market1.value}_{market2.value}_{datetime.now().strftime('%Y%m%d')}",
                        arbitrage_type=ArbitrageType.CROSS_CURRENCY,
                        markets=[market1, market2],
                        expected_return=divergence_score * mean_reversion_prob * 0.3,
                        risk_level=volatility_avg,
                        confidence_score=mean_reversion_prob,
                        entry_conditions={"divergence_threshold": divergence_score},
                        exit_conditions={"convergence_target": divergence_score * 0.3},
                        hedge_requirements={"duration_neutral": True, "currency_hedged": True},
                        estimated_duration=time_to_convergence or 90
                    ))
                
                divergences.append(MarketDivergence(
                    markets=[market1, market2],
                    divergence_score=divergence_score,
                    divergence_drivers={
                        "yield_difference": yield_div,
                        "spread_difference": spread_div,
                        "correlation": correlation
                    },
                    mean_reversion_probability=mean_reversion_prob,
                    time_to_convergence=time_to_convergence,
                    trading_opportunities=trading_opps
                ))
        
        return divergences
    
    def _calculate_rolling_correlation(
        self,
        returns1: np.ndarray,
        returns2: np.ndarray,
        window: int
    ) -> np.ndarray:
        """Calculate rolling correlation between two return series"""
        
        correlations = []
        for i in range(window, len(returns1)):
            corr = np.corrcoef(
                returns1[i-window:i],
                returns2[i-window:i]
            )[0, 1]
            correlations.append(corr)
        
        return np.array(correlations)
    
    def _detect_regime_changes(
        self,
        rolling_correlation: np.ndarray,
        threshold: float = 0.3
    ) -> List[datetime]:
        """Detect regime changes in correlation"""
        
        # Simplified regime change detection
        changes = []
        if len(rolling_correlation) < 2:
            return changes
        
        for i in range(1, len(rolling_correlation)):
            if abs(rolling_correlation[i] - rolling_correlation[i-1]) > threshold:
                # Approximate date (would need actual dates in real implementation)
                change_date = datetime.now() - timedelta(days=len(rolling_correlation)-i)
                changes.append(change_date)
        
        return changes
    
    def _classify_current_regime(self, correlation: float) -> str:
        """Classify current correlation regime"""
        
        if correlation > 0.7:
            return "HIGH_CORRELATION"
        elif correlation > 0.3:
            return "MODERATE_CORRELATION"
        elif correlation > -0.3:
            return "LOW_CORRELATION"
        else:
            return "NEGATIVE_CORRELATION"
    
    def generate_cross_market_report(
        self,
        correlations: Dict[Tuple[MarketRegion, MarketRegion], MarketCorrelation],
        opportunities: List[ArbitrageOpportunity],
        divergences: List[MarketDivergence]
    ) -> Dict[str, Any]:
        """Generate comprehensive cross-market analysis report"""
        
        # Correlation summary
        correlation_summary = {
            "average_correlation": np.mean([c.correlation_coefficient for c in correlations.values()]),
            "correlation_range": {
                "min": min([c.correlation_coefficient for c in correlations.values()]),
                "max": max([c.correlation_coefficient for c in correlations.values()])
            },
            "stable_pairs": [
                pair for pair, corr in correlations.items() 
                if corr.correlation_stability > 0.7
            ],
            "regime_changes_detected": sum([
                len(corr.regime_changes) for corr in correlations.values()
            ])
        }
        
        # Opportunity summary
        opportunity_summary = {
            "total_opportunities": len(opportunities),
            "by_type": {},
            "high_confidence": [opp for opp in opportunities if opp.confidence_score > 0.7],
            "expected_returns": {
                "mean": np.mean([opp.expected_return for opp in opportunities]),
                "max": max([opp.expected_return for opp in opportunities]) if opportunities else 0,
                "total_potential": sum([opp.expected_return for opp in opportunities])
            }
        }
        
        # Count by arbitrage type
        for opp in opportunities:
            arb_type = opp.arbitrage_type.value
            opportunity_summary["by_type"][arb_type] = opportunity_summary["by_type"].get(arb_type, 0) + 1
        
        # Divergence summary
        divergence_summary = {
            "total_divergences": len(divergences),
            "high_divergence_pairs": [
                div for div in divergences if div.divergence_score > 1.5
            ],
            "mean_reversion_candidates": [
                div for div in divergences if div.mean_reversion_probability > 0.6
            ]
        }
        
        return {
            "analysis_timestamp": datetime.now().isoformat(),
            "correlation_analysis": correlation_summary,
            "arbitrage_opportunities": opportunity_summary,
            "market_divergences": divergence_summary,
            "top_opportunities": opportunities[:5],  # Top 5 opportunities
            "recommendations": self._generate_market_recommendations(
                correlations, opportunities, divergences
            )
        }
    
    def _generate_market_recommendations(
        self,
        correlations: Dict[Tuple[MarketRegion, MarketRegion], MarketCorrelation],
        opportunities: List[ArbitrageOpportunity],
        divergences: List[MarketDivergence]
    ) -> List[str]:
        """Generate actionable market recommendations"""
        
        recommendations = []
        
        # High-confidence opportunities
        high_conf_opps = [opp for opp in opportunities if opp.confidence_score > 0.8]
        if high_conf_opps:
            recommendations.append(
                f"Consider {len(high_conf_opps)} high-confidence arbitrage opportunities "
                f"with average expected return of {np.mean([opp.expected_return for opp in high_conf_opps]):.2%}"
            )
        
        # Correlation regime changes
        unstable_pairs = [
            pair for pair, corr in correlations.items() 
            if corr.correlation_stability < 0.5
        ]
        if unstable_pairs:
            recommendations.append(
                f"Monitor {len(unstable_pairs)} market pairs showing correlation instability"
            )
        
        # Mean reversion opportunities
        mean_rev_divs = [div for div in divergences if div.mean_reversion_probability > 0.7]
        if mean_rev_divs:
            recommendations.append(
                f"Strong mean reversion signals detected in {len(mean_rev_divs)} market pairs"
            )
        
        return recommendations