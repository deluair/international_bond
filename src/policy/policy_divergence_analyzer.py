"""
Policy Divergence Analyzer Module

This module provides comprehensive analysis of central bank policy divergence
and its implications for international bond markets.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class PolicyStance(Enum):
    """Central bank policy stance"""
    VERY_DOVISH = "very_dovish"
    DOVISH = "dovish"
    NEUTRAL = "neutral"
    HAWKISH = "hawkish"
    VERY_HAWKISH = "very_hawkish"

class CentralBank(Enum):
    """Major central banks"""
    FED = "Federal Reserve"
    ECB = "European Central Bank"
    BOJ = "Bank of Japan"
    BOE = "Bank of England"
    BOC = "Bank of Canada"
    RBA = "Reserve Bank of Australia"
    SNB = "Swiss National Bank"
    PBOC = "People's Bank of China"

class PolicyTool(Enum):
    """Monetary policy tools"""
    INTEREST_RATES = "interest_rates"
    QUANTITATIVE_EASING = "quantitative_easing"
    FORWARD_GUIDANCE = "forward_guidance"
    YIELD_CURVE_CONTROL = "yield_curve_control"
    NEGATIVE_RATES = "negative_rates"
    RESERVE_REQUIREMENTS = "reserve_requirements"

@dataclass
class PolicyExpectation:
    """Market expectations for policy changes"""
    central_bank: CentralBank
    expected_rate_change: float
    probability: float
    time_horizon: int  # months
    market_pricing: float
    consensus_forecast: float
    
@dataclass
class PolicyDivergence:
    """Policy divergence between central banks"""
    bank_pair: Tuple[CentralBank, CentralBank]
    rate_differential: float
    stance_divergence: float
    policy_cycle_phase: Dict[CentralBank, str]
    divergence_trend: str  # "widening", "narrowing", "stable"
    expected_duration: int  # months
    
@dataclass
class DivergenceMetrics:
    """Comprehensive divergence metrics"""
    current_divergence: float
    historical_percentile: float
    volatility: float
    mean_reversion_speed: float
    half_life: int  # days
    correlation_breakdown: bool

@dataclass
class PolicyImpact:
    """Impact of policy divergence on markets"""
    fx_impact: float
    yield_curve_impact: Dict[str, float]  # by tenor
    credit_spread_impact: float
    volatility_impact: float
    flow_impact: Dict[str, float]  # capital flows

class PolicyDivergenceAnalyzer:
    """
    Comprehensive policy divergence analysis engine
    """
    
    def __init__(self):
        self.policy_history = {}
        self.market_data = {}
        self.divergence_cache = {}
        
    def analyze_policy_divergence(
        self,
        policy_rates: Dict[CentralBank, float],
        policy_stances: Dict[CentralBank, PolicyStance],
        market_expectations: Dict[CentralBank, PolicyExpectation]
    ) -> Dict[Tuple[CentralBank, CentralBank], PolicyDivergence]:
        """Analyze policy divergence between central bank pairs"""
        
        divergences = {}
        banks = list(policy_rates.keys())
        
        for i, bank1 in enumerate(banks):
            for bank2 in banks[i+1:]:
                # Rate differential
                rate_diff = policy_rates[bank1] - policy_rates[bank2]
                
                # Stance divergence (numerical conversion)
                stance1_num = self._stance_to_number(policy_stances[bank1])
                stance2_num = self._stance_to_number(policy_stances[bank2])
                stance_div = stance1_num - stance2_num
                
                # Policy cycle analysis
                cycle_phase = {
                    bank1: self._determine_policy_cycle(bank1, policy_rates, market_expectations),
                    bank2: self._determine_policy_cycle(bank2, policy_rates, market_expectations)
                }
                
                # Divergence trend
                trend = self._calculate_divergence_trend(bank1, bank2, rate_diff)
                
                # Expected duration
                expected_duration = self._estimate_divergence_duration(
                    bank1, bank2, market_expectations
                )
                
                divergences[(bank1, bank2)] = PolicyDivergence(
                    bank_pair=(bank1, bank2),
                    rate_differential=rate_diff,
                    stance_divergence=stance_div,
                    policy_cycle_phase=cycle_phase,
                    divergence_trend=trend,
                    expected_duration=expected_duration
                )
        
        return divergences
    
    def calculate_divergence_metrics(
        self,
        bank_pair: Tuple[CentralBank, CentralBank],
        historical_rates: Dict[CentralBank, np.ndarray],
        lookback_period: int = 252  # 1 year
    ) -> DivergenceMetrics:
        """Calculate comprehensive divergence metrics"""
        
        bank1, bank2 = bank_pair
        rates1 = historical_rates[bank1][-lookback_period:]
        rates2 = historical_rates[bank2][-lookback_period:]
        
        # Current divergence
        current_div = rates1[-1] - rates2[-1]
        
        # Historical divergence series
        historical_div = rates1 - rates2
        
        # Historical percentile
        hist_percentile = (np.sum(historical_div <= current_div) / len(historical_div)) * 100
        
        # Volatility of divergence
        div_volatility = np.std(historical_div) * np.sqrt(252)
        
        # Mean reversion analysis
        mean_div = np.mean(historical_div)
        deviations = historical_div - mean_div
        
        # Simple mean reversion speed (AR(1) coefficient)
        if len(deviations) > 1:
            lagged_deviations = deviations[:-1]
            current_deviations = deviations[1:]
            mean_reversion_speed = -np.corrcoef(lagged_deviations, current_deviations)[0, 1]
        else:
            mean_reversion_speed = 0.0
        
        # Half-life calculation
        if mean_reversion_speed > 0:
            half_life = int(np.log(0.5) / np.log(1 - mean_reversion_speed))
        else:
            half_life = 999  # Very slow mean reversion
        
        # Correlation breakdown detection
        recent_corr = np.corrcoef(rates1[-60:], rates2[-60:])[0, 1]  # 3-month correlation
        long_term_corr = np.corrcoef(rates1, rates2)[0, 1]
        correlation_breakdown = abs(recent_corr - long_term_corr) > 0.3
        
        return DivergenceMetrics(
            current_divergence=current_div,
            historical_percentile=hist_percentile,
            volatility=div_volatility,
            mean_reversion_speed=mean_reversion_speed,
            half_life=half_life,
            correlation_breakdown=correlation_breakdown
        )
    
    def predict_policy_changes(
        self,
        central_bank: CentralBank,
        economic_indicators: Dict[str, float],
        market_expectations: PolicyExpectation,
        historical_patterns: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Predict future policy changes based on economic conditions"""
        
        # Economic indicator analysis
        inflation = economic_indicators.get('inflation', 2.0)
        unemployment = economic_indicators.get('unemployment', 5.0)
        gdp_growth = economic_indicators.get('gdp_growth', 2.0)
        
        # Policy reaction function (simplified Taylor rule)
        neutral_rate = 2.5
        inflation_target = 2.0
        
        taylor_rate = neutral_rate + 1.5 * (inflation - inflation_target) + 0.5 * gdp_growth
        
        # Current policy rate
        current_rate = economic_indicators.get('current_rate', 2.0)
        
        # Predicted change
        predicted_change = (taylor_rate - current_rate) * 0.3  # Gradual adjustment
        
        # Probability calculation based on market expectations vs model
        model_market_diff = abs(predicted_change - market_expectations.expected_rate_change)
        probability = max(0.1, 1.0 - model_market_diff / 2.0)  # Simple probability
        
        # Time horizon
        if abs(predicted_change) > 0.5:
            time_horizon = 3  # 3 months for significant changes
        elif abs(predicted_change) > 0.25:
            time_horizon = 6  # 6 months for moderate changes
        else:
            time_horizon = 12  # 12 months for small changes
        
        return {
            'predicted_rate_change': predicted_change,
            'taylor_rule_rate': taylor_rate,
            'probability': probability,
            'time_horizon_months': time_horizon,
            'key_drivers': {
                'inflation_gap': inflation - inflation_target,
                'growth_contribution': gdp_growth * 0.5,
                'unemployment_factor': max(0, 6.0 - unemployment) * 0.1
            },
            'risk_factors': self._identify_policy_risks(central_bank, economic_indicators)
        }
    
    def calculate_policy_impact(
        self,
        divergence: PolicyDivergence,
        market_conditions: Dict[str, Any]
    ) -> PolicyImpact:
        """Calculate market impact of policy divergence"""
        
        rate_diff = divergence.rate_differential
        stance_diff = divergence.stance_divergence
        
        # FX impact (simplified)
        fx_impact = rate_diff * 2.0 + stance_diff * 0.5  # Higher rates = stronger currency
        
        # Yield curve impact by tenor
        yield_impact = {}
        tenors = ['2Y', '5Y', '10Y', '30Y']
        base_impact = rate_diff * 0.8  # 80% pass-through
        
        for i, tenor in enumerate(tenors):
            # Longer tenors less sensitive to short-term rate changes
            sensitivity = 1.0 - (i * 0.15)
            yield_impact[tenor] = base_impact * sensitivity
        
        # Credit spread impact
        # Policy divergence can affect credit spreads through risk appetite
        risk_on_factor = -stance_diff * 0.1  # Hawkish policy = risk off
        credit_impact = risk_on_factor * market_conditions.get('credit_beta', 1.0)
        
        # Volatility impact
        # Uncertainty about policy divergence increases volatility
        uncertainty_factor = abs(stance_diff) * 0.05
        vol_impact = uncertainty_factor * market_conditions.get('vol_sensitivity', 1.0)
        
        # Capital flow impact
        flow_impact = {
            'bond_flows': rate_diff * 1.5,  # Higher rates attract bond flows
            'equity_flows': -stance_diff * 0.3,  # Hawkish policy negative for equities
            'fx_flows': fx_impact * 0.8
        }
        
        return PolicyImpact(
            fx_impact=fx_impact,
            yield_curve_impact=yield_impact,
            credit_spread_impact=credit_impact,
            volatility_impact=vol_impact,
            flow_impact=flow_impact
        )
    
    def _stance_to_number(self, stance: PolicyStance) -> float:
        """Convert policy stance to numerical value"""
        stance_map = {
            PolicyStance.VERY_DOVISH: -2.0,
            PolicyStance.DOVISH: -1.0,
            PolicyStance.NEUTRAL: 0.0,
            PolicyStance.HAWKISH: 1.0,
            PolicyStance.VERY_HAWKISH: 2.0
        }
        return stance_map.get(stance, 0.0)
    
    def _determine_policy_cycle(
        self,
        central_bank: CentralBank,
        policy_rates: Dict[CentralBank, float],
        market_expectations: Dict[CentralBank, PolicyExpectation]
    ) -> str:
        """Determine current policy cycle phase"""
        
        current_rate = policy_rates[central_bank]
        expected_change = market_expectations[central_bank].expected_rate_change
        
        if expected_change > 0.25:
            return "tightening"
        elif expected_change < -0.25:
            return "easing"
        elif current_rate < 1.0:
            return "accommodative"
        elif current_rate > 4.0:
            return "restrictive"
        else:
            return "neutral"
    
    def _calculate_divergence_trend(
        self,
        bank1: CentralBank,
        bank2: CentralBank,
        current_diff: float
    ) -> str:
        """Calculate divergence trend direction"""
        
        # Simplified trend calculation (would use historical data in practice)
        # For now, use random trend based on current difference
        if abs(current_diff) > 2.0:
            return "narrowing"  # Extreme divergences tend to narrow
        elif abs(current_diff) < 0.5:
            return "widening"   # Small divergences tend to widen
        else:
            return "stable"
    
    def _estimate_divergence_duration(
        self,
        bank1: CentralBank,
        bank2: CentralBank,
        market_expectations: Dict[CentralBank, PolicyExpectation]
    ) -> int:
        """Estimate how long divergence will persist"""
        
        exp1 = market_expectations[bank1]
        exp2 = market_expectations[bank2]
        
        # Take the longer of the two time horizons
        duration = max(exp1.time_horizon, exp2.time_horizon)
        
        # Adjust based on policy uncertainty
        uncertainty1 = 1.0 - exp1.probability
        uncertainty2 = 1.0 - exp2.probability
        avg_uncertainty = (uncertainty1 + uncertainty2) / 2
        
        # Higher uncertainty = longer duration
        duration_adjustment = int(avg_uncertainty * 6)  # Up to 6 months extra
        
        return duration + duration_adjustment
    
    def _identify_policy_risks(
        self,
        central_bank: CentralBank,
        economic_indicators: Dict[str, float]
    ) -> List[str]:
        """Identify key risks to policy outlook"""
        
        risks = []
        
        inflation = economic_indicators.get('inflation', 2.0)
        unemployment = economic_indicators.get('unemployment', 5.0)
        gdp_growth = economic_indicators.get('gdp_growth', 2.0)
        
        if inflation > 4.0:
            risks.append("High inflation pressure")
        elif inflation < 1.0:
            risks.append("Deflationary risk")
        
        if unemployment > 7.0:
            risks.append("High unemployment")
        elif unemployment < 3.0:
            risks.append("Labor market overheating")
        
        if gdp_growth < 0:
            risks.append("Recession risk")
        elif gdp_growth > 4.0:
            risks.append("Overheating economy")
        
        # Central bank specific risks
        if central_bank == CentralBank.ECB:
            risks.append("Fragmentation risk")
        elif central_bank == CentralBank.BOJ:
            risks.append("Yield curve control sustainability")
        elif central_bank == CentralBank.PBOC:
            risks.append("Capital flow management")
        
        return risks
    
    def generate_policy_report(
        self,
        divergences: Dict[Tuple[CentralBank, CentralBank], PolicyDivergence],
        metrics: Dict[Tuple[CentralBank, CentralBank], DivergenceMetrics],
        predictions: Dict[CentralBank, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate comprehensive policy analysis report"""
        
        # Key divergences
        significant_divergences = [
            (pair, div) for pair, div in divergences.items()
            if abs(div.rate_differential) > 1.0 or abs(div.stance_divergence) > 1.0
        ]
        
        # Extreme divergences
        extreme_divergences = [
            (pair, metrics[pair]) for pair in divergences.keys()
            if metrics[pair].historical_percentile > 90 or metrics[pair].historical_percentile < 10
        ]
        
        # Policy change probabilities
        high_prob_changes = [
            (bank, pred) for bank, pred in predictions.items()
            if pred['probability'] > 0.7
        ]
        
        # Summary statistics
        avg_divergence = np.mean([abs(div.rate_differential) for div in divergences.values()])
        max_divergence = max([abs(div.rate_differential) for div in divergences.values()])
        
        return {
            "analysis_timestamp": datetime.now().isoformat(),
            "summary": {
                "total_bank_pairs": len(divergences),
                "significant_divergences": len(significant_divergences),
                "extreme_divergences": len(extreme_divergences),
                "average_rate_divergence": avg_divergence,
                "maximum_rate_divergence": max_divergence
            },
            "key_findings": {
                "widening_divergences": [
                    pair for pair, div in divergences.items()
                    if div.divergence_trend == "widening"
                ],
                "narrowing_divergences": [
                    pair for pair, div in divergences.items()
                    if div.divergence_trend == "narrowing"
                ],
                "correlation_breakdowns": [
                    pair for pair in divergences.keys()
                    if metrics[pair].correlation_breakdown
                ]
            },
            "policy_outlook": {
                "high_probability_changes": high_prob_changes,
                "policy_risks": {
                    bank.value: pred.get('risk_factors', [])
                    for bank, pred in predictions.items()
                }
            },
            "trading_implications": self._generate_trading_implications(
                divergences, metrics, predictions
            )
        }
    
    def _generate_trading_implications(
        self,
        divergences: Dict[Tuple[CentralBank, CentralBank], PolicyDivergence],
        metrics: Dict[Tuple[CentralBank, CentralBank], DivergenceMetrics],
        predictions: Dict[CentralBank, Dict[str, Any]]
    ) -> List[str]:
        """Generate actionable trading implications"""
        
        implications = []
        
        # Mean reversion opportunities
        mean_rev_pairs = [
            pair for pair, metric in metrics.items()
            if metric.historical_percentile > 85 and metric.mean_reversion_speed > 0.1
        ]
        
        if mean_rev_pairs:
            implications.append(
                f"Mean reversion opportunities identified in {len(mean_rev_pairs)} currency pairs"
            )
        
        # Trend continuation
        trending_pairs = [
            pair for pair, div in divergences.items()
            if div.divergence_trend == "widening" and div.expected_duration > 6
        ]
        
        if trending_pairs:
            implications.append(
                f"Trend continuation expected in {len(trending_pairs)} diverging pairs"
            )
        
        # Volatility opportunities
        high_vol_pairs = [
            pair for pair, metric in metrics.items()
            if metric.volatility > np.mean([m.volatility for m in metrics.values()]) * 1.5
        ]
        
        if high_vol_pairs:
            implications.append(
                f"Elevated volatility expected in {len(high_vol_pairs)} pairs"
            )
        
        return implications