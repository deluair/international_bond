"""
Scenario Analysis Module

This module provides comprehensive scenario analysis capabilities for international bond portfolios,
including economic scenarios, policy scenarios, and custom what-if analysis.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class ScenarioType(Enum):
    """Types of scenarios"""
    ECONOMIC_SCENARIO = "economic_scenario"
    POLICY_SCENARIO = "policy_scenario"
    MARKET_SCENARIO = "market_scenario"
    GEOPOLITICAL_SCENARIO = "geopolitical_scenario"
    CUSTOM_SCENARIO = "custom_scenario"

class EconomicCondition(Enum):
    """Economic conditions"""
    RECESSION = "recession"
    SLOW_GROWTH = "slow_growth"
    MODERATE_GROWTH = "moderate_growth"
    STRONG_GROWTH = "strong_growth"
    STAGFLATION = "stagflation"
    DEFLATION = "deflation"

class PolicyStance(Enum):
    """Central bank policy stances"""
    VERY_DOVISH = "very_dovish"
    DOVISH = "dovish"
    NEUTRAL = "neutral"
    HAWKISH = "hawkish"
    VERY_HAWKISH = "very_hawkish"

class MarketRegime(Enum):
    """Market regimes"""
    RISK_ON = "risk_on"
    RISK_OFF = "risk_off"
    LOW_VOLATILITY = "low_volatility"
    HIGH_VOLATILITY = "high_volatility"
    FLIGHT_TO_QUALITY = "flight_to_quality"
    CARRY_TRADE = "carry_trade"

@dataclass
class ScenarioParameter:
    """Individual scenario parameter"""
    parameter_name: str
    base_value: float
    scenario_value: float
    change_magnitude: float
    change_percentage: float
    confidence_level: float = 0.5  # 0-1 scale

@dataclass
class EconomicScenario:
    """Economic scenario specification"""
    scenario_name: str
    scenario_type: ScenarioType
    economic_condition: EconomicCondition
    gdp_growth: ScenarioParameter
    inflation_rate: ScenarioParameter
    unemployment_rate: ScenarioParameter
    policy_rate: ScenarioParameter
    duration_months: int
    probability: float
    description: str

@dataclass
class PolicyScenario:
    """Central bank policy scenario"""
    scenario_name: str
    central_bank: str
    policy_stance: PolicyStance
    rate_changes: List[Tuple[datetime, float]]  # (date, rate_change_bp)
    qe_changes: Optional[float] = None  # Change in QE program size
    forward_guidance: Optional[str] = None
    probability: float = 0.5
    market_impact_estimate: Dict[str, float] = None

@dataclass
class MarketScenario:
    """Market scenario specification"""
    scenario_name: str
    market_regime: MarketRegime
    yield_curve_shifts: Dict[str, Dict[str, float]]  # country -> {tenor: shift_bp}
    credit_spread_changes: Dict[str, float]  # rating -> spread_change_bp
    fx_changes: Dict[str, float]  # currency_pair -> percentage_change
    volatility_changes: Dict[str, float]  # asset_class -> vol_change
    correlation_changes: Dict[Tuple[str, str], float]  # (asset1, asset2) -> corr_change

@dataclass
class ScenarioResult:
    """Result of scenario analysis"""
    scenario_name: str
    scenario_type: ScenarioType
    portfolio_pnl: float
    portfolio_pnl_percentage: float
    asset_level_impacts: Dict[str, float]
    risk_factor_impacts: Dict[str, float]
    duration_impact: float
    credit_impact: float
    currency_impact: float
    key_drivers: List[Tuple[str, float]]
    scenario_probability: float

@dataclass
class ScenarioComparison:
    """Comparison of multiple scenarios"""
    scenarios_analyzed: List[str]
    best_case_scenario: str
    worst_case_scenario: str
    most_likely_scenario: str
    expected_return: float  # Probability-weighted return
    risk_metrics: Dict[str, float]
    scenario_correlations: Dict[Tuple[str, str], float]

class ScenarioAnalyzer:
    """
    Comprehensive scenario analysis system for international bond portfolios
    """
    
    def __init__(self):
        self.predefined_scenarios = {}
        self.scenario_templates = {}
        self.historical_precedents = {}
        self._initialize_scenario_templates()
    
    def analyze_economic_scenario(
        self,
        portfolio_positions: Dict[str, float],
        asset_sensitivities: Dict[str, Dict[str, float]],
        scenario: EconomicScenario,
        portfolio_value: float = 1000000
    ) -> ScenarioResult:
        """Analyze impact of economic scenario on portfolio"""
        
        total_pnl = 0
        asset_impacts = {}
        risk_factor_impacts = {
            'duration': 0,
            'credit': 0,
            'currency': 0,
            'inflation': 0
        }
        
        # Calculate impacts based on economic parameters
        for asset_id, position_size in portfolio_positions.items():
            asset_pnl = 0
            
            if asset_id in asset_sensitivities:
                sensitivities = asset_sensitivities[asset_id]
                
                # GDP growth impact
                if 'gdp_sensitivity' in sensitivities:
                    gdp_impact = (position_size * sensitivities['gdp_sensitivity'] * 
                                scenario.gdp_growth.change_percentage / 100)
                    asset_pnl += gdp_impact
                    risk_factor_impacts['duration'] += gdp_impact
                
                # Inflation impact
                if 'inflation_sensitivity' in sensitivities:
                    inflation_impact = (position_size * sensitivities['inflation_sensitivity'] * 
                                      scenario.inflation_rate.change_percentage / 100)
                    asset_pnl += inflation_impact
                    risk_factor_impacts['inflation'] += inflation_impact
                
                # Policy rate impact (duration effect)
                if 'duration' in sensitivities:
                    rate_impact = (position_size * sensitivities['duration'] * 
                                 scenario.policy_rate.change_magnitude / 10000)  # bp to decimal
                    asset_pnl += rate_impact
                    risk_factor_impacts['duration'] += rate_impact
                
                # Credit impact (based on economic condition)
                if 'credit_sensitivity' in sensitivities:
                    credit_multiplier = self._get_credit_multiplier(scenario.economic_condition)
                    credit_impact = (position_size * sensitivities['credit_sensitivity'] * 
                                   credit_multiplier)
                    asset_pnl += credit_impact
                    risk_factor_impacts['credit'] += credit_impact
            
            asset_impacts[asset_id] = asset_pnl
            total_pnl += asset_pnl
        
        # Calculate percentage impact
        pnl_percentage = (total_pnl / portfolio_value) * 100
        
        # Identify key drivers
        key_drivers = sorted(
            risk_factor_impacts.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:3]
        
        return ScenarioResult(
            scenario_name=scenario.scenario_name,
            scenario_type=scenario.scenario_type,
            portfolio_pnl=total_pnl,
            portfolio_pnl_percentage=pnl_percentage,
            asset_level_impacts=asset_impacts,
            risk_factor_impacts=risk_factor_impacts,
            duration_impact=risk_factor_impacts['duration'],
            credit_impact=risk_factor_impacts['credit'],
            currency_impact=risk_factor_impacts['currency'],
            key_drivers=key_drivers,
            scenario_probability=scenario.probability
        )
    
    def analyze_policy_scenario(
        self,
        portfolio_positions: Dict[str, float],
        asset_sensitivities: Dict[str, Dict[str, float]],
        scenario: PolicyScenario,
        portfolio_value: float = 1000000
    ) -> ScenarioResult:
        """Analyze impact of policy scenario on portfolio"""
        
        total_pnl = 0
        asset_impacts = {}
        risk_factor_impacts = {
            'duration': 0,
            'credit': 0,
            'currency': 0,
            'policy': 0
        }
        
        # Calculate total rate change
        total_rate_change = sum(change for _, change in scenario.rate_changes)
        
        for asset_id, position_size in portfolio_positions.items():
            asset_pnl = 0
            
            if asset_id in asset_sensitivities:
                sensitivities = asset_sensitivities[asset_id]
                
                # Duration impact from rate changes
                if 'duration' in sensitivities:
                    duration_impact = (position_size * sensitivities['duration'] * 
                                     total_rate_change / 10000)  # bp to decimal
                    asset_pnl += duration_impact
                    risk_factor_impacts['duration'] += duration_impact
                
                # Policy-specific impacts
                if 'policy_sensitivity' in sensitivities:
                    policy_multiplier = self._get_policy_multiplier(scenario.policy_stance)
                    policy_impact = (position_size * sensitivities['policy_sensitivity'] * 
                                   policy_multiplier)
                    asset_pnl += policy_impact
                    risk_factor_impacts['policy'] += policy_impact
                
                # QE impact
                if scenario.qe_changes and 'qe_sensitivity' in sensitivities:
                    qe_impact = (position_size * sensitivities['qe_sensitivity'] * 
                               scenario.qe_changes / 100)
                    asset_pnl += qe_impact
                    risk_factor_impacts['duration'] += qe_impact
                
                # Currency impact (for international bonds)
                if 'currency_sensitivity' in sensitivities:
                    currency_multiplier = self._get_currency_policy_impact(scenario.policy_stance)
                    currency_impact = (position_size * sensitivities['currency_sensitivity'] * 
                                     currency_multiplier)
                    asset_pnl += currency_impact
                    risk_factor_impacts['currency'] += currency_impact
            
            asset_impacts[asset_id] = asset_pnl
            total_pnl += asset_pnl
        
        # Calculate percentage impact
        pnl_percentage = (total_pnl / portfolio_value) * 100
        
        # Identify key drivers
        key_drivers = sorted(
            risk_factor_impacts.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:3]
        
        return ScenarioResult(
            scenario_name=scenario.scenario_name,
            scenario_type=ScenarioType.POLICY_SCENARIO,
            portfolio_pnl=total_pnl,
            portfolio_pnl_percentage=pnl_percentage,
            asset_level_impacts=asset_impacts,
            risk_factor_impacts=risk_factor_impacts,
            duration_impact=risk_factor_impacts['duration'],
            credit_impact=risk_factor_impacts['credit'],
            currency_impact=risk_factor_impacts['currency'],
            key_drivers=key_drivers,
            scenario_probability=scenario.probability
        )
    
    def analyze_market_scenario(
        self,
        portfolio_positions: Dict[str, float],
        asset_characteristics: Dict[str, Dict[str, Any]],
        scenario: MarketScenario,
        portfolio_value: float = 1000000
    ) -> ScenarioResult:
        """Analyze impact of market scenario on portfolio"""
        
        total_pnl = 0
        asset_impacts = {}
        risk_factor_impacts = {
            'duration': 0,
            'credit': 0,
            'currency': 0,
            'volatility': 0
        }
        
        for asset_id, position_size in portfolio_positions.items():
            asset_pnl = 0
            
            if asset_id in asset_characteristics:
                characteristics = asset_characteristics[asset_id]
                
                # Yield curve impact
                country = characteristics.get('country', 'US')
                duration = characteristics.get('duration', 5.0)
                
                if country in scenario.yield_curve_shifts:
                    # Find closest tenor
                    tenor_str = f"{int(duration)}Y"
                    if tenor_str in scenario.yield_curve_shifts[country]:
                        yield_shift = scenario.yield_curve_shifts[country][tenor_str]
                        duration_impact = -position_size * duration * yield_shift / 10000
                        asset_pnl += duration_impact
                        risk_factor_impacts['duration'] += duration_impact
                
                # Credit spread impact
                rating = characteristics.get('rating', 'AAA')
                if rating in scenario.credit_spread_changes:
                    spread_change = scenario.credit_spread_changes[rating]
                    credit_duration = characteristics.get('credit_duration', duration * 0.8)
                    credit_impact = -position_size * credit_duration * spread_change / 10000
                    asset_pnl += credit_impact
                    risk_factor_impacts['credit'] += credit_impact
                
                # FX impact
                currency = characteristics.get('currency', 'USD')
                fx_pair = f"{currency}/USD"
                if fx_pair in scenario.fx_changes:
                    fx_change = scenario.fx_changes[fx_pair]
                    fx_impact = position_size * fx_change / 100
                    asset_pnl += fx_impact
                    risk_factor_impacts['currency'] += fx_impact
                
                # Volatility impact (affects option-like features)
                asset_class = characteristics.get('asset_class', 'government')
                if asset_class in scenario.volatility_changes:
                    vol_change = scenario.volatility_changes[asset_class]
                    convexity = characteristics.get('convexity', 0.5)
                    vol_impact = position_size * convexity * vol_change / 100
                    asset_pnl += vol_impact
                    risk_factor_impacts['volatility'] += vol_impact
            
            asset_impacts[asset_id] = asset_pnl
            total_pnl += asset_pnl
        
        # Apply correlation effects
        if scenario.correlation_changes:
            correlation_impact = self._calculate_correlation_impact(
                asset_impacts, scenario.correlation_changes
            )
            total_pnl += correlation_impact
        
        # Calculate percentage impact
        pnl_percentage = (total_pnl / portfolio_value) * 100
        
        # Identify key drivers
        key_drivers = sorted(
            risk_factor_impacts.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:3]
        
        return ScenarioResult(
            scenario_name=scenario.scenario_name,
            scenario_type=ScenarioType.MARKET_SCENARIO,
            portfolio_pnl=total_pnl,
            portfolio_pnl_percentage=pnl_percentage,
            asset_level_impacts=asset_impacts,
            risk_factor_impacts=risk_factor_impacts,
            duration_impact=risk_factor_impacts['duration'],
            credit_impact=risk_factor_impacts['credit'],
            currency_impact=risk_factor_impacts['currency'],
            key_drivers=key_drivers,
            scenario_probability=0.5  # Default probability
        )
    
    def run_multiple_scenarios(
        self,
        portfolio_positions: Dict[str, float],
        scenarios: List[Union[EconomicScenario, PolicyScenario, MarketScenario]],
        asset_data: Dict[str, Dict[str, Any]],
        portfolio_value: float = 1000000
    ) -> Tuple[List[ScenarioResult], ScenarioComparison]:
        """Run multiple scenarios and provide comparison"""
        
        results = []
        
        for scenario in scenarios:
            if isinstance(scenario, EconomicScenario):
                result = self.analyze_economic_scenario(
                    portfolio_positions, asset_data, scenario, portfolio_value
                )
            elif isinstance(scenario, PolicyScenario):
                result = self.analyze_policy_scenario(
                    portfolio_positions, asset_data, scenario, portfolio_value
                )
            elif isinstance(scenario, MarketScenario):
                result = self.analyze_market_scenario(
                    portfolio_positions, asset_data, scenario, portfolio_value
                )
            else:
                continue
            
            results.append(result)
        
        # Generate comparison
        comparison = self._generate_scenario_comparison(results)
        
        return results, comparison
    
    def create_custom_scenario(
        self,
        scenario_name: str,
        parameter_changes: Dict[str, float],
        scenario_probability: float = 0.5
    ) -> Dict[str, Any]:
        """Create custom scenario from parameter changes"""
        
        custom_scenario = {
            'scenario_name': scenario_name,
            'scenario_type': ScenarioType.CUSTOM_SCENARIO,
            'parameter_changes': parameter_changes,
            'probability': scenario_probability,
            'description': f"Custom scenario: {scenario_name}"
        }
        
        return custom_scenario
    
    def _initialize_scenario_templates(self):
        """Initialize predefined scenario templates"""
        
        # Economic scenarios
        self.scenario_templates['recession'] = {
            'gdp_growth': -2.0,
            'inflation_rate': 1.0,
            'unemployment_rate': 8.0,
            'policy_rate': 0.5,
            'credit_spreads': 200,  # bp widening
            'flight_to_quality': True
        }
        
        self.scenario_templates['stagflation'] = {
            'gdp_growth': 0.5,
            'inflation_rate': 6.0,
            'unemployment_rate': 7.0,
            'policy_rate': 4.0,
            'credit_spreads': 150,
            'currency_volatility': 'high'
        }
        
        self.scenario_templates['strong_growth'] = {
            'gdp_growth': 4.0,
            'inflation_rate': 3.0,
            'unemployment_rate': 4.0,
            'policy_rate': 3.5,
            'credit_spreads': -50,  # bp tightening
            'risk_on': True
        }
        
        # Policy scenarios
        self.scenario_templates['aggressive_tightening'] = {
            'rate_increases': 300,  # bp over 12 months
            'qe_tapering': -50,     # % reduction
            'forward_guidance': 'hawkish'
        }
        
        self.scenario_templates['emergency_easing'] = {
            'rate_cuts': -200,      # bp
            'qe_expansion': 100,    # % increase
            'forward_guidance': 'very_dovish'
        }
    
    def _get_credit_multiplier(self, economic_condition: EconomicCondition) -> float:
        """Get credit spread multiplier based on economic condition"""
        
        multipliers = {
            EconomicCondition.RECESSION: 2.0,
            EconomicCondition.SLOW_GROWTH: 1.2,
            EconomicCondition.MODERATE_GROWTH: 1.0,
            EconomicCondition.STRONG_GROWTH: 0.8,
            EconomicCondition.STAGFLATION: 1.5,
            EconomicCondition.DEFLATION: 1.8
        }
        
        return multipliers.get(economic_condition, 1.0)
    
    def _get_policy_multiplier(self, policy_stance: PolicyStance) -> float:
        """Get policy impact multiplier"""
        
        multipliers = {
            PolicyStance.VERY_DOVISH: -1.5,
            PolicyStance.DOVISH: -1.0,
            PolicyStance.NEUTRAL: 0.0,
            PolicyStance.HAWKISH: 1.0,
            PolicyStance.VERY_HAWKISH: 1.5
        }
        
        return multipliers.get(policy_stance, 0.0)
    
    def _get_currency_policy_impact(self, policy_stance: PolicyStance) -> float:
        """Get currency impact from policy stance"""
        
        # Hawkish policy typically strengthens currency
        multipliers = {
            PolicyStance.VERY_DOVISH: -0.05,  # -5% currency impact
            PolicyStance.DOVISH: -0.02,
            PolicyStance.NEUTRAL: 0.0,
            PolicyStance.HAWKISH: 0.02,
            PolicyStance.VERY_HAWKISH: 0.05
        }
        
        return multipliers.get(policy_stance, 0.0)
    
    def _calculate_correlation_impact(
        self,
        asset_impacts: Dict[str, float],
        correlation_changes: Dict[Tuple[str, str], float]
    ) -> float:
        """Calculate additional impact from correlation changes"""
        
        # Simplified correlation impact calculation
        total_correlation_impact = 0
        
        for (asset1, asset2), corr_change in correlation_changes.items():
            if asset1 in asset_impacts and asset2 in asset_impacts:
                # Additional loss/gain from increased/decreased correlation
                impact1 = asset_impacts[asset1]
                impact2 = asset_impacts[asset2]
                
                # If both negative (losses), increased correlation increases total loss
                if impact1 < 0 and impact2 < 0:
                    additional_impact = corr_change * abs(impact1 * impact2) ** 0.5
                    total_correlation_impact -= additional_impact
                # If opposite signs, increased correlation provides diversification
                elif impact1 * impact2 < 0:
                    diversification_benefit = corr_change * abs(impact1 * impact2) ** 0.5
                    total_correlation_impact += diversification_benefit
        
        return total_correlation_impact
    
    def _generate_scenario_comparison(
        self,
        results: List[ScenarioResult]
    ) -> ScenarioComparison:
        """Generate comparison of scenario results"""
        
        if not results:
            return ScenarioComparison(
                scenarios_analyzed=[],
                best_case_scenario="None",
                worst_case_scenario="None",
                most_likely_scenario="None",
                expected_return=0,
                risk_metrics={},
                scenario_correlations={}
            )
        
        # Find best and worst cases
        best_result = max(results, key=lambda x: x.portfolio_pnl_percentage)
        worst_result = min(results, key=lambda x: x.portfolio_pnl_percentage)
        
        # Find most likely scenario (highest probability)
        most_likely_result = max(results, key=lambda x: x.scenario_probability)
        
        # Calculate probability-weighted expected return
        total_probability = sum(r.scenario_probability for r in results)
        if total_probability > 0:
            expected_return = sum(
                r.portfolio_pnl_percentage * r.scenario_probability 
                for r in results
            ) / total_probability
        else:
            expected_return = np.mean([r.portfolio_pnl_percentage for r in results])
        
        # Calculate risk metrics
        returns = [r.portfolio_pnl_percentage for r in results]
        risk_metrics = {
            'standard_deviation': np.std(returns),
            'downside_deviation': np.std([r for r in returns if r < expected_return]),
            'maximum_loss': min(returns),
            'maximum_gain': max(returns),
            'probability_of_loss': sum(1 for r in returns if r < 0) / len(returns)
        }
        
        # Calculate scenario correlations (simplified)
        scenario_correlations = {}
        for i, result1 in enumerate(results):
            for j, result2 in enumerate(results[i+1:], i+1):
                # Correlation based on similar risk factor impacts
                corr = self._calculate_scenario_correlation(result1, result2)
                scenario_correlations[(result1.scenario_name, result2.scenario_name)] = corr
        
        return ScenarioComparison(
            scenarios_analyzed=[r.scenario_name for r in results],
            best_case_scenario=best_result.scenario_name,
            worst_case_scenario=worst_result.scenario_name,
            most_likely_scenario=most_likely_result.scenario_name,
            expected_return=expected_return,
            risk_metrics=risk_metrics,
            scenario_correlations=scenario_correlations
        )
    
    def _calculate_scenario_correlation(
        self,
        result1: ScenarioResult,
        result2: ScenarioResult
    ) -> float:
        """Calculate correlation between two scenario results"""
        
        # Extract risk factor impacts as vectors
        factors = ['duration_impact', 'credit_impact', 'currency_impact']
        
        vector1 = np.array([
            getattr(result1, factor, 0) for factor in factors
        ])
        
        vector2 = np.array([
            getattr(result2, factor, 0) for factor in factors
        ])
        
        # Calculate correlation
        if np.std(vector1) > 0 and np.std(vector2) > 0:
            correlation = np.corrcoef(vector1, vector2)[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
        else:
            return 0.0