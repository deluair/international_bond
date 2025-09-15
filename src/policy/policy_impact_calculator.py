"""
Policy Impact Calculator Module

This module calculates and quantifies the impact of central bank policy changes
on various financial markets and economic indicators.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Import CentralBank and PolicyAction from other modules
from .policy_divergence_analyzer import CentralBank
from .monetary_policy_tracker import PolicyAction

class ImpactTimeframe(Enum):
    """Timeframes for impact analysis"""
    IMMEDIATE = "immediate"      # 0-1 days
    SHORT_TERM = "short_term"    # 1-7 days
    MEDIUM_TERM = "medium_term"  # 1-4 weeks
    LONG_TERM = "long_term"      # 1-6 months

class MarketSegment(Enum):
    """Market segments for impact analysis"""
    MONEY_MARKETS = "money_markets"
    GOVERNMENT_BONDS = "government_bonds"
    CORPORATE_BONDS = "corporate_bonds"
    FOREIGN_EXCHANGE = "foreign_exchange"
    EQUITIES = "equities"
    COMMODITIES = "commodities"
    VOLATILITY = "volatility"

class TransmissionChannel(Enum):
    """Policy transmission channels"""
    INTEREST_RATE = "interest_rate"
    CREDIT = "credit"
    EXCHANGE_RATE = "exchange_rate"
    ASSET_PRICE = "asset_price"
    EXPECTATIONS = "expectations"
    CONFIDENCE = "confidence"

@dataclass
class PolicyShock:
    """Policy shock specification"""
    central_bank: CentralBank
    shock_type: PolicyAction
    magnitude: float  # in basis points for rate changes
    surprise_component: float  # unexpected portion
    announcement_date: datetime
    effective_date: Optional[datetime]
    
@dataclass
class MarketImpact:
    """Market impact from policy change"""
    market_segment: MarketSegment
    timeframe: ImpactTimeframe
    impact_magnitude: float  # percentage or basis points
    confidence_interval: Tuple[float, float]
    statistical_significance: float
    transmission_channels: List[TransmissionChannel]
    
@dataclass
class EconomicImpact:
    """Economic impact from policy change"""
    indicator: str  # GDP, inflation, unemployment, etc.
    timeframe: ImpactTimeframe
    impact_magnitude: float
    confidence_interval: Tuple[float, float]
    peak_impact_timing: int  # quarters
    persistence: float  # how long impact lasts
    
@dataclass
class CrossAssetImpact:
    """Cross-asset impact analysis"""
    primary_asset: str
    secondary_asset: str
    correlation_change: float
    spillover_magnitude: float
    timing_lag: int  # days
    
@dataclass
class ImpactScenario:
    """Policy impact scenario"""
    scenario_name: str
    probability: float
    market_impacts: List[MarketImpact]
    economic_impacts: List[EconomicImpact]
    risk_factors: List[str]

class PolicyImpactCalculator:
    """
    Comprehensive policy impact calculation and analysis system
    """
    
    def __init__(self):
        self.impact_models = {}
        self.historical_impacts = {}
        self.transmission_parameters = {}
        self._initialize_models()
    
    def calculate_market_impact(
        self,
        policy_shock: PolicyShock,
        market_conditions: Dict[str, Any]
    ) -> Dict[MarketSegment, List[MarketImpact]]:
        """Calculate comprehensive market impact of policy shock"""
        
        impacts = {}
        
        for segment in MarketSegment:
            segment_impacts = []
            
            for timeframe in ImpactTimeframe:
                impact = self._calculate_segment_impact(
                    policy_shock, segment, timeframe, market_conditions
                )
                segment_impacts.append(impact)
            
            impacts[segment] = segment_impacts
        
        return impacts
    
    def calculate_economic_impact(
        self,
        policy_shock: PolicyShock,
        economic_conditions: Dict[str, Any]
    ) -> List[EconomicImpact]:
        """Calculate macroeconomic impact of policy shock"""
        
        indicators = ['gdp_growth', 'inflation', 'unemployment', 'consumption', 'investment']
        impacts = []
        
        for indicator in indicators:
            for timeframe in ImpactTimeframe:
                impact = self._calculate_economic_indicator_impact(
                    policy_shock, indicator, timeframe, economic_conditions
                )
                impacts.append(impact)
        
        return impacts
    
    def analyze_transmission_mechanisms(
        self,
        policy_shock: PolicyShock,
        market_structure: Dict[str, Any]
    ) -> Dict[TransmissionChannel, float]:
        """Analyze policy transmission through different channels"""
        
        transmission_strength = {}
        
        for channel in TransmissionChannel:
            strength = self._calculate_transmission_strength(
                policy_shock, channel, market_structure
            )
            transmission_strength[channel] = strength
        
        return transmission_strength
    
    def calculate_cross_asset_spillovers(
        self,
        policy_shock: PolicyShock,
        asset_correlations: Dict[str, Dict[str, float]]
    ) -> List[CrossAssetImpact]:
        """Calculate spillover effects across asset classes"""
        
        spillovers = []
        
        # Primary impact assets
        primary_assets = self._identify_primary_impact_assets(policy_shock)
        
        for primary in primary_assets:
            for secondary in asset_correlations.get(primary, {}):
                if secondary != primary:
                    spillover = self._calculate_spillover_impact(
                        policy_shock, primary, secondary, asset_correlations
                    )
                    spillovers.append(spillover)
        
        return spillovers
    
    def generate_impact_scenarios(
        self,
        policy_shock: PolicyShock,
        market_conditions: Dict[str, Any],
        confidence_level: float = 0.95
    ) -> List[ImpactScenario]:
        """Generate multiple impact scenarios with probabilities"""
        
        scenarios = []
        
        # Base case scenario
        base_scenario = self._create_base_scenario(policy_shock, market_conditions)
        scenarios.append(base_scenario)
        
        # Stress scenarios
        stress_scenarios = self._create_stress_scenarios(policy_shock, market_conditions)
        scenarios.extend(stress_scenarios)
        
        # Benign scenarios
        benign_scenarios = self._create_benign_scenarios(policy_shock, market_conditions)
        scenarios.extend(benign_scenarios)
        
        # Normalize probabilities
        total_prob = sum(s.probability for s in scenarios)
        for scenario in scenarios:
            scenario.probability /= total_prob
        
        return scenarios
    
    def calculate_policy_effectiveness(
        self,
        policy_shock: PolicyShock,
        target_variables: List[str],
        time_horizon: int = 8  # quarters
    ) -> Dict[str, float]:
        """Calculate policy effectiveness for target variables"""
        
        effectiveness = {}
        
        for target in target_variables:
            # Calculate cumulative impact over time horizon
            cumulative_impact = 0
            
            for quarter in range(1, time_horizon + 1):
                quarterly_impact = self._calculate_quarterly_impact(
                    policy_shock, target, quarter
                )
                cumulative_impact += quarterly_impact
            
            # Normalize by policy magnitude
            if policy_shock.magnitude != 0:
                effectiveness[target] = cumulative_impact / abs(policy_shock.magnitude)
            else:
                effectiveness[target] = 0
        
        return effectiveness
    
    def estimate_market_volatility_impact(
        self,
        policy_shock: PolicyShock,
        current_volatility: Dict[str, float]
    ) -> Dict[str, Dict[str, float]]:
        """Estimate impact on market volatility"""
        
        volatility_impacts = {}
        
        markets = ['bonds', 'fx', 'equities', 'commodities']
        
        for market in markets:
            current_vol = current_volatility.get(market, 0.15)  # Default 15%
            
            # Calculate volatility impact
            vol_impact = self._calculate_volatility_impact(policy_shock, market, current_vol)
            
            volatility_impacts[market] = {
                'current_volatility': current_vol,
                'impact_magnitude': vol_impact,
                'new_volatility': current_vol + vol_impact,
                'duration_days': self._estimate_volatility_duration(policy_shock, market)
            }
        
        return volatility_impacts
    
    def _initialize_models(self):
        """Initialize impact calculation models"""
        
        # Interest rate sensitivity models
        self.impact_models['duration'] = {
            'short_term': 0.8,   # 80% pass-through to short rates
            'medium_term': 0.5,  # 50% pass-through to medium rates
            'long_term': 0.3     # 30% pass-through to long rates
        }
        
        # FX impact models
        self.impact_models['fx'] = {
            'immediate': 1.2,    # 120% impact on FX
            'short_term': 0.9,
            'medium_term': 0.6,
            'long_term': 0.4
        }
        
        # Equity impact models
        self.impact_models['equity'] = {
            'immediate': -0.5,   # Negative impact for rate hikes
            'short_term': -0.3,
            'medium_term': -0.2,
            'long_term': 0.1     # Positive long-term if growth supportive
        }
        
        # Credit spread models
        self.impact_models['credit'] = {
            'investment_grade': 0.2,  # 20% pass-through to IG spreads
            'high_yield': 0.4,        # 40% pass-through to HY spreads
            'emerging_market': 0.6    # 60% pass-through to EM spreads
        }
    
    def _calculate_segment_impact(
        self,
        policy_shock: PolicyShock,
        segment: MarketSegment,
        timeframe: ImpactTimeframe,
        market_conditions: Dict[str, Any]
    ) -> MarketImpact:
        """Calculate impact for specific market segment and timeframe"""
        
        # Base impact calculation
        base_impact = self._get_base_impact(policy_shock, segment, timeframe)
        
        # Adjust for market conditions
        condition_adjustment = self._calculate_condition_adjustment(
            segment, market_conditions
        )
        
        # Adjust for surprise component
        surprise_adjustment = policy_shock.surprise_component * 0.5
        
        # Final impact
        impact_magnitude = base_impact * (1 + condition_adjustment + surprise_adjustment)
        
        # Calculate confidence interval
        volatility = self._estimate_impact_volatility(segment, timeframe)
        confidence_interval = (
            impact_magnitude - 1.96 * volatility,
            impact_magnitude + 1.96 * volatility
        )
        
        # Determine transmission channels
        channels = self._identify_transmission_channels(policy_shock, segment)
        
        return MarketImpact(
            market_segment=segment,
            timeframe=timeframe,
            impact_magnitude=impact_magnitude,
            confidence_interval=confidence_interval,
            statistical_significance=0.95,  # Simplified
            transmission_channels=channels
        )
    
    def _calculate_economic_indicator_impact(
        self,
        policy_shock: PolicyShock,
        indicator: str,
        timeframe: ImpactTimeframe,
        economic_conditions: Dict[str, Any]
    ) -> EconomicImpact:
        """Calculate impact on specific economic indicator"""
        
        # Impact parameters by indicator
        impact_params = {
            'gdp_growth': {'sensitivity': 0.1, 'lag': 2, 'persistence': 0.8},
            'inflation': {'sensitivity': 0.05, 'lag': 4, 'persistence': 0.9},
            'unemployment': {'sensitivity': -0.15, 'lag': 3, 'persistence': 0.7},
            'consumption': {'sensitivity': 0.08, 'lag': 1, 'persistence': 0.6},
            'investment': {'sensitivity': 0.2, 'lag': 2, 'persistence': 0.5}
        }
        
        params = impact_params.get(indicator, {'sensitivity': 0, 'lag': 0, 'persistence': 0})
        
        # Calculate base impact
        base_impact = policy_shock.magnitude * params['sensitivity'] / 100
        
        # Adjust for timeframe
        timeframe_multiplier = self._get_timeframe_multiplier(timeframe, params['lag'])
        impact_magnitude = base_impact * timeframe_multiplier
        
        # Calculate confidence interval
        uncertainty = abs(impact_magnitude) * 0.3  # 30% uncertainty
        confidence_interval = (
            impact_magnitude - uncertainty,
            impact_magnitude + uncertainty
        )
        
        return EconomicImpact(
            indicator=indicator,
            timeframe=timeframe,
            impact_magnitude=impact_magnitude,
            confidence_interval=confidence_interval,
            peak_impact_timing=params['lag'],
            persistence=params['persistence']
        )
    
    def _calculate_transmission_strength(
        self,
        policy_shock: PolicyShock,
        channel: TransmissionChannel,
        market_structure: Dict[str, Any]
    ) -> float:
        """Calculate strength of transmission through specific channel"""
        
        # Base transmission strengths
        base_strengths = {
            TransmissionChannel.INTEREST_RATE: 0.8,
            TransmissionChannel.CREDIT: 0.6,
            TransmissionChannel.EXCHANGE_RATE: 0.7,
            TransmissionChannel.ASSET_PRICE: 0.5,
            TransmissionChannel.EXPECTATIONS: 0.9,
            TransmissionChannel.CONFIDENCE: 0.4
        }
        
        base_strength = base_strengths.get(channel, 0.5)
        
        # Adjust for market structure
        structure_adjustment = 0
        
        if channel == TransmissionChannel.CREDIT:
            # Credit channel stronger in bank-based systems
            bank_dependence = market_structure.get('bank_dependence', 0.5)
            structure_adjustment = (bank_dependence - 0.5) * 0.3
        
        elif channel == TransmissionChannel.EXCHANGE_RATE:
            # Exchange rate channel stronger in open economies
            trade_openness = market_structure.get('trade_openness', 0.5)
            structure_adjustment = (trade_openness - 0.5) * 0.4
        
        # Adjust for policy shock characteristics
        shock_adjustment = 0
        if policy_shock.surprise_component > 0.5:
            shock_adjustment = 0.2  # Surprise shocks have stronger transmission
        
        final_strength = base_strength + structure_adjustment + shock_adjustment
        return max(0, min(1, final_strength))  # Bound between 0 and 1
    
    def _identify_primary_impact_assets(self, policy_shock: PolicyShock) -> List[str]:
        """Identify assets with primary impact from policy shock"""
        
        primary_assets = []
        
        if policy_shock.shock_type in [PolicyAction.RATE_HIKE, PolicyAction.RATE_CUT]:
            primary_assets.extend(['short_rates', 'government_bonds', 'fx'])
        
        if policy_shock.shock_type in [PolicyAction.QE_EXPANSION, PolicyAction.QE_TAPERING]:
            primary_assets.extend(['long_bonds', 'credit_spreads', 'equities'])
        
        return primary_assets
    
    def _calculate_spillover_impact(
        self,
        policy_shock: PolicyShock,
        primary_asset: str,
        secondary_asset: str,
        correlations: Dict[str, Dict[str, float]]
    ) -> CrossAssetImpact:
        """Calculate spillover impact between assets"""
        
        # Get correlation
        correlation = correlations.get(primary_asset, {}).get(secondary_asset, 0)
        
        # Calculate primary impact
        primary_impact = self._get_asset_impact(policy_shock, primary_asset)
        
        # Calculate spillover
        spillover_magnitude = primary_impact * correlation * 0.7  # 70% spillover efficiency
        
        # Estimate timing lag
        timing_lag = self._estimate_spillover_lag(primary_asset, secondary_asset)
        
        # Calculate correlation change
        correlation_change = policy_shock.surprise_component * 0.1  # Surprise increases correlation
        
        return CrossAssetImpact(
            primary_asset=primary_asset,
            secondary_asset=secondary_asset,
            correlation_change=correlation_change,
            spillover_magnitude=spillover_magnitude,
            timing_lag=timing_lag
        )
    
    def _create_base_scenario(
        self,
        policy_shock: PolicyShock,
        market_conditions: Dict[str, Any]
    ) -> ImpactScenario:
        """Create base case impact scenario"""
        
        market_impacts = []
        economic_impacts = []
        
        # Calculate market impacts
        for segment in MarketSegment:
            impact = self._calculate_segment_impact(
                policy_shock, segment, ImpactTimeframe.SHORT_TERM, market_conditions
            )
            market_impacts.append(impact)
        
        # Calculate economic impacts
        indicators = ['gdp_growth', 'inflation', 'unemployment']
        for indicator in indicators:
            impact = self._calculate_economic_indicator_impact(
                policy_shock, indicator, ImpactTimeframe.MEDIUM_TERM, {}
            )
            economic_impacts.append(impact)
        
        return ImpactScenario(
            scenario_name="Base Case",
            probability=0.6,
            market_impacts=market_impacts,
            economic_impacts=economic_impacts,
            risk_factors=["Standard market conditions"]
        )
    
    def _create_stress_scenarios(
        self,
        policy_shock: PolicyShock,
        market_conditions: Dict[str, Any]
    ) -> List[ImpactScenario]:
        """Create stress test scenarios"""
        
        scenarios = []
        
        # High volatility scenario
        stress_conditions = market_conditions.copy()
        stress_conditions['volatility_regime'] = 'high'
        
        market_impacts = []
        for segment in MarketSegment:
            base_impact = self._calculate_segment_impact(
                policy_shock, segment, ImpactTimeframe.SHORT_TERM, stress_conditions
            )
            # Amplify impact by 50%
            base_impact.impact_magnitude *= 1.5
            market_impacts.append(base_impact)
        
        stress_scenario = ImpactScenario(
            scenario_name="High Volatility Stress",
            probability=0.2,
            market_impacts=market_impacts,
            economic_impacts=[],
            risk_factors=["High market volatility", "Liquidity constraints", "Risk-off sentiment"]
        )
        
        scenarios.append(stress_scenario)
        
        return scenarios
    
    def _create_benign_scenarios(
        self,
        policy_shock: PolicyShock,
        market_conditions: Dict[str, Any]
    ) -> List[ImpactScenario]:
        """Create benign impact scenarios"""
        
        scenarios = []
        
        # Low volatility scenario
        benign_conditions = market_conditions.copy()
        benign_conditions['volatility_regime'] = 'low'
        
        market_impacts = []
        for segment in MarketSegment:
            base_impact = self._calculate_segment_impact(
                policy_shock, segment, ImpactTimeframe.SHORT_TERM, benign_conditions
            )
            # Reduce impact by 30%
            base_impact.impact_magnitude *= 0.7
            market_impacts.append(base_impact)
        
        benign_scenario = ImpactScenario(
            scenario_name="Benign Market Conditions",
            probability=0.2,
            market_impacts=market_impacts,
            economic_impacts=[],
            risk_factors=["Low volatility", "Strong liquidity", "Stable market conditions"]
        )
        
        scenarios.append(benign_scenario)
        
        return scenarios
    
    def _get_base_impact(
        self,
        policy_shock: PolicyShock,
        segment: MarketSegment,
        timeframe: ImpactTimeframe
    ) -> float:
        """Get base impact for market segment"""
        
        # Simplified impact mapping
        impact_matrix = {
            MarketSegment.MONEY_MARKETS: {
                ImpactTimeframe.IMMEDIATE: 0.9,
                ImpactTimeframe.SHORT_TERM: 0.8,
                ImpactTimeframe.MEDIUM_TERM: 0.6,
                ImpactTimeframe.LONG_TERM: 0.4
            },
            MarketSegment.GOVERNMENT_BONDS: {
                ImpactTimeframe.IMMEDIATE: 0.7,
                ImpactTimeframe.SHORT_TERM: 0.6,
                ImpactTimeframe.MEDIUM_TERM: 0.4,
                ImpactTimeframe.LONG_TERM: 0.3
            },
            MarketSegment.FOREIGN_EXCHANGE: {
                ImpactTimeframe.IMMEDIATE: 1.2,
                ImpactTimeframe.SHORT_TERM: 0.9,
                ImpactTimeframe.MEDIUM_TERM: 0.6,
                ImpactTimeframe.LONG_TERM: 0.4
            }
        }
        
        base_multiplier = impact_matrix.get(segment, {}).get(timeframe, 0.5)
        
        # Apply to policy shock magnitude
        return policy_shock.magnitude * base_multiplier / 100  # Convert bp to decimal
    
    def _calculate_condition_adjustment(
        self,
        segment: MarketSegment,
        market_conditions: Dict[str, Any]
    ) -> float:
        """Calculate adjustment for market conditions"""
        
        adjustment = 0
        
        # Volatility regime adjustment
        volatility_regime = market_conditions.get('volatility_regime', 'normal')
        if volatility_regime == 'high':
            adjustment += 0.3
        elif volatility_regime == 'low':
            adjustment -= 0.2
        
        # Liquidity adjustment
        liquidity = market_conditions.get('liquidity', 'normal')
        if liquidity == 'tight':
            adjustment += 0.2
        elif liquidity == 'abundant':
            adjustment -= 0.1
        
        return adjustment
    
    def _estimate_impact_volatility(
        self,
        segment: MarketSegment,
        timeframe: ImpactTimeframe
    ) -> float:
        """Estimate volatility of impact estimates"""
        
        # Base volatilities by segment
        base_volatilities = {
            MarketSegment.MONEY_MARKETS: 0.1,
            MarketSegment.GOVERNMENT_BONDS: 0.15,
            MarketSegment.CORPORATE_BONDS: 0.2,
            MarketSegment.FOREIGN_EXCHANGE: 0.25,
            MarketSegment.EQUITIES: 0.3,
            MarketSegment.COMMODITIES: 0.35
        }
        
        base_vol = base_volatilities.get(segment, 0.2)
        
        # Adjust for timeframe (longer timeframes have higher uncertainty)
        timeframe_multipliers = {
            ImpactTimeframe.IMMEDIATE: 0.8,
            ImpactTimeframe.SHORT_TERM: 1.0,
            ImpactTimeframe.MEDIUM_TERM: 1.3,
            ImpactTimeframe.LONG_TERM: 1.6
        }
        
        multiplier = timeframe_multipliers.get(timeframe, 1.0)
        
        return base_vol * multiplier