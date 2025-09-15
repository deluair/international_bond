"""
Stress Testing Module

This module provides comprehensive stress testing capabilities for international bond portfolios
including historical scenarios, hypothetical shocks, and Monte Carlo stress testing.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class StressType(Enum):
    """Types of stress tests"""
    HISTORICAL_SCENARIO = "historical_scenario"
    HYPOTHETICAL_SHOCK = "hypothetical_shock"
    MONTE_CARLO_STRESS = "monte_carlo_stress"
    REGULATORY_STRESS = "regulatory_stress"
    TAIL_RISK_SCENARIO = "tail_risk_scenario"

class ScenarioSeverity(Enum):
    """Severity levels for stress scenarios"""
    MILD = "mild"           # 1-2 standard deviations
    MODERATE = "moderate"   # 2-3 standard deviations
    SEVERE = "severe"       # 3-4 standard deviations
    EXTREME = "extreme"     # >4 standard deviations

class RiskFactor(Enum):
    """Risk factors for stress testing"""
    INTEREST_RATES = "interest_rates"
    CREDIT_SPREADS = "credit_spreads"
    FOREIGN_EXCHANGE = "foreign_exchange"
    VOLATILITY = "volatility"
    LIQUIDITY = "liquidity"
    CORRELATION = "correlation"
    INFLATION = "inflation"

@dataclass
class StressShock:
    """Individual stress shock specification"""
    risk_factor: RiskFactor
    shock_magnitude: float  # in basis points or percentage
    shock_direction: str    # "up", "down", "both"
    time_horizon: int       # days
    probability: Optional[float] = None
    
@dataclass
class StressScenario:
    """Complete stress scenario"""
    scenario_name: str
    scenario_type: StressType
    severity: ScenarioSeverity
    shocks: List[StressShock]
    correlation_adjustments: Dict[str, float]
    description: str
    historical_precedent: Optional[str] = None
    
@dataclass
class StressResult:
    """Stress test result"""
    scenario: StressScenario
    portfolio_pnl: float
    portfolio_pnl_percentage: float
    asset_level_pnl: Dict[str, float]
    risk_factor_contributions: Dict[RiskFactor, float]
    worst_performing_assets: List[Tuple[str, float]]
    recovery_time_estimate: Optional[int] = None  # days
    
@dataclass
class StressSummary:
    """Summary of multiple stress tests"""
    worst_case_scenario: str
    worst_case_loss: float
    average_loss: float
    scenarios_tested: int
    pass_rate: float  # percentage of scenarios within risk tolerance
    key_vulnerabilities: List[str]

class StressTester:
    """
    Comprehensive stress testing system for international bond portfolios
    """
    
    def __init__(self):
        self.historical_scenarios = {}
        self.risk_factor_sensitivities = {}
        self.correlation_matrices = {}
        self._initialize_scenarios()
    
    def run_stress_test(
        self,
        portfolio_positions: Dict[str, float],
        asset_sensitivities: Dict[str, Dict[RiskFactor, float]],
        scenario: StressScenario,
        portfolio_value: float = 1000000
    ) -> StressResult:
        """Run individual stress test"""
        
        total_pnl = 0
        asset_level_pnl = {}
        risk_factor_contributions = {factor: 0 for factor in RiskFactor}
        
        # Apply shocks to each position
        for asset_id, position_size in portfolio_positions.items():
            asset_pnl = 0
            
            if asset_id in asset_sensitivities:
                sensitivities = asset_sensitivities[asset_id]
                
                # Apply each shock
                for shock in scenario.shocks:
                    if shock.risk_factor in sensitivities:
                        sensitivity = sensitivities[shock.risk_factor]
                        
                        # Calculate P&L impact
                        shock_pnl = self._calculate_shock_impact(
                            position_size, sensitivity, shock
                        )
                        
                        asset_pnl += shock_pnl
                        risk_factor_contributions[shock.risk_factor] += shock_pnl
            
            asset_level_pnl[asset_id] = asset_pnl
            total_pnl += asset_pnl
        
        # Apply correlation adjustments
        if scenario.correlation_adjustments:
            correlation_impact = self._apply_correlation_adjustments(
                asset_level_pnl, scenario.correlation_adjustments
            )
            total_pnl += correlation_impact
        
        # Calculate percentage impact
        pnl_percentage = (total_pnl / portfolio_value) * 100
        
        # Identify worst performing assets
        worst_assets = sorted(
            asset_level_pnl.items(), 
            key=lambda x: x[1]
        )[:5]  # Top 5 worst performers
        
        # Estimate recovery time
        recovery_time = self._estimate_recovery_time(scenario, abs(pnl_percentage))
        
        return StressResult(
            scenario=scenario,
            portfolio_pnl=total_pnl,
            portfolio_pnl_percentage=pnl_percentage,
            asset_level_pnl=asset_level_pnl,
            risk_factor_contributions=risk_factor_contributions,
            worst_performing_assets=worst_assets,
            recovery_time_estimate=recovery_time
        )
    
    def run_multiple_stress_tests(
        self,
        portfolio_positions: Dict[str, float],
        asset_sensitivities: Dict[str, Dict[RiskFactor, float]],
        scenarios: List[StressScenario],
        portfolio_value: float = 1000000,
        risk_tolerance: float = -5.0  # -5% loss tolerance
    ) -> Tuple[List[StressResult], StressSummary]:
        """Run multiple stress tests and provide summary"""
        
        results = []
        
        # Run each stress test
        for scenario in scenarios:
            result = self.run_stress_test(
                portfolio_positions, asset_sensitivities, scenario, portfolio_value
            )
            results.append(result)
        
        # Generate summary
        summary = self._generate_stress_summary(results, risk_tolerance)
        
        return results, summary
    
    def create_historical_scenario(
        self,
        scenario_name: str,
        start_date: datetime,
        end_date: datetime,
        market_data: Dict[str, pd.Series]
    ) -> StressScenario:
        """Create stress scenario based on historical period"""
        
        shocks = []
        
        for risk_factor_name, data in market_data.items():
            # Filter data for the period
            period_data = data[(data.index >= start_date) & (data.index <= end_date)]
            
            if len(period_data) > 0:
                # Calculate total change over period
                total_change = period_data.iloc[-1] - period_data.iloc[0]
                
                # Map to risk factor enum
                risk_factor = self._map_to_risk_factor(risk_factor_name)
                
                if risk_factor:
                    shock = StressShock(
                        risk_factor=risk_factor,
                        shock_magnitude=total_change,
                        shock_direction="historical",
                        time_horizon=(end_date - start_date).days
                    )
                    shocks.append(shock)
        
        # Determine severity based on magnitude
        severity = self._determine_scenario_severity(shocks)
        
        return StressScenario(
            scenario_name=scenario_name,
            scenario_type=StressType.HISTORICAL_SCENARIO,
            severity=severity,
            shocks=shocks,
            correlation_adjustments={},
            description=f"Historical scenario from {start_date.date()} to {end_date.date()}",
            historical_precedent=scenario_name
        )
    
    def create_hypothetical_scenario(
        self,
        scenario_name: str,
        shock_specifications: Dict[RiskFactor, Tuple[float, str]],
        severity: ScenarioSeverity = ScenarioSeverity.MODERATE
    ) -> StressScenario:
        """Create hypothetical stress scenario"""
        
        shocks = []
        
        for risk_factor, (magnitude, direction) in shock_specifications.items():
            shock = StressShock(
                risk_factor=risk_factor,
                shock_magnitude=magnitude,
                shock_direction=direction,
                time_horizon=1  # Instantaneous shock
            )
            shocks.append(shock)
        
        return StressScenario(
            scenario_name=scenario_name,
            scenario_type=StressType.HYPOTHETICAL_SHOCK,
            severity=severity,
            shocks=shocks,
            correlation_adjustments={},
            description=f"Hypothetical {severity.value} stress scenario"
        )
    
    def generate_monte_carlo_scenarios(
        self,
        n_scenarios: int,
        risk_factor_distributions: Dict[RiskFactor, Dict[str, float]],
        correlation_matrix: Optional[np.ndarray] = None
    ) -> List[StressScenario]:
        """Generate Monte Carlo stress scenarios"""
        
        scenarios = []
        np.random.seed(42)  # For reproducibility
        
        risk_factors = list(risk_factor_distributions.keys())
        n_factors = len(risk_factors)
        
        # Generate correlated random shocks
        if correlation_matrix is not None and correlation_matrix.shape == (n_factors, n_factors):
            # Use Cholesky decomposition for correlated shocks
            L = np.linalg.cholesky(correlation_matrix)
            random_shocks = np.random.normal(0, 1, (n_scenarios, n_factors))
            correlated_shocks = np.dot(random_shocks, L.T)
        else:
            # Independent shocks
            correlated_shocks = np.random.normal(0, 1, (n_scenarios, n_factors))
        
        for i in range(n_scenarios):
            shocks = []
            
            for j, risk_factor in enumerate(risk_factors):
                distribution = risk_factor_distributions[risk_factor]
                mean = distribution.get('mean', 0)
                std = distribution.get('std', 1)
                
                # Scale random shock
                shock_magnitude = mean + std * correlated_shocks[i, j]
                
                shock = StressShock(
                    risk_factor=risk_factor,
                    shock_magnitude=shock_magnitude,
                    shock_direction="both",
                    time_horizon=1
                )
                shocks.append(shock)
            
            # Determine severity
            severity = self._determine_scenario_severity(shocks)
            
            scenario = StressScenario(
                scenario_name=f"Monte Carlo Scenario {i+1}",
                scenario_type=StressType.MONTE_CARLO_STRESS,
                severity=severity,
                shocks=shocks,
                correlation_adjustments={},
                description=f"Monte Carlo generated stress scenario {i+1}"
            )
            scenarios.append(scenario)
        
        return scenarios
    
    def create_regulatory_scenarios(self) -> List[StressScenario]:
        """Create regulatory stress scenarios (e.g., CCAR, EBA)"""
        
        scenarios = []
        
        # Severely Adverse Scenario (similar to Fed CCAR)
        severely_adverse = StressScenario(
            scenario_name="Severely Adverse Economic Conditions",
            scenario_type=StressType.REGULATORY_STRESS,
            severity=ScenarioSeverity.SEVERE,
            shocks=[
                StressShock(RiskFactor.INTEREST_RATES, 200, "up", 90),      # +200bp rates
                StressShock(RiskFactor.CREDIT_SPREADS, 300, "up", 90),      # +300bp spreads
                StressShock(RiskFactor.FOREIGN_EXCHANGE, -15, "down", 90),  # -15% USD
                StressShock(RiskFactor.VOLATILITY, 50, "up", 90)            # +50% volatility
            ],
            correlation_adjustments={'stress_correlation': 0.3},
            description="Regulatory severely adverse economic scenario"
        )
        scenarios.append(severely_adverse)
        
        # Adverse Scenario
        adverse = StressScenario(
            scenario_name="Adverse Economic Conditions",
            scenario_type=StressType.REGULATORY_STRESS,
            severity=ScenarioSeverity.MODERATE,
            shocks=[
                StressShock(RiskFactor.INTEREST_RATES, 100, "up", 90),      # +100bp rates
                StressShock(RiskFactor.CREDIT_SPREADS, 150, "up", 90),      # +150bp spreads
                StressShock(RiskFactor.FOREIGN_EXCHANGE, -8, "down", 90),   # -8% USD
                StressShock(RiskFactor.VOLATILITY, 25, "up", 90)            # +25% volatility
            ],
            correlation_adjustments={'stress_correlation': 0.2},
            description="Regulatory adverse economic scenario"
        )
        scenarios.append(adverse)
        
        return scenarios
    
    def _initialize_scenarios(self):
        """Initialize predefined historical scenarios"""
        
        # 2008 Financial Crisis
        self.historical_scenarios['2008_crisis'] = {
            'start_date': datetime(2008, 9, 1),
            'end_date': datetime(2009, 3, 31),
            'description': 'Global Financial Crisis period'
        }
        
        # COVID-19 Market Shock
        self.historical_scenarios['covid_2020'] = {
            'start_date': datetime(2020, 2, 1),
            'end_date': datetime(2020, 4, 30),
            'description': 'COVID-19 market disruption'
        }
        
        # European Debt Crisis
        self.historical_scenarios['eurozone_2011'] = {
            'start_date': datetime(2011, 7, 1),
            'end_date': datetime(2012, 7, 31),
            'description': 'European sovereign debt crisis'
        }
    
    def _calculate_shock_impact(
        self,
        position_size: float,
        sensitivity: float,
        shock: StressShock
    ) -> float:
        """Calculate P&L impact of shock on position"""
        
        # Basic linear sensitivity calculation
        # P&L = Position Size × Sensitivity × Shock Magnitude
        
        shock_magnitude = shock.shock_magnitude
        
        # Adjust for shock direction
        if shock.shock_direction == "up":
            shock_magnitude = abs(shock_magnitude)
        elif shock.shock_direction == "down":
            shock_magnitude = -abs(shock_magnitude)
        # For "both" or "historical", use as-is
        
        # Convert basis points to decimal if needed
        if shock.risk_factor in [RiskFactor.INTEREST_RATES, RiskFactor.CREDIT_SPREADS]:
            shock_magnitude /= 10000  # bp to decimal
        elif shock.risk_factor == RiskFactor.FOREIGN_EXCHANGE:
            shock_magnitude /= 100    # percentage to decimal
        
        pnl_impact = position_size * sensitivity * shock_magnitude
        
        return pnl_impact
    
    def _apply_correlation_adjustments(
        self,
        asset_pnl: Dict[str, float],
        correlation_adjustments: Dict[str, float]
    ) -> float:
        """Apply correlation adjustments to portfolio P&L"""
        
        # Simplified correlation adjustment
        # In practice, this would use full correlation matrix
        
        total_adjustment = 0
        
        if 'stress_correlation' in correlation_adjustments:
            # Increase correlation during stress
            correlation_increase = correlation_adjustments['stress_correlation']
            
            # Calculate additional loss due to increased correlation
            individual_losses = [pnl for pnl in asset_pnl.values() if pnl < 0]
            
            if len(individual_losses) > 1:
                # Additional loss = correlation increase × sum of individual losses
                additional_loss = correlation_increase * sum(individual_losses) * 0.1
                total_adjustment += additional_loss
        
        return total_adjustment
    
    def _estimate_recovery_time(
        self,
        scenario: StressScenario,
        loss_percentage: float
    ) -> Optional[int]:
        """Estimate portfolio recovery time in days"""
        
        # Simple heuristic based on scenario severity and loss magnitude
        base_recovery_days = {
            ScenarioSeverity.MILD: 30,
            ScenarioSeverity.MODERATE: 90,
            ScenarioSeverity.SEVERE: 180,
            ScenarioSeverity.EXTREME: 365
        }
        
        base_days = base_recovery_days.get(scenario.severity, 90)
        
        # Adjust based on loss magnitude
        if loss_percentage > 10:
            multiplier = 1.5
        elif loss_percentage > 5:
            multiplier = 1.2
        else:
            multiplier = 1.0
        
        return int(base_days * multiplier)
    
    def _generate_stress_summary(
        self,
        results: List[StressResult],
        risk_tolerance: float
    ) -> StressSummary:
        """Generate summary of stress test results"""
        
        if not results:
            return StressSummary(
                worst_case_scenario="None",
                worst_case_loss=0,
                average_loss=0,
                scenarios_tested=0,
                pass_rate=100,
                key_vulnerabilities=[]
            )
        
        # Find worst case
        worst_result = min(results, key=lambda x: x.portfolio_pnl_percentage)
        worst_case_scenario = worst_result.scenario.scenario_name
        worst_case_loss = worst_result.portfolio_pnl_percentage
        
        # Calculate average loss
        average_loss = np.mean([r.portfolio_pnl_percentage for r in results])
        
        # Calculate pass rate (scenarios within risk tolerance)
        passing_scenarios = sum(
            1 for r in results if r.portfolio_pnl_percentage >= risk_tolerance
        )
        pass_rate = (passing_scenarios / len(results)) * 100
        
        # Identify key vulnerabilities
        vulnerabilities = self._identify_vulnerabilities(results)
        
        return StressSummary(
            worst_case_scenario=worst_case_scenario,
            worst_case_loss=worst_case_loss,
            average_loss=average_loss,
            scenarios_tested=len(results),
            pass_rate=pass_rate,
            key_vulnerabilities=vulnerabilities
        )
    
    def _map_to_risk_factor(self, factor_name: str) -> Optional[RiskFactor]:
        """Map string to RiskFactor enum"""
        
        mapping = {
            'interest_rates': RiskFactor.INTEREST_RATES,
            'rates': RiskFactor.INTEREST_RATES,
            'credit_spreads': RiskFactor.CREDIT_SPREADS,
            'credit': RiskFactor.CREDIT_SPREADS,
            'fx': RiskFactor.FOREIGN_EXCHANGE,
            'foreign_exchange': RiskFactor.FOREIGN_EXCHANGE,
            'volatility': RiskFactor.VOLATILITY,
            'vol': RiskFactor.VOLATILITY,
            'liquidity': RiskFactor.LIQUIDITY,
            'correlation': RiskFactor.CORRELATION,
            'inflation': RiskFactor.INFLATION
        }
        
        return mapping.get(factor_name.lower())
    
    def _determine_scenario_severity(self, shocks: List[StressShock]) -> ScenarioSeverity:
        """Determine scenario severity based on shock magnitudes"""
        
        max_magnitude = 0
        
        for shock in shocks:
            # Normalize shock magnitude to standard deviations
            if shock.risk_factor == RiskFactor.INTEREST_RATES:
                # Assume 1 std dev = 50bp for rates
                normalized_magnitude = abs(shock.shock_magnitude) / 50
            elif shock.risk_factor == RiskFactor.CREDIT_SPREADS:
                # Assume 1 std dev = 25bp for credit spreads
                normalized_magnitude = abs(shock.shock_magnitude) / 25
            elif shock.risk_factor == RiskFactor.FOREIGN_EXCHANGE:
                # Assume 1 std dev = 5% for FX
                normalized_magnitude = abs(shock.shock_magnitude) / 5
            else:
                # Default normalization
                normalized_magnitude = abs(shock.shock_magnitude) / 100
            
            max_magnitude = max(max_magnitude, normalized_magnitude)
        
        # Classify severity
        if max_magnitude >= 4:
            return ScenarioSeverity.EXTREME
        elif max_magnitude >= 3:
            return ScenarioSeverity.SEVERE
        elif max_magnitude >= 2:
            return ScenarioSeverity.MODERATE
        else:
            return ScenarioSeverity.MILD
    
    def _identify_vulnerabilities(self, results: List[StressResult]) -> List[str]:
        """Identify key portfolio vulnerabilities from stress test results"""
        
        vulnerabilities = []
        
        # Analyze risk factor contributions
        risk_factor_impacts = {}
        for result in results:
            for factor, contribution in result.risk_factor_contributions.items():
                if factor not in risk_factor_impacts:
                    risk_factor_impacts[factor] = []
                risk_factor_impacts[factor].append(contribution)
        
        # Find most impactful risk factors
        for factor, impacts in risk_factor_impacts.items():
            avg_impact = np.mean(impacts)
            if avg_impact < -50000:  # Significant negative impact
                vulnerabilities.append(f"High sensitivity to {factor.value}")
        
        # Analyze asset concentrations
        asset_impacts = {}
        for result in results:
            for asset, pnl in result.asset_level_pnl.items():
                if asset not in asset_impacts:
                    asset_impacts[asset] = []
                asset_impacts[asset].append(pnl)
        
        # Find consistently worst performing assets
        for asset, impacts in asset_impacts.items():
            avg_impact = np.mean(impacts)
            if avg_impact < -20000:  # Significant negative impact
                vulnerabilities.append(f"Concentration risk in {asset}")
        
        return vulnerabilities[:5]  # Return top 5 vulnerabilities