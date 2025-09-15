"""
Credit analytics and risk assessment functionality.
"""

import numpy as np
from datetime import date, datetime, timedelta
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import math

from ..models.cds import CDSCurve, CDSQuote
from ..models.bond import SovereignBond
from ..pricing.yield_curve import YieldCurve
from .cds_curve_builder import CDSBootstrapResult


class CreditRiskMetric(Enum):
    """Types of credit risk metrics."""
    PROBABILITY_OF_DEFAULT = "probability_of_default"
    SURVIVAL_PROBABILITY = "survival_probability"
    HAZARD_RATE = "hazard_rate"
    CREDIT_SPREAD = "credit_spread"
    EXPECTED_LOSS = "expected_loss"
    CREDIT_VAR = "credit_var"
    JUMP_TO_DEFAULT = "jump_to_default"


@dataclass
class CreditRiskMetrics:
    """Container for credit risk metrics."""
    entity: str
    currency: str
    calculation_date: date
    time_horizon: float  # Years
    
    # Core metrics
    probability_of_default: float
    survival_probability: float
    hazard_rate: float
    credit_spread: float
    expected_loss: float
    
    # Advanced metrics
    credit_var_95: Optional[float] = None
    credit_var_99: Optional[float] = None
    jump_to_default_risk: Optional[float] = None
    
    # Term structure metrics
    term_structure: Optional[Dict[float, Dict[str, float]]] = None
    
    def __str__(self) -> str:
        return (f"Credit Risk Metrics for {self.entity} ({self.currency})\n"
                f"Time Horizon: {self.time_horizon} years\n"
                f"PD: {self.probability_of_default:.4f} ({self.probability_of_default*100:.2f}%)\n"
                f"Hazard Rate: {self.hazard_rate:.4f} ({self.hazard_rate*100:.2f}%)\n"
                f"Credit Spread: {self.credit_spread*10000:.1f} bp\n"
                f"Expected Loss: {self.expected_loss:.4f} ({self.expected_loss*100:.2f}%)")


@dataclass
class CreditCorrelationMatrix:
    """Credit correlation matrix for portfolio analysis."""
    entities: List[str]
    correlation_matrix: np.ndarray
    calculation_date: date
    methodology: str = "asset_correlation"
    
    def get_correlation(self, entity1: str, entity2: str) -> float:
        """Get correlation between two entities."""
        try:
            idx1 = self.entities.index(entity1)
            idx2 = self.entities.index(entity2)
            return self.correlation_matrix[idx1, idx2]
        except ValueError:
            return 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            'entities': self.entities,
            'correlation_matrix': self.correlation_matrix.tolist(),
            'calculation_date': self.calculation_date.isoformat(),
            'methodology': self.methodology
        }


class CreditAnalytics:
    """
    Comprehensive credit analytics and risk assessment.
    """
    
    def __init__(self, recovery_rate: float = 0.4):
        """
        Initialize credit analytics.
        
        Args:
            recovery_rate: Default recovery rate assumption
        """
        self.recovery_rate = recovery_rate
    
    def calculate_credit_metrics(self, cds_curve: CDSCurve,
                               time_horizon: float = 1.0,
                               risk_free_curve: Optional[YieldCurve] = None) -> CreditRiskMetrics:
        """
        Calculate comprehensive credit risk metrics.
        
        Args:
            cds_curve: CDS curve for the entity
            time_horizon: Time horizon in years
            risk_free_curve: Risk-free yield curve
            
        Returns:
            CreditRiskMetrics object
        """
        # Get survival probability and hazard rate
        survival_prob = cds_curve.get_survival_probability(time_horizon)
        probability_of_default = 1.0 - survival_prob
        hazard_rate = cds_curve.get_hazard_rate(time_horizon)
        
        # Calculate credit spread
        if risk_free_curve:
            risk_free_rate = risk_free_curve.get_yield(time_horizon)
            risky_rate = -math.log(survival_prob) / time_horizon
            credit_spread = risky_rate - risk_free_rate
        else:
            # Approximate credit spread from CDS spread
            cds_spread = cds_curve.get_spread(time_horizon)
            credit_spread = cds_spread / (1 - self.recovery_rate)
        
        # Calculate expected loss
        expected_loss = probability_of_default * (1 - self.recovery_rate)
        
        # Calculate VaR metrics
        credit_var_95 = self._calculate_credit_var(probability_of_default, 0.95)
        credit_var_99 = self._calculate_credit_var(probability_of_default, 0.99)
        
        # Calculate jump-to-default risk
        jtd_risk = self._calculate_jump_to_default_risk(cds_curve, time_horizon)
        
        # Calculate term structure
        term_structure = self._calculate_term_structure(cds_curve, risk_free_curve)
        
        return CreditRiskMetrics(
            entity=cds_curve.entity,
            currency=cds_curve.currency,
            calculation_date=cds_curve.curve_date,
            time_horizon=time_horizon,
            probability_of_default=probability_of_default,
            survival_probability=survival_prob,
            hazard_rate=hazard_rate,
            credit_spread=credit_spread,
            expected_loss=expected_loss,
            credit_var_95=credit_var_95,
            credit_var_99=credit_var_99,
            jump_to_default_risk=jtd_risk,
            term_structure=term_structure
        )
    
    def compare_credit_risk(self, cds_curves: List[CDSCurve],
                          time_horizon: float = 1.0,
                          risk_free_curve: Optional[YieldCurve] = None) -> Dict[str, CreditRiskMetrics]:
        """
        Compare credit risk across multiple entities.
        
        Args:
            cds_curves: List of CDS curves
            time_horizon: Time horizon for comparison
            risk_free_curve: Risk-free yield curve
            
        Returns:
            Dictionary of entity to credit metrics
        """
        comparison = {}
        
        for curve in cds_curves:
            metrics = self.calculate_credit_metrics(curve, time_horizon, risk_free_curve)
            comparison[curve.entity] = metrics
        
        return comparison
    
    def calculate_relative_value(self, base_curve: CDSCurve,
                               comparison_curves: List[CDSCurve],
                               time_horizon: float = 1.0) -> Dict[str, Dict[str, float]]:
        """
        Calculate relative value metrics between entities.
        
        Args:
            base_curve: Base CDS curve for comparison
            comparison_curves: List of curves to compare against base
            time_horizon: Time horizon for analysis
            
        Returns:
            Dictionary of relative value metrics
        """
        base_metrics = self.calculate_credit_metrics(base_curve, time_horizon)
        relative_values = {}
        
        for curve in comparison_curves:
            comp_metrics = self.calculate_credit_metrics(curve, time_horizon)
            
            # Calculate relative metrics
            spread_ratio = comp_metrics.credit_spread / base_metrics.credit_spread if base_metrics.credit_spread > 0 else 1.0
            pd_ratio = comp_metrics.probability_of_default / base_metrics.probability_of_default if base_metrics.probability_of_default > 0 else 1.0
            hazard_ratio = comp_metrics.hazard_rate / base_metrics.hazard_rate if base_metrics.hazard_rate > 0 else 1.0
            
            # Spread differential in basis points
            spread_diff = (comp_metrics.credit_spread - base_metrics.credit_spread) * 10000
            
            relative_values[curve.entity] = {
                'spread_ratio': spread_ratio,
                'spread_differential_bp': spread_diff,
                'pd_ratio': pd_ratio,
                'hazard_ratio': hazard_ratio,
                'relative_cheapness': spread_ratio - pd_ratio,  # Simple relative value measure
                'z_score': self._calculate_z_score(comp_metrics.credit_spread, base_metrics.credit_spread)
            }
        
        return relative_values
    
    def calculate_portfolio_credit_risk(self, positions: List[Tuple[CDSCurve, float]],
                                      correlation_matrix: Optional[CreditCorrelationMatrix] = None,
                                      time_horizon: float = 1.0) -> Dict[str, float]:
        """
        Calculate portfolio-level credit risk metrics.
        
        Args:
            positions: List of (CDS curve, weight) tuples
            correlation_matrix: Credit correlation matrix
            time_horizon: Time horizon for analysis
            
        Returns:
            Dictionary of portfolio risk metrics
        """
        # Individual position metrics
        individual_pds = []
        individual_els = []
        weights = []
        entities = []
        
        for curve, weight in positions:
            metrics = self.calculate_credit_metrics(curve, time_horizon)
            individual_pds.append(metrics.probability_of_default)
            individual_els.append(metrics.expected_loss)
            weights.append(weight)
            entities.append(curve.entity)
        
        individual_pds = np.array(individual_pds)
        individual_els = np.array(individual_els)
        weights = np.array(weights)
        
        # Portfolio expected loss
        portfolio_el = np.sum(weights * individual_els)
        
        # Portfolio default probability (assuming independence if no correlation matrix)
        if correlation_matrix is None:
            # Independent case
            portfolio_survival = np.prod((1 - individual_pds) ** weights)
            portfolio_pd = 1 - portfolio_survival
        else:
            # Correlated case - use Monte Carlo or analytical approximation
            portfolio_pd = self._calculate_correlated_portfolio_pd(
                individual_pds, weights, correlation_matrix, entities
            )
        
        # Portfolio VaR
        portfolio_var_95 = self._calculate_portfolio_var(individual_els, weights, 0.95, correlation_matrix, entities)
        portfolio_var_99 = self._calculate_portfolio_var(individual_els, weights, 0.99, correlation_matrix, entities)
        
        # Concentration risk
        concentration_risk = self._calculate_concentration_risk(weights)
        
        return {
            'portfolio_expected_loss': portfolio_el,
            'portfolio_default_probability': portfolio_pd,
            'portfolio_var_95': portfolio_var_95,
            'portfolio_var_99': portfolio_var_99,
            'concentration_risk': concentration_risk,
            'diversification_benefit': np.sum(individual_els) - portfolio_el
        }
    
    def stress_test_credit_curves(self, cds_curves: List[CDSCurve],
                                scenarios: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, CreditRiskMetrics]]:
        """
        Perform stress testing on credit curves.
        
        Args:
            cds_curves: List of CDS curves
            scenarios: Dictionary of scenario name to stress parameters
            
        Returns:
            Dictionary of scenario to entity to stressed metrics
        """
        stress_results = {}
        
        for scenario_name, stress_params in scenarios.items():
            scenario_results = {}
            
            for curve in cds_curves:
                # Apply stress to curve
                stressed_curve = self._apply_stress_to_curve(curve, stress_params)
                
                # Calculate metrics under stress
                stressed_metrics = self.calculate_credit_metrics(stressed_curve)
                scenario_results[curve.entity] = stressed_metrics
            
            stress_results[scenario_name] = scenario_results
        
        return stress_results
    
    def _calculate_credit_var(self, probability_of_default: float, confidence_level: float) -> float:
        """Calculate Credit VaR for given confidence level."""
        # For binary default event, VaR is either 0 or full loss
        if probability_of_default >= (1 - confidence_level):
            return 1 - self.recovery_rate  # Full loss amount
        else:
            return 0.0
    
    def _calculate_jump_to_default_risk(self, cds_curve: CDSCurve, time_horizon: float) -> float:
        """Calculate jump-to-default risk."""
        # Simplified calculation - in practice would use more sophisticated models
        hazard_rate = cds_curve.get_hazard_rate(time_horizon)
        
        # Jump-to-default risk as instantaneous default probability
        jtd_risk = hazard_rate * (1 - self.recovery_rate)
        
        return jtd_risk
    
    def _calculate_term_structure(self, cds_curve: CDSCurve,
                                risk_free_curve: Optional[YieldCurve] = None) -> Dict[float, Dict[str, float]]:
        """Calculate term structure of credit metrics."""
        maturities = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20]
        term_structure = {}
        
        for maturity in maturities:
            survival_prob = cds_curve.get_survival_probability(maturity)
            pd = 1 - survival_prob
            hazard_rate = cds_curve.get_hazard_rate(maturity)
            
            if risk_free_curve:
                risk_free_rate = risk_free_curve.get_yield(maturity)
                risky_rate = -math.log(survival_prob) / maturity if survival_prob > 0 else 0
                credit_spread = risky_rate - risk_free_rate
            else:
                cds_spread = cds_curve.get_spread(maturity)
                credit_spread = cds_spread / (1 - self.recovery_rate)
            
            term_structure[maturity] = {
                'survival_probability': survival_prob,
                'default_probability': pd,
                'hazard_rate': hazard_rate,
                'credit_spread': credit_spread,
                'expected_loss': pd * (1 - self.recovery_rate)
            }
        
        return term_structure
    
    def _calculate_z_score(self, value: float, benchmark: float, volatility: float = 0.01) -> float:
        """Calculate Z-score for relative value analysis."""
        return (value - benchmark) / volatility
    
    def _calculate_correlated_portfolio_pd(self, individual_pds: np.ndarray,
                                         weights: np.ndarray,
                                         correlation_matrix: CreditCorrelationMatrix,
                                         entities: List[str]) -> float:
        """Calculate portfolio default probability with correlations."""
        # Simplified analytical approximation
        # In practice, would use Monte Carlo simulation
        
        # Weighted average PD
        avg_pd = np.sum(weights * individual_pds)
        
        # Correlation adjustment
        avg_correlation = 0.0
        total_weight_pairs = 0.0
        
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities):
                if i != j:
                    corr = correlation_matrix.get_correlation(entity1, entity2)
                    weight_product = weights[i] * weights[j]
                    avg_correlation += corr * weight_product
                    total_weight_pairs += weight_product
        
        if total_weight_pairs > 0:
            avg_correlation /= total_weight_pairs
        
        # Adjust portfolio PD based on correlation
        # Higher correlation increases portfolio risk
        correlation_adjustment = 1 + avg_correlation * 0.5  # Simplified adjustment
        
        return min(avg_pd * correlation_adjustment, 1.0)
    
    def _calculate_portfolio_var(self, individual_els: np.ndarray,
                               weights: np.ndarray,
                               confidence_level: float,
                               correlation_matrix: Optional[CreditCorrelationMatrix],
                               entities: List[str]) -> float:
        """Calculate portfolio Credit VaR."""
        # Portfolio expected loss
        portfolio_el = np.sum(weights * individual_els)
        
        if correlation_matrix is None:
            # Independent case - use normal approximation
            portfolio_variance = np.sum((weights * individual_els) ** 2)
        else:
            # Correlated case
            portfolio_variance = 0.0
            for i, entity1 in enumerate(entities):
                for j, entity2 in enumerate(entities):
                    corr = correlation_matrix.get_correlation(entity1, entity2) if i != j else 1.0
                    portfolio_variance += weights[i] * weights[j] * individual_els[i] * individual_els[j] * corr
        
        portfolio_std = math.sqrt(portfolio_variance)
        
        # Normal approximation for VaR
        from scipy.stats import norm
        z_score = norm.ppf(confidence_level)
        
        portfolio_var = portfolio_el + z_score * portfolio_std
        
        return max(portfolio_var, 0.0)
    
    def _calculate_concentration_risk(self, weights: np.ndarray) -> float:
        """Calculate concentration risk using Herfindahl index."""
        # Herfindahl-Hirschman Index
        hhi = np.sum(weights ** 2)
        
        # Normalize to [0, 1] where 1 is maximum concentration
        n = len(weights)
        normalized_hhi = (hhi - 1/n) / (1 - 1/n) if n > 1 else 0
        
        return normalized_hhi
    
    def _apply_stress_to_curve(self, curve: CDSCurve, stress_params: Dict[str, float]) -> CDSCurve:
        """Apply stress scenario to CDS curve."""
        # Create a copy of the curve with stressed parameters
        stressed_quotes = []
        
        for quote in curve.quotes:
            stressed_value = quote.value
            
            # Apply parallel shift
            if 'parallel_shift_bp' in stress_params:
                if quote.quote_type == "spread":
                    stressed_value += stress_params['parallel_shift_bp']
            
            # Apply multiplicative shock
            if 'spread_multiplier' in stress_params:
                if quote.quote_type == "spread":
                    stressed_value *= stress_params['spread_multiplier']
            
            # Apply steepening/flattening
            if 'steepening_bp' in stress_params:
                if quote.quote_type == "spread":
                    # Apply more stress to longer maturities
                    steepening_factor = min(quote.maturity_years / 10.0, 1.0)
                    stressed_value += stress_params['steepening_bp'] * steepening_factor
            
            stressed_quote = CDSQuote(
                maturity_years=quote.maturity_years,
                value=max(stressed_value, 0.1),  # Minimum 0.1bp spread
                quote_type=quote.quote_type,
                convention=quote.convention,
                quote_date=quote.quote_date,
                running_spread=quote.running_spread
            )
            stressed_quotes.append(stressed_quote)
        
        # Create new curve with stressed quotes
        from .cds_curve_builder import CDSCurveBuilder
        from ..pricing.yield_curve import YieldCurve, YieldPoint
        
        # Create a dummy risk-free curve for bootstrapping
        dummy_points = [YieldPoint(maturity=i, yield_rate=0.02) for i in [1, 5, 10]]
        dummy_curve = YieldCurve("USD", curve.curve_date, dummy_points)
        
        builder = CDSCurveBuilder()
        stressed_curve = builder.build_curve(
            curve.entity, curve.currency, curve.curve_date,
            stressed_quotes, dummy_curve, curve.recovery_rate
        )
        
        return stressed_curve


class CreditCorrelationEstimator:
    """Estimator for credit correlations."""
    
    @staticmethod
    def estimate_asset_correlations(cds_curves: List[CDSCurve],
                                  historical_data: Optional[Dict[str, List[float]]] = None) -> CreditCorrelationMatrix:
        """
        Estimate asset correlations between entities.
        
        Args:
            cds_curves: List of CDS curves
            historical_data: Historical CDS spread data
            
        Returns:
            Credit correlation matrix
        """
        entities = [curve.entity for curve in cds_curves]
        n = len(entities)
        
        if historical_data:
            # Use historical data to estimate correlations
            correlation_matrix = np.eye(n)
            
            for i, entity1 in enumerate(entities):
                for j, entity2 in enumerate(entities):
                    if i != j and entity1 in historical_data and entity2 in historical_data:
                        data1 = np.array(historical_data[entity1])
                        data2 = np.array(historical_data[entity2])
                        
                        if len(data1) == len(data2) and len(data1) > 1:
                            corr = np.corrcoef(data1, data2)[0, 1]
                            correlation_matrix[i, j] = corr if not np.isnan(corr) else 0.0
        else:
            # Use default correlation structure based on regions/sectors
            correlation_matrix = CreditCorrelationEstimator._default_correlation_structure(entities)
        
        return CreditCorrelationMatrix(
            entities=entities,
            correlation_matrix=correlation_matrix,
            calculation_date=date.today(),
            methodology="asset_correlation"
        )
    
    @staticmethod
    def _default_correlation_structure(entities: List[str]) -> np.ndarray:
        """Create default correlation structure."""
        n = len(entities)
        correlation_matrix = np.eye(n)
        
        # Simple heuristic: higher correlation for similar entities
        base_correlation = 0.3
        
        for i in range(n):
            for j in range(i + 1, n):
                # Default correlation
                corr = base_correlation
                
                # Increase correlation for similar country names (simplified)
                entity1, entity2 = entities[i].upper(), entities[j].upper()
                
                # Regional correlations (simplified)
                if any(region in entity1 and region in entity2 for region in ['EU', 'EUROPE', 'ASIA', 'AMERICA']):
                    corr += 0.2
                
                correlation_matrix[i, j] = corr
                correlation_matrix[j, i] = corr
        
        return correlation_matrix