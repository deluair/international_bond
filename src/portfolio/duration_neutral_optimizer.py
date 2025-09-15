"""
Duration-neutral portfolio optimizer for constructing interest rate risk-neutral bond portfolios.
"""

import numpy as np
from datetime import date, datetime
from typing import List, Dict, Tuple, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import math
from scipy import optimize, linalg
import warnings

from ..models.bond import SovereignBond
from ..models.portfolio import Portfolio, Position
from ..pricing.duration_analytics import DurationAnalytics, DurationType


class OptimizationObjective(Enum):
    """Portfolio optimization objectives."""
    MAXIMIZE_YIELD = "maximize_yield"
    MINIMIZE_RISK = "minimize_risk"
    MAXIMIZE_SHARPE = "maximize_sharpe"
    MAXIMIZE_ALPHA = "maximize_alpha"
    MINIMIZE_TRACKING_ERROR = "minimize_tracking_error"
    RISK_PARITY = "risk_parity"
    EQUAL_WEIGHT = "equal_weight"


class DurationMatchingMethod(Enum):
    """Duration matching methods."""
    EXACT_MATCH = "exact_match"
    TOLERANCE_BASED = "tolerance_based"
    WEIGHTED_AVERAGE = "weighted_average"
    KEY_RATE_DURATION = "key_rate_duration"
    EFFECTIVE_DURATION = "effective_duration"


@dataclass
class PortfolioConstraints:
    """Portfolio optimization constraints."""
    # Duration constraints
    target_duration: float = 0.0
    duration_tolerance: float = 0.1
    duration_matching_method: DurationMatchingMethod = DurationMatchingMethod.TOLERANCE_BASED
    
    # Weight constraints
    min_weight: float = 0.0
    max_weight: float = 1.0
    max_single_position: float = 0.2
    
    # Country/sector constraints
    max_country_weight: Dict[str, float] = field(default_factory=dict)
    min_country_weight: Dict[str, float] = field(default_factory=dict)
    max_sector_concentration: float = 0.5
    
    # Credit quality constraints
    min_credit_rating: Optional[str] = None
    max_credit_risk: float = 1.0
    
    # Maturity constraints
    min_maturity: Optional[float] = None
    max_maturity: Optional[float] = None
    
    # Liquidity constraints
    min_issue_size: Optional[float] = None
    min_trading_volume: Optional[float] = None
    
    # Turnover constraints
    max_turnover: float = 1.0
    transaction_cost: float = 0.001
    
    # Risk constraints
    max_tracking_error: Optional[float] = None
    max_var: Optional[float] = None
    
    def validate(self) -> bool:
        """Validate constraint consistency."""
        if self.min_weight > self.max_weight:
            return False
        if self.target_duration < 0:
            return False
        if self.duration_tolerance < 0:
            return False
        return True


@dataclass
class OptimizationResult:
    """Portfolio optimization result."""
    success: bool
    optimal_weights: Dict[str, float]
    portfolio_duration: float
    duration_error: float
    expected_return: float
    expected_risk: float
    sharpe_ratio: float
    tracking_error: Optional[float]
    turnover: float
    transaction_costs: float
    objective_value: float
    constraint_violations: List[str]
    optimization_time: float
    iterations: int
    
    def __str__(self) -> str:
        return (f"Optimization Result:\n"
                f"Success: {self.success}\n"
                f"Portfolio Duration: {self.portfolio_duration:.3f}\n"
                f"Duration Error: {self.duration_error:.4f}\n"
                f"Expected Return: {self.expected_return:.2%}\n"
                f"Expected Risk: {self.expected_risk:.2%}\n"
                f"Sharpe Ratio: {self.sharpe_ratio:.2f}\n"
                f"Turnover: {self.turnover:.2%}\n"
                f"Transaction Costs: {self.transaction_costs:.4f}")


class DurationNeutralOptimizer:
    """
    Advanced duration-neutral portfolio optimizer.
    """
    
    def __init__(self, 
                 duration_analytics: Optional[DurationAnalytics] = None,
                 risk_free_rate: float = 0.02):
        """
        Initialize duration-neutral optimizer.
        
        Args:
            duration_analytics: Duration analytics engine
            risk_free_rate: Risk-free rate for Sharpe ratio calculations
        """
        self.duration_analytics = duration_analytics or DurationAnalytics()
        self.risk_free_rate = risk_free_rate
        self.optimization_history: List[OptimizationResult] = []
    
    def optimize_portfolio(self, 
                         bonds: List[SovereignBond],
                         constraints: PortfolioConstraints,
                         objective: OptimizationObjective = OptimizationObjective.MAXIMIZE_SHARPE,
                         current_weights: Optional[Dict[str, float]] = None,
                         expected_returns: Optional[Dict[str, float]] = None,
                         covariance_matrix: Optional[np.ndarray] = None,
                         benchmark_weights: Optional[Dict[str, float]] = None) -> OptimizationResult:
        """
        Optimize portfolio with duration-neutral constraints.
        
        Args:
            bonds: List of bonds to optimize over
            constraints: Portfolio constraints
            objective: Optimization objective
            current_weights: Current portfolio weights
            expected_returns: Expected returns for each bond
            covariance_matrix: Return covariance matrix
            benchmark_weights: Benchmark weights for tracking error
            
        Returns:
            OptimizationResult object
        """
        start_time = datetime.now()
        
        # Validate inputs
        if not bonds:
            return self._create_failed_result("No bonds provided")
        
        if not constraints.validate():
            return self._create_failed_result("Invalid constraints")
        
        # Prepare optimization data
        n = len(bonds)
        bond_ids = [bond.isin for bond in bonds]
        
        # Calculate durations
        durations = self._calculate_bond_durations(bonds, constraints.duration_matching_method)
        
        # Prepare expected returns
        if expected_returns is None:
            expected_returns = {bond.isin: bond.yield_to_maturity or 0.03 for bond in bonds}
        
        returns_vector = np.array([expected_returns.get(bond_id, 0.0) for bond_id in bond_ids])
        
        # Prepare covariance matrix
        if covariance_matrix is None:
            covariance_matrix = self._estimate_covariance_matrix(bonds, returns_vector)
        
        # Current weights
        if current_weights is None:
            current_weights = {bond_id: 0.0 for bond_id in bond_ids}
        
        current_weights_vector = np.array([current_weights.get(bond_id, 0.0) for bond_id in bond_ids])
        
        # Set up optimization problem
        try:
            result = self._solve_optimization(
                bonds=bonds,
                bond_ids=bond_ids,
                durations=durations,
                returns_vector=returns_vector,
                covariance_matrix=covariance_matrix,
                current_weights_vector=current_weights_vector,
                constraints=constraints,
                objective=objective,
                benchmark_weights=benchmark_weights
            )
            
            # Calculate portfolio metrics
            optimal_weights_dict = {bond_ids[i]: result.x[i] for i in range(n)}
            portfolio_duration = np.dot(result.x, durations)
            duration_error = abs(portfolio_duration - constraints.target_duration)
            
            expected_return = np.dot(result.x, returns_vector)
            expected_risk = math.sqrt(np.dot(result.x, np.dot(covariance_matrix, result.x)))
            sharpe_ratio = (expected_return - self.risk_free_rate) / expected_risk if expected_risk > 0 else 0.0
            
            # Calculate turnover and transaction costs
            turnover = np.sum(np.abs(result.x - current_weights_vector))
            transaction_costs = turnover * constraints.transaction_cost
            
            # Calculate tracking error if benchmark provided
            tracking_error = None
            if benchmark_weights is not None:
                benchmark_vector = np.array([benchmark_weights.get(bond_id, 0.0) for bond_id in bond_ids])
                active_weights = result.x - benchmark_vector
                tracking_error = math.sqrt(np.dot(active_weights, np.dot(covariance_matrix, active_weights)))
            
            # Check constraint violations
            violations = self._check_constraint_violations(result.x, bonds, constraints, durations)
            
            optimization_time = (datetime.now() - start_time).total_seconds()
            
            opt_result = OptimizationResult(
                success=result.success,
                optimal_weights=optimal_weights_dict,
                portfolio_duration=portfolio_duration,
                duration_error=duration_error,
                expected_return=expected_return,
                expected_risk=expected_risk,
                sharpe_ratio=sharpe_ratio,
                tracking_error=tracking_error,
                turnover=turnover,
                transaction_costs=transaction_costs,
                objective_value=result.fun,
                constraint_violations=violations,
                optimization_time=optimization_time,
                iterations=getattr(result, 'nit', 0)
            )
            
            self.optimization_history.append(opt_result)
            return opt_result
            
        except Exception as e:
            return self._create_failed_result(f"Optimization failed: {str(e)}")
    
    def create_duration_ladder(self, 
                             bonds: List[SovereignBond],
                             target_duration: float,
                             duration_buckets: int = 5,
                             equal_weight: bool = True) -> Dict[str, float]:
        """
        Create a duration ladder portfolio.
        
        Args:
            bonds: Available bonds
            target_duration: Target portfolio duration
            duration_buckets: Number of duration buckets
            equal_weight: Whether to use equal weights
            
        Returns:
            Dictionary of bond weights
        """
        if not bonds:
            return {}
        
        # Calculate bond durations
        durations = self._calculate_bond_durations(bonds, DurationMatchingMethod.EFFECTIVE_DURATION)
        
        # Sort bonds by duration
        bond_duration_pairs = list(zip(bonds, durations))
        bond_duration_pairs.sort(key=lambda x: x[1])
        
        # Create duration buckets
        min_duration = min(durations)
        max_duration = max(durations)
        
        if max_duration <= min_duration:
            # All bonds have same duration
            if equal_weight:
                weight = 1.0 / len(bonds)
                return {bond.isin: weight for bond in bonds}
            else:
                return {bonds[0].isin: 1.0}
        
        bucket_size = (max_duration - min_duration) / duration_buckets
        buckets = []
        
        for i in range(duration_buckets):
            bucket_min = min_duration + i * bucket_size
            bucket_max = min_duration + (i + 1) * bucket_size
            bucket_bonds = [
                bond for bond, duration in bond_duration_pairs
                if bucket_min <= duration < bucket_max or (i == duration_buckets - 1 and duration <= bucket_max)
            ]
            if bucket_bonds:
                buckets.append(bucket_bonds)
        
        # Allocate weights
        weights = {}
        
        if equal_weight:
            # Equal weight across all bonds
            weight_per_bond = 1.0 / len(bonds)
            for bond in bonds:
                weights[bond.isin] = weight_per_bond
        else:
            # Equal weight per bucket, then equal within bucket
            weight_per_bucket = 1.0 / len(buckets)
            
            for bucket in buckets:
                weight_per_bond_in_bucket = weight_per_bucket / len(bucket)
                for bond in bucket:
                    weights[bond.isin] = weight_per_bond_in_bucket
        
        return weights
    
    def optimize_key_rate_duration_neutral(self, 
                                         bonds: List[SovereignBond],
                                         key_rate_durations: Dict[str, np.ndarray],
                                         target_key_rates: np.ndarray,
                                         constraints: PortfolioConstraints) -> OptimizationResult:
        """
        Optimize portfolio to be neutral to specific key rate durations.
        
        Args:
            bonds: Available bonds
            key_rate_durations: Key rate durations for each bond
            target_key_rates: Target key rate duration profile
            constraints: Portfolio constraints
            
        Returns:
            OptimizationResult object
        """
        if not bonds or not key_rate_durations:
            return self._create_failed_result("No bonds or key rate durations provided")
        
        n = len(bonds)
        bond_ids = [bond.isin for bond in bonds]
        
        # Build key rate duration matrix
        krd_matrix = []
        for bond_id in bond_ids:
            if bond_id in key_rate_durations:
                krd_matrix.append(key_rate_durations[bond_id])
            else:
                # Default to zero key rate durations
                krd_matrix.append(np.zeros(len(target_key_rates)))
        
        krd_matrix = np.array(krd_matrix).T  # Transpose for matrix multiplication
        
        # Objective: minimize tracking error to target key rate profile
        def objective(weights):
            portfolio_krd = np.dot(krd_matrix, weights)
            krd_error = np.sum((portfolio_krd - target_key_rates) ** 2)
            return krd_error
        
        # Constraints
        constraint_list = []
        
        # Sum to 1
        constraint_list.append({
            'type': 'eq',
            'fun': lambda w: np.sum(w) - 1.0
        })
        
        # Weight bounds
        bounds = [(constraints.min_weight, constraints.max_weight) for _ in range(n)]
        
        # Initial guess
        initial_weights = np.ones(n) / n
        
        # Optimize
        try:
            result = optimize.minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraint_list,
                options={'maxiter': 1000}
            )
            
            if result.success:
                optimal_weights = {bond_ids[i]: result.x[i] for i in range(n)}
                portfolio_krd = np.dot(krd_matrix, result.x)
                krd_error = np.linalg.norm(portfolio_krd - target_key_rates)
                
                return OptimizationResult(
                    success=True,
                    optimal_weights=optimal_weights,
                    portfolio_duration=np.sum(portfolio_krd),  # Sum of key rate durations
                    duration_error=krd_error,
                    expected_return=0.0,  # Not calculated
                    expected_risk=0.0,    # Not calculated
                    sharpe_ratio=0.0,     # Not calculated
                    tracking_error=None,
                    turnover=0.0,         # Not calculated
                    transaction_costs=0.0,
                    objective_value=result.fun,
                    constraint_violations=[],
                    optimization_time=0.0,
                    iterations=getattr(result, 'nit', 0)
                )
            else:
                return self._create_failed_result("Key rate duration optimization failed")
                
        except Exception as e:
            return self._create_failed_result(f"Key rate duration optimization error: {str(e)}")
    
    def calculate_duration_contribution(self, 
                                      weights: Dict[str, float],
                                      bonds: List[SovereignBond]) -> Dict[str, float]:
        """
        Calculate duration contribution of each bond to portfolio.
        
        Args:
            weights: Bond weights
            bonds: List of bonds
            
        Returns:
            Dictionary of duration contributions
        """
        contributions = {}
        
        for bond in bonds:
            weight = weights.get(bond.isin, 0.0)
            duration = self.duration_analytics.calculate_modified_duration(
                bond, bond.yield_to_maturity or 0.03
            )
            contributions[bond.isin] = weight * duration
        
        return contributions
    
    def _calculate_bond_durations(self, 
                                bonds: List[SovereignBond],
                                method: DurationMatchingMethod) -> np.ndarray:
        """Calculate durations for bonds using specified method."""
        durations = []
        
        for bond in bonds:
            ytm = bond.yield_to_maturity or 0.03
            
            if method == DurationMatchingMethod.EFFECTIVE_DURATION:
                duration = self.duration_analytics.calculate_effective_duration(bond, ytm)
            elif method == DurationMatchingMethod.KEY_RATE_DURATION:
                # Use sum of key rate durations as proxy
                krd_result = self.duration_analytics.calculate_key_rate_duration(bond, ytm)
                duration = sum(krd_result.key_rate_durations.values())
            else:
                # Default to modified duration
                duration = self.duration_analytics.calculate_modified_duration(bond, ytm)
            
            durations.append(duration)
        
        return np.array(durations)
    
    def _estimate_covariance_matrix(self, 
                                  bonds: List[SovereignBond],
                                  returns: np.ndarray) -> np.ndarray:
        """Estimate covariance matrix for bonds."""
        n = len(bonds)
        
        # Simple correlation model based on country and maturity
        correlation_matrix = np.eye(n)
        
        for i in range(n):
            for j in range(i + 1, n):
                # Country correlation
                country_corr = 0.7 if bonds[i].country == bonds[j].country else 0.3
                
                # Maturity correlation
                mat_i = bonds[i].maturity_date
                mat_j = bonds[j].maturity_date
                
                if mat_i and mat_j:
                    mat_diff = abs((mat_i - mat_j).days) / 365.25
                    maturity_corr = max(0.2, 1.0 - mat_diff * 0.1)
                else:
                    maturity_corr = 0.5
                
                # Combined correlation
                correlation = country_corr * maturity_corr
                correlation_matrix[i, j] = correlation
                correlation_matrix[j, i] = correlation
        
        # Estimate volatilities (simple approach)
        volatilities = np.array([0.05 + abs(ret) * 0.1 for ret in returns])
        
        # Build covariance matrix
        cov_matrix = np.outer(volatilities, volatilities) * correlation_matrix
        
        return cov_matrix
    
    def _solve_optimization(self, 
                          bonds: List[SovereignBond],
                          bond_ids: List[str],
                          durations: np.ndarray,
                          returns_vector: np.ndarray,
                          covariance_matrix: np.ndarray,
                          current_weights_vector: np.ndarray,
                          constraints: PortfolioConstraints,
                          objective: OptimizationObjective,
                          benchmark_weights: Optional[Dict[str, float]] = None) -> optimize.OptimizeResult:
        """Solve the optimization problem."""
        n = len(bonds)
        
        # Define objective function
        def objective_function(weights):
            if objective == OptimizationObjective.MAXIMIZE_YIELD:
                return -np.dot(weights, returns_vector)  # Minimize negative return
            elif objective == OptimizationObjective.MINIMIZE_RISK:
                return np.dot(weights, np.dot(covariance_matrix, weights))
            elif objective == OptimizationObjective.MAXIMIZE_SHARPE:
                portfolio_return = np.dot(weights, returns_vector)
                portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
                portfolio_vol = math.sqrt(max(0, portfolio_variance))
                if portfolio_vol > 0:
                    sharpe = (portfolio_return - self.risk_free_rate) / portfolio_vol
                    return -sharpe  # Minimize negative Sharpe
                else:
                    return 0.0
            elif objective == OptimizationObjective.RISK_PARITY:
                # Risk parity: minimize sum of squared risk contributions
                portfolio_vol = math.sqrt(np.dot(weights, np.dot(covariance_matrix, weights)))
                if portfolio_vol > 0:
                    marginal_risk = np.dot(covariance_matrix, weights) / portfolio_vol
                    risk_contributions = weights * marginal_risk
                    target_risk = portfolio_vol / n  # Equal risk contribution
                    return np.sum((risk_contributions - target_risk) ** 2)
                else:
                    return 0.0
            elif objective == OptimizationObjective.EQUAL_WEIGHT:
                # Minimize deviation from equal weights
                equal_weight = 1.0 / n
                return np.sum((weights - equal_weight) ** 2)
            else:
                # Default: maximize return
                return -np.dot(weights, returns_vector)
        
        # Set up constraints
        constraint_list = []
        
        # Sum to 1 constraint
        constraint_list.append({
            'type': 'eq',
            'fun': lambda w: np.sum(w) - 1.0
        })
        
        # Duration constraint
        if constraints.duration_matching_method == DurationMatchingMethod.EXACT_MATCH:
            constraint_list.append({
                'type': 'eq',
                'fun': lambda w: np.dot(w, durations) - constraints.target_duration
            })
        elif constraints.duration_matching_method == DurationMatchingMethod.TOLERANCE_BASED:
            # Duration within tolerance
            constraint_list.append({
                'type': 'ineq',
                'fun': lambda w: constraints.duration_tolerance - abs(np.dot(w, durations) - constraints.target_duration)
            })
        
        # Maximum single position constraint
        if constraints.max_single_position < 1.0:
            for i in range(n):
                constraint_list.append({
                    'type': 'ineq',
                    'fun': lambda w, idx=i: constraints.max_single_position - w[idx]
                })
        
        # Country concentration constraints
        if constraints.max_country_weight:
            country_groups = {}
            for i, bond in enumerate(bonds):
                country = bond.country
                if country not in country_groups:
                    country_groups[country] = []
                country_groups[country].append(i)
            
            for country, max_weight in constraints.max_country_weight.items():
                if country in country_groups:
                    indices = country_groups[country]
                    constraint_list.append({
                        'type': 'ineq',
                        'fun': lambda w, idx_list=indices: max_weight - np.sum([w[i] for i in idx_list])
                    })
        
        # Weight bounds
        bounds = []
        for i in range(n):
            min_w = constraints.min_weight
            max_w = min(constraints.max_weight, constraints.max_single_position)
            bounds.append((min_w, max_w))
        
        # Initial guess
        initial_weights = np.ones(n) / n
        
        # Solve optimization
        result = optimize.minimize(
            objective_function,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraint_list,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        return result
    
    def _check_constraint_violations(self, 
                                   weights: np.ndarray,
                                   bonds: List[SovereignBond],
                                   constraints: PortfolioConstraints,
                                   durations: np.ndarray) -> List[str]:
        """Check for constraint violations."""
        violations = []
        
        # Check weight sum
        weight_sum = np.sum(weights)
        if abs(weight_sum - 1.0) > 1e-6:
            violations.append(f"Weight sum violation: {weight_sum:.6f}")
        
        # Check duration constraint
        portfolio_duration = np.dot(weights, durations)
        duration_error = abs(portfolio_duration - constraints.target_duration)
        
        if constraints.duration_matching_method == DurationMatchingMethod.EXACT_MATCH:
            if duration_error > 1e-6:
                violations.append(f"Duration exact match violation: {duration_error:.6f}")
        elif constraints.duration_matching_method == DurationMatchingMethod.TOLERANCE_BASED:
            if duration_error > constraints.duration_tolerance:
                violations.append(f"Duration tolerance violation: {duration_error:.6f}")
        
        # Check weight bounds
        for i, weight in enumerate(weights):
            if weight < constraints.min_weight - 1e-6:
                violations.append(f"Min weight violation for {bonds[i].isin}: {weight:.6f}")
            if weight > constraints.max_weight + 1e-6:
                violations.append(f"Max weight violation for {bonds[i].isin}: {weight:.6f}")
            if weight > constraints.max_single_position + 1e-6:
                violations.append(f"Max single position violation for {bonds[i].isin}: {weight:.6f}")
        
        return violations
    
    def _create_failed_result(self, error_message: str) -> OptimizationResult:
        """Create a failed optimization result."""
        return OptimizationResult(
            success=False,
            optimal_weights={},
            portfolio_duration=0.0,
            duration_error=float('inf'),
            expected_return=0.0,
            expected_risk=0.0,
            sharpe_ratio=0.0,
            tracking_error=None,
            turnover=0.0,
            transaction_costs=0.0,
            objective_value=float('inf'),
            constraint_violations=[error_message],
            optimization_time=0.0,
            iterations=0
        )