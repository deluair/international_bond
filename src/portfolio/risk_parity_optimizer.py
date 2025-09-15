"""
Risk parity optimizer for equal risk contribution portfolio construction.
"""

import numpy as np
from scipy.optimize import minimize
from datetime import date, datetime
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import math

from ..models.bond import SovereignBond
from ..models.portfolio import Portfolio, Position


class RiskParityMethod(Enum):
    """Risk parity optimization methods."""
    EQUAL_RISK_CONTRIBUTION = "equal_risk_contribution"
    EQUAL_VOLATILITY = "equal_volatility"
    INVERSE_VOLATILITY = "inverse_volatility"
    HIERARCHICAL_RISK_PARITY = "hierarchical_risk_parity"
    RISK_BUDGETING = "risk_budgeting"


class RiskMeasure(Enum):
    """Risk measures for optimization."""
    VOLATILITY = "volatility"
    VALUE_AT_RISK = "value_at_risk"
    EXPECTED_SHORTFALL = "expected_shortfall"
    MAXIMUM_DRAWDOWN = "maximum_drawdown"
    TRACKING_ERROR = "tracking_error"


@dataclass
class RiskBudget:
    """Risk budget specification."""
    asset_id: str
    target_risk_contribution: float  # Target percentage of total risk
    min_weight: float = 0.0
    max_weight: float = 1.0
    
    def __post_init__(self):
        if not 0 <= self.target_risk_contribution <= 1:
            raise ValueError("Target risk contribution must be between 0 and 1")
        if not 0 <= self.min_weight <= self.max_weight <= 1:
            raise ValueError("Invalid weight constraints")


@dataclass
class RiskParityConstraints:
    """Risk parity optimization constraints."""
    # Weight constraints
    min_weight: float = 0.0
    max_weight: float = 1.0
    sum_weights: float = 1.0
    
    # Risk constraints
    max_portfolio_volatility: float = 0.20  # 20%
    max_concentration: float = 0.50  # 50% max in single asset
    
    # Sector/country constraints
    max_country_weight: float = 0.30  # 30%
    max_sector_weight: float = 0.40  # 40%
    
    # Risk contribution constraints
    max_risk_contribution: float = 0.50  # 50% max risk from single asset
    min_risk_contribution: float = 0.01  # 1% min risk from single asset
    
    # Optimization parameters
    tolerance: float = 1e-8
    max_iterations: int = 1000
    
    # Risk budgets (optional)
    risk_budgets: List[RiskBudget] = field(default_factory=list)
    
    def validate_risk_budgets(self) -> bool:
        """Validate that risk budgets sum to 1."""
        if not self.risk_budgets:
            return True
        
        total_budget = sum(budget.target_risk_contribution for budget in self.risk_budgets)
        return abs(total_budget - 1.0) < self.tolerance


@dataclass
class RiskParityResult:
    """Risk parity optimization result."""
    # Optimal weights
    weights: np.ndarray
    asset_ids: List[str]
    
    # Risk contributions
    risk_contributions: np.ndarray
    risk_contribution_pct: np.ndarray
    
    # Portfolio metrics
    portfolio_volatility: float
    portfolio_return: float
    sharpe_ratio: float
    
    # Risk parity metrics
    risk_concentration: float  # Herfindahl index of risk contributions
    diversification_ratio: float
    effective_number_of_assets: float
    
    # Optimization details
    optimization_success: bool
    iterations: int
    objective_value: float
    
    # Constraints satisfaction
    weight_constraints_satisfied: bool
    risk_constraints_satisfied: bool
    
    def __str__(self) -> str:
        return (f"Risk Parity Portfolio:\n"
                f"Portfolio Volatility: {self.portfolio_volatility:.2%}\n"
                f"Portfolio Return: {self.portfolio_return:.2%}\n"
                f"Sharpe Ratio: {self.sharpe_ratio:.2f}\n"
                f"Risk Concentration: {self.risk_concentration:.3f}\n"
                f"Diversification Ratio: {self.diversification_ratio:.2f}\n"
                f"Effective Assets: {self.effective_number_of_assets:.1f}\n"
                f"Optimization Success: {self.optimization_success}")


class RiskParityOptimizer:
    """
    Comprehensive risk parity optimizer for bond portfolios.
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize risk parity optimizer.
        
        Args:
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
        """
        self.risk_free_rate = risk_free_rate
        self.optimization_history: List[RiskParityResult] = []
    
    def optimize_portfolio(self, 
                          bonds: List[SovereignBond],
                          expected_returns: np.ndarray,
                          covariance_matrix: np.ndarray,
                          constraints: RiskParityConstraints,
                          method: RiskParityMethod = RiskParityMethod.EQUAL_RISK_CONTRIBUTION) -> RiskParityResult:
        """
        Optimize portfolio using risk parity approach.
        
        Args:
            bonds: List of sovereign bonds
            expected_returns: Expected returns vector
            covariance_matrix: Covariance matrix
            constraints: Optimization constraints
            method: Risk parity method
            
        Returns:
            RiskParityResult object
        """
        n_assets = len(bonds)
        asset_ids = [bond.isin for bond in bonds]
        
        if len(expected_returns) != n_assets or covariance_matrix.shape != (n_assets, n_assets):
            raise ValueError("Dimension mismatch between bonds, returns, and covariance matrix")
        
        # Validate constraints
        if constraints.risk_budgets and not constraints.validate_risk_budgets():
            raise ValueError("Risk budgets do not sum to 1")
        
        # Initial weights (equal weight)
        initial_weights = np.ones(n_assets) / n_assets
        
        # Set up optimization based on method
        if method == RiskParityMethod.EQUAL_RISK_CONTRIBUTION:
            result = self._optimize_equal_risk_contribution(
                initial_weights, covariance_matrix, constraints
            )
        elif method == RiskParityMethod.EQUAL_VOLATILITY:
            result = self._optimize_equal_volatility(
                initial_weights, covariance_matrix, constraints
            )
        elif method == RiskParityMethod.INVERSE_VOLATILITY:
            result = self._optimize_inverse_volatility(
                covariance_matrix, constraints
            )
        elif method == RiskParityMethod.HIERARCHICAL_RISK_PARITY:
            result = self._optimize_hierarchical_risk_parity(
                covariance_matrix, constraints
            )
        elif method == RiskParityMethod.RISK_BUDGETING:
            result = self._optimize_risk_budgeting(
                initial_weights, covariance_matrix, constraints
            )
        else:
            raise ValueError(f"Unknown risk parity method: {method}")
        
        # Calculate portfolio metrics
        portfolio_return = np.dot(result.x, expected_returns)
        portfolio_volatility = np.sqrt(np.dot(result.x, np.dot(covariance_matrix, result.x)))
        
        # Calculate risk contributions
        risk_contributions = self._calculate_risk_contributions(result.x, covariance_matrix)
        risk_contribution_pct = risk_contributions / np.sum(risk_contributions)
        
        # Calculate risk parity metrics
        risk_concentration = np.sum(risk_contribution_pct ** 2)  # Herfindahl index
        diversification_ratio = self._calculate_diversification_ratio(result.x, covariance_matrix)
        effective_number_of_assets = 1.0 / risk_concentration
        
        # Calculate Sharpe ratio
        if portfolio_volatility > 0:
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
        else:
            sharpe_ratio = 0.0
        
        # Check constraints satisfaction
        weight_constraints_satisfied = self._check_weight_constraints(result.x, constraints)
        risk_constraints_satisfied = self._check_risk_constraints(
            result.x, covariance_matrix, risk_contribution_pct, constraints
        )
        
        # Create result object
        rp_result = RiskParityResult(
            weights=result.x,
            asset_ids=asset_ids,
            risk_contributions=risk_contributions,
            risk_contribution_pct=risk_contribution_pct,
            portfolio_volatility=portfolio_volatility,
            portfolio_return=portfolio_return,
            sharpe_ratio=sharpe_ratio,
            risk_concentration=risk_concentration,
            diversification_ratio=diversification_ratio,
            effective_number_of_assets=effective_number_of_assets,
            optimization_success=result.success,
            iterations=result.nit if hasattr(result, 'nit') else 0,
            objective_value=result.fun,
            weight_constraints_satisfied=weight_constraints_satisfied,
            risk_constraints_satisfied=risk_constraints_satisfied
        )
        
        self.optimization_history.append(rp_result)
        return rp_result
    
    def _optimize_equal_risk_contribution(self, 
                                        initial_weights: np.ndarray,
                                        covariance_matrix: np.ndarray,
                                        constraints: RiskParityConstraints) -> object:
        """Optimize for equal risk contribution."""
        
        def objective(weights):
            """Minimize sum of squared deviations from equal risk contribution."""
            risk_contributions = self._calculate_risk_contributions(weights, covariance_matrix)
            total_risk = np.sum(risk_contributions)
            
            if total_risk == 0:
                return 1e10
            
            risk_contribution_pct = risk_contributions / total_risk
            target_contribution = 1.0 / len(weights)
            
            # Sum of squared deviations from equal contribution
            return np.sum((risk_contribution_pct - target_contribution) ** 2)
        
        # Set up constraints
        optimization_constraints = self._setup_optimization_constraints(constraints, len(initial_weights))
        
        # Bounds
        bounds = [(constraints.min_weight, constraints.max_weight) for _ in range(len(initial_weights))]
        
        # Optimize
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=optimization_constraints,
            options={'maxiter': constraints.max_iterations, 'ftol': constraints.tolerance}
        )
        
        return result
    
    def _optimize_equal_volatility(self, 
                                 initial_weights: np.ndarray,
                                 covariance_matrix: np.ndarray,
                                 constraints: RiskParityConstraints) -> object:
        """Optimize for equal volatility contribution."""
        
        # Calculate individual asset volatilities
        asset_volatilities = np.sqrt(np.diag(covariance_matrix))
        
        # Inverse volatility weights
        inv_vol_weights = 1.0 / asset_volatilities
        weights = inv_vol_weights / np.sum(inv_vol_weights)
        
        # Create mock result object
        class MockResult:
            def __init__(self, weights):
                self.x = weights
                self.success = True
                self.fun = 0.0
                self.nit = 1
        
        return MockResult(weights)
    
    def _optimize_inverse_volatility(self, 
                                   covariance_matrix: np.ndarray,
                                   constraints: RiskParityConstraints) -> object:
        """Optimize using inverse volatility weighting."""
        
        # Calculate individual asset volatilities
        asset_volatilities = np.sqrt(np.diag(covariance_matrix))
        
        # Inverse volatility weights
        inv_vol_weights = 1.0 / asset_volatilities
        weights = inv_vol_weights / np.sum(inv_vol_weights)
        
        # Apply weight constraints
        weights = np.clip(weights, constraints.min_weight, constraints.max_weight)
        weights = weights / np.sum(weights)  # Renormalize
        
        # Create mock result object
        class MockResult:
            def __init__(self, weights):
                self.x = weights
                self.success = True
                self.fun = 0.0
                self.nit = 1
        
        return MockResult(weights)
    
    def _optimize_hierarchical_risk_parity(self, 
                                         covariance_matrix: np.ndarray,
                                         constraints: RiskParityConstraints) -> object:
        """Optimize using hierarchical risk parity (simplified version)."""
        
        # Simplified HRP: use inverse volatility as starting point
        asset_volatilities = np.sqrt(np.diag(covariance_matrix))
        
        # Calculate correlation matrix
        correlation_matrix = self._cov_to_corr(covariance_matrix)
        
        # Hierarchical clustering (simplified)
        # In practice, this would use proper clustering algorithms
        n_assets = len(asset_volatilities)
        
        # For simplicity, create two clusters based on correlation
        if n_assets > 2:
            # Find most correlated pair
            np.fill_diagonal(correlation_matrix, 0)  # Remove self-correlation
            max_corr_idx = np.unravel_index(np.argmax(correlation_matrix), correlation_matrix.shape)
            
            cluster1 = [max_corr_idx[0], max_corr_idx[1]]
            cluster2 = [i for i in range(n_assets) if i not in cluster1]
            
            # Allocate 50% to each cluster, then use inverse vol within clusters
            cluster1_weights = np.zeros(n_assets)
            cluster2_weights = np.zeros(n_assets)
            
            if cluster1:
                cluster1_vols = asset_volatilities[cluster1]
                cluster1_inv_vol = 1.0 / cluster1_vols
                cluster1_inv_vol = cluster1_inv_vol / np.sum(cluster1_inv_vol) * 0.5
                cluster1_weights[cluster1] = cluster1_inv_vol
            
            if cluster2:
                cluster2_vols = asset_volatilities[cluster2]
                cluster2_inv_vol = 1.0 / cluster2_vols
                cluster2_inv_vol = cluster2_inv_vol / np.sum(cluster2_inv_vol) * 0.5
                cluster2_weights[cluster2] = cluster2_inv_vol
            
            weights = cluster1_weights + cluster2_weights
        else:
            # Fall back to inverse volatility
            inv_vol_weights = 1.0 / asset_volatilities
            weights = inv_vol_weights / np.sum(inv_vol_weights)
        
        # Create mock result object
        class MockResult:
            def __init__(self, weights):
                self.x = weights
                self.success = True
                self.fun = 0.0
                self.nit = 1
        
        return MockResult(weights)
    
    def _optimize_risk_budgeting(self, 
                               initial_weights: np.ndarray,
                               covariance_matrix: np.ndarray,
                               constraints: RiskParityConstraints) -> object:
        """Optimize for specific risk budgets."""
        
        if not constraints.risk_budgets:
            # Fall back to equal risk contribution
            return self._optimize_equal_risk_contribution(initial_weights, covariance_matrix, constraints)
        
        # Create target risk contribution vector
        n_assets = len(initial_weights)
        target_risk_contributions = np.zeros(n_assets)
        
        for budget in constraints.risk_budgets:
            # Find asset index (simplified - assumes asset_id is index)
            try:
                asset_idx = int(budget.asset_id)
                if 0 <= asset_idx < n_assets:
                    target_risk_contributions[asset_idx] = budget.target_risk_contribution
            except (ValueError, IndexError):
                continue
        
        # If not all assets have budgets, distribute remaining equally
        remaining_budget = 1.0 - np.sum(target_risk_contributions)
        unassigned_assets = np.where(target_risk_contributions == 0)[0]
        
        if len(unassigned_assets) > 0 and remaining_budget > 0:
            target_risk_contributions[unassigned_assets] = remaining_budget / len(unassigned_assets)
        
        def objective(weights):
            """Minimize sum of squared deviations from target risk contributions."""
            risk_contributions = self._calculate_risk_contributions(weights, covariance_matrix)
            total_risk = np.sum(risk_contributions)
            
            if total_risk == 0:
                return 1e10
            
            risk_contribution_pct = risk_contributions / total_risk
            
            # Sum of squared deviations from target contributions
            return np.sum((risk_contribution_pct - target_risk_contributions) ** 2)
        
        # Set up constraints
        optimization_constraints = self._setup_optimization_constraints(constraints, len(initial_weights))
        
        # Add risk budget constraints
        for budget in constraints.risk_budgets:
            try:
                asset_idx = int(budget.asset_id)
                if 0 <= asset_idx < n_assets:
                    # Weight bounds for this asset
                    optimization_constraints.append({
                        'type': 'ineq',
                        'fun': lambda w, idx=asset_idx, min_w=budget.min_weight: w[idx] - min_w
                    })
                    optimization_constraints.append({
                        'type': 'ineq', 
                        'fun': lambda w, idx=asset_idx, max_w=budget.max_weight: max_w - w[idx]
                    })
            except (ValueError, IndexError):
                continue
        
        # Bounds
        bounds = [(constraints.min_weight, constraints.max_weight) for _ in range(len(initial_weights))]
        
        # Optimize
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=optimization_constraints,
            options={'maxiter': constraints.max_iterations, 'ftol': constraints.tolerance}
        )
        
        return result
    
    def _calculate_risk_contributions(self, 
                                    weights: np.ndarray,
                                    covariance_matrix: np.ndarray) -> np.ndarray:
        """Calculate marginal risk contributions."""
        portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
        
        if portfolio_variance == 0:
            return np.zeros_like(weights)
        
        # Marginal contribution to risk
        marginal_contrib = np.dot(covariance_matrix, weights)
        
        # Risk contribution = weight * marginal contribution
        risk_contributions = weights * marginal_contrib
        
        return risk_contributions
    
    def _calculate_diversification_ratio(self, 
                                       weights: np.ndarray,
                                       covariance_matrix: np.ndarray) -> float:
        """Calculate diversification ratio."""
        # Weighted average of individual volatilities
        individual_vols = np.sqrt(np.diag(covariance_matrix))
        weighted_avg_vol = np.dot(weights, individual_vols)
        
        # Portfolio volatility
        portfolio_vol = np.sqrt(np.dot(weights, np.dot(covariance_matrix, weights)))
        
        if portfolio_vol > 0:
            return weighted_avg_vol / portfolio_vol
        else:
            return 1.0
    
    def _setup_optimization_constraints(self, 
                                      constraints: RiskParityConstraints,
                                      n_assets: int) -> List[Dict]:
        """Set up optimization constraints."""
        opt_constraints = []
        
        # Sum of weights constraint
        opt_constraints.append({
            'type': 'eq',
            'fun': lambda w: np.sum(w) - constraints.sum_weights
        })
        
        # Maximum concentration constraint
        if constraints.max_concentration < 1.0:
            for i in range(n_assets):
                opt_constraints.append({
                    'type': 'ineq',
                    'fun': lambda w, idx=i: constraints.max_concentration - w[idx]
                })
        
        return opt_constraints
    
    def _check_weight_constraints(self, 
                                weights: np.ndarray,
                                constraints: RiskParityConstraints) -> bool:
        """Check if weight constraints are satisfied."""
        # Check individual weight bounds
        if np.any(weights < constraints.min_weight - constraints.tolerance):
            return False
        if np.any(weights > constraints.max_weight + constraints.tolerance):
            return False
        
        # Check sum of weights
        if abs(np.sum(weights) - constraints.sum_weights) > constraints.tolerance:
            return False
        
        # Check concentration constraint
        if np.any(weights > constraints.max_concentration + constraints.tolerance):
            return False
        
        return True
    
    def _check_risk_constraints(self, 
                              weights: np.ndarray,
                              covariance_matrix: np.ndarray,
                              risk_contribution_pct: np.ndarray,
                              constraints: RiskParityConstraints) -> bool:
        """Check if risk constraints are satisfied."""
        # Check portfolio volatility
        portfolio_vol = np.sqrt(np.dot(weights, np.dot(covariance_matrix, weights)))
        if portfolio_vol > constraints.max_portfolio_volatility + constraints.tolerance:
            return False
        
        # Check risk contribution bounds
        if np.any(risk_contribution_pct > constraints.max_risk_contribution + constraints.tolerance):
            return False
        if np.any(risk_contribution_pct < constraints.min_risk_contribution - constraints.tolerance):
            return False
        
        return True
    
    def _cov_to_corr(self, covariance_matrix: np.ndarray) -> np.ndarray:
        """Convert covariance matrix to correlation matrix."""
        std_devs = np.sqrt(np.diag(covariance_matrix))
        correlation_matrix = covariance_matrix / np.outer(std_devs, std_devs)
        return correlation_matrix
    
    def calculate_risk_attribution(self, 
                                 weights: np.ndarray,
                                 covariance_matrix: np.ndarray,
                                 asset_ids: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Calculate detailed risk attribution.
        
        Args:
            weights: Portfolio weights
            covariance_matrix: Covariance matrix
            asset_ids: Asset identifiers
            
        Returns:
            Dictionary with risk attribution details
        """
        risk_contributions = self._calculate_risk_contributions(weights, covariance_matrix)
        total_risk = np.sum(risk_contributions)
        
        attribution = {}
        
        for i, asset_id in enumerate(asset_ids):
            attribution[asset_id] = {
                'weight': weights[i],
                'risk_contribution': risk_contributions[i],
                'risk_contribution_pct': risk_contributions[i] / total_risk if total_risk > 0 else 0.0,
                'individual_volatility': np.sqrt(covariance_matrix[i, i]),
                'marginal_risk': np.dot(covariance_matrix[i], weights)
            }
        
        return attribution