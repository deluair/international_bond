"""
Value at Risk (VaR) Calculator Module

This module provides comprehensive VaR calculation capabilities for international bond portfolios
using multiple methodologies including parametric, historical simulation, and Monte Carlo.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy import stats
from scipy.optimize import minimize

class VaRMethod(Enum):
    """VaR calculation methods"""
    PARAMETRIC = "parametric"
    HISTORICAL_SIMULATION = "historical_simulation"
    MONTE_CARLO = "monte_carlo"
    CORNISH_FISHER = "cornish_fisher"
    EXTREME_VALUE_THEORY = "extreme_value_theory"
    FILTERED_HISTORICAL = "filtered_historical"

class RiskMeasure(Enum):
    """Risk measures"""
    VAR = "var"
    EXPECTED_SHORTFALL = "expected_shortfall"
    CONDITIONAL_VAR = "conditional_var"
    MAXIMUM_DRAWDOWN = "maximum_drawdown"

class DistributionType(Enum):
    """Distribution types for parametric VaR"""
    NORMAL = "normal"
    T_DISTRIBUTION = "t_distribution"
    SKEWED_T = "skewed_t"
    GENERALIZED_ERROR = "generalized_error"

@dataclass
class VaRResult:
    """VaR calculation result"""
    method: VaRMethod
    confidence_level: float
    time_horizon: int  # days
    var_absolute: float  # absolute VaR
    var_relative: float  # relative VaR (%)
    expected_shortfall: float
    portfolio_value: float
    calculation_date: datetime
    
@dataclass
class ComponentVaR:
    """Component VaR for portfolio decomposition"""
    asset_id: str
    component_var: float
    marginal_var: float
    contribution_percentage: float
    standalone_var: float
    
@dataclass
class BacktestResult:
    """VaR backtesting result"""
    method: VaRMethod
    confidence_level: float
    total_observations: int
    violations: int
    violation_rate: float
    expected_violations: int
    kupiec_test_statistic: float
    kupiec_p_value: float
    christoffersen_test_statistic: float
    christoffersen_p_value: float
    
@dataclass
class RiskDecomposition:
    """Risk decomposition analysis"""
    total_var: float
    systematic_var: float
    idiosyncratic_var: float
    currency_var: float
    interest_rate_var: float
    credit_var: float
    diversification_benefit: float

class VaRCalculator:
    """
    Comprehensive Value at Risk calculator for international bond portfolios
    """
    
    def __init__(self):
        self.return_data = {}
        self.risk_factors = {}
        self.correlation_matrices = {}
        
    def calculate_var(
        self,
        portfolio_returns: np.ndarray,
        method: VaRMethod = VaRMethod.HISTORICAL_SIMULATION,
        confidence_level: float = 0.95,
        time_horizon: int = 1,
        portfolio_value: float = 1000000
    ) -> VaRResult:
        """Calculate VaR using specified method"""
        
        if method == VaRMethod.PARAMETRIC:
            var_value = self._calculate_parametric_var(
                portfolio_returns, confidence_level, time_horizon
            )
        elif method == VaRMethod.HISTORICAL_SIMULATION:
            var_value = self._calculate_historical_var(
                portfolio_returns, confidence_level, time_horizon
            )
        elif method == VaRMethod.MONTE_CARLO:
            var_value = self._calculate_monte_carlo_var(
                portfolio_returns, confidence_level, time_horizon
            )
        elif method == VaRMethod.CORNISH_FISHER:
            var_value = self._calculate_cornish_fisher_var(
                portfolio_returns, confidence_level, time_horizon
            )
        elif method == VaRMethod.EXTREME_VALUE_THEORY:
            var_value = self._calculate_evt_var(
                portfolio_returns, confidence_level, time_horizon
            )
        else:
            raise ValueError(f"Unsupported VaR method: {method}")
        
        # Calculate Expected Shortfall
        expected_shortfall = self._calculate_expected_shortfall(
            portfolio_returns, confidence_level, time_horizon, method
        )
        
        # Convert to absolute terms
        var_absolute = abs(var_value * portfolio_value)
        var_relative = abs(var_value * 100)
        
        return VaRResult(
            method=method,
            confidence_level=confidence_level,
            time_horizon=time_horizon,
            var_absolute=var_absolute,
            var_relative=var_relative,
            expected_shortfall=abs(expected_shortfall * portfolio_value),
            portfolio_value=portfolio_value,
            calculation_date=datetime.now()
        )
    
    def calculate_component_var(
        self,
        portfolio_weights: np.ndarray,
        asset_returns: np.ndarray,
        confidence_level: float = 0.95,
        method: VaRMethod = VaRMethod.PARAMETRIC
    ) -> List[ComponentVaR]:
        """Calculate component VaR for portfolio decomposition"""
        
        n_assets = len(portfolio_weights)
        component_vars = []
        
        # Calculate portfolio returns
        portfolio_returns = np.dot(asset_returns, portfolio_weights)
        
        # Calculate portfolio VaR
        portfolio_var = self.calculate_var(
            portfolio_returns, method, confidence_level
        ).var_relative / 100
        
        # Calculate covariance matrix
        cov_matrix = np.cov(asset_returns.T)
        
        # Calculate marginal VaR for each asset
        for i in range(n_assets):
            # Marginal VaR calculation
            marginal_var = self._calculate_marginal_var(
                portfolio_weights, cov_matrix, confidence_level, i
            )
            
            # Component VaR
            component_var = portfolio_weights[i] * marginal_var
            
            # Standalone VaR
            standalone_var = self._calculate_standalone_var(
                asset_returns[:, i], confidence_level, method
            )
            
            # Contribution percentage
            contribution_pct = component_var / portfolio_var * 100
            
            component_vars.append(ComponentVaR(
                asset_id=f"Asset_{i+1}",
                component_var=component_var,
                marginal_var=marginal_var,
                contribution_percentage=contribution_pct,
                standalone_var=standalone_var
            ))
        
        return component_vars
    
    def backtest_var(
        self,
        portfolio_returns: np.ndarray,
        var_forecasts: np.ndarray,
        confidence_level: float = 0.95,
        method: VaRMethod = VaRMethod.HISTORICAL_SIMULATION
    ) -> BacktestResult:
        """Backtest VaR model performance"""
        
        # Count violations
        violations = np.sum(portfolio_returns < -var_forecasts)
        total_observations = len(portfolio_returns)
        violation_rate = violations / total_observations
        expected_violations = int(total_observations * (1 - confidence_level))
        
        # Kupiec test (unconditional coverage)
        kupiec_stat, kupiec_p = self._kupiec_test(
            violations, total_observations, confidence_level
        )
        
        # Christoffersen test (conditional coverage)
        christoffersen_stat, christoffersen_p = self._christoffersen_test(
            portfolio_returns, var_forecasts, confidence_level
        )
        
        return BacktestResult(
            method=method,
            confidence_level=confidence_level,
            total_observations=total_observations,
            violations=violations,
            violation_rate=violation_rate,
            expected_violations=expected_violations,
            kupiec_test_statistic=kupiec_stat,
            kupiec_p_value=kupiec_p,
            christoffersen_test_statistic=christoffersen_stat,
            christoffersen_p_value=christoffersen_p
        )
    
    def decompose_risk(
        self,
        portfolio_returns: np.ndarray,
        risk_factors: Dict[str, np.ndarray],
        confidence_level: float = 0.95
    ) -> RiskDecomposition:
        """Decompose portfolio risk into systematic and idiosyncratic components"""
        
        # Calculate total VaR
        total_var = self.calculate_var(
            portfolio_returns, VaRMethod.PARAMETRIC, confidence_level
        ).var_relative / 100
        
        # Factor model regression
        factor_matrix = np.column_stack(list(risk_factors.values()))
        
        # Regression: returns = alpha + beta * factors + epsilon
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(factor_matrix, portfolio_returns)
        
        # Systematic component (explained by factors)
        systematic_returns = model.predict(factor_matrix)
        systematic_var = np.percentile(systematic_returns, (1 - confidence_level) * 100)
        
        # Idiosyncratic component (residuals)
        residuals = portfolio_returns - systematic_returns
        idiosyncratic_var = np.percentile(residuals, (1 - confidence_level) * 100)
        
        # Factor-specific VaRs
        currency_var = 0
        interest_rate_var = 0
        credit_var = 0
        
        if 'currency' in risk_factors:
            currency_returns = model.coef_[0] * risk_factors['currency']
            currency_var = np.percentile(currency_returns, (1 - confidence_level) * 100)
        
        if 'interest_rate' in risk_factors:
            ir_returns = model.coef_[1] * risk_factors['interest_rate']
            interest_rate_var = np.percentile(ir_returns, (1 - confidence_level) * 100)
        
        if 'credit' in risk_factors:
            credit_returns = model.coef_[2] * risk_factors['credit']
            credit_var = np.percentile(credit_returns, (1 - confidence_level) * 100)
        
        # Diversification benefit
        sum_component_vars = abs(systematic_var) + abs(idiosyncratic_var)
        diversification_benefit = sum_component_vars - abs(total_var)
        
        return RiskDecomposition(
            total_var=total_var,
            systematic_var=systematic_var,
            idiosyncratic_var=idiosyncratic_var,
            currency_var=currency_var,
            interest_rate_var=interest_rate_var,
            credit_var=credit_var,
            diversification_benefit=diversification_benefit
        )
    
    def _calculate_parametric_var(
        self,
        returns: np.ndarray,
        confidence_level: float,
        time_horizon: int,
        distribution: DistributionType = DistributionType.NORMAL
    ) -> float:
        """Calculate parametric VaR"""
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Adjust for time horizon
        horizon_mean = mean_return * time_horizon
        horizon_std = std_return * np.sqrt(time_horizon)
        
        if distribution == DistributionType.NORMAL:
            z_score = stats.norm.ppf(1 - confidence_level)
            var_value = horizon_mean + z_score * horizon_std
            
        elif distribution == DistributionType.T_DISTRIBUTION:
            # Fit t-distribution
            df, loc, scale = stats.t.fit(returns)
            t_score = stats.t.ppf(1 - confidence_level, df)
            var_value = horizon_mean + t_score * horizon_std
            
        else:
            # Default to normal
            z_score = stats.norm.ppf(1 - confidence_level)
            var_value = horizon_mean + z_score * horizon_std
        
        return var_value
    
    def _calculate_historical_var(
        self,
        returns: np.ndarray,
        confidence_level: float,
        time_horizon: int
    ) -> float:
        """Calculate historical simulation VaR"""
        
        # Adjust returns for time horizon
        if time_horizon > 1:
            # Overlapping periods method
            horizon_returns = []
            for i in range(len(returns) - time_horizon + 1):
                horizon_return = np.sum(returns[i:i+time_horizon])
                horizon_returns.append(horizon_return)
            returns = np.array(horizon_returns)
        
        # Calculate VaR as percentile
        var_value = np.percentile(returns, (1 - confidence_level) * 100)
        
        return var_value
    
    def _calculate_monte_carlo_var(
        self,
        returns: np.ndarray,
        confidence_level: float,
        time_horizon: int,
        n_simulations: int = 10000
    ) -> float:
        """Calculate Monte Carlo VaR"""
        
        # Fit distribution to returns
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Generate random scenarios
        np.random.seed(42)  # For reproducibility
        simulated_returns = np.random.normal(
            mean_return, std_return, (n_simulations, time_horizon)
        )
        
        # Calculate cumulative returns for each scenario
        cumulative_returns = np.sum(simulated_returns, axis=1)
        
        # Calculate VaR
        var_value = np.percentile(cumulative_returns, (1 - confidence_level) * 100)
        
        return var_value
    
    def _calculate_cornish_fisher_var(
        self,
        returns: np.ndarray,
        confidence_level: float,
        time_horizon: int
    ) -> float:
        """Calculate Cornish-Fisher VaR (accounts for skewness and kurtosis)"""
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns, fisher=True)  # Excess kurtosis
        
        # Adjust for time horizon
        horizon_mean = mean_return * time_horizon
        horizon_std = std_return * np.sqrt(time_horizon)
        
        # Standard normal quantile
        z = stats.norm.ppf(1 - confidence_level)
        
        # Cornish-Fisher adjustment
        z_cf = (z + 
                (z**2 - 1) * skewness / 6 +
                (z**3 - 3*z) * kurtosis / 24 -
                (2*z**3 - 5*z) * skewness**2 / 36)
        
        var_value = horizon_mean + z_cf * horizon_std
        
        return var_value
    
    def _calculate_evt_var(
        self,
        returns: np.ndarray,
        confidence_level: float,
        time_horizon: int,
        threshold_percentile: float = 0.1
    ) -> float:
        """Calculate VaR using Extreme Value Theory"""
        
        # Define threshold (e.g., 10th percentile)
        threshold = np.percentile(returns, threshold_percentile * 100)
        
        # Extract exceedances
        exceedances = returns[returns < threshold] - threshold
        
        if len(exceedances) < 10:  # Need sufficient data
            # Fall back to historical simulation
            return self._calculate_historical_var(returns, confidence_level, time_horizon)
        
        # Fit Generalized Pareto Distribution to exceedances
        try:
            shape, loc, scale = stats.genpareto.fit(-exceedances, floc=0)
            
            # Calculate VaR using GPD
            n = len(returns)
            n_exceedances = len(exceedances)
            
            # Probability of exceedance
            prob_exceedance = n_exceedances / n
            
            # VaR calculation
            if shape != 0:
                var_exceedance = (scale / shape) * (
                    ((n / n_exceedances) * (1 - confidence_level))**(-shape) - 1
                )
            else:
                var_exceedance = scale * np.log(
                    (n / n_exceedances) * (1 - confidence_level)
                )
            
            var_value = threshold - var_exceedance
            
            # Adjust for time horizon
            var_value *= np.sqrt(time_horizon)
            
        except:
            # Fall back to historical simulation if fitting fails
            var_value = self._calculate_historical_var(returns, confidence_level, time_horizon)
        
        return var_value
    
    def _calculate_expected_shortfall(
        self,
        returns: np.ndarray,
        confidence_level: float,
        time_horizon: int,
        method: VaRMethod
    ) -> float:
        """Calculate Expected Shortfall (Conditional VaR)"""
        
        # Calculate VaR first
        var_value = self._calculate_historical_var(returns, confidence_level, time_horizon)
        
        # Expected Shortfall is the mean of returns below VaR
        tail_returns = returns[returns <= var_value]
        
        if len(tail_returns) > 0:
            expected_shortfall = np.mean(tail_returns)
        else:
            expected_shortfall = var_value
        
        return expected_shortfall
    
    def _calculate_marginal_var(
        self,
        weights: np.ndarray,
        cov_matrix: np.ndarray,
        confidence_level: float,
        asset_index: int
    ) -> float:
        """Calculate marginal VaR for specific asset"""
        
        # Portfolio variance
        portfolio_var = np.dot(weights, np.dot(cov_matrix, weights))
        portfolio_vol = np.sqrt(portfolio_var)
        
        # Marginal contribution to portfolio volatility
        marginal_vol = np.dot(cov_matrix[asset_index], weights) / portfolio_vol
        
        # Convert to VaR using normal distribution assumption
        z_score = stats.norm.ppf(1 - confidence_level)
        marginal_var = -z_score * marginal_vol
        
        return marginal_var
    
    def _calculate_standalone_var(
        self,
        asset_returns: np.ndarray,
        confidence_level: float,
        method: VaRMethod
    ) -> float:
        """Calculate standalone VaR for individual asset"""
        
        return self.calculate_var(
            asset_returns, method, confidence_level
        ).var_relative / 100
    
    def _kupiec_test(
        self,
        violations: int,
        total_observations: int,
        confidence_level: float
    ) -> Tuple[float, float]:
        """Kupiec test for unconditional coverage"""
        
        expected_rate = 1 - confidence_level
        observed_rate = violations / total_observations
        
        if violations == 0 or violations == total_observations:
            return 0, 1  # Avoid log(0)
        
        # Likelihood ratio test statistic
        lr_stat = -2 * (
            total_observations * np.log(expected_rate) +
            violations * np.log(observed_rate / expected_rate) +
            (total_observations - violations) * np.log(
                (1 - observed_rate) / (1 - expected_rate)
            )
        )
        
        # P-value from chi-square distribution with 1 degree of freedom
        p_value = 1 - stats.chi2.cdf(lr_stat, df=1)
        
        return lr_stat, p_value
    
    def _christoffersen_test(
        self,
        returns: np.ndarray,
        var_forecasts: np.ndarray,
        confidence_level: float
    ) -> Tuple[float, float]:
        """Christoffersen test for conditional coverage"""
        
        # Create violation indicator
        violations = (returns < -var_forecasts).astype(int)
        
        # Count transitions
        n00 = n01 = n10 = n11 = 0
        
        for i in range(1, len(violations)):
            if violations[i-1] == 0 and violations[i] == 0:
                n00 += 1
            elif violations[i-1] == 0 and violations[i] == 1:
                n01 += 1
            elif violations[i-1] == 1 and violations[i] == 0:
                n10 += 1
            elif violations[i-1] == 1 and violations[i] == 1:
                n11 += 1
        
        # Avoid division by zero
        if n00 + n01 == 0 or n10 + n11 == 0:
            return 0, 1
        
        # Transition probabilities
        pi_01 = n01 / (n00 + n01)
        pi_11 = n11 / (n10 + n11)
        pi = (n01 + n11) / (n00 + n01 + n10 + n11)
        
        # Likelihood ratio test statistic
        if pi_01 > 0 and pi_11 > 0 and pi > 0:
            lr_stat = -2 * (
                (n00 + n10) * np.log(1 - pi) +
                (n01 + n11) * np.log(pi) -
                n00 * np.log(1 - pi_01) -
                n01 * np.log(pi_01) -
                n10 * np.log(1 - pi_11) -
                n11 * np.log(pi_11)
            )
        else:
            lr_stat = 0
        
        # P-value from chi-square distribution with 1 degree of freedom
        p_value = 1 - stats.chi2.cdf(lr_stat, df=1)
        
        return lr_stat, p_value