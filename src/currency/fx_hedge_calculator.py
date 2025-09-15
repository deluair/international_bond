"""
Foreign exchange hedging calculator and risk management.
"""

import numpy as np
from datetime import date, datetime, timedelta
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import math
from scipy import optimize, stats

from ..models.currency import CurrencyPair, CurrencyType


class HedgeType(Enum):
    """Types of FX hedging strategies."""
    FULL_HEDGE = "full_hedge"
    PARTIAL_HEDGE = "partial_hedge"
    DYNAMIC_HEDGE = "dynamic_hedge"
    MINIMUM_VARIANCE = "minimum_variance"
    RISK_PARITY = "risk_parity"
    NO_HEDGE = "no_hedge"


class HedgeInstrument(Enum):
    """FX hedging instruments."""
    FORWARD = "forward"
    SWAP = "swap"
    OPTION = "option"
    FUTURE = "future"
    NDF = "ndf"  # Non-deliverable forward


@dataclass
class HedgeRatio:
    """FX hedge ratio calculation result."""
    currency_pair: str
    hedge_ratio: float
    hedge_type: HedgeType
    hedge_instrument: HedgeInstrument
    effectiveness: float
    tracking_error: float
    var_reduction: float
    confidence_level: float
    calculation_date: date
    
    def __str__(self) -> str:
        return (f"Hedge Ratio for {self.currency_pair}: {self.hedge_ratio:.4f}\n"
                f"Type: {self.hedge_type.value}\n"
                f"Instrument: {self.hedge_instrument.value}\n"
                f"Effectiveness: {self.effectiveness:.2%}\n"
                f"Tracking Error: {self.tracking_error:.4f}\n"
                f"VaR Reduction: {self.var_reduction:.2%}")


@dataclass
class HedgeEffectiveness:
    """Hedge effectiveness measurement."""
    regression_r_squared: float
    dollar_offset_ratio: float
    variance_reduction_ratio: float
    correlation: float
    beta: float
    alpha: float
    tracking_error: float
    information_ratio: float
    
    @property
    def is_highly_effective(self) -> bool:
        """Check if hedge meets high effectiveness criteria (80-125% rule)."""
        return 0.80 <= abs(self.dollar_offset_ratio) <= 1.25


@dataclass
class FXRiskMetrics:
    """FX risk measurement metrics."""
    currency_exposure: Dict[str, float]
    portfolio_fx_var: float
    portfolio_fx_cvar: float
    fx_volatility: float
    correlation_matrix: np.ndarray
    marginal_var: Dict[str, float]
    component_var: Dict[str, float]
    diversification_ratio: float
    
    def __str__(self) -> str:
        return (f"FX Risk Metrics:\n"
                f"Portfolio FX VaR (95%): {self.portfolio_fx_var:.2%}\n"
                f"Portfolio FX CVaR (95%): {self.portfolio_fx_cvar:.2%}\n"
                f"FX Volatility: {self.fx_volatility:.2%}\n"
                f"Diversification Ratio: {self.diversification_ratio:.4f}")


class FXHedgeCalculator:
    """
    Comprehensive FX hedging calculator and risk manager.
    """
    
    def __init__(self, base_currency: CurrencyType = CurrencyType.MAJOR):
        """
        Initialize FX hedge calculator.
        
        Args:
            base_currency: Base currency for calculations
        """
        self.base_currency = base_currency
        self.fx_data_cache = {}
        self.correlation_cache = {}
    
    def calculate_hedge_ratio(self, 
                            currency_pair: CurrencyPair,
                            portfolio_exposure: float,
                            hedge_type: HedgeType = HedgeType.MINIMUM_VARIANCE,
                            lookback_days: int = 252,
                            confidence_level: float = 0.95) -> HedgeRatio:
        """
        Calculate optimal hedge ratio for currency exposure.
        
        Args:
            currency_pair: Currency pair to hedge
            portfolio_exposure: Portfolio exposure amount
            hedge_type: Type of hedging strategy
            lookback_days: Historical data lookback period
            confidence_level: Confidence level for risk calculations
            
        Returns:
            HedgeRatio object
        """
        if hedge_type == HedgeType.FULL_HEDGE:
            hedge_ratio = 1.0
            effectiveness = 1.0
            tracking_error = 0.0
            var_reduction = 1.0
            
        elif hedge_type == HedgeType.NO_HEDGE:
            hedge_ratio = 0.0
            effectiveness = 0.0
            tracking_error = currency_pair.volatility or 0.15
            var_reduction = 0.0
            
        elif hedge_type == HedgeType.MINIMUM_VARIANCE:
            hedge_ratio, effectiveness, tracking_error, var_reduction = self._calculate_minimum_variance_hedge(
                currency_pair, lookback_days, confidence_level
            )
            
        elif hedge_type == HedgeType.PARTIAL_HEDGE:
            # Default to 50% hedge
            hedge_ratio = 0.5
            effectiveness = 0.7  # Approximate
            tracking_error = (currency_pair.volatility or 0.15) * 0.5
            var_reduction = 0.5
            
        elif hedge_type == HedgeType.DYNAMIC_HEDGE:
            hedge_ratio, effectiveness, tracking_error, var_reduction = self._calculate_dynamic_hedge(
                currency_pair, lookback_days, confidence_level
            )
            
        elif hedge_type == HedgeType.RISK_PARITY:
            hedge_ratio, effectiveness, tracking_error, var_reduction = self._calculate_risk_parity_hedge(
                currency_pair, portfolio_exposure, confidence_level
            )
            
        else:
            raise ValueError(f"Unsupported hedge type: {hedge_type}")
        
        return HedgeRatio(
            currency_pair=f"{currency_pair.base_currency.value}/{currency_pair.quote_currency.value}",
            hedge_ratio=hedge_ratio,
            hedge_type=hedge_type,
            hedge_instrument=HedgeInstrument.FORWARD,  # Default
            effectiveness=effectiveness,
            tracking_error=tracking_error,
            var_reduction=var_reduction,
            confidence_level=confidence_level,
            calculation_date=date.today()
        )
    
    def calculate_hedge_effectiveness(self, 
                                   hedged_returns: np.ndarray,
                                   unhedged_returns: np.ndarray,
                                   hedge_returns: np.ndarray) -> HedgeEffectiveness:
        """
        Calculate hedge effectiveness metrics.
        
        Args:
            hedged_returns: Hedged portfolio returns
            unhedged_returns: Unhedged portfolio returns  
            hedge_returns: Hedge instrument returns
            
        Returns:
            HedgeEffectiveness object
        """
        # Regression analysis
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            hedge_returns, unhedged_returns - hedged_returns
        )
        
        regression_r_squared = r_value ** 2
        beta = slope
        alpha = intercept
        
        # Dollar offset ratio
        hedge_pnl = np.sum(hedge_returns)
        exposure_pnl = np.sum(unhedged_returns - hedged_returns)
        
        if abs(exposure_pnl) > 1e-10:
            dollar_offset_ratio = hedge_pnl / exposure_pnl
        else:
            dollar_offset_ratio = 0.0
        
        # Variance reduction ratio
        unhedged_var = np.var(unhedged_returns)
        hedged_var = np.var(hedged_returns)
        
        if unhedged_var > 0:
            variance_reduction_ratio = 1.0 - (hedged_var / unhedged_var)
        else:
            variance_reduction_ratio = 0.0
        
        # Correlation
        correlation = np.corrcoef(hedge_returns, unhedged_returns - hedged_returns)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0
        
        # Tracking error
        tracking_error = np.std(hedged_returns - unhedged_returns)
        
        # Information ratio
        excess_return = np.mean(hedged_returns - unhedged_returns)
        information_ratio = excess_return / tracking_error if tracking_error > 0 else 0.0
        
        return HedgeEffectiveness(
            regression_r_squared=regression_r_squared,
            dollar_offset_ratio=dollar_offset_ratio,
            variance_reduction_ratio=variance_reduction_ratio,
            correlation=correlation,
            beta=beta,
            alpha=alpha,
            tracking_error=tracking_error,
            information_ratio=information_ratio
        )
    
    def calculate_fx_risk_metrics(self, 
                                currency_exposures: Dict[str, float],
                                fx_volatilities: Dict[str, float],
                                fx_correlations: np.ndarray,
                                confidence_level: float = 0.95) -> FXRiskMetrics:
        """
        Calculate comprehensive FX risk metrics for portfolio.
        
        Args:
            currency_exposures: Dictionary of currency exposures
            fx_volatilities: Dictionary of FX volatilities
            fx_correlations: FX correlation matrix
            confidence_level: Confidence level for VaR calculations
            
        Returns:
            FXRiskMetrics object
        """
        currencies = list(currency_exposures.keys())
        exposures = np.array([currency_exposures[curr] for curr in currencies])
        volatilities = np.array([fx_volatilities[curr] for curr in currencies])
        
        # Portfolio FX variance
        weighted_vols = exposures * volatilities
        portfolio_variance = np.dot(weighted_vols, np.dot(fx_correlations, weighted_vols))
        portfolio_volatility = math.sqrt(max(0, portfolio_variance))
        
        # VaR and CVaR calculations
        z_score = stats.norm.ppf(confidence_level)
        portfolio_fx_var = portfolio_volatility * z_score
        
        # CVaR (Expected Shortfall)
        portfolio_fx_cvar = portfolio_volatility * stats.norm.pdf(z_score) / (1 - confidence_level)
        
        # Marginal VaR
        marginal_var = {}
        for i, curr in enumerate(currencies):
            marginal_contribution = (weighted_vols[i] * np.dot(fx_correlations[i], weighted_vols)) / portfolio_volatility
            marginal_var[curr] = marginal_contribution * z_score
        
        # Component VaR
        component_var = {}
        for i, curr in enumerate(currencies):
            component_var[curr] = exposures[i] * marginal_var[curr]
        
        # Diversification ratio
        undiversified_risk = np.sum(np.abs(exposures) * volatilities)
        diversification_ratio = portfolio_volatility / undiversified_risk if undiversified_risk > 0 else 1.0
        
        return FXRiskMetrics(
            currency_exposure=currency_exposures,
            portfolio_fx_var=portfolio_fx_var,
            portfolio_fx_cvar=portfolio_fx_cvar,
            fx_volatility=portfolio_volatility,
            correlation_matrix=fx_correlations,
            marginal_var=marginal_var,
            component_var=component_var,
            diversification_ratio=diversification_ratio
        )
    
    def optimize_hedge_portfolio(self, 
                               currency_exposures: Dict[str, float],
                               available_instruments: List[HedgeInstrument],
                               target_hedge_ratio: float = 0.8,
                               cost_constraints: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Optimize hedge portfolio across multiple currencies.
        
        Args:
            currency_exposures: Dictionary of currency exposures
            available_instruments: List of available hedge instruments
            target_hedge_ratio: Target overall hedge ratio
            cost_constraints: Dictionary of hedging costs by currency
            
        Returns:
            Dictionary of optimal hedge amounts by currency
        """
        currencies = list(currency_exposures.keys())
        n_currencies = len(currencies)
        
        if n_currencies == 0:
            return {}
        
        # Objective function: minimize tracking error while achieving target hedge ratio
        def objective(hedge_amounts):
            hedge_amounts = np.array(hedge_amounts)
            exposures = np.array([currency_exposures[curr] for curr in currencies])
            
            # Calculate overall hedge ratio
            total_exposure = np.sum(np.abs(exposures))
            total_hedged = np.sum(np.abs(hedge_amounts))
            
            if total_exposure > 0:
                overall_hedge_ratio = total_hedged / total_exposure
            else:
                overall_hedge_ratio = 0.0
            
            # Penalty for deviating from target hedge ratio
            hedge_ratio_penalty = 1000 * (overall_hedge_ratio - target_hedge_ratio) ** 2
            
            # Tracking error penalty (simplified)
            tracking_error_penalty = np.sum((hedge_amounts - target_hedge_ratio * exposures) ** 2)
            
            # Cost penalty
            cost_penalty = 0.0
            if cost_constraints:
                for i, curr in enumerate(currencies):
                    cost = cost_constraints.get(curr, 0.01)
                    cost_penalty += cost * abs(hedge_amounts[i])
            
            return hedge_ratio_penalty + tracking_error_penalty + cost_penalty
        
        # Constraints
        constraints = []
        
        # Bounds (can't hedge more than 150% of exposure)
        bounds = []
        for curr in currencies:
            exposure = abs(currency_exposures[curr])
            bounds.append((-1.5 * exposure, 1.5 * exposure))
        
        # Initial guess
        initial_guess = [target_hedge_ratio * currency_exposures[curr] for curr in currencies]
        
        # Optimize
        try:
            result = optimize.minimize(
                objective, 
                initial_guess, 
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if result.success:
                optimal_hedges = result.x
            else:
                optimal_hedges = initial_guess
                
        except Exception:
            optimal_hedges = initial_guess
        
        return {curr: hedge for curr, hedge in zip(currencies, optimal_hedges)}
    
    def _calculate_minimum_variance_hedge(self, 
                                        currency_pair: CurrencyPair,
                                        lookback_days: int,
                                        confidence_level: float) -> Tuple[float, float, float, float]:
        """Calculate minimum variance hedge ratio."""
        # Simulate historical data (in practice, would use real data)
        np.random.seed(42)  # For reproducibility
        
        # Generate correlated returns
        spot_vol = currency_pair.volatility or 0.15
        forward_vol = spot_vol * 0.95  # Forward typically less volatile
        correlation = 0.85  # Typical spot-forward correlation
        
        returns = np.random.multivariate_normal(
            [0, 0], 
            [[spot_vol**2, correlation * spot_vol * forward_vol],
             [correlation * spot_vol * forward_vol, forward_vol**2]],
            lookback_days
        )
        
        spot_returns = returns[:, 0]
        forward_returns = returns[:, 1]
        
        # Minimum variance hedge ratio
        covariance = np.cov(spot_returns, forward_returns)[0, 1]
        forward_variance = np.var(forward_returns)
        
        if forward_variance > 0:
            hedge_ratio = covariance / forward_variance
        else:
            hedge_ratio = 0.0
        
        # Hedge effectiveness
        spot_variance = np.var(spot_returns)
        hedged_variance = spot_variance + hedge_ratio**2 * forward_variance - 2 * hedge_ratio * covariance
        
        if spot_variance > 0:
            effectiveness = 1.0 - (hedged_variance / spot_variance)
            var_reduction = effectiveness
        else:
            effectiveness = 0.0
            var_reduction = 0.0
        
        tracking_error = math.sqrt(max(0, hedged_variance))
        
        return hedge_ratio, max(0, effectiveness), tracking_error, max(0, var_reduction)
    
    def _calculate_dynamic_hedge(self, 
                               currency_pair: CurrencyPair,
                               lookback_days: int,
                               confidence_level: float) -> Tuple[float, float, float, float]:
        """Calculate dynamic hedge ratio based on volatility regime."""
        # Simplified dynamic hedging - in practice would use more sophisticated models
        current_vol = currency_pair.volatility or 0.15
        long_term_vol = 0.12  # Assumed long-term average
        
        # Adjust hedge ratio based on volatility regime
        vol_ratio = current_vol / long_term_vol
        
        if vol_ratio > 1.5:  # High volatility regime
            base_hedge_ratio = 0.9
        elif vol_ratio < 0.7:  # Low volatility regime
            base_hedge_ratio = 0.5
        else:  # Normal regime
            base_hedge_ratio = 0.7
        
        # Use minimum variance as baseline
        mv_hedge_ratio, mv_effectiveness, mv_tracking_error, mv_var_reduction = self._calculate_minimum_variance_hedge(
            currency_pair, lookback_days, confidence_level
        )
        
        # Blend with regime-based adjustment
        hedge_ratio = 0.7 * mv_hedge_ratio + 0.3 * base_hedge_ratio
        effectiveness = mv_effectiveness * 0.9  # Slightly lower due to dynamic adjustment
        tracking_error = mv_tracking_error * 1.1  # Slightly higher
        var_reduction = mv_var_reduction * 0.9
        
        return hedge_ratio, effectiveness, tracking_error, var_reduction
    
    def _calculate_risk_parity_hedge(self, 
                                   currency_pair: CurrencyPair,
                                   portfolio_exposure: float,
                                   confidence_level: float) -> Tuple[float, float, float, float]:
        """Calculate risk parity hedge ratio."""
        # Risk parity approach: hedge to equalize risk contributions
        spot_vol = currency_pair.volatility or 0.15
        
        # Assume portfolio has other risk factors
        portfolio_vol = 0.08  # Assumed portfolio volatility
        fx_risk_contribution = (abs(portfolio_exposure) * spot_vol) / portfolio_vol
        
        # Target FX risk contribution (e.g., 20% of total risk)
        target_fx_contribution = 0.20
        
        if fx_risk_contribution > target_fx_contribution:
            # Need to hedge to reduce FX risk
            hedge_ratio = 1.0 - (target_fx_contribution / fx_risk_contribution)
        else:
            # FX risk already at target level
            hedge_ratio = 0.0
        
        # Clip hedge ratio
        hedge_ratio = max(0.0, min(1.0, hedge_ratio))
        
        # Estimate effectiveness
        effectiveness = hedge_ratio * 0.8  # Approximate
        tracking_error = spot_vol * (1 - hedge_ratio)
        var_reduction = hedge_ratio * 0.7  # Approximate
        
        return hedge_ratio, effectiveness, tracking_error, var_reduction


class FXHedgeOptimizer:
    """Advanced FX hedge optimization engine."""
    
    def __init__(self, hedge_calculator: FXHedgeCalculator):
        """Initialize with hedge calculator."""
        self.hedge_calculator = hedge_calculator
    
    def optimize_multi_currency_hedge(self, 
                                    exposures: Dict[str, float],
                                    correlations: np.ndarray,
                                    volatilities: Dict[str, float],
                                    hedge_costs: Dict[str, float],
                                    risk_budget: float = 0.05) -> Dict[str, float]:
        """
        Optimize hedge ratios across multiple currencies.
        
        Args:
            exposures: Currency exposures
            correlations: FX correlation matrix
            volatilities: FX volatilities
            hedge_costs: Hedging costs by currency
            risk_budget: Maximum acceptable FX risk
            
        Returns:
            Optimal hedge ratios by currency
        """
        currencies = list(exposures.keys())
        n = len(currencies)
        
        if n == 0:
            return {}
        
        # Set up optimization problem
        def objective(hedge_ratios):
            # Calculate portfolio risk after hedging
            hedged_exposures = np.array([
                exposures[curr] * (1 - hedge_ratios[i]) 
                for i, curr in enumerate(currencies)
            ])
            
            vols = np.array([volatilities[curr] for curr in currencies])
            weighted_vols = hedged_exposures * vols
            
            portfolio_variance = np.dot(weighted_vols, np.dot(correlations, weighted_vols))
            portfolio_risk = math.sqrt(max(0, portfolio_variance))
            
            # Calculate hedging costs
            total_cost = sum(
                hedge_costs.get(curr, 0.01) * abs(exposures[curr]) * hedge_ratios[i]
                for i, curr in enumerate(currencies)
            )
            
            # Objective: minimize cost subject to risk constraint
            # Use penalty method for risk constraint
            risk_penalty = 1000 * max(0, portfolio_risk - risk_budget) ** 2
            
            return total_cost + risk_penalty
        
        # Bounds: hedge ratios between 0 and 1
        bounds = [(0.0, 1.0) for _ in currencies]
        
        # Initial guess
        initial_guess = [0.5] * n
        
        # Optimize
        try:
            result = optimize.minimize(
                objective,
                initial_guess,
                method='L-BFGS-B',
                bounds=bounds
            )
            
            if result.success:
                optimal_ratios = result.x
            else:
                optimal_ratios = initial_guess
                
        except Exception:
            optimal_ratios = initial_guess
        
        return {curr: ratio for curr, ratio in zip(currencies, optimal_ratios)}