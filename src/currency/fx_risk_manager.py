"""
FX risk management module for comprehensive currency risk monitoring and control.
"""

import numpy as np
from datetime import date, datetime, timedelta
from typing import List, Dict, Tuple, Optional, Union, Set
from dataclasses import dataclass, field
from enum import Enum
import math
from scipy import stats

from ..models.currency import CurrencyPair, CurrencyType
from ..models.portfolio import Portfolio, Position


class RiskMetric(Enum):
    """FX risk metrics."""
    VAR_95 = "var_95"
    VAR_99 = "var_99"
    EXPECTED_SHORTFALL = "expected_shortfall"
    MAXIMUM_DRAWDOWN = "maximum_drawdown"
    VOLATILITY = "volatility"
    BETA = "beta"
    CORRELATION = "correlation"
    TRACKING_ERROR = "tracking_error"


class RiskLimit(Enum):
    """Types of risk limits."""
    ABSOLUTE_VAR = "absolute_var"
    RELATIVE_VAR = "relative_var"
    NOTIONAL_EXPOSURE = "notional_exposure"
    CONCENTRATION = "concentration"
    LEVERAGE = "leverage"
    DRAWDOWN = "drawdown"
    VOLATILITY_LIMIT = "volatility_limit"


@dataclass
class FXRiskLimit:
    """FX risk limit definition."""
    limit_type: RiskLimit
    currency_pair: Optional[str] = None
    limit_value: float = 0.0
    warning_threshold: float = 0.8  # Warning at 80% of limit
    breach_action: str = "alert"  # alert, reduce_position, hedge
    
    def is_breached(self, current_value: float) -> bool:
        """Check if limit is breached."""
        return abs(current_value) > self.limit_value
    
    def is_warning(self, current_value: float) -> bool:
        """Check if warning threshold is reached."""
        return abs(current_value) > (self.limit_value * self.warning_threshold)


@dataclass
class FXRiskMetrics:
    """FX risk metrics for a portfolio or position."""
    currency_exposure: Dict[str, float]
    total_fx_var_95: float
    total_fx_var_99: float
    fx_expected_shortfall: float
    fx_volatility: float
    max_currency_exposure: float
    fx_concentration_ratio: float
    fx_leverage: float
    currency_correlations: Dict[Tuple[str, str], float]
    
    def __str__(self) -> str:
        return (f"FX Risk Metrics:\n"
                f"Total FX VaR (95%): {self.total_fx_var_95:.2%}\n"
                f"Total FX VaR (99%): {self.total_fx_var_99:.2%}\n"
                f"FX Expected Shortfall: {self.fx_expected_shortfall:.2%}\n"
                f"FX Volatility: {self.fx_volatility:.2%}\n"
                f"Max Currency Exposure: {self.max_currency_exposure:.2%}\n"
                f"FX Concentration Ratio: {self.fx_concentration_ratio:.2f}\n"
                f"FX Leverage: {self.fx_leverage:.2f}")


@dataclass
class RiskAlert:
    """Risk alert notification."""
    alert_type: str
    severity: str  # low, medium, high, critical
    currency_pair: Optional[str]
    limit_type: RiskLimit
    current_value: float
    limit_value: float
    breach_percentage: float
    timestamp: datetime
    message: str
    recommended_action: str


@dataclass
class FXStressTest:
    """FX stress test scenario."""
    scenario_name: str
    currency_shocks: Dict[str, float]  # Currency pair -> shock percentage
    correlation_adjustment: float = 1.0  # Correlation multiplier
    volatility_adjustment: float = 1.0  # Volatility multiplier
    
    def apply_shock(self, fx_rates: Dict[str, float]) -> Dict[str, float]:
        """Apply stress scenario to FX rates."""
        shocked_rates = fx_rates.copy()
        
        for pair, shock in self.currency_shocks.items():
            if pair in shocked_rates:
                shocked_rates[pair] *= (1 + shock)
        
        return shocked_rates


class FXRiskManager:
    """
    Comprehensive FX risk management system.
    """
    
    def __init__(self, 
                 base_currency: CurrencyType = CurrencyType.MAJOR,
                 confidence_levels: List[float] = None):
        """
        Initialize FX risk manager.
        
        Args:
            base_currency: Base currency for risk calculations
            confidence_levels: VaR confidence levels
        """
        self.base_currency = base_currency
        self.confidence_levels = confidence_levels or [0.95, 0.99]
        self.risk_limits: List[FXRiskLimit] = []
        self.alerts: List[RiskAlert] = []
        self.stress_scenarios: List[FXStressTest] = []
        
        # Initialize default stress scenarios
        self._initialize_default_scenarios()
    
    def add_risk_limit(self, risk_limit: FXRiskLimit) -> None:
        """Add a risk limit."""
        self.risk_limits.append(risk_limit)
    
    def calculate_fx_risk_metrics(self, 
                                portfolio: Portfolio,
                                fx_rates: Dict[str, float],
                                fx_volatilities: Dict[str, float],
                                correlation_matrix: Optional[np.ndarray] = None) -> FXRiskMetrics:
        """
        Calculate comprehensive FX risk metrics for portfolio.
        
        Args:
            portfolio: Portfolio to analyze
            fx_rates: Current FX rates
            fx_volatilities: FX volatilities
            correlation_matrix: Currency correlation matrix
            
        Returns:
            FXRiskMetrics object
        """
        # Calculate currency exposures
        currency_exposure = self._calculate_currency_exposure(portfolio, fx_rates)
        
        # Calculate VaR metrics
        var_95, var_99, expected_shortfall = self._calculate_fx_var(
            currency_exposure, fx_volatilities, correlation_matrix
        )
        
        # Calculate other risk metrics
        fx_volatility = self._calculate_portfolio_fx_volatility(
            currency_exposure, fx_volatilities, correlation_matrix
        )
        
        max_exposure = max(abs(exp) for exp in currency_exposure.values()) if currency_exposure else 0.0
        
        concentration_ratio = self._calculate_concentration_ratio(currency_exposure)
        
        fx_leverage = sum(abs(exp) for exp in currency_exposure.values())
        
        # Calculate correlations
        currency_correlations = self._extract_currency_correlations(
            list(currency_exposure.keys()), correlation_matrix
        )
        
        return FXRiskMetrics(
            currency_exposure=currency_exposure,
            total_fx_var_95=var_95,
            total_fx_var_99=var_99,
            fx_expected_shortfall=expected_shortfall,
            fx_volatility=fx_volatility,
            max_currency_exposure=max_exposure,
            fx_concentration_ratio=concentration_ratio,
            fx_leverage=fx_leverage,
            currency_correlations=currency_correlations
        )
    
    def monitor_risk_limits(self, 
                          risk_metrics: FXRiskMetrics,
                          portfolio_value: float) -> List[RiskAlert]:
        """
        Monitor risk limits and generate alerts.
        
        Args:
            risk_metrics: Current risk metrics
            portfolio_value: Total portfolio value
            
        Returns:
            List of risk alerts
        """
        alerts = []
        current_time = datetime.now()
        
        for limit in self.risk_limits:
            current_value = self._get_metric_value(limit, risk_metrics, portfolio_value)
            
            if limit.is_breached(current_value):
                severity = "critical"
                breach_pct = (abs(current_value) / limit.limit_value - 1.0) * 100
                
                alert = RiskAlert(
                    alert_type="limit_breach",
                    severity=severity,
                    currency_pair=limit.currency_pair,
                    limit_type=limit.limit_type,
                    current_value=current_value,
                    limit_value=limit.limit_value,
                    breach_percentage=breach_pct,
                    timestamp=current_time,
                    message=f"{limit.limit_type.value} limit breached by {breach_pct:.1f}%",
                    recommended_action=limit.breach_action
                )
                alerts.append(alert)
                
            elif limit.is_warning(current_value):
                severity = "medium"
                warning_pct = (abs(current_value) / limit.limit_value) * 100
                
                alert = RiskAlert(
                    alert_type="limit_warning",
                    severity=severity,
                    currency_pair=limit.currency_pair,
                    limit_type=limit.limit_type,
                    current_value=current_value,
                    limit_value=limit.limit_value,
                    breach_percentage=warning_pct,
                    timestamp=current_time,
                    message=f"{limit.limit_type.value} at {warning_pct:.1f}% of limit",
                    recommended_action="monitor_closely"
                )
                alerts.append(alert)
        
        self.alerts.extend(alerts)
        return alerts
    
    def run_fx_stress_tests(self, 
                          portfolio: Portfolio,
                          fx_rates: Dict[str, float],
                          fx_volatilities: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """
        Run FX stress tests on portfolio.
        
        Args:
            portfolio: Portfolio to stress test
            fx_rates: Current FX rates
            fx_volatilities: FX volatilities
            
        Returns:
            Dictionary of stress test results
        """
        stress_results = {}
        
        for scenario in self.stress_scenarios:
            # Apply stress scenario
            stressed_rates = scenario.apply_shock(fx_rates)
            
            # Adjust volatilities if needed
            stressed_vols = fx_volatilities.copy()
            if scenario.volatility_adjustment != 1.0:
                stressed_vols = {
                    pair: vol * scenario.volatility_adjustment 
                    for pair, vol in stressed_vols.items()
                }
            
            # Calculate stressed exposures
            stressed_exposure = self._calculate_currency_exposure(portfolio, stressed_rates)
            
            # Calculate P&L impact
            base_exposure = self._calculate_currency_exposure(portfolio, fx_rates)
            pnl_impact = {}
            
            for currency in set(list(base_exposure.keys()) + list(stressed_exposure.keys())):
                base_exp = base_exposure.get(currency, 0.0)
                stressed_exp = stressed_exposure.get(currency, 0.0)
                pnl_impact[currency] = stressed_exp - base_exp
            
            total_pnl_impact = sum(pnl_impact.values())
            
            stress_results[scenario.scenario_name] = {
                'total_pnl_impact': total_pnl_impact,
                'currency_pnl_impact': pnl_impact,
                'stressed_exposure': stressed_exposure
            }
        
        return stress_results
    
    def calculate_optimal_hedge_ratios(self, 
                                     portfolio: Portfolio,
                                     fx_rates: Dict[str, float],
                                     fx_volatilities: Dict[str, float],
                                     correlation_matrix: Optional[np.ndarray] = None,
                                     target_risk_reduction: float = 0.5) -> Dict[str, float]:
        """
        Calculate optimal hedge ratios for FX exposures.
        
        Args:
            portfolio: Portfolio to hedge
            fx_rates: Current FX rates
            fx_volatilities: FX volatilities
            correlation_matrix: Currency correlation matrix
            target_risk_reduction: Target risk reduction (0.5 = 50% reduction)
            
        Returns:
            Dictionary of optimal hedge ratios by currency
        """
        # Calculate current exposures
        exposures = self._calculate_currency_exposure(portfolio, fx_rates)
        
        if not exposures:
            return {}
        
        currencies = list(exposures.keys())
        n = len(currencies)
        
        if n == 0:
            return {}
        
        # Build covariance matrix
        if correlation_matrix is None:
            correlation_matrix = np.eye(n)
        
        volatility_vector = np.array([fx_volatilities.get(curr, 0.15) for curr in currencies])
        cov_matrix = np.outer(volatility_vector, volatility_vector) * correlation_matrix
        
        # Exposure vector
        exposure_vector = np.array([exposures[curr] for curr in currencies])
        
        # Calculate minimum variance hedge ratios
        if np.linalg.det(cov_matrix) != 0:
            inv_cov = np.linalg.inv(cov_matrix)
            
            # Minimum variance hedge ratios
            hedge_ratios = np.dot(inv_cov, exposure_vector) / np.dot(exposure_vector, np.dot(inv_cov, exposure_vector))
            
            # Scale by target risk reduction
            hedge_ratios *= target_risk_reduction
            
            # Convert back to dictionary
            hedge_dict = {}
            for i, currency in enumerate(currencies):
                hedge_dict[currency] = hedge_ratios[i]
            
            return hedge_dict
        else:
            # Fallback: proportional hedging
            total_exposure = sum(abs(exp) for exp in exposures.values())
            if total_exposure > 0:
                return {
                    curr: (exp / total_exposure) * target_risk_reduction 
                    for curr, exp in exposures.items()
                }
            else:
                return {}
    
    def generate_fx_risk_report(self, 
                              risk_metrics: FXRiskMetrics,
                              alerts: List[RiskAlert],
                              stress_results: Dict[str, Dict[str, float]]) -> str:
        """
        Generate comprehensive FX risk report.
        
        Args:
            risk_metrics: Current risk metrics
            alerts: Risk alerts
            stress_results: Stress test results
            
        Returns:
            Formatted risk report string
        """
        report = []
        report.append("=" * 60)
        report.append("FX RISK MANAGEMENT REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Risk Metrics Section
        report.append("CURRENT RISK METRICS")
        report.append("-" * 30)
        report.append(str(risk_metrics))
        report.append("")
        
        # Currency Exposures
        report.append("CURRENCY EXPOSURES")
        report.append("-" * 30)
        for currency, exposure in sorted(risk_metrics.currency_exposure.items()):
            report.append(f"{currency}: {exposure:,.0f} ({exposure/sum(abs(e) for e in risk_metrics.currency_exposure.values()):.1%})")
        report.append("")
        
        # Risk Alerts
        if alerts:
            report.append("RISK ALERTS")
            report.append("-" * 30)
            for alert in alerts:
                report.append(f"[{alert.severity.upper()}] {alert.message}")
                report.append(f"  Action: {alert.recommended_action}")
            report.append("")
        
        # Stress Test Results
        if stress_results:
            report.append("STRESS TEST RESULTS")
            report.append("-" * 30)
            for scenario, results in stress_results.items():
                total_impact = results['total_pnl_impact']
                report.append(f"{scenario}: {total_impact:,.0f} ({total_impact/1000000:.1f}M)")
            report.append("")
        
        # Risk Limits Status
        report.append("RISK LIMITS STATUS")
        report.append("-" * 30)
        for limit in self.risk_limits:
            status = "OK"
            if any(alert.limit_type == limit.limit_type for alert in alerts):
                status = "BREACH" if any(alert.alert_type == "limit_breach" and alert.limit_type == limit.limit_type for alert in alerts) else "WARNING"
            
            report.append(f"{limit.limit_type.value}: {status}")
        
        return "\n".join(report)
    
    def _calculate_currency_exposure(self, 
                                   portfolio: Portfolio,
                                   fx_rates: Dict[str, float]) -> Dict[str, float]:
        """Calculate currency exposure for portfolio."""
        exposures = {}
        
        for position in portfolio.positions:
            # Get position currency
            position_currency = getattr(position, 'currency', self.base_currency.value)
            
            if position_currency != self.base_currency.value:
                # Convert to base currency
                fx_rate = fx_rates.get(f"{position_currency}/{self.base_currency.value}", 1.0)
                exposure_value = position.market_value * fx_rate
                
                if position_currency in exposures:
                    exposures[position_currency] += exposure_value
                else:
                    exposures[position_currency] = exposure_value
        
        return exposures
    
    def _calculate_fx_var(self, 
                        currency_exposure: Dict[str, float],
                        fx_volatilities: Dict[str, float],
                        correlation_matrix: Optional[np.ndarray] = None) -> Tuple[float, float, float]:
        """Calculate FX VaR metrics."""
        if not currency_exposure:
            return 0.0, 0.0, 0.0
        
        currencies = list(currency_exposure.keys())
        exposures = np.array([currency_exposure[curr] for curr in currencies])
        volatilities = np.array([fx_volatilities.get(curr, 0.15) for curr in currencies])
        
        # Build covariance matrix
        if correlation_matrix is None:
            correlation_matrix = np.eye(len(currencies))
        
        cov_matrix = np.outer(volatilities, volatilities) * correlation_matrix
        
        # Portfolio variance
        portfolio_variance = np.dot(exposures, np.dot(cov_matrix, exposures))
        portfolio_vol = math.sqrt(max(0, portfolio_variance))
        
        # VaR calculations (assuming normal distribution)
        var_95 = portfolio_vol * stats.norm.ppf(0.95)
        var_99 = portfolio_vol * stats.norm.ppf(0.99)
        
        # Expected Shortfall (CVaR)
        expected_shortfall = portfolio_vol * stats.norm.pdf(stats.norm.ppf(0.95)) / 0.05
        
        return var_95, var_99, expected_shortfall
    
    def _calculate_portfolio_fx_volatility(self, 
                                         currency_exposure: Dict[str, float],
                                         fx_volatilities: Dict[str, float],
                                         correlation_matrix: Optional[np.ndarray] = None) -> float:
        """Calculate portfolio FX volatility."""
        if not currency_exposure:
            return 0.0
        
        currencies = list(currency_exposure.keys())
        exposures = np.array([currency_exposure[curr] for curr in currencies])
        volatilities = np.array([fx_volatilities.get(curr, 0.15) for curr in currencies])
        
        if correlation_matrix is None:
            correlation_matrix = np.eye(len(currencies))
        
        # Normalize exposures
        total_exposure = np.sum(np.abs(exposures))
        if total_exposure > 0:
            weights = exposures / total_exposure
        else:
            weights = np.ones(len(exposures)) / len(exposures)
        
        # Portfolio volatility
        cov_matrix = np.outer(volatilities, volatilities) * correlation_matrix
        portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
        
        return math.sqrt(max(0, portfolio_variance))
    
    def _calculate_concentration_ratio(self, currency_exposure: Dict[str, float]) -> float:
        """Calculate concentration ratio (Herfindahl index)."""
        if not currency_exposure:
            return 0.0
        
        total_exposure = sum(abs(exp) for exp in currency_exposure.values())
        if total_exposure == 0:
            return 0.0
        
        # Calculate Herfindahl index
        concentration = sum((abs(exp) / total_exposure) ** 2 for exp in currency_exposure.values())
        
        return concentration
    
    def _extract_currency_correlations(self, 
                                     currencies: List[str],
                                     correlation_matrix: Optional[np.ndarray] = None) -> Dict[Tuple[str, str], float]:
        """Extract currency correlations from matrix."""
        correlations = {}
        
        if correlation_matrix is None or len(currencies) == 0:
            return correlations
        
        n = len(currencies)
        if correlation_matrix.shape != (n, n):
            return correlations
        
        for i in range(n):
            for j in range(i + 1, n):
                pair = (currencies[i], currencies[j])
                correlations[pair] = correlation_matrix[i, j]
        
        return correlations
    
    def _get_metric_value(self, 
                        limit: FXRiskLimit,
                        risk_metrics: FXRiskMetrics,
                        portfolio_value: float) -> float:
        """Get current value for a risk metric."""
        if limit.limit_type == RiskLimit.ABSOLUTE_VAR:
            return risk_metrics.total_fx_var_95
        elif limit.limit_type == RiskLimit.RELATIVE_VAR:
            return risk_metrics.total_fx_var_95 / portfolio_value if portfolio_value > 0 else 0.0
        elif limit.limit_type == RiskLimit.NOTIONAL_EXPOSURE:
            if limit.currency_pair:
                return abs(risk_metrics.currency_exposure.get(limit.currency_pair, 0.0))
            else:
                return sum(abs(exp) for exp in risk_metrics.currency_exposure.values())
        elif limit.limit_type == RiskLimit.CONCENTRATION:
            return risk_metrics.fx_concentration_ratio
        elif limit.limit_type == RiskLimit.LEVERAGE:
            return risk_metrics.fx_leverage
        elif limit.limit_type == RiskLimit.VOLATILITY_LIMIT:
            return risk_metrics.fx_volatility
        else:
            return 0.0
    
    def _initialize_default_scenarios(self) -> None:
        """Initialize default stress test scenarios."""
        # Major currency crisis
        self.stress_scenarios.append(FXStressTest(
            scenario_name="Major Currency Crisis",
            currency_shocks={
                "EUR/USD": -0.15,  # 15% EUR depreciation
                "GBP/USD": -0.20,  # 20% GBP depreciation
                "JPY/USD": 0.10,   # 10% JPY appreciation
                "CHF/USD": 0.08,   # 8% CHF appreciation
            },
            correlation_adjustment=1.5,  # Increased correlations
            volatility_adjustment=2.0    # Doubled volatility
        ))
        
        # Emerging market crisis
        self.stress_scenarios.append(FXStressTest(
            scenario_name="Emerging Market Crisis",
            currency_shocks={
                "BRL/USD": -0.25,  # 25% BRL depreciation
                "MXN/USD": -0.20,  # 20% MXN depreciation
                "ZAR/USD": -0.30,  # 30% ZAR depreciation
                "TRY/USD": -0.35,  # 35% TRY depreciation
            },
            correlation_adjustment=1.8,
            volatility_adjustment=2.5
        ))
        
        # Safe haven flow
        self.stress_scenarios.append(FXStressTest(
            scenario_name="Safe Haven Flow",
            currency_shocks={
                "USD/JPY": -0.08,  # JPY appreciation
                "USD/CHF": -0.06,  # CHF appreciation
                "EUR/USD": -0.05,  # EUR depreciation
                "GBP/USD": -0.08,  # GBP depreciation
            },
            correlation_adjustment=0.8,
            volatility_adjustment=1.5
        ))