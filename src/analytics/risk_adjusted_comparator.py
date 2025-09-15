"""
Risk-Adjusted Comparator Module

This module provides comprehensive risk-adjusted comparison capabilities for international bonds,
including Sharpe ratio analysis, information ratio calculations, and risk-adjusted performance metrics.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class RiskAdjustmentMethod(Enum):
    """Risk adjustment methods for bond comparison"""
    SHARPE_RATIO = "sharpe_ratio"
    INFORMATION_RATIO = "information_ratio"
    SORTINO_RATIO = "sortino_ratio"
    CALMAR_RATIO = "calmar_ratio"
    TREYNOR_RATIO = "treynor_ratio"
    JENSEN_ALPHA = "jensen_alpha"
    TRACKING_ERROR = "tracking_error"
    VAR_ADJUSTED = "var_adjusted"

class PerformanceMetric(Enum):
    """Performance metrics for comparison"""
    TOTAL_RETURN = "total_return"
    EXCESS_RETURN = "excess_return"
    VOLATILITY = "volatility"
    DOWNSIDE_DEVIATION = "downside_deviation"
    MAX_DRAWDOWN = "max_drawdown"
    BETA = "beta"
    CORRELATION = "correlation"
    INFORMATION_COEFFICIENT = "information_coefficient"

@dataclass
class RiskAdjustedMetrics:
    """Risk-adjusted performance metrics"""
    sharpe_ratio: float
    information_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    treynor_ratio: float
    jensen_alpha: float
    tracking_error: float
    var_95: float
    cvar_95: float
    max_drawdown: float
    volatility: float
    downside_deviation: float
    beta: float
    correlation: float
    
@dataclass
class ComparisonResult:
    """Result of risk-adjusted comparison"""
    bond_id: str
    benchmark_id: str
    metrics: RiskAdjustedMetrics
    relative_performance: Dict[str, float]
    risk_attribution: Dict[str, float]
    recommendation: str
    confidence_score: float
    
@dataclass
class PerformanceAttribution:
    """Performance attribution analysis"""
    total_return: float
    duration_contribution: float
    credit_contribution: float
    currency_contribution: float
    carry_contribution: float
    residual_contribution: float
    interaction_effects: Dict[str, float]

class RiskAdjustedComparator:
    """
    Comprehensive risk-adjusted comparison engine for international bonds
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        self.comparison_cache = {}
        
    def calculate_risk_metrics(
        self,
        returns: np.ndarray,
        benchmark_returns: Optional[np.ndarray] = None,
        risk_free_rate: Optional[float] = None
    ) -> RiskAdjustedMetrics:
        """Calculate comprehensive risk-adjusted metrics"""
        
        rf_rate = risk_free_rate or self.risk_free_rate
        excess_returns = returns - rf_rate / 252  # Daily risk-free rate
        
        # Basic metrics
        volatility = np.std(returns) * np.sqrt(252)
        mean_return = np.mean(returns) * 252
        
        # Sharpe ratio
        sharpe_ratio = (mean_return - rf_rate) / volatility if volatility > 0 else 0
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_deviation = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = (mean_return - rf_rate) / downside_deviation if downside_deviation > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdowns)
        
        # Calmar ratio
        calmar_ratio = (mean_return - rf_rate) / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # VaR and CVaR
        var_95 = np.percentile(returns, 5) * np.sqrt(252)
        cvar_95 = np.mean(returns[returns <= np.percentile(returns, 5)]) * np.sqrt(252)
        
        # Benchmark-dependent metrics
        if benchmark_returns is not None:
            # Beta
            covariance = np.cov(returns, benchmark_returns)[0, 1]
            benchmark_variance = np.var(benchmark_returns)
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
            
            # Correlation
            correlation = np.corrcoef(returns, benchmark_returns)[0, 1]
            
            # Tracking error
            tracking_error = np.std(returns - benchmark_returns) * np.sqrt(252)
            
            # Information ratio
            excess_return = mean_return - np.mean(benchmark_returns) * 252
            information_ratio = excess_return / tracking_error if tracking_error > 0 else 0
            
            # Treynor ratio
            treynor_ratio = (mean_return - rf_rate) / beta if beta != 0 else 0
            
            # Jensen's alpha
            benchmark_mean = np.mean(benchmark_returns) * 252
            jensen_alpha = mean_return - (rf_rate + beta * (benchmark_mean - rf_rate))
            
        else:
            beta = 1.0
            correlation = 1.0
            tracking_error = 0.0
            information_ratio = 0.0
            treynor_ratio = sharpe_ratio
            jensen_alpha = 0.0
        
        return RiskAdjustedMetrics(
            sharpe_ratio=sharpe_ratio,
            information_ratio=information_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            treynor_ratio=treynor_ratio,
            jensen_alpha=jensen_alpha,
            tracking_error=tracking_error,
            var_95=var_95,
            cvar_95=cvar_95,
            max_drawdown=max_drawdown,
            volatility=volatility,
            downside_deviation=downside_deviation,
            beta=beta,
            correlation=correlation
        )
    
    def compare_bonds(
        self,
        bond_returns: Dict[str, np.ndarray],
        benchmark_returns: np.ndarray,
        benchmark_id: str = "benchmark"
    ) -> Dict[str, ComparisonResult]:
        """Compare multiple bonds against a benchmark"""
        
        results = {}
        
        for bond_id, returns in bond_returns.items():
            metrics = self.calculate_risk_metrics(returns, benchmark_returns)
            
            # Calculate relative performance
            relative_performance = self._calculate_relative_performance(
                metrics, benchmark_returns
            )
            
            # Risk attribution
            risk_attribution = self._calculate_risk_attribution(
                returns, benchmark_returns
            )
            
            # Generate recommendation
            recommendation, confidence = self._generate_recommendation(metrics)
            
            results[bond_id] = ComparisonResult(
                bond_id=bond_id,
                benchmark_id=benchmark_id,
                metrics=metrics,
                relative_performance=relative_performance,
                risk_attribution=risk_attribution,
                recommendation=recommendation,
                confidence_score=confidence
            )
        
        return results
    
    def _calculate_relative_performance(
        self,
        metrics: RiskAdjustedMetrics,
        benchmark_returns: np.ndarray
    ) -> Dict[str, float]:
        """Calculate relative performance metrics"""
        
        benchmark_metrics = self.calculate_risk_metrics(benchmark_returns)
        
        return {
            "excess_sharpe": metrics.sharpe_ratio - benchmark_metrics.sharpe_ratio,
            "excess_return": metrics.jensen_alpha,
            "relative_volatility": metrics.volatility / benchmark_metrics.volatility - 1,
            "relative_max_drawdown": metrics.max_drawdown / benchmark_metrics.max_drawdown - 1,
            "tracking_error": metrics.tracking_error,
            "information_ratio": metrics.information_ratio
        }
    
    def _calculate_risk_attribution(
        self,
        returns: np.ndarray,
        benchmark_returns: np.ndarray
    ) -> Dict[str, float]:
        """Calculate risk attribution components"""
        
        # Simplified risk attribution
        excess_returns = returns - benchmark_returns
        total_excess_var = np.var(excess_returns)
        
        # Systematic vs idiosyncratic risk
        beta = np.cov(returns, benchmark_returns)[0, 1] / np.var(benchmark_returns)
        systematic_var = (beta ** 2) * np.var(benchmark_returns)
        idiosyncratic_var = np.var(returns) - systematic_var
        
        return {
            "systematic_risk": systematic_var / np.var(returns),
            "idiosyncratic_risk": idiosyncratic_var / np.var(returns),
            "tracking_error_contribution": total_excess_var,
            "beta_contribution": beta - 1.0
        }
    
    def _generate_recommendation(
        self,
        metrics: RiskAdjustedMetrics
    ) -> Tuple[str, float]:
        """Generate investment recommendation with confidence score"""
        
        # Scoring system
        score = 0
        factors = 0
        
        # Sharpe ratio scoring
        if metrics.sharpe_ratio > 1.0:
            score += 2
        elif metrics.sharpe_ratio > 0.5:
            score += 1
        factors += 1
        
        # Information ratio scoring
        if metrics.information_ratio > 0.5:
            score += 2
        elif metrics.information_ratio > 0.0:
            score += 1
        factors += 1
        
        # Maximum drawdown scoring
        if metrics.max_drawdown > -0.05:
            score += 2
        elif metrics.max_drawdown > -0.10:
            score += 1
        factors += 1
        
        # Volatility scoring (lower is better)
        if metrics.volatility < 0.05:
            score += 2
        elif metrics.volatility < 0.10:
            score += 1
        factors += 1
        
        avg_score = score / (factors * 2)  # Normalize to 0-1
        
        if avg_score >= 0.75:
            recommendation = "STRONG_BUY"
        elif avg_score >= 0.6:
            recommendation = "BUY"
        elif avg_score >= 0.4:
            recommendation = "HOLD"
        elif avg_score >= 0.25:
            recommendation = "SELL"
        else:
            recommendation = "STRONG_SELL"
        
        confidence = min(avg_score * 1.2, 1.0)  # Boost confidence slightly
        
        return recommendation, confidence
    
    def calculate_performance_attribution(
        self,
        portfolio_returns: np.ndarray,
        factor_returns: Dict[str, np.ndarray],
        weights: Optional[Dict[str, float]] = None
    ) -> PerformanceAttribution:
        """Calculate detailed performance attribution"""
        
        if weights is None:
            weights = {factor: 1.0/len(factor_returns) for factor in factor_returns}
        
        total_return = np.sum(portfolio_returns)
        
        # Factor contributions
        contributions = {}
        for factor, returns in factor_returns.items():
            factor_contribution = weights.get(factor, 0) * np.sum(returns)
            contributions[f"{factor}_contribution"] = factor_contribution
        
        # Interaction effects (simplified)
        interaction_effects = {}
        factors = list(factor_returns.keys())
        for i, factor1 in enumerate(factors):
            for factor2 in factors[i+1:]:
                interaction = np.corrcoef(
                    factor_returns[factor1], 
                    factor_returns[factor2]
                )[0, 1] * 0.01  # Simplified interaction
                interaction_effects[f"{factor1}_{factor2}"] = interaction
        
        # Residual
        explained_return = sum(contributions.values())
        residual = total_return - explained_return
        
        return PerformanceAttribution(
            total_return=total_return,
            duration_contribution=contributions.get("duration_contribution", 0),
            credit_contribution=contributions.get("credit_contribution", 0),
            currency_contribution=contributions.get("currency_contribution", 0),
            carry_contribution=contributions.get("carry_contribution", 0),
            residual_contribution=residual,
            interaction_effects=interaction_effects
        )
    
    def generate_comparison_report(
        self,
        comparison_results: Dict[str, ComparisonResult]
    ) -> Dict[str, Any]:
        """Generate comprehensive comparison report"""
        
        # Ranking by different metrics
        rankings = {
            "sharpe_ratio": sorted(
                comparison_results.items(),
                key=lambda x: x[1].metrics.sharpe_ratio,
                reverse=True
            ),
            "information_ratio": sorted(
                comparison_results.items(),
                key=lambda x: x[1].metrics.information_ratio,
                reverse=True
            ),
            "max_drawdown": sorted(
                comparison_results.items(),
                key=lambda x: x[1].metrics.max_drawdown,
                reverse=True  # Less negative is better
            )
        }
        
        # Summary statistics
        metrics_summary = {}
        for metric in ["sharpe_ratio", "information_ratio", "volatility", "max_drawdown"]:
            values = [getattr(result.metrics, metric) for result in comparison_results.values()]
            metrics_summary[metric] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "median": np.median(values)
            }
        
        # Top performers
        top_performers = {
            "best_sharpe": rankings["sharpe_ratio"][0] if rankings["sharpe_ratio"] else None,
            "best_information_ratio": rankings["information_ratio"][0] if rankings["information_ratio"] else None,
            "lowest_drawdown": rankings["max_drawdown"][0] if rankings["max_drawdown"] else None
        }
        
        return {
            "rankings": rankings,
            "summary_statistics": metrics_summary,
            "top_performers": top_performers,
            "total_bonds_analyzed": len(comparison_results),
            "analysis_timestamp": datetime.now().isoformat()
        }