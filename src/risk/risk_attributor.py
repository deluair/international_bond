"""
Risk Attribution Module

This module provides comprehensive risk attribution capabilities for international bond portfolios,
decomposing risk into various factors and providing detailed analysis of risk sources.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from datetime import datetime

class AttributionMethod(Enum):
    """Risk attribution methods"""
    FACTOR_DECOMPOSITION = "factor_decomposition"
    MARGINAL_CONTRIBUTION = "marginal_contribution"
    COMPONENT_VAR = "component_var"
    INCREMENTAL_VAR = "incremental_var"
    BRINSON_ATTRIBUTION = "brinson_attribution"

class RiskSource(Enum):
    """Sources of portfolio risk"""
    DURATION_RISK = "duration_risk"
    CREDIT_RISK = "credit_risk"
    CURRENCY_RISK = "currency_risk"
    COUNTRY_RISK = "country_risk"
    SECTOR_RISK = "sector_risk"
    LIQUIDITY_RISK = "liquidity_risk"
    CONCENTRATION_RISK = "concentration_risk"
    CORRELATION_RISK = "correlation_risk"

class AttributionLevel(Enum):
    """Levels of risk attribution"""
    PORTFOLIO = "portfolio"
    ASSET_CLASS = "asset_class"
    COUNTRY = "country"
    SECTOR = "sector"
    INDIVIDUAL_BOND = "individual_bond"
    RISK_FACTOR = "risk_factor"

@dataclass
class RiskContribution:
    """Individual risk contribution"""
    source: str
    risk_type: RiskSource
    contribution_absolute: float  # Absolute risk contribution
    contribution_percentage: float  # Percentage of total risk
    marginal_contribution: float  # Marginal risk contribution
    component_var: float  # Component VaR
    
@dataclass
class AttributionResult:
    """Risk attribution result"""
    attribution_method: AttributionMethod
    attribution_level: AttributionLevel
    total_portfolio_risk: float
    risk_contributions: List[RiskContribution]
    diversification_benefit: float
    concentration_metrics: Dict[str, float]
    top_risk_contributors: List[Tuple[str, float]]
    attribution_date: datetime

@dataclass
class FactorExposure:
    """Factor exposure for attribution"""
    factor_name: str
    exposure: float
    factor_return: float
    factor_volatility: float
    factor_beta: float

@dataclass
class AttributionSummary:
    """Summary of risk attribution analysis"""
    dominant_risk_sources: List[Tuple[RiskSource, float]]
    diversification_ratio: float
    effective_number_of_positions: float
    risk_concentration_index: float
    attribution_quality: float  # R-squared of attribution model

class RiskAttributor:
    """
    Comprehensive risk attribution system for international bond portfolios
    """
    
    def __init__(self):
        self.factor_models = {}
        self.risk_factor_loadings = {}
        self.correlation_matrices = {}
        self.benchmark_weights = {}
    
    def perform_risk_attribution(
        self,
        portfolio_positions: Dict[str, float],
        asset_returns: Dict[str, pd.Series],
        factor_exposures: Dict[str, Dict[str, float]],
        method: AttributionMethod = AttributionMethod.FACTOR_DECOMPOSITION,
        level: AttributionLevel = AttributionLevel.INDIVIDUAL_BOND
    ) -> AttributionResult:
        """Perform comprehensive risk attribution analysis"""
        
        if method == AttributionMethod.FACTOR_DECOMPOSITION:
            return self._factor_decomposition_attribution(
                portfolio_positions, asset_returns, factor_exposures, level
            )
        elif method == AttributionMethod.MARGINAL_CONTRIBUTION:
            return self._marginal_contribution_attribution(
                portfolio_positions, asset_returns, level
            )
        elif method == AttributionMethod.COMPONENT_VAR:
            return self._component_var_attribution(
                portfolio_positions, asset_returns, level
            )
        elif method == AttributionMethod.INCREMENTAL_VAR:
            return self._incremental_var_attribution(
                portfolio_positions, asset_returns, level
            )
        else:
            raise ValueError(f"Unsupported attribution method: {method}")
    
    def calculate_factor_contributions(
        self,
        portfolio_positions: Dict[str, float],
        factor_exposures: Dict[str, Dict[str, float]],
        factor_covariance_matrix: np.ndarray,
        factor_names: List[str]
    ) -> Dict[str, float]:
        """Calculate risk contributions by factor"""
        
        # Calculate portfolio factor exposures
        portfolio_exposures = np.zeros(len(factor_names))
        total_weight = sum(portfolio_positions.values())
        
        for i, factor in enumerate(factor_names):
            factor_exposure = 0
            for asset, weight in portfolio_positions.items():
                if asset in factor_exposures and factor in factor_exposures[asset]:
                    factor_exposure += (weight / total_weight) * factor_exposures[asset][factor]
            portfolio_exposures[i] = factor_exposure
        
        # Calculate factor contributions to portfolio variance
        portfolio_variance = np.dot(portfolio_exposures, 
                                  np.dot(factor_covariance_matrix, portfolio_exposures))
        
        factor_contributions = {}
        for i, factor in enumerate(factor_names):
            # Marginal contribution to variance
            marginal_contrib = 2 * np.dot(factor_covariance_matrix[i, :], portfolio_exposures)
            
            # Factor contribution = exposure × marginal contribution
            factor_contrib = portfolio_exposures[i] * marginal_contrib
            factor_contributions[factor] = factor_contrib / portfolio_variance if portfolio_variance > 0 else 0
        
        return factor_contributions
    
    def calculate_asset_contributions(
        self,
        portfolio_positions: Dict[str, float],
        covariance_matrix: np.ndarray,
        asset_names: List[str]
    ) -> Dict[str, RiskContribution]:
        """Calculate risk contributions by individual assets"""
        
        # Convert positions to weights
        total_value = sum(portfolio_positions.values())
        weights = np.array([portfolio_positions.get(asset, 0) / total_value for asset in asset_names])
        
        # Calculate portfolio variance
        portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        asset_contributions = {}
        
        for i, asset in enumerate(asset_names):
            if asset in portfolio_positions:
                # Marginal contribution to variance
                marginal_contrib_var = 2 * np.dot(covariance_matrix[i, :], weights)
                
                # Component contribution to variance
                component_var = weights[i] * marginal_contrib_var
                
                # Convert to volatility terms
                marginal_contrib_vol = marginal_contrib_var / (2 * portfolio_volatility) if portfolio_volatility > 0 else 0
                component_vol = component_var / (2 * portfolio_volatility) if portfolio_volatility > 0 else 0
                
                # Percentage contribution
                contrib_percentage = (component_vol / portfolio_volatility * 100) if portfolio_volatility > 0 else 0
                
                # Determine risk source (simplified)
                risk_source = self._classify_asset_risk_source(asset)
                
                asset_contributions[asset] = RiskContribution(
                    source=asset,
                    risk_type=risk_source,
                    contribution_absolute=component_vol,
                    contribution_percentage=contrib_percentage,
                    marginal_contribution=marginal_contrib_vol,
                    component_var=component_var
                )
        
        return asset_contributions
    
    def calculate_diversification_metrics(
        self,
        portfolio_positions: Dict[str, float],
        asset_volatilities: Dict[str, float],
        correlation_matrix: np.ndarray,
        asset_names: List[str]
    ) -> Dict[str, float]:
        """Calculate portfolio diversification metrics"""
        
        total_value = sum(portfolio_positions.values())
        weights = np.array([portfolio_positions.get(asset, 0) / total_value for asset in asset_names])
        
        # Portfolio volatility
        volatilities = np.array([asset_volatilities.get(asset, 0) for asset in asset_names])
        covariance_matrix = np.outer(volatilities, volatilities) * correlation_matrix
        portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Weighted average volatility (undiversified)
        weighted_avg_volatility = np.dot(weights, volatilities)
        
        # Diversification ratio
        diversification_ratio = weighted_avg_volatility / portfolio_volatility if portfolio_volatility > 0 else 1
        
        # Effective number of positions (inverse Herfindahl index)
        herfindahl_index = np.sum(weights ** 2)
        effective_positions = 1 / herfindahl_index if herfindahl_index > 0 else 1
        
        # Concentration metrics
        max_weight = np.max(weights)
        top_5_concentration = np.sum(np.sort(weights)[-5:]) if len(weights) >= 5 else np.sum(weights)
        
        # Diversification benefit
        diversification_benefit = weighted_avg_volatility - portfolio_volatility
        
        return {
            'diversification_ratio': diversification_ratio,
            'effective_positions': effective_positions,
            'herfindahl_index': herfindahl_index,
            'max_weight': max_weight,
            'top_5_concentration': top_5_concentration,
            'diversification_benefit': diversification_benefit,
            'portfolio_volatility': portfolio_volatility,
            'weighted_avg_volatility': weighted_avg_volatility
        }
    
    def perform_brinson_attribution(
        self,
        portfolio_weights: Dict[str, float],
        benchmark_weights: Dict[str, float],
        portfolio_returns: Dict[str, float],
        benchmark_returns: Dict[str, float],
        categories: Dict[str, str]  # asset -> category mapping
    ) -> Dict[str, Dict[str, float]]:
        """Perform Brinson-style performance attribution"""
        
        # Group by categories
        category_data = {}
        
        for asset in set(list(portfolio_weights.keys()) + list(benchmark_weights.keys())):
            category = categories.get(asset, 'Other')
            
            if category not in category_data:
                category_data[category] = {
                    'portfolio_weight': 0,
                    'benchmark_weight': 0,
                    'portfolio_return': 0,
                    'benchmark_return': 0,
                    'assets': []
                }
            
            category_data[category]['portfolio_weight'] += portfolio_weights.get(asset, 0)
            category_data[category]['benchmark_weight'] += benchmark_weights.get(asset, 0)
            category_data[category]['assets'].append(asset)
        
        # Calculate category returns (weighted average)
        for category, data in category_data.items():
            portfolio_return = 0
            benchmark_return = 0
            portfolio_weight_sum = 0
            benchmark_weight_sum = 0
            
            for asset in data['assets']:
                p_weight = portfolio_weights.get(asset, 0)
                b_weight = benchmark_weights.get(asset, 0)
                p_return = portfolio_returns.get(asset, 0)
                b_return = benchmark_returns.get(asset, 0)
                
                if p_weight > 0:
                    portfolio_return += p_weight * p_return
                    portfolio_weight_sum += p_weight
                
                if b_weight > 0:
                    benchmark_return += b_weight * b_return
                    benchmark_weight_sum += b_weight
            
            data['portfolio_return'] = portfolio_return / portfolio_weight_sum if portfolio_weight_sum > 0 else 0
            data['benchmark_return'] = benchmark_return / benchmark_weight_sum if benchmark_weight_sum > 0 else 0
        
        # Calculate attribution effects
        attribution_results = {}
        
        for category, data in category_data.items():
            wp = data['portfolio_weight']
            wb = data['benchmark_weight']
            rp = data['portfolio_return']
            rb = data['benchmark_return']
            
            # Allocation effect: (wp - wb) × rb
            allocation_effect = (wp - wb) * rb
            
            # Selection effect: wb × (rp - rb)
            selection_effect = wb * (rp - rb)
            
            # Interaction effect: (wp - wb) × (rp - rb)
            interaction_effect = (wp - wb) * (rp - rb)
            
            attribution_results[category] = {
                'allocation_effect': allocation_effect,
                'selection_effect': selection_effect,
                'interaction_effect': interaction_effect,
                'total_effect': allocation_effect + selection_effect + interaction_effect
            }
        
        return attribution_results
    
    def _factor_decomposition_attribution(
        self,
        portfolio_positions: Dict[str, float],
        asset_returns: Dict[str, pd.Series],
        factor_exposures: Dict[str, Dict[str, float]],
        level: AttributionLevel
    ) -> AttributionResult:
        """Perform factor decomposition attribution"""
        
        # Extract unique factors
        all_factors = set()
        for asset_factors in factor_exposures.values():
            all_factors.update(asset_factors.keys())
        factor_names = list(all_factors)
        
        # Build factor return matrix (simplified - using asset returns as proxy)
        factor_returns = {}
        for factor in factor_names:
            factor_return_series = []
            for asset, returns in asset_returns.items():
                if asset in factor_exposures and factor in factor_exposures[asset]:
                    exposure = factor_exposures[asset][factor]
                    factor_return_series.append(returns * exposure)
            
            if factor_return_series:
                factor_returns[factor] = pd.concat(factor_return_series, axis=1).mean(axis=1)
        
        # Calculate factor covariance matrix
        factor_return_df = pd.DataFrame(factor_returns)
        factor_covariance = factor_return_df.cov().values
        
        # Calculate factor contributions
        factor_contributions = self.calculate_factor_contributions(
            portfolio_positions, factor_exposures, factor_covariance, factor_names
        )
        
        # Convert to RiskContribution objects
        risk_contributions = []
        total_risk = sum(abs(contrib) for contrib in factor_contributions.values())
        
        for factor, contribution in factor_contributions.items():
            risk_source = self._map_factor_to_risk_source(factor)
            
            risk_contrib = RiskContribution(
                source=factor,
                risk_type=risk_source,
                contribution_absolute=abs(contribution),
                contribution_percentage=(abs(contribution) / total_risk * 100) if total_risk > 0 else 0,
                marginal_contribution=contribution,
                component_var=contribution ** 2
            )
            risk_contributions.append(risk_contrib)
        
        # Calculate diversification benefit
        individual_risks = [contrib.contribution_absolute for contrib in risk_contributions]
        diversification_benefit = sum(individual_risks) - total_risk
        
        # Top contributors
        top_contributors = sorted(
            [(contrib.source, contrib.contribution_absolute) for contrib in risk_contributions],
            key=lambda x: x[1], reverse=True
        )[:5]
        
        return AttributionResult(
            attribution_method=AttributionMethod.FACTOR_DECOMPOSITION,
            attribution_level=level,
            total_portfolio_risk=total_risk,
            risk_contributions=risk_contributions,
            diversification_benefit=diversification_benefit,
            concentration_metrics={},
            top_risk_contributors=top_contributors,
            attribution_date=datetime.now()
        )
    
    def _marginal_contribution_attribution(
        self,
        portfolio_positions: Dict[str, float],
        asset_returns: Dict[str, pd.Series],
        level: AttributionLevel
    ) -> AttributionResult:
        """Perform marginal contribution attribution"""
        
        # Build return matrix
        return_df = pd.DataFrame(asset_returns)
        covariance_matrix = return_df.cov().values
        asset_names = list(return_df.columns)
        
        # Calculate asset contributions
        asset_contributions = self.calculate_asset_contributions(
            portfolio_positions, covariance_matrix, asset_names
        )
        
        # Calculate total portfolio risk
        total_value = sum(portfolio_positions.values())
        weights = np.array([portfolio_positions.get(asset, 0) / total_value for asset in asset_names])
        portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
        total_portfolio_risk = np.sqrt(portfolio_variance)
        
        # Convert to list
        risk_contributions = list(asset_contributions.values())
        
        # Calculate diversification benefit
        individual_risks = [contrib.contribution_absolute for contrib in risk_contributions]
        diversification_benefit = sum(individual_risks) - total_portfolio_risk
        
        # Top contributors
        top_contributors = sorted(
            [(contrib.source, contrib.contribution_absolute) for contrib in risk_contributions],
            key=lambda x: x[1], reverse=True
        )[:5]
        
        # Concentration metrics
        weights_array = np.array([portfolio_positions.get(asset, 0) / total_value for asset in asset_names])
        concentration_metrics = {
            'herfindahl_index': np.sum(weights_array ** 2),
            'effective_positions': 1 / np.sum(weights_array ** 2) if np.sum(weights_array ** 2) > 0 else 1,
            'max_weight': np.max(weights_array)
        }
        
        return AttributionResult(
            attribution_method=AttributionMethod.MARGINAL_CONTRIBUTION,
            attribution_level=level,
            total_portfolio_risk=total_portfolio_risk,
            risk_contributions=risk_contributions,
            diversification_benefit=diversification_benefit,
            concentration_metrics=concentration_metrics,
            top_risk_contributors=top_contributors,
            attribution_date=datetime.now()
        )
    
    def _component_var_attribution(
        self,
        portfolio_positions: Dict[str, float],
        asset_returns: Dict[str, pd.Series],
        level: AttributionLevel
    ) -> AttributionResult:
        """Perform Component VaR attribution"""
        
        # Similar to marginal contribution but focuses on VaR
        return_df = pd.DataFrame(asset_returns)
        
        # Calculate portfolio returns
        total_value = sum(portfolio_positions.values())
        weights = {asset: pos / total_value for asset, pos in portfolio_positions.items()}
        
        portfolio_returns = pd.Series(0, index=return_df.index)
        for asset, weight in weights.items():
            if asset in return_df.columns:
                portfolio_returns += weight * return_df[asset]
        
        # Calculate portfolio VaR (95% confidence)
        portfolio_var_95 = np.percentile(portfolio_returns, 5)
        
        # Calculate component VaRs
        risk_contributions = []
        
        for asset in portfolio_positions.keys():
            if asset in return_df.columns:
                # Calculate marginal VaR
                asset_returns_series = return_df[asset]
                
                # Correlation with portfolio
                correlation = portfolio_returns.corr(asset_returns_series)
                
                # Asset volatility
                asset_vol = asset_returns_series.std()
                
                # Marginal VaR approximation
                marginal_var = correlation * asset_vol * 1.645  # 95% confidence
                
                # Component VaR
                weight = weights.get(asset, 0)
                component_var = weight * marginal_var
                
                # Risk source classification
                risk_source = self._classify_asset_risk_source(asset)
                
                risk_contrib = RiskContribution(
                    source=asset,
                    risk_type=risk_source,
                    contribution_absolute=abs(component_var),
                    contribution_percentage=(abs(component_var) / abs(portfolio_var_95) * 100) if portfolio_var_95 != 0 else 0,
                    marginal_contribution=marginal_var,
                    component_var=component_var
                )
                risk_contributions.append(risk_contrib)
        
        # Top contributors
        top_contributors = sorted(
            [(contrib.source, contrib.contribution_absolute) for contrib in risk_contributions],
            key=lambda x: x[1], reverse=True
        )[:5]
        
        return AttributionResult(
            attribution_method=AttributionMethod.COMPONENT_VAR,
            attribution_level=level,
            total_portfolio_risk=abs(portfolio_var_95),
            risk_contributions=risk_contributions,
            diversification_benefit=0,  # Not applicable for VaR
            concentration_metrics={},
            top_risk_contributors=top_contributors,
            attribution_date=datetime.now()
        )
    
    def _incremental_var_attribution(
        self,
        portfolio_positions: Dict[str, float],
        asset_returns: Dict[str, pd.Series],
        level: AttributionLevel
    ) -> AttributionResult:
        """Perform Incremental VaR attribution"""
        
        return_df = pd.DataFrame(asset_returns)
        
        # Calculate baseline portfolio VaR
        total_value = sum(portfolio_positions.values())
        weights = {asset: pos / total_value for asset, pos in portfolio_positions.items()}
        
        baseline_returns = pd.Series(0, index=return_df.index)
        for asset, weight in weights.items():
            if asset in return_df.columns:
                baseline_returns += weight * return_df[asset]
        
        baseline_var = np.percentile(baseline_returns, 5)
        
        # Calculate incremental VaRs
        risk_contributions = []
        
        for asset in portfolio_positions.keys():
            if asset in return_df.columns:
                # Create portfolio without this asset
                modified_weights = {a: w for a, w in weights.items() if a != asset}
                
                # Renormalize weights
                total_modified_weight = sum(modified_weights.values())
                if total_modified_weight > 0:
                    modified_weights = {a: w / total_modified_weight for a, w in modified_weights.items()}
                
                # Calculate modified portfolio returns
                modified_returns = pd.Series(0, index=return_df.index)
                for a, w in modified_weights.items():
                    if a in return_df.columns:
                        modified_returns += w * return_df[a]
                
                # Calculate VaR without this asset
                modified_var = np.percentile(modified_returns, 5) if len(modified_weights) > 0 else 0
                
                # Incremental VaR
                incremental_var = baseline_var - modified_var
                
                # Risk source classification
                risk_source = self._classify_asset_risk_source(asset)
                
                risk_contrib = RiskContribution(
                    source=asset,
                    risk_type=risk_source,
                    contribution_absolute=abs(incremental_var),
                    contribution_percentage=(abs(incremental_var) / abs(baseline_var) * 100) if baseline_var != 0 else 0,
                    marginal_contribution=incremental_var,
                    component_var=incremental_var
                )
                risk_contributions.append(risk_contrib)
        
        # Top contributors
        top_contributors = sorted(
            [(contrib.source, contrib.contribution_absolute) for contrib in risk_contributions],
            key=lambda x: x[1], reverse=True
        )[:5]
        
        return AttributionResult(
            attribution_method=AttributionMethod.INCREMENTAL_VAR,
            attribution_level=level,
            total_portfolio_risk=abs(baseline_var),
            risk_contributions=risk_contributions,
            diversification_benefit=0,  # Not applicable for incremental VaR
            concentration_metrics={},
            top_risk_contributors=top_contributors,
            attribution_date=datetime.now()
        )
    
    def _classify_asset_risk_source(self, asset_id: str) -> RiskSource:
        """Classify asset into primary risk source"""
        
        # Simple classification based on asset ID patterns
        asset_lower = asset_id.lower()
        
        if 'corp' in asset_lower or 'credit' in asset_lower:
            return RiskSource.CREDIT_RISK
        elif any(curr in asset_lower for curr in ['eur', 'jpy', 'gbp', 'cad', 'aud']):
            return RiskSource.CURRENCY_RISK
        elif any(country in asset_lower for country in ['us', 'de', 'jp', 'uk', 'fr']):
            return RiskSource.COUNTRY_RISK
        elif 'long' in asset_lower or 'short' in asset_lower or 'duration' in asset_lower:
            return RiskSource.DURATION_RISK
        else:
            return RiskSource.DURATION_RISK  # Default
    
    def _map_factor_to_risk_source(self, factor_name: str) -> RiskSource:
        """Map factor name to risk source"""
        
        factor_lower = factor_name.lower()
        
        if 'duration' in factor_lower or 'rate' in factor_lower:
            return RiskSource.DURATION_RISK
        elif 'credit' in factor_lower or 'spread' in factor_lower:
            return RiskSource.CREDIT_RISK
        elif 'currency' in factor_lower or 'fx' in factor_lower:
            return RiskSource.CURRENCY_RISK
        elif 'country' in factor_lower:
            return RiskSource.COUNTRY_RISK
        elif 'sector' in factor_lower:
            return RiskSource.SECTOR_RISK
        elif 'liquidity' in factor_lower:
            return RiskSource.LIQUIDITY_RISK
        else:
            return RiskSource.DURATION_RISK  # Default