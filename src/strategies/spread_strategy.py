"""
Spread Strategy Module

This module implements spread trading strategies including credit spreads,
sector spreads, quality spreads, and other relative value spread trades.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

class SpreadType(Enum):
    """Types of spread trades"""
    CREDIT_SPREAD = "credit_spread"
    SECTOR_SPREAD = "sector_spread"
    QUALITY_SPREAD = "quality_spread"
    SOVEREIGN_SPREAD = "sovereign_spread"
    CURRENCY_SPREAD = "currency_spread"
    MATURITY_SPREAD = "maturity_spread"
    SWAP_SPREAD = "swap_spread"

class SpreadDirection(Enum):
    """Direction of spread trade"""
    TIGHTEN = "tighten"    # Expect spread to narrow
    WIDEN = "widen"        # Expect spread to widen
    NEUTRAL = "neutral"

class CreditRating(Enum):
    """Credit rating categories"""
    AAA = "AAA"
    AA = "AA"
    A = "A"
    BBB = "BBB"
    BB = "BB"
    B = "B"
    CCC = "CCC"
    UNRATED = "UNRATED"

class Sector(Enum):
    """Bond sectors"""
    GOVERNMENT = "government"
    CORPORATE = "corporate"
    FINANCIAL = "financial"
    UTILITY = "utility"
    INDUSTRIAL = "industrial"
    SUPRANATIONAL = "supranational"
    AGENCY = "agency"
    MUNICIPAL = "municipal"

@dataclass
class SpreadPosition:
    """Represents a position in a spread trade"""
    bond_id: str
    sector: Sector
    rating: CreditRating
    maturity: float
    spread_to_benchmark: float
    duration: float
    weight: float
    notional: Optional[float] = None

@dataclass
class SpreadOpportunity:
    """Represents a spread trading opportunity"""
    opportunity_id: str
    spread_type: SpreadType
    direction: SpreadDirection
    long_leg: SpreadPosition
    short_leg: SpreadPosition
    current_spread: float
    fair_value_spread: float
    spread_z_score: float
    expected_pnl: float
    risk_metrics: Dict[str, float]
    confidence_score: float
    entry_date: datetime
    recommended_holding_period: int
    stop_loss_level: Optional[float]
    take_profit_level: Optional[float]

@dataclass
class SpreadAnalysis:
    """Spread analysis results"""
    analysis_date: datetime
    spread_levels: Dict[str, float]
    spread_changes: Dict[str, float]
    spread_volatilities: Dict[str, float]
    spread_correlations: Dict[Tuple[str, str], float]
    historical_percentiles: Dict[str, float]
    z_scores: Dict[str, float]
    mean_reversion_signals: Dict[str, float]
    momentum_signals: Dict[str, float]

@dataclass
class SpreadPerformance:
    """Performance tracking for spread trades"""
    opportunity_id: str
    entry_date: datetime
    exit_date: Optional[datetime]
    entry_spread: float
    exit_spread: Optional[float]
    spread_pnl: float
    duration_pnl: float
    carry_pnl: float
    total_pnl: float
    days_held: int
    max_favorable: float
    max_adverse: float
    hit_stop_loss: bool
    hit_take_profit: bool

class SpreadStrategy:
    """
    Comprehensive spread trading strategy for international bonds
    """
    
    def __init__(
        self,
        strategy_name: str = "Spread Strategy",
        lookback_period: int = 252,  # 1 year
        volatility_lookback: int = 63,  # 3 months
        mean_reversion_threshold: float = 1.5,  # 1.5 standard deviations
        momentum_threshold: float = 1.0,  # 1.0 standard deviation
        min_spread_threshold: float = 0.0005,  # 5 bps minimum spread
        max_duration_mismatch: float = 0.5  # 0.5 years max duration mismatch
    ):
        self.strategy_name = strategy_name
        self.lookback_period = lookback_period
        self.volatility_lookback = volatility_lookback
        self.mean_reversion_threshold = mean_reversion_threshold
        self.momentum_threshold = momentum_threshold
        self.min_spread_threshold = min_spread_threshold
        self.max_duration_mismatch = max_duration_mismatch
        
        # Strategy state
        self.active_opportunities: Dict[str, SpreadOpportunity] = {}
        self.historical_analysis: List[SpreadAnalysis] = []
        self.performance_history: List[SpreadPerformance] = []
        
        # Market data
        self.spread_history: pd.DataFrame = pd.DataFrame()
        self.bond_universe: Dict[str, Dict[str, Any]] = {}
    
    def analyze_spreads(
        self,
        bond_data: Dict[str, Dict[str, Any]],
        benchmark_yields: Dict[str, float],
        analysis_date: datetime
    ) -> SpreadAnalysis:
        """
        Perform comprehensive spread analysis
        """
        
        # Calculate current spread levels
        spread_levels = self._calculate_spread_levels(bond_data, benchmark_yields)
        
        # Calculate spread changes
        spread_changes = self._calculate_spread_changes(spread_levels, analysis_date)
        
        # Calculate spread volatilities
        spread_volatilities = self._calculate_spread_volatilities(analysis_date)
        
        # Calculate spread correlations
        spread_correlations = self._calculate_spread_correlations(analysis_date)
        
        # Calculate historical percentiles
        historical_percentiles = self._calculate_spread_percentiles(spread_levels)
        
        # Calculate z-scores
        z_scores = self._calculate_spread_z_scores(spread_levels)
        
        # Generate mean reversion signals
        mean_reversion_signals = self._generate_mean_reversion_signals(z_scores)
        
        # Generate momentum signals
        momentum_signals = self._generate_momentum_signals(spread_changes, spread_volatilities)
        
        analysis = SpreadAnalysis(
            analysis_date=analysis_date,
            spread_levels=spread_levels,
            spread_changes=spread_changes,
            spread_volatilities=spread_volatilities,
            spread_correlations=spread_correlations,
            historical_percentiles=historical_percentiles,
            z_scores=z_scores,
            mean_reversion_signals=mean_reversion_signals,
            momentum_signals=momentum_signals
        )
        
        self.historical_analysis.append(analysis)
        return analysis
    
    def identify_spread_opportunities(
        self,
        spread_analysis: SpreadAnalysis,
        bond_universe: Dict[str, Dict[str, Any]],
        market_conditions: Dict[str, Any]
    ) -> List[SpreadOpportunity]:
        """
        Identify spread trading opportunities
        """
        
        opportunities = []
        
        # Credit spread opportunities
        credit_opportunities = self._identify_credit_spread_opportunities(
            spread_analysis, bond_universe
        )
        opportunities.extend(credit_opportunities)
        
        # Sector spread opportunities
        sector_opportunities = self._identify_sector_spread_opportunities(
            spread_analysis, bond_universe
        )
        opportunities.extend(sector_opportunities)
        
        # Quality spread opportunities
        quality_opportunities = self._identify_quality_spread_opportunities(
            spread_analysis, bond_universe
        )
        opportunities.extend(quality_opportunities)
        
        # Sovereign spread opportunities
        sovereign_opportunities = self._identify_sovereign_spread_opportunities(
            spread_analysis, bond_universe
        )
        opportunities.extend(sovereign_opportunities)
        
        # Currency spread opportunities
        currency_opportunities = self._identify_currency_spread_opportunities(
            spread_analysis, bond_universe
        )
        opportunities.extend(currency_opportunities)
        
        # Filter and rank opportunities
        opportunities = self._filter_spread_opportunities(opportunities, market_conditions)
        opportunities.sort(key=lambda x: x.confidence_score, reverse=True)
        
        return opportunities[:15]  # Return top 15 opportunities
    
    def construct_spread_trade(
        self,
        long_bond: Dict[str, Any],
        short_bond: Dict[str, Any],
        spread_type: SpreadType,
        direction: SpreadDirection,
        target_notional: float = 1000000
    ) -> Optional[SpreadOpportunity]:
        """
        Construct a specific spread trade
        """
        
        # Create positions
        long_position = self._create_spread_position(long_bond, 1.0, target_notional)
        short_position = self._create_spread_position(short_bond, -1.0, target_notional)
        
        # Duration hedge the positions
        long_position, short_position = self._duration_hedge_positions(
            long_position, short_position
        )
        
        # Calculate spread metrics
        current_spread = long_position.spread_to_benchmark - short_position.spread_to_benchmark
        fair_value_spread = self._calculate_fair_value_spread(long_bond, short_bond)
        spread_z_score = self._calculate_spread_z_score(current_spread, long_bond, short_bond)
        
        # Calculate expected P&L
        expected_pnl = self._calculate_expected_spread_pnl(
            current_spread, fair_value_spread, long_position, short_position
        )
        
        # Calculate risk metrics
        risk_metrics = self._calculate_spread_risk_metrics(long_position, short_position)
        
        # Calculate confidence score
        confidence_score = self._calculate_spread_confidence(
            spread_z_score, risk_metrics, spread_type
        )
        
        opportunity = SpreadOpportunity(
            opportunity_id=f"{spread_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            spread_type=spread_type,
            direction=direction,
            long_leg=long_position,
            short_leg=short_position,
            current_spread=current_spread,
            fair_value_spread=fair_value_spread,
            spread_z_score=spread_z_score,
            expected_pnl=expected_pnl,
            risk_metrics=risk_metrics,
            confidence_score=confidence_score,
            entry_date=datetime.now(),
            recommended_holding_period=90,
            stop_loss_level=-0.01,
            take_profit_level=0.02
        )
        
        return opportunity
    
    def calculate_spread_pnl(
        self,
        opportunity: SpreadOpportunity,
        current_long_spread: float,
        current_short_spread: float,
        days_elapsed: int
    ) -> Dict[str, float]:
        """
        Calculate P&L components for a spread trade
        """
        
        # Spread P&L
        initial_spread = opportunity.current_spread
        current_spread = current_long_spread - current_short_spread
        spread_change = current_spread - initial_spread
        
        # For tightening trades, positive spread change is negative P&L
        if opportunity.direction == SpreadDirection.TIGHTEN:
            spread_pnl = -spread_change * opportunity.long_leg.duration * opportunity.long_leg.notional / 10000
        else:
            spread_pnl = spread_change * opportunity.long_leg.duration * opportunity.long_leg.notional / 10000
        
        # Duration P&L (simplified - assumes parallel yield curve shifts)
        duration_pnl = 0.0  # Would need yield curve changes
        
        # Carry P&L
        long_carry = opportunity.long_leg.spread_to_benchmark * (days_elapsed / 365)
        short_carry = opportunity.short_leg.spread_to_benchmark * (days_elapsed / 365)
        carry_pnl = (long_carry - short_carry) * opportunity.long_leg.notional / 10000
        
        total_pnl = spread_pnl + duration_pnl + carry_pnl
        
        return {
            'spread_pnl': spread_pnl,
            'duration_pnl': duration_pnl,
            'carry_pnl': carry_pnl,
            'total_pnl': total_pnl
        }
    
    def monitor_spread_trades(
        self,
        current_spreads: Dict[str, float],
        monitoring_date: datetime
    ) -> Dict[str, Dict[str, Any]]:
        """
        Monitor active spread trades and generate signals
        """
        
        monitoring_results = {}
        
        for opp_id, opportunity in self.active_opportunities.items():
            # Get current spreads for the bonds
            long_bond_id = opportunity.long_leg.bond_id
            short_bond_id = opportunity.short_leg.bond_id
            
            current_long_spread = current_spreads.get(long_bond_id, opportunity.long_leg.spread_to_benchmark)
            current_short_spread = current_spreads.get(short_bond_id, opportunity.short_leg.spread_to_benchmark)
            
            # Calculate current P&L
            days_elapsed = (monitoring_date - opportunity.entry_date).days
            pnl_components = self.calculate_spread_pnl(
                opportunity, current_long_spread, current_short_spread, days_elapsed
            )
            
            # Check exit conditions
            exit_signal = self._check_spread_exit_conditions(
                opportunity, pnl_components, days_elapsed
            )
            
            # Update performance tracking
            performance = SpreadPerformance(
                opportunity_id=opp_id,
                entry_date=opportunity.entry_date,
                exit_date=monitoring_date if exit_signal else None,
                entry_spread=opportunity.current_spread,
                exit_spread=current_long_spread - current_short_spread if exit_signal else None,
                spread_pnl=pnl_components['spread_pnl'],
                duration_pnl=pnl_components['duration_pnl'],
                carry_pnl=pnl_components['carry_pnl'],
                total_pnl=pnl_components['total_pnl'],
                days_held=days_elapsed,
                max_favorable=pnl_components['total_pnl'],  # Simplified
                max_adverse=pnl_components['total_pnl'],   # Simplified
                hit_stop_loss=pnl_components['total_pnl'] <= (opportunity.stop_loss_level or -float('inf')),
                hit_take_profit=pnl_components['total_pnl'] >= (opportunity.take_profit_level or float('inf'))
            )
            
            monitoring_results[opp_id] = {
                'pnl_components': pnl_components,
                'exit_signal': exit_signal,
                'performance': performance,
                'current_spread': current_long_spread - current_short_spread,
                'spread_change': (current_long_spread - current_short_spread) - opportunity.current_spread
            }
        
        return monitoring_results
    
    def _calculate_spread_levels(
        self,
        bond_data: Dict[str, Dict[str, Any]],
        benchmark_yields: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate current spread levels for all bonds"""
        
        spread_levels = {}
        
        for bond_id, bond_info in bond_data.items():
            # Get appropriate benchmark
            benchmark_key = self._get_benchmark_key(bond_info)
            benchmark_yield = benchmark_yields.get(benchmark_key, 0.0)
            
            # Calculate spread
            bond_yield = bond_info.get('yield', 0.0)
            spread = bond_yield - benchmark_yield
            
            spread_levels[bond_id] = spread
        
        return spread_levels
    
    def _calculate_spread_changes(
        self,
        current_spreads: Dict[str, float],
        analysis_date: datetime
    ) -> Dict[str, float]:
        """Calculate spread changes from previous period"""
        
        spread_changes = {}
        
        # Get previous spreads (simplified - would use historical data)
        for bond_id, current_spread in current_spreads.items():
            # Placeholder for historical lookup
            previous_spread = current_spread * (1 + np.random.normal(0, 0.01))
            spread_changes[bond_id] = current_spread - previous_spread
        
        return spread_changes
    
    def _calculate_spread_volatilities(self, analysis_date: datetime) -> Dict[str, float]:
        """Calculate spread volatilities"""
        
        # Simplified volatility calculation
        volatilities = {}
        
        # Would use historical spread data to calculate realized volatilities
        # For now, using placeholder values
        for bond_id in self.bond_universe.keys():
            # Typical spread volatilities by rating
            rating = self.bond_universe.get(bond_id, {}).get('rating', 'BBB')
            if rating in ['AAA', 'AA']:
                vol = 0.002  # 20 bps
            elif rating == 'A':
                vol = 0.003  # 30 bps
            elif rating == 'BBB':
                vol = 0.005  # 50 bps
            else:
                vol = 0.008  # 80 bps
            
            volatilities[bond_id] = vol
        
        return volatilities
    
    def _calculate_spread_correlations(self, analysis_date: datetime) -> Dict[Tuple[str, str], float]:
        """Calculate spread correlations between bonds"""
        
        correlations = {}
        
        bond_ids = list(self.bond_universe.keys())
        
        for i, bond1 in enumerate(bond_ids):
            for j, bond2 in enumerate(bond_ids[i+1:], i+1):
                # Simplified correlation calculation
                # Would use historical spread data
                
                # Higher correlation for same sector/rating
                bond1_info = self.bond_universe.get(bond1, {})
                bond2_info = self.bond_universe.get(bond2, {})
                
                same_sector = bond1_info.get('sector') == bond2_info.get('sector')
                same_rating = bond1_info.get('rating') == bond2_info.get('rating')
                
                if same_sector and same_rating:
                    corr = 0.8
                elif same_sector or same_rating:
                    corr = 0.6
                else:
                    corr = 0.4
                
                correlations[(bond1, bond2)] = corr
        
        return correlations
    
    def _calculate_spread_percentiles(self, current_spreads: Dict[str, float]) -> Dict[str, float]:
        """Calculate historical percentiles for current spreads"""
        
        percentiles = {}
        
        for bond_id, current_spread in current_spreads.items():
            # Simplified percentile calculation
            # Would use historical spread distribution
            percentiles[bond_id] = np.random.uniform(10, 90)  # Placeholder
        
        return percentiles
    
    def _calculate_spread_z_scores(self, current_spreads: Dict[str, float]) -> Dict[str, float]:
        """Calculate z-scores for current spreads"""
        
        z_scores = {}
        
        for bond_id, current_spread in current_spreads.items():
            # Simplified z-score calculation
            # Would use historical mean and standard deviation
            historical_mean = current_spread * 0.95  # Placeholder
            historical_std = current_spread * 0.1    # Placeholder
            
            z_score = (current_spread - historical_mean) / historical_std
            z_scores[bond_id] = z_score
        
        return z_scores
    
    def _generate_mean_reversion_signals(self, z_scores: Dict[str, float]) -> Dict[str, float]:
        """Generate mean reversion signals based on z-scores"""
        
        signals = {}
        
        for bond_id, z_score in z_scores.items():
            if abs(z_score) > self.mean_reversion_threshold:
                # Strong mean reversion signal
                signal = -np.sign(z_score) * min(abs(z_score) / 2, 1.0)
            else:
                signal = 0.0
            
            signals[bond_id] = signal
        
        return signals
    
    def _generate_momentum_signals(
        self,
        spread_changes: Dict[str, float],
        spread_volatilities: Dict[str, float]
    ) -> Dict[str, float]:
        """Generate momentum signals based on spread changes"""
        
        signals = {}
        
        for bond_id, spread_change in spread_changes.items():
            volatility = spread_volatilities.get(bond_id, 0.005)
            
            # Normalize by volatility
            normalized_change = spread_change / volatility
            
            if abs(normalized_change) > self.momentum_threshold:
                signal = np.sign(normalized_change) * min(abs(normalized_change) / 2, 1.0)
            else:
                signal = 0.0
            
            signals[bond_id] = signal
        
        return signals
    
    def _identify_credit_spread_opportunities(
        self,
        spread_analysis: SpreadAnalysis,
        bond_universe: Dict[str, Dict[str, Any]]
    ) -> List[SpreadOpportunity]:
        """Identify credit spread opportunities"""
        
        opportunities = []
        
        # Group bonds by similar characteristics
        rating_groups = self._group_bonds_by_rating(bond_universe)
        
        for rating, bonds in rating_groups.items():
            if len(bonds) < 2:
                continue
            
            # Find pairs with significant z-score differences
            for i, bond1_id in enumerate(bonds):
                for bond2_id in bonds[i+1:]:
                    z1 = spread_analysis.z_scores.get(bond1_id, 0)
                    z2 = spread_analysis.z_scores.get(bond2_id, 0)
                    
                    z_diff = abs(z1 - z2)
                    
                    if z_diff > 1.0:  # Significant difference
                        # Determine long/short legs
                        if z1 > z2:
                            long_bond = bond_universe[bond2_id]  # Cheap bond
                            short_bond = bond_universe[bond1_id]  # Rich bond
                        else:
                            long_bond = bond_universe[bond1_id]
                            short_bond = bond_universe[bond2_id]
                        
                        opportunity = self.construct_spread_trade(
                            long_bond, short_bond,
                            SpreadType.CREDIT_SPREAD,
                            SpreadDirection.TIGHTEN
                        )
                        
                        if opportunity and opportunity.confidence_score > 0.6:
                            opportunities.append(opportunity)
        
        return opportunities
    
    def _identify_sector_spread_opportunities(
        self,
        spread_analysis: SpreadAnalysis,
        bond_universe: Dict[str, Dict[str, Any]]
    ) -> List[SpreadOpportunity]:
        """Identify sector spread opportunities"""
        
        opportunities = []
        
        # Group bonds by sector
        sector_groups = self._group_bonds_by_sector(bond_universe)
        
        # Compare sectors
        sectors = list(sector_groups.keys())
        for i, sector1 in enumerate(sectors):
            for sector2 in sectors[i+1:]:
                # Calculate average z-scores for each sector
                sector1_z_scores = [
                    spread_analysis.z_scores.get(bond_id, 0)
                    for bond_id in sector_groups[sector1]
                ]
                sector2_z_scores = [
                    spread_analysis.z_scores.get(bond_id, 0)
                    for bond_id in sector_groups[sector2]
                ]
                
                avg_z1 = np.mean(sector1_z_scores) if sector1_z_scores else 0
                avg_z2 = np.mean(sector2_z_scores) if sector2_z_scores else 0
                
                if abs(avg_z1 - avg_z2) > 0.8:
                    # Find representative bonds from each sector
                    if avg_z1 > avg_z2:
                        long_sector, short_sector = sector2, sector1
                    else:
                        long_sector, short_sector = sector1, sector2
                    
                    long_bond_id = self._select_representative_bond(
                        sector_groups[long_sector], bond_universe
                    )
                    short_bond_id = self._select_representative_bond(
                        sector_groups[short_sector], bond_universe
                    )
                    
                    if long_bond_id and short_bond_id:
                        opportunity = self.construct_spread_trade(
                            bond_universe[long_bond_id],
                            bond_universe[short_bond_id],
                            SpreadType.SECTOR_SPREAD,
                            SpreadDirection.TIGHTEN
                        )
                        
                        if opportunity and opportunity.confidence_score > 0.5:
                            opportunities.append(opportunity)
        
        return opportunities
    
    def _identify_quality_spread_opportunities(
        self,
        spread_analysis: SpreadAnalysis,
        bond_universe: Dict[str, Dict[str, Any]]
    ) -> List[SpreadOpportunity]:
        """Identify quality spread opportunities"""
        
        opportunities = []
        
        # Compare different rating categories
        rating_order = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B']
        
        for i, rating1 in enumerate(rating_order[:-1]):
            rating2 = rating_order[i + 1]
            
            # Get bonds for each rating
            rating1_bonds = [
                bond_id for bond_id, bond_info in bond_universe.items()
                if bond_info.get('rating') == rating1
            ]
            rating2_bonds = [
                bond_id for bond_id, bond_info in bond_universe.items()
                if bond_info.get('rating') == rating2
            ]
            
            if not rating1_bonds or not rating2_bonds:
                continue
            
            # Calculate average spreads
            rating1_spreads = [
                spread_analysis.spread_levels.get(bond_id, 0)
                for bond_id in rating1_bonds
            ]
            rating2_spreads = [
                spread_analysis.spread_levels.get(bond_id, 0)
                for bond_id in rating2_bonds
            ]
            
            avg_spread1 = np.mean(rating1_spreads) if rating1_spreads else 0
            avg_spread2 = np.mean(rating2_spreads) if rating2_spreads else 0
            
            quality_spread = avg_spread2 - avg_spread1
            
            # Check if quality spread is extreme
            historical_quality_spread = 0.005 * (i + 1)  # Placeholder
            z_score = (quality_spread - historical_quality_spread) / (historical_quality_spread * 0.3)
            
            if abs(z_score) > 1.5:
                # Select representative bonds
                bond1_id = self._select_representative_bond(rating1_bonds, bond_universe)
                bond2_id = self._select_representative_bond(rating2_bonds, bond_universe)
                
                if bond1_id and bond2_id:
                    if z_score > 0:  # Quality spread is wide
                        direction = SpreadDirection.TIGHTEN
                        long_bond = bond_universe[bond2_id]  # Lower quality
                        short_bond = bond_universe[bond1_id]  # Higher quality
                    else:  # Quality spread is tight
                        direction = SpreadDirection.WIDEN
                        long_bond = bond_universe[bond1_id]  # Higher quality
                        short_bond = bond_universe[bond2_id]  # Lower quality
                    
                    opportunity = self.construct_spread_trade(
                        long_bond, short_bond,
                        SpreadType.QUALITY_SPREAD,
                        direction
                    )
                    
                    if opportunity and opportunity.confidence_score > 0.6:
                        opportunities.append(opportunity)
        
        return opportunities
    
    def _identify_sovereign_spread_opportunities(
        self,
        spread_analysis: SpreadAnalysis,
        bond_universe: Dict[str, Dict[str, Any]]
    ) -> List[SpreadOpportunity]:
        """Identify sovereign spread opportunities"""
        
        opportunities = []
        
        # Get sovereign bonds
        sovereign_bonds = {
            bond_id: bond_info for bond_id, bond_info in bond_universe.items()
            if bond_info.get('sector') == Sector.GOVERNMENT.value
        }
        
        # Group by country
        country_groups = {}
        for bond_id, bond_info in sovereign_bonds.items():
            country = bond_info.get('country', 'Unknown')
            if country not in country_groups:
                country_groups[country] = []
            country_groups[country].append(bond_id)
        
        # Compare countries
        countries = list(country_groups.keys())
        for i, country1 in enumerate(countries):
            for country2 in countries[i+1:]:
                # Calculate average z-scores
                country1_z_scores = [
                    spread_analysis.z_scores.get(bond_id, 0)
                    for bond_id in country_groups[country1]
                ]
                country2_z_scores = [
                    spread_analysis.z_scores.get(bond_id, 0)
                    for bond_id in country_groups[country2]
                ]
                
                avg_z1 = np.mean(country1_z_scores) if country1_z_scores else 0
                avg_z2 = np.mean(country2_z_scores) if country2_z_scores else 0
                
                if abs(avg_z1 - avg_z2) > 1.0:
                    # Select representative bonds
                    if avg_z1 > avg_z2:
                        long_country, short_country = country2, country1
                    else:
                        long_country, short_country = country1, country2
                    
                    long_bond_id = self._select_representative_bond(
                        country_groups[long_country], bond_universe
                    )
                    short_bond_id = self._select_representative_bond(
                        country_groups[short_country], bond_universe
                    )
                    
                    if long_bond_id and short_bond_id:
                        opportunity = self.construct_spread_trade(
                            bond_universe[long_bond_id],
                            bond_universe[short_bond_id],
                            SpreadType.SOVEREIGN_SPREAD,
                            SpreadDirection.TIGHTEN
                        )
                        
                        if opportunity and opportunity.confidence_score > 0.55:
                            opportunities.append(opportunity)
        
        return opportunities
    
    def _identify_currency_spread_opportunities(
        self,
        spread_analysis: SpreadAnalysis,
        bond_universe: Dict[str, Dict[str, Any]]
    ) -> List[SpreadOpportunity]:
        """Identify currency spread opportunities"""
        
        opportunities = []
        
        # Group bonds by currency
        currency_groups = {}
        for bond_id, bond_info in bond_universe.items():
            currency = bond_info.get('currency', 'USD')
            if currency not in currency_groups:
                currency_groups[currency] = []
            currency_groups[currency].append(bond_id)
        
        # Compare currencies (focus on major pairs)
        major_currencies = ['USD', 'EUR', 'GBP', 'JPY']
        
        for i, currency1 in enumerate(major_currencies):
            for currency2 in major_currencies[i+1:]:
                if currency1 not in currency_groups or currency2 not in currency_groups:
                    continue
                
                # Calculate average spreads for similar bonds in different currencies
                # This is simplified - would need more sophisticated currency spread analysis
                
                currency1_bonds = currency_groups[currency1]
                currency2_bonds = currency_groups[currency2]
                
                # Find similar maturity/rating bonds
                for bond1_id in currency1_bonds:
                    bond1_info = bond_universe[bond1_id]
                    
                    for bond2_id in currency2_bonds:
                        bond2_info = bond_universe[bond2_id]
                        
                        # Check if bonds are similar
                        maturity_diff = abs(bond1_info.get('maturity', 0) - bond2_info.get('maturity', 0))
                        same_rating = bond1_info.get('rating') == bond2_info.get('rating')
                        
                        if maturity_diff < 1.0 and same_rating:
                            z1 = spread_analysis.z_scores.get(bond1_id, 0)
                            z2 = spread_analysis.z_scores.get(bond2_id, 0)
                            
                            if abs(z1 - z2) > 1.2:
                                if z1 > z2:
                                    long_bond, short_bond = bond2_info, bond1_info
                                else:
                                    long_bond, short_bond = bond1_info, bond2_info
                                
                                opportunity = self.construct_spread_trade(
                                    long_bond, short_bond,
                                    SpreadType.CURRENCY_SPREAD,
                                    SpreadDirection.TIGHTEN
                                )
                                
                                if opportunity and opportunity.confidence_score > 0.5:
                                    opportunities.append(opportunity)
                                    break  # Only one opportunity per currency pair
        
        return opportunities
    
    def _filter_spread_opportunities(
        self,
        opportunities: List[SpreadOpportunity],
        market_conditions: Dict[str, Any]
    ) -> List[SpreadOpportunity]:
        """Filter opportunities based on market conditions and risk limits"""
        
        filtered = []
        
        for opportunity in opportunities:
            # Check minimum confidence
            if opportunity.confidence_score < 0.5:
                continue
            
            # Check minimum spread
            if abs(opportunity.current_spread) < self.min_spread_threshold:
                continue
            
            # Check duration mismatch
            duration_diff = abs(opportunity.long_leg.duration - opportunity.short_leg.duration)
            if duration_diff > self.max_duration_mismatch:
                continue
            
            # Check expected P&L
            if opportunity.expected_pnl < 0.005:  # 50 bps minimum
                continue
            
            filtered.append(opportunity)
        
        return filtered
    
    def _create_spread_position(
        self,
        bond_info: Dict[str, Any],
        direction: float,
        target_notional: float
    ) -> SpreadPosition:
        """Create a spread position from bond information"""
        
        return SpreadPosition(
            bond_id=bond_info.get('bond_id', ''),
            sector=Sector(bond_info.get('sector', 'corporate')),
            rating=CreditRating(bond_info.get('rating', 'BBB')),
            maturity=bond_info.get('maturity', 5.0),
            spread_to_benchmark=bond_info.get('spread', 0.01),
            duration=bond_info.get('duration', bond_info.get('maturity', 5.0) * 0.85),
            weight=direction,
            notional=target_notional * abs(direction)
        )
    
    def _duration_hedge_positions(
        self,
        long_position: SpreadPosition,
        short_position: SpreadPosition
    ) -> Tuple[SpreadPosition, SpreadPosition]:
        """Adjust position weights to be duration neutral"""
        
        # Calculate duration-neutral weights
        long_duration = long_position.duration
        short_duration = short_position.duration
        
        # Adjust short position weight to match long position duration
        duration_ratio = long_duration / short_duration
        short_position.weight = -duration_ratio
        short_position.notional = long_position.notional * duration_ratio
        
        return long_position, short_position
    
    def _calculate_fair_value_spread(
        self,
        long_bond: Dict[str, Any],
        short_bond: Dict[str, Any]
    ) -> float:
        """Calculate fair value spread between two bonds"""
        
        # Simplified fair value calculation
        # Would use more sophisticated models in practice
        
        long_spread = long_bond.get('spread', 0.01)
        short_spread = short_bond.get('spread', 0.005)
        
        # Adjust for rating difference
        rating_adjustment = self._get_rating_adjustment(
            long_bond.get('rating', 'BBB'),
            short_bond.get('rating', 'A')
        )
        
        # Adjust for maturity difference
        maturity_adjustment = self._get_maturity_adjustment(
            long_bond.get('maturity', 5.0),
            short_bond.get('maturity', 5.0)
        )
        
        fair_value_spread = (long_spread - short_spread) + rating_adjustment + maturity_adjustment
        
        return fair_value_spread
    
    def _calculate_spread_z_score(
        self,
        current_spread: float,
        long_bond: Dict[str, Any],
        short_bond: Dict[str, Any]
    ) -> float:
        """Calculate z-score for the spread"""
        
        # Simplified z-score calculation
        # Would use historical spread distribution
        
        fair_value_spread = self._calculate_fair_value_spread(long_bond, short_bond)
        spread_volatility = 0.002  # 20 bps typical spread volatility
        
        z_score = (current_spread - fair_value_spread) / spread_volatility
        
        return z_score
    
    def _calculate_expected_spread_pnl(
        self,
        current_spread: float,
        fair_value_spread: float,
        long_position: SpreadPosition,
        short_position: SpreadPosition
    ) -> float:
        """Calculate expected P&L from spread convergence"""
        
        spread_convergence = fair_value_spread - current_spread
        
        # P&L from spread change
        avg_duration = (long_position.duration + short_position.duration) / 2
        expected_pnl = spread_convergence * avg_duration * long_position.notional / 10000
        
        return expected_pnl
    
    def _calculate_spread_risk_metrics(
        self,
        long_position: SpreadPosition,
        short_position: SpreadPosition
    ) -> Dict[str, float]:
        """Calculate risk metrics for the spread trade"""
        
        # Duration risk
        net_duration = long_position.duration * long_position.weight + short_position.duration * short_position.weight
        duration_risk = abs(net_duration)
        
        # Spread volatility risk
        spread_vol = 0.002  # 20 bps
        spread_risk = spread_vol * long_position.duration * long_position.notional / 10000
        
        # Correlation risk (simplified)
        correlation_risk = 0.3  # Assume 30% correlation
        
        return {
            'duration_risk': duration_risk,
            'spread_risk': spread_risk,
            'correlation_risk': correlation_risk,
            'total_risk': spread_risk * (1 - correlation_risk)
        }
    
    def _calculate_spread_confidence(
        self,
        spread_z_score: float,
        risk_metrics: Dict[str, float],
        spread_type: SpreadType
    ) -> float:
        """Calculate confidence score for the spread trade"""
        
        # Base confidence from z-score
        z_score_confidence = min(abs(spread_z_score) / 2.0, 1.0)
        
        # Risk adjustment
        risk_penalty = min(risk_metrics.get('total_risk', 0) / 0.01, 0.3)
        
        # Spread type adjustment
        type_multiplier = {
            SpreadType.CREDIT_SPREAD: 1.0,
            SpreadType.SECTOR_SPREAD: 0.9,
            SpreadType.QUALITY_SPREAD: 0.95,
            SpreadType.SOVEREIGN_SPREAD: 0.85,
            SpreadType.CURRENCY_SPREAD: 0.8
        }.get(spread_type, 0.8)
        
        confidence = (z_score_confidence - risk_penalty) * type_multiplier
        
        return max(0.0, min(1.0, confidence))
    
    def _check_spread_exit_conditions(
        self,
        opportunity: SpreadOpportunity,
        pnl_components: Dict[str, float],
        days_elapsed: int
    ) -> bool:
        """Check if spread trade should be exited"""
        
        total_pnl_pct = pnl_components['total_pnl'] / opportunity.long_leg.notional
        
        # Stop loss
        if opportunity.stop_loss_level and total_pnl_pct <= opportunity.stop_loss_level:
            return True
        
        # Take profit
        if opportunity.take_profit_level and total_pnl_pct >= opportunity.take_profit_level:
            return True
        
        # Max holding period
        if days_elapsed >= opportunity.recommended_holding_period:
            return True
        
        return False
    
    def _get_benchmark_key(self, bond_info: Dict[str, Any]) -> str:
        """Get appropriate benchmark key for a bond"""
        
        currency = bond_info.get('currency', 'USD')
        maturity = bond_info.get('maturity', 5.0)
        
        # Simplified benchmark mapping
        if maturity <= 2:
            return f"{currency}_2Y"
        elif maturity <= 5:
            return f"{currency}_5Y"
        elif maturity <= 10:
            return f"{currency}_10Y"
        else:
            return f"{currency}_30Y"
    
    def _group_bonds_by_rating(self, bond_universe: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
        """Group bonds by credit rating"""
        
        rating_groups = {}
        
        for bond_id, bond_info in bond_universe.items():
            rating = bond_info.get('rating', 'UNRATED')
            if rating not in rating_groups:
                rating_groups[rating] = []
            rating_groups[rating].append(bond_id)
        
        return rating_groups
    
    def _group_bonds_by_sector(self, bond_universe: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
        """Group bonds by sector"""
        
        sector_groups = {}
        
        for bond_id, bond_info in bond_universe.items():
            sector = bond_info.get('sector', 'corporate')
            if sector not in sector_groups:
                sector_groups[sector] = []
            sector_groups[sector].append(bond_id)
        
        return sector_groups
    
    def _select_representative_bond(
        self,
        bond_ids: List[str],
        bond_universe: Dict[str, Dict[str, Any]]
    ) -> Optional[str]:
        """Select a representative bond from a group"""
        
        if not bond_ids:
            return None
        
        # Select bond with median maturity
        maturities = [
            (bond_id, bond_universe[bond_id].get('maturity', 5.0))
            for bond_id in bond_ids
        ]
        maturities.sort(key=lambda x: x[1])
        
        median_idx = len(maturities) // 2
        return maturities[median_idx][0]
    
    def _get_rating_adjustment(self, rating1: str, rating2: str) -> float:
        """Get spread adjustment for rating difference"""
        
        rating_spreads = {
            'AAA': 0.0005,
            'AA': 0.001,
            'A': 0.002,
            'BBB': 0.004,
            'BB': 0.008,
            'B': 0.015,
            'CCC': 0.030
        }
        
        spread1 = rating_spreads.get(rating1, 0.004)
        spread2 = rating_spreads.get(rating2, 0.004)
        
        return spread1 - spread2
    
    def _get_maturity_adjustment(self, maturity1: float, maturity2: float) -> float:
        """Get spread adjustment for maturity difference"""
        
        # Simplified maturity adjustment
        maturity_diff = maturity1 - maturity2
        
        # Longer maturity typically has higher spread
        return maturity_diff * 0.0002  # 2 bps per year