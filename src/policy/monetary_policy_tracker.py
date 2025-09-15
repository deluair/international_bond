"""
Monetary Policy Tracker Module

This module provides comprehensive tracking and analysis of central bank monetary policy
decisions, communications, and market expectations.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Import CentralBank from policy_divergence_analyzer
from .policy_divergence_analyzer import CentralBank

class PolicyAction(Enum):
    """Types of monetary policy actions"""
    RATE_HIKE = "rate_hike"
    RATE_CUT = "rate_cut"
    HOLD = "hold"
    QE_EXPANSION = "qe_expansion"
    QE_TAPERING = "qe_tapering"
    QE_END = "qe_end"
    FORWARD_GUIDANCE_CHANGE = "forward_guidance_change"
    YIELD_CURVE_CONTROL = "yield_curve_control"

class PolicySignal(Enum):
    """Policy communication signals"""
    VERY_HAWKISH = "very_hawkish"
    HAWKISH = "hawkish"
    NEUTRAL = "neutral"
    DOVISH = "dovish"
    VERY_DOVISH = "very_dovish"
    DATA_DEPENDENT = "data_dependent"
    PATIENT = "patient"

class PolicyCycle(Enum):
    """Policy cycle phases"""
    EASING_CYCLE = "easing_cycle"
    PAUSE_AFTER_EASING = "pause_after_easing"
    NEUTRAL = "neutral"
    PAUSE_BEFORE_TIGHTENING = "pause_before_tightening"
    TIGHTENING_CYCLE = "tightening_cycle"
    PAUSE_AFTER_TIGHTENING = "pause_after_tightening"

@dataclass
class PolicyDecision:
    """Central bank policy decision"""
    central_bank: CentralBank
    date: datetime
    action: PolicyAction
    rate_change: float
    new_rate: float
    vote_split: Optional[str]
    statement_tone: PolicySignal
    forward_guidance: str
    market_surprise: float  # vs expectations
    
@dataclass
class PolicyCommunication:
    """Central bank communication analysis"""
    central_bank: CentralBank
    date: datetime
    speaker: str
    communication_type: str  # speech, interview, testimony
    hawkish_dovish_score: float  # -2 to +2
    key_themes: List[str]
    market_reaction: Dict[str, float]
    
@dataclass
class PolicyExpectations:
    """Market expectations for policy"""
    central_bank: CentralBank
    next_meeting_date: datetime
    probability_hike: float
    probability_cut: float
    probability_hold: float
    expected_terminal_rate: float
    expected_timing: Dict[str, datetime]  # first hike/cut dates
    
@dataclass
class EconomicConditions:
    """Economic conditions affecting policy"""
    central_bank: CentralBank
    inflation_current: float
    inflation_target: float
    unemployment_rate: float
    gdp_growth: float
    core_pce: Optional[float]
    wage_growth: Optional[float]
    financial_conditions_index: Optional[float]

class MonetaryPolicyTracker:
    """
    Comprehensive monetary policy tracking and analysis system
    """
    
    def __init__(self):
        self.policy_history = {}
        self.communication_history = {}
        self.expectations_history = {}
        self.economic_data = {}
        
    def track_policy_decision(
        self,
        decision: PolicyDecision
    ) -> Dict[str, Any]:
        """Track and analyze a new policy decision"""
        
        bank = decision.central_bank
        
        # Initialize history if needed
        if bank not in self.policy_history:
            self.policy_history[bank] = []
        
        # Add to history
        self.policy_history[bank].append(decision)
        
        # Analyze decision
        analysis = {
            'surprise_factor': self._calculate_surprise_factor(decision),
            'cycle_position': self._determine_cycle_position(bank),
            'consistency_score': self._calculate_consistency_score(bank, decision),
            'market_impact_prediction': self._predict_market_impact(decision),
            'next_move_probability': self._calculate_next_move_probability(bank, decision)
        }
        
        return analysis
    
    def analyze_communication(
        self,
        communication: PolicyCommunication
    ) -> Dict[str, Any]:
        """Analyze central bank communication"""
        
        bank = communication.central_bank
        
        # Initialize history if needed
        if bank not in self.communication_history:
            self.communication_history[bank] = []
        
        # Add to history
        self.communication_history[bank].append(communication)
        
        # Calculate communication metrics
        recent_comms = [
            c for c in self.communication_history[bank]
            if (datetime.now() - c.date).days <= 30
        ]
        
        avg_tone = np.mean([c.hawkish_dovish_score for c in recent_comms])
        tone_consistency = 1 - np.std([c.hawkish_dovish_score for c in recent_comms])
        
        # Detect tone shifts
        tone_shift = self._detect_tone_shift(bank, communication)
        
        # Theme analysis
        theme_frequency = self._analyze_themes(recent_comms)
        
        return {
            'average_tone_30d': avg_tone,
            'tone_consistency': tone_consistency,
            'tone_shift_detected': tone_shift,
            'dominant_themes': theme_frequency,
            'communication_frequency': len(recent_comms),
            'market_sensitivity': self._calculate_market_sensitivity(communication)
        }
    
    def update_market_expectations(
        self,
        expectations: PolicyExpectations
    ) -> Dict[str, Any]:
        """Update and analyze market expectations"""
        
        bank = expectations.central_bank
        
        # Initialize history if needed
        if bank not in self.expectations_history:
            self.expectations_history[bank] = []
        
        # Add to history
        self.expectations_history[bank].append(expectations)
        
        # Calculate expectation metrics
        analysis = {
            'expectations_volatility': self._calculate_expectations_volatility(bank),
            'consensus_strength': self._calculate_consensus_strength(expectations),
            'timing_uncertainty': self._calculate_timing_uncertainty(expectations),
            'terminal_rate_evolution': self._track_terminal_rate_evolution(bank),
            'market_positioning': self._infer_market_positioning(expectations)
        }
        
        return analysis
    
    def predict_next_policy_move(
        self,
        central_bank: CentralBank,
        economic_conditions: EconomicConditions,
        market_expectations: PolicyExpectations
    ) -> Dict[str, Any]:
        """Predict next policy move using multiple factors"""
        
        # Economic factor analysis
        economic_score = self._calculate_economic_policy_score(economic_conditions)
        
        # Historical pattern analysis
        historical_score = self._analyze_historical_patterns(central_bank, economic_conditions)
        
        # Communication analysis
        communication_score = self._analyze_recent_communication_tone(central_bank)
        
        # Market expectation alignment
        market_alignment = self._calculate_market_alignment(central_bank, market_expectations)
        
        # Combined prediction
        weights = {
            'economic': 0.4,
            'historical': 0.2,
            'communication': 0.25,
            'market': 0.15
        }
        
        combined_score = (
            economic_score * weights['economic'] +
            historical_score * weights['historical'] +
            communication_score * weights['communication'] +
            market_alignment * weights['market']
        )
        
        # Convert to probabilities
        if combined_score > 0.5:
            prob_hike = min(0.9, combined_score)
            prob_cut = max(0.05, (1 - combined_score) / 2)
            prob_hold = 1 - prob_hike - prob_cut
        elif combined_score < -0.5:
            prob_cut = min(0.9, abs(combined_score))
            prob_hike = max(0.05, (1 - abs(combined_score)) / 2)
            prob_hold = 1 - prob_hike - prob_cut
        else:
            prob_hold = 0.7
            prob_hike = 0.15
            prob_cut = 0.15
        
        return {
            'probability_hike': prob_hike,
            'probability_cut': prob_cut,
            'probability_hold': prob_hold,
            'confidence_level': abs(combined_score),
            'key_factors': {
                'economic_conditions': economic_score,
                'historical_patterns': historical_score,
                'communication_tone': communication_score,
                'market_expectations': market_alignment
            },
            'risk_factors': self._identify_prediction_risks(central_bank, economic_conditions),
            'timing_estimate': self._estimate_policy_timing(central_bank, combined_score)
        }
    
    def _calculate_surprise_factor(self, decision: PolicyDecision) -> float:
        """Calculate how surprising the decision was"""
        return abs(decision.market_surprise) / 0.25  # Normalize by 25bp
    
    def _determine_cycle_position(self, central_bank: CentralBank) -> PolicyCycle:
        """Determine current position in policy cycle"""
        
        if central_bank not in self.policy_history:
            return PolicyCycle.NEUTRAL
        
        recent_decisions = self.policy_history[central_bank][-6:]  # Last 6 decisions
        
        if not recent_decisions:
            return PolicyCycle.NEUTRAL
        
        # Count hikes and cuts
        hikes = sum(1 for d in recent_decisions if d.action == PolicyAction.RATE_HIKE)
        cuts = sum(1 for d in recent_decisions if d.action == PolicyAction.RATE_CUT)
        holds = len(recent_decisions) - hikes - cuts
        
        if hikes >= 3 and cuts == 0:
            return PolicyCycle.TIGHTENING_CYCLE
        elif cuts >= 3 and hikes == 0:
            return PolicyCycle.EASING_CYCLE
        elif hikes > 0 and holds >= 2:
            return PolicyCycle.PAUSE_AFTER_TIGHTENING
        elif cuts > 0 and holds >= 2:
            return PolicyCycle.PAUSE_AFTER_EASING
        else:
            return PolicyCycle.NEUTRAL
    
    def _calculate_consistency_score(
        self,
        central_bank: CentralBank,
        decision: PolicyDecision
    ) -> float:
        """Calculate consistency with previous communications"""
        
        if central_bank not in self.communication_history:
            return 0.5  # Neutral if no communication history
        
        recent_comms = [
            c for c in self.communication_history[central_bank]
            if (decision.date - c.date).days <= 60  # 2 months before decision
        ]
        
        if not recent_comms:
            return 0.5
        
        avg_tone = np.mean([c.hawkish_dovish_score for c in recent_comms])
        
        # Map decision to tone
        decision_tone = 0
        if decision.action == PolicyAction.RATE_HIKE:
            decision_tone = 1
        elif decision.action == PolicyAction.RATE_CUT:
            decision_tone = -1
        
        # Calculate consistency (1 = perfect consistency, 0 = complete inconsistency)
        consistency = 1 - abs(avg_tone - decision_tone) / 2
        
        return max(0, consistency)
    
    def _predict_market_impact(self, decision: PolicyDecision) -> Dict[str, float]:
        """Predict market impact of policy decision"""
        
        surprise = decision.market_surprise
        
        # Base impacts (in basis points or percentage points)
        impacts = {
            'short_rates': surprise * 0.8,  # 80% pass-through to short rates
            'long_rates': surprise * 0.4,   # 40% pass-through to long rates
            'fx_impact': surprise * 1.2,    # 120% impact on FX (stronger currency for hikes)
            'equity_impact': -surprise * 0.5, # Negative impact on equities for hikes
            'credit_spreads': surprise * 0.2   # Widening spreads for hikes
        }
        
        return impacts
    
    def _calculate_next_move_probability(
        self,
        central_bank: CentralBank,
        decision: PolicyDecision
    ) -> Dict[str, float]:
        """Calculate probability of next move based on current decision"""
        
        cycle_position = self._determine_cycle_position(central_bank)
        
        # Base probabilities based on cycle position
        if cycle_position == PolicyCycle.TIGHTENING_CYCLE:
            return {'hike': 0.6, 'hold': 0.35, 'cut': 0.05}
        elif cycle_position == PolicyCycle.EASING_CYCLE:
            return {'hike': 0.05, 'hold': 0.35, 'cut': 0.6}
        elif cycle_position == PolicyCycle.PAUSE_AFTER_TIGHTENING:
            return {'hike': 0.2, 'hold': 0.7, 'cut': 0.1}
        elif cycle_position == PolicyCycle.PAUSE_AFTER_EASING:
            return {'hike': 0.1, 'hold': 0.7, 'cut': 0.2}
        else:
            return {'hike': 0.3, 'hold': 0.4, 'cut': 0.3}
    
    def _detect_tone_shift(
        self,
        central_bank: CentralBank,
        communication: PolicyCommunication
    ) -> bool:
        """Detect significant shifts in communication tone"""
        
        if central_bank not in self.communication_history:
            return False
        
        recent_comms = self.communication_history[central_bank][-5:]  # Last 5 communications
        
        if len(recent_comms) < 3:
            return False
        
        # Calculate moving average tone
        prev_avg = np.mean([c.hawkish_dovish_score for c in recent_comms[:-1]])
        current_tone = communication.hawkish_dovish_score
        
        # Detect shift (threshold of 1.0 point change)
        return abs(current_tone - prev_avg) > 1.0
    
    def _analyze_themes(
        self,
        communications: List[PolicyCommunication]
    ) -> Dict[str, int]:
        """Analyze frequency of communication themes"""
        
        theme_count = {}
        
        for comm in communications:
            for theme in comm.key_themes:
                theme_count[theme] = theme_count.get(theme, 0) + 1
        
        # Sort by frequency
        return dict(sorted(theme_count.items(), key=lambda x: x[1], reverse=True))
    
    def _calculate_market_sensitivity(
        self,
        communication: PolicyCommunication
    ) -> float:
        """Calculate market sensitivity to communication"""
        
        # Sum absolute market reactions
        total_reaction = sum(abs(reaction) for reaction in communication.market_reaction.values())
        
        # Normalize by tone strength
        tone_strength = abs(communication.hawkish_dovish_score)
        
        if tone_strength > 0:
            sensitivity = total_reaction / tone_strength
        else:
            sensitivity = 0
        
        return sensitivity
    
    def _calculate_expectations_volatility(self, central_bank: CentralBank) -> float:
        """Calculate volatility of market expectations"""
        
        if central_bank not in self.expectations_history:
            return 0
        
        recent_expectations = self.expectations_history[central_bank][-10:]  # Last 10 observations
        
        if len(recent_expectations) < 2:
            return 0
        
        # Calculate volatility of hike probabilities
        hike_probs = [exp.probability_hike for exp in recent_expectations]
        return np.std(hike_probs)
    
    def _calculate_consensus_strength(self, expectations: PolicyExpectations) -> float:
        """Calculate strength of market consensus"""
        
        # Measure how concentrated probabilities are
        probs = [expectations.probability_hike, expectations.probability_cut, expectations.probability_hold]
        max_prob = max(probs)
        
        # Strong consensus if one probability > 70%
        return max_prob
    
    def _calculate_timing_uncertainty(self, expectations: PolicyExpectations) -> float:
        """Calculate uncertainty about policy timing"""
        
        # Simple measure: how far out is the expected first move
        if expectations.expected_timing:
            first_move_date = min(expectations.expected_timing.values())
            days_to_move = (first_move_date - datetime.now()).days
            
            # Normalize: longer time = more uncertainty
            return min(1.0, days_to_move / 365)
        
        return 0.5  # Default moderate uncertainty
    
    def _track_terminal_rate_evolution(self, central_bank: CentralBank) -> Dict[str, float]:
        """Track evolution of terminal rate expectations"""
        
        if central_bank not in self.expectations_history:
            return {}
        
        recent_expectations = self.expectations_history[central_bank][-5:]
        
        if len(recent_expectations) < 2:
            return {}
        
        terminal_rates = [exp.expected_terminal_rate for exp in recent_expectations]
        
        return {
            'current_terminal_rate': terminal_rates[-1],
            'change_1m': terminal_rates[-1] - terminal_rates[-2] if len(terminal_rates) >= 2 else 0,
            'volatility': np.std(terminal_rates),
            'trend': 'rising' if terminal_rates[-1] > terminal_rates[0] else 'falling'
        }
    
    def _infer_market_positioning(self, expectations: PolicyExpectations) -> Dict[str, str]:
        """Infer market positioning from expectations"""
        
        positioning = {}
        
        if expectations.probability_hike > 0.7:
            positioning['bias'] = 'hawkish'
            positioning['confidence'] = 'high'
        elif expectations.probability_cut > 0.7:
            positioning['bias'] = 'dovish'
            positioning['confidence'] = 'high'
        else:
            positioning['bias'] = 'neutral'
            positioning['confidence'] = 'low'
        
        return positioning
    
    def _calculate_economic_policy_score(self, conditions: EconomicConditions) -> float:
        """Calculate policy score based on economic conditions"""
        
        # Simplified Taylor rule approach
        inflation_gap = conditions.inflation_current - conditions.inflation_target
        
        # Unemployment gap (assume NAIRU of 4.5%)
        unemployment_gap = 4.5 - conditions.unemployment_rate
        
        # GDP growth (assume potential of 2.5%)
        growth_gap = conditions.gdp_growth - 2.5
        
        # Policy score (positive = hawkish, negative = dovish)
        score = (
            inflation_gap * 1.5 +      # 1.5x weight on inflation
            unemployment_gap * 0.5 +   # 0.5x weight on unemployment
            growth_gap * 0.3           # 0.3x weight on growth
        )
        
        # Normalize to -1 to +1 range
        return np.tanh(score / 2)
    
    def _analyze_historical_patterns(
        self,
        central_bank: CentralBank,
        conditions: EconomicConditions
    ) -> float:
        """Analyze historical policy patterns"""
        
        # Simplified pattern analysis
        # In practice, this would use machine learning on historical data
        
        if central_bank not in self.policy_history:
            return 0
        
        # Look for similar economic conditions in the past
        # For now, use simple heuristics
        
        if conditions.inflation_current > conditions.inflation_target + 1:
            return 0.7  # Historically hawkish response to high inflation
        elif conditions.unemployment_rate > 6:
            return -0.7  # Historically dovish response to high unemployment
        else:
            return 0  # Neutral
    
    def _analyze_recent_communication_tone(self, central_bank: CentralBank) -> float:
        """Analyze recent communication tone"""
        
        if central_bank not in self.communication_history:
            return 0
        
        recent_comms = [
            c for c in self.communication_history[central_bank]
            if (datetime.now() - c.date).days <= 30
        ]
        
        if not recent_comms:
            return 0
        
        return np.mean([c.hawkish_dovish_score for c in recent_comms])
    
    def _calculate_market_alignment(
        self,
        central_bank: CentralBank,
        expectations: PolicyExpectations
    ) -> float:
        """Calculate alignment with market expectations"""
        
        # Simple alignment measure
        if expectations.probability_hike > 0.6:
            return 0.5  # Market expects hikes
        elif expectations.probability_cut > 0.6:
            return -0.5  # Market expects cuts
        else:
            return 0  # Market neutral
    
    def _identify_prediction_risks(
        self,
        central_bank: CentralBank,
        conditions: EconomicConditions
    ) -> List[str]:
        """Identify risks to policy prediction"""
        
        risks = []
        
        if conditions.inflation_current > 5:
            risks.append("Inflation significantly above target")
        
        if conditions.unemployment_rate > 8:
            risks.append("Very high unemployment")
        
        if conditions.gdp_growth < -1:
            risks.append("Economic contraction")
        
        if hasattr(conditions, 'financial_conditions_index') and conditions.financial_conditions_index:
            if conditions.financial_conditions_index > 1:
                risks.append("Tight financial conditions")
        
        # Central bank specific risks
        if central_bank == CentralBank.ECB:
            risks.append("Eurozone fragmentation concerns")
        elif central_bank == CentralBank.BOJ:
            risks.append("Yield curve control policy constraints")
        
        return risks
    
    def _estimate_policy_timing(self, central_bank: CentralBank, policy_score: float) -> Dict[str, Any]:
        """Estimate timing of next policy move"""
        
        urgency = abs(policy_score)
        
        if urgency > 0.8:
            timing = "Next meeting"
            confidence = "High"
        elif urgency > 0.5:
            timing = "Within 3 months"
            confidence = "Medium"
        else:
            timing = "Beyond 6 months"
            confidence = "Low"
        
        return {
            'estimated_timing': timing,
            'confidence': confidence,
            'urgency_score': urgency
        }