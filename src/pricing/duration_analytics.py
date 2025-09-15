"""
Duration and convexity analytics for bond risk management.
"""

import numpy as np
from datetime import date, datetime, timedelta
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import math

from ..models.bond import SovereignBond
from .yield_curve import YieldCurve


class DurationType(Enum):
    """Types of duration calculations."""
    MACAULAY = "macaulay"
    MODIFIED = "modified"
    EFFECTIVE = "effective"
    KEY_RATE = "key_rate"
    DOLLAR = "dollar"


@dataclass
class DurationResult:
    """Result of duration calculation."""
    duration_type: DurationType
    value: float
    currency: str
    calculation_date: date
    yield_used: float
    price_used: float
    
    def __str__(self) -> str:
        return f"{self.duration_type.value.title()} Duration: {self.value:.4f}"


@dataclass
class ConvexityResult:
    """Result of convexity calculation."""
    convexity: float
    currency: str
    calculation_date: date
    yield_used: float
    price_used: float
    
    def __str__(self) -> str:
        return f"Convexity: {self.convexity:.4f}"


@dataclass
class KeyRateDuration:
    """Key rate duration for specific maturity bucket."""
    maturity: float  # Years
    duration: float
    shift_size: float = 0.0001  # 1bp default
    
    def __str__(self) -> str:
        return f"{self.maturity}Y: {self.duration:.4f}"


class DurationAnalytics:
    """
    Comprehensive duration and convexity analytics for bonds.
    """
    
    def __init__(self, shift_size: float = 0.0001):
        """
        Initialize duration analytics.
        
        Args:
            shift_size: Yield shift size for numerical calculations (default 1bp)
        """
        self.shift_size = shift_size
    
    def calculate_macaulay_duration(self, bond: SovereignBond, 
                                  yield_curve: YieldCurve,
                                  settlement_date: date = None) -> DurationResult:
        """
        Calculate Macaulay duration.
        
        Args:
            bond: Sovereign bond
            yield_curve: Yield curve for discounting
            settlement_date: Settlement date for calculation
            
        Returns:
            DurationResult with Macaulay duration
        """
        if settlement_date is None:
            settlement_date = date.today()
        
        cash_flows = self._generate_cash_flows(bond, settlement_date)
        ytm = self._calculate_ytm(bond, yield_curve, settlement_date)
        
        weighted_time = 0.0
        total_pv = 0.0
        
        for cf_date, cf_amount in cash_flows:
            time_to_cf = (cf_date - settlement_date).days / 365.25
            discount_factor = (1 + ytm) ** (-time_to_cf)
            pv = cf_amount * discount_factor
            
            weighted_time += time_to_cf * pv
            total_pv += pv
        
        macaulay_duration = weighted_time / total_pv if total_pv > 0 else 0.0
        
        return DurationResult(
            duration_type=DurationType.MACAULAY,
            value=macaulay_duration,
            currency=bond.currency.value,
            calculation_date=settlement_date,
            yield_used=ytm,
            price_used=total_pv
        )
    
    def calculate_modified_duration(self, bond: SovereignBond,
                                  yield_curve: YieldCurve,
                                  settlement_date: date = None) -> DurationResult:
        """
        Calculate Modified duration.
        
        Args:
            bond: Sovereign bond
            yield_curve: Yield curve for discounting
            settlement_date: Settlement date for calculation
            
        Returns:
            DurationResult with Modified duration
        """
        macaulay_result = self.calculate_macaulay_duration(bond, yield_curve, settlement_date)
        ytm = macaulay_result.yield_used
        
        # Modified duration = Macaulay duration / (1 + YTM/frequency)
        frequency = self._get_coupon_frequency(bond)
        modified_duration = macaulay_result.value / (1 + ytm / frequency)
        
        return DurationResult(
            duration_type=DurationType.MODIFIED,
            value=modified_duration,
            currency=bond.currency.value,
            calculation_date=macaulay_result.calculation_date,
            yield_used=ytm,
            price_used=macaulay_result.price_used
        )
    
    def calculate_effective_duration(self, bond: SovereignBond,
                                   yield_curve: YieldCurve,
                                   settlement_date: date = None,
                                   shift_size: float = None) -> DurationResult:
        """
        Calculate Effective duration using numerical differentiation.
        
        Args:
            bond: Sovereign bond
            yield_curve: Yield curve for discounting
            settlement_date: Settlement date for calculation
            shift_size: Yield shift size (default uses instance shift_size)
            
        Returns:
            DurationResult with Effective duration
        """
        if settlement_date is None:
            settlement_date = date.today()
        
        if shift_size is None:
            shift_size = self.shift_size
        
        # Calculate base price
        base_price = self._calculate_bond_price(bond, yield_curve, settlement_date)
        
        # Calculate price with yield up
        up_curve = yield_curve.parallel_shift(shift_size)
        price_up = self._calculate_bond_price(bond, up_curve, settlement_date)
        
        # Calculate price with yield down
        down_curve = yield_curve.parallel_shift(-shift_size)
        price_down = self._calculate_bond_price(bond, down_curve, settlement_date)
        
        # Effective duration = -(P- - P+) / (2 * P0 * Δy)
        effective_duration = -(price_down - price_up) / (2 * base_price * shift_size)
        
        ytm = self._calculate_ytm(bond, yield_curve, settlement_date)
        
        return DurationResult(
            duration_type=DurationType.EFFECTIVE,
            value=effective_duration,
            currency=bond.currency.value,
            calculation_date=settlement_date,
            yield_used=ytm,
            price_used=base_price
        )
    
    def calculate_key_rate_durations(self, bond: SovereignBond,
                                   yield_curve: YieldCurve,
                                   key_maturities: List[float] = None,
                                   settlement_date: date = None) -> List[KeyRateDuration]:
        """
        Calculate Key Rate Durations for specific maturity buckets.
        
        Args:
            bond: Sovereign bond
            yield_curve: Yield curve for discounting
            key_maturities: List of key maturities (years)
            settlement_date: Settlement date for calculation
            
        Returns:
            List of KeyRateDuration objects
        """
        if settlement_date is None:
            settlement_date = date.today()
        
        if key_maturities is None:
            key_maturities = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30]
        
        base_price = self._calculate_bond_price(bond, yield_curve, settlement_date)
        key_rate_durations = []
        
        for maturity in key_maturities:
            # Create shifted curve for this key rate
            shifted_curve = self._create_key_rate_shifted_curve(
                yield_curve, maturity, self.shift_size
            )
            
            # Calculate price with shifted curve
            shifted_price = self._calculate_bond_price(bond, shifted_curve, settlement_date)
            
            # Key rate duration = -(P_shifted - P_base) / (P_base * shift_size)
            krd = -(shifted_price - base_price) / (base_price * self.shift_size)
            
            key_rate_durations.append(KeyRateDuration(
                maturity=maturity,
                duration=krd,
                shift_size=self.shift_size
            ))
        
        return key_rate_durations
    
    def calculate_dollar_duration(self, bond: SovereignBond,
                                yield_curve: YieldCurve,
                                notional: float = 1000000,
                                settlement_date: date = None) -> DurationResult:
        """
        Calculate Dollar Duration (DV01).
        
        Args:
            bond: Sovereign bond
            yield_curve: Yield curve for discounting
            notional: Notional amount
            settlement_date: Settlement date for calculation
            
        Returns:
            DurationResult with Dollar duration
        """
        modified_duration = self.calculate_modified_duration(bond, yield_curve, settlement_date)
        
        # Dollar duration = Modified Duration * Price * 0.0001 * Notional
        price_per_unit = modified_duration.price_used
        dollar_duration = modified_duration.value * price_per_unit * 0.0001 * notional
        
        return DurationResult(
            duration_type=DurationType.DOLLAR,
            value=dollar_duration,
            currency=bond.currency.value,
            calculation_date=modified_duration.calculation_date,
            yield_used=modified_duration.yield_used,
            price_used=price_per_unit * notional
        )
    
    def calculate_convexity(self, bond: SovereignBond,
                          yield_curve: YieldCurve,
                          settlement_date: date = None,
                          shift_size: float = None) -> ConvexityResult:
        """
        Calculate bond convexity.
        
        Args:
            bond: Sovereign bond
            yield_curve: Yield curve for discounting
            settlement_date: Settlement date for calculation
            shift_size: Yield shift size
            
        Returns:
            ConvexityResult
        """
        if settlement_date is None:
            settlement_date = date.today()
        
        if shift_size is None:
            shift_size = self.shift_size
        
        # Calculate base price
        base_price = self._calculate_bond_price(bond, yield_curve, settlement_date)
        
        # Calculate price with yield up
        up_curve = yield_curve.parallel_shift(shift_size)
        price_up = self._calculate_bond_price(bond, up_curve, settlement_date)
        
        # Calculate price with yield down
        down_curve = yield_curve.parallel_shift(-shift_size)
        price_down = self._calculate_bond_price(bond, down_curve, settlement_date)
        
        # Convexity = (P+ + P- - 2*P0) / (P0 * (Δy)^2)
        convexity = (price_up + price_down - 2 * base_price) / (base_price * shift_size ** 2)
        
        ytm = self._calculate_ytm(bond, yield_curve, settlement_date)
        
        return ConvexityResult(
            convexity=convexity,
            currency=bond.currency.value,
            calculation_date=settlement_date,
            yield_used=ytm,
            price_used=base_price
        )
    
    def duration_hedge_ratio(self, target_bond: SovereignBond,
                           hedge_bond: SovereignBond,
                           yield_curve: YieldCurve,
                           settlement_date: date = None) -> float:
        """
        Calculate hedge ratio to make portfolio duration neutral.
        
        Args:
            target_bond: Bond to hedge
            hedge_bond: Bond used for hedging
            yield_curve: Yield curve for calculations
            settlement_date: Settlement date
            
        Returns:
            Hedge ratio (negative for short position)
        """
        target_duration = self.calculate_modified_duration(target_bond, yield_curve, settlement_date)
        hedge_duration = self.calculate_modified_duration(hedge_bond, yield_curve, settlement_date)
        
        target_price = target_duration.price_used
        hedge_price = hedge_duration.price_used
        
        # Hedge ratio = -(Target Duration * Target Price) / (Hedge Duration * Hedge Price)
        hedge_ratio = -(target_duration.value * target_price) / (hedge_duration.value * hedge_price)
        
        return hedge_ratio
    
    def portfolio_duration(self, bonds: List[Tuple[SovereignBond, float]],
                         yield_curve: YieldCurve,
                         settlement_date: date = None) -> DurationResult:
        """
        Calculate portfolio duration.
        
        Args:
            bonds: List of (bond, weight) tuples
            yield_curve: Yield curve for calculations
            settlement_date: Settlement date
            
        Returns:
            DurationResult for portfolio
        """
        if settlement_date is None:
            settlement_date = date.today()
        
        total_duration = 0.0
        total_weight = 0.0
        weighted_yield = 0.0
        total_value = 0.0
        
        for bond, weight in bonds:
            duration_result = self.calculate_modified_duration(bond, yield_curve, settlement_date)
            
            total_duration += duration_result.value * weight
            total_weight += weight
            weighted_yield += duration_result.yield_used * weight
            total_value += duration_result.price_used * weight
        
        portfolio_duration = total_duration / total_weight if total_weight > 0 else 0.0
        avg_yield = weighted_yield / total_weight if total_weight > 0 else 0.0
        
        return DurationResult(
            duration_type=DurationType.MODIFIED,
            value=portfolio_duration,
            currency="PORTFOLIO",
            calculation_date=settlement_date,
            yield_used=avg_yield,
            price_used=total_value
        )
    
    def _generate_cash_flows(self, bond: SovereignBond, 
                           settlement_date: date) -> List[Tuple[date, float]]:
        """Generate bond cash flows."""
        cash_flows = []
        
        # Calculate coupon payment dates
        current_date = settlement_date
        maturity_date = bond.maturity_date
        
        # Determine coupon frequency
        frequency = self._get_coupon_frequency(bond)
        months_between = 12 // frequency
        
        # Generate coupon dates
        coupon_date = maturity_date
        while coupon_date > settlement_date:
            # Add coupon payment
            coupon_amount = bond.coupon_rate * bond.face_value / frequency
            cash_flows.append((coupon_date, coupon_amount))
            
            # Move to previous coupon date
            coupon_date = coupon_date.replace(
                month=coupon_date.month - months_between if coupon_date.month > months_between 
                else coupon_date.month - months_between + 12,
                year=coupon_date.year if coupon_date.month > months_between 
                else coupon_date.year - 1
            )
        
        # Add principal repayment at maturity
        if cash_flows and cash_flows[0][0] == maturity_date:
            # Combine with final coupon
            final_coupon = cash_flows[0][1]
            cash_flows[0] = (maturity_date, final_coupon + bond.face_value)
        else:
            cash_flows.insert(0, (maturity_date, bond.face_value))
        
        # Sort by date
        cash_flows.sort(key=lambda x: x[0])
        
        return cash_flows
    
    def _get_coupon_frequency(self, bond: SovereignBond) -> int:
        """Get coupon payment frequency per year."""
        # Default to semi-annual for most sovereign bonds
        # In practice, this would be a property of the bond
        return 2
    
    def _calculate_bond_price(self, bond: SovereignBond,
                            yield_curve: YieldCurve,
                            settlement_date: date) -> float:
        """Calculate bond price using yield curve."""
        cash_flows = self._generate_cash_flows(bond, settlement_date)
        
        total_pv = 0.0
        for cf_date, cf_amount in cash_flows:
            time_to_cf = (cf_date - settlement_date).days / 365.25
            discount_factor = yield_curve.get_discount_factor(time_to_cf)
            total_pv += cf_amount * discount_factor
        
        return total_pv
    
    def _calculate_ytm(self, bond: SovereignBond,
                      yield_curve: YieldCurve,
                      settlement_date: date) -> float:
        """Calculate yield to maturity."""
        time_to_maturity = (bond.maturity_date - settlement_date).days / 365.25
        return yield_curve.get_yield(time_to_maturity)
    
    def _create_key_rate_shifted_curve(self, base_curve: YieldCurve,
                                     key_maturity: float,
                                     shift_size: float) -> YieldCurve:
        """Create yield curve with key rate shift."""
        # Create new points with shift applied to key maturity
        shifted_points = []
        
        for point in base_curve.points:
            # Apply triangular weight function centered at key_maturity
            distance = abs(point.maturity - key_maturity)
            
            # Triangular weight: 1 at key_maturity, 0 at ±1 year
            if distance <= 1.0:
                weight = 1.0 - distance
                shift = shift_size * weight
            else:
                shift = 0.0
            
            from ..models.bond import YieldPoint  # Import here to avoid circular import
            shifted_point = YieldPoint(
                maturity=point.maturity,
                yield_rate=point.yield_rate + shift,
                instrument_type=point.instrument_type,
                quote_date=point.quote_date
            )
            shifted_points.append(shifted_point)
        
        return YieldCurve(
            currency=base_curve.currency,
            curve_date=base_curve.curve_date,
            points=shifted_points,
            interpolation_method=base_curve.interpolation_method
        )


class DurationMatchingOptimizer:
    """Optimizer for duration matching strategies."""
    
    def __init__(self, target_duration: float, tolerance: float = 0.01):
        """
        Initialize duration matching optimizer.
        
        Args:
            target_duration: Target portfolio duration
            tolerance: Acceptable duration tolerance
        """
        self.target_duration = target_duration
        self.tolerance = tolerance
    
    def optimize_weights(self, bonds: List[SovereignBond],
                        yield_curve: YieldCurve,
                        analytics: DurationAnalytics,
                        settlement_date: date = None) -> Dict[str, float]:
        """
        Optimize bond weights to achieve target duration.
        
        Args:
            bonds: List of available bonds
            yield_curve: Yield curve for calculations
            analytics: Duration analytics instance
            settlement_date: Settlement date
            
        Returns:
            Dictionary of bond ISIN to weight mappings
        """
        if settlement_date is None:
            settlement_date = date.today()
        
        # Calculate duration for each bond
        durations = []
        prices = []
        
        for bond in bonds:
            duration_result = analytics.calculate_modified_duration(
                bond, yield_curve, settlement_date
            )
            durations.append(duration_result.value)
            prices.append(duration_result.price_used)
        
        durations = np.array(durations)
        prices = np.array(prices)
        
        # Simple optimization: find weights that minimize duration difference
        # Subject to: sum(weights) = 1, weights >= 0
        
        from scipy.optimize import minimize
        
        def objective(weights):
            portfolio_duration = np.sum(weights * durations)
            return (portfolio_duration - self.target_duration) ** 2
        
        def constraint_sum_to_one(weights):
            return np.sum(weights) - 1.0
        
        # Initial guess: equal weights
        n_bonds = len(bonds)
        initial_weights = np.ones(n_bonds) / n_bonds
        
        # Constraints
        constraints = [{'type': 'eq', 'fun': constraint_sum_to_one}]
        
        # Bounds: weights between 0 and 1
        bounds = [(0, 1) for _ in range(n_bonds)]
        
        # Optimize
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            weights = result.x
            # Create result dictionary
            weight_dict = {}
            for i, bond in enumerate(bonds):
                if weights[i] > 0.001:  # Only include significant weights
                    weight_dict[bond.isin] = weights[i]
            
            return weight_dict
        else:
            # Fallback: equal weights
            return {bond.isin: 1.0 / len(bonds) for bond in bonds}
    
    def calculate_tracking_error(self, weights: Dict[str, float],
                               bonds: List[SovereignBond],
                               yield_curve: YieldCurve,
                               analytics: DurationAnalytics,
                               settlement_date: date = None) -> float:
        """Calculate duration tracking error for given weights."""
        if settlement_date is None:
            settlement_date = date.today()
        
        # Create weighted bond list
        weighted_bonds = []
        for bond in bonds:
            weight = weights.get(bond.isin, 0.0)
            if weight > 0:
                weighted_bonds.append((bond, weight))
        
        # Calculate portfolio duration
        portfolio_result = analytics.portfolio_duration(
            weighted_bonds, yield_curve, settlement_date
        )
        
        # Return absolute difference from target
        return abs(portfolio_result.value - self.target_duration)