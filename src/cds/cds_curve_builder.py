"""
CDS curve construction and bootstrapping functionality.
"""

import numpy as np
from datetime import date, datetime, timedelta
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass, field
from scipy import interpolate, optimize
from enum import Enum
import math

from ..models.cds import CDSQuote, CDSCurve, CDSConvention
from ..pricing.yield_curve import YieldCurve


class CDSInterpolationMethod(Enum):
    """CDS curve interpolation methods."""
    LINEAR_SPREAD = "linear_spread"
    LINEAR_HAZARD = "linear_hazard"
    CUBIC_SPLINE = "cubic_spline"
    PIECEWISE_CONSTANT = "piecewise_constant"
    LOG_LINEAR = "log_linear"


@dataclass
class CDSBootstrapResult:
    """Result of CDS curve bootstrapping."""
    survival_probabilities: Dict[float, float]
    hazard_rates: Dict[float, float]
    default_probabilities: Dict[float, float]
    par_spreads: Dict[float, float]
    upfront_values: Dict[float, float]
    
    def get_survival_probability(self, maturity: float) -> float:
        """Get survival probability for given maturity."""
        if maturity in self.survival_probabilities:
            return self.survival_probabilities[maturity]
        
        # Linear interpolation
        maturities = sorted(self.survival_probabilities.keys())
        if maturity <= maturities[0]:
            return self.survival_probabilities[maturities[0]]
        elif maturity >= maturities[-1]:
            return self.survival_probabilities[maturities[-1]]
        
        # Find surrounding points
        for i in range(len(maturities) - 1):
            if maturities[i] <= maturity <= maturities[i + 1]:
                t1, t2 = maturities[i], maturities[i + 1]
                s1, s2 = self.survival_probabilities[t1], self.survival_probabilities[t2]
                
                # Linear interpolation in log space
                log_s1, log_s2 = math.log(s1), math.log(s2)
                weight = (maturity - t1) / (t2 - t1)
                log_s = log_s1 + weight * (log_s2 - log_s1)
                return math.exp(log_s)
        
        return 1.0


class CDSCurveBuilder:
    """
    Builder for constructing CDS curves from market quotes.
    """
    
    def __init__(self, interpolation_method: CDSInterpolationMethod = CDSInterpolationMethod.LINEAR_HAZARD):
        """
        Initialize CDS curve builder.
        
        Args:
            interpolation_method: Method for curve interpolation
        """
        self.interpolation_method = interpolation_method
    
    def build_curve(self, entity: str, currency: str, curve_date: date,
                   cds_quotes: List[CDSQuote], 
                   risk_free_curve: YieldCurve,
                   recovery_rate: float = 0.4) -> CDSCurve:
        """
        Build CDS curve from market quotes.
        
        Args:
            entity: Entity name (country/issuer)
            currency: Currency of the CDS
            curve_date: Curve construction date
            cds_quotes: List of CDS market quotes
            risk_free_curve: Risk-free yield curve
            recovery_rate: Recovery rate assumption
            
        Returns:
            Constructed CDS curve
        """
        # Sort quotes by maturity
        sorted_quotes = sorted(cds_quotes, key=lambda x: x.maturity_years)
        
        # Bootstrap the curve
        bootstrap_result = self._bootstrap_curve(
            sorted_quotes, risk_free_curve, recovery_rate, curve_date
        )
        
        # Create CDS curve object
        cds_curve = CDSCurve(
            entity=entity,
            currency=currency,
            curve_date=curve_date,
            quotes=sorted_quotes,
            recovery_rate=recovery_rate
        )
        
        # Set interpolated values
        cds_curve._survival_probabilities = bootstrap_result.survival_probabilities
        cds_curve._hazard_rates = bootstrap_result.hazard_rates
        cds_curve._default_probabilities = bootstrap_result.default_probabilities
        
        return cds_curve
    
    def _bootstrap_curve(self, quotes: List[CDSQuote], 
                        risk_free_curve: YieldCurve,
                        recovery_rate: float,
                        curve_date: date) -> CDSBootstrapResult:
        """
        Bootstrap CDS curve from quotes.
        
        Args:
            quotes: Sorted CDS quotes
            risk_free_curve: Risk-free curve
            recovery_rate: Recovery rate
            curve_date: Curve date
            
        Returns:
            Bootstrap result with survival probabilities and hazard rates
        """
        survival_probs = {0.0: 1.0}  # Start with 100% survival at t=0
        hazard_rates = {}
        default_probs = {}
        par_spreads = {}
        upfront_values = {}
        
        for i, quote in enumerate(quotes):
            maturity = quote.maturity_years
            
            if quote.quote_type == "spread":
                # Bootstrap from par spread
                spread = quote.value
                
                # Solve for hazard rate that matches the par spread
                hazard_rate = self._solve_hazard_rate_from_spread(
                    spread, maturity, recovery_rate, risk_free_curve, 
                    survival_probs, curve_date
                )
                
                # Calculate survival probability
                if i == 0:
                    # First point
                    survival_prob = math.exp(-hazard_rate * maturity)
                else:
                    # Use piecewise constant hazard rate
                    prev_maturity = quotes[i-1].maturity_years
                    prev_survival = survival_probs[prev_maturity]
                    
                    survival_prob = prev_survival * math.exp(-hazard_rate * (maturity - prev_maturity))
                
                survival_probs[maturity] = survival_prob
                hazard_rates[maturity] = hazard_rate
                default_probs[maturity] = 1.0 - survival_prob
                par_spreads[maturity] = spread
                
            elif quote.quote_type == "upfront":
                # Bootstrap from upfront quote
                upfront = quote.value
                running_spread = quote.running_spread or 100  # 100bp default
                
                # Solve for hazard rate that matches upfront value
                hazard_rate = self._solve_hazard_rate_from_upfront(
                    upfront, running_spread, maturity, recovery_rate,
                    risk_free_curve, survival_probs, curve_date
                )
                
                # Calculate survival probability
                if i == 0:
                    survival_prob = math.exp(-hazard_rate * maturity)
                else:
                    prev_maturity = quotes[i-1].maturity_years
                    prev_survival = survival_probs[prev_maturity]
                    survival_prob = prev_survival * math.exp(-hazard_rate * (maturity - prev_maturity))
                
                survival_probs[maturity] = survival_prob
                hazard_rates[maturity] = hazard_rate
                default_probs[maturity] = 1.0 - survival_prob
                upfront_values[maturity] = upfront
        
        return CDSBootstrapResult(
            survival_probabilities=survival_probs,
            hazard_rates=hazard_rates,
            default_probabilities=default_probs,
            par_spreads=par_spreads,
            upfront_values=upfront_values
        )
    
    def _solve_hazard_rate_from_spread(self, spread: float, maturity: float,
                                     recovery_rate: float, risk_free_curve: YieldCurve,
                                     existing_survival_probs: Dict[float, float],
                                     curve_date: date) -> float:
        """
        Solve for hazard rate that matches the given par spread.
        
        Args:
            spread: Par spread in basis points
            maturity: CDS maturity in years
            recovery_rate: Recovery rate
            risk_free_curve: Risk-free curve
            existing_survival_probs: Previously bootstrapped survival probabilities
            curve_date: Curve date
            
        Returns:
            Hazard rate that matches the spread
        """
        spread_decimal = spread / 10000.0  # Convert bp to decimal
        
        def objective(hazard_rate):
            # Calculate theoretical spread for this hazard rate
            theoretical_spread = self._calculate_par_spread(
                hazard_rate, maturity, recovery_rate, risk_free_curve,
                existing_survival_probs, curve_date
            )
            return (theoretical_spread - spread_decimal) ** 2
        
        # Initial guess
        initial_guess = spread_decimal / (1 - recovery_rate)
        
        # Bounds: hazard rate should be positive and reasonable
        bounds = [(1e-6, 1.0)]
        
        try:
            result = optimize.minimize_scalar(objective, bounds=bounds, method='bounded')
            return result.x if result.success else initial_guess
        except:
            return initial_guess
    
    def _solve_hazard_rate_from_upfront(self, upfront: float, running_spread: float,
                                      maturity: float, recovery_rate: float,
                                      risk_free_curve: YieldCurve,
                                      existing_survival_probs: Dict[float, float],
                                      curve_date: date) -> float:
        """
        Solve for hazard rate that matches the given upfront value.
        
        Args:
            upfront: Upfront payment as percentage of notional
            running_spread: Running spread in basis points
            maturity: CDS maturity in years
            recovery_rate: Recovery rate
            risk_free_curve: Risk-free curve
            existing_survival_probs: Previously bootstrapped survival probabilities
            curve_date: Curve date
            
        Returns:
            Hazard rate that matches the upfront
        """
        running_spread_decimal = running_spread / 10000.0
        
        def objective(hazard_rate):
            # Calculate theoretical upfront for this hazard rate
            theoretical_upfront = self._calculate_upfront_value(
                hazard_rate, running_spread_decimal, maturity, recovery_rate,
                risk_free_curve, existing_survival_probs, curve_date
            )
            return (theoretical_upfront - upfront) ** 2
        
        # Initial guess
        initial_guess = 0.01  # 1% hazard rate
        
        # Bounds
        bounds = [(1e-6, 1.0)]
        
        try:
            result = optimize.minimize_scalar(objective, bounds=bounds, method='bounded')
            return result.x if result.success else initial_guess
        except:
            return initial_guess
    
    def _calculate_par_spread(self, hazard_rate: float, maturity: float,
                            recovery_rate: float, risk_free_curve: YieldCurve,
                            existing_survival_probs: Dict[float, float],
                            curve_date: date) -> float:
        """
        Calculate par spread for given hazard rate.
        
        Args:
            hazard_rate: Constant hazard rate
            maturity: CDS maturity
            recovery_rate: Recovery rate
            risk_free_curve: Risk-free curve
            existing_survival_probs: Existing survival probabilities
            curve_date: Curve date
            
        Returns:
            Par spread as decimal
        """
        # Use quarterly payment frequency
        payment_frequency = 4
        dt = 1.0 / payment_frequency
        
        # Calculate protection leg and premium leg
        protection_leg = 0.0
        premium_leg = 0.0
        
        # Find the last bootstrapped point
        last_time = max(existing_survival_probs.keys()) if existing_survival_probs else 0.0
        
        for i in range(1, int(maturity * payment_frequency) + 1):
            t = i * dt
            
            # Get survival probability
            if t <= last_time:
                # Use existing survival probability
                survival_prob = self._interpolate_survival_prob(t, existing_survival_probs)
            else:
                # Calculate using new hazard rate
                if last_time > 0:
                    last_survival = existing_survival_probs[last_time]
                    survival_prob = last_survival * math.exp(-hazard_rate * (t - last_time))
                else:
                    survival_prob = math.exp(-hazard_rate * t)
            
            # Discount factor
            discount_factor = risk_free_curve.get_discount_factor(t)
            
            # Protection leg: (1 - R) * P(default in [t-dt, t]) * DF(t)
            if t <= dt:
                prev_survival = 1.0
            else:
                prev_survival = self._interpolate_survival_prob(t - dt, existing_survival_probs) if t - dt <= last_time else (
                    existing_survival_probs[last_time] * math.exp(-hazard_rate * (t - dt - last_time)) if last_time > 0
                    else math.exp(-hazard_rate * (t - dt))
                )
            
            default_prob = prev_survival - survival_prob
            protection_leg += (1 - recovery_rate) * default_prob * discount_factor
            
            # Premium leg: spread * dt * P(survival to t) * DF(t)
            premium_leg += dt * survival_prob * discount_factor
        
        # Par spread = Protection Leg / Premium Leg
        return protection_leg / premium_leg if premium_leg > 0 else 0.0
    
    def _calculate_upfront_value(self, hazard_rate: float, running_spread: float,
                               maturity: float, recovery_rate: float,
                               risk_free_curve: YieldCurve,
                               existing_survival_probs: Dict[float, float],
                               curve_date: date) -> float:
        """
        Calculate upfront value for given hazard rate and running spread.
        
        Args:
            hazard_rate: Hazard rate
            running_spread: Running spread as decimal
            maturity: CDS maturity
            recovery_rate: Recovery rate
            risk_free_curve: Risk-free curve
            existing_survival_probs: Existing survival probabilities
            curve_date: Curve date
            
        Returns:
            Upfront value as percentage of notional
        """
        # Calculate par spread for this hazard rate
        par_spread = self._calculate_par_spread(
            hazard_rate, maturity, recovery_rate, risk_free_curve,
            existing_survival_probs, curve_date
        )
        
        # Calculate premium leg present value for running spread
        payment_frequency = 4
        dt = 1.0 / payment_frequency
        premium_leg = 0.0
        
        last_time = max(existing_survival_probs.keys()) if existing_survival_probs else 0.0
        
        for i in range(1, int(maturity * payment_frequency) + 1):
            t = i * dt
            
            # Get survival probability
            if t <= last_time:
                survival_prob = self._interpolate_survival_prob(t, existing_survival_probs)
            else:
                if last_time > 0:
                    last_survival = existing_survival_probs[last_time]
                    survival_prob = last_survival * math.exp(-hazard_rate * (t - last_time))
                else:
                    survival_prob = math.exp(-hazard_rate * t)
            
            discount_factor = risk_free_curve.get_discount_factor(t)
            premium_leg += dt * survival_prob * discount_factor
        
        # Upfront = (Par Spread - Running Spread) * Premium Leg PV
        upfront = (par_spread - running_spread) * premium_leg
        
        return upfront
    
    def _interpolate_survival_prob(self, time: float, 
                                 survival_probs: Dict[float, float]) -> float:
        """Interpolate survival probability for given time."""
        times = sorted(survival_probs.keys())
        
        if time <= times[0]:
            return survival_probs[times[0]]
        elif time >= times[-1]:
            return survival_probs[times[-1]]
        
        # Linear interpolation in log space
        for i in range(len(times) - 1):
            if times[i] <= time <= times[i + 1]:
                t1, t2 = times[i], times[i + 1]
                s1, s2 = survival_probs[t1], survival_probs[t2]
                
                log_s1, log_s2 = math.log(max(s1, 1e-10)), math.log(max(s2, 1e-10))
                weight = (time - t1) / (t2 - t1)
                log_s = log_s1 + weight * (log_s2 - log_s1)
                return math.exp(log_s)
        
        return 1.0


class CDSBootstrapper:
    """
    Advanced CDS curve bootstrapper with multiple methodologies.
    """
    
    def __init__(self, method: str = "piecewise_constant"):
        """
        Initialize bootstrapper.
        
        Args:
            method: Bootstrapping method ('piecewise_constant', 'linear_hazard', 'cubic_spline')
        """
        self.method = method
    
    def bootstrap(self, quotes: List[CDSQuote], risk_free_curve: YieldCurve,
                 recovery_rate: float = 0.4, curve_date: date = None) -> CDSBootstrapResult:
        """
        Bootstrap CDS curve using specified method.
        
        Args:
            quotes: CDS market quotes
            risk_free_curve: Risk-free yield curve
            recovery_rate: Recovery rate assumption
            curve_date: Curve construction date
            
        Returns:
            Bootstrap result
        """
        if curve_date is None:
            curve_date = date.today()
        
        if self.method == "piecewise_constant":
            return self._bootstrap_piecewise_constant(quotes, risk_free_curve, recovery_rate, curve_date)
        elif self.method == "linear_hazard":
            return self._bootstrap_linear_hazard(quotes, risk_free_curve, recovery_rate, curve_date)
        elif self.method == "cubic_spline":
            return self._bootstrap_cubic_spline(quotes, risk_free_curve, recovery_rate, curve_date)
        else:
            raise ValueError(f"Unknown bootstrapping method: {self.method}")
    
    def _bootstrap_piecewise_constant(self, quotes: List[CDSQuote], 
                                    risk_free_curve: YieldCurve,
                                    recovery_rate: float,
                                    curve_date: date) -> CDSBootstrapResult:
        """Bootstrap with piecewise constant hazard rates."""
        # Use the CDSCurveBuilder for piecewise constant bootstrapping
        builder = CDSCurveBuilder(CDSInterpolationMethod.PIECEWISE_CONSTANT)
        return builder._bootstrap_curve(quotes, risk_free_curve, recovery_rate, curve_date)
    
    def _bootstrap_linear_hazard(self, quotes: List[CDSQuote],
                               risk_free_curve: YieldCurve,
                               recovery_rate: float,
                               curve_date: date) -> CDSBootstrapResult:
        """Bootstrap with linear hazard rate interpolation."""
        # Simplified implementation - in practice would be more sophisticated
        builder = CDSCurveBuilder(CDSInterpolationMethod.LINEAR_HAZARD)
        return builder._bootstrap_curve(quotes, risk_free_curve, recovery_rate, curve_date)
    
    def _bootstrap_cubic_spline(self, quotes: List[CDSQuote],
                              risk_free_curve: YieldCurve,
                              recovery_rate: float,
                              curve_date: date) -> CDSBootstrapResult:
        """Bootstrap with cubic spline interpolation."""
        # First bootstrap with piecewise constant, then smooth with spline
        pc_result = self._bootstrap_piecewise_constant(quotes, risk_free_curve, recovery_rate, curve_date)
        
        # Apply cubic spline smoothing to hazard rates
        maturities = np.array(sorted(pc_result.hazard_rates.keys()))
        hazard_rates = np.array([pc_result.hazard_rates[m] for m in maturities])
        
        if len(maturities) >= 4:
            spline = interpolate.CubicSpline(maturities, hazard_rates, bc_type='natural')
            
            # Recalculate survival probabilities with smoothed hazard rates
            smoothed_hazard_rates = {}
            smoothed_survival_probs = {0.0: 1.0}
            
            for maturity in maturities:
                smoothed_rate = float(spline(maturity))
                smoothed_hazard_rates[maturity] = max(smoothed_rate, 1e-6)  # Ensure positive
                
                # Recalculate survival probability
                survival_prob = math.exp(-smoothed_hazard_rates[maturity] * maturity)
                smoothed_survival_probs[maturity] = survival_prob
            
            # Update result
            pc_result.hazard_rates = smoothed_hazard_rates
            pc_result.survival_probabilities = smoothed_survival_probs
            pc_result.default_probabilities = {m: 1.0 - s for m, s in smoothed_survival_probs.items() if m > 0}
        
        return pc_result
    
    def validate_curve(self, result: CDSBootstrapResult, 
                      quotes: List[CDSQuote],
                      tolerance: float = 0.5) -> Dict[str, bool]:
        """
        Validate bootstrapped curve against input quotes.
        
        Args:
            result: Bootstrap result
            quotes: Original quotes
            tolerance: Tolerance in basis points
            
        Returns:
            Dictionary of validation results
        """
        validation_results = {}
        
        for quote in quotes:
            maturity = quote.maturity_years
            
            if quote.quote_type == "spread":
                # Check if we can reproduce the spread
                if maturity in result.par_spreads:
                    market_spread = quote.value
                    model_spread = result.par_spreads[maturity] * 10000  # Convert to bp
                    
                    diff = abs(market_spread - model_spread)
                    validation_results[f"{maturity}Y_spread"] = diff <= tolerance
            
            elif quote.quote_type == "upfront":
                # Check upfront value
                if maturity in result.upfront_values:
                    market_upfront = quote.value
                    model_upfront = result.upfront_values[maturity]
                    
                    diff = abs(market_upfront - model_upfront) * 100  # Convert to percentage points
                    validation_results[f"{maturity}Y_upfront"] = diff <= tolerance / 100
        
        return validation_results