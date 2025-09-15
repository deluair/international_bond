"""
Comprehensive bond pricing engine with various pricing models.
"""

import numpy as np
from datetime import date, datetime
from typing import Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

from ..models.bond import SovereignBond


class PricingModel(Enum):
    """Bond pricing model types."""
    DISCOUNTED_CASH_FLOW = "dcf"
    BINOMIAL_TREE = "binomial"
    MONTE_CARLO = "monte_carlo"
    BLACK_SCHOLES = "black_scholes"


@dataclass
class CashFlow:
    """Individual cash flow from a bond."""
    date: date
    amount: float
    type: str  # "coupon", "principal", "call"
    discount_factor: float = 1.0
    present_value: float = 0.0


class BondPricer:
    """
    Advanced bond pricing engine with multiple models and risk calculations.
    """
    
    def __init__(self, risk_free_curve: Optional[object] = None):
        """
        Initialize bond pricer.
        
        Args:
            risk_free_curve: Yield curve for discounting (optional)
        """
        self.risk_free_curve = risk_free_curve
        self.default_risk_free_rate = 0.02  # 2% default risk-free rate
    
    def price_bond(self, bond: SovereignBond, 
                   discount_rate: Optional[float] = None,
                   model: PricingModel = PricingModel.DISCOUNTED_CASH_FLOW) -> float:
        """
        Price a bond using specified model.
        
        Args:
            bond: SovereignBond to price
            discount_rate: Override discount rate (optional)
            model: Pricing model to use
            
        Returns:
            Bond price as percentage of face value
        """
        if model == PricingModel.DISCOUNTED_CASH_FLOW:
            return self._price_dcf(bond, discount_rate)
        elif model == PricingModel.BINOMIAL_TREE:
            return self._price_binomial(bond, discount_rate)
        elif model == PricingModel.MONTE_CARLO:
            return self._price_monte_carlo(bond, discount_rate)
        else:
            raise ValueError(f"Unsupported pricing model: {model}")
    
    def _price_dcf(self, bond: SovereignBond, discount_rate: Optional[float] = None) -> float:
        """Price bond using discounted cash flow model."""
        if not bond.maturity_date:
            return bond.face_value
        
        cash_flows = self.generate_cash_flows(bond)
        
        if not cash_flows:
            return bond.face_value
        
        # Use provided discount rate or bond's YTM or default rate
        rate = discount_rate or bond.yield_to_maturity or self.default_risk_free_rate
        
        total_pv = 0.0
        valuation_date = date.today()
        
        for cf in cash_flows:
            days_to_payment = (cf.date - valuation_date).days
            if days_to_payment > 0:
                years_to_payment = days_to_payment / 365.25
                discount_factor = 1 / (1 + rate) ** years_to_payment
                cf.discount_factor = discount_factor
                cf.present_value = cf.amount * discount_factor
                total_pv += cf.present_value
        
        # Return as percentage of face value
        return (total_pv / bond.face_value) * 100
    
    def _price_binomial(self, bond: SovereignBond, discount_rate: Optional[float] = None) -> float:
        """Price bond using binomial tree (for callable bonds)."""
        if not bond.callable:
            return self._price_dcf(bond, discount_rate)
        
        # Simplified binomial tree implementation
        rate = discount_rate or bond.yield_to_maturity or self.default_risk_free_rate
        volatility = 0.15  # 15% interest rate volatility assumption
        
        years_to_maturity = bond.years_to_maturity
        if years_to_maturity <= 0:
            return bond.face_value
        
        # Tree parameters
        n_steps = min(int(years_to_maturity * 4), 100)  # Quarterly steps, max 100
        dt = years_to_maturity / n_steps
        
        u = np.exp(volatility * np.sqrt(dt))  # Up factor
        d = 1 / u  # Down factor
        p = (np.exp(rate * dt) - d) / (u - d)  # Risk-neutral probability
        
        # Initialize rate tree
        rates = np.zeros((n_steps + 1, n_steps + 1))
        for i in range(n_steps + 1):
            for j in range(i + 1):
                rates[i, j] = rate * (u ** (i - j)) * (d ** j)
        
        # Initialize bond values at maturity
        bond_values = np.zeros((n_steps + 1, n_steps + 1))
        for j in range(n_steps + 1):
            bond_values[n_steps, j] = bond.face_value
        
        # Backward induction
        coupon_payment = bond.coupon_rate * bond.face_value / 2  # Semi-annual
        
        for i in range(n_steps - 1, -1, -1):
            for j in range(i + 1):
                # Calculate continuation value
                continuation_value = (p * bond_values[i + 1, j] + 
                                    (1 - p) * bond_values[i + 1, j + 1]) / (1 + rates[i, j] * dt)
                
                # Add coupon if payment date
                if i % 2 == 0:  # Semi-annual coupon payments
                    continuation_value += coupon_payment
                
                # Check call option
                call_price = bond.face_value * 1.02  # Assume 2% call premium
                if bond.first_call_date and i * dt >= 1:  # Callable after 1 year
                    bond_values[i, j] = min(continuation_value, call_price)
                else:
                    bond_values[i, j] = continuation_value
        
        return (bond_values[0, 0] / bond.face_value) * 100
    
    def _price_monte_carlo(self, bond: SovereignBond, discount_rate: Optional[float] = None,
                          n_simulations: int = 10000) -> float:
        """Price bond using Monte Carlo simulation."""
        rate = discount_rate or bond.yield_to_maturity or self.default_risk_free_rate
        volatility = 0.15  # Interest rate volatility
        
        years_to_maturity = bond.years_to_maturity
        if years_to_maturity <= 0:
            return bond.face_value
        
        dt = 1/252  # Daily time steps
        n_steps = int(years_to_maturity * 252)
        
        prices = []
        
        for _ in range(n_simulations):
            # Simulate interest rate path
            rates = [rate]
            for _ in range(n_steps):
                dW = np.random.normal(0, np.sqrt(dt))
                dr = 0.1 * (0.05 - rates[-1]) * dt + volatility * dW  # Vasicek model
                rates.append(max(0.001, rates[-1] + dr))  # Floor at 0.1%
            
            # Calculate bond price for this path
            cash_flows = self.generate_cash_flows(bond)
            total_pv = 0.0
            
            for cf in cash_flows:
                days_to_payment = (cf.date - date.today()).days
                if days_to_payment > 0:
                    step_index = min(days_to_payment, len(rates) - 1)
                    discount_rate_sim = rates[step_index]
                    years_to_payment = days_to_payment / 365.25
                    discount_factor = np.exp(-discount_rate_sim * years_to_payment)
                    total_pv += cf.amount * discount_factor
            
            prices.append(total_pv)
        
        average_price = np.mean(prices)
        return (average_price / bond.face_value) * 100
    
    def generate_cash_flows(self, bond: SovereignBond) -> List[CashFlow]:
        """Generate all cash flows for a bond."""
        if not bond.maturity_date or not bond.issue_date:
            return []
        
        cash_flows = []
        current_date = date.today()
        
        # Generate coupon payments
        if bond.coupon_rate > 0:
            # Assume semi-annual payments
            payment_frequency = 2
            months_between_payments = 12 // payment_frequency
            
            # Find next coupon date
            next_coupon_date = bond.issue_date
            while next_coupon_date <= current_date:
                next_coupon_date = self._add_months(next_coupon_date, months_between_payments)
            
            # Generate all future coupon payments
            coupon_amount = bond.coupon_rate * bond.face_value / payment_frequency
            
            while next_coupon_date <= bond.maturity_date:
                cash_flows.append(CashFlow(
                    date=next_coupon_date,
                    amount=coupon_amount,
                    type="coupon"
                ))
                next_coupon_date = self._add_months(next_coupon_date, months_between_payments)
        
        # Add principal repayment at maturity
        cash_flows.append(CashFlow(
            date=bond.maturity_date,
            amount=bond.face_value,
            type="principal"
        ))
        
        return cash_flows
    
    def calculate_yield_to_maturity(self, bond: SovereignBond, price: float,
                                  max_iterations: int = 100, tolerance: float = 1e-6) -> float:
        """Calculate yield to maturity using Newton-Raphson method."""
        if bond.years_to_maturity <= 0:
            return 0.0
        
        # Initial guess
        ytm = bond.coupon_rate or 0.05
        
        for iteration in range(max_iterations):
            calculated_price = self._price_dcf(bond, ytm)
            price_diff = calculated_price - price
            
            if abs(price_diff) < tolerance:
                break
            
            # Calculate derivative (modified duration approximation)
            ytm_up = ytm + 0.0001
            price_up = self._price_dcf(bond, ytm_up)
            derivative = (price_up - calculated_price) / 0.0001
            
            if abs(derivative) < 1e-10:
                break
            
            ytm = ytm - price_diff / derivative
            ytm = max(0.0001, ytm)  # Ensure positive yield
        
        return ytm
    
    def calculate_option_adjusted_spread(self, bond: SovereignBond, market_price: float,
                                       benchmark_curve: object) -> float:
        """Calculate Option-Adjusted Spread (OAS)."""
        # Simplified OAS calculation
        # In practice, this would use Monte Carlo with stochastic interest rates
        
        if not benchmark_curve:
            return 0.0
        
        # Calculate theoretical price using benchmark curve
        theoretical_price = self._price_dcf(bond)
        
        # OAS is the spread that makes theoretical price equal market price
        spread_guess = 0.01  # 100 bps initial guess
        
        for _ in range(50):  # Max iterations
            adjusted_rate = bond.yield_to_maturity + spread_guess
            adjusted_price = self._price_dcf(bond, adjusted_rate)
            
            if abs(adjusted_price - market_price) < 0.01:
                break
            
            # Adjust spread
            if adjusted_price > market_price:
                spread_guess += 0.001  # Increase spread
            else:
                spread_guess -= 0.001  # Decrease spread
            
            spread_guess = max(0, spread_guess)
        
        return spread_guess * 10000  # Return in basis points
    
    def calculate_effective_duration(self, bond: SovereignBond, 
                                   yield_shock: float = 0.01) -> float:
        """Calculate effective duration using yield shock method."""
        base_price = self._price_dcf(bond)
        
        # Price with yield up
        price_up = self._price_dcf(bond, bond.yield_to_maturity + yield_shock)
        
        # Price with yield down
        price_down = self._price_dcf(bond, bond.yield_to_maturity - yield_shock)
        
        # Effective duration formula
        effective_duration = (price_down - price_up) / (2 * base_price * yield_shock)
        
        return effective_duration
    
    def calculate_key_rate_durations(self, bond: SovereignBond, 
                                   key_rates: List[float] = None) -> dict:
        """Calculate key rate durations for different maturity points."""
        if key_rates is None:
            key_rates = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30]  # Standard key rates
        
        base_price = self._price_dcf(bond)
        key_rate_durations = {}
        
        shock_size = 0.01  # 1% shock
        
        for key_rate in key_rates:
            if key_rate <= bond.years_to_maturity:
                # This is simplified - in practice would shock the yield curve
                # at specific points and reprice
                shocked_yield = bond.yield_to_maturity + shock_size
                shocked_price = self._price_dcf(bond, shocked_yield)
                
                duration = -(shocked_price - base_price) / (base_price * shock_size)
                key_rate_durations[f"{key_rate}Y"] = duration
        
        return key_rate_durations
    
    @staticmethod
    def _add_months(start_date: date, months: int) -> date:
        """Add months to a date."""
        month = start_date.month - 1 + months
        year = start_date.year + month // 12
        month = month % 12 + 1
        day = min(start_date.day, [31,
                                   29 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 28,
                                   31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month - 1])
        return date(year, month, day)
    
    def price_with_credit_spread(self, bond: SovereignBond, credit_spread: float) -> float:
        """Price bond with additional credit spread."""
        adjusted_yield = bond.yield_to_maturity + credit_spread
        return self._price_dcf(bond, adjusted_yield)
    
    def calculate_z_spread(self, bond: SovereignBond, market_price: float,
                          treasury_curve: object) -> float:
        """Calculate Z-spread (zero-volatility spread)."""
        # Simplified Z-spread calculation
        # In practice, would use actual treasury curve
        
        spread_guess = 0.01  # 100 bps initial guess
        
        for _ in range(50):
            adjusted_yield = bond.yield_to_maturity + spread_guess
            calculated_price = self._price_dcf(bond, adjusted_yield)
            
            if abs(calculated_price - market_price) < 0.01:
                break
            
            if calculated_price > market_price:
                spread_guess += 0.001
            else:
                spread_guess -= 0.001
            
            spread_guess = max(0, spread_guess)
        
        return spread_guess * 10000  # Return in basis points