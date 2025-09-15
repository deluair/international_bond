"""
Sovereign Bond data model with pricing and risk metrics.
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Optional, Dict, List
import numpy as np
from enum import Enum


class BondType(Enum):
    """Bond type classification."""
    GOVERNMENT = "government"
    CORPORATE = "corporate"
    MUNICIPAL = "municipal"
    SUPRANATIONAL = "supranational"


class RatingAgency(Enum):
    """Credit rating agencies."""
    MOODY = "moody"
    SP = "sp"
    FITCH = "fitch"


@dataclass
class CreditRating:
    """Credit rating information."""
    agency: RatingAgency
    rating: str
    outlook: str
    date_assigned: date


@dataclass
class SovereignBond:
    """
    Sovereign bond data model with comprehensive attributes for analysis.
    """
    # Basic identifiers
    isin: str
    cusip: Optional[str] = None
    ticker: Optional[str] = None
    
    # Bond characteristics
    issuer: str = ""
    country: str = ""
    currency: str = ""
    bond_type: BondType = BondType.GOVERNMENT
    
    # Financial terms
    face_value: float = 1000.0
    coupon_rate: float = 0.0
    issue_date: Optional[date] = None
    maturity_date: Optional[date] = None
    first_call_date: Optional[date] = None
    
    # Market data
    current_price: float = 100.0
    yield_to_maturity: float = 0.0
    duration: float = 0.0
    modified_duration: float = 0.0
    convexity: float = 0.0
    
    # Credit information
    credit_ratings: List[CreditRating] = field(default_factory=list)
    cds_spread: Optional[float] = None
    
    # Market metrics
    bid_price: Optional[float] = None
    ask_price: Optional[float] = None
    last_trade_date: Optional[datetime] = None
    daily_volume: Optional[float] = None
    
    # Risk metrics
    credit_spread: Optional[float] = None
    z_spread: Optional[float] = None
    oas_spread: Optional[float] = None
    
    # Additional metadata
    sector: Optional[str] = None
    subsector: Optional[str] = None
    callable: bool = False
    puttable: bool = False
    
    def __post_init__(self):
        """Validate and compute derived fields."""
        if self.maturity_date and self.issue_date:
            self.time_to_maturity = (self.maturity_date - self.issue_date).days / 365.25
        else:
            self.time_to_maturity = 0.0
    
    @property
    def years_to_maturity(self) -> float:
        """Calculate years to maturity from current date."""
        if not self.maturity_date:
            return 0.0
        return (self.maturity_date - date.today()).days / 365.25
    
    @property
    def is_investment_grade(self) -> bool:
        """Determine if bond is investment grade based on ratings."""
        if not self.credit_ratings:
            return False
        
        # Simple mapping - in practice would be more sophisticated
        investment_grade_ratings = {
            'AAA', 'AA+', 'AA', 'AA-', 'A+', 'A', 'A-', 
            'BBB+', 'BBB', 'BBB-', 'Aaa', 'Aa1', 'Aa2', 
            'Aa3', 'A1', 'A2', 'A3', 'Baa1', 'Baa2', 'Baa3'
        }
        
        return any(rating.rating in investment_grade_ratings 
                  for rating in self.credit_ratings)
    
    @property
    def accrued_interest(self) -> float:
        """Calculate accrued interest (simplified)."""
        if not self.issue_date or self.coupon_rate == 0:
            return 0.0
        
        days_since_last_payment = (date.today() - self.issue_date).days % 182.5
        return (self.coupon_rate / 2) * (days_since_last_payment / 182.5) * self.face_value
    
    @property
    def clean_price(self) -> float:
        """Clean price (price without accrued interest)."""
        return self.current_price - (self.accrued_interest / self.face_value * 100)
    
    def calculate_price_from_yield(self, yield_rate: float) -> float:
        """Calculate theoretical price from yield."""
        if not self.maturity_date or self.years_to_maturity <= 0:
            return self.face_value
        
        periods = int(self.years_to_maturity * 2)  # Semi-annual payments
        coupon_payment = self.coupon_rate * self.face_value / 2
        discount_rate = yield_rate / 2
        
        if discount_rate == 0:
            return self.face_value + coupon_payment * periods
        
        # Present value of coupon payments
        pv_coupons = coupon_payment * (1 - (1 + discount_rate) ** -periods) / discount_rate
        
        # Present value of principal
        pv_principal = self.face_value / (1 + discount_rate) ** periods
        
        return pv_coupons + pv_principal
    
    def calculate_yield_from_price(self, price: float, max_iterations: int = 100) -> float:
        """Calculate yield to maturity from price using Newton-Raphson method."""
        if not self.maturity_date or self.years_to_maturity <= 0:
            return 0.0
        
        # Initial guess
        yield_guess = self.coupon_rate
        
        for _ in range(max_iterations):
            calculated_price = self.calculate_price_from_yield(yield_guess)
            price_diff = calculated_price - price
            
            if abs(price_diff) < 0.01:  # Convergence threshold
                break
            
            # Calculate derivative (duration approximation)
            yield_up = yield_guess + 0.0001
            price_up = self.calculate_price_from_yield(yield_up)
            derivative = (price_up - calculated_price) / 0.0001
            
            if abs(derivative) < 1e-10:
                break
            
            yield_guess = yield_guess - price_diff / derivative
        
        return max(0, yield_guess)  # Yield cannot be negative
    
    def to_dict(self) -> Dict:
        """Convert bond to dictionary representation."""
        return {
            'isin': self.isin,
            'cusip': self.cusip,
            'ticker': self.ticker,
            'issuer': self.issuer,
            'country': self.country,
            'currency': self.currency,
            'bond_type': self.bond_type.value,
            'face_value': self.face_value,
            'coupon_rate': self.coupon_rate,
            'issue_date': self.issue_date.isoformat() if self.issue_date else None,
            'maturity_date': self.maturity_date.isoformat() if self.maturity_date else None,
            'current_price': self.current_price,
            'yield_to_maturity': self.yield_to_maturity,
            'duration': self.duration,
            'modified_duration': self.modified_duration,
            'convexity': self.convexity,
            'years_to_maturity': self.years_to_maturity,
            'is_investment_grade': self.is_investment_grade,
            'credit_spread': self.credit_spread,
            'cds_spread': self.cds_spread
        }
    
    def __str__(self) -> str:
        """String representation of the bond."""
        return f"{self.issuer} {self.coupon_rate:.2%} {self.maturity_date} ({self.isin})"