"""
Currency pair data model for FX rates and hedging calculations.
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional, Dict, List
import numpy as np
from enum import Enum


class CurrencyType(Enum):
    """Currency classification."""
    MAJOR = "major"
    MINOR = "minor"
    EXOTIC = "exotic"
    CRYPTO = "crypto"


@dataclass
class ForwardPoint:
    """Forward FX point for a specific tenor."""
    tenor: str
    tenor_days: int
    forward_points: float
    outright_rate: float
    
    @property
    def tenor_years(self) -> float:
        """Convert tenor days to years."""
        return self.tenor_days / 365.25


@dataclass
class CurrencyPair:
    """
    Currency pair with spot rates, forwards, and volatility data.
    """
    base_currency: str
    quote_currency: str
    spot_rate: float
    quote_date: date
    
    # Market data
    bid_rate: Optional[float] = None
    ask_rate: Optional[float] = None
    mid_rate: Optional[float] = None
    
    # Forward curve
    forward_points: List[ForwardPoint] = field(default_factory=list)
    
    # Volatility data
    implied_volatility_1m: Optional[float] = None
    implied_volatility_3m: Optional[float] = None
    implied_volatility_6m: Optional[float] = None
    implied_volatility_1y: Optional[float] = None
    
    # Interest rate differentials
    base_interest_rate: Optional[float] = None
    quote_interest_rate: Optional[float] = None
    
    # Classification
    currency_type: CurrencyType = CurrencyType.MAJOR
    
    def __post_init__(self):
        """Initialize derived fields."""
        if self.mid_rate is None and self.bid_rate and self.ask_rate:
            self.mid_rate = (self.bid_rate + self.ask_rate) / 2
        
        if self.mid_rate is None:
            self.mid_rate = self.spot_rate
    
    @property
    def pair_name(self) -> str:
        """Get currency pair name (e.g., EURUSD)."""
        return f"{self.base_currency}{self.quote_currency}"
    
    @property
    def inverse_rate(self) -> float:
        """Get inverse exchange rate."""
        return 1.0 / self.spot_rate if self.spot_rate != 0 else 0.0
    
    @property
    def bid_ask_spread(self) -> float:
        """Calculate bid-ask spread in pips."""
        if not self.bid_rate or not self.ask_rate:
            return 0.0
        
        spread = self.ask_rate - self.bid_rate
        # Convert to pips (assuming 4 decimal places for major pairs)
        pip_factor = 10000 if self.currency_type == CurrencyType.MAJOR else 100
        return spread * pip_factor
    
    @property
    def interest_rate_differential(self) -> Optional[float]:
        """Calculate interest rate differential (base - quote)."""
        if self.base_interest_rate is None or self.quote_interest_rate is None:
            return None
        return self.base_interest_rate - self.quote_interest_rate
    
    def get_forward_rate(self, tenor_days: int) -> float:
        """Get forward rate for a specific tenor."""
        # Find exact match first
        for fp in self.forward_points:
            if fp.tenor_days == tenor_days:
                return fp.outright_rate
        
        # If no exact match, interpolate
        if len(self.forward_points) < 2:
            return self.spot_rate
        
        # Sort by tenor
        sorted_points = sorted(self.forward_points, key=lambda x: x.tenor_days)
        
        # Linear interpolation
        for i in range(len(sorted_points) - 1):
            if sorted_points[i].tenor_days <= tenor_days <= sorted_points[i + 1].tenor_days:
                x1, y1 = sorted_points[i].tenor_days, sorted_points[i].outright_rate
                x2, y2 = sorted_points[i + 1].tenor_days, sorted_points[i + 1].outright_rate
                
                # Linear interpolation
                rate = y1 + (y2 - y1) * (tenor_days - x1) / (x2 - x1)
                return rate
        
        # Extrapolation
        if tenor_days < sorted_points[0].tenor_days:
            return sorted_points[0].outright_rate
        else:
            return sorted_points[-1].outright_rate
    
    def calculate_forward_rate_theoretical(self, tenor_days: int) -> float:
        """Calculate theoretical forward rate using interest rate parity."""
        if self.base_interest_rate is None or self.quote_interest_rate is None:
            return self.spot_rate
        
        tenor_years = tenor_days / 365.25
        
        # Interest rate parity: F = S * (1 + r_quote * t) / (1 + r_base * t)
        forward_rate = self.spot_rate * (
            (1 + self.quote_interest_rate * tenor_years) /
            (1 + self.base_interest_rate * tenor_years)
        )
        
        return forward_rate
    
    def get_implied_volatility(self, tenor_days: int) -> Optional[float]:
        """Get implied volatility for a specific tenor (interpolated)."""
        vol_points = [
            (30, self.implied_volatility_1m),
            (90, self.implied_volatility_3m),
            (180, self.implied_volatility_6m),
            (365, self.implied_volatility_1y)
        ]
        
        # Filter out None values
        valid_points = [(days, vol) for days, vol in vol_points if vol is not None]
        
        if not valid_points:
            return None
        
        if len(valid_points) == 1:
            return valid_points[0][1]
        
        # Linear interpolation
        for i in range(len(valid_points) - 1):
            if valid_points[i][0] <= tenor_days <= valid_points[i + 1][0]:
                x1, y1 = valid_points[i]
                x2, y2 = valid_points[i + 1]
                
                vol = y1 + (y2 - y1) * (tenor_days - x1) / (x2 - x1)
                return vol
        
        # Extrapolation
        if tenor_days < valid_points[0][0]:
            return valid_points[0][1]
        else:
            return valid_points[-1][1]
    
    def calculate_hedge_ratio(self, bond_amount: float, hedge_tenor_days: int = 30) -> float:
        """Calculate FX hedge ratio for a bond position."""
        # Simple hedge ratio - in practice would consider duration and correlation
        forward_rate = self.get_forward_rate(hedge_tenor_days)
        
        # Hedge ratio based on forward rate differential
        hedge_ratio = bond_amount * (forward_rate / self.spot_rate)
        
        return hedge_ratio
    
    def calculate_carry(self, tenor_days: int) -> float:
        """Calculate carry (interest rate differential impact)."""
        if self.interest_rate_differential is None:
            return 0.0
        
        tenor_years = tenor_days / 365.25
        carry = self.interest_rate_differential * tenor_years
        
        return carry
    
    def add_forward_point(self, tenor: str, tenor_days: int, forward_points: float):
        """Add a forward point to the curve."""
        outright_rate = self.spot_rate + (forward_points / 10000)  # Assuming points in pips
        
        fp = ForwardPoint(
            tenor=tenor,
            tenor_days=tenor_days,
            forward_points=forward_points,
            outright_rate=outright_rate
        )
        
        self.forward_points.append(fp)
        self.forward_points.sort(key=lambda x: x.tenor_days)
    
    def to_dict(self) -> Dict:
        """Convert currency pair to dictionary representation."""
        return {
            'pair_name': self.pair_name,
            'base_currency': self.base_currency,
            'quote_currency': self.quote_currency,
            'spot_rate': self.spot_rate,
            'mid_rate': self.mid_rate,
            'quote_date': self.quote_date.isoformat(),
            'bid_ask_spread': self.bid_ask_spread,
            'interest_rate_differential': self.interest_rate_differential,
            'currency_type': self.currency_type.value,
            'forward_points': [
                {
                    'tenor': fp.tenor,
                    'tenor_days': fp.tenor_days,
                    'forward_points': fp.forward_points,
                    'outright_rate': fp.outright_rate
                }
                for fp in self.forward_points
            ],
            'implied_volatilities': {
                '1M': self.implied_volatility_1m,
                '3M': self.implied_volatility_3m,
                '6M': self.implied_volatility_6m,
                '1Y': self.implied_volatility_1y
            }
        }
    
    def __str__(self) -> str:
        """String representation of the currency pair."""
        return f"{self.pair_name}: {self.spot_rate:.4f} ({self.quote_date})"