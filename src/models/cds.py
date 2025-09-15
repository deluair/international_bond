"""
Credit Default Swap (CDS) data model and curve interpolation.
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import List, Dict, Tuple, Optional
import numpy as np
from scipy import interpolate
from enum import Enum


class CDSConvention(Enum):
    """CDS market conventions."""
    STANDARD = "standard"
    OLD = "old"
    EMERGING = "emerging"


@dataclass
class CDSQuote:
    """Individual CDS quote for a specific tenor."""
    tenor: str  # e.g., "1Y", "5Y", "10Y"
    tenor_years: float
    spread_bps: float
    quote_date: date
    bid_spread: Optional[float] = None
    ask_spread: Optional[float] = None
    
    def __post_init__(self):
        """Parse tenor string to years if not provided."""
        if self.tenor_years == 0:
            self.tenor_years = self._parse_tenor(self.tenor)
    
    @staticmethod
    def _parse_tenor(tenor: str) -> float:
        """Parse tenor string to years."""
        tenor = tenor.upper().strip()
        if tenor.endswith('Y'):
            return float(tenor[:-1])
        elif tenor.endswith('M'):
            return float(tenor[:-1]) / 12
        elif tenor.endswith('D'):
            return float(tenor[:-1]) / 365
        else:
            raise ValueError(f"Invalid tenor format: {tenor}")


@dataclass
class CDSCurve:
    """
    Credit Default Swap curve for a specific reference entity.
    """
    reference_entity: str
    currency: str
    quote_date: date
    quotes: List[CDSQuote] = field(default_factory=list)
    convention: CDSConvention = CDSConvention.STANDARD
    recovery_rate: float = 0.4  # Standard assumption
    
    # Interpolated curve data
    _interpolator: Optional[object] = field(default=None, init=False, repr=False)
    _tenors_array: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _spreads_array: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    
    def __post_init__(self):
        """Initialize interpolation after creation."""
        if self.quotes:
            self._build_interpolator()
    
    def add_quote(self, quote: CDSQuote):
        """Add a CDS quote to the curve."""
        self.quotes.append(quote)
        self.quotes.sort(key=lambda x: x.tenor_years)
        self._build_interpolator()
    
    def _build_interpolator(self):
        """Build interpolation function for the CDS curve."""
        if len(self.quotes) < 2:
            return
        
        tenors = np.array([q.tenor_years for q in self.quotes])
        spreads = np.array([q.spread_bps for q in self.quotes])
        
        # Remove duplicates and sort
        unique_indices = np.unique(tenors, return_index=True)[1]
        tenors = tenors[unique_indices]
        spreads = spreads[unique_indices]
        
        self._tenors_array = tenors
        self._spreads_array = spreads
        
        # Use cubic spline interpolation for smooth curves
        if len(tenors) >= 4:
            self._interpolator = interpolate.CubicSpline(
                tenors, spreads, bc_type='natural'
            )
        else:
            # Linear interpolation for fewer points
            self._interpolator = interpolate.interp1d(
                tenors, spreads, kind='linear', 
                bounds_error=False, fill_value='extrapolate'
            )
    
    def get_spread(self, tenor_years: float) -> float:
        """Get interpolated CDS spread for a given tenor."""
        if not self._interpolator:
            if not self.quotes:
                return 0.0
            # Return closest quote if no interpolation available
            closest_quote = min(self.quotes, 
                              key=lambda x: abs(x.tenor_years - tenor_years))
            return closest_quote.spread_bps
        
        # Extrapolate using flat curve beyond available data
        if tenor_years <= self._tenors_array[0]:
            return float(self._spreads_array[0])
        elif tenor_years >= self._tenors_array[-1]:
            return float(self._spreads_array[-1])
        else:
            return float(self._interpolator(tenor_years))
    
    def get_survival_probability(self, tenor_years: float) -> float:
        """Calculate survival probability for a given tenor."""
        spread_bps = self.get_spread(tenor_years)
        spread_decimal = spread_bps / 10000  # Convert bps to decimal
        
        # Simplified survival probability calculation
        # In practice, this would use more sophisticated bootstrapping
        hazard_rate = spread_decimal / (1 - self.recovery_rate)
        survival_prob = np.exp(-hazard_rate * tenor_years)
        
        return max(0.0, min(1.0, survival_prob))
    
    def get_default_probability(self, tenor_years: float) -> float:
        """Calculate cumulative default probability for a given tenor."""
        return 1.0 - self.get_survival_probability(tenor_years)
    
    def calculate_risky_pv01(self, tenor_years: float) -> float:
        """Calculate risky PV01 (present value of 1bp spread)."""
        # Simplified calculation - in practice would integrate over the curve
        survival_prob = self.get_survival_probability(tenor_years)
        discount_factor = np.exp(-0.02 * tenor_years)  # Assume 2% risk-free rate
        
        return survival_prob * discount_factor * tenor_years * 0.0001
    
    def parallel_shift(self, shift_bps: float) -> 'CDSCurve':
        """Create a new curve with parallel shift in spreads."""
        shifted_quotes = []
        for quote in self.quotes:
            shifted_quote = CDSQuote(
                tenor=quote.tenor,
                tenor_years=quote.tenor_years,
                spread_bps=quote.spread_bps + shift_bps,
                quote_date=quote.quote_date,
                bid_spread=quote.bid_spread + shift_bps if quote.bid_spread else None,
                ask_spread=quote.ask_spread + shift_bps if quote.ask_spread else None
            )
            shifted_quotes.append(shifted_quote)
        
        return CDSCurve(
            reference_entity=self.reference_entity,
            currency=self.currency,
            quote_date=self.quote_date,
            quotes=shifted_quotes,
            convention=self.convention,
            recovery_rate=self.recovery_rate
        )
    
    def get_curve_points(self, num_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Get interpolated curve points for plotting."""
        if not self._tenors_array is not None:
            return np.array([]), np.array([])
        
        min_tenor = self._tenors_array[0]
        max_tenor = self._tenors_array[-1]
        tenors = np.linspace(min_tenor, max_tenor, num_points)
        spreads = np.array([self.get_spread(t) for t in tenors])
        
        return tenors, spreads
    
    @property
    def benchmark_spreads(self) -> Dict[str, float]:
        """Get benchmark tenor spreads."""
        benchmarks = ['1Y', '2Y', '3Y', '5Y', '7Y', '10Y']
        result = {}
        
        for tenor in benchmarks:
            try:
                tenor_years = CDSQuote._parse_tenor(tenor)
                spread = self.get_spread(tenor_years)
                result[tenor] = spread
            except:
                continue
        
        return result
    
    @property
    def five_year_spread(self) -> float:
        """Get the 5-year CDS spread (most liquid benchmark)."""
        return self.get_spread(5.0)
    
    def to_dict(self) -> Dict:
        """Convert CDS curve to dictionary representation."""
        return {
            'reference_entity': self.reference_entity,
            'currency': self.currency,
            'quote_date': self.quote_date.isoformat(),
            'convention': self.convention.value,
            'recovery_rate': self.recovery_rate,
            'quotes': [
                {
                    'tenor': q.tenor,
                    'tenor_years': q.tenor_years,
                    'spread_bps': q.spread_bps,
                    'quote_date': q.quote_date.isoformat()
                }
                for q in self.quotes
            ],
            'benchmark_spreads': self.benchmark_spreads,
            'five_year_spread': self.five_year_spread
        }
    
    def __str__(self) -> str:
        """String representation of the CDS curve."""
        return f"CDS Curve: {self.reference_entity} ({self.currency}) - 5Y: {self.five_year_spread:.1f}bps"