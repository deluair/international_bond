"""
International Bond Relative Value System

A comprehensive system for comparing sovereign bonds across countries 
on a risk-adjusted basis with CDS curves and currency overlay.
"""

__version__ = "1.0.0"
__author__ = "International Bond Analysis Team"

from .models import SovereignBond, Portfolio
from .strategies import CarryTradeStrategy, SpreadStrategy

__all__ = [
    "SovereignBond",
    "Portfolio", 
    "CarryTradeStrategy",
    "SpreadStrategy"
]