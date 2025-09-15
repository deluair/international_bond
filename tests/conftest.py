"""
Simplified test configuration for international bond system.
"""

import pytest
import sys
import os
from datetime import date, datetime
from decimal import Decimal

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.bond import SovereignBond
from models.portfolio import Portfolio


@pytest.fixture
def sample_us_bond():
    """Create a sample US Treasury bond for testing."""
    return SovereignBond(
        isin="US912828XG55",
        cusip="912828XG5",
        issuer="US Treasury",
        country="United States",
        currency="USD",
        issue_date=date(2020, 1, 15),
        maturity_date=date(2030, 1, 15),  # Future date
        coupon_rate=0.025,
        face_value=1000.0
    )


@pytest.fixture
def sample_german_bond():
    """Create a sample German Bund for testing."""
    return SovereignBond(
        isin="DE0001102309",
        cusip="D15135AA9",
        issuer="Germany",
        country="Germany",
        currency="EUR",
        issue_date=date(2020, 2, 15),
        maturity_date=date(2030, 2, 15),  # Future date
        coupon_rate=0.015,
        face_value=1000.0
    )


@pytest.fixture
def sample_portfolio(sample_us_bond, sample_german_bond):
    """Create a sample portfolio with bonds."""
    portfolio = Portfolio(
        name="Test Portfolio",
        base_currency="USD"
    )
    
    # Add positions
    portfolio.add_position(sample_us_bond, 100000.0)  # $100k position
    portfolio.add_position(sample_german_bond, 50000.0)   # €50k position
    
    return portfolio


@pytest.fixture
def test_config():
    """Basic test configuration."""
    return {
        'base_currency': 'USD',
        'analysis_date': date.today(),
        'risk_free_rate': 0.02
    }