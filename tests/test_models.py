"""
Essential unit tests for core models.
"""

import pytest
from datetime import date
from models.bond import SovereignBond
from models.portfolio import Portfolio


class TestSovereignBond:
    """Test SovereignBond model functionality."""
    
    def test_bond_creation(self):
        """Test creating a sovereign bond."""
        bond = SovereignBond(
            isin="US912828XG55",
            cusip="912828XG5",
            country="United States",
            currency="USD",
            issue_date=date(2020, 1, 15),
            maturity_date=date(2025, 1, 15),
            coupon_rate=0.025,
            face_value=1000.0
        )
        
        assert bond.isin == "US912828XG55"
        assert bond.country == "United States"
        assert bond.currency == "USD"
        assert bond.coupon_rate == 0.025
        assert bond.face_value == 1000.0
    
    def test_bond_years_to_maturity(self, sample_us_bond):
        """Test years to maturity calculation."""
        # This will vary based on current date, just ensure it returns a number
        years = sample_us_bond.years_to_maturity
        assert isinstance(years, (int, float))
        assert years >= 0
    
    def test_bond_string_representation(self, sample_us_bond):
        """Test bond string representation."""
        bond_str = str(sample_us_bond)
        assert "US912828XG55" in bond_str
        # The string representation uses issuer, not country
        assert "2.50%" in bond_str


class TestPortfolio:
    """Test Portfolio model functionality."""
    
    def test_portfolio_creation(self):
        """Test creating a portfolio."""
        portfolio = Portfolio(
            name="Test Portfolio",
            base_currency="USD"
        )
        
        assert portfolio.name == "Test Portfolio"
        assert portfolio.base_currency == "USD"
        assert len(portfolio.positions) == 0
    
    def test_add_position(self, sample_portfolio):
        """Test adding positions to portfolio."""
        assert len(sample_portfolio.positions) == 2
        
        # Check positions exist
        position_isins = [pos.bond.isin for pos in sample_portfolio.positions]
        assert "US912828XG55" in position_isins
        assert "DE0001102309" in position_isins
    
    def test_portfolio_total_market_value(self, sample_portfolio):
        """Test portfolio total market value calculation."""
        # This should return a number (exact value depends on pricing logic)
        total_value = sample_portfolio.total_market_value
        assert isinstance(total_value, (int, float))
        assert total_value > 0
    
    def test_portfolio_string_representation(self, sample_portfolio):
        """Test portfolio string representation."""
        portfolio_str = str(sample_portfolio)
        assert "Test Portfolio" in portfolio_str