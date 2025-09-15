"""
Test Package for International Bond Relative Value System

This package contains comprehensive test suites for all modules of the
international bond relative value analysis system.

Test Structure:
- test_analytics/: Tests for analytics modules
- test_data/: Tests for data management modules  
- test_policy/: Tests for policy analysis modules
- test_risk/: Tests for risk management modules
- test_strategies/: Tests for trading strategy modules
- test_integration/: Integration tests
- test_performance/: Performance tests
- fixtures/: Test data and fixtures
- conftest.py: Pytest configuration and shared fixtures
"""

# Test configuration
TEST_DATA_PATH = "tests/fixtures/test_data"
TEST_CONFIG_PATH = "tests/fixtures/test_config"

# Test constants
DEFAULT_TEST_TIMEOUT = 30  # seconds
PERFORMANCE_TEST_ITERATIONS = 100
INTEGRATION_TEST_BONDS = [
    "US10Y", "DE10Y", "JP10Y", "GB10Y", "FR10Y",
    "IT10Y", "ES10Y", "CA10Y", "AU10Y", "CH10Y"
]

__version__ = "1.0.0"
__author__ = "International Bond Analysis Team"