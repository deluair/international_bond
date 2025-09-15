"""
Data Management Module Tests

This package contains comprehensive tests for the data management modules,
ensuring robust data collection, processing, and storage capabilities.

Test Modules:
- test_bond_data_collector: Bond market data collection tests
- test_economic_data_collector: Economic indicators data collection tests
- test_fx_data_collector: Foreign exchange data collection tests
- test_data_processor: Data cleaning and processing tests
- test_data_validator: Data quality validation tests

Test Coverage:
- Unit tests for individual data collectors
- Integration tests for data pipeline workflows
- Performance tests for large data volumes
- Validation tests for data quality and consistency
- Error handling tests for data source failures
"""

__version__ = "1.0.0"
__author__ = "International Bond System"

# Test configuration
TEST_DATA_SOURCES = ['bloomberg', 'refinitiv', 'fred', 'ecb']
TEST_DATE_RANGES = {
    'short': 30,    # 30 days
    'medium': 252,  # 1 year
    'long': 1260    # 5 years
}
TEST_CURRENCIES = ['USD', 'EUR', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD']
TEST_COUNTRIES = ['US', 'DE', 'GB', 'JP', 'FR', 'IT', 'ES', 'CA', 'AU']

# Data quality thresholds
MIN_DATA_COMPLETENESS = 0.95    # 95% data availability
MAX_OUTLIER_RATIO = 0.05        # 5% outliers allowed
MAX_DATA_LATENCY = 3600         # 1 hour max latency (seconds)
MIN_UPDATE_FREQUENCY = 86400    # Daily updates minimum (seconds)